"""Governed NL→SQL (FR-D10) — the safety layer between a model and a tenant's database.

A natural-language question is translated to SQL by the tenant's own model, but the model
is NEVER trusted: every generated statement is parsed with sqlglot and must pass a hard
allowlist before it can touch the database. The guarantees enforced here:

* **read-only** — exactly one statement, and it must be a SELECT/UNION; any INSERT/UPDATE/
  DELETE/DDL/`Command` (SET, COPY, VACUUM, …) or `SELECT … INTO` is rejected;
* **table allowlist** — every referenced table must be in the connection's allowlist (CTE
  names introduced by `WITH` are permitted, since they are not base tables);
* **no data-exfiltration functions** — a denylist of file/network/sleep builtins
  (pg_read_file, lo_import, dblink, pg_sleep, load_file, …) is refused;
* **bounded** — the validated query is wrapped with an outer ``LIMIT`` so no question can
  return an unbounded result set, regardless of the inner query's shape.

The validator returns the safe SQL to execute; on any violation it raises ``ValidationError``
so the caller abstains rather than running something risky. Pure and deterministic — the bulk
of this module is unit-tested with adversarial inputs and needs no database.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from rag_common.errors import ValidationError

# Builtins that read files, open networks, or stall the server — never legitimate for NL Q&A.
_DENIED_FUNCTIONS = frozenset(
    {
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_sleep",
        "lo_import",
        "lo_export",
        "dblink",
        "dblink_exec",
        "copy",
        "load_file",
        "sleep",
        "benchmark",
        "load_extension",
        "readfile",
        "sys_exec",
        "xp_cmdshell",
        "current_setting",
        "set_config",
    }
)

# Statement node types that must never appear (DML/DDL/side-effecting commands).
_FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Into,
    exp.Set,
)


def _tables_referenced(tree: exp.Expression) -> set[str]:
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name}


def _cte_names(tree: exp.Expression) -> set[str]:
    names: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        alias = cte.alias
        if alias:
            names.add(alias.lower())
    return names


# sqlglot's dialect names differ from our DSN-scheme names for some engines — notably SQL Server
# is ``tsql``, not ``mssql``. Map ours → sqlglot's; an unknown engine maps to ``None`` (generic
# ANSI parsing), which never crashes and is still fully governed — the checks below are AST-based,
# not dialect-specific. This is what lets NL→SQL govern ANY connected database, not just Postgres.
_SQLGLOT_DIALECT = {
    "postgres": "postgres",
    "postgresql": "postgres",
    "mysql": "mysql",
    "mariadb": "mysql",
    "mssql": "tsql",
    "tsql": "tsql",
    "oracle": "oracle",
    "sqlite": "sqlite",
}


def _sqlglot_dialect(dialect: str) -> str | None:
    return _SQLGLOT_DIALECT.get(dialect.lower())


def validate_sql(  # noqa: PLR0912  # SQL governance: many narrow guard branches, each a distinct refusal rule
    sql: str,
    *,
    allowed_tables: set[str],
    dialect: str = "postgres",
    max_rows: int = 1000,
) -> str:
    """Parse + govern a model-generated SQL string; return the safe (bounded) SQL to run.

    Raises :class:`ValidationError` on anything that is not a single read-only SELECT over the
    allowlisted tables.
    """
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise ValidationError("empty SQL")

    sg = _sqlglot_dialect(dialect)
    try:
        statements = sqlglot.parse(text, read=sg)
    except Exception as exc:  # sqlglot ParseError and friends
        raise ValidationError("could not parse SQL", details={"reason": str(exc)[:200]}) from exc

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise ValidationError("exactly one statement is allowed", details={"statements": len(statements)})
    tree = statements[0]

    if not isinstance(tree, exp.Select | exp.Union | exp.Subquery | exp.With):
        raise ValidationError("only read-only SELECT queries are allowed", details={"got": type(tree).__name__})
    for node_type in _FORBIDDEN_NODES:
        if isinstance(tree, node_type) or tree.find(node_type) is not None:
            raise ValidationError("write/DDL/command statements are not permitted")

    allowed = {t.lower() for t in allowed_tables}
    referenced = _tables_referenced(tree) - _cte_names(tree)
    unknown = referenced - allowed
    if unknown:
        raise ValidationError("query references tables outside the allowlist", details={"tables": sorted(unknown)})

    for anon in tree.find_all(exp.Anonymous):
        if (anon.name or "").lower() in _DENIED_FUNCTIONS:
            raise ValidationError("query uses a forbidden function", details={"function": anon.name.lower()})
    for func in tree.find_all(exp.Func):
        fname = (func.sql_name() or "").lower()
        if fname in _DENIED_FUNCTIONS:
            raise ValidationError("query uses a forbidden function", details={"function": fname})

    # A derived table needs EVERY column named on strict dialects — SQL Server rejects an unnamed
    # COUNT(*) inside the subquery below ("No column name was specified for column 1"). Alias any
    # unnamed top-level projection so the LIMIT-wrap is valid on every dialect (Postgres/MySQL are
    # lenient, which made this fail only intermittently, whenever the model omitted an alias).
    if isinstance(tree, exp.Select):
        for i, proj in enumerate(list(tree.expressions)):
            # A bare Column/Alias/Star is already named; anything else (COUNT(*), a+b, CASE …) has no
            # column name in a derived table, so give it one.
            if not isinstance(proj, exp.Column | exp.Alias | exp.Star):
                proj.replace(exp.alias_(proj.copy(), f"col_{i + 1}"))

    # Bound the result set. SQL Server REJECTS a derived table that carries its own ORDER BY
    # (error 1033), so the subquery wrap below breaks every ranked question — "break down by
    # status", "top 5 categories" — against MSSQL. T-SQL instead takes the cap as TOP on the
    # statement itself, which leaves the ORDER BY at the top level where it is legal.
    if sg == "tsql":
        capped = tree.limit(max_rows) if isinstance(tree, exp.Select) else tree
        return capped.sql(dialect=sg)

    # Every other dialect: wrap with a hard outer LIMIT so the bound holds whatever the inner shape.
    safe = exp.select("*").from_(exp.Subquery(this=tree, alias="_rag")).limit(max_rows)
    return safe.sql(dialect=sg)


def format_schema(tables: dict[str, list[str]], samples: dict[str, dict[str, list[str]]] | None = None) -> str:
    """Render a compact schema description for the NL→SQL prompt. When ``samples`` gives the known
    values of a categorical column, annotate it inline (``stage=[Won|Lost|Open]``) so the model
    filters with REAL values instead of guessing."""
    samples = samples or {}
    lines: list[str] = []
    for name, cols in sorted(tables.items()):
        rendered = []
        for c in cols:
            vals = samples.get(name, {}).get(c)
            rendered.append(f"{c}=[{'|'.join(vals)}]" if vals else c)
        lines.append(f"- {name}({', '.join(rendered)})")
    return "\n".join(lines)


def build_sql_prompt(
    question: str,
    tables: dict[str, list[str]],
    *,
    dialect: str = "postgres",
    samples: dict[str, dict[str, list[str]]] | None = None,
) -> str:
    """Prompt the tenant's model to emit ONE read-only SQL query for the question."""
    return (
        f"You are a careful data analyst. Write ONE read-only {dialect} SQL SELECT query that "
        "answers the question, using ONLY these tables and columns:\n"
        f"{format_schema(tables, samples)}\n\n"
        f"Question: {question}\n\n"
        "Rules: SELECT only (no INSERT/UPDATE/DELETE/DDL); reference only the tables above; "
        "prefer explicit column lists; apply ONLY the filters the question explicitly asks for. "
        "When a column shows its known values as name=[A|B|C], filter using EXACTLY those values "
        "(match the given casing) — never invent status/enum values. If the needed value is not "
        "listed, do not guess a literal; filter on the column without assuming its contents. "
        "Return ONLY the SQL, no prose."
    )


def extract_sql(model_output: str) -> str:
    """Pull the SQL out of a model response (strip ``` fences / 'sql:' preambles)."""
    text = model_output.strip()
    if "```" in text:
        blocks = text.split("```")
        # take the block after the first fence, dropping an optional 'sql' language tag
        candidate = blocks[1] if len(blocks) > 1 else text
        candidate = candidate.split("\n", 1)[1] if candidate.lower().startswith("sql") else candidate
        text = candidate.strip()
    return text.strip().rstrip(";").strip()
