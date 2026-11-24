"""Governed NL→SQL validator: adversarial SQL must be refused; safe SELECTs bounded."""

from __future__ import annotations

import pytest
from svc_control_plane.nl2sql import build_sql_prompt, extract_sql, validate_sql

from rag_common.errors import ValidationError

_ALLOW = {"employees", "departments"}


def _ok(sql: str) -> str:
    return validate_sql(sql, allowed_tables=_ALLOW)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name, salary FROM employees WHERE salary > 100",
        "SELECT d.name, count(*) FROM employees e JOIN departments d ON e.dept_id=d.id GROUP BY d.name",
        "WITH t AS (SELECT * FROM employees) SELECT * FROM t",  # CTE name not a base table
        "SELECT name FROM employees UNION SELECT name FROM departments",
    ],
)
def test_allows_read_only_selects(sql: str) -> None:
    out = _ok(sql)
    assert out.lower().startswith("select")
    assert "limit 1000" in out.lower()  # bounded


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM employees",
        "UPDATE employees SET salary = 0",
        "DROP TABLE employees",
        "INSERT INTO employees VALUES (1)",
        "SELECT * FROM employees; DROP TABLE employees",  # stacked
        "SELECT * FROM salaries",  # not allowlisted
        "SELECT pg_read_file('/etc/passwd')",  # forbidden fn
        "SELECT * FROM employees WHERE id = pg_sleep(10)",  # forbidden fn
        "SELECT * INTO backup FROM employees",  # SELECT INTO (write)
        "VACUUM",  # command
        "",  # empty
    ],
)
def test_blocks_unsafe(sql: str) -> None:
    with pytest.raises(ValidationError):
        _ok(sql)


def test_limit_is_clamped_and_present() -> None:
    out = validate_sql("SELECT * FROM employees", allowed_tables=_ALLOW, max_rows=50)
    assert "limit 50" in out.lower()


def test_extract_sql_strips_fences() -> None:
    assert extract_sql("```sql\nSELECT 1 FROM employees;\n```").lower() == "select 1 from employees"
    assert extract_sql("SELECT 2 FROM employees").lower() == "select 2 from employees"


def test_prompt_lists_only_allowed_schema() -> None:
    p = build_sql_prompt("how many?", {"employees": ["id", "name"]}, dialect="postgres")
    assert "employees(id, name)" in p
    assert "SELECT only" in p
