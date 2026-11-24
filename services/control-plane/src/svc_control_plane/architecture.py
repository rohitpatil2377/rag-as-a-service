"""Architecture extraction — turn technical documents (and diagram images) into a typed system
architecture, render editable diagrams (Mermaid / PlantUML), and explain it in plain language.

Two directions:
* **Docs -> architecture**: an LLM extracts a typed :class:`Architecture` (applications, services,
  APIs, databases, integrations + their relationships) from retrieved text; a diagram IMAGE is first
  described by the sovereign vision model, then extracted the same way. :func:`to_mermaid` /
  :func:`to_plantuml` render an EDITABLE diagram, and :func:`build_explain_prompt` asks the model to
  explain it.
* **Architecture -> plain answer**: a NON-technical user asks "explain the architecture of X"; the
  same :class:`Architecture` is summarised and explained in jargon-free terms, with the diagram as a
  visual aid.

This module is the deterministic core: the schema, the renderers, a no-model heuristic summary, and
the prompt builders — all pure and testable without a model. The model/vision calls live in thin
wrappers (:func:`extract_architecture`, :func:`architecture_from_image`) that reuse the tenant
gateway and the multimodal vision path.
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag_common import get_settings
from rag_core.grounding import abstain, make_citations
from rag_schemas import GroundedAnswer, RetrievalQuery
from rag_schemas.citation import Citation
from rag_schemas.common import AnswerStatus

# The component taxonomy an architecture diagram is built from. Kept small + closed so the model
# maps everything onto known shapes; an unknown kind falls back to "service" (see parse_architecture).
ComponentKind = Literal["application", "service", "api", "database", "queue", "frontend", "integration", "external"]
_KINDS: frozenset[str] = frozenset(
    ("application", "service", "api", "database", "queue", "frontend", "integration", "external")
)


class ArchComponent(BaseModel):
    """One box in the architecture: an application, service, API, database, queue, frontend, an
    external system, or an integration point."""

    id: str = Field(..., description="Stable diagram id (slug of the name).")
    name: str = Field(..., description="Human-readable component name.")
    kind: ComponentKind = Field(default="service", description="Component type — drives the diagram shape.")
    description: str = Field(default="", description="One-line role of the component.")
    group: str = Field(default="", description="Optional cluster/layer (e.g. 'Control Plane') for subgraph grouping.")


class ArchEdge(BaseModel):
    """A directed relationship between two components (calls / reads / writes / integrates …)."""

    source: str = Field(..., description="Source component id.")
    target: str = Field(..., description="Target component id.")
    relation: str = Field(default="depends_on", description="e.g. calls | reads | writes | integrates | publishes.")
    label: str = Field(default="", description="Optional edge label; defaults to the relation.")


class Architecture(BaseModel):
    """A typed system architecture extracted from documents — the source of truth both renderers and
    the plain-language explanation are generated from."""

    title: str = Field(default="System Architecture")
    components: list[ArchComponent] = Field(default_factory=list)
    edges: list[ArchEdge] = Field(default_factory=list)
    summary: str = Field(default="", description="Short natural-language summary.")
    note: str = Field(
        default="",
        description="Optional render note (e.g. 'Showing 20 of 34 components (most connected).').",
    )


# --- id / text helpers ---------------------------------------------------------------


def slug_id(name: str) -> str:
    """A diagram-safe id from a component name: lowercase alphanumerics + underscores. Mermaid and
    PlantUML both choke on spaces/punctuation in ids, so every id goes through this."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return s or "node"


# Mermaid flowchart keywords: a bare node id equal to one of these corrupts the grammar (a node named
# "End" -> id `end`, which Mermaid reads as a subgraph closer -> the whole diagram fails to parse).
# Such an id is suffixed exactly like a slug collision (`end` -> `end_1`).
_RESERVED_IDS: frozenset[str] = frozenset(
    ("graph", "flowchart", "subgraph", "end", "class", "classdef", "style", "linkstyle", "click", "direction")
)


def _unique_id(base: str, used: set[str]) -> str:
    """Return a diagram-safe id derived from ``base`` that is neither already in ``used`` nor a
    Mermaid reserved word, suffixing numerically (``_1``, ``_2``, …) until both hold. Mutates
    ``used`` with the chosen id so repeated calls stay collision-free."""
    if base.lower() in _RESERVED_IDS:
        cid, n = f"{base}_1", 2
    else:
        cid, n = base, 2
    while cid in used:
        cid, n = f"{base}_{n}", n + 1
    used.add(cid)
    return cid


def _clean(text: str) -> str:
    """Collapse whitespace (newlines included) + neutralise every character that breaks Mermaid
    label/edge syntax, so a label can never corrupt the diagram. The pipe is the critical one: an
    edge label renders as ``-->|{label}|``, so a ``|`` in the label produces ``-->|a|b|`` which
    Mermaid REJECTS. ``"`` breaks quoted node labels; ``;`` is a statement separator; the shape
    brackets ``[] {} ()`` and ``< > # ` `` all confuse the parser inside a label. Each is mapped to
    a harmless look-alike (kept, not dropped, so the label stays readable)."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    for bad, good in (
        ('"', "'"),
        ("|", "/"),
        ("<", "("),
        (">", ")"),
        ("#", ""),
        ("`", "'"),
        (";", ","),
        ("[", "("),
        ("]", ")"),
        ("{", "("),
        ("}", ")"),
    ):
        cleaned = cleaned.replace(bad, good)
    return cleaned


# Longest label rendered inside a node before it is clamped with an ellipsis (the full name rides in
# the model + a tooltip title). Keeps a verbose extraction from blowing out the box width.
_MAX_LABEL = 40


def _label(text: str, *, limit: int = _MAX_LABEL) -> str:
    """A cleaned, length-clamped node/edge label. Deterministic — same input, same output."""
    clean = _clean(text)
    if len(clean) > limit:
        return clean[: limit - 1].rstrip() + "…"
    return clean


# --- parse (model JSON -> validated Architecture) ------------------------------------


def parse_architecture(data: dict[str, Any], *, title: str = "System Architecture") -> Architecture:
    """Build a validated :class:`Architecture` from loosely-typed extractor output.

    Assigns stable slug ids from names, maps an unknown ``kind`` to ``service`` (never fails on a
    kind the model invented), de-duplicates components by id, and DROPS any edge that references a
    component we did not keep — so the rendered diagram is always internally consistent."""
    comps: dict[str, ArchComponent] = {}
    name_to_id: dict[str, str] = {}
    seen_names: set[str] = set()
    used_ids: set[str] = set()
    for raw in data.get("components", []) or []:
        name = _clean(str(raw.get("name", ""))) if isinstance(raw, dict) else _clean(str(raw))
        if not name:
            continue
        if name.lower() in seen_names:  # a true duplicate NAME collapses to one node
            continue
        seen_names.add(name.lower())
        kind = str(raw.get("kind", "service")).lower() if isinstance(raw, dict) else "service"
        if kind not in _KINDS:
            kind = "service"
        desc = _clean(str(raw.get("description", ""))) if isinstance(raw, dict) else ""
        grp = _clean(str(raw.get("group", ""))) if isinstance(raw, dict) else ""
        # Two DISTINCT names can slug to the same id ("Web App" / "Web-App" -> "web_app"), and a name
        # can slug to a Mermaid reserved word ("End" -> "end"); _unique_id suffixes both so no data
        # is dropped and the id is always safe.
        cid = _unique_id(slug_id(name), used_ids)
        comps[cid] = ArchComponent(id=cid, name=name, kind=kind, description=desc, group=grp)
        name_to_id[name.lower()] = cid

    def _resolve(ref: str) -> str | None:
        ref = _clean(ref)
        if not ref:
            return None
        if ref in comps:
            return ref
        if ref.lower() in name_to_id:
            return name_to_id[ref.lower()]
        cid = slug_id(ref)
        return cid if cid in comps else None

    edges: list[ArchEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for raw in data.get("edges", []) or []:
        if isinstance(raw, dict):
            src, dst = _resolve(str(raw.get("source", ""))), _resolve(str(raw.get("target", "")))
            rel = _clean(str(raw.get("relation", "depends_on"))) or "depends_on"
            label = _clean(str(raw.get("label", "")))
        elif isinstance(raw, list | tuple) and len(raw) >= 2:  # noqa: PLR2004 - a [source, target] pair
            src, dst, rel, label = _resolve(str(raw[0])), _resolve(str(raw[1])), "depends_on", ""
        else:
            continue
        if not src or not dst or src == dst:
            continue
        key = (src, dst, rel)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(ArchEdge(source=src, target=dst, relation=rel, label=label))

    return Architecture(
        title=_clean(str(data.get("title", ""))) or title,
        components=list(comps.values()),
        edges=edges,
        summary=_clean(str(data.get("summary", ""))),
    )


# --- validate / repair / truncate (pure, deterministic) ------------------------------


def sanitize_architecture(arch: Architecture) -> Architecture:
    """Return a copy of ``arch`` GUARANTEED to render: every node id is a unique, diagram-safe slug
    (a ``slug_id`` collision from two distinct names gets a numeric suffix), every label is non-empty
    and neutralised, and every edge references a surviving component with no self-loop. Idempotent —
    a clean architecture passes through unchanged, so determinism is preserved — and defensive, so a
    hand-built (non-parsed) :class:`Architecture` with dangling edges, duplicate ids or empty labels
    still yields valid Mermaid/PlantUML instead of a broken diagram."""
    comps: list[ArchComponent] = []
    id_remap: dict[str, str] = {}
    used: set[str] = set()
    for c in arch.components:
        cid = _unique_id(slug_id(c.id), used)  # unique + non-reserved (end -> end_1, etc.)
        # Assumes distinct original ids (parse_architecture guarantees this); a hand-built
        # Architecture with duplicate original ids keeps only the FIRST id_remap mapping, so edges
        # referencing the collided id route to the first component — acceptable for a repair net.
        id_remap.setdefault(c.id, cid)
        name = _clean(c.name) or cid
        comps.append(c.model_copy(update={"id": cid, "name": name}))
    edges: list[ArchEdge] = []
    for e in arch.edges:
        src, dst = id_remap.get(e.source, e.source), id_remap.get(e.target, e.target)
        if src not in used or dst not in used or src == dst:
            continue  # drop dangling edges + self-loops so the render can't break
        edges.append(e.model_copy(update={"source": src, "target": dst}))
    return arch.model_copy(update={"components": comps, "edges": edges})


def truncate_architecture(arch: Architecture, cap: int) -> Architecture:
    """Cap a large architecture at ``cap`` components, KEEPING the most-connected ones (by edge
    degree) rather than truncating arbitrarily, and record a ``note`` ("Showing N of M components").
    Ties break on the component's original position, so the result is deterministic and preserves the
    extractor's ordering among the kept nodes. Edges to dropped components are removed."""
    total = len(arch.components)
    if total <= cap:
        return arch
    degree: dict[str, int] = {c.id: 0 for c in arch.components}
    for e in arch.edges:
        if e.source in degree:
            degree[e.source] += 1
        if e.target in degree:
            degree[e.target] += 1
    order = sorted(range(total), key=lambda i: (-degree[arch.components[i].id], i))
    keep = set(order[:cap])
    kept = [c for i, c in enumerate(arch.components) if i in keep]  # preserve original order
    kept_ids = {c.id for c in kept}
    edges = [e for e in arch.edges if e.source in kept_ids and e.target in kept_ids]
    note = f"Showing {cap} of {total} components (most connected)."
    return arch.model_copy(update={"components": kept, "edges": edges, "note": note})


def _merge_edges(edges: list[ArchEdge]) -> list[tuple[str, str, str]]:
    """Collapse duplicate ``source -> target`` edges into one, joining DISTINCT labels with ' / '
    (e.g. a 'reads' and a 'writes' edge between the same pair render as one 'reads / writes' arrow).
    Deterministic: pairs keep first-seen order and labels keep encounter order."""
    order: list[tuple[str, str]] = []
    labels: dict[tuple[str, str], list[str]] = {}
    for e in edges:
        key = (e.source, e.target)
        if key not in labels:
            labels[key] = []
            order.append(key)
        lbl = _label(e.label or e.relation)
        if lbl and lbl not in labels[key]:
            labels[key].append(lbl)
    return [(s, t, " / ".join(labels[(s, t)])) for (s, t) in order]


# --- renderers (pure, deterministic) -------------------------------------------------

# Mermaid node shape per kind — a database is a cylinder, a queue a subroutine box, an external
# system a hexagon, a frontend a stadium, everything else a rectangle. Shapes make the diagram
# readable at a glance without a legend.
_MERMAID_SHAPE: dict[str, tuple[str, str]] = {
    "database": ("[(", ")]"),
    "queue": ("[[", "]]"),
    "external": ("{{", "}}"),
    "frontend": ("([", "])"),
    "api": ("[/", "/]"),
}

# Deterministic per-kind palette (fill, stroke, text) applied as Mermaid ``classDef`` styles so the
# diagram looks professional and colour-codes component types at a glance. Ordering is fixed (kept in
# sync with _KINDS) so the emitted style block is byte-identical for a given architecture.
_KIND_STYLE: dict[str, tuple[str, str, str]] = {
    "application": ("#EEF2FF", "#6366F1", "#1E1B4B"),
    "service": ("#ECFDF5", "#10B981", "#064E3B"),
    "api": ("#FEF3C7", "#F59E0B", "#78350F"),
    "database": ("#E0F2FE", "#0EA5E9", "#0C4A6E"),
    "queue": ("#FAE8FF", "#D946EF", "#701A75"),
    "frontend": ("#FFE4E6", "#F43F5E", "#881337"),
    "integration": ("#F1F5F9", "#64748B", "#1E293B"),
    "external": ("#F3F4F6", "#9CA3AF", "#374151"),
}

# Above this many components the diagram is laid out top-to-bottom (TB) — a tall stack stays readable
# where a wide left-to-right (LR) flow would sprawl off-screen. Small diagrams keep the LR flow.
_TB_THRESHOLD = 8


def to_mermaid(arch: Architecture) -> str:
    """Render ``arch`` as an editable, styled Mermaid flowchart. Deterministic — same architecture
    yields byte-identical output — so it diffs cleanly and can be pasted straight into docs. Direction
    is ``TB`` for larger diagrams (>8 components) and ``LR`` otherwise; nodes are colour-coded per
    kind via ``classDef``, duplicate edges are merged, and the input is sanitised first so the output
    is always parseable."""
    arch = sanitize_architecture(arch)
    direction = "TB" if len(arch.components) > _TB_THRESHOLD else "LR"
    lines = [f"flowchart {direction}", f"  %% {_clean(arch.title)}"]

    def node_line(c: ArchComponent, indent: str) -> str:
        open_b, close_b = _MERMAID_SHAPE.get(c.kind, ("[", "]"))
        return f'{indent}{c.id}{open_b}"{_label(c.name)}"{close_b}'

    # Cluster components into subgraph boxes when the extraction assigned >=2 distinct groups — this
    # reads like a real architecture diagram (Control Plane / Data Plane) instead of a flat web. Falls
    # back to the flat list when there's no useful grouping, so ungrouped extractions don't regress.
    groups: dict[str, list[ArchComponent]] = {}
    for c in arch.components:
        groups.setdefault(c.group.strip(), []).append(c)
    real_groups = [g for g in groups if g]
    if len(real_groups) >= 2:  # noqa: PLR2004 - need at least two clusters for grouping to help
        for gi, gname in enumerate(real_groups):
            lines.append(f'  subgraph cluster_{gi} ["{_label(gname)}"]')
            for c in groups[gname]:
                lines.append(node_line(c, "    "))
            lines.append("  end")
        for c in groups.get("", []):  # ungrouped components sit outside any cluster
            lines.append(node_line(c, "  "))
    else:
        for c in arch.components:
            lines.append(node_line(c, "  "))

    for src, dst, label in _merge_edges(arch.edges):
        lines.append(f"  {src} -->|{label}| {dst}" if label else f"  {src} --> {dst}")

    # Per-kind colour styling (a lightweight legend the colours themselves provide). Only the kinds
    # actually present are emitted, in the fixed _KIND_STYLE order, so the block stays deterministic.
    present = [k for k in _KIND_STYLE if any(c.kind == k for c in arch.components)]
    if present:
        lines.append("  %% styling")
        for kind in present:
            fill, stroke, text = _KIND_STYLE[kind]
            lines.append(f"  classDef {kind} fill:{fill},stroke:{stroke},color:{text}")
        for kind in present:
            ids = ",".join(c.id for c in arch.components if c.kind == kind)
            lines.append(f"  class {ids} {kind}")
    return "\n".join(lines)


_PLANTUML_DECL: dict[str, str] = {
    "database": "database",
    "queue": "queue",
    "external": "cloud",
    "frontend": "actor",
    "api": "interface",
    "application": "component",
    "service": "component",
    "integration": "component",
}


def to_plantuml(arch: Architecture) -> str:
    """Render ``arch`` as an editable, lightly-styled PlantUML component diagram. Deterministic.
    Nodes carry the same per-kind fill colour as the Mermaid output (parity is best-effort) and
    duplicate edges are merged; the input is sanitised first so the output always parses."""
    arch = sanitize_architecture(arch)
    lines = ["@startuml", f"title {_clean(arch.title)}", "skinparam shadowing false", "skinparam roundcorner 8"]
    for c in arch.components:
        decl = _PLANTUML_DECL.get(c.kind, "component")
        fill = _KIND_STYLE.get(c.kind, ("#F1F5F9", "", ""))[0]
        lines.append(f'{decl} "{_label(c.name)}" as {c.id} {fill}')
    for src, dst, label in _merge_edges(arch.edges):
        lines.append(f"{src} --> {dst} : {label}" if label else f"{src} --> {dst}")
    lines.append("@enduml")
    return "\n".join(lines)


# --- no-model heuristic summary ------------------------------------------------------


def heuristic_summary(arch: Architecture) -> str:
    """A deterministic, model-free summary — a safe fallback when no generator is available and a
    quick sanity check in tests. Counts components by kind and names the primary flows."""
    if not arch.components:
        return "No architecture components were identified."
    by_kind: dict[str, int] = {}
    for c in arch.components:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    parts = ", ".join(f"{n} {k}{'s' if n > 1 else ''}" for k, n in sorted(by_kind.items()))
    id_to_name = {c.id: c.name for c in arch.components}
    flows = [
        f"{id_to_name.get(e.source, e.source)} {e.relation.replace('_', ' ')} {id_to_name.get(e.target, e.target)}"
        for e in arch.edges[:5]
    ]
    text = f"{arch.title}: {len(arch.components)} components ({parts})."
    if flows:
        text += " Key flows: " + "; ".join(flows) + "."
    if arch.note:
        text += " " + arch.note
    return text


# --- prompt builders (LLM extraction + vision + plain explanation) -------------------


def build_architecture_prompt(text: str, *, max_components: int = 20) -> str:
    """Prompt an LLM to extract a TYPED architecture as JSON from technical text. Mirrors the graph
    extractor's JSON contract so the same gateway + parsing style applies. ``max_components`` caps the
    number of components requested (configurable via ``arch_max_components``) so large systems keep
    their structure instead of losing nodes to a hard-coded ceiling."""
    return (
        "You are an architecture analyst. From the TECHNICAL TEXT, extract the system architecture "
        "as STRICT, COMPACT JSON with this shape:\n"
        '{"title": "...", "components": [{"name": "...", "kind": "...", "group": "..."}], '
        '"edges": [{"source": "<name>", "target": "<name>", "relation": "..."}]}\n'
        f"kind MUST be one of: {', '.join(sorted(_KINDS))}. group is the logical layer/subsystem the "
        "component belongs to (e.g. 'Control Plane', 'Data Plane', 'Frontend', 'External') — give the "
        'SAME group string to components that belong together, or "" if it has no natural group. '
        "relation is a short verb like calls, reads, writes, integrates, depends_on. Use component NAMES "
        f"in edges. BE TERSE so the JSON stays small: at MOST {max_components} components (prefer the "
        "most important, most-connected ones if there are more), do NOT emit a description field, and "
        "keep every string short. You MUST include a non-empty edges array showing how the components "
        "connect — the diagram needs the arrows, so always relate the components. "
        "Output ONLY the JSON.\n\nTECHNICAL TEXT:\n" + text
    )


def build_diagram_description_prompt() -> str:
    """Vision prompt: describe an ARCHITECTURE DIAGRAM image as text the extractor can consume."""
    return (
        "This is a system architecture diagram. List every component (box/node) with its label and "
        "type (application, service, api, database, queue, frontend, external, integration), and every "
        "connection as 'A -> B (relationship)'. Be exhaustive and literal; do not infer beyond the image."
    )


def build_explain_prompt(arch: Architecture, *, audience: str = "non-technical") -> str:
    """Prompt to EXPLAIN the architecture for a given audience. For a non-technical audience the model
    is told to avoid jargon and use plain analogies — the 'explain it so anyone understands' path."""
    body = json.dumps(arch.model_dump(), separators=(",", ":"))
    if audience == "non-technical":
        style = (
            "Explain this system's architecture to someone with NO technical background. Avoid jargon; "
            "when a technical term is unavoidable, explain it in one plain phrase. Use a simple analogy "
            "where it helps. Describe what each part does and how they work together, as a short story."
        )
    else:
        style = (
            "Explain this system's architecture for a technical reader: responsibilities, data flow, and integrations."
        )
    return f"{style}\n\nARCHITECTURE (JSON):\n{body}"


# --- model / vision wrappers (thin; gateway injected for testability) ----------------


def _extract_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON object from a model response — tolerates ```json fences and prose around the
    object by taking the outermost ``{ ... }``. Falls back to salvaging a TRUNCATED object (a verbose
    model overruns max_tokens mid-array), so complete components survive even when a trailing
    edge/description was cut off. Returns ``{}`` on anything unparseable."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        return {}
    if end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            pass
    return _repair_truncated_json(text[start:])


def _repair_truncated_json(text: str) -> dict[str, Any]:
    """Parse a JSON object that was cut off mid-generation: rewind to the last cleanly-CLOSED nested
    container (a completed component/edge), drop any dangling comma, and close the still-open outer
    brackets. Preserves every fully-emitted element before the truncation point."""
    in_str = esc = False
    stack: list[str] = []
    safe: tuple[int, list[str]] | None = None  # (index after a clean nested close, still-open closers)
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            if stack:  # a nested container just closed while an outer one is still open
                safe = (i + 1, list(stack))
    if safe is None:
        return {}
    idx, open_closers = safe
    candidate = text[:idx].rstrip().rstrip(",") + "".join(reversed(open_closers))
    try:
        obj = json.loads(candidate)
    except (ValueError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


async def extract_architecture(
    text: str,
    *,
    gateway: Any,
    title: str = "System Architecture",
    max_tokens: int = 3000,
    retries: int = 1,
    max_components: int | None = None,
) -> Architecture:
    """Extract a typed :class:`Architecture` from technical TEXT via the tenant's LLM ``gateway``
    (any object with ``async complete(prompt, *, max_tokens)``). Fail-safe: a gateway error or a
    non-JSON response yields an EMPTY architecture, never an exception — extraction degrades, the
    request lives.

    RETRIES an empty extraction: a reasoning model (sarvam-105b) intermittently spends its whole
    budget in <think> and returns no final content, so one attempt is unreliable. A hard gateway
    error (e.g. a 400 for a max_tokens above the model's own ceiling) is NOT retried — the same
    request would fail identically.

    ``max_components`` caps the diagram size (defaults to ``arch_max_components``); when the model
    returns more, the most-connected components are kept and a ``note`` records the truncation."""
    cap = max_components if max_components is not None else get_settings().arch_max_components
    prompt = build_architecture_prompt(text, max_components=cap)
    arch = Architecture(title=title)
    for _ in range(retries + 1):
        try:
            raw = await gateway.complete(prompt, max_tokens=max_tokens)
        except Exception:
            return Architecture(title=title)
        arch = parse_architecture(_extract_json(raw), title=title)
        if arch.components:
            return truncate_architecture(arch, cap)
    return arch


async def architecture_from_image(
    image: bytes, *, vision_gateway: Any, gateway: Any, title: str = "System Architecture", max_tokens: int = 1200
) -> Architecture:
    """Diagram IMAGE -> typed :class:`Architecture`: the sovereign vision model describes the diagram
    (``vision_gateway.complete_with_images``), then :func:`extract_architecture` turns that
    description into the typed model. Fail-safe at both hops."""
    try:
        description = await vision_gateway.complete_with_images(build_diagram_description_prompt(), [image])
    except Exception:
        return Architecture(title=title)
    return await extract_architecture(description, gateway=gateway, title=title, max_tokens=max_tokens)


# --- query path: explain a KB's architecture for a (non-)technical user --------------


class ArchitectureAnswer(BaseModel):
    """The answer to 'explain/generate the architecture of X': the typed model, both editable
    diagrams, a plain-language explanation, and citations back to the source documents."""

    architecture: Architecture
    mermaid: str
    plantuml: str
    explanation: str = Field(..., description="Audience-tuned prose (plain language for non-technical users).")
    citations: list[Citation] = Field(default_factory=list)


async def answer_architecture(
    kb_id: str,
    topic: str,
    *,
    audience: str = "non-technical",
    actor: str = "",
    top_k: int = 12,
    model: str | None = None,
) -> ArchitectureAnswer:
    """Retrieve the KB content about ``topic``, extract its architecture, render both diagrams, and
    EXPLAIN it for ``audience`` — the 'a non-technical user asks to understand the architecture'
    path. Grounded (extraction runs over retrieved chunks) and cited (the chunks used). Fail-safe:
    an explanation-model error falls back to the deterministic summary."""
    from svc_control_plane import wiring  # noqa: PLC0415 - local import: control-plane wiring cycle

    _ = actor  # reserved for future per-request audit
    retrieval = await (await wiring.retrieval_service_for(kb_id)).retrieve(
        RetrievalQuery(tenant_id=kb_id, text=topic, top_k=top_k, acl=[])
    )
    corpus = "\n\n".join(s.text for s in retrieval.spans)
    # Use the SELECTED model (per-question pick), else the KB's default — there is no dedicated
    # architecture model. NOTE the extraction is one LARGE (~4000-token) typed-JSON call, so a slow
    # shared endpoint (qwen/dev-aiops) can 504; the self-hosted Sarvam-30B handles it fast/reliably.
    gateway = await wiring.resolve_tenant_gateway(kb_id, model)
    # Generous, tunable budget: a large architecture's JSON must not truncate, and a reasoning
    # model needs headroom above its <think> pass to emit anything at all (1200 -> empty on sarvam).
    settings = get_settings()
    max_tokens = settings.architecture_max_tokens
    # Several tries with early-exit: a reasoning model returns an empty completion ~2/3 of the time,
    # so one attempt renders the diagram unreliably (auto-render then degrades to a chip). Retrying
    # on empty makes the strong-intent auto-render dependable; the loop stops at the first success.
    arch = await extract_architecture(
        corpus, gateway=gateway, title=topic, max_tokens=max_tokens, retries=settings.architecture_extract_retries
    )
    if not arch.summary:
        arch = arch.model_copy(update={"summary": heuristic_summary(arch)})
    # The explanation is the low-value half of the flow: it is a SECOND slow reasoning-model call
    # (~25s on sarvam) on top of the extraction, while the DIAGRAM is what the user asked for. Use
    # the instant deterministic summary by default; the LLM prose is opt-in (architecture_llm_explanation)
    # for deployments that want richer non-technical narration and can afford the latency.
    explanation = heuristic_summary(arch)
    if get_settings().architecture_llm_explanation:
        with contextlib.suppress(Exception):
            explanation = (
                await gateway.complete(build_explain_prompt(arch, audience=audience), max_tokens=max_tokens)
            ).strip() or explanation
    return ArchitectureAnswer(
        architecture=arch,
        mermaid=to_mermaid(arch),
        plantuml=to_plantuml(arch),
        explanation=explanation or heuristic_summary(arch),
        citations=make_citations(retrieval.spans, topic),
    )


# --- auto-intent: route architecture questions through the normal /query path --------


def is_architecture_query(text: str) -> bool:
    """True ONLY when the query contains the literal word 'architecture'. Deliberately simple (per
    the owner: "only the architecture word should trigger it") so a diagram is drawn when — and only
    when — the user actually asks for the architecture, not on incidental words like 'services' or
    'components'. Fail-safe regardless: the flow abstains if it can't ground an architecture."""
    return "architecture" in text.lower()


def has_architecture_potential(text: str) -> bool:
    """Disabled: the weak-signal 'offer a diagram chip' path used to fire on services / components /
    data-flow / topology and pestered ordinary questions with a diagram chip. Per the owner, ONLY the
    literal word 'architecture' triggers the architecture flow — and that case is already covered by
    :func:`is_architecture_query` (auto-render), so there is no separate potential-chip signal."""
    _ = text
    return False


async def attach_diagram(answer: GroundedAnswer, kb_id: str, topic: str, model: str | None = None) -> GroundedAnswer:
    """Layer an architecture diagram onto an already-GROUNDED answer WITHOUT touching its prose:
    extract the architecture from the KB and ride the Mermaid/PlantUML/typed model in ``structured``.
    Fail-safe — an extraction failure (or nothing groundable) leaves the answer untouched but flips
    ``diagram_available`` so the console can still offer the diagram as an on-demand chip."""
    try:
        arch_ans = await answer_architecture(kb_id, topic, model=model)
    except Exception:
        return answer.model_copy(update={"diagram_available": True})
    if not arch_ans.architecture.components:
        return answer.model_copy(update={"diagram_available": True})
    structured = dict(answer.structured or {})
    structured.update(
        {
            "mermaid": arch_ans.mermaid,
            "plantuml": arch_ans.plantuml,
            "architecture": arch_ans.architecture.model_dump(mode="json"),
            "diagram_note": arch_ans.architecture.note,
        }
    )
    # Diagram-first: for an architecture question the DIAGRAM is the deliverable, so trim the verbose
    # grounded prose to its opening paragraph (a brief, still-cited intro). BUT if the answer is a
    # Markdown table (the user asked for BOTH a table and the architecture), keep it whole — splitting
    # on paragraph breaks would chop the table — and show the table above the diagram.
    body = answer.answer or ""
    is_table = "|" in body and "---" in body
    new_answer = body if is_table else ((body.split("\n\n", 1)[0].strip()) or body)
    # Surface the truncation note in the prose too (structured.diagram_note may not be rendered by
    # every client), so a capped large system always tells the user components were dropped.
    note = arch_ans.architecture.note
    if note:
        new_answer = f"{new_answer}\n\n_{note}_" if new_answer else note
    return answer.model_copy(update={"structured": structured, "answer": new_answer})


async def architecture_grounded_answer(
    query: RetrievalQuery, *, actor: str = "", trace_id: str | None = None
) -> GroundedAnswer:
    """Answer an architecture-intent question as an ordinary :class:`GroundedAnswer`: the
    plain-language explanation is the answer text, the editable diagrams ride in ``structured``
    (``mermaid`` / ``plantuml`` / ``architecture``), and the retrieved chunks are the citations.
    Abstains — never hallucinates a diagram — when no architecture could be grounded from the KB."""
    ans = await answer_architecture(query.tenant_id, query.text, audience="non-technical", actor=actor)
    if not ans.architecture.components or not ans.citations:
        return abstain(query.tenant_id, query.text, 0.0, reason="no_architecture_grounded", trace_id=trace_id)
    confidence = max((c.score for c in ans.citations), default=0.5)
    return GroundedAnswer(
        tenant_id=query.tenant_id,
        query=query.text,
        status=AnswerStatus.GROUNDED,
        answer=ans.explanation,
        citations=ans.citations,
        confidence=confidence,
        structured={
            "mermaid": ans.mermaid,
            "plantuml": ans.plantuml,
            "architecture": ans.architecture.model_dump(),
            "diagram_note": ans.architecture.note,
        },
        model="architecture",
        trace_id=trace_id,
    )
