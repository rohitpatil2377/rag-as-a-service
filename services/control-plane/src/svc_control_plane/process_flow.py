"""Workflow understanding + optimization — turn SOPs and process documents into a typed process
flow, render it, explain it, and find gaps to recommend improvements.

Mirrors :mod:`architecture` (typed model → renderers → LLM extractor → query path → auto-intent),
specialised for PROCESSES: steps with actors + kinds (start/task/decision/end/subprocess), directed
transitions with conditions, and roles. The distinctive piece is :func:`detect_gaps` — a pure,
deterministic checker for structural workflow problems (unreachable steps, dead ends, one-armed
decisions, ownerless steps, dangling transitions) — the basis of "identify gaps + recommend
improvements". This module is separate from :mod:`workflows` (the FR-F agentic runtime).

Pure, testable core; the model/vision calls are thin injected wrappers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag_core.grounding import abstain, make_citations
from rag_schemas import GroundedAnswer, RetrievalQuery
from rag_schemas.citation import Citation
from rag_schemas.common import AnswerStatus

StepKind = Literal["start", "task", "decision", "end", "subprocess", "event"]
_KINDS: frozenset[str] = frozenset(("start", "task", "decision", "end", "subprocess", "event"))


class ProcessStep(BaseModel):
    """One node in the process: a start/end marker, a task, a decision, a sub-process, or an event."""

    id: str = Field(..., description="Stable diagram id (slug of the name).")
    name: str
    kind: StepKind = Field(default="task")
    actor: str = Field(default="", description="Role/owner responsible for this step.")
    description: str = Field(default="")


class ProcessTransition(BaseModel):
    """A directed edge from one step to another, with an optional condition (decision branch)."""

    source: str
    target: str
    condition: str = Field(default="", description="Guard/branch label, e.g. 'approved' / 'rejected'.")


class WorkflowProcess(BaseModel):
    """A typed business process extracted from a document/SOP."""

    title: str = Field(default="Process")
    steps: list[ProcessStep] = Field(default_factory=list)
    transitions: list[ProcessTransition] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    summary: str = Field(default="")


class ProcessGap(BaseModel):
    """A structural problem or improvement opportunity found in the process."""

    kind: str = Field(..., description="e.g. no_start | dead_end | unreachable | one_armed_decision | no_actor.")
    detail: str
    step_id: str = Field(default="")


# --- helpers -------------------------------------------------------------------------


def slug_id(name: str) -> str:
    """Diagram-safe id from a step name (lowercase alphanumerics + underscores)."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return s or "step"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).replace('"', "'").strip()


# --- parse (loose model JSON -> validated WorkflowProcess) ---------------------------


def parse_process(data: dict[str, Any], *, title: str = "Process") -> WorkflowProcess:
    """Build a validated :class:`WorkflowProcess` from loose extractor output: slug ids, unknown
    kind -> task, dedupe steps, resolve transitions by id or name, drop transitions to unknown
    steps."""
    steps: dict[str, ProcessStep] = {}
    name_to_id: dict[str, str] = {}
    for raw in data.get("steps", []) or []:
        name = _clean(str(raw.get("name", ""))) if isinstance(raw, dict) else _clean(str(raw))
        if not name:
            continue
        sid = slug_id(name)
        kind = str(raw.get("kind", "task")).lower() if isinstance(raw, dict) else "task"
        if kind not in _KINDS:
            kind = "task"
        actor = _clean(str(raw.get("actor", ""))) if isinstance(raw, dict) else ""
        desc = _clean(str(raw.get("description", ""))) if isinstance(raw, dict) else ""
        if sid not in steps:
            steps[sid] = ProcessStep(id=sid, name=name, kind=kind, actor=actor, description=desc)
            name_to_id[name.lower()] = sid

    def _resolve(ref: str) -> str | None:
        ref = _clean(ref)
        if ref in steps:
            return ref
        if ref.lower() in name_to_id:
            return name_to_id[ref.lower()]
        sid = slug_id(ref)
        return sid if sid in steps else None

    transitions: list[ProcessTransition] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in data.get("transitions", []) or []:
        if not isinstance(raw, dict):
            continue
        src, dst = _resolve(str(raw.get("source", ""))), _resolve(str(raw.get("target", "")))
        cond = _clean(str(raw.get("condition", "")))
        if not src or not dst:
            continue
        key = (src, dst, cond)
        if key in seen:
            continue
        seen.add(key)
        transitions.append(ProcessTransition(source=src, target=dst, condition=cond))

    roles = sorted({_clean(str(r)) for r in (data.get("roles", []) or []) if _clean(str(r))})
    roles = roles or sorted({s.actor for s in steps.values() if s.actor})
    return WorkflowProcess(
        title=_clean(str(data.get("title", ""))) or title,
        steps=list(steps.values()),
        transitions=transitions,
        roles=roles,
        summary=_clean(str(data.get("summary", ""))),
    )


# --- gap detection (pure, deterministic — the optimization edge) ---------------------


def detect_gaps(process: WorkflowProcess) -> list[ProcessGap]:  # noqa: PLR0912 - one branch per gap kind
    """Find structural problems in the process, deterministically. Covers: no start / no end,
    steps unreachable from a start, dead-end steps (non-end with no outgoing), one-armed decisions
    (a decision with fewer than two outgoing branches), ownerless task/subprocess steps, and
    transitions pointing at unknown steps. This is the evidence a recommendation is built on."""
    gaps: list[ProcessGap] = []
    ids = {s.id for s in process.steps}
    out: dict[str, list[ProcessTransition]] = {sid: [] for sid in ids}
    for t in process.transitions:
        if t.source not in ids or t.target not in ids:
            gaps.append(
                ProcessGap(kind="dangling_transition", detail=f"{t.source} -> {t.target} references an unknown step")
            )
            continue
        out[t.source].append(t)

    starts = [s for s in process.steps if s.kind == "start"]
    ends = [s for s in process.steps if s.kind == "end"]
    if not starts:
        gaps.append(ProcessGap(kind="no_start", detail="The process has no start step."))
    if not ends:
        gaps.append(ProcessGap(kind="no_end", detail="The process has no end step."))

    # Reachability from any start (BFS).
    reachable: set[str] = set()
    frontier = [s.id for s in starts]
    while frontier:
        cur = frontier.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        frontier.extend(t.target for t in out.get(cur, []))
    if starts:
        for s in process.steps:
            if s.id not in reachable:
                gaps.append(
                    ProcessGap(kind="unreachable", detail=f"'{s.name}' is unreachable from the start.", step_id=s.id)
                )

    for s in process.steps:
        outgoing = out.get(s.id, [])
        if s.kind != "end" and not outgoing:
            gaps.append(ProcessGap(kind="dead_end", detail=f"'{s.name}' has no next step (dead end).", step_id=s.id))
        if s.kind == "decision" and len(outgoing) < 2:  # noqa: PLR2004 - a decision needs >=2 branches
            gaps.append(
                ProcessGap(
                    kind="one_armed_decision",
                    detail=f"Decision '{s.name}' has fewer than two branches.",
                    step_id=s.id,
                )
            )
        if s.kind in ("task", "subprocess") and not s.actor:
            gaps.append(ProcessGap(kind="no_actor", detail=f"'{s.name}' has no responsible role/owner.", step_id=s.id))
    return gaps


# --- renderers (pure) ----------------------------------------------------------------

_MERMAID_SHAPE: dict[str, tuple[str, str]] = {
    "start": ("([", "])"),
    "end": ("([", "])"),
    "decision": ("{", "}"),
    "subprocess": ("[[", "]]"),
    "event": (">", "]"),
}


def to_mermaid(process: WorkflowProcess) -> str:
    """Render the process as an editable Mermaid ``flowchart TD`` (top-down): decisions are
    diamonds, start/end stadiums, sub-processes subroutines. Deterministic."""
    lines = ["flowchart TD", f"  %% {_clean(process.title)}"]
    for s in process.steps:
        open_b, close_b = _MERMAID_SHAPE.get(s.kind, ("[", "]"))
        label = _clean(f"{s.name}" + (f" ({s.actor})" if s.actor and s.kind in ("task", "subprocess") else ""))
        lines.append(f'  {s.id}{open_b}"{label}"{close_b}')
    for t in process.transitions:
        cond = _clean(t.condition)
        lines.append(f"  {t.source} -->|{cond}| {t.target}" if cond else f"  {t.source} --> {t.target}")
    return "\n".join(lines)


def to_plantuml(process: WorkflowProcess) -> str:
    """Render the process as an editable PlantUML state diagram. Deterministic."""
    lines = ["@startuml", f"title {_clean(process.title)}"]
    for s in process.steps:
        lines.append(f'state "{_clean(s.name)}" as {s.id}')
    for s in process.steps:
        if s.kind == "start":
            lines.append(f"[*] --> {s.id}")
        if s.kind == "end":
            lines.append(f"{s.id} --> [*]")
    for t in process.transitions:
        cond = _clean(t.condition)
        lines.append(f"{t.source} --> {t.target} : {cond}" if cond else f"{t.source} --> {t.target}")
    lines.append("@enduml")
    return "\n".join(lines)


# --- prompt builders -----------------------------------------------------------------


def build_process_prompt(text: str) -> str:
    """Prompt an LLM to extract a typed process from an SOP / process document as STRICT JSON."""
    return (
        "You are a business-process analyst. From the SOP / PROCESS TEXT, extract the workflow as "
        "STRICT JSON:\n"
        '{"title":"...","roles":["..."],"steps":[{"name":"...","kind":"...","actor":"...",'
        '"description":"..."}],"transitions":[{"source":"<step name>","target":"<step name>",'
        '"condition":"..."}]}\n'
        f"kind MUST be one of: {', '.join(sorted(_KINDS))}. Use step NAMES in transitions; put the "
        "branch label (e.g. approved/rejected) in condition. Include a start and an end. Output ONLY "
        "the JSON.\n\nPROCESS TEXT:\n" + text
    )


def build_diagram_description_prompt() -> str:
    """Vision prompt: describe a WORKFLOW/flowchart image as text the extractor can consume."""
    return (
        "This is a workflow / flowchart / process diagram. List every step (box) with its label and "
        "type (start, task, decision, end, subprocess, event) and the responsible role if shown, and "
        "every arrow as 'A -> B (condition)'. Be exhaustive and literal."
    )


def build_explain_prompt(process: WorkflowProcess, gaps: list[ProcessGap], *, audience: str = "business") -> str:
    """Prompt to EXPLAIN the process (steps, dependencies, business logic) and RECOMMEND improvements
    from the detected gaps — the understand + optimize output."""
    body = json.dumps(process.model_dump(), separators=(",", ":"))
    gap_lines = "; ".join(f"{g.kind}: {g.detail}" for g in gaps) or "none found"
    lens = (
        "for a business audience in plain terms" if audience == "business" else "for an operations/engineering audience"
    )
    return (
        f"Explain this business process {lens}: what each step does, who owns it, the dependencies and "
        "decision logic, as a short narrative. Then, using the DETECTED GAPS, recommend concrete "
        f"improvements (each: the gap and a fix).\n\nPROCESS (JSON):\n{body}\n\nDETECTED GAPS: {gap_lines}"
    )


# --- model / vision wrappers (gateway injected) --------------------------------------


def _extract_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON object from a model response (strips ```fences/prose)."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


async def extract_process(
    text: str, *, gateway: Any, title: str = "Process", max_tokens: int = 1500
) -> WorkflowProcess:
    """Extract a typed process from SOP text via the tenant LLM gateway. Fail-safe: an error or
    non-JSON response yields an empty process."""
    try:
        raw = await gateway.complete(build_process_prompt(text), max_tokens=max_tokens)
    except Exception:
        return WorkflowProcess(title=title)
    return parse_process(_extract_json(raw), title=title)


async def process_from_image(
    image: bytes, *, vision_gateway: Any, gateway: Any, title: str = "Process", max_tokens: int = 1500
) -> WorkflowProcess:
    """Flowchart IMAGE -> typed process: the vision model describes the diagram, then the extractor
    structures it. Fail-safe at both hops."""
    try:
        description = await vision_gateway.complete_with_images(build_diagram_description_prompt(), [image])
    except Exception:
        return WorkflowProcess(title=title)
    return await extract_process(description, gateway=gateway, title=title, max_tokens=max_tokens)


# --- query path: understand + optimize a KB's process --------------------------------


class ProcessAnswer(BaseModel):
    """The answer to 'explain/optimize the process of X': the typed process, both diagrams, the
    detected gaps, a plain-language explanation + recommendations, and citations."""

    process: WorkflowProcess
    mermaid: str
    plantuml: str
    gaps: list[ProcessGap] = Field(default_factory=list)
    explanation: str = ""
    citations: list[Citation] = Field(default_factory=list)


async def answer_process(
    kb_id: str, topic: str, *, audience: str = "business", actor: str = "", top_k: int = 12
) -> ProcessAnswer:
    """Retrieve the KB content about ``topic``, extract its process, render both diagrams, detect
    gaps, and EXPLAIN the process + RECOMMEND improvements. Grounded + cited. Fail-safe."""
    from svc_control_plane import wiring  # noqa: PLC0415 - control-plane wiring cycle

    _ = actor
    retrieval = await (await wiring.retrieval_service_for(kb_id)).retrieve(
        RetrievalQuery(tenant_id=kb_id, text=topic, top_k=top_k, acl=[])
    )
    corpus = "\n\n".join(s.text for s in retrieval.spans)
    gateway = await wiring.resolve_tenant_gateway(kb_id)
    process = await extract_process(corpus, gateway=gateway, title=topic)
    gaps = detect_gaps(process)
    try:
        explanation = (
            await gateway.complete(build_explain_prompt(process, gaps, audience=audience), max_tokens=900)
        ).strip()
    except Exception:
        explanation = _heuristic_explanation(process, gaps)
    return ProcessAnswer(
        process=process,
        mermaid=to_mermaid(process),
        plantuml=to_plantuml(process),
        gaps=gaps,
        explanation=explanation or _heuristic_explanation(process, gaps),
        citations=make_citations(retrieval.spans, topic),
    )


def _heuristic_explanation(process: WorkflowProcess, gaps: list[ProcessGap]) -> str:
    """A no-model summary + gap list — the safe fallback."""
    if not process.steps:
        return "No process steps were identified."
    head = f"{process.title}: {len(process.steps)} steps across {len(process.roles) or 'unspecified'} role(s)."
    if gaps:
        head += " Gaps: " + "; ".join(f"{g.kind} — {g.detail}" for g in gaps[:6]) + "."
    else:
        head += " No structural gaps detected."
    return head


# --- auto-intent: route process questions through the normal /query path -------------

_PROC_VERB_RE = re.compile(
    r"\b(explain|describe|show|map|diagram|optimi[sz]e|improve|understand|how does|walk me through|steps)\b", re.I
)


def is_process_query(text: str) -> bool:
    """True when a question asks to explain/optimize a PROCESS / WORKFLOW / SOP — so the normal query
    path can route it here. Heuristic: it mentions process/workflow/SOP/procedure with an
    explain-or-optimize intent."""
    low = text.lower()
    if not any(w in low for w in ("process", "workflow", "sop", "procedure")):
        return False
    return bool(_PROC_VERB_RE.search(low)) or "process of" in low or "workflow of" in low


async def process_grounded_answer(
    query: RetrievalQuery, *, actor: str = "", trace_id: str | None = None
) -> GroundedAnswer:
    """Answer a process-intent question as an ordinary GroundedAnswer: the explanation + recommendations
    are the answer text, the diagrams + gaps ride in ``structured``, chunks are the citations. Abstains
    when nothing grounds."""
    ans = await answer_process(query.tenant_id, query.text, audience="business", actor=actor)
    if not ans.process.steps or not ans.citations:
        return abstain(query.tenant_id, query.text, 0.0, reason="no_process_grounded", trace_id=trace_id)
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
            "process": ans.process.model_dump(),
            "gaps": [g.model_dump() for g in ans.gaps],
        },
        model="process_flow",
        trace_id=trace_id,
    )
