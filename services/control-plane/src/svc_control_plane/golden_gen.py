"""Synthetic golden-set generation (FR-G8) — a quality baseline with zero tenant effort.

Samples chunks across a KB's documents and drafts evaluation Q&A pairs from them: the KB's
own model when one is connected (real questions + expected facts), else a deterministic
heuristic fallback (entity-anchored) so it works on ANY corpus with no model. The result is
a DRAFT — the tenant reviews/edits it in the Quality page before saving into the existing
golden-set storage. Nothing is auto-saved.
"""

from __future__ import annotations

import json
import re

from rag_clients.factory import get_catalog
from rag_common import get_logger
from rag_core.graph import extract_entities
from rag_core.parent_child import not_parent
from rag_schemas import Chunk
from svc_control_plane import wiring
from svc_control_plane.quality import EvalQuestion

_log = get_logger(__name__)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_MIN_CHUNK_LEN = 80  # skip tiny/boilerplate chunks — too little to ask about
_MAX_COUNT = 20


def build_golden_prompt(text: str) -> str:
    """Prompt the KB model for ONE factual Q&A grounded in the passage."""
    return (
        "From the PASSAGE below, write ONE clear factual question a user might ask, and the "
        "SHORT exact answer as stated in the passage. The answer MUST be a fact present in the "
        'passage (a phrase, name, or number). Return only JSON: {"question": "...", "expected": "..."}.\n\n'
        f"PASSAGE:\n{text.strip()[:1200]}\n\nJSON:"
    )


def parse_golden(raw: str) -> EvalQuestion | None:
    """Parse a model reply into an EvalQuestion, or None if it isn't usable."""
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    q = str(data.get("question", "")).strip()
    expected = str(data.get("expected", "")).strip()
    if len(q) < 8 or "?" not in q:  # noqa: PLR2004 - a real question has some length + a mark
        return None
    return EvalQuestion(question=q, expected=expected or None)


def heuristic_golden(chunk: Chunk) -> EvalQuestion | None:
    """Model-free fallback: anchor a question on the chunk's most salient entity (domain-
    neutral — works on legal, medical, finance, HR, engineering text alike)."""
    entities = extract_entities(chunk.text)
    if not entities:
        return None
    anchor = entities[0]
    return EvalQuestion(question=f"What does the document say about {anchor}?", expected=anchor)


async def _sample_chunks(kb_id: str, count: int) -> list[Chunk]:
    """Pick up to ``count`` substantive chunks spread across the KB's documents (round-robin
    so a single large doc doesn't dominate the sample)."""
    docs = await get_catalog().list_documents(kb_id)
    per_doc: list[list[Chunk]] = []
    for d in docs:
        # FR-C2: never draw a golden question from a parent context record (it is not a search unit).
        raw = not_parent(await get_catalog().document_chunks(kb_id, d.doc_id))
        chunks = [c for c in raw if len(c.text) >= _MIN_CHUNK_LEN]
        if chunks:
            per_doc.append(chunks)
    picked: list[Chunk] = []
    idx = 0
    while per_doc and len(picked) < count:
        made_progress = False
        for chunks in per_doc:
            if idx < len(chunks):
                picked.append(chunks[idx])
                made_progress = True
                if len(picked) >= count:
                    break
        if not made_progress:
            break
        idx += 1
    return picked


async def generate_golden(kb_id: str, count: int = 5) -> list[EvalQuestion]:
    """Draft up to ``count`` golden Q&A pairs from the KB's documents (FR-G8). Uses the KB's
    model per chunk, falling back to the heuristic when there's no usable JSON (or no model)."""
    count = max(1, min(_MAX_COUNT, count))
    chunks = await _sample_chunks(kb_id, count)
    if not chunks:
        return []
    gateway = await wiring.resolve_tenant_gateway(kb_id)
    out: list[EvalQuestion] = []
    seen: set[str] = set()
    for chunk in chunks:
        item: EvalQuestion | None = None
        try:
            # 3000, not a small cap: the tenant's own model can be a reasoning model (e.g.
            # sarvam-105b) that spends tokens on <think> before the JSON line -- a low cap
            # (256 previously) starves it into empty content, which silently falls through to
            # heuristic_golden below every time (same failure mode as FR-I8 follow-ups).
            raw = await gateway.complete(build_golden_prompt(chunk.text), max_tokens=3000, temperature=0.0)
            item = parse_golden(raw)
        except Exception as exc:  # a flaky model never blocks generation — fall back
            _log.warning("golden_gen_model_failed_fallback", kb_id=kb_id, error=str(exc)[:160])
        if item is None:
            item = heuristic_golden(chunk)
        if item and item.question.lower() not in seen:
            seen.add(item.question.lower())
            out.append(item)
    _log.info("golden_generated", kb_id=kb_id, drafted=len(out), sampled=len(chunks))
    return out
