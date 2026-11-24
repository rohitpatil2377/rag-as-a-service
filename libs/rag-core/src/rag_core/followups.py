"""Suggested follow-up questions (FR-I8) — pure logic, no framework.

After a grounded answer, offer 2-3 questions the user might ask next — generated ONLY from
the RETRIEVED context, never the model's parametric knowledge (so a follow-up can't lead the
user off the grounded path). Two backends behind one seam (heuristic default / LLM opt-in):
* :class:`HeuristicFollowupGenerator` — entity-anchored, deterministic, model-free, CI-safe;
* the LLM backend (in ``rag_clients``) prompts the KB's model over the same context.

Follow-ups are NON-AUTHORITATIVE hints — the platform never answers them itself here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_core.graph import extract_entities

_MAX_CONTEXT_CHARS = 2000


@runtime_checkable
class FollowupGenerator(Protocol):
    async def suggest(self, query: str, context: str, count: int) -> list[str]: ...


def build_followups_prompt(query: str, context: str, count: int) -> str:
    """Prompt the LLM backend for follow-ups grounded strictly in the context."""
    return (
        f'A user asked: "{query.strip()}".\n'
        f"Using ONLY the CONTEXT below (never outside knowledge), suggest {count} short follow-up "
        "questions the user might ask next. Each must be answerable from the context. Return one "
        "question per line, no numbering, no preamble.\n\n"
        f"CONTEXT:\n{context.strip()[:_MAX_CONTEXT_CHARS]}\n\nFOLLOW-UPS:"
    )


def parse_followups(raw: str, count: int) -> list[str]:
    """Extract clean question lines from a model reply (deduped, capped)."""
    out: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        q = line.strip().lstrip("-*0123456789. )").strip()
        if len(q) < 8 or "?" not in q:  # noqa: PLR2004 - a real question has length + a mark
            continue
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= count:
            break
    return out


def heuristic_followups(query: str, context: str, count: int) -> list[str]:
    """Model-free follow-ups: anchor on salient entities in the context that the user's own
    question did NOT already name. Domain-neutral (works on any corpus)."""
    asked = {e.lower() for e in extract_entities(query)}
    out: list[str] = []
    for ent in extract_entities(context):
        el = ent.lower()
        # The context is PII-masked; don't anchor a follow-up on the redaction token itself
        # ("What does the document say about REDACTED?" — QA ISS-013).
        if el in asked or "redacted" in el:
            continue
        out.append(f"What does the document say about {ent}?")
        if len(out) >= count:
            break
    return out


class HeuristicFollowupGenerator:
    """Deterministic entity-anchored follow-ups (default; no model, CI-safe)."""

    async def suggest(self, query: str, context: str, count: int) -> list[str]:
        return heuristic_followups(query, context, count)
