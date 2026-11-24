"""Guardrails (FR-G1 prompt-injection · FR-G3 toxicity) — interfaces + heuristic defaults.

A guardrail inspects text and returns a :class:`GuardrailVerdict`. Input guards run on the
user query BEFORE any model call (block ⇒ never call the model); output guards run on the
generated answer (block ⇒ drop it). A block surfaces as ``ABSTAINED`` with a
``blocked_*`` reason and a distinct ``AuditAction.GUARDRAIL`` event (ADR-0010).

The heuristics here are deterministic, dependency-free, and conservative (default ON).
They are a FLOOR, not a ceiling — production should swap in NeMo Guardrails / LlamaGuard /
Detoxify via the LLM/guard backend (opt-in). Do not tune the wordlists to look thorough;
the real classifier is the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GuardrailVerdict:
    allowed: bool
    category: str | None = None  # e.g. "prompt_injection", "toxicity"
    reason: str = ""


@runtime_checkable
class Guardrail(Protocol):
    async def check(self, text: str) -> GuardrailVerdict: ...


# --- Prompt-injection (input) — FR-G1 ---------------------------------------
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+instructions",
        r"disregard\s+(the\s+)?(previous|above|system|prior)",
        r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions)",
        r"(print|show|repeat)\s+(your\s+)?(system\s+)?prompt",
        r"you\s+are\s+now\s+",
        r"pretend\s+to\s+be\s+",
        r"developer\s+mode",
        r"jailbreak",
        r"override\s+(the\s+)?(safety|guard|rules)",
    )
]


class HeuristicInjectionGuard:
    """Pattern-based prompt-injection detector (input). Conservative, deterministic."""

    async def check(self, text: str) -> GuardrailVerdict:
        for pat in _INJECTION_PATTERNS:
            if pat.search(text):
                return GuardrailVerdict(
                    allowed=False, category="prompt_injection", reason=f"matched:{pat.pattern[:40]}"
                )
        return GuardrailVerdict(allowed=True)


# --- Toxicity (output) — FR-G3 ----------------------------------------------
# Minimal illustrative wordlist — a PLACEHOLDER for a real moderation model.
# TODO(Phase-1): replace with LlamaGuard / Detoxify / NeMo content moderation.
_TOXIC_TERMS = frozenset({"idiot", "moron", "scum", "kill yourself", "hate you"})


class HeuristicToxicityGuard:
    """Wordlist-based toxicity screen (output). Placeholder; deterministic."""

    async def check(self, text: str) -> GuardrailVerdict:
        low = text.lower()
        for term in _TOXIC_TERMS:
            if term in low:
                return GuardrailVerdict(allowed=False, category="toxicity", reason=f"matched:{term}")
        return GuardrailVerdict(allowed=True)
