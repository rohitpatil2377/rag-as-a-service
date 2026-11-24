"""Prompt token budgets — RAG v2 Phase 12 ("compute token counts before calls; never exceed
``max_model_len``").

A prompt that overflows the model's window does not fail loudly. Measured on the real AEGCL
corrigendum, twice, in two different shapes:

* The Phase-9 MAP calls were given a 2048-token completion budget and eighteen of thirty-eight
  replies came back cut off mid-object — valid JSON up to the cut, unparseable as a whole.
* The Phase-9 REDUCE was handed all 322 extracted items in one prompt: 107,000 characters, and
  the answer it produced stopped mid-word with the coverage statement never reached.

In both cases the model answered, the pipeline logged success, and content was missing from an
answer that read complete. That is the failure this module exists to make impossible to reach by
accident.

WHAT IS AND IS NOT CLAIMED. The estimate is the codebase's existing chars/4 heuristic
(``llm_gateway._estimate_usage``, ``pipeline``'s whole-document budget), kept deliberately
rather than replaced by a real tokeniser: a tokeniser is per-model, would have to be shipped and
kept in step with whatever endpoint a tenant attached, and would turn a config-only model swap
into a dependency change. chars/4 is wrong in both directions — dense technical text with many
numbers and identifiers tokenises worse than 4 chars/token, prose better — so every budget here
is spent against a SAFETY FRACTION rather than to the last token, and the functions are named
``estimate_*`` so no caller mistakes them for a measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

__all__ = [
    "CHARS_PER_TOKEN",
    "SAFETY",
    "batch_to_budget",
    "estimate_tokens",
    "prompt_budget_chars",
]

#: The codebase's existing heuristic. Matches ``llm_gateway._estimate_usage`` and the
#: whole-document read budget in ``pipeline``, deliberately: three different numbers for the same
#: quantity is how two of them end up wrong.
CHARS_PER_TOKEN = 4

#: How much of a computed budget a caller may actually fill. The estimate is a heuristic that is
#: wrong in both directions, and the cost of the two errors is not symmetric: underfilling costs
#: one extra call, overfilling costs content from an answer that still reads complete.
SAFETY = 0.85

T = TypeVar("T")


def estimate_tokens(text: str) -> int:
    """Roughly how many tokens ``text`` will cost. Never zero for non-empty text."""
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


def prompt_budget_chars(*, context_window: int, max_output_tokens: int, scaffolding_chars: int = 0) -> int:
    """How many characters of VARIABLE content one prompt may carry.

    ``context_window`` is the model's limit, ``max_output_tokens`` is what the answer is allowed
    to take, and ``scaffolding_chars`` is the fixed part of the prompt — the instructions, the
    schema, the profile. What is left is for the part that grows: a section's text, or a facet's
    extracted items.

    Returns 0 when the fixed cost already exceeds the window, which is a real configuration
    (a 4,096-token window and a 4,000-token completion budget) and must not come back negative
    for a caller to slice a list with."""
    usable = int(max(0, context_window - max_output_tokens) * SAFETY)
    return max(0, usable * CHARS_PER_TOKEN - scaffolding_chars)


def batch_to_budget(
    items: Sequence[T], size_of: Callable[[T], int], budget_chars: int, *, max_items: int | None = None
) -> list[list[T]]:
    """Pack ``items`` into the fewest batches that each fit ``budget_chars``.

    GREEDY AND ORDER-PRESERVING. The order is meaningful at every call site — extracted items
    arrive in document order — so this never sorts, and a batch boundary is only ever placed
    between two adjacent items.

    AN ITEM LARGER THAN THE WHOLE BUDGET GETS ITS OWN BATCH rather than being dropped or split.
    Dropping it would lose a requirement; splitting it would cut a verbatim quote in half and the
    quote is the thing that makes the item checkable. A batch of one oversized item may still
    overflow the model, and that is the honest outcome: the alternative is silently discarding
    evidence, and a caller that wants to know can compare the batch's size to the budget.

    A budget of 0 or less puts everything in one batch — the caller has no room to work with and
    splitting into single-item batches would multiply the calls without fixing anything.

    ``max_items`` is a SECOND, independent ceiling, for the caller whose cost is not the prompt it
    sends but the reply it asks for. ``budget_chars`` bounds the INPUT; a batch that fits the
    input window can still be one the model cannot answer, because the answer has its own token
    budget and nothing here knew about it. Measured on the real AEGCL corrigendum: one facet
    arrived with 446 items, fitted the prompt window comfortably, and needed roughly 13,000
    output tokens against a 4,000-token reply budget — so about two thirds of that facet could
    never be rendered, and was silently absent from the answer. A caller that knows how much
    output one item costs passes that ceiling here."""
    if not items:
        return []
    if budget_chars <= 0:
        return [list(items)]
    batches: list[list[T]] = []
    current: list[T] = []
    used = 0
    for item in items:
        cost = max(0, size_of(item))
        full = used + cost > budget_chars or (max_items is not None and len(current) >= max_items)
        if current and full:
            batches.append(current)
            current, used = [], 0
        current.append(item)
        used += cost
    if current:
        batches.append(current)
    return batches
