"""Prompt token budgets (Phase 12's "never exceed max_model_len").

The reason this exists is that an overflowing prompt does NOT fail loudly — measured twice on
the real AEGCL corrigendum, in two different shapes, and both times the model answered, the
pipeline logged success, and content was missing from an answer that read complete.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_core.budget import CHARS_PER_TOKEN, SAFETY, batch_to_budget, estimate_tokens, prompt_budget_chars


def test_an_empty_string_costs_nothing_and_anything_else_costs_at_least_one() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("x" * 400) == 400 // CHARS_PER_TOKEN


def test_the_budget_is_what_is_left_after_the_output_and_the_scaffolding() -> None:
    budget = prompt_budget_chars(context_window=32_000, max_output_tokens=4_000, scaffolding_chars=2_000)

    expected = int((32_000 - 4_000) * SAFETY) * CHARS_PER_TOKEN - 2_000
    assert budget == expected


def test_the_budget_is_spent_against_a_safety_fraction_not_to_the_last_token() -> None:
    """chars/4 is wrong in both directions, and the two errors do not cost the same: underfilling
    costs one extra call, overfilling costs content from an answer that still reads complete."""
    full = prompt_budget_chars(context_window=32_000, max_output_tokens=0)
    assert full < 32_000 * CHARS_PER_TOKEN
    assert SAFETY < 1.0


def test_a_window_already_spent_on_the_output_leaves_zero_not_a_negative() -> None:
    """A real configuration — a 4,096-token window with a 4,000-token completion budget — and a
    caller is about to slice a list with this."""
    assert prompt_budget_chars(context_window=4_096, max_output_tokens=4_096) == 0
    assert prompt_budget_chars(context_window=1_000, max_output_tokens=9_999) == 0
    assert prompt_budget_chars(context_window=4_096, max_output_tokens=0, scaffolding_chars=10**9) == 0


# --- batching -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Item:
    name: str
    size: int


def _size(item: _Item) -> int:
    return item.size


def _names(batches: list[list[_Item]]) -> list[list[str]]:
    return [[i.name for i in b] for b in batches]


def test_items_pack_into_the_fewest_batches_that_fit() -> None:
    items = [_Item("a", 40), _Item("b", 40), _Item("c", 40), _Item("d", 40)]
    assert _names(batch_to_budget(items, _size, 100)) == [["a", "b"], ["c", "d"]]


def test_everything_fitting_is_one_batch() -> None:
    items = [_Item("a", 10), _Item("b", 10)]
    assert _names(batch_to_budget(items, _size, 1000)) == [["a", "b"]]


def test_the_order_is_never_changed() -> None:
    """The order is meaningful at every call site — extracted items arrive in document order —
    so a batch boundary only ever lands between two ADJACENT items."""
    items = [_Item(str(i), 60) for i in range(5)]
    batches = batch_to_budget(items, _size, 100)
    assert [i.name for b in batches for i in b] == ["0", "1", "2", "3", "4"]


def test_an_item_larger_than_the_whole_budget_gets_its_own_batch() -> None:
    """Dropping it would lose a requirement; splitting it would cut a verbatim quote in half,
    and the quote is the thing that makes the item checkable. A batch of one oversized item may
    still overflow the model, and that is the honest outcome."""
    items = [_Item("small", 10), _Item("huge", 10_000), _Item("also_small", 10)]

    assert _names(batch_to_budget(items, _size, 100)) == [["small"], ["huge"], ["also_small"]]


def test_no_items_is_no_batches() -> None:
    assert batch_to_budget([], _size, 100) == []


def test_no_budget_puts_everything_in_one_batch() -> None:
    """The caller has no room to work with, and splitting into single-item batches would
    multiply the calls without fixing anything."""
    items = [_Item("a", 10), _Item("b", 10)]
    assert _names(batch_to_budget(items, _size, 0)) == [["a", "b"]]
    assert _names(batch_to_budget(items, _size, -5)) == [["a", "b"]]


def test_a_negative_size_cannot_shrink_the_running_total() -> None:
    items = [_Item("a", -100), _Item("b", 60), _Item("c", 60)]
    assert _names(batch_to_budget(items, _size, 100)) == [["a", "b"], ["c"]]


def test_a_batch_is_also_capped_by_how_much_can_be_rendered() -> None:
    """The second ceiling, and the bug it exists for.

    Packing by prompt size alone produced batches the model could not answer. Measured on the
    real AEGCL corrigendum: one facet arrived with 446 items, fitted the 32k window with room to
    spare, and needed ~13,000 output tokens against a 4,000-token reply budget — so two thirds of
    it could never be rendered and was silently missing from the answer."""
    items = ["x"] * 446
    batches = batch_to_budget(items, lambda _: 1, 10_000, max_items=100)

    assert sum(len(b) for b in batches) == 446, "no item may be dropped by capping"
    assert max(len(b) for b in batches) <= 100
    assert len(batches) == 5


def test_the_item_cap_is_independent_of_the_character_budget() -> None:
    """Whichever ceiling is hit first closes the batch — they bound different things."""
    # Character budget bites first: a batch closes BEFORE it would overflow, so 3 of 30 chars
    # fit a 100-char budget and the 4th starts a new batch. The item cap is far out of reach.
    by_chars = batch_to_budget(["x"] * 10, lambda _: 30, 100, max_items=1000)
    assert [len(b) for b in by_chars] == [3, 3, 3, 1]

    # Item cap bites first: the same ten items, generous characters, a cap of 2.
    by_items = batch_to_budget(["x"] * 10, lambda _: 1, 10_000, max_items=2)
    assert [len(b) for b in by_items] == [2, 2, 2, 2, 2]


def test_omitting_the_cap_keeps_the_old_behaviour_exactly() -> None:
    """Callers that only care about the prompt must be unaffected."""
    items = ["x"] * 50
    assert batch_to_budget(items, lambda _: 1, 10_000) == batch_to_budget(items, lambda _: 1, 10_000, max_items=None)
    assert len(batch_to_budget(items, lambda _: 1, 10_000)) == 1
