"""A per-item quota is pointless if the PROMPT cannot hold it.

Live on the SOP corpus: "explain PBSOP001 ... PBSOP010" retrieved passages for all ten items --
coverage did its job -- and the answer still reported "No detailed explanation is provided in the
sources" for three of them, while asking those three individually returned full detail. The
passages were fetched and then dropped on the way into the prompt, because the prompt is capped at
the query's top_k, which defaults to 5.

That is the worst shape a failure can take here: the answer is truthful about what it was shown,
so nothing looks broken -- it just quietly under-reports.
"""

from __future__ import annotations

from svc_control_plane.pipeline import _prompt_budget_for_items

_DEFAULT_TOP_K = 5


def test_ten_items_get_a_budget_that_can_hold_them() -> None:
    assert _prompt_budget_for_items(_DEFAULT_TOP_K, items=10, available=40) >= 10


def test_the_budget_never_shrinks_what_the_caller_asked_for() -> None:
    """A caller who asked for 30 passages keeps them; coverage only ever widens."""
    assert _prompt_budget_for_items(30, items=3, available=40) == 30


def test_it_cannot_exceed_what_retrieval_actually_found() -> None:
    """Asking the prompt for more passages than exist would be a lie about the evidence."""
    assert _prompt_budget_for_items(_DEFAULT_TOP_K, items=10, available=7) == 7


def test_no_enumeration_leaves_the_budget_alone() -> None:
    assert _prompt_budget_for_items(_DEFAULT_TOP_K, items=0, available=40) == _DEFAULT_TOP_K


def test_the_budget_is_bounded_for_a_very_long_enumeration() -> None:
    """A question naming fifty identifiers must not blow the context window open."""
    assert _prompt_budget_for_items(_DEFAULT_TOP_K, items=50, available=200) <= 40
