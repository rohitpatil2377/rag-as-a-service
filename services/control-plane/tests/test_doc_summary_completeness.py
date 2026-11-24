"""Fix 2: per-doc completeness is aggregated into the combined doc-summary answer.

Unit-tests the pure aggregation helper directly (no infra), matching how the doc-summary path folds
each per-doc summary's completeness signal into one combined signal."""

from __future__ import annotations

from svc_control_plane.pipeline import _aggregate_doc_completeness


def test_disabled_returns_none_byte_identical() -> None:
    assert _aggregate_doc_completeness(3, [], enabled=False) is None
    assert _aggregate_doc_completeness(3, ["Handbook"], enabled=False) is None


def test_all_complete_is_complete() -> None:
    sig = _aggregate_doc_completeness(3, [], enabled=True)
    assert sig is not None
    assert sig.complete is True
    assert sig.reason is None
    assert (sig.covered, sig.expected, sig.truncated) == (3, 3, False)


def test_some_incomplete_names_the_docs() -> None:
    sig = _aggregate_doc_completeness(3, ["Handbook", "Policy"], enabled=True)
    assert sig is not None
    assert sig.complete is False
    assert sig.reason is not None
    assert "Handbook" in sig.reason and "Policy" in sig.reason
    assert (sig.covered, sig.expected, sig.truncated) == (1, 3, True)
