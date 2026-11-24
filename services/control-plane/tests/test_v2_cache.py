"""The two upstream caches on the RAG v2 query path (Phase 12).

The claim that makes these safe as plain LRUs is that neither caches a JUDGEMENT. A router
decision is a function of the question's words and the cards in scope; a section's reranker
score is a function of (probe, candidate text). Both are deterministic for fixed inputs, so a
hit cannot serve something an operator's threshold change should have stopped — which is why
``semantic_cache`` has to re-check its stored gate (``still_passes``) and these do not.

What the tests therefore pin is: the key covers every input, a change of any input misses, and
the KB's version counter clears both together.
"""

from __future__ import annotations

import pytest
from svc_control_plane.semantic_cache import _version_key
from svc_control_plane.v2_cache import V2Cache

from rag_common import get_coordinator, get_settings
from rag_core.router import RouterDecision, RouterIntent

_KB = "cachekb"
_QUERY = "give me the compliance details from this document"


@pytest.fixture
def cache(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> V2Cache:
    for flag in ("RAG_V2_ROUTER_CACHE_ENABLED", "RAG_V2_SCORE_CACHE_ENABLED"):
        monkeypatch.setenv(flag, "true")
    get_settings.cache_clear()
    return V2Cache()


def _decision(intent: RouterIntent = "aggregate", docs: tuple[str, ...] = ("d1",)) -> RouterDecision:
    return RouterDecision(intent=intent, doc_ids=docs, facets=(("d1", "certifications"),))


# --- the router decision cache --------------------------------------------------------------------


def test_a_stored_decision_comes_back_whole(cache: V2Cache) -> None:
    """The DECISION OBJECT is cached, not its JSON. Re-parsing stored JSON would need the scope
    rebuilt, and ``parse_router_decision`` validates against it — handed an empty scope it drops
    every doc_id and facet, turning a hit into a decision that scopes retrieval to nothing."""
    cache.put_router(_KB, _QUERY, ("d1",), _decision())

    hit = cache.router(_KB, _QUERY, ("d1",))

    assert hit is not None
    assert hit.intent == "aggregate"
    assert hit.doc_ids == ("d1",), "the scope survived"
    assert hit.facets == (("d1", "certifications"),), "and so did the facets"


def test_a_miss_is_none(cache: V2Cache) -> None:
    assert cache.router(_KB, _QUERY, ("d1",)) is None


def test_the_query_is_normalised_but_the_doc_set_is_part_of_the_key(cache: V2Cache) -> None:
    """The same question over a different scope is a different decision. A cache keyed on the
    query alone would scope one question's answer to another question's documents."""
    cache.put_router(_KB, _QUERY, ("d1",), _decision())

    assert cache.router(_KB, f"  {_QUERY.upper()}  ", ("d1",)) is not None, "case and padding are not meaning"
    assert cache.router(_KB, _QUERY, ("d2",)) is None, "a different scope misses"
    assert cache.router(_KB, _QUERY, ("d1", "d2")) is None, "so does a wider one"
    assert cache.router(_KB, "a different question", ("d1",)) is None


def test_the_doc_set_is_a_set_not_a_sequence(cache: V2Cache) -> None:
    cache.put_router(_KB, _QUERY, ("d1", "d2"), _decision())
    assert cache.router(_KB, _QUERY, ("d2", "d1")) is not None


def test_one_kb_cannot_read_anothers_decision(cache: V2Cache) -> None:
    cache.put_router(_KB, _QUERY, ("d1",), _decision())
    assert cache.router("other-kb", _QUERY, ("d1",)) is None


def test_the_router_cache_can_be_turned_off(cache: V2Cache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_V2_ROUTER_CACHE_ENABLED", "false")
    get_settings.cache_clear()

    cache.put_router(_KB, _QUERY, ("d1",), _decision())
    assert cache.router(_KB, _QUERY, ("d1",)) is None


def test_the_router_cache_is_bounded(cache: V2Cache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_V2_ROUTER_CACHE_SIZE", "2")
    get_settings.cache_clear()

    for i in range(4):
        cache.put_router(_KB, f"question {i}", (), _decision())

    assert cache.router(_KB, "question 0", ()) is None, "evicted"
    assert cache.router(_KB, "question 3", ()) is not None, "kept"


# --- the section score cache ----------------------------------------------------------------------

_TEXTS = {"s1": "21. Switches — UL 62368", "s2": "22. Firewalls — HBOM and SBOM"}


def test_scores_come_back_per_section(cache: V2Cache) -> None:
    cache.put_scores(_KB, "certifications", _TEXTS, {"s1": 0.9, "s2": 0.1})

    assert cache.scores(_KB, "certifications", _TEXTS) == {"s1": 0.9, "s2": 0.1}


def test_a_partial_hit_returns_only_what_is_known(cache: V2Cache) -> None:
    """This is the shape the caller depends on: it scores the misses and merges, so a cache that
    returned nothing on a partial hit would re-score sections it already knew."""
    cache.put_scores(_KB, "certifications", _TEXTS, {"s1": 0.9})

    assert cache.scores(_KB, "certifications", _TEXTS) == {"s1": 0.9}


def test_a_different_probe_misses(cache: V2Cache) -> None:
    cache.put_scores(_KB, "certifications", _TEXTS, {"s1": 0.9})
    assert cache.scores(_KB, "supply chain", _TEXTS) == {}


def test_changed_candidate_text_misses_even_at_the_same_section_id(cache: V2Cache) -> None:
    """Keyed on what the cross-encoder READ, not on the section id: a section whose summary was
    re-derived must miss even though its id did not change."""
    cache.put_scores(_KB, "certifications", _TEXTS, {"s1": 0.9})

    rewritten = {"s1": "21. Switches — re-summarised by a newer model"}
    assert cache.scores(_KB, "certifications", rewritten) == {}


def test_the_score_cache_can_be_turned_off(cache: V2Cache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_V2_SCORE_CACHE_ENABLED", "false")
    get_settings.cache_clear()

    cache.put_scores(_KB, "certifications", _TEXTS, {"s1": 0.9})
    assert cache.scores(_KB, "certifications", _TEXTS) == {}


# --- invalidation ---------------------------------------------------------------------------------


def test_bumping_the_kbs_version_clears_both_caches(cache: V2Cache) -> None:
    """The SAME signal ``semantic_cache`` uses, bumped on any ingest, delete or reindex. One
    counter per KB means one invalidation event clears every derived cache it feeds — a separate
    namespace would let the answer cache and these drift apart after a reindex, the answer gone
    and the scores that produced it still live."""
    cache.put_router(_KB, _QUERY, ("d1",), _decision())
    cache.put_scores(_KB, "certifications", _TEXTS, {"s1": 0.9})
    assert cache.router(_KB, _QUERY, ("d1",)) is not None

    get_coordinator().bump(_version_key(_KB))

    assert cache.router(_KB, _QUERY, ("d1",)) is None
    assert cache.scores(_KB, "certifications", _TEXTS) == {}


def test_another_kbs_bump_does_not_clear_this_one(cache: V2Cache) -> None:
    cache.put_router(_KB, _QUERY, ("d1",), _decision())

    get_coordinator().bump(_version_key("unrelated-kb"))

    assert cache.router(_KB, _QUERY, ("d1",)) is not None
