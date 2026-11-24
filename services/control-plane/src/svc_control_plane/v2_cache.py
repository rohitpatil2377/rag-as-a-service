"""In-process caches for the two repeated computations on the RAG v2 query path (Phase 12:
"cache router output by normalised query + doc set; cache section reranker scores per (query, doc)").

NEITHER OF THESE IS AN ANSWER CACHE. ``semantic_cache`` already caches answers, and it has to
re-check the stored gate every time it serves one (``still_passes``) because an answer is a
judgement that a threshold could have changed. What is cached here is upstream of any judgement:

* a ROUTER DECISION is a function of the question's words and the cards in scope, and
* a SECTION'S RERANKER SCORE is a function of (probe, candidate text).

Both are deterministic for fixed inputs, so a hit cannot serve something an operator's config
change should have stopped — the way a stale ANSWER could. That is what makes these safe as
plain LRUs with no revalidation.

WHY IT IS WORTH CACHING AT ALL. The section pass is the expensive part of an aggregate question
and almost none of it depends on the question's exact wording: on the real AEGCL corrigendum it
is nine probes over seventy-two candidates, and eight of those nine probes are the document's
facet names — identical for every question asked of that document. Two differently-worded
compliance questions over the same corrigendum therefore share eight of their nine rerank calls.

INVALIDATION IS BY THE SAME SIGNAL ``semantic_cache`` USES — the per-KB coordinator version
counter, bumped on any ingest, delete or reindex. These caches key on text the catalog holds, so
a re-ingest that rewrites a section's summary must drop its scores; piggybacking on the existing
signal means there is one place that says "this KB's derived state moved" rather than two that
can disagree.

IN-PROCESS, like ``semantic_cache``: a second replica has its own copy, which costs a cold start
per replica and nothing else, because every entry is reproducible from the catalog.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from rag_common import get_coordinator, get_logger, get_settings
from rag_core.router import RouterDecision

_log = get_logger(__name__)

__all__ = ["V2Cache", "v2_cache"]


def _version_key(kb_id: str) -> str:
    """The SAME namespace ``semantic_cache`` uses, deliberately.

    One counter per KB means one invalidation event clears every derived cache it feeds. A
    separate namespace would let the answer cache and these two drift apart after a reindex —
    the answer gone and the scores that produced it still live."""
    return f"semcache:{kb_id}"


def _digest(*parts: str) -> str:
    """A bounded key from unbounded text. The parts are joined with NUL, which cannot appear in
    any of them, so two different splits cannot collide into one key."""
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


class V2Cache:
    """Per-KB LRUs for router decisions and section scores, cleared together on KB change."""

    def __init__(self) -> None:
        # kb -> key -> value. Two maps rather than one with prefixed keys: they have different
        # sizes and different hit profiles, and sharing an LRU would let a hundred section
        # scores evict the one router decision that every question needs.
        self._router: dict[str, OrderedDict[str, RouterDecision]] = {}
        self._scores: dict[str, OrderedDict[str, float]] = {}
        self._versions: dict[str, int] = {}

    def _fresh(self, kb_id: str) -> None:
        """Drop this KB's entries if its version moved. Called before every read and write."""
        try:
            version = get_coordinator().version(_version_key(kb_id))
        except Exception as exc:  # a coordinator fault must not break the query path
            _log.warning("v2_cache_version_unavailable", kb_id=kb_id, error=str(exc)[:120])
            return
        if self._versions.get(kb_id) != version:
            self._versions[kb_id] = version
            self._router.pop(kb_id, None)
            self._scores.pop(kb_id, None)

    def _trim(self, entries: OrderedDict[str, object], cap: int) -> None:
        while len(entries) > cap:
            entries.popitem(last=False)

    # --- router decisions ---------------------------------------------------------------------

    def router(self, kb_id: str, query: str, doc_ids: tuple[str, ...]) -> RouterDecision | None:
        """The stored decision for this (normalised query, doc set), or None.

        The DECISION OBJECT is cached, not its JSON. Round-tripping through
        ``parse_router_decision`` would be the natural-looking choice and is wrong: that parser
        VALIDATES against a scope, so re-parsing a stored decision would need the scope rebuilt
        — and handed an empty one it silently drops every ``doc_id`` and facet, turning a cache
        hit into a decision that scopes retrieval to nothing. The cache is in-process and the
        decision is a frozen dataclass, so there is nothing to serialise for.

        The DOC SET is part of the key because it is part of the input: the same question over
        a different scope is a different decision, and a cache keyed on the query alone would
        scope one KB's answer to another document set."""
        settings = get_settings()
        if not settings.v2_router_cache_enabled:
            return None
        self._fresh(kb_id)
        entries = self._router.get(kb_id)
        if entries is None:
            return None
        key = _digest(query.strip().casefold(), *sorted(doc_ids))
        hit = entries.get(key)
        if hit is not None:
            entries.move_to_end(key)
        return hit

    def put_router(self, kb_id: str, query: str, doc_ids: tuple[str, ...], decision: RouterDecision) -> None:
        settings = get_settings()
        if not settings.v2_router_cache_enabled:
            return
        self._fresh(kb_id)
        entries = self._router.setdefault(kb_id, OrderedDict())
        key = _digest(query.strip().casefold(), *sorted(doc_ids))
        entries[key] = decision
        entries.move_to_end(key)
        self._trim(entries, settings.v2_router_cache_size)  # type: ignore[arg-type]

    # --- section reranker scores --------------------------------------------------------------

    def scores(self, kb_id: str, probe: str, candidates: dict[str, str]) -> dict[str, float]:
        """The cached score for each candidate this probe has already been scored against.

        Keyed on (probe, candidate TEXT) rather than (probe, section_id): the score is a
        function of what the cross-encoder read, so a section whose summary was re-derived must
        miss even though its id did not change. The version counter covers the ordinary reindex;
        this covers the case where it does not fire."""
        if not get_settings().v2_score_cache_enabled:
            return {}
        self._fresh(kb_id)
        entries = self._scores.get(kb_id)
        if entries is None:
            return {}
        out: dict[str, float] = {}
        for section_id, text in candidates.items():
            key = _digest(probe, text)
            hit = entries.get(key)
            if hit is not None:
                entries.move_to_end(key)
                out[section_id] = hit
        return out

    def put_scores(self, kb_id: str, probe: str, texts: dict[str, str], scored: dict[str, float]) -> None:
        settings = get_settings()
        if not settings.v2_score_cache_enabled:
            return
        self._fresh(kb_id)
        entries = self._scores.setdefault(kb_id, OrderedDict())
        for section_id, score in scored.items():
            text = texts.get(section_id)
            if text is not None:
                entries[_digest(probe, text)] = score
        self._trim(entries, settings.v2_score_cache_size)  # type: ignore[arg-type]


#: One per process, like ``semantic_cache``'s.
v2_cache = V2Cache()
