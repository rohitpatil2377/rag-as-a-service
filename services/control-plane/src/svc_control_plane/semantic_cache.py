"""Semantic answer cache (FR-E5, P2) — repeat questions answer instantly.

Caches GROUNDED answers per knowledge base, keyed by the query's embedding: a new
question whose embedding is within ``semantic_cache_threshold`` cosine similarity of a
cached one returns the cached cited answer with no retrieval or generation cost.

Safety rules (why this can't serve a wrong or stale answer silently):
* only GROUNDED answers are cached — never abstains or flagged-for-review;
* the whole KB's cache is invalidated on ANY ingest / delete / re-index in that KB;
* the caller's ACL groups are part of the key — an entitled user's answer is never
  replayed to a caller with different document visibility (FR-A6);
* cache hits are audited (QUERY event with ``cache_hit: true``), so metering and the
  quality dashboards stay honest.

The cached answers live in-process (per-replica LRU — the expensive vector scan stays
local and lock-free), but INVALIDATION is coherent across replicas via the coordination
seam: every invalidate bumps a shared per-KB version counter, and each read drops its
local KB cache when it observes a newer version. So when replica A ingests into a KB,
replica B stops serving that KB's now-stale answers on its next lookup. Single-process
(no ``RAG_REDIS_URL``) the counter is process-local — identical to the original
behaviour, zero overhead.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

from rag_common import get_coordinator, get_logger, get_settings
from rag_schemas import GroundedAnswer

_log = get_logger(__name__)


def _version_key(kb_id: str) -> str:
    return f"semcache:{kb_id}"


def still_passes(answer: GroundedAnswer, *, abstain: float, faithful: float, flag: float) -> bool:
    """Would this cached answer still clear TODAY's gate?

    A cache hit returns a stored GroundedAnswer verbatim, which skips the grounding and
    faithfulness gates that produced it. Those thresholds are per-KB and operator-editable, so
    an operator who raises their abstain floor — the one control they have for "stop answering
    marginal questions" — would keep receiving the marginal answers it was meant to stop, for as
    long as the entry lived. Re-checking the stored numbers against the current ones is cheap
    (no model call, no retrieval) and keeps the gate the single decision point.

    Only GROUNDED answers are cached, so anything that would now route to FLAGGED_FOR_REVIEW or
    ABSTAINED must not be served from here."""
    if answer.confidence < abstain:
        return False
    return not (answer.faithfulness_score is not None and answer.faithfulness_score < faithful)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass(slots=True)
class _Entry:
    vector: list[float]
    scope_key: str
    answer: GroundedAnswer


class SemanticCache:
    """Per-KB LRU of (query embedding, acl, grounded answer)."""

    def __init__(self) -> None:
        self._kb: dict[str, OrderedDict[str, _Entry]] = {}
        self._seen_version: dict[str, int] = {}  # last shared version this replica has honoured

    @staticmethod
    def _scope_key(acl: list[str], clearance: int = 0, top_k: int = 0) -> str:
        """Everything besides the embedding that changes which answer is correct.

        Clearance: a Confidential-cleared answer must NEVER be served from cache to a
        lower-clearance caller (banking pack). Same rule as the ACL groups.

        top_k: it decides how much evidence the generator is shown, so it changes the answer —
        an enumerated list answered from 5 passages is genuinely less complete than the same
        question answered from 15. Without it in the key the cache replayed the 5-passage answer
        to the 15-passage request, which also silently defeats any attempt to tune top_k."""
        return f"{clearance}|{top_k}|" + ",".join(sorted(acl))

    def _sync_coherence(self, kb_id: str) -> None:
        """Drop this replica's local cache for the KB if another replica has invalidated it."""
        current = get_coordinator().version(_version_key(kb_id))
        if self._seen_version.get(kb_id, 0) != current:
            self._kb.pop(kb_id, None)
            self._seen_version[kb_id] = current

    def get(
        self, kb_id: str, vector: list[float], acl: list[str], clearance: int = 0, top_k: int = 0
    ) -> GroundedAnswer | None:
        settings = get_settings()
        self._sync_coherence(kb_id)
        entries = self._kb.get(kb_id)
        if not entries:
            return None
        scope_key = self._scope_key(acl, clearance, top_k)
        best: tuple[float, str, _Entry] | None = None
        for key, entry in entries.items():
            if entry.scope_key != scope_key:
                continue
            # Defence-in-depth behind the invalidation on embedder switch: an entry from a
            # different vector space (different dimension) is stale, not comparable —
            # zip(strict=True) in _cosine would raise, 500ing the query.
            if len(entry.vector) != len(vector):
                continue
            sim = _cosine(vector, entry.vector)
            if best is None or sim > best[0]:
                best = (sim, key, entry)
        if best is not None and best[0] >= settings.semantic_cache_threshold:
            entries.move_to_end(best[1])
            _log.info("semantic_cache_hit", kb_id=kb_id, similarity=round(best[0], 4))
            return best[2].answer
        return None

    def put(
        self,
        kb_id: str,
        query_text: str,
        vector: list[float],
        acl: list[str],
        answer: GroundedAnswer,
        clearance: int = 0,
        top_k: int = 0,
    ) -> None:
        if answer.status.value != "grounded":
            return  # never cache abstains or flagged answers
        settings = get_settings()
        entries = self._kb.setdefault(kb_id, OrderedDict())
        entries[query_text] = _Entry(vector=vector, scope_key=self._scope_key(acl, clearance, top_k), answer=answer)
        entries.move_to_end(query_text)
        while len(entries) > settings.semantic_cache_max_entries:
            entries.popitem(last=False)

    def invalidate(self, kb_id: str) -> None:
        """The KB's knowledge changed — every cached answer may now be stale. Bump the shared
        version so OTHER replicas drop their copies on their next read too."""
        version = get_coordinator().bump(_version_key(kb_id))
        self._seen_version[kb_id] = version
        if self._kb.pop(kb_id, None) is not None:
            _log.info("semantic_cache_invalidated", kb_id=kb_id)

    def clear(self) -> None:
        self._kb.clear()


cache = SemanticCache()
