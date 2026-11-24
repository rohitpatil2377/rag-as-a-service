"""Per-stage timing for a single request (RAG v2 Phase 12 · discovery §8.2).

The upgrade plan's §0 success criteria are latency targets — "<= 8 s lookup / <= 60 s aggregate
p95" — and discovery found they are not measurable: the query path records ONE number, the
round trip, so a slow answer cannot be attributed to routing, retrieval or generation. That is
why §8.2 asks for this slice before the phases it is supposed to measure, and it is also why the
Phase-0 baseline was never run.

DURATIONS, NOT OFFSETS, which is where this differs from the ingest path's ``stage_ms``
(``svc_control_plane.ingest_progress``). The ingestion DAG draws a timeline, so offsets from the
start are what it needs. A query is tuned, not drawn: the question is "which stage is the 5
seconds in", and that is a duration. The two shapes are deliberately not unified — one is a
picture of a pipeline, the other is a bill.

FAIL-SOFT, ALWAYS. Nothing here raises, and nothing here is allowed to be on the critical path
of a correct answer: a mis-named stage is logged and dropped rather than aborting the request it
was measuring. An observability helper that can fail a query is worse than no observability.

ONE OBJECT PER REQUEST, passed down the call, rather than a process-global keyed by trace id.
``ingest_progress`` is global because the ingest DAG polls it from another request while the
ingest is still running; a query has no such reader — it finishes inside one call stack — so a
global would add a TTL, a GC sweep and a cross-request leak for nothing.
"""

from __future__ import annotations

import time
from typing import Any

from rag_common.logging import get_logger

__all__ = ["TOTAL_KEY", "StageTimeline"]

_log = get_logger(__name__)

# Reserved in :meth:`StageTimeline.payload`, which adds the request's whole duration under it. A
# stage may not use the name — see :meth:`StageTimeline.mark`.
TOTAL_KEY = "total"


class StageTimeline:
    """Accumulating per-stage durations for one request, in whole milliseconds.

    Each :meth:`mark` closes the interval since the previous mark (or since construction), so the
    stages partition the request's wall clock and sum to roughly :attr:`total_ms` — "roughly"
    because the tail after the last mark is not attributed to any stage, which is honest: it is
    whatever the caller did after it stopped measuring.

    Marking the same stage twice ACCUMULATES rather than overwriting. A query path has loops in it
    (an agentic retry re-runs retrieval; a map step runs per section), and reporting only the last
    pass would understate exactly the case worth seeing.
    """

    __slots__ = ("_last", "_stages", "_started")

    def __init__(self) -> None:
        now = time.perf_counter()
        self._started = now
        self._last = now
        self._stages: dict[str, int] = {}

    def mark(self, stage: str) -> None:
        """Attribute the time since the previous mark to ``stage``."""
        now = time.perf_counter()
        elapsed = max(0, round((now - self._last) * 1000))
        self._last = now
        if not stage or stage == TOTAL_KEY:
            # A programming error in a call site, not a runtime condition — but this runs inside a
            # live query, so it is reported and dropped rather than raised. The elapsed time is
            # still consumed, so the NEXT stage is not credited with this one's duration too.
            _log.warning("stage_timeline_bad_stage_name", stage=stage or "(empty)")
            return
        self._stages[stage] = self._stages.get(stage, 0) + elapsed

    def skip(self) -> None:
        """Discard the time since the previous mark without attributing it to any stage.

        For a branch that leaves the measured path — an early return into another function with
        its own timeline — where charging the elapsed time to the next stage would be a lie."""
        self._last = time.perf_counter()

    @property
    def total_ms(self) -> int:
        """The whole request so far, independent of how much of it has been attributed."""
        return max(0, round((time.perf_counter() - self._started) * 1000))

    def stages(self) -> dict[str, int]:
        """The per-stage durations, in the order they were first marked."""
        return dict(self._stages)

    def payload(self) -> dict[str, Any]:
        """The shape written to the audit payload and to ``query_log.latency_ms``.

        Flat — ``{"route": 180, "retrieve": 420, "generate": 5200, "total": 5815}`` — so a
        dashboard can graph one key without digging through a nested object, and so it reads the
        same way as the ingest path's ``stage_ms`` beside it. :data:`TOTAL_KEY` is reserved, which
        is the whole reason :meth:`mark` refuses it."""
        return {**self._stages, TOTAL_KEY: self.total_ms}
