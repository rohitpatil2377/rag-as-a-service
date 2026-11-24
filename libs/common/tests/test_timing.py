"""``StageTimeline``: per-stage request timing (RAG v2 Phase 12 · discovery §8.2).

Timing code is awkward to test without making the test slow or flaky, so these pin the SHAPE and
the ARITHMETIC rather than real durations: that stages accumulate rather than overwrite, that a
skipped interval is charged to nobody, that the reserved total key cannot be taken by a stage,
and that a bad stage name is dropped instead of raising. The last one is the point of the whole
module — an observability helper that can fail the request it is measuring is worse than none.
"""

from __future__ import annotations

import time

from rag_common.timing import TOTAL_KEY, StageTimeline


def test_stages_partition_the_request_in_the_order_they_were_marked() -> None:
    timeline = StageTimeline()
    for stage in ("guard", "cache", "route", "retrieve", "generate"):
        timeline.mark(stage)

    assert list(timeline.stages()) == ["guard", "cache", "route", "retrieve", "generate"]
    assert all(isinstance(ms, int) and ms >= 0 for ms in timeline.stages().values())


def test_marking_a_stage_twice_accumulates_rather_than_overwrites() -> None:
    """A query path has loops in it — an agentic retry re-runs retrieval — and reporting only the
    last pass would understate exactly the case worth seeing."""
    timeline = StageTimeline()
    timeline.mark("retrieve")
    time.sleep(0.02)
    timeline.mark("retrieve")

    assert len(timeline.stages()) == 1
    assert timeline.stages()["retrieve"] >= 15, "the second pass was added to the first, not dropped"


def test_a_stage_measures_the_interval_since_the_previous_mark() -> None:
    timeline = StageTimeline()
    timeline.mark("fast")
    time.sleep(0.03)
    timeline.mark("slow")

    stages = timeline.stages()
    assert stages["slow"] >= 25
    assert stages["slow"] > stages["fast"]


def test_skip_charges_the_elapsed_time_to_nobody() -> None:
    """For a branch that leaves the measured path — charging it to the next stage would be a lie."""
    timeline = StageTimeline()
    time.sleep(0.03)
    timeline.skip()
    timeline.mark("after")

    assert timeline.stages()["after"] < 20
    assert timeline.total_ms >= 25, "...but the request's own total still counts it"


def test_the_payload_is_flat_and_carries_the_total() -> None:
    """Flat so a dashboard can graph one key, and so it reads like the ingest path's `stage_ms`."""
    timeline = StageTimeline()
    timeline.mark("route")
    payload = timeline.payload()

    assert set(payload) == {"route", TOTAL_KEY}
    assert payload[TOTAL_KEY] >= payload["route"]


def test_a_stage_may_not_take_the_reserved_total_key() -> None:
    timeline = StageTimeline()
    timeline.mark(TOTAL_KEY)
    timeline.mark("route")

    assert list(timeline.stages()) == ["route"]


def test_a_bad_stage_name_is_dropped_and_does_not_raise() -> None:
    """It is a programming error in a call site, but it happens inside a live query."""
    timeline = StageTimeline()
    timeline.mark("")
    assert timeline.stages() == {}


def test_a_dropped_mark_still_consumes_its_interval() -> None:
    """Otherwise the NEXT stage is credited with the dropped one's duration too, which would turn
    a naming slip into a wrong latency attribution — the one thing this module exists to get right."""
    timeline = StageTimeline()
    time.sleep(0.03)
    timeline.mark(TOTAL_KEY)
    timeline.mark("route")

    assert timeline.stages()["route"] < 20


def test_an_unmarked_timeline_reports_a_total_and_no_stages() -> None:
    """A request that returned before any mark — a guardrail block — still has a duration, and
    claiming a stage it never ran would be worse than reporting none."""
    timeline = StageTimeline()
    payload = timeline.payload()

    assert list(payload) == [TOTAL_KEY]
    assert timeline.stages() == {}
