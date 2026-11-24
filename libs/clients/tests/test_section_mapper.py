"""``LLMSectionMapper``: the per-section map call, its one retry, and what a failure costs.

What an item CONTAINS is ``libs/rag-core/tests/test_mapreduce.py``. What is tested here is the
adapter's contract, and the posture that distinguishes it from the Phase-4 enricher it is
otherwise shaped like: this runs on the LATENCY PATH, once per selected section per question, so
a long section is truncated rather than windowed, and a section that fails costs that section's
items rather than the answer.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from rag_clients.section_mapper import LLMSectionMapper, MapJob
from rag_core import LLMGateway
from rag_core.section_enrich import section_windows

_TEXT = "The switches shall comply with UL 62368, CSA 22.2, CE marking and FCC Part 15 Class A."
_FACETS = ("certifications", "supply_chain")


def _job(section_id: str = "s1", text: str = _TEXT) -> MapJob:
    return MapJob(section_id=section_id, doc_id="d1", breadcrumb="21. Switches", text=text)


def _blocks(n: int) -> str:
    """A section long enough to window, where EVERY block carries the quotable phrase — so an
    item dropped by the tests below was dropped by the thing under test, not by the fixture."""
    return "\n\n".join(f"Block {i}. UL 62368 applies. " + "x" * 280 for i in range(n))


def _items(*quotes: str) -> str:
    return json.dumps([{"facet_id": "certifications", "statement": f"requires {q}", "quote": q} for q in quotes])


class _Gateway:
    """A scripted gateway: one reply per call, recording the prompts it was given."""

    def __init__(self, *replies: str | Exception) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        reply = self._replies.pop(0) if self._replies else "[]"
        if isinstance(reply, Exception):
            raise reply
        return reply

    def stream(self, prompt: str, **_: Any) -> Any:  # pragma: no cover — unused by this adapter
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "test-model"

    async def list_models(self) -> list[str]:
        return ["test-model"]


async def _map(gw: LLMGateway, jobs: list[MapJob] | None = None, **kw: Any) -> Any:
    mapper = LLMSectionMapper(gw, **kw)
    return await mapper.map_sections(
        [_job()] if jobs is None else jobs,
        query="what are the switch certifications?",
        profile_json="",
        facet_ids=_FACETS,
    )


# --- the call ---------------------------------------------------------------------------------------


async def test_a_usable_reply_becomes_items() -> None:
    items, report = await _map(_Gateway(_items("UL 62368", "FCC Part 15 Class A")))

    assert [i.item_id for i in items] == ["m1", "m2"]
    assert report.sections_mapped == 1 and report.sections_contributed == 1 and report.items == 2


async def test_the_prompt_carries_the_section_and_the_question() -> None:
    gw = _Gateway(_items("UL 62368"))
    await _map(gw)

    assert "21. Switches" in gw.prompts[0]
    assert "what are the switch certifications?" in gw.prompts[0]
    assert "UL 62368" in gw.prompts[0], "the section text itself"


async def test_an_empty_extraction_is_mapped_but_did_not_contribute() -> None:
    """The numerator of the coverage statement. A section that was read and had nothing relevant
    in it did not contribute, and an answer that counted it would overstate how much of the
    document it used."""
    _items_out, report = await _map(_Gateway("[]"))

    assert report.sections_mapped == 1
    assert report.sections_contributed == 0
    assert report.sections_failed == 0


# --- the retry --------------------------------------------------------------------------------------


async def test_an_unusable_reply_is_retried_once_with_the_reason() -> None:
    gw = _Gateway("not json", _items("UL 62368"))
    items, report = await _map(gw)

    assert len(gw.prompts) == 2
    assert "could not be used" in gw.prompts[1] and "no JSON array" in gw.prompts[1]
    assert len(items) == 1 and report.retries == 1


async def test_a_second_unusable_reply_fails_that_section_only() -> None:
    gw = _Gateway("nope", "still nope")
    items, report = await _map(gw, [_job("s1"), _job("s2")])

    assert report.sections_failed >= 1
    assert isinstance(items, list), "the pass still returns, with whatever survived"


async def test_a_gateway_error_is_retried_like_any_other_unusable_reply() -> None:
    """Unlike the Phase-7 router, which does not retry a raising gateway. There, one decision
    gates the whole answer and a second call is latency for an answer already in hand; here the
    section's items are the answer's content, and the retry is scoped to one section."""
    gw = _Gateway(RuntimeError("model unavailable"), _items("UL 62368"))
    items, report = await _map(gw)

    assert len(items) == 1 and report.retries == 1


async def test_one_failing_section_does_not_stop_the_others() -> None:
    gw = _Gateway("bad", "bad again", _items("UL 62368"))
    items, report = await _map(gw, [_job("s1"), _job("s2")])

    assert report.sections_failed == 1 and report.sections_mapped == 1
    assert len(items) == 1, "the healthy section still contributed"


# --- the latency-path posture ---------------------------------------------------------------------


async def test_a_long_section_is_windowed_not_truncated() -> None:
    """MEASURED on the real AEGCL corrigendum: truncating at 12,000 characters was the single
    largest cause of a missed requirement. A tender section puts its specification table first
    and its certification and bill-of-material clauses LAST, so the end of a long section is
    the part a compliance question most needs — and it was the part being cut."""
    long_job = _job(text=_blocks(10))
    gw = _Gateway(*[_items("UL 62368 applies")] * 3)

    items, report = await _map(gw, [long_job], section_chars=1200, max_windows=3)

    assert len(gw.prompts) == 3, "three windows, three calls"
    assert report.windows == 3
    assert len(items) == 3, "every window's items are merged"
    assert [i.item_id for i in items] == ["m1", "m2", "m3"]


async def test_a_section_past_the_window_bound_is_counted_as_truncated() -> None:
    """The bound still exists — this runs on the latency path — but what it costs is now
    VISIBLE. ``sections_truncated`` is the number to watch before widening it."""
    long_job = _job(text=_blocks(20))
    gw = _Gateway(*[_items("UL 62368 applies")] * 2)

    _items_out, report = await _map(gw, [long_job], section_chars=1200, max_windows=2)

    assert len(gw.prompts) == 2 and report.windows == 2
    assert report.sections_truncated == 1


async def test_a_multi_window_section_that_fits_is_not_reported_truncated() -> None:
    """The separator the splitter drops is not lost content. Comparing character SUMS reported
    every multi-window section as truncated — on the real corrigendum, 12 instead of 3 — which
    is precisely the number an operator is told to watch before widening the bound."""
    job = _job(text=_blocks(6))
    needed = len(section_windows(job.text, 1200, 99))
    gw = _Gateway(*[_items("UL 62368 applies")] * needed)

    _items_out, report = await _map(gw, [job], section_chars=1200, max_windows=needed)

    assert report.windows == needed
    assert report.sections_truncated == 0, "fully covered, just not in one window"


async def test_a_section_inside_the_budget_is_one_window_and_not_truncated() -> None:
    gw = _Gateway(_items("UL 62368"))
    _items_out, report = await _map(gw, [_job()], section_chars=12_000)

    assert len(gw.prompts) == 1 and report.windows == 1 and report.sections_truncated == 0


async def test_one_fumbled_window_does_not_fail_the_section() -> None:
    """A long section whose second window the model fumbled still contributed the first
    window's items, and counting the whole section failed would make the coverage denominator
    punish it for that."""
    long_job = _job(text=_blocks(6))
    gw = _Gateway(_items("UL 62368 applies"), "bad", "bad again")

    items, report = await _map(gw, [long_job], section_chars=1200, max_windows=2)

    assert report.sections_mapped == 1 and report.sections_failed == 0
    assert report.sections_contributed == 1 and len(items) == 1


async def test_a_quote_from_outside_the_window_is_dropped() -> None:
    """Each window's items are checked against THAT WINDOW's text, not the whole section —
    otherwise a model that remembered the previous window could quote text this call never
    saw, and the quote rule would be checking the wrong haystack."""
    job = _job(text="FIRST PART. " + "x" * 500 + " UL 62368 appears only here.")
    gw = _Gateway(*[_items("UL 62368 appears only here")] * 6)

    items, _report = await _map(gw, [job], section_chars=40, max_windows=1)

    assert items == []


async def test_no_jobs_makes_no_calls() -> None:
    gw = _Gateway()
    items, report = await _map(gw, [])
    assert items == [] and report.sections_mapped == 0 and gw.prompts == []


async def test_ids_are_dense_and_ascending_across_sections() -> None:
    """The reduce prompt cites one flat list, so per-section numbering would collide."""
    gw = _Gateway(_items("UL 62368", "CSA 22.2"), _items("CE marking"))
    items, _report = await _map(gw, [_job("s1"), _job("s2")])

    assert [i.item_id for i in items] == ["m1", "m2", "m3"]
    assert {i.section_id for i in items} == {"s1", "s2"}


# --- the concurrency bound --------------------------------------------------------------------------
#
# The map stage's latency was measured, over six runs, to equal the LONGEST SECTION'S window chain
# to within 0.1 s, while only 6.9-8.4 of 12 permitted calls were ever in flight: the bound counted
# SECTIONS, so one long section's windows queued behind each other with the endpoint idle. The
# permit now covers a CALL. These tests pin both halves of that -- the bound is still respected,
# and a single section's windows no longer serialise -- because either one alone can be satisfied
# by a wrong implementation.


class _TrackingGateway:
    """Records how many calls are in flight at once, and suspends so they genuinely overlap."""

    def __init__(self, reply: str = "[]") -> None:
        self._reply = reply
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    async def complete(self, prompt: str, **_: Any) -> str:
        self.in_flight += 1
        self.calls += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            # Two hops, so a caller that merely *starts* its coroutines without awaiting past
            # the first suspension point cannot pass for concurrency.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return self._reply
        finally:
            self.in_flight -= 1

    def stream(self, prompt: str, **_: Any) -> Any:  # pragma: no cover — unused by this adapter
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "test-model"

    async def list_models(self) -> list[str]:
        return ["test-model"]


async def test_the_bound_counts_calls_in_flight_across_the_whole_stage() -> None:
    """Twelve permits, eight sections of four windows each: never a thirteenth call in flight."""
    gw = _TrackingGateway(_items("UL 62368"))
    jobs = [_job(section_id=f"s{i}", text=_blocks(4)) for i in range(8)]
    _items_out, report = await _map(gw, jobs, concurrency=12, section_chars=400, max_windows=4)

    assert report.windows == 32, "the fixture must actually produce the windows being bounded"
    assert gw.calls == 32
    assert gw.peak <= 12, f"{gw.peak} calls were in flight against a bound of 12"


async def test_one_sections_windows_do_not_wait_for_each_other() -> None:
    """The regression that mattered: a SINGLE section's windows used to run strictly in sequence,
    so one long section set the whole stage's wall time while the endpoint sat idle."""
    gw = _TrackingGateway(_items("UL 62368"))
    _items_out, report = await _map(gw, [_job(text=_blocks(6))], concurrency=12, section_chars=400, max_windows=6)

    assert report.windows == 6
    assert gw.peak > 1, "one section's windows are still serialised"
    assert gw.peak == 6, f"all six windows fit under the bound of 12 but only {gw.peak} overlapped"


async def test_a_sections_windows_still_queue_when_they_exceed_the_bound() -> None:
    """A section may offer more windows than the whole bound. They queue; they do not dispatch."""
    gw = _TrackingGateway(_items("UL 62368"))
    _items_out, report = await _map(gw, [_job(text=_blocks(6))], concurrency=2, section_chars=400, max_windows=6)

    assert report.windows == 6
    assert gw.peak == 2, f"{gw.peak} calls were in flight against a bound of 2"


async def test_items_follow_window_order_not_completion_order() -> None:
    """Ids are assigned from the parses, so a result ordered by whichever call returned first
    would make item ids -- and the map cache keyed on them -- depend on scheduling."""

    class _ReversingGateway(_TrackingGateway):
        """Answers the LAST window first: the first window waits longest."""

        def __init__(self) -> None:
            super().__init__()
            self._seen = 0

        async def complete(self, prompt: str, **_: Any) -> str:
            mine = self._seen
            self._seen += 1
            for _hop in range(8 - mine):
                await asyncio.sleep(0)
            return _items(f"Block {mine}. UL 62368 applies.")

    gw = _ReversingGateway()
    items, report = await _map(gw, [_job(text=_blocks(4))], concurrency=12, section_chars=400, max_windows=4)

    assert report.windows == 4
    assert [i.item_id for i in items] == ["m1", "m2", "m3", "m4"]
    # Window 0's quote must be item 1 even though its call returned last.
    assert items[0].quote.startswith("Block 0."), [i.quote[:9] for i in items]


# --- what a failure WAS, not just that there was one -------------------------------------------


class _ClassifiedError(Exception):
    """A gateway error carrying the classification `OpenAICompatGateway` attaches: `details`
    with a `failure_kind`. The mapper must pass that through rather than flatten every failure
    into one count."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"gateway said {kind}")
        self.details = {"failure_kind": kind, "timeout_s": 240.0, "elapsed_s": 240.1}


async def test_a_read_timeout_is_counted_as_a_read_timeout() -> None:
    """The distinction that cost a day: 28 of 63 calls died at exactly the configured limit
    during the scheduler benchmark, and `sections_failed=11` alone could not say whether the
    endpoint was slow or the replies were malformed."""
    gw = _Gateway(_ClassifiedError("read_timeout"), _ClassifiedError("read_timeout"))
    _items_out, report = await _map(gw)

    assert report.sections_failed == 1
    assert report.failures == (("read_timeout", 1),)


async def test_an_unreachable_endpoint_is_not_a_timeout() -> None:
    gw = _Gateway(_ClassifiedError("unreachable"), _ClassifiedError("unreachable"))
    _items_out, report = await _map(gw)

    assert report.failures == (("unreachable", 1),)


async def test_a_malformed_reply_is_its_own_kind() -> None:
    """A reply that arrived and would not parse is the one failure that is OUR fault, and it
    must not share a bucket with the endpoint's."""
    gw = _Gateway("not json", "still not json")
    _items_out, report = await _map(gw)

    assert report.failures == (("unusable_reply", 1),)


async def test_an_unclassified_exception_keeps_its_type_name() -> None:
    gw = _Gateway(RuntimeError("boom"), RuntimeError("boom"))
    _items_out, report = await _map(gw)

    assert report.failures == (("RuntimeError", 1),)


async def test_kinds_are_counted_separately_and_ordered_by_count() -> None:
    gw = _Gateway(
        _ClassifiedError("read_timeout"),
        _ClassifiedError("read_timeout"),
        "bad",
        "bad again",
        _ClassifiedError("read_timeout"),
        _ClassifiedError("read_timeout"),
    )
    _items_out, report = await _map(gw, [_job("s1"), _job("s2"), _job("s3")])

    assert report.sections_failed == 3
    assert report.failures == (("read_timeout", 2), ("unusable_reply", 1))


async def test_a_window_that_recovers_on_retry_is_not_a_failure() -> None:
    """The kind recorded is the LAST attempt's, because that is the one that decided the
    outcome. A first call that timed out and a retry that parsed is a success that cost an extra
    call, and counting it as a read timeout would make the metric fire on healthy runs."""
    gw = _Gateway(_ClassifiedError("read_timeout"), _items("UL 62368"))
    items, report = await _map(gw)

    assert len(items) == 1 and report.retries == 1
    assert report.sections_failed == 0 and report.failures == ()


async def test_a_clean_pass_reports_no_failures() -> None:
    _items_out, report = await _map(_Gateway(_items("UL 62368")))

    assert report.failures == ()


class _MeteredGateway(_Gateway):
    """A gateway with the usage-reporting path, so `finish_reason` and token counts exist.

    Scripted per call as ``(reply, finish_reason, completion_tokens)``. `_Gateway` deliberately
    has NO ``complete_with_usage`` — the tests above are the fallback path, and these are the
    reporting one. Both must work: a stub gateway with only ``complete()`` is a supported
    gateway, it just cannot say what it spent."""

    def __init__(self, *calls: tuple[str, str, int]) -> None:
        super().__init__(*(c[0] for c in calls))
        self._meta = [(c[1], c[2]) for c in calls]

    async def complete_with_usage(self, prompt: str, **kw: Any) -> tuple[str, dict[str, Any]]:
        reply = await self.complete(prompt, **kw)
        finish, tokens = self._meta.pop(0) if self._meta else ("stop", 0)
        return reply, {"finish_reason": finish, "completion_tokens": tokens}


async def test_a_window_cut_off_by_the_output_cap_is_counted() -> None:
    """The point of the field: this reply PARSED and the section CONTRIBUTED, so every other
    measure on the report reads clean. Only `calls_at_cap` says the model ran out of room to
    finish reporting what it had already read."""
    gw = _MeteredGateway((_items("UL 62368"), "length", 4000))
    items, report = await _map(gw)

    assert len(items) == 1
    assert report.calls_at_cap == 1 and report.calls == 1
    assert report.completion_tokens == 4000
    # Not a failure, not coverage truncation, and not a dropped item — the four are independent.
    assert report.sections_failed == 0 and report.sections_truncated == 0
    assert report.sections_contributed == 1 and report.dropped_over_cap == 0


async def test_output_truncation_is_not_coverage_truncation() -> None:
    """A section SHORT enough to need one window can still be cut off by the output cap, and a
    section too long for `max_windows` can finish every reply. Opposite remedies, so a report
    that merged them would send the operator to the wrong knob."""
    gw = _MeteredGateway(*[(_items("UL 62368"), "stop", 10)] * 2)
    _out, report = await _map(gw, [_job("s1", _blocks(6))], section_chars=600, max_windows=2)

    assert report.sections_truncated == 1, "text past the windows bound was never shown"
    assert report.calls_at_cap == 0, "but every reply the model did give finished"


async def test_a_retry_costs_a_second_call_and_its_tokens() -> None:
    """`calls` is the denominator a failure or at-cap RATE divides by, and it is not `windows`:
    the retried window below spent two calls, and dividing by one window would report a 100%
    at-cap rate for a pass where half the calls finished."""
    gw = _MeteredGateway(("not json", "length", 4000), (_items("UL 62368"), "stop", 120))
    items, report = await _map(gw)

    assert len(items) == 1 and report.retries == 1
    assert report.windows == 1 and report.calls == 2
    assert report.calls_at_cap == 1
    assert report.completion_tokens == 4120, "both attempts were really paid for"


async def test_a_gateway_that_cannot_report_usage_still_maps() -> None:
    """The capability check, not an interface requirement. `LLMGateway` does not declare
    `complete_with_usage`, so a conforming gateway may not have it — it maps as before and
    reports zero spend, which is 'unmeasurable' rather than 'free'."""
    items, report = await _map(_Gateway(_items("UL 62368")))

    assert len(items) == 1 and report.calls == 1
    assert report.calls_at_cap == 0 and report.completion_tokens == 0


async def test_the_map_call_stays_unclassified_so_its_measured_sampling_is_untouched() -> None:
    """A DELIBERATE NON-CHANGE, pinned because it would be easy to "finish the job" and break it.

    Phase 4 and the Phase-3 profile were classified as structured extraction so their
    `temperature=0.0` reaches the endpoint (`rag_core.sampling`). The map call was NOT, and
    that is not an oversight: its behaviour at 0.6 is a measured baseline -- the window-size
    experiment verified every one of 1,251 map and 411 reduce requests on the wire at
    `temperature=0.6` (see `docs/rag_v2/goal_scorecard.md`). Reclassifying it would invalidate
    that baseline and change answers, so it is its own decision."""

    class _PolicyGateway(_Gateway):
        honours_sampling_task = True

        def __init__(self, *replies: str | Exception) -> None:
            super().__init__(*replies)
            self.kwargs: list[dict[str, Any]] = []

        async def complete_with_usage(self, prompt: str, **kw: Any) -> tuple[str, dict[str, Any]]:
            self.kwargs.append(dict(kw))
            return await self.complete(prompt), {"completion_tokens": 100, "finish_reason": "stop"}

    gw = _PolicyGateway(_items("UL 62368"))
    await _map(gw)

    assert gw.kwargs, "the mapper went through the usage-reporting path"
    assert all("task" not in kw for kw in gw.kwargs), "the map call is unclassified, by design"
