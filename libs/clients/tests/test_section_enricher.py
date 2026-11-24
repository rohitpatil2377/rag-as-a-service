"""``LLMSectionEnricher``: the gateway calls, the retry, the windowing, the bound, the fail-safe.

What a section's enrichment CONTAINS is not tested here and cannot be — see the header of
``libs/rag-core/tests/test_section_enrich.py``. What is tested is the adapter's contract, and the
part of it that is new relative to Phase 3's one-call-per-document profiler: a failure is scoped
to ONE SECTION, the calls are bounded rather than fired all at once, and a long section is read
in windows whose quotes are validated against the window the model actually saw.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from rag_clients.section_enricher import QUOTE_RATE_FLOOR, LLMSectionEnricher, SectionJob

_TEXT = "The bidder shall furnish an EMD of Rs. 5 lakh. Certificates shall accompany the bid."
_FACETS = ("eligibility",)


def _reply(summary: str = "What this clause requires.", **overrides: object) -> str:
    body: dict[str, object] = {
        "summary": summary,
        "facet_ids": ["eligibility"],
        "key_values": [{"name": "emd", "value": "5", "unit": "lakh", "quote": "an EMD of Rs. 5 lakh"}],
        "amends": None,
        "propositions": [],
    }
    body.update(overrides)
    return json.dumps(body)


class _Gateway:
    """A scripted gateway: one reply per call, recording prompts and peak concurrency."""

    def __init__(self, *replies: str | Exception, delay: float = 0.0) -> None:
        self._replies = list(replies)
        self._delay = delay
        self.prompts: list[str] = []
        self.live = 0
        self.peak = 0

    async def complete(self, prompt: str, **_: Any) -> str:
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            self.prompts.append(prompt)
            reply = self._replies.pop(0) if self._replies else _reply()
            if isinstance(reply, Exception):
                raise reply
            return reply
        finally:
            self.live -= 1

    def stream(self, prompt: str, **_: Any) -> Any:  # pragma: no cover — unused by this adapter
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "test-model"

    async def list_models(self) -> list[str]:
        return ["test-model"]


def _jobs(n: int = 1, *, text: str = _TEXT) -> list[SectionJob]:
    return [SectionJob(section_id=f"sec-{i}", breadcrumb=f"4.{i} Clause", text=text) for i in range(n)]


async def _enrich(gw: _Gateway, jobs: list[SectionJob], **kwargs: Any) -> Any:
    enricher = LLMSectionEnricher(gw, **kwargs)
    return await enricher.enrich(jobs, profile_json="{}", facet_ids=_FACETS, doc_id="d1")


async def test_one_call_per_section_and_the_results_are_keyed_by_section_id() -> None:
    gw = _Gateway(_reply(), _reply(), _reply())
    enrichment, report = await _enrich(gw, _jobs(3))

    assert len(gw.prompts) == 3
    assert set(enrichment) == {"sec-0", "sec-1", "sec-2"}
    assert report.sections_enriched == 3 and report.sections_failed == 0


async def test_a_failing_section_costs_that_section_and_nothing_else() -> None:
    """The whole posture. One slow or broken section must not cost a document its ingest — at one
    call per section the chance of at least one hiccup per document is far higher than Phase 3's."""
    gw = _Gateway(RuntimeError("gateway down"), _reply())
    enrichment, report = await _enrich(gw, _jobs(2))

    assert set(enrichment) == {"sec-1"}
    assert report.sections_enriched == 1 and report.sections_failed == 1


async def test_an_unusable_reply_is_retried_once_with_the_reason() -> None:
    gw = _Gateway("not json at all", _reply())
    enrichment, _ = await _enrich(gw, _jobs(1))

    assert len(gw.prompts) == 2
    assert "could not be used" in gw.prompts[1] and "no JSON object" in gw.prompts[1]
    assert set(enrichment) == {"sec-0"}


async def test_a_gateway_error_is_not_retried() -> None:
    """There is nothing to retry a parse on, and a second call to a gateway that just raised is
    one more call for an answer already in hand."""
    gw = _Gateway(RuntimeError("down"))
    enrichment, report = await _enrich(gw, _jobs(1))
    assert len(gw.prompts) == 1, "the one attempt, and no retry after it"
    assert enrichment == {} and report.sections_failed == 1


async def test_a_second_unusable_reply_ends_it() -> None:
    gw = _Gateway("nope", "still nope")
    enrichment, report = await _enrich(gw, _jobs(1))
    assert len(gw.prompts) == 2 and enrichment == {} and report.sections_failed == 1


async def test_concurrency_is_bounded() -> None:
    """ "Parallel, batched" — unbounded parallelism against a tenant's own endpoint is a
    self-inflicted denial of service, and on a CPU tier it is a queue with extra steps."""
    gw = _Gateway(*[_reply()] * 8, delay=0.01)
    await _enrich(gw, _jobs(8), concurrency=2)
    assert gw.peak <= 2


async def test_no_jobs_makes_no_calls() -> None:
    gw = _Gateway()
    enrichment, report = await _enrich(gw, [])
    assert enrichment == {} and report.sections_enriched == 0 and not gw.prompts


# --- windowing ------------------------------------------------------------------------------------

_LONG_A = "The bidder shall furnish an EMD of Rs. 5 lakh."
_LONG_B = "Certificates shall accompany the bid."


async def test_a_long_section_is_read_in_windows_and_merged_into_one_enrichment() -> None:
    long_text = "\n\n".join([_LONG_A, "x" * 200, _LONG_B])
    gw = _Gateway(
        _reply("First window.", key_values=[{"name": "emd", "value": "5", "unit": None, "quote": _LONG_A}]),
        _reply("Second window."),
        _reply("Third window.", propositions=[{"text": "Certificates are required.", "quote": _LONG_B}]),
    )
    enrichment, report = await _enrich(gw, _jobs(1, text=long_text), window_chars=120, max_windows=4)

    assert report.windows == 3 and len(gw.prompts) == 3
    merged = enrichment["sec-0"]
    assert merged.summary == "First window. Second window. Third window."
    assert [kv.name for kv in merged.key_values] == ["emd"]
    assert [p.text for p in merged.propositions] == ["Certificates are required."]


async def test_a_quote_is_validated_against_its_own_window_not_the_whole_section() -> None:
    """A window's reply citing text that window never saw is a fabrication even when a later
    window happens to contain it — which is exactly the case this check exists to catch."""
    long_text = "\n\n".join([_LONG_A, "x" * 200, _LONG_B])
    gw = _Gateway(
        # The FIRST window quotes text that only appears in the THIRD.
        _reply("First.", key_values=[{"name": "certs", "value": "yes", "unit": None, "quote": _LONG_B}]),
        _reply("Second.", key_values=[]),
        _reply("Third.", key_values=[]),
    )
    enrichment, report = await _enrich(gw, _jobs(1, text=long_text), window_chars=120, max_windows=4)

    assert enrichment["sec-0"].key_values == ()
    assert report.quotes_seen == 1 and report.quotes_kept == 0


async def test_a_partially_parsed_section_is_refused_not_merged() -> None:
    """THE INVARIANT THIS FILE PREVIOUSLY ASSERTED THE OPPOSITE OF.

    The old test was `test_one_failed_window_does_not_lose_the_others`, and it asserted exactly
    the behaviour below with `sections_enriched == 1` and the merged summary of the two windows
    that answered. It passed, and it documented a defect: the section is stored and counted as a
    success with one window's facets, key values and summary missing, and NOTHING -- not the
    report, not the manifest, not the answer -- says so. Phase 8 then selects on that metadata.

    A clean failure is disclosed. This was not, which made it worse than the failure it hid.
    """
    long_text = "\n\n".join([_LONG_A, "x" * 200, _LONG_B])
    gw = _Gateway(RuntimeError("down"), _reply("Second window."), _reply("Third window."))
    enrichment, report = await _enrich(gw, _jobs(1, text=long_text), window_chars=120, max_windows=4)

    assert "sec-0" not in enrichment
    assert report.sections_enriched == 0 and report.sections_failed == 1
    assert report.sections_partial == 1 and report.partial_section_ids == ("sec-0",)
    # The window never reached a reply, so the class is the gateway's and not the model's.
    assert report.failure_classes == {"GATEWAY_FAILURE": 1}


async def test_a_truncated_window_refuses_its_section_and_keeps_the_parser_code() -> None:
    """The measured shape: the LAST window of a table-dense section runs out of output budget.

    This is what the rejected windowing arm did in all three of its runs (2026-11-02), and what
    made it rejectable: the merged result looked enriched while its facets collapsed from 5-6 to
    2. The class must stay TRUNCATED -- the fix for this section is room, not a prompt."""
    long_text = "\n\n".join([_LONG_A, "x" * 200, _LONG_B])
    gw = _Gateway(
        _reply("First window."),
        _reply("Second window."),
        '{"summary": "the table begins',  # third window, cut off mid-object
        '{"summary": "and again',  # ...and its retry
    )
    enrichment, report = await _enrich(gw, _jobs(1, text=long_text), window_chars=120, max_windows=4)

    assert "sec-0" not in enrichment
    assert report.sections_partial == 1 and report.failure_classes == {"TRUNCATED": 1}


async def test_a_single_window_section_is_never_partial() -> None:
    """`sections_partial` must stay 0 on the ordinary document. A one-window section that fails is
    a plain failure, and reporting it as a partial would make every truncation look like a
    windowing problem."""
    gw = _Gateway("just prose", "still prose")
    _enrichment, report = await _enrich(gw, _jobs(1))

    assert report.sections_failed == 1 and report.sections_partial == 0
    assert report.partial_section_ids == ()


async def test_every_window_parsing_still_merges_into_one_enrichment() -> None:
    """The atomicity rule must not cost the case it was built around -- a long section that worked
    before still works, with every window's contribution in it."""
    long_text = "\n\n".join([_LONG_A, "x" * 200, _LONG_B])
    gw = _Gateway(_reply("First window."), _reply("Second window."), _reply("Third window."))
    enrichment, report = await _enrich(gw, _jobs(1, text=long_text), window_chars=120, max_windows=4)

    assert enrichment["sec-0"].summary == "First window. Second window. Third window."
    assert report.sections_enriched == 1 and report.sections_partial == 0


async def test_the_input_cap_is_not_a_partial() -> None:
    """A section longer than `max_windows` allows is a DISCLOSED truncation of the INPUT, decided
    before any call. Every window that was read parsed, so the section is enriched -- counting it
    as partial would mark every over-long section in a corpus incomplete."""
    long_text = "\n\n".join(f"Block {i} of this very long section." for i in range(40))
    gw = _Gateway(*[_reply()] * 40)
    enrichment, report = await _enrich(gw, _jobs(1, text=long_text), window_chars=100, max_windows=2)

    assert "sec-0" in enrichment
    assert report.windows == 2 and report.sections_partial == 0


async def test_the_window_cap_stops_the_calls_too() -> None:
    long_text = "\n\n".join(f"Block {i} of this very long section." for i in range(40))
    gw = _Gateway(*[_reply()] * 40)
    _, report = await _enrich(gw, _jobs(1, text=long_text), window_chars=100, max_windows=2)

    assert report.windows == 2 and len(gw.prompts) == 2


# --- the report -------------------------------------------------------------------------------------


async def test_the_report_counts_quotes_kept_against_quotes_offered() -> None:
    """The plan's acceptance is a RATE (">= 95% of quotes validate"), so the denominator has to be
    what the model offered, not what survived."""
    gw = _Gateway(
        _reply(
            key_values=[
                {"name": "emd", "value": "5", "unit": None, "quote": "an EMD of Rs. 5 lakh"},
                {"name": "ghost", "value": "1", "unit": None, "quote": "a bank guarantee of Rs. 1 crore"},
            ]
        )
    )
    _, report = await _enrich(gw, _jobs(1))
    assert report.quotes_seen == 2 and report.quotes_kept == 1 and report.quote_rate == 0.5


async def test_an_unknown_facet_is_reported_once_per_document() -> None:
    gw = _Gateway(_reply(facet_ids=["eligibility", "invented"]), _reply(facet_ids=["invented"]))
    enrichment, report = await _enrich(gw, _jobs(2))
    assert report.unknown_facets == ("invented",)
    assert enrichment["sec-0"].facet_ids == ("eligibility",)


# --- the attempt budget --------------------------------------------------------------------


async def test_the_default_budget_is_the_two_attempts_it_replaced() -> None:
    """`max_attempts` must change nothing until an operator raises it. It is TOTAL calls per
    window counting the first, not a retry count, so the default of 2 is exactly the pair of
    hardcoded literals it replaced."""
    gw = _Gateway("nope", "still nope", _reply())
    enrichment, report = await _enrich(gw, _jobs(1))
    assert len(gw.prompts) == 2, "two attempts, and the third reply is never asked for"
    assert enrichment == {} and report.sections_failed == 1


async def test_a_raised_budget_recovers_a_section_the_pair_of_attempts_lost() -> None:
    """What the budget DOES, which is not the same as what it was reached for.

    A section that fails enrichment does not merely lose its summary: it is written with no
    `facet_ids` and no `amends` either, which closes the score, facet AND amendment paths into
    Phase-8 selection at once. So a recoverable reply is worth recovering.

    BUT THE REAL CORPUS FAILURE IS NOT THIS SHAPE, and raising the budget did not fix it: on
    `corrigendum-5.pdf` 2 attempts lost three substantive sections and 5 attempts lost two,
    swapping which ones. Those replies were valid JSON truncated at `section_enrich_max_tokens`,
    so another attempt only re-rolls the reply's length. This test covers a GENUINELY flaky
    reply -- prose, then prose, then JSON -- which is the case the budget is for."""
    gw = _Gateway("prose", "more prose", "yet more prose", _reply())
    enrichment, report = await _enrich(gw, _jobs(1), max_attempts=5)
    assert len(gw.prompts) == 4, "it kept going and stopped the moment a reply was usable"
    assert set(enrichment) == {"sec-0"} and report.sections_failed == 0


async def test_a_raised_budget_costs_nothing_on_a_section_that_already_works() -> None:
    """Early exit on the first usable reply — the same shape as `architecture_extract_retries`
    and `doc_profile_retries`, so raising the ceiling is not a bill for every document."""
    gw = _Gateway(_reply(), _reply())
    enrichment, _ = await _enrich(gw, _jobs(1), max_attempts=5)
    assert len(gw.prompts) == 1 and set(enrichment) == {"sec-0"}


async def test_a_gateway_error_still_does_not_consume_the_budget() -> None:
    """A gateway failure is not a parse failure and there is nothing to re-prompt on, so a raised
    budget must not turn one unreachable endpoint into five calls against it."""
    gw = _Gateway(RuntimeError("down"), _reply())
    enrichment, report = await _enrich(gw, _jobs(1), max_attempts=5)
    assert len(gw.prompts) == 1, "the budget is for unusable REPLIES, not for a dead endpoint"
    assert enrichment == {} and report.sections_failed == 1


async def test_a_budget_of_one_means_one_call_and_no_retry() -> None:
    gw = _Gateway("nope", _reply())
    enrichment, _ = await _enrich(gw, _jobs(1), max_attempts=1)
    assert len(gw.prompts) == 1 and enrichment == {}


async def test_every_retry_carries_the_reason_the_last_reply_was_unusable() -> None:
    """The nudge is what makes a retry different from the same call again."""
    gw = _Gateway("nope", "still nope", _reply())
    await _enrich(gw, _jobs(1), max_attempts=3)
    assert len(gw.prompts) == 3
    for prompt in gw.prompts[1:]:
        assert "could not be used" in prompt and "no JSON object" in prompt


async def test_a_failed_section_is_named_and_classified_not_just_counted() -> None:
    """`sections_failed` is a number; the question is WHICH sections and WHY.

    A section that fails enrichment is written with no summary, no `facet_ids` and no `amends`,
    which closes the score, facet AND amendment paths into Phase-8 selection at once. So the ids
    are the only record of what an aggregate answer can no longer see, and the class is what
    decides the fix: TRUNCATED wants output budget, UNUSABLE_OUTPUT wants a prompt."""
    gw = _Gateway(
        '{"summary": "it opened and never closed',  # sec-0, truncated, twice
        '{"summary": "again truncated',
        "just prose",  # sec-1, unusable, twice
        "still prose",
        _reply(),  # sec-2 succeeds
    )
    _enrichment, report = await _enrich(gw, _jobs(3), concurrency=1)

    assert report.sections_failed == 2 and report.sections_enriched == 1
    assert report.failed_section_ids == ("sec-0", "sec-1")
    assert report.failure_classes == {"TRUNCATED": 1, "UNUSABLE_OUTPUT": 1}


async def test_a_section_the_gateway_never_answered_is_classified_as_such() -> None:
    """No reply is not a bad reply. A gateway failure has no `reason_code` from the model, and
    recording it as an unusable OUTPUT would blame the model for an unreachable endpoint."""
    gw = _Gateway(RuntimeError("down"))
    _enrichment, report = await _enrich(gw, _jobs(1))
    assert report.failed_section_ids == ("sec-0",)
    assert report.failure_classes == {"GATEWAY_FAILURE": 1}


# --- what it actually spent -------------------------------------------------------------------------


class _MeteredGateway(_Gateway):
    """A gateway that reports usage, like the real `OpenAICompatGateway` does.

    The plain `_Gateway` above has NO `complete_with_usage` -- every other test in this file
    therefore exercises the fallback path, which is the regression guard that a conforming gateway
    with only `complete()` still enriches."""

    def __init__(self, *replies: str | Exception, tokens: int = 700, finish_reason: str = "stop") -> None:
        super().__init__(*replies)
        self._tokens = tokens
        self._finish_reason = finish_reason

    async def complete_with_usage(self, prompt: str, **kw: Any) -> tuple[str, dict[str, Any]]:
        text = await self.complete(prompt, **kw)
        return text, {
            "prompt_tokens": 1200,
            "completion_tokens": self._tokens,
            "total_tokens": 1200 + self._tokens,
            "estimated": False,
            "finish_reason": self._finish_reason,
        }


async def test_the_report_carries_what_the_calls_actually_spent() -> None:
    """A CEILING IS NOT A SPEND, and the argument for raising `section_enrich_max_tokens` from
    1,200 to 3,000 rests on that distinction. It is only honest if the real number is observable:
    a section that finishes in 700 tokens still costs 700 under a 3,000 ceiling."""
    gw = _MeteredGateway(_reply(), _reply(), tokens=700)
    _enrichment, report = await _enrich(gw, _jobs(2), max_tokens=3000)

    assert report.completion_tokens == 1400
    assert report.calls_at_cap == 0


async def test_a_call_that_stopped_at_the_ceiling_is_counted() -> None:
    """`finish_reason="length"` is the server's own statement that it ran out of room. It is the
    early warning that 3,000 is becoming the next 1,200 -- and NOT a failure count: this reply
    stopped at the cap with its JSON already closed and the section is enriched."""
    gw = _MeteredGateway(_reply(), tokens=3000, finish_reason="length")
    enrichment, report = await _enrich(gw, _jobs(1), max_tokens=3000)

    assert "sec-0" in enrichment and report.sections_failed == 0
    assert report.calls_at_cap == 1 and report.completion_tokens == 3000


async def test_the_server_saying_length_classifies_the_failure_as_truncated() -> None:
    """The parser infers TRUNCATED from an object that opened and never closed, which is right in
    the common case and still an inference. A reply cut off BETWEEN two complete objects parses as
    valid JSON of the wrong shape -- INVALID_SCHEMA, which sends an operator to the schema for a
    budget problem. The server's own `finish_reason` settles it."""
    gw = _MeteredGateway('{"facet_ids": ["eligibility"]}', tokens=3000, finish_reason="length")
    _enrichment, report = await _enrich(gw, _jobs(1), max_tokens=3000, max_attempts=1)

    assert report.sections_failed == 1
    assert report.failure_classes == {"TRUNCATED": 1}


async def test_a_gateway_without_usage_reporting_still_enriches() -> None:
    """`complete_with_usage` is a capability, not a requirement of the `LLMGateway` Protocol. A
    gateway that does not have it reports nothing and loses nothing else."""
    gw = _Gateway(_reply())
    enrichment, report = await _enrich(gw, _jobs(1))

    assert "sec-0" in enrichment
    assert report.completion_tokens == 0 and report.calls_at_cap == 0


# --- what the call is FOR -----------------------------------------------------------------------


class _PolicyGateway(_Gateway):
    """Records the sampling task it was given, and advertises that it honours one.

    Shaped like `OpenAICompatGateway`, which is the only adapter that acts on a task."""

    honours_sampling_task = True

    def __init__(self, *replies: str | Exception) -> None:
        super().__init__(*replies)
        self.tasks: list[str | None] = []

    async def complete_with_usage(self, prompt: str, **kw: Any) -> tuple[str, dict[str, Any]]:
        self.tasks.append(kw.get("task"))
        return await self.complete(prompt), {"completion_tokens": 400, "finish_reason": "stop"}


async def test_enrichment_calls_are_classified_as_structured_extraction() -> None:
    """Phase 4's reply is stored as selection metadata, not read as prose, so it must reach the
    endpoint at the temperature it asked for. Before this it was nudged to 0.6 and three ingests of
    identical bytes extracted different content every time."""
    from rag_core import sampling

    gw = _PolicyGateway(_reply(), _reply())
    enrichment, _report = await _enrich(gw, _jobs(2))

    assert len(enrichment) == 2
    assert gw.tasks == [sampling.STRUCTURED_EXTRACTION] * 2


async def test_the_claims_that_failed_quote_validation_are_named() -> None:
    """A RATE WITHOUT EVIDENCE IS NOT ACTIONABLE, and that is what the floor warning used to be.

    `parse_section_enrichment` has always returned the items it dropped for an unfindable quote,
    and nothing read them -- so when three of fourteen 3,000-token ingests landed under the 0.95
    floor, WHICH claims the document could not support was unrecoverable. The items are
    masked-derived (the model only ever saw masked text), the same standing as `reply_head`."""
    gw = _Gateway(
        _reply(
            key_values=[
                {"name": "emd", "value": "5", "unit": None, "quote": "an EMD of Rs. 5 lakh"},
                {"name": "ghost", "value": "1", "unit": None, "quote": "a bank guarantee never written"},
            ],
            propositions=[{"text": "Invented requirement.", "quote": "text that is not in the section"}],
        )
    )
    _enrichment, report = await _enrich(gw, _jobs(1))

    assert report.quotes_seen == 3 and report.quotes_kept == 1
    assert report.sections_with_unquoted == ("sec-0",)
    assert report.quote_rate is not None and report.quote_rate < QUOTE_RATE_FLOOR


async def test_a_document_whose_every_quote_validates_names_no_section() -> None:
    gw = _Gateway(_reply(), _reply())
    _enrichment, report = await _enrich(gw, _jobs(2))

    assert report.quotes_kept == report.quotes_seen
    assert report.sections_with_unquoted == ()
