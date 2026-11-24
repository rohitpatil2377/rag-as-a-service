"""The Phase-9 MAP call: one extraction per selected section, through the KB's own gateway.

What an item CONTAINS and what makes it valid is ``rag_core.mapreduce``. What is here is
the adapter's contract, and the two ways it differs in posture from the Phase-4 enricher it is
otherwise shaped like:

* **It is on the LATENCY PATH.** Enrichment runs once per document at ingest; this runs on every
  aggregate question, once per selected section. It still WINDOWS a long section, but with a
  tight bound — and that bound was set by measurement, not by taste. Truncating instead was the
  first design, and on the real AEGCL corrigendum it was the single largest cause of a missed
  requirement: twelve of thirty-eight selected sections overran a 12,000-character budget, and
  a tender section puts its specification table first and its certification and bill-of-material
  clauses LAST. Every "UL 62368", "CSA 22.2", "FCC Part 15", "HBOM" and "SBOM" in the document
  fell off the end of a truncated prompt and appeared nowhere in the answer — the precise
  failure map-reduce exists to prevent, reintroduced inside the map step.
* **A failed section is not a failed answer.** Each section's call is independent and its
  failure costs exactly that section's items; the reduce step still runs on what came back, and
  the coverage statement is what tells the reader how much that was. This is the opposite of the
  router's posture (one decision, so a failure must fall back to a decision) and the same as
  Phase 4's (many independent calls, so a partial result is the ordinary outcome).

ONE RETRY, on an UNUSABLE REPLY only — the same rule as every other JSON call in this package.
A gateway that just raised is not retried: there is nothing to retry a parse on, and a second
call to a failing endpoint is one more call on the latency path for a section whose items the
answer can do without.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from rag_common import get_logger
from rag_core import LLMGateway
from rag_core.mapreduce import MapParse, MappedItem, build_map_prompt, parse_map_items
from rag_core.section_enrich import section_windows

_log = get_logger(__name__)

_RETRY_SUFFIX = (
    "\n\nYour previous reply could not be used: {reason}. Reply with ONLY the JSON list "
    "described above — no prose before or after it, no code fence. An empty list [] is a valid "
    "reply if nothing in the section is relevant."
)


@dataclass(frozen=True, slots=True)
class MapJob:
    """One section to extract from. EVERY FIELD EXCEPT THE IDS MUST ALREADY BE MASKED.

    ``text`` is the section's CHUNK text, not ``blocks.text`` — see the note in
    ``rag_core.mapreduce``. ``section_id`` and ``doc_id`` are identity and stay as stored:
    they are what an item is attributed to and what its citation resolves through."""

    section_id: str
    doc_id: str
    breadcrumb: str
    text: str


@dataclass(frozen=True, slots=True)
class MapReport:
    """What a whole map pass cost and lost.

    ``sections_contributed`` is the numerator of the plan's coverage statement, and is
    deliberately NOT ``sections_mapped``: a section that was read and had nothing relevant in it
    did not contribute, and an answer that counted it would overstate how much of the document
    it actually used.

    ``sections_truncated`` counts the sections still too long for ``max_windows`` windows — the
    ones where the map genuinely did not read to the end. It is the signal that the windows
    bound is costing content, and it is the number to watch before widening it.

    TWO DIFFERENT TRUNCATIONS LIVE HERE AND MUST NOT BE ADDED TOGETHER. ``sections_truncated``
    is COVERAGE truncation: text past ``section_chars * max_windows`` that was never shown to
    the model. ``calls_at_cap`` is OUTPUT truncation: a window whose reply stopped at
    ``max_tokens``, so the model saw everything and ran out of room to report it. The first is
    fixed by widening the windows bound, the second by raising the output cap or narrowing the
    windows — opposite remedies, and a single "truncated" number cannot tell an operator which
    one they need."""

    sections_mapped: int = 0
    sections_failed: int = 0
    sections_contributed: int = 0
    sections_truncated: int = 0
    windows: int = 0
    items: int = 0
    dropped_unquoted: int = 0
    dropped_unknown_facet: int = 0
    #: Items the model returned and `_MAX_ITEMS_PER_WINDOW` discarded. Non-zero means the
    #: extraction was CUT, not completed — the section holds more than this call could report,
    #: and the answer is short by that much with nothing else to show for it.
    dropped_over_cap: int = 0
    #: A bounded sample of the quotes the validator rejected, across the whole run. The RATE is
    #: not diagnosable without them: a paraphrase is the rule working as intended, a verbatim
    #: quote lost to a punctuation difference is the rule costing real evidence, and the count
    #: looks identical either way. Measured on a real tender at 7.9% of items, and the split is
    #: lopsided: the model reformats (elides with `...`, welds a list stem to its bullet) far
    #: more than it paraphrases, so most rejections discard evidence the section really holds.
    dropped_quotes: tuple[str, ...] = ()
    retries: int = 0
    #: Every call that did not yield usable items, counted BY KIND, highest first. `sections_failed`
    #: says how much was lost; this says what to do about it, and the two answers are unrelated:
    #: `read_timeout` means the budget is too small for this `max_tokens` or the endpoint is slow,
    #: `unreachable` means it is down, `http_status` means it rejected us, and `unusable_reply`
    #: means it answered fine and the model's JSON was wrong. Measured during the scheduler
    #: benchmark, a twenty-minute endpoint degradation lost 28 of 63 calls — ALL of them
    #: `read_timeout` at exactly the configured limit, with every other kind at zero. A single
    #: `sections_failed=6` cannot be told apart from six bad replies, and the response to those
    #: two situations is not the same one.
    failures: tuple[tuple[str, int], ...] = ()
    #: Every model call the pass made, retries included — the DENOMINATOR `failures` and
    #: `calls_at_cap` are rates of. `windows` is not it: a window that was retried cost two
    #: calls, and dividing by windows understates both.
    calls: int = 0
    #: Windows whose reply stopped because it hit `max_tokens` (`finish_reason="length"`).
    #: OUTPUT truncation, not coverage truncation — see the class docstring. The model read the
    #: whole window and ran out of room to report it, so the items it had not yet written are
    #: gone with NOTHING to show for them: the reply parses, the section contributes, and no
    #: failure, retry or warning is recorded anywhere. That silence is why this is counted.
    #: Measured on a real tender at 63% of windows at 12,000 chars and 8% at 3,000 — the
    #: dominant evidence-loss mechanism in the stage, and until this field existed it was
    #: invisible outside a bespoke experiment.
    calls_at_cap: int = 0
    #: Completion tokens the pass spent, server-reported where the endpoint says so. Zero when
    #: the gateway cannot report usage, which is not the same as a free pass.
    completion_tokens: int = 0


#: How many rejected quotes one RUN keeps. Enough to see the shape of the failures across a
#: document without carrying a log around in a dataclass.
_QUOTE_SAMPLE = 12


@dataclass(frozen=True, slots=True)
class _CallCost:
    """What ONE model call spent, and whether the token ceiling cut its reply short.

    Threaded back through the return values rather than accumulated on the mapper instance: a
    mapper is not guaranteed to be per-pass, and an instance counter would silently pool two
    concurrent questions' costs into both their reports."""

    completion_tokens: int = 0
    at_cap: bool = False


class LLMSectionMapper:
    """Extracts question-relevant items from selected sections, in parallel up to ``concurrency``."""

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        max_tokens: int = 2400,
        concurrency: int = 4,
        section_chars: int = 12_000,
        max_windows: int = 3,
    ) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens
        self._concurrency = max(1, concurrency)
        self._section_chars = section_chars
        self._max_windows = max(1, max_windows)

    async def map_sections(
        self,
        jobs: Sequence[MapJob],
        *,
        query: str,
        profile_json: str,
        facet_ids: Sequence[str],
    ) -> tuple[list[MappedItem], MapReport]:
        """Extract from every job. Returns the items in SECTION ORDER and a report.

        Section order, not completion order: ``item_id`` is assigned here and the reduce prompt
        groups by facet, but a reader following citations through a long answer still reads the
        document forwards. Completion order would make the ids depend on which call happened to
        return first, which would also make the map cache's contents depend on scheduling.
        """
        if not jobs:
            return [], MapReport()
        # ONE permit PER MODEL CALL, for the whole stage -- not one per section.
        #
        # The bound exists to protect the tenant's endpoint, and what loads an endpoint is a
        # request in flight, so that is what it should count. Bounding SECTIONS instead left the
        # endpoint idle whenever the selected sections outnumbered neither the bound nor each
        # other in length: measured over six runs, the map stage ran at 6.9-8.4 calls in flight
        # against a bound of 12, and its wall time equalled the LONGEST SECTION'S window chain to
        # within 0.1 s in every one of them -- one 58,484-char section decided the latency of the
        # whole stage while thirty-three others sat finished. See the scorecard's critical-path
        # row for the measurement.
        #
        # The gate is passed DOWN to the call site. `_map_one` must never hold it: it awaits its
        # windows, each of which acquires it, so a permit held across that await would be a
        # permit waiting on itself.
        gate = asyncio.Semaphore(self._concurrency)
        # `gather` returns in ARGUMENT order, not completion order, so the per-section results
        # below -- and the item ids assigned from them -- do not depend on which call came back
        # first. That property is what makes this change invisible downstream.
        results = await asyncio.gather(
            *(
                self._map_one(job, query=query, profile_json=profile_json, facet_ids=facet_ids, gate=gate)
                for job in jobs
            )
        )

        items: list[MappedItem] = []
        unknown: dict[str, None] = {}
        mapped = failed = contributed = truncated_n = unquoted = unknown_n = retried = window_n = over_cap = 0
        quote_sample: list[str] = []
        failures: Counter[str] = Counter()
        call_n = at_cap_n = completion_n = 0
        for _job, parses, truncated, retries, window_count, kinds, costs in results:
            failures.update(kinds)
            call_n += len(costs)
            at_cap_n += sum(1 for c in costs if c.at_cap)
            completion_n += sum(c.completion_tokens for c in costs)
            # A SECTION counts as mapped when ANY of its windows parsed: the others cost their
            # items, not the section. Counting it failed would make the coverage denominator
            # punish a long section for the one window the model fumbled.
            section_items = [item for parse in parses for item in parse.items]
            mapped += 1 if any(p.usable for p in parses) else 0
            failed += 0 if any(p.usable for p in parses) else 1
            contributed += 1 if section_items else 0
            truncated_n += 1 if truncated else 0
            window_n += window_count
            unquoted += sum(p.dropped_unquoted for p in parses)
            over_cap += sum(p.dropped_over_cap for p in parses)
            for parse in parses:
                room = _QUOTE_SAMPLE - len(quote_sample)
                if room > 0:
                    quote_sample.extend(parse.dropped_quotes[:room])
            unknown_n += sum(p.dropped_unknown_facet for p in parses)
            retried += retries
            for parse in parses:
                for facet in parse.unknown_facets:
                    unknown.setdefault(facet, None)
            # Re-numbered as they are appended, so the ids the reduce prompt cites are dense and
            # ascending across the WHOLE answer rather than restarting per section.
            for item in section_items:
                items.append(
                    MappedItem(
                        item_id=f"m{len(items) + 1}",
                        section_id=item.section_id,
                        doc_id=item.doc_id,
                        facet_id=item.facet_id,
                        statement=item.statement,
                        quote=item.quote,
                        values=item.values,
                        clause_ref=item.clause_ref,
                        is_amendment=item.is_amendment,
                        old_value=item.old_value,
                        new_value=item.new_value,
                    )
                )
        report = MapReport(
            sections_mapped=mapped,
            sections_failed=failed,
            sections_contributed=contributed,
            sections_truncated=truncated_n,
            windows=window_n,
            items=len(items),
            dropped_unquoted=unquoted,
            dropped_over_cap=over_cap,
            dropped_quotes=tuple(quote_sample),
            dropped_unknown_facet=unknown_n,
            retries=retried,
            # Sorted by count so the dominant kind reads first, then by name so the tuple is
            # deterministic for equal counts — a report that reorders between runs is a report
            # nobody can diff.
            failures=tuple(sorted(failures.items(), key=lambda kv: (-kv[1], kv[0]))),
            calls=call_n,
            calls_at_cap=at_cap_n,
            completion_tokens=completion_n,
        )
        # DELIBERATELY NOT A WARNING, unlike `dropped_over_cap` next door. Being at the output
        # cap is the NORMAL state of the shipped configuration — measured at 63% of windows at
        # 12,000 chars on a real tender — so a per-pass warning would fire on every aggregate
        # question until the window size changes, and a warning that always fires is one nobody
        # reads. The count rides the `rag_v2_map_reduce` line instead, where a dashboard can
        # alert on the RATE and an operator can see it move when they change the knob.
        if unknown:
            _log.info("map_unknown_facets", facets=sorted(unknown)[:10], count=len(unknown))
        if failures:
            # WARNING, not a field on an info line: every one of these is evidence the answer is
            # short, and the kinds are what say whether that is the endpoint's fault or ours.
            _log.warning(
                "map_call_failures",
                sections_failed=failed,
                windows=window_n,
                # The rate's real denominator, beside the window count: a retried window cost
                # two calls, so `windows` alone makes every failure rate look worse than it is.
                calls=call_n,
                kinds=dict(report.failures),
            )
        return items, report

    async def _map_one(
        self,
        job: MapJob,
        *,
        query: str,
        profile_json: str,
        facet_ids: Sequence[str],
        gate: asyncio.Semaphore,
    ) -> tuple[MapJob, list[MapParse], bool, int, int, list[str], list[_CallCost]]:
        """Every window of one section, concurrently under the stage-wide gate. Returns the
        parses plus the costs, and the KIND of each window's final failure.

        The windows used to run in SEQUENCE because the semaphore was held per SECTION, so
        fanning them out here would have let one long section open ``max_windows`` times more
        connections than the bound allowed. That is no longer how the bound works: ``gate``
        guards each individual call, so a section can offer all its windows at once and the
        stage still never exceeds ``concurrency`` requests in flight. What this buys is the
        critical path -- a section's windows no longer queue behind each other while the
        endpoint is idle.

        A section may now offer more windows than the whole bound. That is intentional and
        safe: they QUEUE on the gate rather than dispatch. It does mean a very long section can
        hold several permits at once, which is a fairness property, not a safety one -- the
        total in flight is still bounded, and the long section is precisely the one on the
        critical path.

        The RETRY stays sequential WITHIN a window, because it is a response to that window's
        own failed parse and has nothing to retry until the first call returns."""
        windows = section_windows(job.text, self._section_chars, self._max_windows)
        # Truncation is "the splitter wanted MORE windows than it was allowed", asked by letting
        # it have one more. NOT a character-count comparison: `section_windows` splits on the
        # block separator and drops it, so the windows always sum to slightly less than the
        # section even when they cover all of it — a sum comparison reports every multi-window
        # section as truncated, which is the number an operator is meant to watch before
        # widening the bound, and it was wrong by a factor of four on the real corrigendum
        # (12 reported, 3 actual).
        truncated = len(section_windows(job.text, self._section_chars, self._max_windows + 1)) > self._max_windows

        async def one_window(window: str) -> tuple[MapParse, int, str, list[_CallCost]]:
            """One window's call, its one retry, whether that retry was spent, and — when the
            window ended with nothing usable — WHY. The kind is the LAST attempt's, because that
            is the one that decided the outcome: a window whose first call timed out and whose
            retry parsed is not a failure at all.

            The costs are EVERY attempt's, not the last one's: a retried window really did spend
            two calls' tokens, and a report that counted one would under-bill the pass."""
            prompt = build_map_prompt(
                profile=profile_json,
                breadcrumb=job.breadcrumb,
                section_text=window,
                query=query,
                facet_ids=facet_ids,
            )
            parse, kind, cost = await self._attempt(prompt, job, window, facet_ids, gate=gate)
            if parse.usable:
                return parse, 0, "", [cost]
            parse, kind, retry_cost = await self._attempt(
                prompt + _RETRY_SUFFIX.format(reason=parse.error), job, window, facet_ids, gate=gate
            )
            if not parse.usable:
                _log.warning(
                    "map_section_unusable",
                    section_id=job.section_id,
                    failure_kind=kind or "unusable_reply",
                    reason=parse.error[:120],
                )
            return parse, 1, "" if parse.usable else (kind or "unusable_reply"), [cost, retry_cost]

        # Window order, not completion order: a parse is only meaningful against the window it
        # was asked about, and `gather` keeps the two aligned by position.
        outcomes = await asyncio.gather(*(one_window(window) for window in windows))
        parses = [parse for parse, _retried, _kind, _costs in outcomes]
        retries = sum(retried for _parse, retried, _kind, _costs in outcomes)
        kinds = [kind for _parse, _retried, kind, _costs in outcomes if kind]
        costs = [cost for _parse, _retried, _kind, cs in outcomes for cost in cs]
        return job, parses, truncated, retries, len(windows), kinds, costs

    async def _attempt(
        self, prompt: str, job: MapJob, text: str, facet_ids: Sequence[str], *, gate: asyncio.Semaphore
    ) -> tuple[MapParse, str, _CallCost]:
        """One model call, under one permit. The permit covers the REQUEST and nothing else.

        Parsing and quote-validation run after it is released: they are local CPU work that
        loads nothing at the endpoint, and holding a permit through them would under-use the
        bound for exactly as long as they take.

        The second element is the gateway's own failure classification, or ``""`` when the call
        returned. It is returned rather than logged-and-forgotten so the stage can COUNT kinds:
        a per-call warning tells an operator reading logs what happened, and tells a dashboard
        reading `MapReport` nothing.

        The third is what the call SPENT, taken from the usage-reporting path when the gateway
        has one — the same ``getattr`` capability check the generation service uses, so a plain
        stub with only ``complete()`` still works and simply reports nothing. This is the only
        place ``finish_reason`` is observable: it decides `MapReport.calls_at_cap`, and without
        it a window that ran out of output budget is indistinguishable from one that finished."""
        try:
            async with gate:
                with_usage = getattr(self._gw, "complete_with_usage", None)
                if with_usage is not None:
                    reply, metered = await with_usage(prompt, max_tokens=self._max_tokens)
                else:
                    reply, metered = await self._gw.complete(prompt, max_tokens=self._max_tokens), {}
        except Exception as exc:  # the answer survives without this section's items
            # Carry the gateway's own classification through. A read timeout and an unreachable
            # endpoint both land here, and telling them apart is the difference between "our
            # timeout is too low for this max_tokens" and "the endpoint is down".
            detail = getattr(exc, "details", None) or {}
            kind = detail.get("failure_kind", "")
            _log.warning(
                "map_section_call_failed",
                section_id=job.section_id,
                failure_kind=kind,
                timeout_s=detail.get("timeout_s"),
                elapsed_s=detail.get("elapsed_s"),
                error=str(exc)[:160],
            )
            return (
                MapParse(error=f"gateway error: {type(exc).__name__}{f' ({kind})' if kind else ''}"),
                kind or type(exc).__name__,
                # A call that raised reported no usage. Not "free" — unmeasurable, and the two
                # are only the same number by coincidence.
                _CallCost(),
            )
        return (
            parse_map_items(
                reply, section_id=job.section_id, doc_id=job.doc_id, section_text=text, facet_ids=facet_ids
            ),
            "",
            _CallCost(
                completion_tokens=int(metered.get("completion_tokens") or 0),
                at_cap=metered.get("finish_reason") == "length",
            ),
        )
