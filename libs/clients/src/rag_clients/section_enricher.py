"""Section-enrichment adapter (RAG v2 Phase 4): one gateway call per section, strictly validated.

Mirrors ``rag_clients.doc_profiler`` exactly — the prompt and the parser are pure logic in
``rag_core.section_enrich``; this holds the gateway, the retry, the windowing, the
concurrency bound and the fail-safe. Where Phase 3 made ONE call per document, this makes one per
SECTION, and everything that is different here follows from that.

FAIL-SAFE, PER SECTION. A section whose call fails, times out, or answers unusably contributes no
enrichment and is counted; the document's other sections are unaffected and the document indexes
exactly as Phase 5 indexed it minus those sections' summaries and propositions. ``enrich`` never
raises. The alternative — one slow section failing a document's ingest — would make a model
hiccup an ingest outage, and at one call per section the chance of at least one hiccup per
document is far higher than it was in Phase 3.

BOUNDED CONCURRENCY, which Phase 3 had no need for. The plan says "parallel, batched"; a 200-
section tender fired at a gateway all at once is a self-inflicted denial of service against a
tenant's own model endpoint, and on a CPU llama.cpp tier it is simply a queue with extra steps.
A semaphore of ``concurrency`` is the whole mechanism.

WINDOWING, ATOMICALLY. A section longer than ``window_chars`` is split (``section_windows``) and
each window enriched separately, then merged (``merge_windows``) -- which refuses the merge unless
EVERY window parsed, because a half-merged section is stored and counted as a success while
Phase 8 selects on a third of its metadata. Each window's quotes are validated against THAT
WINDOW, because that is the text the model saw.

ONE RETRY per window, on the same terms and for the same reason as the profile call: nothing
behind ``LLMGateway.complete`` speaks a constrained-decoding API, so strict validation plus one
retry carrying a CORRECTIVE LINE is what is available, and an unmodified retry at
``temperature=0.0`` would re-read the same prompt and write the same bytes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace

from rag_clients.llm_task import complete_for_task
from rag_common import get_logger
from rag_core import sampling
from rag_core.interfaces import LLMGateway
from rag_core.section_enrich import (
    EnrichParse,
    EnrichReport,
    SectionEnrichment,
    build_section_enrich_prompt,
    merge_windows,
    parse_section_enrichment,
    section_windows,
)

_log = get_logger(__name__)

_RETRY_SUFFIX = (
    "\n\nYour previous reply could not be used: {reason}. Reply with ONLY the JSON object "
    "described above — no prose before or after it, no code fence."
)

# The plan's Phase-4 acceptance threshold (">= 95% of quotes validate"), here as the level below
# which a document's enrichment is worth an operator's attention. NOT an enforcement gate: the
# validated items are still correct and still worth storing when the rate is poor — what a low
# rate says is that the model is fabricating spans, which is a prompt or model problem and not a
# reason to throw away the evidence that did check out.
QUOTE_RATE_FLOOR = 0.95


@dataclass(frozen=True, slots=True)
class _CallCost:
    """What ONE enrichment call spent, and whether the output ceiling cut its reply short.

    WHY THE SPEND IS CARRIED AT ALL. `section_enrich_max_tokens` is a CEILING, and the argument
    for raising it from 1,200 to 3,000 (2026-11-02) was that a ceiling is not a spend -- a section
    that finishes in 800 tokens still costs 800. That argument is only honest if the real number
    is observable, so it is measured here rather than assumed from the setting.

    ``at_cap`` is ``finish_reason == "length"``, the server's own statement that it stopped
    because it ran out of budget. It is the ONLY authoritative truncation signal available: the
    parser can say a JSON object opened and never closed, which is strong evidence and still an
    inference. Same shape and same reasoning as `section_mapper._CallCost`.
    """

    completion_tokens: int = 0
    at_cap: bool = False


@dataclass(frozen=True, slots=True)
class SectionJob:
    """One section to enrich. EVERY FIELD EXCEPT ``section_id`` MUST ALREADY BE MASKED.

    ``section_id`` is identity and stays as the tree wrote it — it is the key the enrichment is
    stored under and the ``Chunk.section_id`` of every unit built from it, so masking it would
    point the enrichment at a row that does not exist."""

    section_id: str
    breadcrumb: str
    text: str


class LLMSectionEnricher:
    """Enriches a document's sections through the KB's own sovereign gateway."""

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        max_tokens: int = 3000,
        concurrency: int = 4,
        window_chars: int = 12_000,
        max_windows: int = 4,
        max_attempts: int = 2,
    ) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens
        self._concurrency = max(1, concurrency)
        self._window_chars = window_chars
        self._max_windows = max_windows
        # TOTAL attempts, counting the first call -- not a retry count. `max_attempts=1` means
        # one call and no retry, and the default of 2 is exactly the behaviour this parameter
        # replaced, so adding it changes nothing until an operator raises it.
        self._max_attempts = max(1, max_attempts)

    async def enrich(
        self,
        jobs: Sequence[SectionJob],
        *,
        profile_json: str,
        facet_ids: Sequence[str],
        doc_id: str = "",
    ) -> tuple[dict[str, SectionEnrichment], EnrichReport]:
        """Enrich every job, in parallel up to ``concurrency``. Returns what succeeded and a report.

        The dict is keyed by ``section_id`` because that is how both consumers read it: the catalog
        updates one ``sections`` row per key, and the chunker looks up the section it is currently
        walking. A section absent from the dict was not enriched, which the report counts and
        distinguishes from one that was enriched and had nothing to say.
        """
        if not jobs:
            return {}, EnrichReport()
        gate = asyncio.Semaphore(self._concurrency)

        async def run(job: SectionJob) -> tuple[SectionJob, list[EnrichParse], int, list[_CallCost]]:
            async with gate:
                return await self._enrich_one(job, profile_json=profile_json, facet_ids=facet_ids, doc_id=doc_id)

        results = await asyncio.gather(*(run(job) for job in jobs))

        out: dict[str, SectionEnrichment] = {}
        enriched = failed = windows = seen = kept = amends_found = amends_unquoted = 0
        partial = tokens = at_cap = 0
        unknown: dict[str, None] = {}
        failed_ids: list[str] = []
        partial_ids: list[str] = []
        unquoted_ids: dict[str, None] = {}
        classes: dict[str, int] = {}
        for job, parses, window_count, costs in results:
            windows += window_count
            tokens += sum(c.completion_tokens for c in costs)
            at_cap += sum(1 for c in costs if c.at_cap)
            for parse in parses:
                seen += parse.quotes_seen
                kept += parse.quotes_kept
                amends_unquoted += 1 if parse.amends_unquoted else 0
                for facet in parse.unknown_facets:
                    unknown.setdefault(facet, None)
                # WHICH CLAIMS THE DOCUMENT COULD NOT SUPPORT, named. The parser has always
                # carried these and nothing ever read them, so a run under the quote floor
                # reported a RATE and no evidence: three of fourteen 3,000-token ingests landed
                # below 0.95 and which claims failed could not be established afterwards at all.
                # Masked-derived, like `reply_head` -- the model only ever saw masked text.
                if parse.unquoted_key_values or parse.unquoted_propositions:
                    unquoted_ids.setdefault(job.section_id, None)
                    _log.info(
                        "section_enrich_quote_dropped",
                        doc_id=doc_id,
                        section_id=job.section_id,
                        key_values=list(parse.unquoted_key_values)[:8],
                        propositions=[p[:80] for p in parse.unquoted_propositions][:5],
                        dropped=len(parse.unquoted_key_values) + len(parse.unquoted_propositions),
                    )
            # ATOMIC: every window, or none of them. See `merge_windows` -- a section merged
            # from two of its three windows is a stored, counted success with a third of its
            # facets, key values and summary missing, and nothing in the system says so.
            merged, was_partial = merge_windows(
                [p.enrichment for p in parses if p.enrichment is not None], windows=window_count
            )
            if merged is None:
                failed += 1
                failed_ids.append(job.section_id)
                if was_partial:
                    partial += 1
                    partial_ids.append(job.section_id)
                    # A WARNING, because this is the one failure the model half-answered: the
                    # section is demonstrably readable, the loss is a single window's worth of
                    # room, and the result being discarded is the correct outcome rather than an
                    # obvious one. Named separately from `section_enrich_sections_unreachable` so
                    # it is not read as a section the model could not handle at all.
                    _log.warning(
                        "section_enrich_partial_windows_refused",
                        doc_id=doc_id,
                        section_id=job.section_id,
                        windows=window_count,
                        windows_parsed=sum(1 for p in parses if p.enrichment is not None),
                    )
                # THE LAST ATTEMPT'S CODE, because that is the state the section was abandoned
                # in. A section whose first window truncated and whose retry answered prose is
                # recorded as the prose it ended on; recording the first would describe a failure
                # that was superseded. A section with no parses at all (every call raised) has no
                # code from the model and is recorded as a gateway failure instead.
                code = next((p.reason_code for p in reversed(parses) if p.reason_code), "")
                # A window that produced NO PARSE AT ALL never reached the model's reply: every
                # attempt on it raised. There is no model-side code for that, so it is the
                # gateway's. This covers both "nothing parsed" and the partial case where one
                # window's calls all raised while its siblings answered.
                if not code and len(parses) < window_count:
                    code = "GATEWAY_FAILURE"
                if code:
                    classes[code] = classes.get(code, 0) + 1
                continue
            enriched += 1
            amends_found += 1 if merged.amends is not None else 0
            out[job.section_id] = merged
        report = EnrichReport(
            sections_enriched=enriched,
            sections_failed=failed,
            windows=windows,
            quotes_seen=seen,
            quotes_kept=kept,
            unknown_facets=tuple(unknown),
            amends_found=amends_found,
            amends_unquoted=amends_unquoted,
            failed_section_ids=tuple(failed_ids),
            failure_classes=dict(classes),
            sections_partial=partial,
            partial_section_ids=tuple(partial_ids),
            completion_tokens=tokens,
            calls_at_cap=at_cap,
            sections_with_unquoted=tuple(unquoted_ids),
        )
        _report(report, doc_id)
        return out, report

    async def _enrich_one(
        self, job: SectionJob, *, profile_json: str, facet_ids: Sequence[str], doc_id: str
    ) -> tuple[SectionJob, list[EnrichParse], int, list[_CallCost]]:
        """One section: one call per window, each with its own retry. Never raises.

        The fourth element is what the section's calls SPENT, one entry per call including
        retries and failures -- the per-section record of actual consumption against
        ``max_tokens``, which a configured ceiling cannot provide."""
        windows = section_windows(job.text, self._window_chars, self._max_windows)
        if not windows:
            return job, [], 0, []
        parses: list[EnrichParse] = []
        costs: list[_CallCost] = []
        for index, window in enumerate(windows):
            prompt = build_section_enrich_prompt(
                profile_json=profile_json,
                facet_ids=facet_ids,
                breadcrumb=job.breadcrumb,
                section_text=window,
            )
            # ATTEMPTS ARE A BUDGET, NOT A PAIR, and this was hardcoded at two literals.
            #
            # DO NOT RAISE THIS EXPECTING TO FIX "no JSON object in the reply". That was measured
            # on `corrigendum-5.pdf` (2026-11-02) and it does NOT work: at 2 attempts three
            # substantive sections failed (Video Wall, Enterprise Cabinet/Rack, Firewalls); at 5
            # attempts two of those recovered and `21. Switches`, which had succeeded at 2, failed
            # all five. 3 failures became 2 by swapping which sections they were. That is churn
            # around a ceiling, not a reliability gain.
            #
            # THE CEILING IS `section_enrich_max_tokens` (1200). Every failing reply is a VALID
            # JSON OBJECT cut off mid-structure -- `reply_head` shows `{ "summary": "This section
            # outlines ...` every time -- at 3,569-4,539 characters, i.e. 965-1,227 tokens against
            # that 1,200 cap. `parse_section_enrichment` reports "no JSON object in the reply"
            # because it cannot find a COMPLETE one, which is a misleading message for a
            # truncation and is what made this look like an instruction-following failure.
            # Retrying a truncated reply re-rolls its length; it does not buy it more room.
            #
            # The budget is kept because it is configurable and costs nothing at its default of 2
            # (the behaviour it replaced), and because a GENUINELY flaky reply is a different
            # failure that it would help. Early exit on the first usable reply, so raising it
            # costs nothing on sections that already work.
            parse = None
            for attempt in range(1, self._max_attempts + 1):
                nudge = "" if parse is None else _RETRY_SUFFIX.format(reason=parse.error)
                parse, cost = await self._attempt(
                    prompt + nudge,
                    job=job,
                    source_text=window,
                    facet_ids=facet_ids,
                    doc_id=doc_id,
                    attempt=attempt,
                    window=index,
                )
                costs.append(cost)
                # A GATEWAY failure is not a parse failure and there is nothing to re-prompt on:
                # `_attempt` returns None for it, and retrying the same call is the caller's
                # concern, not this loop's. A usable reply ends the budget.
                if parse is None or parse.enrichment is not None:
                    break
            if parse is not None:
                parses.append(parse)
        # ONE LINE PER SECTION, at info. Phase 4 already makes one model call per section, so a
        # line per section is proportionate -- and the question the owner asked of the raised
        # ceiling ("monitor actual consumption, not the configured ceiling") cannot be answered
        # from a per-document total: a document whose 50 sections average 600 tokens and whose
        # one table-dense section spends 2,900 is the case that matters, and the average hides it.
        _log.info(
            "section_enrich_section_done",
            doc_id=doc_id,
            section_id=job.section_id,
            windows=len(windows),
            calls=len(costs),
            completion_tokens=sum(c.completion_tokens for c in costs),
            max_tokens=self._max_tokens,
            calls_at_cap=sum(1 for c in costs if c.at_cap),
        )
        return job, parses, len(windows), costs

    async def _attempt(
        self,
        prompt: str,
        *,
        job: SectionJob,
        source_text: str,
        facet_ids: Sequence[str],
        doc_id: str,
        attempt: int,
        window: int,
    ) -> tuple[EnrichParse | None, _CallCost]:
        """One call, and what it spent. None when the GATEWAY failed (there is nothing to retry a
        parse on); a parse with ``enrichment=None`` when the model answered and the answer was
        unusable.

        STRUCTURED_EXTRACTION, so the caller's ``temperature=0.0`` reaches the endpoint instead of
        being nudged to 0.6 (`rag_core.sampling`). `complete_for_task` also carries the usage
        back where the gateway reports it, so a stub with only ``complete()`` still works and
        simply reports nothing."""
        try:
            raw, metered = await complete_for_task(
                self._gw,
                prompt,
                max_tokens=self._max_tokens,
                temperature=0.0,
                task=sampling.STRUCTURED_EXTRACTION,
            )
        except Exception as exc:  # fail-safe: a model error costs this section, never the document
            _log.warning(
                "section_enrich_call_failed",
                doc_id=doc_id,
                section_id=job.section_id,
                window=window,
                attempt=attempt,
                error=str(exc)[:160],
            )
            # A call that raised reported no usage. NOT free -- unmeasurable.
            return None, _CallCost()
        cost = _CallCost(
            completion_tokens=int(metered.get("completion_tokens") or 0),
            at_cap=str(metered.get("finish_reason") or "") == "length",
        )
        parse = parse_section_enrichment(raw, section_id=job.section_id, source_text=source_text, facet_ids=facet_ids)
        if parse.enrichment is None and cost.at_cap and parse.reason_code != "TRUNCATED":
            # THE SERVER SAID IT RAN OUT OF ROOM, so the code is TRUNCATED whatever the reply's
            # brace state looks like. The parser's own TRUNCATED is an inference from an unclosed
            # object and is right in the common case; a reply cut off between two complete objects
            # parses as INVALID_SCHEMA and would send an operator to the schema for a budget
            # problem. Narrowed to failures: a call can stop at the cap with its JSON already
            # closed, which is not a failure at all.
            parse = replace(parse, reason_code="TRUNCATED")
        if parse.enrichment is None:
            # `reply_chars` and a bounded head, because `reason` alone cannot tell PROSE from
            # MALFORMED JSON and those have different fixes: the first is an instruction-following
            # failure, the second a schema one. The reply derives from text this caller already
            # masked, and this is a log line rather than anything indexed or persisted.
            _log.info(
                "section_enrich_unusable_reply",
                doc_id=doc_id,
                section_id=job.section_id,
                window=window,
                attempt=attempt,
                reason=parse.error,
                reply_chars=len(raw or ""),
                reply_head=(raw or "")[:120].replace("\n", " "),
                completion_tokens=cost.completion_tokens,
                at_cap=cost.at_cap,
            )
        return parse, cost


def _report(report: EnrichReport, doc_id: str) -> None:
    """One line per document, and a WARNING only where something was genuinely lost.

    A quote rate below the plan's acceptance threshold is a WARNING even when every section
    enriched: the rows are written either way, and the number is the only thing that says how much
    of what the model claimed the document actually supports. An amends dropped for an unfindable
    quote is called out separately — it is the single most consequential loss this phase has, and
    with no line it is indistinguishable from a section that amends nothing."""
    rate = report.quote_rate
    _log.info(
        "section_enrich_done",
        doc_id=doc_id,
        sections_enriched=report.sections_enriched,
        sections_failed=report.sections_failed,
        windows=report.windows,
        quotes_seen=report.quotes_seen,
        quotes_kept=report.quotes_kept,
        # SIX PLACES, NOT THREE. The acceptance floor is 0.95 and the margin on real documents
        # is thin: 0.949474 rounds to 0.949 and 0.9496 would round to 0.950, which READS as
        # passing a floor it does not pass. The comparison below has always used the unrounded
        # rate, so the gate was never wrong -- what was wrong is that the number an operator reads
        # could not be trusted to the precision the decision turns on. `quotes_kept`/`quotes_seen`
        # are logged beside it so the exact fraction is always recoverable.
        quote_rate=round(rate, 6) if rate is not None else None,
        amends_found=report.amends_found,
        failure_classes=dict(report.failure_classes) or None,
        sections_partial=report.sections_partial or None,
        # WHAT IT SPENT, beside what it was allowed to spend. These two numbers in one line are
        # what makes a raised ceiling reviewable instead of a leap of faith.
        completion_tokens=report.completion_tokens or None,
        calls_at_cap=report.calls_at_cap or None,
    )
    # A WARNING, NOT A COUNT BURIED IN THE LINE ABOVE. A failed section is written with no
    # summary, no `facet_ids` and no `amends`, which closes every Phase-8 selection path into it:
    # the document keeps the section and an aggregate answer can no longer see it. On
    # `corrigendum-5.pdf` that silently cost three substantive sections, and `sections_failed=3`
    # sat in an info line nobody read for a day.
    if report.sections_failed:
        _log.warning(
            "section_enrich_sections_unreachable",
            doc_id=doc_id,
            sections_failed=report.sections_failed,
            failure_classes=dict(report.failure_classes),
            failed_section_ids=list(report.failed_section_ids[:20]),
            # Of those, the ones refused for a PARTIAL window set. They failed for a different
            # reason than the rest -- the model answered, just not for every window -- and an
            # operator chasing "why is this section unreachable" needs that split.
            sections_partial=report.sections_partial or None,
            partial_section_ids=list(report.partial_section_ids[:20]) or None,
            detail="these sections have no summary, facet_ids or amends, so Phase-8 selection "
            "cannot reach them by score, facet or amendment",
        )
    if report.amends_unquoted:
        _log.warning("section_enrich_amends_dropped", doc_id=doc_id, dropped=report.amends_unquoted)
    if rate is not None and rate < QUOTE_RATE_FLOOR:
        _log.warning(
            "section_enrich_quote_rate_low",
            doc_id=doc_id,
            quote_rate=round(rate, 6),
            floor=QUOTE_RATE_FLOOR,
            quotes_seen=report.quotes_seen,
            quotes_kept=report.quotes_kept,
            # WHERE the shortfall is, so the rate is actionable. `section_enrich_quote_dropped`
            # carries the individual claims for each of these.
            sections=list(report.sections_with_unquoted)[:20],
            # The exact fraction, because a rate is a rounded view of it and this line is the one
            # an operator acts on.
            shortfall=report.quotes_seen - report.quotes_kept,
        )
    if report.unknown_facets:
        _log.warning("section_enrich_unknown_facets", doc_id=doc_id, facets=list(report.unknown_facets)[:20])
