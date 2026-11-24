"""Document-profile adapter (RAG v2 Phase 3): one gateway call per document, strictly validated.

Mirrors every other "one small JSON call" in this package — ``query_planner_llm.LLMQueryPlanner``,
``routing.LLMRouter``, ``intent_classifier.LLMIntentClassifier``: the prompt and the parser are
pure logic in ``rag_core.doc_profile``; this holds the gateway, the retry and the fail-safe.

FAIL-SAFE, AND THAT IS THE WHOLE POSTURE. ``profile`` returns None on an unavailable model, a
timeout, an unparseable reply or a reply that is valid JSON but not a usable card, and NEVER
raises. A document with no profile is a document that indexes exactly as Phase 5 indexed it —
which is the behaviour of this whole deployment today, since the flag is off everywhere. The
alternative, letting a profile call decide whether a document gets ingested, would make a model
outage an ingest outage.

RETRIES, per plan §14 ("constrained/guided decoding if the server supports it; otherwise strict
schema validation + one retry"). Nothing in this codebase speaks a constrained-decoding API —
there is no ``response_format``, ``guided_json`` or grammar parameter anywhere behind
``LLMGateway.complete`` — so the strict-validation half is what is available, and every retry
carries a CORRECTIVE LINE naming what was wrong with the previous reply.

IT USED TO BE ONE RETRY, on the reasoning that "retrying the identical prompt would be pointless
at ``temperature=0.0``: a greedy decode re-reads the same prompt and writes the same bytes". That
is sound, and it is not what this deployment does.
:meth:`OpenAICompatGateway._eff_temp` raises a requested 0.0 to 0.6 whenever reasoning is
suppressed — which ``RAG_LLM_REASONING_EFFORT=off`` does for EVERY model — deliberately, because
a no-reasoning model decoding greedily loops and echoes the prompt's own examples back. So the
attempts are independent samples, not one answer re-fetched, and the count is now
``doc_profile_retries``.

WHY IT IS WORTH SPENDING CALLS ON. The failure is silent by design: this returns None, the
document ingests with no card, and Phase 4's enrichment prompt, Phase 7's routing and Phase 8's
facets all quietly get less. Nothing downstream reports a missing profile as a fault, so a
document that loses a coin flip is simply a worse document forever after, until it is re-ingested.
"""

from __future__ import annotations

from rag_clients.llm_task import complete_for_task
from rag_common import get_logger, get_settings
from rag_core import sampling
from rag_core.doc_profile import DocProfile, ProfileParse, parse_doc_profile
from rag_core.interfaces import LLMGateway

_log = get_logger(__name__)

# Appended to the prompt on a retry. Short and specific: the model is told what failed, not asked
# to try harder.
_RETRY_SUFFIX = (
    "\n\nYour previous reply could not be used: {reason}. Reply with ONLY the JSON object "
    "described above — no prose before or after it, no code fence."
)


class LLMDocProfiler:
    """Builds a document's profile card through the KB's own sovereign gateway."""

    def __init__(self, gateway: LLMGateway, *, max_tokens: int = 1400) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens

    async def profile(
        self, prompt: str, *, source_text: str, title: str, lang: str | None = None, doc_id: str = ""
    ) -> DocProfile | None:
        """The card, or None. ``source_text`` is what an evidence quote is checked against — the
        same text the prompt was given, which is the check's whole point; ``lang`` is the
        platform's own detection, which overrides the model's (see
        :func:`~rag_core.doc_profile.parse_doc_profile`)."""
        attempts = 1 + max(0, get_settings().doc_profile_retries)
        parsed: ProfileParse | None = None
        for attempt in range(1, attempts + 1):
            # The corrective line names what the LAST reply got wrong, so a retry is a different
            # prompt as well as a different sample. The first attempt has nothing to correct.
            correction = "" if parsed is None or parsed.error is None else _RETRY_SUFFIX.format(reason=parsed.error)
            text = prompt + correction
            parsed = await self._attempt(
                text, source_text=source_text, title=title, lang=lang, doc_id=doc_id, attempt=attempt
            )
            if parsed is not None and parsed.profile is not None:
                if attempt > 1:
                    _log.info("doc_profile_recovered_on_retry", doc_id=doc_id, attempt=attempt)
                _report(parsed, doc_id)
                return parsed.profile
            if parsed is None:
                # The GATEWAY failed, not the model: there is no reply to correct and no reason to
                # believe the next call reaches it either. Retrying a transport fault here would
                # turn one outage into `attempts` timeouts on the ingest path.
                break
        _log.warning("doc_profile_gave_up", doc_id=doc_id, attempts=attempts)
        return None

    async def _attempt(
        self, prompt: str, *, source_text: str, title: str, lang: str | None, doc_id: str, attempt: int
    ) -> ProfileParse | None:
        """One call. None when the GATEWAY failed (there is nothing to retry a parse on); a
        ``ProfileParse`` with ``profile=None`` when the model answered and the answer was unusable
        (which is exactly the case a retry can fix)."""
        try:
            # STRUCTURED_EXTRACTION, and this is the call where it matters most. This reply becomes
            # the document's FACET VOCABULARY, and a facet id is Phase 8's facet-union join key
            # while its label is a cross-encoder probe -- so re-rolling this re-rolls section
            # selection. Measured: three ingests of identical bytes produced three 10-facet
            # vocabularies sharing ONE id, because `temperature=0.0` here was being nudged to 0.6
            # before it reached the endpoint. See `rag_core.sampling`.
            raw, _usage = await complete_for_task(
                self._gw,
                prompt,
                max_tokens=self._max_tokens,
                temperature=0.0,
                task=sampling.STRUCTURED_EXTRACTION,
            )
        except Exception as exc:  # fail-safe: a model error must never cost a document its ingest
            _log.warning("doc_profile_call_failed", doc_id=doc_id, attempt=attempt, error=str(exc)[:160])
            return None
        parsed = parse_doc_profile(raw, source_text=source_text, title=title, lang=lang)
        if parsed.profile is None:
            _log.warning("doc_profile_unusable_reply", doc_id=doc_id, attempt=attempt, reason=parsed.error)
        return parsed


def _report(parsed: ProfileParse, doc_id: str) -> None:
    """Log what validation threw away.

    A dropped relation is the consequential one and gets its own WARNING: an "amends" the model
    found and the quote check rejected is the difference between a corrigendum that supersedes a
    clause and one that does not, and without this line its only trace is a row that is not there.
    """
    profile = parsed.profile
    assert profile is not None  # only called on a usable card
    if parsed.unquoted or parsed.unknown_types:
        _log.warning(
            "doc_profile_relations_dropped",
            doc_id=doc_id,
            unquoted=len(parsed.unquoted),
            unknown_types=sorted(set(parsed.unknown_types)),
            quotes=[q[:80] for q in parsed.unquoted[:5]],
        )
    _log.info(
        "doc_profile_built",
        doc_id=doc_id,
        doc_type=profile.doc_type,
        relations=len(profile.relations),
        facets=len(profile.facets),
        # The prompt asks for 6-15; outside that the card is KEPT and this says so, because a
        # two-facet taxonomy groups an aggregate answer into two buckets and nothing else reports it.
        facet_count_off_band=parsed.facet_count_off_band,
        dropped_facets=len(parsed.dropped_facets),
        has_issue_date=profile.issue_date is not None,
    )
