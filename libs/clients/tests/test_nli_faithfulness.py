"""Unit tests: NLI faithfulness aggregation + TEI /predict response parsing.

The scoring logic is exercised with a DETERMINISTIC fake entailment scorer (no model,
no network), so it runs in the zero-infra CI profile like every other unit test.
"""

from __future__ import annotations

import pytest

from rag_clients.nli_faithfulness import EntailFn, NLIFaithfulnessChecker, _focus, _support_prob, _terms
from rag_schemas import RetrievedSpan


def _span(text: str) -> RetrievedSpan:
    # NLI reads only the text (premise); score/scorer are irrelevant here, so leave them default.
    return RetrievedSpan(chunk_id="c1", doc_id="d1", text=text)


def _entail_by_keyword(keyword: str) -> EntailFn:
    """A fake NLI scorer: a hypothesis is entailed iff it mentions ``keyword``."""

    async def _fn(premise: str, hypothesis: str) -> float:
        return 0.95 if keyword in hypothesis.lower() else 0.05

    return _fn


async def test_fully_entailed_answer_is_grounded() -> None:
    checker = NLIFaithfulnessChecker(_entail_by_keyword("leave"))
    spans = [_span("Employees are entitled to 24 days of paid annual leave.")]
    verdict = await checker.check("Staff get annual leave.", spans)
    assert verdict.faithful is True
    assert verdict.score == 1.0
    assert verdict.unsupported_claims == []


async def test_unsupported_sentence_flagged() -> None:
    checker = NLIFaithfulnessChecker(_entail_by_keyword("leave"))
    spans = [_span("Employees are entitled to 24 days of paid annual leave.")]
    # Two sentences: one about leave (entailed), one about bonuses (not).
    answer = "Staff get annual leave. Everyone also receives a huge cash bonus."
    verdict = await checker.check(answer, spans)
    assert verdict.faithful is False
    # Weighted by CONTENT WORDS, not by sentence count: "staff get annual leave" carries 4 of the
    # answer's 10 content words, so 0.4 — not the 0.5 a plain sentence fraction would give. This
    # is what makes the score format-invariant (prose vs bullets), the whole point of weighting.
    assert verdict.score == pytest.approx(0.4)
    assert verdict.unsupported_claims == ["Everyone also receives a huge cash bonus."]


async def test_contradiction_is_unsupported() -> None:
    # Scorer says nothing is entailed -> every sentence unsupported, score 0.
    async def _never(premise: str, hypothesis: str) -> float:
        return 0.0

    checker = NLIFaithfulnessChecker(_never)
    spans = [_span("The office closes at 6 PM.")]
    verdict = await checker.check("The office never closes.", spans)
    assert verdict.faithful is False
    assert verdict.score == 0.0
    assert verdict.unsupported_claims == ["The office never closes."]


async def test_support_threshold_is_respected() -> None:
    async def _mid(premise: str, hypothesis: str) -> float:
        return 0.55

    # anchor_threshold=0 isolates the support threshold: these fixtures share no vocabulary with
    # the span, so the lexical anchor would (correctly) reject them and mask what is under test.
    strict = NLIFaithfulnessChecker(_mid, support_threshold=0.6, anchor_threshold=0.0)
    lenient = NLIFaithfulnessChecker(_mid, support_threshold=0.5, anchor_threshold=0.0)
    spans = [_span("source text")]
    assert (await strict.check("a claim.", spans)).faithful is False
    assert (await lenient.check("a claim.", spans)).faithful is True


async def test_empty_answer_or_no_spans_is_vacuously_faithful() -> None:
    checker = NLIFaithfulnessChecker(_entail_by_keyword("x"))
    assert (await checker.check("", [_span("something")])).score == 1.0
    assert (await checker.check("An answer.", [])).score == 1.0


@pytest.mark.parametrize(
    "payload,expected",
    [
        ([{"label": "entailment", "score": 0.9}, {"label": "neutral", "score": 0.1}], 0.9),
        ([[{"label": "ENTAILMENT", "score": 0.7}]], 0.7),  # batch-of-one nesting + uppercase
        # Contradiction is now the sharp signal: support = 1 - P(contradiction) (see _support_prob).
        ([{"label": "contradiction", "score": 0.8}], pytest.approx(0.2)),
        ([{"label": "entail", "score": 1.5}], 1.0),  # clamped to [0,1]
    ],
)
def test_support_prob_parsing(payload: object, expected: float) -> None:
    assert _support_prob(payload) == expected


def test_support_prob_bad_shape_raises() -> None:
    from rag_common import ConfigError

    with pytest.raises(ConfigError):
        _support_prob("not a list")


async def test_true_but_unrelated_sentence_is_not_supported() -> None:
    """A sentence recalled from the model's own memory must not pass as grounded.

    Observed live: asking an RFP knowledge base "prime minister of india" returned "The current
    prime minister of India is Narendra Modi", cited a passage about a constitutional audit
    authority that never mentions him, and scored 1.00 GROUNDED. NLI alone cannot catch this —
    the sentence contradicts nothing, it is merely unrelated, so P(contradiction) is ~0 and
    "not contradicted" rubber-stamps it. The lexical anchor is the test that does catch it.
    """

    async def _never_contradicted(premise: str, hypothesis: str) -> float:
        return 1.0  # exactly what NLI reports for a true-but-unrelated sentence

    checker = NLIFaithfulnessChecker(_never_contradicted)
    spans = [_span("The Comptroller and Auditor General of India is a constitutional authority.")]
    verdict = await checker.check("The current prime minister of India is Narendra Modi.", spans)
    assert verdict.faithful is False, "an unrelated sentence must not be reported as faithful"
    assert verdict.score == 0.0
    assert verdict.unsupported_claims == ["The current prime minister of India is Narendra Modi."]


async def test_score_is_invariant_to_prose_vs_bullets() -> None:
    """The same facts must score the same whether written as prose or as a list.

    Before weighting, formatting moved the number more than truth did: identical correct content
    scored 1.00 as one prose sentence and 0.25 as four bullets, because each bullet became an
    independent pass/fail against a tiny denominator. A quality metric that punishes structure is
    measuring the wrong thing.
    """
    checker = NLIFaithfulnessChecker(_entail_by_keyword("uptime"))
    spans = [_span("Platform uptime is 99.5 percent at baseline and 99.9 percent at peak.")]
    prose = "Platform uptime is 99.5 percent at baseline and 99.9 percent at peak."
    bullets = "Platform uptime is 99.5 percent at baseline.\n- Uptime is 99.9 percent at peak."
    assert (await checker.check(prose, spans)).score == (await checker.check(bullets, spans)).score


def test_the_premise_is_bounded_by_characters_not_span_count() -> None:
    """`keep=3` bounded the NUMBER of spans, and the docstring claimed that kept the premise
    "inside the cross-encoder's window". It did while chunks were small; once the chunker capped
    at 700 characters, three spans became ~2 000 characters and ran past mDeBERTa's 512-token
    window.

    Measured live on the HR corpus, one answer sentence quoted VERBATIM from its source:
    premise 400ch -> 0.969, 700ch -> 0.950, 1400ch -> 0.981, 1983ch -> 0.279. Past the window
    the model returns CONTRADICTION for a sentence it can no longer see, so a correct, cited
    answer is scored unfaithful and the console badges it "verify"."""
    from rag_clients.nli_faithfulness import _candidate_premises

    spans = [_span("x" * 800), _span("y" * 800), _span("z" * 800)]
    premises = _candidate_premises("some sentence about x", spans, max_chars=1000)
    # Each candidate is now judged on its OWN, so the bound that matters is per candidate. The
    # join this once asserted is exactly what dropped a supporting span (see _candidate_premises).
    assert premises and all(len(p) <= 1000 for p in premises), [len(p) for p in premises]


def test_the_best_span_survives_even_when_it_alone_exceeds_the_budget() -> None:
    """A single long chunk must still be used — truncated — rather than dropped, or a sentence
    supported only by that chunk would be judged against nothing."""
    from rag_clients.nli_faithfulness import _candidate_premises

    premises = _candidate_premises("night shift allowance", [_span("night shift allowance " * 200)], max_chars=500)
    assert premises and 0 < len(premises[0]) <= 500


# --- the compound-query relevance gate ------------------------------------------------------
#
# The gate exists to stop a compound question's UNANSWERED half being rubber-stamped: NLI only
# asks "does this conflict with the sources?", and an invented figure for a topic the KB has no
# coverage of conflicts with nothing. It shipped with no tests, and measured live it was also
# rejecting compound questions the corpus DOES answer. Both directions are pinned here.


def _doc_span(doc_id: str, text: str) -> RetrievedSpan:
    return RetrievedSpan(chunk_id=doc_id + "::1", doc_id=doc_id, text=text)


async def test_a_compound_querys_unanswered_half_is_still_gated() -> None:
    """The adversarial case the gate was built for: the KB answers the leave half and has NOTHING
    on the budget half, but an unrelated budget document shares enough surface vocabulary to be
    retrieved and cited. Nothing in the sources CONTRADICTS the invented figure, so entailment
    alone passes it — the sentence must be rejected for drawing on a span that answers neither
    sub-question."""

    async def _always_entailed(premise: str, hypothesis: str) -> float:
        return 0.95

    checker = NLIFaithfulnessChecker(_always_entailed)
    spans = [
        _doc_span("leave.pdf", "Employees are entitled to 24 days of paid annual leave each year."),
        _doc_span("ganga.pdf", "The Project Ganga budget for FY2027 is 4.2 crore, approved by the steering group."),
    ]
    verdict = await checker.check(
        "Employees get 24 days of paid annual leave. The FY2027 Platform Engineering budget is 4.2 crore.",
        spans,
        "What is the annual leave policy, and what is the FY2027 Platform Engineering budget?",
    )
    assert verdict.faithful is False
    assert verdict.unsupported_claims == ["The FY2027 Platform Engineering budget is 4.2 crore."]


async def test_an_inflected_form_still_counts_as_the_clauses_own_term() -> None:
    """Measured live: "Who approves a business travel expense claim, and who approves a facility
    staff child education reimbursement?" ABSTAINED on a corpus that answers both halves. The
    travel policy writes "approved"/"expenses" where the clause says "approves"/"expense", so
    exact matching scored a chunk that plainly answers the sub-question at 0.40 (2 of 5 terms)
    and threw the answer away. A word and its inflection are the same word."""

    async def _always_entailed(premise: str, hypothesis: str) -> float:
        return 0.95

    checker = NLIFaithfulnessChecker(_always_entailed)
    spans = [
        _doc_span(
            "travel.pdf",
            "Business travel expenses are approved by the Business Unit Head and the Corporate "
            "Administration Department before reimbursement is settled.",
        ),
        _doc_span(
            "education.pdf",
            "School fees for the children of facility staff are reimbursed on production of a receipt.",
        ),
    ]
    verdict = await checker.check(
        "Business travel expenses are approved by the Business Unit Head.",
        spans,
        "Who approves a business travel expense claim, and who approves a facility staff child "
        "education reimbursement?",
    )
    assert verdict.unsupported_claims == []
    assert verdict.faithful is True


async def test_a_sub_question_the_corpus_answers_elsewhere_is_not_gated() -> None:
    """Measured live, the last false abstain: the chunk that actually answers "who approves a
    facility staff child education reimbursement" is the MIDDLE of that policy — "Verification and
    Approval: the admin department..." — and does not repeat the words "child", "education" or
    "facility", because the heading carrying them is a different chunk. Judged alone it covers
    3 of 6 clause terms and was thrown away, even though the policy's own heading chunk was
    retrieved alongside it and covers the sub-question completely.

    The gate's question is whether the CORPUS answers this half of the compound query — that is
    the failure it was built for — not whether one chunk happens to restate the question's nouns."""

    async def _always_entailed(premise: str, hypothesis: str) -> float:
        return 0.95

    checker = NLIFaithfulnessChecker(_always_entailed)
    spans = [
        _doc_span(
            "handbook.pdf",
            "Facility Staff Child Education Reimbursement Policy. Objective: to offer financial "
            "assistance for the education of the children of our facility staff members.",
        ),
        _doc_span(
            "handbook.pdf",
            "Reimbursement Process: staff submit a reimbursement application form with fee "
            "receipts to the admin department. Verification and Approval: the admin department "
            "reviews the application and forwards it to the appropriate authority.",
        ),
        _doc_span("travel.pdf", "Business travel expenses are approved by the Business Unit Head."),
    ]
    verdict = await checker.check(
        "Reimbursement applications are reviewed by the admin department and forwarded to the "
        "appropriate authority for approval.",
        spans,
        "Who approves a business travel expense claim, and who approves a facility staff child "
        "education reimbursement?",
    )
    assert verdict.unsupported_claims == []


# --- the weighted anchor (rag_core.anchoring), pinned by live measurements -------------


async def test_an_invention_built_from_the_corpus_common_words_is_caught() -> None:
    """NLI passes anything that does not contradict the sources, so the anchor is the only check
    between this sentence and the reader. Retrieval for a night-shift question returns passages
    that ALL say "Acme employees ... night shift" (neighbour expansion adds more of them), so
    half of this sentence's words are in every passage: unweighted that is 0.50 and it passed.
    Weighted, the words no passage contains dominate and it fails."""

    async def _never_contradicted(premise: str, hypothesis: str) -> float:
        return 0.95

    topics = [
        "are paid a night shift allowance of Rs.200 per day",
        "must report attendance through the shift register",
        "are eligible after completing their probation",
        "working past midnight are provided a cab home",
        "are rostered by their department head each month",
        "record overtime separately from the allowance",
        "may swap a rostered slot with prior approval",
        "follow the same leave rules as day staff",
    ]
    spans = [_doc_span("night.pdf", f"Acme employees on the night shift {t}.") for t in topics]
    verdict = await NLIFaithfulnessChecker(_never_contradicted, anchor_threshold=0.25).check(
        "Acme employees on the night shift receive a quarterly performance bonus.", spans
    )
    assert verdict.faithful is False


async def test_a_hindi_sentence_carrying_a_number_is_not_failed_for_its_digits() -> None:
    """Measured on a live Hindi answer: the ASCII-only anchor saw nothing in a Hindi sentence but
    the digits of "₹1,500", found them in no English passage, and scored a correct sentence 0.00.
    A Hindi sentence shares no words with English sources by construction; that is for the
    multilingual NLI check to judge, not a reason to fail it."""

    async def _entailed(premise: str, hypothesis: str) -> float:
        return 0.95

    spans = [
        # As live: the English passage states the perk without the figure in the same form, so the
        # digits were the ONLY "words" the old anchor could see, and they matched nothing.
        _doc_span("torch.pdf", "Torch Bearers receive sponsored meals every month and a fuel sponsorship."),
    ]
    verdict = await NLIFaithfulnessChecker(_entailed, anchor_threshold=0.25).check(
        "कंपनी द्वारा प्रति माह ₹1,500 तक के प्रायोजित भोजन दिए जाएंगे।", spans
    )
    assert verdict.faithful is True


# --- tables ---------------------------------------------------------------------------------
# A sentence-pair NLI model needs a premise that CONTAINS the fact. _focus picks the window with
# the most claim vocabulary, and on a table that is the wrong instinct: the heading and header row
# carry every word of the question ("department", "code", "Quality Control") while the row holding
# the ANSWER carries almost none of it.
#
# Measured live on the SOP corpus, a correct and correctly-cited answer scored 0.0 and abstained:
#   span  : 698 chars, contains "| Quality Control | Q | G | O | C | QG/QO/QC |"
#   focus : 300-char window ending at "| Capsules |" -- the header, none of the answer
#   verdict: 0.000 for the full span, 1.000 for the same table trimmed under the focus threshold
_TABLE_DOC = (
    "5.7.3 Department wise Code for SOP Numbering system for Manufacturing and Quality Control: "
    "| Department | Department Code | General | Operational & Calibration | Cleaning | Combined |\n"
    "| --- | --- | --- | --- | --- | --- |\n"
    "| Beta Lactam | | | | | |\n"
    "| Mfg. & Packing | B | G | O | C | BG/BO/BC |\n"
    "| Non Beta Lactam | | | | | |\n"
    "| Capsules | C | G | O | C | CG/CO/CC |\n"
    "| Liquid Orals | L | G | O | C | LG/LO/LC |\n"
    "| Packing | P | G | O | C | PG/PO/PC |\n"
    "| Quality Control | Q | G | O | C | QG/QO/QC |\n"
    "| Tablets | T | G | O | C | TG/TO/TC |\n"
    "| Engineering | E | G | O | C | EG/EO/EC |\n"
)


def test_the_premise_for_a_table_keeps_the_header_with_the_matching_row() -> None:
    """Either half alone is useless: the row without the header is "| Q |", and the header
    without the row does not contain the answer at all."""
    want = _terms("The department code for Quality Control is Q.")
    premise = _focus(_TABLE_DOC, want)
    assert "| Quality Control | Q |" in premise, "the answering row was cut out of the premise"
    assert "| Department | Department Code |" in premise, "the row lost the header that names its columns"


def test_a_different_row_selects_its_own_evidence() -> None:
    want = _terms("The department code for Engineering is E.")
    premise = _focus(_TABLE_DOC, want)
    assert "| Engineering | E |" in premise
    assert "| Department | Department Code |" in premise


def test_the_table_premise_stays_bounded() -> None:
    """A wide table must not hand the model the whole document back."""
    want = _terms("The department code for Quality Control is Q.")
    assert len(_focus(_TABLE_DOC, want)) <= len(_TABLE_DOC)


def test_prose_is_still_focused_by_window_not_by_rows() -> None:
    """The table path must not change how ordinary prose is judged."""
    prose = (
        "The retention period for audit logs is seven years. " * 6
        + "Deviations must be recorded within 24 hours of detection. "
        + "The escalation matrix has four tiers. " * 6
    )
    want = _terms("Deviations must be recorded within 24 hours.")
    premise = _focus(prose, want)
    assert "24 hours" in premise
    assert len(premise) < len(prose)


def test_a_short_table_is_unchanged() -> None:
    small = "| Dept | Code |\n| --- | --- |\n| Quality Control | Q |"
    assert _focus(small, _terms("The department code for Quality Control is Q.")) == small


async def test_a_long_answer_never_exceeds_the_concurrency_bound() -> None:
    """The real failure this guards: a map-reduce answer of hundreds of claims used to open a
    request per claim at once, overflow httpx's connection pool, and send the WHOLE answer down
    the fail-soft path to the lexical checker — reported only as a warning log, so FR-E4 looked
    like it was running NLI when it was not. The bound is what makes a long answer measurable."""
    import asyncio

    peak = 0
    live = 0

    async def _counting(premise: str, hypothesis: str) -> float:
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)  # yield, so every started call really is concurrent
        live -= 1
        return 0.95

    answer = " ".join(f"Clause {i} requires an annual review." for i in range(300))
    checker = NLIFaithfulnessChecker(_counting, anchor_threshold=0.0, max_concurrency=8)
    verdict = await checker.check(answer, [_span("Every clause requires an annual review.")])

    assert verdict.score == pytest.approx(1.0)
    assert peak <= 8, f"{peak} calls in flight against a bound of 8"


async def test_an_answer_shorter_than_the_bound_is_unaffected() -> None:
    """The bound must not change the short-answer case it was never about."""
    checker = NLIFaithfulnessChecker(_entail_by_keyword("leave"), max_concurrency=8)
    verdict = await checker.check("Annual leave is 24 days.", [_span("Annual leave is 24 days per year.")])
    assert verdict.faithful and verdict.score == pytest.approx(1.0)


async def test_concurrent_checks_share_one_bound_at_the_scorer() -> None:
    """The bound the per-call semaphore missed.

    A limit applied inside `check()` is per CALLER, so N concurrent checks hold N x the limit.
    Measured against a live run: four faithfulness checks ran at once and together held 65
    sockets, exhausting the pool exactly as the unbounded version had, and every PoolTimeout
    silently downgraded that answer to the lexical checker. The scorer owns the pool and is a
    singleton, so the bound has to live there to mean anything."""
    import asyncio

    from rag_clients.nli_faithfulness import TEINLIScorer

    scorer = TEINLIScorer(url="http://localhost:1")
    scorer._gate = asyncio.Semaphore(4)

    peak = live = 0

    async def _predict(premise: str, hypothesis: str) -> float:
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return 0.95

    scorer._predict = _predict  # type: ignore[method-assign]

    checker = NLIFaithfulnessChecker(scorer, anchor_threshold=0.0, max_concurrency=16)
    answer = " ".join(f"Clause {i} requires an annual review." for i in range(60))
    spans = [_span("Every clause requires an annual review.")]

    # Four concurrent checks, each allowed 16 of its own — only the scorer's 4 may actually fly.
    await asyncio.gather(*(checker.check(answer, spans) for _ in range(4)))

    assert peak <= 4, f"{peak} NLI calls in flight against a scorer bound of 4"
