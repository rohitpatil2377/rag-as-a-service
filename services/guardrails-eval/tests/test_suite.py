"""The RAG v2 gold-set format and its metrics (Phase 11).

These are the metrics a release gate will be read off, so what they are tested for is that each
one is TRUE WHEN THE PLATFORM IS BROKEN, not just when it is healthy. ``evaluation.py``'s own
docstring records two that were not — ``citation_coverage`` scored 1.0 for a build that abstained
on everything, and ``context_hit_rate`` was computed and then ignored by the gate — and the same
class of defect is what this file is written against:

* a metric the suite did not ask for must be ``None``, never ``0.0``, or a half-written gold set
  reads as a broken build;
* a metric that is ``None`` must never fail a floor, or the gate fires on the suite's own
  incompleteness;
* a malformed gold file must RAISE, not be skipped, because a dropped gold query makes the suite
  smaller and its score better.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from svc_guardrails_eval.suite import (
    INTENTS,
    SUITE_VERSION,
    ChecklistItem,
    GoldQuery,
    GoldSuite,
    SuiteFloors,
    SuiteFormatError,
    SuiteReport,
    Supersession,
    contains_cue,
    extract_values,
    fold_text,
    load_suite,
    parse_suite,
    score_query,
    score_suite,
    strip_emphasis,
)

from rag_schemas import Citation, GroundedAnswer
from rag_schemas.common import AnswerStatus

_REPO = Path(__file__).resolve().parents[3]


def _answer(
    text: str = "The approver is Asst. Manager.",
    *,
    status: AnswerStatus = AnswerStatus.GROUNDED,
    doc_ids: tuple[str, ...] = ("sop.pdf",),
    faithfulness: float | None = 0.91,
) -> GroundedAnswer:
    citations = [Citation(chunk_id=f"c{i}", doc_id=doc_id, quote="q", score=0.9) for i, doc_id in enumerate(doc_ids)]
    return GroundedAnswer(
        tenant_id="t",
        query="who approves an SOP?",
        answer=text,
        status=status,
        citations=citations if status is not AnswerStatus.ABSTAINED else [],
        confidence=0.8,
        faithfulness_score=faithfulness,
    )


def _abstained(text: str = "I could not find that in the documents.") -> GroundedAnswer:
    return _answer(text, status=AnswerStatus.ABSTAINED, doc_ids=(), faithfulness=None)


# --- the format ------------------------------------------------------------------------------------


def test_the_shipped_sop_suite_loads() -> None:
    """The one gold suite this repository can honestly carry — see the file's own header for where
    its items come from and why there is only one document class."""
    suite = load_suite(_REPO / "eval" / "gold" / "sop.yaml")

    assert suite.doc_type == "sop" and suite.suite_version == SUITE_VERSION
    assert len(suite.queries) >= 5
    assert {q.intent for q in suite.queries} <= INTENTS
    assert any(q.expect_abstain for q in suite.queries), "the refusal half is not padding"
    assert any(q.checklist for q in suite.queries), "the aggregate half has something to score"


def test_a_minimal_lookup_query_parses() -> None:
    suite = parse_suite({"doc_type": "sop", "queries": [{"query": "q?", "expected_answer": "x"}]}, name="t")
    assert suite.queries[0].intent == "lookup"
    assert suite.queries[0].expected_answer == "x"


def test_a_checklist_defaults_to_the_must_tier() -> None:
    suite = parse_suite(
        {
            "doc_type": "sop",
            "queries": [{"query": "q?", "intent": "aggregate", "checklist": [{"must_contain": ["a"]}]}],
        },
        name="t",
    )
    item = suite.queries[0].checklist[0]
    assert item.tier == "must" and item.id == "a"


@pytest.mark.parametrize(
    ("data", "because"),
    [
        ({"queries": [{"query": "q"}]}, "doc_type is what corpus diversity counts"),
        ({"doc_type": "sop"}, "a suite with no queries gates nothing"),
        ({"doc_type": "sop", "queries": []}, "same"),
        ({"doc_type": "sop", "queries": [{"query": ""}]}, "a query needs text"),
        ({"doc_type": "sop", "queries": [{"query": "q", "intent": "lookup2"}]}, "unknown intent"),
        (
            {"doc_type": "sop", "queries": [{"query": "q", "checklist": [{"must_contain": []}]}]},
            "a checklist entry with nothing to contain asserts nothing",
        ),
        (
            {"doc_type": "sop", "queries": [{"query": "q", "checklist": [{"must_contain": ["a"], "tier": "nice"}]}]},
            "unknown tier",
        ),
        (
            {
                "doc_type": "sop",
                "queries": [{"query": "q", "expected_answer": "x", "checklist": [{"must_contain": ["a"]}]}],
            },
            "expected_answer OR checklist, not both",
        ),
        ({"doc_type": "sop", "queries": [{"query": "q", "supersessions": [{"old": "a"}]}]}, "needs both old and new"),
        ({"suite_version": 2, "doc_type": "sop", "queries": [{"query": "q"}]}, "a future contract version"),
        ("not a mapping", "the file must be a mapping"),
    ],
)
def test_a_malformed_gold_file_raises_rather_than_being_skipped(data: object, because: str) -> None:
    """A dropped gold query makes the suite smaller and its score better — the one failure mode a
    regression gate must never have. Hence strict parsing here, unlike the tolerant parsing this
    codebase uses on the ingest and query paths, where the alternative is taking a tenant down."""
    with pytest.raises(SuiteFormatError):
        parse_suite(data, name="t")
    assert because  # the reason is the point of the case


def test_folding_matches_the_gate_scripts_own_normalisation() -> None:
    """Shared so the script and the metrics cannot drift: an extraction that lost a subscript
    should not read as a wrong answer."""
    assert fold_text("NO₃⁻") == fold_text("NO3-")
    assert fold_text("  Asst.   MANAGER ") == fold_text("asst. manager")


# --- lookup scoring ---------------------------------------------------------------------------------


def test_a_lookup_hit_needs_the_fact_and_a_stood_behind_answer() -> None:
    gold = GoldQuery(query="q", expected_answer="Asst. Manager")
    assert score_query(gold, _answer("The approver is Asst. Manager.")).lookup_hit is True
    assert score_query(gold, _answer("The approver is the Executive.")).lookup_hit is False


def test_an_abstention_quoting_the_expected_fact_is_not_a_hit() -> None:
    """Otherwise a refusal that happens to echo the question's wording scores as a correct answer."""
    gold = GoldQuery(query="q", expected_answer="Asst. Manager")
    assert score_query(gold, _abstained("I could not find who the Asst. Manager is.")).lookup_hit is False


def test_abstain_correctness_is_scored_in_both_directions() -> None:
    wanted = GoldQuery(query="q", expect_abstain=True)
    assert score_query(wanted, _abstained()).abstain_ok is True
    assert score_query(wanted, _answer()).abstain_ok is False
    answerable = GoldQuery(query="q", expected_answer="x")
    assert score_query(answerable, _abstained()).abstain_ok is False


def test_context_precision_is_not_scored_on_an_abstain_item() -> None:
    """An item that SHOULD abstain cites nothing by definition, so counting it as a context miss
    would penalise correct behaviour — the same rule the v1 gate already follows."""
    assert score_query(GoldQuery(query="q", expect_abstain=True), _abstained()).context_hit is None


def test_context_precision_needs_an_expected_document() -> None:
    gold = GoldQuery(query="q", expected_answer="x", expected_doc_ids=("sop.pdf",))
    assert score_query(gold, _answer(doc_ids=("sop.pdf",))).context_hit is True
    assert score_query(gold, _answer(doc_ids=("other.pdf",))).context_hit is False


# --- aggregate scoring --------------------------------------------------------------------------------

_CHECKLIST = (
    ChecklistItem(id="prepares", facet="responsibilities", must_contain=("Executive",), tier="must"),
    ChecklistItem(id="approves", facet="responsibilities", must_contain=("Asst. Manager",), tier="must"),
    ChecklistItem(id="extra", facet="responsibilities", must_contain=("Head of Department",), tier="should"),
)


def test_item_recall_is_reported_per_tier() -> None:
    """Averaging the tiers would let a build trade a required item for two optional ones and call
    it progress, which is why the plan asks for them separately."""
    gold = GoldQuery(query="q", intent="aggregate", checklist=_CHECKLIST)
    score = score_query(gold, _answer("The Executive prepares it and the Head of Department is accountable."))

    assert score.item_recall_must == 0.5
    assert score.item_recall_should == 1.0
    assert score.missed_items == ("approves",)


def test_an_item_needs_every_one_of_its_must_contain_entries() -> None:
    """All, not any: an entry is there because the item is not adequately answered without it."""
    gold = GoldQuery(
        query="q",
        intent="aggregate",
        checklist=(ChecklistItem(id="both", facet="f", must_contain=("HBOM", "SBOM")),),
    )
    assert score_query(gold, _answer("Supply the SBOM.")).item_recall_must == 0.0
    assert score_query(gold, _answer("Supply the HBOM and the SBOM.")).item_recall_must == 1.0


def test_value_accuracy_is_separate_from_item_recall() -> None:
    """Naming the facet with the wrong figure and omitting the facet entirely are different bugs.

    This used to make the point with "Rs. 50,000" against "Rs. 50000" and assert that RECALL
    failed on the thousands separator while value accuracy passed. `fold_text` now folds that
    separator, deliberately: traced through the real pipeline, a correct answer saying
    "sustained 6,000 EPS" was scored a miss against a gold of `6000`, and lookup accuracy read
    0.714 when it was 1.000. So the distinction is shown here with wording the answer genuinely
    does not contain, which is what recall is actually for."""
    gold = GoldQuery(
        query="q",
        intent="aggregate",
        checklist=(ChecklistItem(id="fee", facet="fees", must_contain=("Rs. 50,000 per annum",)),),
    )
    wrong = score_query(gold, _answer("The fee is Rs. 60,000 per annum."))
    right = score_query(gold, _answer("The fee is Rs. 50000."))

    assert wrong.value_accuracy == 0.0
    assert right.value_accuracy == 1.0, "a figure written differently is the same figure"
    assert right.item_recall_must == 0.0, "...and recall still needs the wording the item names"


def test_a_checklist_entry_with_no_value_in_it_does_not_affect_value_accuracy() -> None:
    gold = GoldQuery(
        query="q", intent="aggregate", checklist=(ChecklistItem(id="role", facet="f", must_contain=("Executive",)),)
    )
    assert score_query(gold, _answer("The Executive prepares it.")).value_accuracy is None


# --- supersessions ------------------------------------------------------------------------------------


def _maat() -> GoldQuery:
    return GoldQuery(
        query="q",
        supersessions=(Supersession(field="maat", old="Rs. 35 Crores", new="Rs. 19.44 Crores"),),
    )


def test_quoting_the_replaced_figure_alone_still_fails() -> None:
    """The original purpose, unchanged: the only metric that can catch an answer which is
    grounded, cited, faithful and quoting the figure that was replaced."""
    assert score_query(_maat(), _answer("The requirement is Rs. 35 Crores.")).supersession_ok is False


def test_stating_only_the_current_figure_is_an_unmarked_change_not_a_pass() -> None:
    """A DELIBERATE TIGHTENING, and it reverses what this file previously asserted.

    The old contract was "states new, and does not state old unmarked", so an answer giving only
    `Rs. 19.44 Crores` passed. Two things argue against that. The scorer's own docstring already
    did -- "a compliance reader needs to know a figure CHANGED, not merely what it is now; an
    answer that silently states ₹19.44 Cr gives a bidder no way to tell that their ₹35 Cr
    assumption is stale". And the looser rule could not be made correct: because freshness was a
    GLOBAL search for the new value, the real corpus's `D.3` entry
    ("Individual Confidentiality Declaration -> withdrawn") passed on the word "withdrawn"
    appearing anywhere in a 47,000-character answer -- it was about Appendix D's Vendor
    Undertaking Formats, and the declaration itself was never mentioned. Tying the value to the
    gold's `field` would be the obvious fix and does not work: the fields are clause identifiers
    and only `MAAT` appears in any answer.

    So the pair is the assertion. An answer stating only the current figure is not stale -- it
    asserts nothing superseded -- and is not a correct report either, so it is named rather than
    silently passed or counted as staleness."""
    score = score_query(_maat(), _answer("The requirement is Rs. 19.44 Crores."))

    assert score.supersession_ok is False
    assert score.stale_values == (), "nothing stale was asserted; the old figure is simply absent"
    assert score.unmarked_changes == ("maat",)


def test_the_wording_of_a_figure_does_not_decide_whether_it_was_reported() -> None:
    """Measured across all nine stored runs: the gold says `Rs. 19.44 Crores` and the answers say
    `₹19.44 crore` or `₹**19.44** crores`. `fold_text` folds case and separators but not the
    currency symbol and not the plural, so a correctly reported amendment scored as unreported on
    9 runs out of 9.

    NO NEW EQUIVALENCE IS INVENTED FOR THIS. `extract_values` already canonicalises all of those
    to the same `qty:19.44 crore` key -- the inconsistency was between that vocabulary and the
    plain substring search this metric used."""
    for wording in (
        "was Rs. 35 Crores, now Rs. 19.44 Crores",
        "was ₹35 crore, now ₹19.44 crore",
        "was ₹**35** crores and is now ₹**19.44** crores",
        "changed from Rs 35 crores to 19.44 crore",
    ):
        score = score_query(_maat(), _answer(f"The MAAT {wording}."))
        assert score.supersession_ok is True, f"not recognised: {wording!r}"

    # And a DIFFERENT figure is still a different figure.
    assert score_query(_maat(), _answer("The MAAT was Rs. 35 Crores, now Rs. 19.45 Crores.")).supersession_ok is False


def test_showing_old_then_new_together_is_correct_not_stale() -> None:
    """CORRECTED. The rule used to be "states new and does not state old AT ALL", and it
    contradicted the plan's own §9.2 reduce instruction — rule 3 there is "for amendments, show
    'old -> new' explicitly". When Phase 9 was finally scored against this metric, every run
    failed on an answer that had done exactly what §9.2 asked.

    §9.2 is the one worth keeping: a compliance reader needs to know a figure CHANGED, not just
    what it is now. An answer that silently states ₹19.44 Cr gives a bidder no way to tell their
    ₹35 Cr assumption is stale."""
    for rendering in (
        "MAAT: Rs. 35 Crores -> Rs. 19.44 Crores",
        "MAAT: Rs. 35 Crores $\\rightarrow$ Rs. 19.44 Crores",
        "It was Rs. 35 Crores and is now Rs. 19.44 Crores.",
        "The turnover floor, previously Rs. 35 Crores, is amended to Rs. 19.44 Crores.",
    ):
        score = score_query(_maat(), _answer(rendering))
        assert score.supersession_ok is True, rendering
        assert score.stale_values == ()


def test_the_old_value_far_from_its_replacement_is_still_stale() -> None:
    """The failure this metric exists for, which the correction above must not let through: both
    figures appear, but nothing ties them together, so a reader cannot tell which is in force."""
    far = "The requirement is Rs. 35 Crores. " + ("Unrelated prose about switches. " * 40) + "Rs. 19.44 Crores."
    score = score_query(_maat(), _answer(far))

    assert score.supersession_ok is False
    assert score.stale_values == ("Rs. 35 Crores",)


def test_a_real_amendment_clause_restates_both_sides_and_still_passes() -> None:
    """MEASURED on the real AEGCL corrigendum: its DRS amendment puts about 200 characters
    between "letterhead of the respective OEM" and "GTP/Data Sheet", because the clause restates
    the whole requirement on each side of the change. A tight proximity bound scored that correct
    answer as stale."""
    gold = GoldQuery(
        query="q",
        supersessions=(Supersession(field="bds", old="letterhead of the respective OEM", new="GTP/Data Sheet"),),
    )
    answer = (
        "The Bidder's compliance to the Data Requirement Sheet shall be submitted on the "
        "letterhead of the respective OEM, duly signed by an authorized signatory of that OEM, "
        "is amended as follows: the bidder shall offer their solution against the requirements "
        "of the DRS (Volume-II, Part B, Appendix D) and shall also provide the GTP/Data Sheet "
        "offered by the respective OEM(s) against each requirement."
    )

    assert score_query(gold, _answer(answer)).supersession_ok is True


# --- the suite report and the gate ---------------------------------------------------------------------


def _suite(*queries: GoldQuery) -> GoldSuite:
    return GoldSuite(name="t", doc_type="sop", queries=queries)


def test_a_metric_the_suite_never_asked_for_is_none_not_zero() -> None:
    """A half-written gold set must not read as a broken build."""
    suite = _suite(GoldQuery(query="q", expected_answer="Asst. Manager"))
    report = score_suite(suite, [score_query(suite.queries[0], _answer())])

    assert report.lookup_accuracy == 1.0
    assert report.item_recall_must is None
    assert report.supersession_correctness is None
    assert report.value_accuracy is None


def test_a_none_metric_never_fails_a_floor() -> None:
    """Otherwise the gate fires on the suite's own incompleteness rather than on the build — the
    SOP suite has no supersession items because the document is not an amendment."""
    report = SuiteReport(suite="t", doc_type="sop", n=1, scored=1, lookup_accuracy=1.0)
    floors = SuiteFloors(min_supersession_correctness=1.0, min_item_recall_must=1.0)

    assert report.failures(floors) == ()
    assert report.passes(floors)


def test_a_breached_floor_is_named_with_both_numbers() -> None:
    report = SuiteReport(suite="t", doc_type="sop", n=2, scored=2, lookup_accuracy=0.5)
    failures = report.failures(SuiteFloors(min_lookup_accuracy=0.9))

    assert len(failures) == 1
    assert "lookup_accuracy" in failures[0] and "0.500" in failures[0] and "0.900" in failures[0]


def test_no_floor_set_means_the_gate_passes() -> None:
    """Defaulting to ungated is the difference between a gate and a guess: the plan's §0 targets
    are owner-confirmable numbers, and an invented floor teaches a team to edit the threshold."""
    report = SuiteReport(suite="t", doc_type="sop", n=1, scored=1, lookup_accuracy=0.0)
    assert report.passes(SuiteFloors())


def test_the_report_counts_intents_so_a_single_intent_suite_is_visible() -> None:
    suite = _suite(
        GoldQuery(query="a", intent="lookup", expected_answer="x"),
        GoldQuery(query="b", intent="aggregate", checklist=_CHECKLIST),
    )
    scores = [score_query(q, _answer()) for q in suite.queries]
    assert score_suite(suite, scores).intent_counts == {"lookup": 1, "aggregate": 1}


def test_latency_p95_is_per_stage_and_nearest_rank() -> None:
    """Per stage because a suite that is slow tells you nothing and a suite that is slow IN
    RETRIEVAL tells you where to look. Nearest-rank because interpolating between two samples at
    suite size invents a number no question actually took."""
    suite = _suite(*(GoldQuery(query=f"q{i}", expected_answer="x") for i in range(4)))
    scores = [
        score_query(q, _answer(), latency_ms={"retrieve": ms, "generate": ms * 10})
        for q, ms in zip(suite.queries, (10, 20, 30, 40), strict=True)
    ]
    report = score_suite(suite, scores)

    assert report.latency_p95 == {"retrieve": 40, "generate": 400}


def test_a_scores_length_mismatch_is_refused() -> None:
    suite = _suite(GoldQuery(query="a", expected_answer="x"))
    with pytest.raises(ValueError, match="length mismatch"):
        score_suite(suite, [])


# --- the intent metric now has a producer (Phase 7) ----------------------------------------------


def test_the_router_labels_the_shipped_suites_intents() -> None:
    """Phase 11 shipped `intent` as a fixture waiting for its consumer — discovery §8.23 recorded
    that nothing measured whether a router AGREED, because no Phase-7 router existed and the old
    classifier labelled a different taxonomy. It exists now, so the loop closes here: the gold
    labels are checked against the real router's own classifier.

    Asserted at 100% because the suite is nine queries, all of them shapes the cue tables are
    built for. It is a REGRESSION guard, not an accuracy estimate — the estimate lives in
    `libs/rag-core/tests/test_router.py`, over forty queries, with its own caveat about having
    co-evolved with the classifier."""
    from rag_core.router import classify_router_intent

    suite = load_suite(_REPO / "eval" / "gold" / "sop.yaml")
    wrong = [(q.query, q.intent, classify_router_intent(q.query)) for q in suite.queries]
    wrong = [w for w in wrong if w[1] != w[2]]

    assert not wrong, f"the router disagrees with the gold labels: {wrong}"


def test_a_thousands_separator_is_not_a_different_number() -> None:
    """Traced through the real pipeline: the model answered "sustained 6,000 EPS" and the gold
    said `6000`, so a correct answer scored as a miss and lookup accuracy read 0.714 when it was
    1.000."""
    assert fold_text("6000") in fold_text("The SIEM must handle a sustained 6,000 EPS")
    assert fold_text("115000") in fold_text("about 115,000 output tokens")


def test_a_spelled_out_gloss_is_not_a_different_duration() -> None:
    """Same trace, second case: the document writes "6 (six) months" and the model answered
    "minimum 6 months". The parenthetical re-spells the number it follows; dropping it says
    nothing different."""
    assert fold_text("6 (six) months") in fold_text("a continuous period of minimum 6 months")
    assert fold_text("3 (three) years") in fold_text("within 3 years of award")
    # and the gloss is still matched when the answer DOES carry it
    assert fold_text("6 (six) months") in fold_text("for 6 (six) months from go-live")


def test_folding_numbers_does_not_merge_genuinely_different_values() -> None:
    """The guard on the two rules above: they normalise rendering, not magnitude."""
    assert fold_text("6000") not in fold_text("the system handles 600 EPS")
    assert fold_text("19.44") not in fold_text("Rs. 1944 Crores")
    # a parenthetical that is NOT a number gloss must survive, or real text is lost
    assert "(annexure a)" in fold_text("see Annexure (Annexure A) for details")


def test_markdown_emphasis_does_not_hide_a_value() -> None:
    """The answers ARE markdown and the model bolds the figures. Measured against the live API:
    a correct answer reading "minimum **6** months" scored a miss, because `**6** months` does
    not contain `6 months` however the rest is normalised."""
    assert fold_text("6 (six) months") in fold_text("for a continuous period of minimum **6** months.")
    assert fold_text("15 crores") in fold_text("shall not be less than **15 crores** (Rs 15 crore)")
    assert fold_text("UL 62368") in fold_text("certified to `UL 62368` and CSA 22.2")


def test_underscores_are_left_alone() -> None:
    """`_` is markdown emphasis too, but snake_case facet ids are real content and stripping it
    would silently merge `ng_siem` into `ngsiem` on one side of a comparison only."""
    assert fold_text("snake_case_id") in fold_text("the facet snake_case_id applies here")
    assert fold_text("ng_siem") in fold_text("items grouped under ng_siem")


def test_a_figure_does_not_match_inside_a_longer_figure() -> None:
    """The defect that survived the folding fixes: a substring test cannot tell "600" from the
    first three characters of "6000", so an answer stating the WRONG capacity scored as stating
    the right one. Boundaries are applied only at whichever end of the cue is itself a digit."""
    assert not contains_cue(fold_text("the system handles 6000 EPS"), "600")
    assert not contains_cue(fold_text("uplift to 16000 EPS"), "6000")
    assert not contains_cue(fold_text("Rs. 119.44 Crores"), "19.44")
    assert not contains_cue(fold_text("Rs. 19.447 Crores"), "19.44")
    assert not contains_cue(fold_text("SAN of 175 TB usable"), "75 TB")
    assert not contains_cue(fold_text("certified to UL 623689"), "UL 62368")
    assert not contains_cue(fold_text("diode per IEC 62443-4-25"), "IEC 62443-4-2")


def test_the_boundary_rule_still_accepts_every_real_rendering() -> None:
    """A guard that is too eager costs real evidence, which is the failure this whole area has
    been correcting. A figure at the end of a sentence, inside markdown, or carrying a folded
    thousands separator must still match."""
    assert contains_cue(fold_text("a sustained **6,000** EPS"), "6000")
    assert contains_cue(fold_text("the SIEM handles 6000. Next clause."), "6000")
    assert contains_cue(fold_text("Rs. 19.44 Crores"), "19.44")
    assert contains_cue(fold_text("SAN of 75 TB usable"), "75 TB")
    assert contains_cue(fold_text("certified to **UL 62368**"), "UL 62368")
    assert contains_cue(fold_text("diode per IEC 62443-4-2."), "IEC 62443-4-2")
    assert contains_cue(fold_text("minimum **6** months"), "6 (six) months")


def test_text_cues_are_still_plain_substring_matches() -> None:
    """Making these boundary-sensitive would change the meaning of every checklist item in the
    suite to fix a problem none of them has."""
    assert contains_cue(fold_text("the Video Wall shall support"), "Video Wall")
    assert contains_cue(fold_text("certified by PeopleCert"), "PeopleCert")
    assert contains_cue(fold_text("items grouped under ng_siem"), "ng_siem")
    assert not contains_cue(fold_text("the switches shall"), "Data Diode")


def test_an_empty_cue_never_counts_as_covered() -> None:
    assert not contains_cue(fold_text("anything at all"), "")
    assert not contains_cue(fold_text("anything at all"), "   ")


# --- one canonical representation for every evaluator (SPEC, not yet implemented) ----------------
#
# `_checklist_scores` reads the answer two ways on consecutive lines:
#
#     covered       = all(contains_cue(folded, entry) for entry in item.must_contain)   # folded
#     answer_values = extract_values(raw_answer)                                        # raw
#
# `fold_text` was given markdown-folding precisely so a bolded figure would still match -- the
# test above records that fix -- and its sibling kept reading raw text. So the fix landed at one
# of two call sites. It only became visible once an answer path started OBEYING the presentation
# contract's "put EVERY number in **bold**": measured on a real run, `₹**15** crores` scored a
# value MISS against a gold of `15 crores`, dropping value accuracy 58.8% -> 52.9% on a correct,
# better-formatted answer. Any arm that obeys the bold rule is penalised, the incumbent included.
#
# `xfail(strict=True)` rather than fixed: a benchmark is running and `prod_run` imports
# `score_query` from here, so changing the grader mid-run would score later cycles differently
# from earlier ones. The fix is `extract_values(folded)`; the stored answers are then re-scored
# uniformly across all arms, original and corrected reported side by side. Remove with the fix.


def test_a_bolded_figure_is_the_same_figure() -> None:
    """THE DEFECT, minimally. Emphasis is presentation; the value underneath is unchanged."""
    gold = GoldQuery(
        query="q",
        intent="aggregate",
        checklist=(ChecklistItem(id="maat", facet="finance", must_contain=("15 crores",)),),
    )

    plain = score_query(gold, _answer("The combined value shall not be less than 15 crores."))
    bolded = score_query(gold, _answer("The combined value shall not be less than ₹**15** crores."))

    assert plain.value_accuracy == 1.0
    assert bolded.value_accuracy == 1.0, "bold markers between the number and its unit hid the quantity"


@pytest.mark.parametrize(
    "written",
    ["1,25,49,23,246", "**1,25,49,23,246**", "₹**1,25,49,23,246**", "**1,25,49,23,246** (₹125.5 crore)"],
)
def test_bold_around_a_whole_figure_already_matches(written: str) -> None:
    """THE BOUNDARY OF THE DEFECT, and it is narrower than it first looked. `_NUMBER_RE` finds the
    digits whatever surrounds them, so emphasis wrapped around a complete figure has never been a
    problem. These pass today and are pinned so the fix cannot be mis-scoped into "bold used to
    break numbers", which it did not.

    An earlier version of this test asserted these as expected failures. `strict=True` turned the
    four passes into four reported failures, which is what caught the mistake."""
    gold = GoldQuery(
        query="q",
        intent="aggregate",
        checklist=(ChecklistItem(id="amt", facet="finance", must_contain=("1,25,49,23,246",)),),
    )

    assert score_query(gold, _answer(f"The amount is {written}.")).value_accuracy == 1.0


@pytest.mark.parametrize(
    ("gold_entry", "written"),
    [
        ("15 crores", "₹**15** crores"),
        ("15 crores", "**15** crores"),
        ("75 TB", "**75** TB"),
        ("6000 EPS", "**6,000** EPS"),
        ("19.44 crore", "₹**19.44** crore"),
    ],
)
def test_bold_between_a_figure_and_its_unit_keeps_the_quantity(gold_entry: str, written: str) -> None:
    """The defect, now fixed. `extract_values` builds a `qty:<number> <unit>` key by reading the unit
    that FOLLOWS the number, so `**` sitting between them leaves only the bare `num:`:

        '15 crores'       -> ['num:15', 'qty:15 crore']
        '₹**15** crores'  -> ['num:15']

    And the presentation contract asks for exactly that shape -- bold every number, keep the unit
    -- so the rule and the grader are in direct conflict. `contains_cue` already folds markdown;
    `extract_values` must read the same folded text."""
    gold = GoldQuery(
        query="q",
        intent="aggregate",
        checklist=(ChecklistItem(id="v", facet="f", must_contain=(gold_entry,)),),
    )

    assert score_query(gold, _answer(f"The value is {written}.")).value_accuracy == 1.0


def test_a_wrong_figure_still_fails_however_it_is_formatted() -> None:
    """The guard on the fix: normalising presentation must not weaken semantic validation. This
    passes TODAY and must keep passing — a fix that made every bolded number match would also
    make every WRONG bolded number match, which is the false negative the value check exists for."""
    gold = GoldQuery(
        query="q",
        intent="aggregate",
        checklist=(ChecklistItem(id="maat", facet="finance", must_contain=("15 crores",)),),
    )

    assert score_query(gold, _answer("The value shall not be less than ₹**19.44** crores.")).value_accuracy == 0.0


def test_a_unit_bearing_value_keeps_its_unit() -> None:
    """`75 TB` and `75` are not the same claim, and folding must not reduce one to the other."""
    gold = GoldQuery(
        query="q",
        intent="aggregate",
        checklist=(ChecklistItem(id="cap", facet="storage", must_contain=("75 TB",)),),
    )

    assert score_query(gold, _answer("Usable capacity is 75 TB.")).value_accuracy == 1.0
    assert score_query(gold, _answer("There are 75 nodes.")).value_accuracy == 0.0


@pytest.mark.parametrize("identifier", ["TPM 2.0", "IEC 61000-4-5", "UL 62368", "IEC 62443-4-2", "CSA 22.2"])
def test_a_technical_identifier_is_a_name_and_not_a_quantity(identifier: str) -> None:
    """`UL 62368` is not 62,368 of anything. An evaluator that applies numeric tolerance to an
    identifier mistakes a MISSING CERTIFICATION STANDARD for a rounding difference — which is
    exactly what the first version of the articulation report's classifier did to `IEC 61000-4-5`.
    A missing identifier must fail, with no tolerance reachable."""
    gold = GoldQuery(
        query="q",
        intent="aggregate",
        checklist=(ChecklistItem(id="cert", facet="certs", must_contain=(identifier,)),),
    )

    assert score_query(gold, _answer(f"The switch complies with {identifier}.")).value_accuracy == 1.0
    assert score_query(gold, _answer("The switch complies with some standard.")).value_accuracy == 0.0
    # And the near-miss is a miss: a digit changed in an identifier is a DIFFERENT standard.
    near = identifier[:-1] + ("9" if identifier[-1] != "9" else "8")
    assert score_query(gold, _answer(f"The switch complies with {near}.")).value_accuracy == 0.0


def test_identifier_digits_are_not_independent_numeric_evidence() -> None:
    """CROSS-CATEGORY CONTAMINATION, pinned before the fix and not after.

    `IEC 62443-4-2` is one typed thing. Decomposed into `num:62443`, `num:4`, `num:2` it becomes
    three pieces of numeric evidence the answer never offered, and a gold entry wanting the number
    4 would be satisfied by an answer that merely cited a standard. That is a false POSITIVE
    arriving through the door built to stop a false negative.

    This passes today because `extract_values` reads the raw answer and types the whole identifier.
    It is pinned because the obvious fix for the bold-quantity defect -- `extract_values(fold_text
    (answer))` -- breaks it: folding lowercases, the case-sensitive `std:` key is lost, and the
    identifier dissolves into exactly those loose numbers. Measured on a real answer, folding
    destroyed all six standards in it (`std:ieee 802.3`, `std:iec 60870-5-104`, ...) while
    inventing their digits as evidence."""
    gold = GoldQuery(
        query="q",
        intent="aggregate",
        checklist=(ChecklistItem(id="ports", facet="hw", must_contain=("4 ports",)),),
    )

    assert score_query(gold, _answer("The switch complies with IEC 62443-4-2.")).value_accuracy == 0.0
    assert score_query(gold, _answer("The switch complies with **IEC 62443-4-2**.")).value_accuracy == 0.0
    # And the real requirement still matches when it IS stated.
    assert score_query(gold, _answer("The switch provides 4 ports.")).value_accuracy == 1.0


@pytest.mark.parametrize("identifier", ["TPM 2.0", "IEC 61000-4-5", "UL 62368", "IEC 62443-4-2", "CSA 22.2"])
def test_a_bolded_identifier_keeps_its_typed_identity(identifier: str) -> None:
    """The presentation contract says to bold every entity name, so identifiers ARRIVE bolded.
    Any normalisation chosen for the quantity defect has to leave them typed.

    This is the test that rules out `fold_text` and rules out a raw/folded UNION: the union still
    manufactures the decomposed digits, where stripping ONLY markdown emphasis -- preserving case,
    separators and dashes -- recovers the quantities and never creates them. Measured on a real
    answer, emphasis-only extraction is a strict superset of raw: 112 keys against 103, 9 more
    `qty:`, all 6 `std:` intact, nothing lost."""
    gold = GoldQuery(
        query="q",
        intent="aggregate",
        checklist=(ChecklistItem(id="cert", facet="certs", must_contain=(identifier,)),),
    )

    assert score_query(gold, _answer(f"Complies with **{identifier}**.")).value_accuracy == 1.0


# --- the normalizer's nine permanent invariants ---------------------------------------------------
#
# The normalizer is asterisk-only and deliberately not a markdown parser: it removes exactly the
# syntax this application's own prompt generates (`**bold**`) and nothing else. An earlier
# delimiter-aware version that also handled `_` stripped `__init__` to `init` -- correct about
# CommonMark, wrong about a Python dunder -- so underscores are now never touched, which removes
# the dilemma at no cost, because the defect is `**` between a number and its unit.


@pytest.mark.parametrize(
    ("written", "expect"),
    [
        ("**15** crores", "15 crores"),
        ("₹**15** crores", "₹15 crores"),
        ("**75** TB", "75 TB"),
        ("*italic*", "italic"),
        ("**bold**", "bold"),
        ("**a *b* c**", "a b c"),
    ],
)
def test_only_paired_delimiters_hugging_content_are_removed(written: str, expect: str) -> None:
    """Invariants 1 and 2: a valid pair, and the delimiters must hug non-whitespace."""
    assert strip_emphasis(written) == expect


@pytest.mark.parametrize(
    "untouched",
    [
        r"\*literal\*",  # 3: escaped delimiters
        "model_v2",
        "__init__",
        "A_B",
        "snake_case_name",  # 4: underscores are never modified
        "5 * 10",
        "a * b * c",
        "2*3=6",  # 5: mathematical asterisks
        "unpaired * here",
        "**unclosed",
        "trailing **",  # 6: unpaired delimiters
        "IEC 62443-4-2",
        "_Could not be verified_",
    ],
)
def test_everything_that_is_not_a_bold_pair_survives_byte_for_byte(untouched: str) -> None:
    """Invariants 3 to 6. The naive `re.sub(r"[*_]+", "", text)` fails every case here, silently:
    only the COMPARISON would change, never the stored answer, so a corrupted evaluation
    representation leaves no trace in the artifact."""
    assert strip_emphasis(untouched) == untouched


def test_the_raw_answer_is_never_mutated() -> None:
    """Invariant 7. The normalized form is an EVALUATION representation; the raw answer stays
    authoritative for audit and for reproducing any earlier score."""
    raw = "The value is ₹**15** crores."
    before = raw
    strip_emphasis(raw)
    assert raw == before


@pytest.mark.parametrize(
    ("written", "expect_key"),
    [
        ("**IEC 62443-4-2**", "std:iec 62443-4-2"),
        ("**TPM 2.0**", "std:tpm 2.0"),
        ("ISO **27001**", "std:iso 27001"),
        ("**UL 62368**", "std:ul 62368"),
    ],
)
def test_an_identifier_keeps_or_regains_its_typed_representation(written: str, expect_key: str) -> None:
    """Invariant 8, and the one that is an improvement rather than a repair.

    `ISO **27001**` yields a loose `num:27001` today, because the bold splits the identifier from
    its prefix and the `std:` key never forms -- so a standard's digits become independent numeric
    evidence. Stripping the emphasis PROMOTES it to `std:iso 27001`. Measured on a real answer,
    that is the single key that changes identity rather than appearing: `num:27001` out,
    `std:iso 27001` in."""
    assert expect_key in extract_values(strip_emphasis(written))


def test_normalization_does_not_make_a_wrong_value_match() -> None:
    """Invariant 9, the guard on the whole exercise. Recovering presentation-equivalent quantities
    must not turn the value check into a formatting check: a wrong figure stays wrong however it
    is emphasised, and an identifier one digit out is a DIFFERENT standard."""
    got = extract_values(strip_emphasis("The value is ₹**19.44** crores and it meets **UL 62369**."))

    assert "qty:19.44 crore" in got, "the correct quantity is readable"
    assert "qty:15 crore" not in got, "a different figure must not match"
    assert "std:ul 62368" not in got, "one digit out is another standard"
