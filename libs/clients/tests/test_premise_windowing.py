"""The premise handed to the NLI judge must be the part of the source that could support the
sentence — not the whole chunk it happens to sit in.

Measured live on Acme's SOP corpus: the answer sentence "For the first issuance, 'New
introduction' shall be mentioned in the Supersedes column" is written almost verbatim in a
685-character chunk, and the judge returned CONTRADICTION 0.80 for it — because that chunk packs
four unrelated clauses together (5.8.6 Revision No., 5.8.7 Effective Date, 5.8.8 Supersedes,
5.8.9 Review Date) and a sentence-pair model given a multi-topic premise loses the one clause
that matters. The same judge, given a 260-character window around that clause, returned
ENTAILMENT 0.983. The answer was correct and cited, and the platform abstained on it.

This is NOT the premise-length problem already recorded in _premise_for (whole spans overflowing
the 512-token window). 685 characters is well inside the window. Granularity, not length.
"""

from __future__ import annotations

from rag_clients.nli_faithfulness import _candidate_premises
from rag_schemas import RetrievedSpan

# The real chunk, shortened but with its structure intact: the supporting clause is in the middle,
# surrounded by clauses about other fields.
CHUNK = (
    "5.8.6 Revision No.: Revision No. consists of two numerical characters which start from '00' "
    "with increment in one digit after each revision. '00' indicates the first issuance of that "
    "particular SOP. 5.8.7 Effective Date: The effective date shall be stamped by Q.A. after final "
    "authorization before circulating the SOPs. 5.8.8 Supersedes: For the first issuance, 'New "
    "introduction' shall be mentioned in this column, otherwise effective date of obsolete "
    "(previous) SOP shall be mentioned. 5.8.9 Review Date: The normal review period will be 2 years "
    "from the effective date of the SOP. For example if any SOP is having effective date "
    "'01-12-2013', its review date will be '01-12-2015'."
)
SENTENCE = "For the first issuance, 'New introduction' shall be mentioned in the Supersedes column."


def _span(text: str, chunk_id: str = "c1") -> RetrievedSpan:
    return RetrievedSpan(chunk_id=chunk_id, doc_id="sop", text=text, score=0.9, scorer="cross_encoder")


def test_the_premise_is_narrowed_to_the_part_that_could_support_the_sentence() -> None:
    premise = "\n".join(_candidate_premises(SENTENCE, [_span(CHUNK)]))
    assert "New introduction" in premise, "the supporting clause must survive narrowing"
    assert len(premise) < len(CHUNK), "a multi-clause chunk must not be passed whole"
    # The clauses that dominate the chunk but say nothing about Supersedes are what confuse a
    # sentence-pair model; the narrowed premise should not be carrying all of them.
    assert "review date will be" not in premise


def test_a_short_single_topic_chunk_is_passed_through_untouched() -> None:
    """Narrowing must not fire where there is nothing to narrow — a small focused chunk is
    already the right premise, and trimming it could cut the very clause being judged."""
    small = "The night shift allowance is Rs.200 per day, paid with the monthly salary."
    assert _candidate_premises("The night shift allowance is Rs.200 per day.", [_span(small)]) == [small]


def test_the_best_span_still_wins_across_several_spans() -> None:
    other = "Solvent and chemical storage is covered by PBSOP002, revision 3, dated 05-Dec-17."
    premise = "\n".join(_candidate_premises(SENTENCE, [_span(other, "c0"), _span(CHUNK, "c1")]))
    assert "New introduction" in premise


def test_nothing_relevant_still_yields_a_premise() -> None:
    """A sentence with no lexical overlap must still be judged against something, or an
    unsupported claim would be scored against an empty premise and pass by default."""
    unrelated = "Cryogenic fluids may be transported in any elevator in the building."
    assert "".join(_candidate_premises(SENTENCE, [_span(unrelated)])).strip() != ""


# --- one premise per span, not a join of several ------------------------------------------------
# Two distinct ways the premise fails, both measured, and they compose:
#   * JOIN — three spans concatenated into one premise overflow the cross-encoder's window, and
#     the supporting span is the one truncated away. Reported on the other branch from a
#     Lewis-structure question: 0.905 against the full evidence, 0.007 against the truncated join.
#   * GRANULARITY — one 685-char chunk holding four unrelated clauses returns CONTRADICTION 0.80
#     for a sentence written inside it (see the tests above).
# Judging each span SEPARATELY fixes the first; focusing a window inside each fixes the second.


def test_each_candidate_span_is_offered_on_its_own() -> None:
    from rag_clients.nli_faithfulness import _candidate_premises

    a = _span("Two persons shall approve the SOP after review.", "a")
    b = _span(CHUNK, "b")
    premises = _candidate_premises(SENTENCE, [a, b])
    assert len(premises) >= 2, "a claim is supported if ANY passage supports it — ask per passage"
    assert all(len(p) <= 1000 for p in premises), "each candidate must fit the window on its own"


def test_a_supporting_span_is_never_truncated_away_by_a_louder_one() -> None:
    """The join failure: the best-ranked span alone filled the whole budget, so every other span
    was dropped — including the one that actually said it."""
    from rag_clients.nli_faithfulness import _candidate_premises

    filler = _span("Approval and authorization. " * 60, "filler")  # ~1 700 chars, ranks first on length
    premises = _candidate_premises(SENTENCE, [filler, _span(CHUNK, "real")])
    assert any("New introduction" in p for p in premises), premises


def test_windowing_still_applies_inside_each_candidate() -> None:
    from rag_clients.nli_faithfulness import _candidate_premises

    premises = _candidate_premises(SENTENCE, [_span(CHUNK)])
    assert any("New introduction" in p and "review date will be" not in p for p in premises)
