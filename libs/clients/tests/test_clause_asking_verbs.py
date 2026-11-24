"""A question's ASKING VERB is not something a document can contain.

The compound-query relevance gate asks which sub-question the retrieved passages cover, by
counting how many of the clause's terms a passage carries. "What does Supersedes SAY on first
issue?" puts "say" in that term set — and no policy document ever contains the word "say", so the
clause could never reach the strict majority the gate requires. Measured live on Acme's SOP
corpus: the passage that literally answers the question matched 'supersedes' and 'first' but not
'say', scored 2/4 = 0.50, and the clause was declared uncovered. The correct, fully-anchored
answer sentence was then gated as "only supported because nothing contradicts it", and a question
whose answer is written verbatim in the source abstained.

Asking verbs are removed from the CLAUSE term set only. Retrieval is untouched, and the gate's
purpose is untouched: a sub-question whose TOPIC words are absent from every passage is still
uncovered, which is the adversarial case it was built for.
"""

from __future__ import annotations

from rag_clients.nli_faithfulness import _clause_terms, _query_clauses, _terms, _uncovered_clauses
from rag_schemas import RetrievedSpan

SOP_CHUNK = (
    "5.8.8 Supersedes: For the first issuance, 'New introduction' shall be mentioned in this "
    "column, otherwise effective date of obsolete (previous) SOP shall be mentioned."
)


def _span(text: str) -> RetrievedSpan:
    return RetrievedSpan(chunk_id="c1", doc_id="sop", text=text, score=0.85, scorer="cross_encoder")


def test_an_asking_verb_is_not_a_clause_term() -> None:
    assert "say" not in _clause_terms("what does Supersedes say on first issue")
    assert "mention" not in _clause_terms("what do the sources mention about leave")
    assert "describe" not in _clause_terms("describe the approval flow")


def test_the_filter_is_scoped_to_clauses_and_never_touches_premise_ranking() -> None:
    """Why this is not in _terms: premise ranking and window selection use _terms, and the SOURCE
    genuinely writes "shall be MENTIONED in this column". Filtering globally sent the focus window
    to the wrong clause and re-broke the very question the clause fix was for — caught live, after
    the first version of this change shipped."""
    assert "mentioned" in _terms("'New introduction' shall be mentioned in this column")
    assert "mentioned" not in _clause_terms("what does the column mentioned on first issue say")


def test_the_clause_its_passage_answers_is_covered() -> None:
    clauses = _query_clauses(
        "What is the revision number after the third revision, and what does Supersedes say on first issue?"
    )
    uncovered = _uncovered_clauses([_span(SOP_CHUNK)], clauses)
    supersedes = next(c for c in clauses if "supersedes" in c)
    assert supersedes not in uncovered, "the passage that answers it verbatim must cover it"


def test_a_word_that_carries_topic_is_still_a_term() -> None:
    """The narrowing must not eat content. "State" stays: an Indian policy corpus is full of
    State Government, and "require"/"report" are what policies DO."""
    terms = _terms("which state government must approve the report and what does it require")
    assert {"state", "report", "require"} <= terms


def test_a_genuinely_absent_sub_question_is_still_uncovered() -> None:
    """The adversarial case the gate exists for: a compound query whose second half this KB
    cannot answer at all must still be flagged, or a confident wrong figure sails through."""
    clauses = _query_clauses(
        "What does Supersedes say on first issue, and what is the FY2027 Platform Engineering budget?"
    )
    uncovered = _uncovered_clauses([_span(SOP_CHUNK)], clauses)
    assert any("budget" in c for c in uncovered)
