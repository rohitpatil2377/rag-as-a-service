"""When a question NAMES a document, the answer must come from that document.

Measured on a real two-document corpus: "summarise SOP Writing Guidance Pharma" retrieved 5 of 6
passages from "SOPs - Leslie Dan Faculty of Pharmacy" and one from the document actually named.
Both are about SOPs, so on content alone they are near neighbours — and the answer came back
confident, cited, faithfulness 1.0, because that gate asks whether the answer is supported by the
passages and never whether they are the passages the reader asked for.

Scoping is a FILTER, not a better score, and it is conservative by design: a false positive
silently hides the rest of the corpus, which is worse than not scoping at all.
"""

from __future__ import annotations

from rag_core.docscope import distinctive, scope_for_query

CORPUS = [
    "SOPs - Leslie Dan Faculty of Pharmacy - Single Document (1).pdf",
    "SOP Writing Guidance Pharma.pdf",
    "images.jpeg",
]


def test_a_named_document_is_scoped_to() -> None:
    scope = scope_for_query("summarise SOP Writing Guidance Pharma", CORPUS)
    assert scope is not None
    assert scope.doc_id == "SOP Writing Guidance Pharma.pdf"


def test_a_paraphrased_name_still_matches() -> None:
    """People do not retype a filename; they paraphrase it ("pharma" for "pharmacy", "sop" for
    "sops")."""
    scope = scope_for_query("what does the writing guidance for pharma say about revisions?", CORPUS)
    assert scope is not None and scope.doc_id == "SOP Writing Guidance Pharma.pdf"


def test_an_ordinary_question_scopes_to_nothing() -> None:
    """The common case, and the safe one — it must keep searching the whole knowledge base."""
    for q in (
        "what is the revision numbering scheme?",
        "who approves an SOP?",
        "how many documents are there?",
    ):
        assert scope_for_query(q, CORPUS) is None, q


def test_an_ambiguous_reference_refuses_to_choose() -> None:
    """Two documents are both SOPs. Guessing between them is exactly the failure this prevents."""
    assert scope_for_query("summarise the SOP document", CORPUS) is None


def test_generic_filename_words_identify_nothing() -> None:
    """Every corpus has a "Final Document v2". Matching on those words would scope "the final
    document" to whichever file happened to be called that."""
    assert distinctive("Final Document v2.pdf") == set()
    assert scope_for_query("show me the final document", CORPUS) is None


def test_an_empty_corpus_or_query_is_not_a_match() -> None:
    assert scope_for_query("", CORPUS) is None
    assert scope_for_query("summarise SOP Writing Guidance Pharma", []) is None
