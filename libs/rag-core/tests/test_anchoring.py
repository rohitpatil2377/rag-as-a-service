"""Lexical anchoring: is a sentence drawn from ONE retrieved passage?

Each test pins one of the four defects the weighted anchor fixes (see rag_core.anchoring).
"""

from __future__ import annotations

from rag_core.anchoring import anchor_score, content_terms, idf_weights, script_of

_STOP = frozenset({"the", "and", "for", "are", "with", "per", "all", "must", "their", "who"})

_POLICY = [
    "Acme employees working the night shift are paid a night shift allowance of Rs.200 per day.",
    "Acme employees must follow the Acme travel policy; travel advances are settled within 48 hours.",
    "Acme employees nominated as Torch Bearers receive leadership training and an annual vacation.",
]


def test_words_every_passage_shares_carry_less_weight_than_distinctive_ones() -> None:
    evidence = [set(content_terms(p, stopwords=_STOP)) for p in _POLICY]
    w = idf_weights({"acme", "employees", "allowance", "sabbatical"}, evidence)
    assert w["acme"] < w["allowance"] < w["sabbatical"]
    assert w["acme"] == w["employees"]


def test_a_fabrication_built_from_common_corpus_words_no_longer_anchors_like_a_quote() -> None:
    """The defect: "Acme", "employees", "policy" are in every passage, so an invented sentence
    made of them looked as anchored as a real one. Weighted, the invented specifics dominate."""
    invented = "Acme employees receive a sabbatical of six months under the Acme policy."
    quoted = "Acme employees on the night shift are paid a night shift allowance per day."
    unweighted_invented = 3 / 6  # acme, employees, policy of acme/employees/receive/sabbatical/six/months/policy
    weighted_invented = anchor_score(invented, _POLICY, stopwords=_STOP)
    assert weighted_invented < unweighted_invented
    assert anchor_score(quoted, _POLICY, stopwords=_STOP) > 0.9


def test_an_inflected_form_matches_its_source_word() -> None:
    """ "approves" never met "approved" under exact matching."""
    passages = ["Expense claims are approved by the Business Unit Head."]
    assert anchor_score("The Business Unit Head approves expense claims.", passages, stopwords=_STOP) == 1.0


def test_hindi_is_now_judged_against_hindi_evidence() -> None:
    """The ASCII-only tokenizer found no words in Hindi, so every Hindi sentence scored 1.0."""
    passages = ["रात्रि पाली भत्ता 200 रुपये प्रति दिन की दर से दिया जाएगा।"]
    assert script_of("भत्ता") == "DEVANAGARI"
    grounded = anchor_score("रात्रि पाली भत्ता 200 रुपये प्रति दिन है।", passages)
    unrelated = anchor_score("कंपनी हर साल विदेश यात्रा का आयोजन करती है।", passages)
    assert grounded > 0.8
    assert unrelated < 0.3


def test_a_cross_lingual_answer_is_not_punished_for_sharing_no_words() -> None:
    """A Hindi answer drawn from English sources shares no words with them by construction. The
    anchor cannot judge that and must say so, leaving it to the multilingual NLI check."""
    assert anchor_score("रात्रि पाली भत्ता 200 रुपये प्रति दिन है।", _POLICY, stopwords=_STOP) == 1.0


def test_length_normalization_penalises_long_passages_and_never_rewards_short_ones() -> None:
    sentence = "Torch Bearers receive leadership training."
    short = "Torch Bearers receive leadership training."
    long_padding = " ".join(f"clause{i} unrelated wording" for i in range(40))
    long = f"{long_padding} Torch Bearers receive leadership training."
    medium = "Leave and attendance rules apply to every employee across all offices and departments."

    evidence = [long, medium]
    plain = anchor_score(sentence, evidence, stopwords=_STOP)
    normalized = anchor_score(sentence, evidence, stopwords=_STOP, length_norm=0.75)
    assert plain == 1.0
    assert normalized < plain, "a match found only inside a far-longer-than-average span is discounted"

    # A short span is never boosted above its plain coverage (no reward for being short).
    assert anchor_score(sentence, [short, long], stopwords=_STOP, length_norm=0.75) == 1.0


def test_nothing_to_judge_is_not_a_failure() -> None:
    assert anchor_score("It depends.", _POLICY, stopwords=_STOP | {"depends"}) == 1.0
    assert anchor_score("Anything at all.", [], stopwords=_STOP) == 1.0


def test_notation_folds_for_comparison_so_a_formula_answer_still_anchors() -> None:
    """Ingestion PRESERVES subscripts (H₂SO₄ stays H₂SO₄), but a model answering about it writes
    "H2SO4" as often as not. Without folding for comparison, "so₄" and "so4" are different tokens,
    the sentence looks unanchored in every span, and a correct answer scores 0.0 and abstains."""
    passages = ["The reaction uses H₂SO₄ and BaCl₂ to precipitate 4FeO₄²⁻."]
    assert anchor_score("The reaction uses H2SO4 and BaCl2.", passages, stopwords=_STOP) > 0.9
    # ...and the other direction, since either side may carry the notation.
    assert anchor_score("The reaction uses H₂SO₄ and BaCl₂.", ["The reaction uses H2SO4 and BaCl2."]) > 0.9


def test_a_short_word_and_its_plural_are_the_same_word() -> None:
    """A four-letter noun can never reach the five-character shared prefix, so every short plural
    read as a different word — "atom"/"atoms", "bond"/"bonded"."""
    from rag_core.anchoring import same_term

    assert same_term("atom", "atoms")
    assert same_term("bond", "bonded")
    assert same_term("expense", "expenses")
    # ...but a shared prefix that is not the whole shorter word is still a different word.
    assert not same_term("bond", "bone")
    assert not same_term("cat", "car")
