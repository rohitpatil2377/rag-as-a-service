"""Notation grounding (FR-E4, Slice 4) — the notation twin of test_numeracy.

Numeracy checks ASCII figures; this checks the three things its digit-only regex misses: a unit
attached to a number (5 mg vs 5 g), a chemical formula whose sub/superscripts are invisible to
``\\d`` (H₂SO₄ vs H₂SO₃), and invented technical notation (formulae, SMILES). It is conservative
by mandate — a guard that fires on correct answers gets turned off — so the tests pin BOTH
directions: the fabrication is flagged AND the correctly-preserved form is not.
"""

from __future__ import annotations

from rag_core.notation import notation_tokens_in, unsupported_notation
from rag_schemas import RetrievedSpan


def _spans(*texts: str) -> list[RetrievedSpan]:
    return [RetrievedSpan(chunk_id=f"c{i}", doc_id="d", text=t) for i, t in enumerate(texts)]


# --- subscript/ASCII equivalence (the core normalization) -------------------------------------


def test_subscripted_formula_equals_its_ascii_source_form() -> None:
    """The generator preserves H₂SO₄ (grounding rule 8); an OCR'd source may hold H2SO4. Same
    molecule, so it must NOT be flagged — normalized equal on both sides."""
    assert unsupported_notation("The reagent is H₂SO₄.", _spans("Add H2SO4 slowly.")) == []
    # ...and the reverse: ASCII answer against a subscripted source.
    assert unsupported_notation("Add H2SO4 slowly.", _spans("The reagent is H₂SO₄.")) == []


def test_a_real_formula_difference_is_flagged() -> None:
    """H₂SO₃ (sulfurous) is not H₂SO₄ (sulfuric) — a count difference the numeric gate cannot see
    (subscripts aren't \\d), so this is the notation check's job."""
    flagged = unsupported_notation("The reagent is H₂SO₃.", _spans("Add H2SO4 slowly."))
    assert flagged == ["FORMULA:H2S1O3"]


def test_charge_is_part_of_the_formula() -> None:
    assert unsupported_notation("Nitrate is NO₃⁻.", _spans("The NO3- ion is present.")) == []
    assert unsupported_notation("It forms Ca(OH)₂.", _spans("Calcium hydroxide Ca(OH)2 forms.")) == []


def test_caret_and_parenthesised_charge_fold_to_the_unicode_form() -> None:
    """F2 — a source written "SO4^2-" (or "SO4(2-)") must match an answer's preserved "SO₄²⁻"."""
    from rag_core.notation import _canonical_formula

    assert _canonical_formula("SO₄²⁻") == _canonical_formula("SO4^2-") == _canonical_formula("SO4(2-)")
    assert unsupported_notation("Sulfate is SO₄²⁻.", _spans("The SO4^2- ion forms.")) == []


def test_pharma_and_tech_acronyms_are_not_treated_as_formulae() -> None:
    """F1 — common acronyms parse as element runs (IC50→I,C50; CB1→C,B1) but are NOT formulae.
    They must never be flagged, even against sources that do not contain them."""
    from rag_core.notation import notation_tokens_in

    for acronym in ("IC50", "EC50", "LD50", "CB1", "CB2", "S1P", "P2Y12", "USB2", "PS2", "COX2"):
        assert notation_tokens_in(f"The {acronym} value is relevant.") == set(), acronym
        assert unsupported_notation(f"See {acronym} for detail.", _spans("no such token here")) == []


def test_real_formulae_still_detected_after_the_acronym_guard() -> None:
    """The F1 guard must not break genuine formulae."""
    from rag_core.notation import _canonical_formula

    for formula in ("H2SO4", "CO2", "H2O", "Ca(OH)2", "NO3-", "C9H8O4", "SO2", "NH3", "H2O2"):
        assert _canonical_formula(formula) is not None, formula


# --- units (a number + a unit) ----------------------------------------------------------------


def test_a_unit_mismatch_is_flagged_even_though_the_number_is_grounded() -> None:
    """ "5" IS in the source (numeracy passes it); "5 g" is not — the source says 5 mg. The
    unit-bearing pair is what the pharma-safety case turns on."""
    assert unsupported_notation("Give 5 g daily.", _spans("The dose is 5 mg daily.")) == ["UNIT:5g"]


def test_a_matching_unit_quantity_is_not_flagged() -> None:
    assert unsupported_notation("Give 5 mg daily.", _spans("The dose is 5 mg daily.")) == []
    # spacing + trailing-zero tolerance (numeracy's _canonical on the number)
    assert unsupported_notation("Give 5mg daily.", _spans("The dose is 5.0 mg daily.")) == []


def test_percent_symbol_matches_the_spelled_out_source() -> None:
    """A spelled-out source ("20 percent") must not false-flag a symbol answer ("20%")."""
    assert unsupported_notation("Yield rose 20%.", _spans("a yield increase of 20 percent")) == []


def test_rate_units_carry_their_denominator() -> None:
    assert unsupported_notation("Concentration 5 mg/mL.", _spans("prepared at 5 mg/mL")) == []
    # same number+numerator, different denominator -> a real mismatch
    assert unsupported_notation("Concentration 5 mg/L.", _spans("prepared at 5 mg/mL")) == ["UNIT:5mg/l"]


# --- SMILES -----------------------------------------------------------------------------------


def test_masking_stops_a_smiles_ring_digit_being_seen_as_an_invented_figure() -> None:
    """L1 — the ``1`` in ``c1ccccc1`` is a ring closure, not a quantity. Masking notation tokens
    before the numeric scan stops it being mis-reported as an invented figure."""
    from rag_core.notation import mask_notation_tokens
    from rag_core.numeracy import unsupported_numbers

    answer = "The structure CC(=O)Oc1ccccc1C(=O)O appears."
    spans = _spans("Aspirin is a common drug with no digits.")
    assert unsupported_numbers(answer, spans) == ["1"]  # WITHOUT masking, the ring digit flags
    assert unsupported_numbers(mask_notation_tokens(answer), spans) == []  # WITH masking, suppressed


def test_invented_smiles_is_flagged_and_verbatim_is_not() -> None:
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    assert unsupported_notation(f"The structure is {aspirin}.", _spans("Aspirin is a common drug.")) == [
        "SMILES:" + aspirin
    ]
    assert unsupported_notation(f"The structure is {aspirin}.", _spans(f"Aspirin: {aspirin} is used.")) == []


# --- false-positive discipline (the whole ethos) ----------------------------------------------


def test_plain_prose_never_trips_the_check() -> None:
    assert notation_tokens_in("The Content Curator approves all external material.") == set()
    assert unsupported_notation("The reviewer signs off within five working days.", _spans("anything")) == []


def test_plain_numbers_are_left_to_numeracy() -> None:
    """A bare figure with no unit is numeracy's concern, not notation's — no double-handling."""
    assert unsupported_notation("The total is 144 hours over 18 days.", _spans("nothing here")) == []


def test_identifiers_are_not_treated_as_formulae() -> None:
    """SOP-BULK-001 is slice 2a's retrieval concern; its digits are numeric-checked already."""
    assert unsupported_notation("See SOP-BULK-001 for detail.", _spans("no such id")) == []


def test_a_correct_pharma_answer_with_several_notations_is_not_flagged() -> None:
    """The acceptance case: preserved formulae + doses that ARE grounded must all pass."""
    sources = _spans(
        "Dissolve the API in H2SO4 and titrate to pH 7.",
        "The recommended dose is 5 mg/mL, stored at 25°C.",
        "Acetylsalicylic acid has the formula C9H8O4.",
    )
    answer = "Dissolve in H₂SO₄, dose 5 mg/mL, store at 25°C. The active ingredient is C₉H₈O₄."
    assert unsupported_notation(answer, sources) == []
