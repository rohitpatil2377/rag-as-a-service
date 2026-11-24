"""Chemical SMILES normalization (Slice 6, Phase C) — the PURE half (no rdkit).

Two things are tested here, both dependency-free: the conservative candidate DETECTOR (it must catch
real SMILES incl. rings/aromatic/branches and the tiny ethanol chains, and must NOT fire on ordinary
words, plain formulae notation.py owns, citation/list artifacts, or pharma/assay acronyms), and the
:func:`canonical_terms` helper driven by a tiny FAKE normalizer (the real RDKit adapter is tested,
skip-guarded, in the clients suite). The detector is conservative by mandate — a guard that fires on
prose adds a stray searchable token — so the tests pin BOTH directions.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from rag_core.chem import (
    CanonicalMolecule,
    NullMoleculeNormalizer,
    canonical_terms,
    find_smiles_candidates,
)

# --- candidate detection: positives ------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The solvent is CCO in the mixture.",  # ethanol, structureless carbon chain
        "written as OCC by the other lab",  # ethanol, the other way round
        "benzene c1ccccc1 is aromatic",  # aromatic ring (lowercase + ring digit)
        "kekule form C1=CC=CC=C1 of benzene",  # kekulé (bonds + ring digits)
        "aspirin CC(=O)Oc1ccccc1C(=O)O here",  # branches + aromatic ring
        "a stereocentre C[C@@H](N)C in the chain",  # bracket atom
        "acetone CC(=O)C reacts",  # branch + carbonyl
    ],
)
def test_real_smiles_tokens_are_detected(text: str) -> None:
    found = find_smiles_candidates(text)
    assert len(found) == 1, f"expected exactly one candidate in {text!r}, got {found}"


def test_multiple_candidates_are_deduped_in_order() -> None:
    assert find_smiles_candidates("mix CCO then c1ccccc1 then CCO again") == ["CCO", "c1ccccc1"]


# --- candidate detection: negatives (the guards) -----------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The cat sat on a big dog and ran",  # ordinary prose
        "NO ON so con spoon cocoon nib pin son",  # all-atom-letter words (min length / carbon / ring guards)
        "see section 5 and clause 12 hereby",  # legal prose
        "as shown in [12] and [1, 2] above",  # citation artifacts
        "1. first 2) second (i) third a. b.",  # list markers
        "the reagent H2SO4 and gas CO2 form",  # plain formulae — notation.py owns these
        "the IC50 and EC50 and LD50 values",  # assay acronyms
        "the CB1 and CB2 and S1P receptors",  # receptor acronyms
        "plug into USB2 on the PS2 console",  # tech acronyms
        "upgrade to v2 and read 25mg dosing",  # version / dose (notation owns the unit)
    ],
)
def test_ordinary_text_never_qualifies(text: str) -> None:
    assert find_smiles_candidates(text) == [], text


# --- NullMoleculeNormalizer (the fail-safe default) --------------------------------------------


def test_null_normalizer_canonicalizes_nothing() -> None:
    null = NullMoleculeNormalizer()
    assert null.canonicalize("CCO") is None
    assert null.canonicalize("c1ccccc1") is None
    # ...so canonical_terms is always [] with it — the flag-off / rdkit-absent byte-identical path.
    assert canonical_terms("the solvent CCO and benzene c1ccccc1", null) == []


# --- canonical_terms with a FAKE in-test normalizer --------------------------------------------


class _FakeNormalizer:
    """A tiny stub: ethanol's two spellings collapse to CCO, benzene's two to c1ccccc1. Everything
    else is unknown (None) — exactly how a real toolkit behaves for a non-molecule."""

    _CANON: ClassVar[dict[str, tuple[str, str]]] = {
        "OCC": ("CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        "CCO": ("CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        "c1ccccc1": ("c1ccccc1", "UHOVQNZJYSORNB-UHFFFAOYSA-N"),
        "C1=CC=CC=C1": ("c1ccccc1", "UHOVQNZJYSORNB-UHFFFAOYSA-N"),
    }

    def canonicalize(self, smiles: str) -> CanonicalMolecule | None:
        pair = self._CANON.get(smiles)
        return CanonicalMolecule(canonical_smiles=pair[0], inchikey=pair[1]) if pair else None


def test_canonical_terms_returns_canonical_smiles_and_inchikey() -> None:
    terms = canonical_terms("the solvent is OCC in this run", _FakeNormalizer())
    assert terms == ["CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"]


def test_equivalent_molecules_collapse_to_the_same_terms() -> None:
    """OCC and CCO are the same molecule — both canonicalize to the same term set (the whole point:
    a query written one way matches a document written the other)."""
    assert canonical_terms("dose OCC", _FakeNormalizer()) == canonical_terms("dose CCO", _FakeNormalizer())
    assert canonical_terms("ring c1ccccc1", _FakeNormalizer()) == canonical_terms("ring C1=CC=CC=C1", _FakeNormalizer())


def test_canonical_terms_dedupes_repeats_preserving_order() -> None:
    # Two molecules; each contributes canonical SMILES then InChIKey, once, in reading order.
    assert canonical_terms("CCO and c1ccccc1 and CCO", _FakeNormalizer()) == [
        "CCO",
        "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "c1ccccc1",
        "UHOVQNZJYSORNB-UHFFFAOYSA-N",
    ]


def test_canonical_terms_empty_when_no_candidates() -> None:
    assert canonical_terms("the cat sat on the mat", _FakeNormalizer()) == []


def test_canonical_terms_empty_when_normalizer_rejects_candidate() -> None:
    # A detected-but-unknown SMILES (the fake returns None) contributes nothing — mirrors a real
    # toolkit rejecting an invalid structure.
    assert canonical_terms("mystery CC(=O)N1CCCCC1 here", _FakeNormalizer()) == []
