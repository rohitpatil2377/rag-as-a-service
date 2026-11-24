"""RDKitMoleculeNormalizer (Slice 6, Phase C) — the REAL toolkit path.

Skipped entirely when RDKit is not installed (CI does not carry the optional ``rag-clients[chem]``
extra), so `make test` stays green without it. Where RDKit IS present, this proves the property the
whole feature rests on: chemically-identical SMILES canonicalize to the SAME string + InChIKey,
however they were drawn; an invalid SMILES is None; and it never raises.
"""

from __future__ import annotations

import importlib.util

import pytest

from rag_clients.chem_normalize import RDKitMoleculeNormalizer

rdkit_missing = importlib.util.find_spec("rdkit") is None
pytestmark = pytest.mark.skipif(rdkit_missing, reason="rdkit not installed (optional rag-clients[chem] extra)")


def _normalizer() -> RDKitMoleculeNormalizer:
    return RDKitMoleculeNormalizer()


def test_rdkit_is_reported_available() -> None:
    assert _normalizer().available is True


def test_ethanol_spellings_all_canonicalize_equal() -> None:
    """OCC, CCO and C(C)O are three ways to write ethanol — same canonical SMILES AND InChIKey."""
    n = _normalizer()
    mols = [n.canonicalize(s) for s in ("OCC", "CCO", "C(C)O")]
    assert all(m is not None for m in mols)
    canon = {m.canonical_smiles for m in mols if m is not None}
    keys = {m.inchikey for m in mols if m is not None}
    assert len(canon) == 1, f"ethanol spellings disagreed on canonical SMILES: {canon}"
    assert len(keys) == 1, f"ethanol spellings disagreed on InChIKey: {keys}"


def test_benzene_aromatic_equals_kekule() -> None:
    n = _normalizer()
    aromatic = n.canonicalize("c1ccccc1")
    kekule = n.canonicalize("C1=CC=CC=C1")
    assert aromatic is not None and kekule is not None
    assert aromatic.canonical_smiles == kekule.canonical_smiles
    assert aromatic.inchikey == kekule.inchikey


def test_invalid_smiles_returns_none_and_never_raises() -> None:
    n = _normalizer()
    for junk in ("not_a_molecule", "C(((", "[Xx]", "", "12345"):
        assert n.canonicalize(junk) is None, junk


def test_inchikey_has_the_expected_shape() -> None:
    mol = _normalizer().canonicalize("CCO")
    assert mol is not None
    # Standard InChIKey: 14 + 1 + 8 + 1 + 1 blocks separated by hyphens (27 chars, uppercase).
    assert len(mol.inchikey) == 27 and mol.inchikey.count("-") == 2
