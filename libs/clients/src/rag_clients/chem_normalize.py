"""RDKit-backed molecule normalizer (implements the rag-core :class:`MoleculeNormalizer` seam).

RDKit is an OPTIONAL, import-guarded dependency (the ``rag-clients[chem]`` extra): it is a heavy
C++ toolkit and CI must not carry it, so this module imports it defensively and degrades to "not
available" when it is absent. Canonicalization NEVER raises into ingest/query — an invalid SMILES,
an unparseable molecule, or a missing RDKit all resolve to ``None`` — so the whole feature is
fail-safe, exactly like the Null twin the factory falls back to.
"""

from __future__ import annotations

from rag_common import get_logger
from rag_core.chem import CanonicalMolecule

_log = get_logger(__name__)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey

    # RDKit logs parse failures on stderr via its C++ logger; invalid SMILES are EXPECTED here (a
    # conservative detector still hands over the occasional non-molecule), so silence it — a returned
    # None already carries that signal, and the noise would swamp real logs.
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_AVAILABLE = True
except ImportError:  # this branch runs when rdkit is ABSENT (the CI/default case)
    Chem = None
    MolToInchiKey = None
    _RDKIT_AVAILABLE = False


class RDKitMoleculeNormalizer:
    """Canonicalize a SMILES to its RDKit canonical string + InChIKey. ``available`` reports whether
    RDKit imported, so the factory can fall back to the Null twin when it did not."""

    available: bool = _RDKIT_AVAILABLE

    def canonicalize(self, smiles: str) -> CanonicalMolecule | None:
        """The molecule's canonical identity, or ``None`` for a None/invalid SMILES or when RDKit is
        unavailable. Never raises — any RDKit error is logged and swallowed as ``None``."""
        if not _RDKIT_AVAILABLE or not smiles:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            canonical = Chem.MolToSmiles(mol)
            inchikey = MolToInchiKey(mol)
        except Exception as exc:  # RDKit must never fail an ingest or a query
            _log.warning("rdkit_canonicalize_failed", smiles=smiles[:64], error=str(exc)[:160])
            return None
        if not canonical:
            return None
        return CanonicalMolecule(canonical_smiles=canonical, inchikey=inchikey or "")
