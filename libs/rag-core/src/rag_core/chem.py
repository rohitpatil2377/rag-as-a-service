"""Chemical SMILES normalization (Slice 6, Phase C) — PURE detection, NO rdkit import.

The Slice-2a exact-term path already indexes chunk text into a case-preserving whitespace
``text.exact`` sub-field, so a SMILES matches VERBATIM. The gap this module closes: two SMILES that
are the SAME molecule written differently — ``OCC`` vs ``CCO`` for ethanol, aromatic ``c1ccccc1`` vs
Kekulé ``C1=CC=CC=C1`` for benzene — do NOT match verbatim. RDKit collapses any valid SMILES to ONE
canonical string plus an InChIKey, so equivalent molecules become the same token and match. That
canonicalization needs a chemistry toolkit, so it lives behind the :class:`MoleculeNormalizer` seam
(the concrete RDKit adapter is in ``rag_clients``); THIS module is the pure, dependency-free,
CI-safe half: it decides WHICH tokens are worth handing to a normalizer, and turns a normalizer's
verdicts into the deduped canonical terms both ingest and query index/match on.

It mirrors :mod:`rag_core.notation`'s discipline and voice. notation.py is a GROUNDING check
("is this exact token in a cited span") and is verbatim by design; this is a RETRIEVAL-matching
helper ("what canonical form should match this molecule however it was drawn"). The two are
deliberately separate: notation owns chemical FORMULAE (``H₂SO₄``, ``CO2``) and verbatim SMILES;
this owns the CANONICAL identity of a SMILES. A token notation already treats as a formula is left
to it (see the ``_canonical_formula`` guard below), so the two never fight over the same token.

Detection is CONSERVATIVE by mandate, exactly like notation: a guard that fires on ordinary prose
does more harm than good, because a false candidate that a normalizer happens to accept adds a
stray searchable token. The detector therefore admits a token only when it is unambiguously
SMILES-shaped — every character a SMILES character, every bare letter a SMILES atom (the strongest
guard: an ordinary word almost always contains a letter, like a vowel a/e/u or g/k/t, that is not a
SMILES atom) — AND it clears one of a few structural signals that separate a real structure from a
short word that happens to be all-atom-letters. When in doubt it does NOT emit a candidate; the
Null normalizer (and any invalid SMILES a real toolkit rejects) then yields nothing, so the whole
path fails safe to "no extra tokens", byte-identical to today.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_core.notation import (
    _ACRONYM_MIN_INDEX,
    _ACRONYM_RE,
    _FORMULA_DENYLIST,
    _canonical_formula,
)

# The metadata key under which the canonical terms ride on a chunk to the sparse index. A plain
# str value (space-joined) so it fits ``Chunk.metadata`` (``dict[str, str]``) and every sparse
# backend can already carry it with no schema churn.
CHEM_TERMS_META = "chem_terms"

# --- SMILES character / atom alphabets ---------------------------------------------------------
# Aliphatic organic-subset atoms that appear as BARE (un-bracketed) UPPERCASE letters. ``H`` only
# ever appears inside a bracket atom (``[nH]``) in real SMILES, so it is a valid CHARACTER but a
# bare ``H`` is not treated as a chemistry signal. ``l``/``r`` complete the two-letter Cl/Br.
_ALIPHATIC = set("BCNOPSFI")
# Aromatic atoms are the lowercase ``b c n o p s`` — and in valid SMILES they live in a RING, i.e.
# alongside a ring-closure digit. That fact is the guard that keeps lowercase words ("con", "spoon",
# "cocoon") — all of whose letters happen to be aromatic-atom letters — from ever qualifying.
_AROMATIC = set("bcnops")
_BARE_LETTERS = _ALIPHATIC | _AROMATIC | set("lrH")
# The full SMILES character set: atoms + bond/branch/bracket/charge/ring punctuation + digits.
_SMILES_CHARS = _BARE_LETTERS | set("()[]=#@+-.\\/%") | set("0123456789")
# A structural signal beyond a bare atom chain: a bond, a branch, or a bracket atom.
_STRUCT = set("=#()[]")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_TRAILING_PUNCT = ".,;:!?"
_QUOTE_STRIP = "\"'“”‘’"  # noqa: RUF001 - smart quotes stripped as data, never compared

# Long enough to be a molecule, not a two-letter word/atom ("NO", "ON", "IS", "CO"). Ethanol
# ``CCO``/``OCC`` is the shortest thing we must still catch, so the floor is 3.
_MIN_LEN = 3
# A structureless uppercase chain needs a real carbon backbone (CCO has two; "PIN"/"SON" have zero).
_MIN_CARBONS = 2
# Residual collisions the structural guards below cannot catch: ALL-CAPS words/acronyms that are all
# valid SMILES characters AND clear the carbon/structure gates. This is NOT harmless noise — many of
# them (NCC, CCC, CCCP, OCCC, CNBC, ICBC, BCCI) are VALID SMILES that RDKit canonicalizes to a real
# molecule, so inside a chemistry KB they could cross-match; the denylist is the containment. The
# structural guards already reject two-letter words (length), lowercase words with no ring digit
# (aromatic atoms live in rings) and structureless chains with fewer than two carbons ("PIN", "SON").
# The membership check below ALSO strips a trailing plural 'S' (PICNICS→PICNIC, CISCOS→CISCO), so only
# the singular needs listing. Best-effort precision, not perfection — extensible.
_DENYLIST = frozenset(
    {
        "CHIC",
        "PICNIC",
        "CONIC",
        "COMIC",
        "SONIC",
        "IONIC",
        "CISCO",
        "COCCI",
        "CROCS",
        "NCC",
        "CCC",
        "CCCP",
        "OCCC",
        "CNBC",
        "ICBC",
        "BCCI",  # short valid-SMILES acronyms (review)
    }
)


@dataclass(frozen=True)
class CanonicalMolecule:
    """A SMILES resolved to its canonical identity: the canonical SMILES string RDKit emits and the
    InChIKey (a hash of the molecule). Two ways to write the same molecule collapse to the SAME pair,
    which is the whole point — both become the same searchable token."""

    canonical_smiles: str
    inchikey: str


@runtime_checkable
class MoleculeNormalizer(Protocol):
    """The seam a chemistry toolkit sits behind. ``canonicalize`` returns the canonical identity of a
    SMILES, or ``None`` for anything it cannot parse — it NEVER raises, so ingest/query stay safe."""

    def canonicalize(self, smiles: str) -> CanonicalMolecule | None: ...


class NullMoleculeNormalizer:
    """The fail-safe default — canonicalizes nothing. Returned whenever the master flag is off or
    RDKit is unavailable, so :func:`canonical_terms` yields ``[]`` and the ingest/query paths are
    byte-identical to today. (The Slice-4 ``NullMoleculeNormalizer`` twin of a real toolkit.)"""

    def canonicalize(self, smiles: str) -> CanonicalMolecule | None:
        return None


def _has_smiles_signal(tok: str, bare: str) -> bool:
    """Whether a well-formed (all-SMILES-character, all-atom-letter) token clears one structural
    signal that separates a real structure from a short all-atom-letter word: a bond/branch/bracket;
    OR an aromatic lowercase atom WITH a ring digit (aromatic atoms are only valid inside a ring —
    this rejects "con"/"spoon"); OR a ring-closure digit on a carbon-bearing chain (``C1CCCCC1``);
    OR, for a bare uppercase chain with no structure at all, at least two carbons (``CCO`` yes)."""
    if any(c in _STRUCT for c in tok):
        return True
    has_digit = any(c.isdigit() for c in tok)
    carbons = sum(1 for c in bare if c in "Cc")
    if any(c in _AROMATIC for c in bare):
        return has_digit
    if has_digit:
        return carbons >= 1  # ring-closure aliphatic (C1CCCCC1)
    return carbons >= _MIN_CARBONS  # a structureless chain needs a real carbon backbone


def find_smiles_candidates(text: str) -> list[str]:
    """Every SMILES-like token in ``text``, in reading order, deduped.

    Conservative and high-precision. A candidate must:

    * be at least :data:`_MIN_LEN` characters (kills two-letter words/atoms: "NO", "ON", "IS", "CO");
    * NOT be in the ALL-CAPS-word :data:`_DENYLIST` (the residual multi-carbon word collisions);
    * be composed ENTIRELY of SMILES characters, and every BARE (un-bracketed) letter must be a
      SMILES atom letter — the strongest guard, because an ordinary word almost always carries a
      letter (a vowel a/e/u, or g/k/t/…) that is not a SMILES atom;
    * contain at least one atom letter (so a citation/list artifact like ``[12]`` — brackets and
      digits, no atom — never qualifies);
    * NOT already be a chemical FORMULA that :mod:`rag_core.notation` owns (``CO2``, ``H2SO4``);
      notation canonicalizes formulae, this canonicalizes SMILES, and they must not fight; and
    * clear ONE structural signal that separates a real structure from a short all-atom-letter word:
      a bond/branch/bracket (``=`` ``#`` ``(`` ``[``); OR an aromatic lowercase atom **with** a ring
      digit (aromatic atoms are only valid inside a ring — this is what rejects "con"/"spoon"); OR a
      ring-closure digit on a carbon-bearing chain (``C1CCCCC1``); OR, for a bare uppercase chain
      with no structure at all, at least TWO carbons (``CCO`` yes; "PIN"/"SON" — zero carbons — no).

    Detection only chooses what to HAND a normalizer; RDKit is the final gate. A stray candidate that
    is NOT a valid molecule canonicalizes to nothing. But a stray candidate that IS a valid molecule
    (e.g. an all-caps acronym like ``NCC``) canonicalizes to a real token that, inside a CHEMISTRY KB,
    CAN cross-match — this does NOT "match nothing". The containment is: it applies only to a
    chemistry KB, only as an ADDITIVE ``should`` clause on a separate ``chem`` field, so RRF + the
    cross-encoder reranker down-rank a spurious hit and it NEVER removes a correct result — plus the
    denylist above catches the common ones. Best-effort precision, not perfection.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.split():
        tok = raw.strip(_TRAILING_PUNCT).strip(_QUOTE_STRIP)
        if len(tok) < _MIN_LEN:
            continue
        # Reject a denylisted ALL-CAPS word/acronym, AND its plural (strip a trailing 'S' and re-check,
        # so PICNICS/CISCOS fall out with PICNIC/CISCO already listed).
        upper = tok.upper()
        if upper in _DENYLIST or (upper.endswith("S") and upper[:-1] in _DENYLIST):
            continue
        # Pharma/assay/receptor acronyms parse as atom+index (IC50→I,C50; CB1→C,B1; S1P; USB2) but
        # are NOT molecules. notation.py owns exactly these — reuse its denylist AND its all-caps +
        # trailing-index shape (index >= 10, so real diatomics like CO2 are untouched) so the two
        # modules agree on what is an acronym and this never sees one as SMILES.
        if tok.upper() in _FORMULA_DENYLIST:
            continue
        _acronym = _ACRONYM_RE.fullmatch(tok.upper())
        if _acronym and int(_acronym.group(1)) >= _ACRONYM_MIN_INDEX:
            continue
        if any(c not in _SMILES_CHARS for c in tok):
            continue
        bare = _BRACKET_RE.sub("", tok)
        # A bare letter that is not a SMILES atom letter means the token is a word, not a structure.
        if any(c.isalpha() and c not in _BARE_LETTERS for c in bare):
            continue
        if not any(c.isalpha() for c in tok):
            continue  # brackets/digits only (e.g. a citation "[12]") — no atom, not SMILES
        # A formula notation.py already canonicalizes is left to notation, not double-handled here.
        if _canonical_formula(tok) is not None:
            continue
        if not _has_smiles_signal(tok, bare) or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def canonical_terms(text: str, normalizer: MoleculeNormalizer) -> list[str]:
    """The canonical searchable terms for every molecule detected in ``text``, deduped, in order.

    For each detected SMILES candidate, ask the normalizer for its canonical identity and collect
    both the canonical SMILES and the InChIKey (two spellings that let a query match whichever form a
    document happened to store). The SINGLE pure helper both the ingest path (stamp these onto the
    chunk as extra exact tokens) and the query path (add these to the query so it matches them) use.

    Returns ``[]`` when nothing is detected OR the normalizer canonicalizes nothing — so with the
    :class:`NullMoleculeNormalizer` (flag off / RDKit absent) the result is always ``[]`` and both
    paths stay byte-identical."""
    out: list[str] = []
    seen: set[str] = set()
    for candidate in find_smiles_candidates(text):
        mol = normalizer.canonicalize(candidate)
        if mol is None:
            continue
        for term in (mol.canonical_smiles, mol.inchikey):
            if term and term not in seen:
                seen.add(term)
                out.append(term)
    return out
