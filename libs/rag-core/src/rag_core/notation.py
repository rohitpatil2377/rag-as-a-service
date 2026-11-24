"""Notation grounding (FR-E4 support, Slice 4) — pure logic, no I/O.

This EXTENDS :mod:`rag_core.numeracy`. That layer checks whether every ASCII *figure* an
answer asserts was retrieved; its `_NUMBER_RE` deliberately matches digits only, so three things
slip past it:

* **units.** "5 mg" and "5 g" both reduce to the figure "5" — the unit sits OUTSIDE the number
  match, so a dose stated in the wrong unit reads as grounded.
* **sub/superscripts.** H₂SO₄'s subscripts are not ``\\d``, so H₂SO₄ and H₂SO₃ carry the same
  (empty) numeric footprint — a wrong formula reads as grounded.
* **invented technical notation.** a chemical formula or a SMILES string the model produced but
  no source contains is pure structure, not a figure, so numeracy never sees it.

For a pharma / technical-governance corpus these ARE the content. This module is the notation
twin of `numbers_in`/`unsupported_numbers`: same discipline (one canonical spelling per value so
a correctly-preserved form is never called fabricated — see numeracy's `_canonical`), same
question ("is this token in what we retrieved"), same deliberately-narrow, deterministic scope.

It is CONSERVATIVE by mandate: a guard that fires on correct answers gets turned off. It flags a
token ONLY when it is unambiguously a unit-bearing quantity, a chemical formula, or a SMILES
string AND that token is absent from every cited span. Sub/superscript↔ASCII, spacing/case and
unit-spelling variants are normalised on BOTH sides first, so a formula the generator correctly
preserved as H₂SO₄ matches an ASCII source form H2SO4 and passes. When in doubt, it does not flag.
Plain numbers stay with numeracy; plain prose never trips it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from rag_core.numeracy import _canonical
from rag_schemas import RetrievedSpan

# Inserted by the citation assembler, not asserted by the model (mirrors numeracy). Only strips
# purely-numeric bracket groups (``[1]``, ``[1, 2]``) — a SMILES bracket atom like ``[nH]`` or
# ``[C@@H]`` is not numeric and is left intact.
_CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")

# --- sub/superscript → ASCII (the gap numeracy's ASCII-only \d leaves open) --------------------
# Both scripts collapse to plain digits: a subscript is a COUNT (H₂), a superscript is usually a
# CHARGE magnitude (SO₄²⁻). We cannot recover that distinction from an ASCII source (which writes
# both as plain digits/signs), so we flatten identically on both sides — symmetric, so it never
# false-flags — and keep the charge SIGN, which is what actually distinguishes an ion.
_SUB = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
_SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
_ASCII_FOR = "0123456789+-=()"
_SCRIPT_MAP: dict[int, str | None] = {ord(s): a for s, a in zip(_SUB + _SUP, _ASCII_FOR + _ASCII_FOR, strict=True)}
# Unicode minus / middle dot variants that show up in charges and formulae.
_SCRIPT_MAP[ord("−")] = "-"  # noqa: RUF001 - intentional: fold the unicode MINUS to ASCII hyphen
# ASCII caret marks a superscript charge in plain text ("SO4^2-"); drop it so an ASCII source
# form compares EQUAL to the unicode-superscript form the generator emits (SO4^2- ≡ SO₄²⁻).
_SCRIPT_MAP[ord("^")] = None


def _normalize_scripts(text: str) -> str:
    """Fold unicode sub/superscript digits, signs and parens to their ASCII equivalents so the
    subscripted form the generator preserves (H₂SO₄, NO₃⁻) compares EQUAL to an ASCII source
    form (H2SO4, NO3-)."""
    return text.translate(_SCRIPT_MAP)


# --- units (a number immediately followed by a unit) -------------------------------------------
# Small, pharma / governance-relevant, extensible. Each spelling maps to ONE canonical unit so
# spelling/case variants of the SAME unit compare equal, while a genuinely different unit does
# not — that is the whole point (5 g must not match 5 mg). Word spellings (percent, celsius…) are
# folded to the symbol's canonical too, so a spelled-out SOURCE never false-flags a symbol ANSWER.
_UNIT_ALIASES: dict[str, str] = {
    # mass
    "mg": "mg",
    "g": "g",
    "kg": "kg",
    "ng": "ng",
    "mcg": "mcg",
    "µg": "mcg",
    "μg": "mcg",  # the OTHER unicode mu (U+03BC vs U+00B5 above); both occur in the wild
    "ug": "mcg",
    # amount of substance
    "mol": "mol",
    "mmol": "mmol",
    "nmol": "nmol",
    # volume
    "ml": "ml",
    "l": "l",
    # activity / concentration
    "iu": "iu",
    "ppm": "ppm",
    "ppb": "ppb",
    # percent (symbol + words → one unit, so "20%" == "20 percent")
    "%": "%",
    "percent": "%",
    "pct": "%",
    "％": "%",  # noqa: RUF001 - fullwidth percent (CJK/Indic text); intentionally folded to ASCII %
    # NOTE: temperature is handled by `_iter_temperature`, not here — its many spellings ("25°C",
    # "25 °C", "25 degrees Celsius", "25 deg C") need an optional degrees/° prefix that a flat
    # alias cannot express, and getting that equivalence right is what keeps a correct answer from
    # being flagged against a spelled-out source.
}
# Denominators for a rate unit ("mg/mL", "mg/kg", "IU/day"). Kept separate: only meaningful after
# a base unit + slash.
_DENOM_ALIASES: dict[str, str] = {
    "ml": "ml",
    "l": "l",
    "kg": "kg",
    "g": "g",
    "mg": "mg",
    "day": "day",
    "d": "day",
    "h": "h",
    "hr": "h",
    "hour": "h",
    "min": "min",
}
_NUMBER = r"\d[\d,]*(?:\.\d+)?"


def _alt(keys: list[str]) -> str:
    return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))


# Matching is CASE-INSENSITIVE and the same on both sides, which is strictly safe: it can only make
# more things read as the SAME unit, never split a genuine match, so it never introduces a false
# flag (a real unit MISMATCH like g vs mg still differs regardless of case). It also spares us the
# mg-vs-Mg(=magnesium) trap — a bare unit after a number is a unit, whatever its case. The `(?<![A-
# Za-z0-9])` / `(?![A-Za-z])` boundaries stop it firing inside a larger token ("5gb", "v2").
_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + _NUMBER + r")\s?"
    r"(" + _alt(list(_UNIT_ALIASES)) + r")"
    r"(?:\s?/\s?(" + _alt(list(_DENOM_ALIASES)) + r"))?"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)


def _iter_units(text: str) -> Iterator[str]:
    for m in _UNIT_RE.finditer(text):
        unit = _UNIT_ALIASES.get(m.group(2).lower())
        if unit is None:
            continue
        token = _canonical(m.group(1)) + unit
        denom = m.group(3)
        if denom:
            token += "/" + _DENOM_ALIASES[denom.lower()]
        yield token
    yield from _iter_temperature(text)


# Temperature is its own matcher so every spelling normalises to the SAME canonical unit and a
# correct "25°C" is never flagged against a source that spelled out "25 degrees Celsius". Word
# forms (celsius/…) stand alone; the single letters C/F require an explicit degree marker (° or
# "deg[rees]") so a bare "5 C"/"5 F" — which is not a temperature — never trips it.
_DEG = r"(?:°\s*)?(?:deg(?:ree)?s?\.?\s*)?"
_TEMP_WORD_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + _NUMBER + r")\s*" + _DEG + r"(celsius|centigrade|fahrenheit)\b",
    re.IGNORECASE,
)
_TEMP_ABBR_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + _NUMBER + r")\s*(?:°\s*" + r"(?:deg(?:ree)?s?\.?\s*)?|deg(?:ree)?s?\.?\s+)(c|f)(?![A-Za-z])",
    re.IGNORECASE,
)
# Single-codepoint ℃ / ℉.
_TEMP_UNICODE_RE = re.compile(r"(?<![A-Za-z0-9])(" + _NUMBER + r")\s*(℃|℉)")


def _iter_temperature(text: str) -> Iterator[str]:
    for m in _TEMP_WORD_RE.finditer(text):
        unit = "degf" if m.group(2)[0].lower() == "f" else "degc"
        yield _canonical(m.group(1)) + unit
    for m in _TEMP_ABBR_RE.finditer(text):
        unit = "degf" if m.group(2).lower() == "f" else "degc"
        yield _canonical(m.group(1)) + unit
    for m in _TEMP_UNICODE_RE.finditer(text):
        unit = "degf" if m.group(2) == "℉" else "degc"
        yield _canonical(m.group(1)) + unit


# --- chemical formulae (element-symbol + count runs, incl. charge) -----------------------------
_ELEMENTS = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se
    Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb
    Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm
    Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
    """.split()  # noqa: SIM905 - the 118-symbol periodic table reads far better than a list literal
)
# A single element symbol is also an English word ("In", "As", "No") — require at least two
# element groups (plus a digit/charge) before a token is treated as a formula.
_MIN_FORMULA_ELEMENTS = 2
# Common technical / pharma acronyms that PARSE as element runs but are NOT formulae (IC50 → I,C50;
# CB1 → C,B1). Structurally identical to real formulae like CO2/PS2, so a denylist is the only
# reliable separator for the count<10 collisions — extensible; add new ones here.
_FORMULA_DENYLIST = frozenset(
    {
        "IC50",
        "EC50",
        "LD50",
        "ED50",
        "TD50",
        "CC50",
        "GI50",
        "LC50",  # assay / potency measures
        "CB1",
        "CB2",
        "S1P",
        "P2Y12",
        "P2X7",
        "COX1",
        "COX2",
        "PDE4",
        "PDE5",  # receptors / enzymes
        "PD1",
        "PDL1",
        "USB2",
        "USB3",
        "PS2",
        "PS3",
        "PS4",
        "PS5",
        "MP3",
        "MP4",  # targets / tech
    }
)
# All-caps letters then trailing digits with a value ≥ 10 (IC50, LD50, EC50…): a subtype/assay
# index, never a small-molecule subscript. Kept ≥10 so real diatomics (CO2, SO3, NO2) are untouched.
_ACRONYM_RE = re.compile(r"^[A-Z]{2,}(\d+)$")
_ACRONYM_MIN_INDEX = 10


def _canonical_formula(token: str) -> str | None:
    """Ordered ``Elem+count`` serialisation (+ trailing charge sign) of a chemical formula, or
    None if the token is not unambiguously a formula. Symmetric across sub/superscript↔ASCII,
    caret/parenthesised charge and parenthesis wrapping (paren chars are walked, group multipliers
    dropped) so a preserved form matches an ASCII source form; SO₄ vs SO₃ still differ because the
    COUNT differs.

    Conservative: requires ≥2 element groups AND a digit/charge, rejects single-element English
    words ("In", "As", "No") and common technical/pharma acronyms (IC50, CB1, S1P, P2Y12, USB2 …)
    that would otherwise parse as element runs.
    """
    core = _normalize_scripts(token)
    # Acronym guard (before parsing): drop known pharma/tech acronyms and the all-caps+trailing-
    # index shape. Structurally IC50/PS2 are indistinguishable from CO2/SO3, so this is the only
    # reliable separator for the low-index collisions.
    if core.upper() in _FORMULA_DENYLIST:
        return None
    acro = _ACRONYM_RE.fullmatch(core)
    if acro and int(acro.group(1)) >= _ACRONYM_MIN_INDEX:
        return None
    # Fold a parenthesised trailing charge — "(2-)" → "2-" — so SO4(2-) ≡ SO4^2- ≡ SO₄²⁻.
    core = re.sub(r"\((\d*[+-])\)$", r"\1", core)
    charge = ""
    m = re.search(r"[+-]+$", core)
    if m:
        charge = "+" if "+" in m.group(0) else "-"
        core = core[: m.start()]
    parts: list[tuple[str, str]] = []
    i, n = 0, len(core)
    has_digit = False
    while i < n:
        ch = core[i]
        if ch in "()":
            i += 1
            continue
        if ch.isdigit():
            # A bare digit here is a parenthesised-group multiplier — dropped (symmetric), but it
            # still counts as a structural digit marker.
            has_digit = True
            i += 1
            continue
        if ch.isupper():
            sym = ch
            i += 1
            if i < n and core[i].islower():
                sym += core[i]
                i += 1
            if sym not in _ELEMENTS:
                return None
            cnt = ""
            while i < n and core[i].isdigit():
                cnt += core[i]
                i += 1
            if cnt:
                has_digit = True
            parts.append((sym, cnt or "1"))
            continue
        return None
    if len(parts) < _MIN_FORMULA_ELEMENTS or not (has_digit or charge):
        return None
    return "".join(f"{s}{c}" for s, c in parts) + charge


# --- SMILES-like tokens ------------------------------------------------------------------------
# Organic-subset atoms that appear as BARE letters in SMILES (outside [brackets]). Any other bare
# letter means the token is not SMILES — a strong, high-precision guard against ordinary words.
_SMILES_BARE = set("BCNOPSFHI") | set("clr") | set("bcnops")  # cl/br via l,r; aromatic bcnops
_SMILES_CHARS = set("()[]=#@+-/\\%") | set("0123456789")
_SMILES_STRUCT = set("()=#[]@")
# Conservative gates: long enough + enough structure/atoms to be unambiguously a SMILES, not prose.
_SMILES_MIN_LEN = 8
_SMILES_MIN_STRUCT = 2
_SMILES_MIN_DIGITS = 1
_SMILES_MIN_LETTERS = 4
_BRACKET_RE = re.compile(r"\[[^\]]*\]")


def _canonical_smiles(token: str) -> str | None:
    """The token verbatim (sub/superscripts folded) if it is unambiguously a SMILES string, else
    None. Verbatim by design: SMILES canonicalisation needs a chemistry toolkit, so the check is
    "is this exact structure in a cited span". Conservative length + charset + organic-letter +
    structural-signal gates keep it off punctuated prose."""
    s = _normalize_scripts(token)
    if len(s) < _SMILES_MIN_LEN or any(c not in _SMILES_CHARS and not c.isalpha() for c in s):
        return None
    if sum(1 for c in s if c in _SMILES_STRUCT) < _SMILES_MIN_STRUCT:
        return None
    if sum(1 for c in s if c.isdigit()) < _SMILES_MIN_DIGITS or sum(1 for c in s if c.isalpha()) < _SMILES_MIN_LETTERS:
        return None
    bare = _BRACKET_RE.sub("", s)
    if any(c.isalpha() and c not in _SMILES_BARE for c in bare):
        return None
    # A SMILES-specific signal beyond a bare formula-with-parens: a bond, a bracket atom, or an
    # aromatic lowercase atom.
    if not (any(c in "=#[" for c in s) or any(c in "bcnops" for c in bare)):
        return None
    return s


# --- extraction --------------------------------------------------------------------------------
_TRAILING_PUNCT = ".,;:!?"
_QUOTE_STRIP = "\"'“”‘’"  # noqa: RUF001 - smart quotes stripped as data, never compared


def _iter_notation(text: str) -> Iterator[str]:
    """Every high-confidence notation token asserted in ``text``, in reading order.

    Order matters: SMILES first (longest, most specific), then formulae, then unit quantities.
    A whitespace token is classified as at most ONE of SMILES / formula; unit quantities are
    matched by regex over the whole text independently (a "5 mg" is never also a formula)."""
    for token in text.split():
        stripped = token.strip(_TRAILING_PUNCT).strip(_QUOTE_STRIP)
        if not stripped:
            continue
        smiles = _canonical_smiles(stripped)
        if smiles is not None:
            yield "SMILES:" + smiles
            continue
        formula = _canonical_formula(stripped)
        if formula is not None:
            yield "FORMULA:" + formula
    for unit_token in _iter_units(text):
        yield "UNIT:" + unit_token


def mask_notation_tokens(text: str) -> str:
    """Blank out detected chemical-formula and SMILES tokens (replaced with spaces) so a numeric
    grounding scan does not treat their INTERNAL digits — a formula subscript, or a SMILES
    ring-closure like the ``1`` in ``c1ccccc1`` — as an asserted figure. Unit quantities are left
    intact: their number IS a real quantity numeracy legitimately checks. Used by the notation
    checker before the numeric check, only when notation grounding is enabled (so the numeric-only
    path is untouched)."""
    out: list[str] = []
    for token in text.split():
        stripped = token.strip(_TRAILING_PUNCT).strip(_QUOTE_STRIP)
        if stripped and (_canonical_smiles(stripped) is not None or _canonical_formula(stripped) is not None):
            out.append(" " * len(token))
        else:
            out.append(token)
    return " ".join(out)


def notation_tokens_in(text: str) -> set[str]:
    """Every high-confidence notation token asserted in ``text``, canonicalised and tagged by
    kind (``UNIT:`` / ``FORMULA:`` / ``SMILES:``). Citation markers are stripped first (they are
    inserted by the platform, not asserted by the model)."""
    return set(_iter_notation(_CITATION_RE.sub(" ", text)))


def unsupported_notation(answer: str, spans: list[RetrievedSpan]) -> list[str]:
    """Notation tokens the answer asserts that appear in NO retrieved span, in reading order.

    Mirrors :func:`rag_core.numeracy.unsupported_numbers`: sources are read generously (no
    citation stripping — being liberal about what counts as supported keeps the guard aimed at
    fabrication, not formatting), and the same canonicalisation runs on both sides so a correctly
    preserved formula / unit-quantity is never flagged. Tokens keep their ``UNIT:``/``FORMULA:``/
    ``SMILES:`` tag so the caller can report which kind fired."""
    supported = {tok for s in spans for tok in _iter_notation(s.text)}
    seen: set[str] = set()
    out: list[str] = []
    for token in _iter_notation(_CITATION_RE.sub(" ", answer)):
        if token not in supported and token not in seen:
            seen.add(token)
            out.append(token)
    return out
