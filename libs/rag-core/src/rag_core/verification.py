"""Answer verification — RAG v2 Phase 10 (pure logic, no I/O).

The plan's Phase 10 splits an answer into ATOMIC CLAIMS and checks each one against the span it
CITES, three ways: entailment (NLI), a deterministic value check, and an enumeration check. This
module owns the second and third and the claim split they need. The first already exists and is
measured — ``rag_clients.nli_faithfulness`` — and is deliberately not rewritten here (§8.21).

WHY THE DETERMINISTIC HALF IS THE HALF THAT MATTERS. An NLI cross-encoder reads meaning, and a
figure is not meaning: swap "Rs. 19.44 Crores" for "Rs. 35 Crores" and the sentence still entails
beautifully, because everything that carries entailment — the subject, the verb, the unit, the
clause structure — is untouched. The platform already learned this once, which is why
:mod:`rag_core.numeracy` exists. Phase 10 tightens that lesson in two ways the plan names:

* **Per CITED QUOTE, not per retrieved span.** ``numeracy.unsupported_numbers`` asks "does this
  figure appear in ANY retrieved span". A figure that appears in some other passage of the corpus
  but NOT in the span the sentence cites is a MIS-ATTRIBUTION, and that check cannot see it. A
  ``Citation.quote`` is the span the answer says supports the claim, so it is the honest premise.
* **Values, not just bare digits.** A unit swap ("35 Crores" -> "35 lakh"), a standard swap
  ("IEC 62443-4-2" -> "IEC 62443-3-3") and a date swap all keep every digit the sentence had.

AND THE ENUMERATION CHECK, which is the plan's own worked example: a source reading "HBOM and
SBOM" answered as "SBOM" drops half the requirement while every word of the answer is supported.
Nothing in this codebase could catch that — ``coverage.py`` is span-level term overlap, which
measures whether a span was used at all, not whether a list survived it.

FALSE POSITIVES ARE THE FAILURE MODE TO FEAR, not misses. ``numeracy``'s own docstring says it:
"a guard that fires on correct answers gets turned off". So every rule here is narrow by
construction — the unit vocabulary is short and explicit rather than open-ended, standards bodies
are matched case-SENSITIVELY (so "when 50" cannot look like "EN 50"), dates are compared as
format-agnostic component sets rather than as strings, and an enumeration is only compared against
another enumeration. The test suite measures both directions: the catch rate on a seeded error set
(the plan's acceptance) AND the false-positive rate on correct answers.

NOTHING HERE DECIDES AN ANSWER'S STATUS. It reports, and the caller applies the plan's configured
action (drop the claim, or footnote it). Routing on a deterministic check needs the care the
notation guard already demonstrates — see ``faithfulness.notation_flag_band_score`` — and the
plan's own remedy for a failed claim is editorial, not a score nudge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag_core.numeracy import _canonical

__all__ = [
    "VERIFY_VERSION",
    "Claim",
    "ClaimVerdict",
    "VerificationReport",
    "enumeration_items",
    "extract_values",
    "split_claims",
    "unverified_footnote",
    "verify_claims",
    "without_unverified",
]

# The version of the VERIFICATION CONTRACT — this splitter, these rules — stored in
# ``query_log.verification`` beside each report. Same role as ``doc_profile.PROFILE_VERSION``:
# it answers "which stored verdicts did a build this one would no longer produce", which selects
# a set of rows to re-read rather than a number to admire. Bump it when a rule changes in a way
# that would change a stored verdict.
VERIFY_VERSION = 1

# --- claim splitting -----------------------------------------------------------------------------

# Sentence boundaries, Latin plus the Devanagari danda/double-danda, plus NEWLINES so a Markdown
# bullet list becomes one claim per bullet. Mirrors the splitter in
# ``rag_clients.nli_faithfulness`` deliberately — see §8.21 for why there are still two.
# Abbreviations whose full stop is NOT a sentence end. "Rs." is the one that matters and the one
# that was wrong: this corpus states money as "Rs. 19.44 Crores", so splitting after it tore every
# monetary claim in half and left the figure in a fragment too short to be kept as a claim at all
# — the verifier then checked a subject with no value in it and passed everything.
#
# THE BIAS IS DELIBERATELY TOWARDS NOT SPLITTING, including the single-initial rule, which also
# suppresses a real boundary after a sentence ending in a capital letter ("... of Annexure A. The
# bidder ..."). A missed split MERGES two claims, and a merged claim is checked against the union
# of both its citations — still correct, just coarser. A spurious split invents a fragment whose
# values are not in its own quote, which is a false positive, and a guard that fires on correct
# answers gets turned off (`numeracy`'s own words).
_ABBREV = "".join(
    rf"(?<!\b{a}\.)"
    for a in ("Rs", "No", "Sl", "Fig", "cl", "Cl", "vs", "etc", "Mr", "Mrs", "Ms", "Dr", "Sh", "Smt", "Pvt", "Ltd")
)
_SENT_RE = re.compile(r"(?<=[.!?।॥])" + _ABBREV + r"(?<![A-Z]\.)(?<!\b[ivx]\.)\s+|\n+")
# Markdown the generator adds for readability and which is not part of the claim.
_MD_LEAD = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|#{1,6})\s+")
_MD_INLINE = re.compile(r"\*\*|__|\*|_|`")
# Citation markers, as the assembler writes them: "[1]", "[2, 3]".
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A fragment with no letter in any script carries no claim — a bare "[1]" or a lone danda.
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
# Below this a "claim" is a lead-in ("Here are the requirements:"), not an assertion. Short enough
# that a real one-line requirement ("EMD is Rs. 5 lakh.") is still checked.
_MIN_CLAIM_CHARS = 12


@dataclass(frozen=True, slots=True)
class Claim:
    """One atomic assertion out of an answer, with the citations it carries.

    ``citations`` are the 1-based marker numbers as the answer WROTE them, which is how a reader
    resolves them and therefore how this module must: the assembler emits ``[n]`` for the n-th
    entry of ``GroundedAnswer.citations``."""

    text: str
    ordinal: int
    citations: tuple[int, ...] = ()


def _normalise(fragment: str) -> str:
    """One claim's text, as both :func:`split_claims` and :func:`without_unverified` must read it.

    Shared deliberately: the drop path matches a failing verdict back to the fragment it came
    from by TEXT, so if the two normalisations ever diverged the drop would silently stop matching
    and quietly keep every claim it was asked to remove. Same reasoning as
    :func:`rag_core.graph.quote_key` being the one definition of quote matching."""
    text = _MD_INLINE.sub("", _CITE_RE.sub(" ", _MD_LEAD.sub("", fragment.strip())))
    return " ".join(text.split()).strip(" .:;-—")


def split_claims(answer: str) -> tuple[Claim, ...]:
    """Split an answer into atomic claims, in order.

    One claim per sentence or Markdown bullet, with the citation markers it carries recorded and
    then STRIPPED from its text — a marker is produced by the platform's citation assembler, not
    asserted by the model, so leaving it in would make every value check see a stray number.

    A fragment with no letters, or shorter than :data:`_MIN_CLAIM_CHARS`, is dropped: a bare
    ``[1]``, a lone danda and a lead-in colon carry nothing to verify, and counting them would
    dilute the report with claims no reader would call claims.

    NOT the plan's full "split conjunction lists into items when they are enumerations". That
    reading is handled instead by the ENUMERATION CHECK (:func:`enumeration_items`), which
    compares the claim's list against the quote's list in place. Splitting "A, B and C" into three
    claims would multiply one sentence into three, each then separately NLI-scored against the
    same quote and each counted separately in a rate the plan states as a per-claim threshold —
    inflating the denominator with the same assertion three times. Keeping the sentence whole and
    comparing the LISTS is the same check with an honest count. See §8.21.
    """
    out: list[Claim] = []
    for raw in _SENT_RE.split(answer.strip()):
        citations = tuple(int(num) for group in _CITE_RE.findall(raw) for num in re.split(r"\s*,\s*", group) if num)
        text = _normalise(raw)
        if len(text) < _MIN_CLAIM_CHARS or not _HAS_LETTER.search(text):
            continue
        out.append(Claim(text=text, ordinal=len(out), citations=citations))
    return tuple(out)


# --- values --------------------------------------------------------------------------------------

_NUMBER = r"\d[\d,]*(?:\.\d+)?"

# Standards bodies, matched CASE-SENSITIVELY and followed by a designation. Case sensitivity is
# what makes the short ones usable: "EN" and "UL" as lowercase substrings occur in "when" and
# "result", and a case-insensitive rule would fire on ordinary prose. The list is the plan's own
# plus the Indian-standards bodies this corpus actually cites.
_BODIES = ("IEC", "ISO", "IEEE", "EN", "UL", "CSA", "CISPR", "FIPS", "TPM", "ANSI", "NEMA", "BIS", "IS")
_STANDARD_RE = re.compile(r"\b(" + "|".join(_BODIES) + r")[\s/-]{0,2}(\d[\w.\-/]*)")

# Quantities: a number and the unit it is stated in. DELIBERATELY SHORT. The point of carrying the
# unit is to catch a swap that leaves every digit intact ("35 Crores" -> "35 lakh"), and the cost
# of an over-broad vocabulary is a false positive on prose the sources phrase differently. An
# unknown unit is better missed here than guessed: the bare number is still checked either way.
_UNIT_CANON = {
    "cr": "crore",
    "crore": "crore",
    "crores": "crore",
    "lakh": "lakh",
    "lakhs": "lakh",
    "lac": "lakh",
    "kv": "kv",
    "mv": "mv",
    "kw": "kw",
    "mw": "mw",
    "kwh": "kwh",
    "gb": "gb",
    "tb": "tb",
    "mb": "mb",
    "gbps": "gbps",
    "mbps": "mbps",
    "eps": "eps",
    "nits": "nits",
    "%": "%",
    "percent": "%",
    "°c": "°c",
    "°f": "°f",
}
# Longest first, so "crores" is not matched as "crore" with a stray "s", and "gbps" not as "gb".
_ALPHA_UNITS = "|".join(sorted((k for k in _UNIT_CANON if k.isalpha()), key=len, reverse=True))
# The word boundary applies to the ALPHA branch only: "%" and "°C" are non-word characters, so a
# trailing `\b` after them could never match at the end of a sentence ("... 50%.").
_UNIT_RE = re.compile(
    r"(?<![\w.])(" + _NUMBER + r")\s*(%|°[CF]|(?:" + _ALPHA_UNITS + r")\b)",
    re.IGNORECASE,
)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})\b")
# The month word must BE a month. The first cut accepted any `[A-Za-z]{3,9}` and consequently read
# "IEC 62443" as "the 6th of IEC, 2443" — a reminder that an optional-group-heavy date regex will
# match almost anything if nothing in it is anchored to a real calendar.
_MONTH_ALT = "|".join(_MONTHS)
_ORDINAL = r"(?:st|nd|rd|th)?"
_WORD_DATE_RE = re.compile(
    r"\b(?:(\d{1,2})" + _ORDINAL + r"\s+(?:of\s+)?)?(" + _MONTH_ALT + r")[a-z]*\.?,?\s*"
    r"(?:(\d{1,2})" + _ORDINAL + r",?\s*)?(\d{4})\b",
    re.IGNORECASE,
)
_MAX_DAY = 31
_YEAR_FLOOR = 1000
# Inserted by the platform, never asserted: citation markers and ordered-list numbering.
_LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def extract_values(text: str) -> set[str]:
    """Every VALUE the text asserts, canonicalised to one spelling each.

    Four kinds, each the plan's: bare figures, quantities (figure + unit), standards designations,
    and dates. Each is reduced to a single canonical token so a value restated in a different but
    equivalent form is never reported as fabricated — the rule :func:`numeracy._canonical` already
    establishes for figures, applied to the other three.

    A DATE IS A COMPONENT SET, NOT A STRING. "24-09-2026", "24/09/2026" and "September 24, 2026"
    are the same date and must compare equal; "09/24/2026" is the same date under the other
    convention, and Phase 3's ``_date`` refused to guess between the two for good reason. So a
    date canonicalises to its year plus its two other components SORTED — which is identical under
    either reading and across every separator and month spelling. It cannot distinguish 24 March
    from 3 December in a day-month-ambiguous rendering, which is exactly the distinction no reader
    can make either.
    """
    stripped = _LIST_MARKER_RE.sub("", _CITE_RE.sub(" ", text))
    values: set[str] = set()
    for body, designation in _STANDARD_RE.findall(stripped):
        values.add(f"std:{body.lower()} {designation.lower().rstrip('.,;:')}")
    # Standards out of the way FIRST: a designation is ONE identifier, and leaving "62443-4-2" in
    # the text for the date and figure rules to pick over turns one standard into a date-shaped
    # token and three loose numbers.
    masked = _STANDARD_RE.sub(" ", stripped)
    for number, unit in _UNIT_RE.findall(masked):
        canon = _UNIT_CANON.get(unit.lower())
        if canon:
            values.add(f"qty:{_canonical(number)} {canon}")
    values |= {f"date:{d}" for d in _dates(masked)}
    # Bare figures last, over text with the DATE and STANDARD tokens masked out. Masking is what
    # makes the composite rules safe: "24-09-2026" and "September 24, 2026" are the same date and
    # produce the same `date:` token, but their bare digits differ ("09" is in one and not the
    # other), so an unmasked sweep would report `num:09` missing from a quote that states the very
    # same date in words. Same for a standard's designation: "62443-4-2" is one identifier, not
    # three figures. A QUANTITY deliberately keeps its number — "35 Crores" vs "35 lakh" is caught
    # by the `qty:` token, and the shared `num:35` costs nothing.
    bare = _WORD_DATE_RE.sub(" ", _NUMERIC_DATE_RE.sub(" ", masked))
    values |= {f"num:{_canonical(m.group())}" for m in re.finditer(_NUMBER, bare)}
    return values


def _supported(value: str, premise: set[str]) -> bool:
    """Is one of the claim's values present in the premise's?

    Exact for everything except a STANDARD, where the claim's designation may be a PREFIX of the
    quote's. An answer writing "IEC 62443" where its quote says "IEC 62443-4-2" is less specific,
    not wrong, and reporting it would fire on correct answers. The reverse — "IEC 62443-3-3"
    against a quote stating "IEC 62443-4-2" — is not a prefix and is exactly the swap this check
    exists to catch, so the asymmetry is the point rather than a loophole."""
    if value in premise:
        return True
    if value.startswith("std:"):
        return any(p.startswith(value) for p in premise)
    return False


def _dates(text: str) -> set[str]:
    """Date-shaped tokens, each as ``"<year>-<lo>-<hi>"`` over its two non-year components."""
    out: set[str] = set()
    for a, b, c in _NUMERIC_DATE_RE.findall(text):
        parts = [int(a), int(b), int(c)]
        year_at = [i for i, p in enumerate(parts) if p >= _YEAR_FLOOR]
        if len(year_at) != 1:
            # Not a date this module is willing to read: "1-2-3" is a clause number and
            # "2026-2027" with a third part is a range. Refusing beats guessing.
            continue
        year = parts[year_at[0]]
        rest = sorted(p for i, p in enumerate(parts) if i != year_at[0])
        if all(p <= _MAX_DAY for p in rest):
            out.add(f"{year}-{rest[0]}-{rest[1]}")
    for day_before, word, day_after, year in _WORD_DATE_RE.findall(text):
        month = _MONTHS.get(word[:3].lower())
        day = day_before or day_after
        if month is None or not day:
            continue
        rest = sorted((month, int(day)))
        out.add(f"{int(year)}-{rest[0]}-{rest[1]}")
    return out


# --- enumerations --------------------------------------------------------------------------------

# "A, B and C" / "A, B, or C" / "both A and B". The item separators only — the items themselves
# are whatever sits between them.
_ENUM_SPLIT = re.compile(r"\s*,\s*(?:and|or)?\s*|\s+(?:and|or)\s+", re.IGNORECASE)
# An "item" longer than this is a clause, not a list entry, and treating it as one would compare
# two paraphrases of the same sentence and call the difference an omission. The WORD bound is the
# load-bearing one: "the contractor shall provide the EMD" is 36 characters and would pass a
# character bound, while a real list item in a requirements document is a short noun phrase
# ("HBOM", "data diode", "NG-SIEM"). Measured against the false-positive fixtures, the word bound
# is what stops ordinary prose joined by "and" from being read as a two-item list.
_MAX_ITEM_CHARS = 48
_MAX_ITEM_WORDS = 5
_MIN_ENUM_ITEMS = 2
_ITEM_STRIP = " \t.;:!?\"'()[]“”‘’"  # noqa: RUF001 - smart quotes stripped as data, never compared


# Words that are never a list ITEM: articles, prepositions, auxiliaries, and the verbs a
# requirements document joins its lists with. The verbs are the load-bearing half and the reason
# this list is domain-shaped rather than a generic stop list — "the vendor shall supply HBOM and
# SBOM" splits into "the vendor shall supply HBOM" / "SBOM for every component", and only knowing
# that "supply" is not part of the item recovers "HBOM" from the first half. Measured against the
# false-positive fixtures: without the verbs, that sentence yields the items "the vendor shall
# supply hbom" and "sbom for every component", which match nothing and lose the check entirely.
_STOP = frozenset(
    (
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "within",
        "under",
        "per",
        "as",
        "such",
        "other",
        "this",
        "that",
        "these",
        "those",
        "its",
        "their",
        "his",
        "her",
        "our",
        "your",
        "it",
        "they",
        "them",
        "shall",
        "must",
        "should",
        "will",
        "would",
        "may",
        "can",
        "could",
        "be",
        "is",
        "are",
        "was",
        "were",
        "been",
        "being",
        "am",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "supply",
        "supplied",
        "provide",
        "provided",
        "submit",
        "submitted",
        "furnish",
        "furnished",
        "include",
        "included",
        "including",
        "comply",
        "complying",
        "meet",
        "meets",
        "deliver",
        "delivered",
        "maintain",
        "maintained",
        "ensure",
        "ensured",
        "support",
        "required",
        "require",
        "requires",
        "requiring",
        "stated",
        "specified",
        "mentioned",
        "given",
        "both",
        "either",
        "neither",
        "all",
        "any",
        "each",
        "every",
        "some",
        "no",
        "not",
        "also",
        "along",
        "together",
    )
)

_WORD_RE = re.compile(r"[\w./-]+")


def _leading_run(tokens: list[str]) -> list[str]:
    """The content words at the START of a segment, bounded. Empty if it opens on a stop word."""
    out: list[str] = []
    for token in tokens:
        if token in _STOP or len(out) == _MAX_ITEM_WORDS:
            break
        out.append(token)
    return out


def _trailing_run(tokens: list[str]) -> list[str]:
    """The content words at the END of a segment, bounded. Empty if it closes on a stop word."""
    return list(reversed(_leading_run(list(reversed(tokens)))))


def enumeration_items(text: str) -> tuple[str, ...]:
    """The items of the longest conjunction list in ``text``, casefolded; ``()`` if there is none.

    Scoped to ONE clause — the span between ``:``/``;``/sentence boundaries — because a sentence
    can hold a list and then go on to say something else, and running the splitter over the whole
    sentence turns the tail into a phantom item.

    ANCHORED ON THE SEPARATORS, not on the segments between them, which is the correction that
    made this work at all. A list item sits NEXT TO its separator: in "the vendor shall supply
    HBOM and SBOM for every component", the item before "and" is at the END of its segment and
    the item after it is at the START of the next. So the first segment contributes its trailing
    content run, the last its leading one, and a middle segment (bounded by separators on both
    sides) its leading one. Taking whole segments instead produced "the vendor shall supply hbom"
    and "sbom for every component" — two items that match nothing, which is how the plan's own
    worked example silently failed.

    EVERY SEGMENT MUST YIELD AN ITEM or there is no list. That is what keeps ordinary prose out:
    "the bidder shall submit the EMD and the technical bid shall be sealed" splits into two
    clauses, the second of which opens on "the" and so contributes nothing, and the candidate is
    discarded rather than read as a two-item list.

    When that strict reading finds nothing, :func:`_acronym_items` is tried, which relaxes the
    every-segment rule in the one place the evidence is unambiguous. KNOWN MISS: a list whose
    items are article-prefixed and NOT acronyms ("the first schedule, the second schedule and the
    annexure") is not read as a list at all. That is a deliberate miss — the relaxation that would
    catch it is the one that misreads the prose sentence above, and a false positive here costs
    more than a miss: it footnotes a correct claim, and a guard that fires on correct answers gets
    turned off.
    """
    best: tuple[str, ...] = ()
    for clause in re.split(r"[:;]|\.\s", text):
        parts = [p for p in _ENUM_SPLIT.split(clause.strip()) if p.strip()]
        if len(parts) < _MIN_ENUM_ITEMS:
            continue
        items = _strict_items(parts) or _acronym_items(parts)
        if len(items) >= _MIN_ENUM_ITEMS and len(items) > len(best):
            best = items
    return best


def _strict_items(parts: list[str]) -> tuple[str, ...]:
    """Items when EVERY segment yields one. The conservative path — see
    :func:`enumeration_items`."""
    items: list[str] = []
    for index, segment in enumerate(parts):
        tokens = _WORD_RE.findall(segment.casefold())
        run = _trailing_run(tokens) if index == 0 else _leading_run(tokens)
        item = " ".join(run).strip(_ITEM_STRIP)
        if not item or len(item) > _MAX_ITEM_CHARS:
            return ()
        items.append(item)
    return tuple(items)


# An ACRONYM head: two or more capitals, optionally with digits and internal hyphens/slashes.
# "HBOM", "SBOM", "NG-SIEM", "NMS/ITSM", "HCI-1". Matched on the ORIGINAL casing, which is the
# whole signal — this is what a requirements document's list items actually look like.
_ACRONYM_RE = re.compile(r"[A-Z][A-Z0-9]*(?:[-/][A-Z0-9]+)*")
_MIN_ACRONYM_CHARS = 2


def _acronym_items(parts: list[str]) -> tuple[str, ...]:
    """Items when at least two segments have an ACRONYM next to the separator.

    The fallback for the one shape the strict path cannot read: "the HBOM, the SBOM and the
    compliance matrix", where every item carries an article and so every segment opens on a stop
    word. Skipping leading articles in the strict path would fix that sentence and break a more
    important one — "the bidder shall submit the EMD and the technical bid shall be sealed" would
    become the two-item list ["emd", "technical bid"], and a claim mentioning only the EMD would
    be reported as having dropped an item it never owed. So the relaxation is granted only where
    the evidence is unambiguous: an all-capitals token adjacent to the separator.

    Segments WITHOUT an acronym are skipped rather than failing the whole list, which is what
    makes the example above work; two acronyms are still required, so a lone "EMD" beside ordinary
    prose is not a list."""
    found: list[str] = []
    for index, segment in enumerate(parts):
        candidates = _ACRONYM_RE.findall(segment)
        if not candidates:
            continue
        head = candidates[-1] if index == 0 else candidates[0]
        if len(head) >= _MIN_ACRONYM_CHARS:
            found.append(head.casefold())
    return tuple(found) if len(found) >= _MIN_ENUM_ITEMS else ()


def _omitted_items(claim: str, quote: str) -> tuple[str, ...]:
    """Items the QUOTE lists that the claim dropped — the plan's "HBOM and SBOM" -> "SBOM" case.

    Fires only when the quote itself contains a list and the claim mentions a NON-EMPTY STRICT
    SUBSET of it. Both halves of that condition are what keeps it quiet: a claim mentioning none
    of the quote's items is about something else in the quote, and a claim mentioning all of them
    dropped nothing. So the only thing reported is a claim that answered a list PARTIALLY, which
    is the defect, not a style.
    """
    listed = enumeration_items(quote)
    if len(listed) < _MIN_ENUM_ITEMS:
        return ()
    folded = claim.casefold()
    present = [i for i in listed if i in folded]
    if not present or len(present) == len(listed):
        return ()
    return tuple(i for i in listed if i not in present)


# --- the verdict ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """What the deterministic checks made of one claim.

    ``ok`` is the verdict a caller acts on. ``uncited`` is reported but never on its own a
    failure: a lead-in or a connective legitimately carries no marker, and the claim was still
    checked — against the union of every quote, which is the generous reading."""

    claim: str
    ordinal: int
    ok: bool
    missing_values: tuple[str, ...] = ()
    omitted_items: tuple[str, ...] = ()
    uncited: bool = False

    @property
    def reasons(self) -> tuple[str, ...]:
        """Why this claim failed, as stable slugs for a log line or a dashboard."""
        out: list[str] = []
        if self.missing_values:
            out.append("value_not_in_quote")
        if self.omitted_items:
            out.append("enumeration_omitted")
        return tuple(out)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Every claim's verdict, plus the counts the plan's acceptance is stated in."""

    verdicts: tuple[ClaimVerdict, ...] = ()
    verify_version: int = VERIFY_VERSION

    @property
    def checked(self) -> int:
        return len(self.verdicts)

    @property
    def failures(self) -> tuple[ClaimVerdict, ...]:
        return tuple(v for v in self.verdicts if not v.ok)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_json(self) -> str:
        """The shape stored in ``query_log.verification`` (plan §10 item 6).

        Stores the FAILURES in full and the passes as a count. A row exists to answer "what did
        this answer get wrong", and recording every passing claim verbatim would store the whole
        answer a second time in the column beside the one that already holds it."""
        return json.dumps(
            {
                "verify_version": self.verify_version,
                "checked": self.checked,
                "failed": len(self.failures),
                "failures": [
                    {
                        "claim": v.claim,
                        "ordinal": v.ordinal,
                        "reasons": list(v.reasons),
                        "missing_values": list(v.missing_values),
                        "omitted_items": list(v.omitted_items),
                        "uncited": v.uncited,
                    }
                    for v in self.failures
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def verify_claims(answer: str, quotes: list[str]) -> VerificationReport:
    """Verify every claim in ``answer`` against the quote it cites (plan §10 items 1 and 3).

    ``quotes`` are the supporting spans in CITATION ORDER — ``[c.quote for c in
    answer.citations]`` — so a claim's ``[n]`` marker indexes straight into it.

    A claim is checked against ITS OWN quote. That is the whole point of the phase: a figure that
    appears somewhere else in the corpus but not in the span the sentence cites is a
    mis-attribution, and the pre-Phase-10 check (``numeracy.unsupported_numbers``, against the
    union of all retrieved spans) cannot see it.

    Two deliberate generosities, each closing a false-positive path that would otherwise get this
    guard switched off:

    * a claim carrying NO marker is checked against the union of every quote, because a lead-in or
      a connective legitimately has none and punishing the generator's formatting is not the job;
    * a marker pointing outside ``quotes`` falls back to the same union rather than failing the
      claim, because an out-of-range marker is a generator formatting fault, not a false claim.

    With no quotes at all nothing is checked and the report is empty — an abstain has no citations,
    and "every claim failed" would be a false reading of "there was nothing to check against".
    """
    if not quotes:
        return VerificationReport()
    union = "\n".join(quotes)
    union_values = extract_values(union)
    verdicts: list[ClaimVerdict] = []
    for claim in split_claims(answer):
        cited = [quotes[n - 1] for n in claim.citations if 1 <= n <= len(quotes)]
        uncited = not cited
        premise = "\n".join(cited) if cited else union
        premise_values = extract_values(premise) if cited else union_values
        missing = tuple(sorted(v for v in extract_values(claim.text) if not _supported(v, premise_values)))
        omitted = _omitted_items(claim.text, premise)
        verdicts.append(
            ClaimVerdict(
                claim=claim.text,
                ordinal=claim.ordinal,
                ok=not missing and not omitted,
                missing_values=missing,
                omitted_items=omitted,
                uncited=uncited,
            )
        )
    return VerificationReport(verdicts=tuple(verdicts))


# --- the plan's two configured actions -----------------------------------------------------------

_FOOTNOTE_HEADING = "Could not be verified from the cited sources:"


def unverified_footnote(report: VerificationReport) -> str:
    """The plan's "could not be verified" footnote, or ``""`` when nothing failed.

    Appended to the answer rather than replacing anything, so the reader still has what the model
    said AND the fact that part of it does not check out. That is the more conservative of the
    plan's two actions and the default for exactly that reason: dropping a claim that failed on a
    verifier false positive removes real content, while footnoting one is visible and reversible
    by the reader's own judgement."""
    failures = report.failures
    if not failures:
        return ""
    lines = "\n".join(f"- {v.claim}" for v in failures)
    return f"\n\n_{_FOOTNOTE_HEADING}_\n{lines}"


def without_unverified(answer: str, report: VerificationReport) -> str:
    """The plan's "drop it" action: the answer minus every claim that failed, plus the footnote.

    THE FOOTNOTE STAYS EVEN WHEN THE CLAIM IS DROPPED, which is the plan's own wording ("drop it
    and add it to a 'could not be verified' footnote"). A silent deletion would leave the reader
    with a shorter answer and no way to know something was removed, which is the one outcome worse
    than either keeping or removing it.

    EVERY CLAIM FAILING RETURNS THE ANSWER UNCHANGED, with the footnote. An answer stripped to
    nothing would be an empty GROUNDED answer — a contract violation
    (:class:`rag_schemas.GroundedAnswer` requires citations for GROUNDED and forbids them for
    ABSTAINED, and neither describes "grounded but blank"). A verifier that disagrees with every
    claim is far more likely to be wrong about the answer than the answer is to be entirely
    fabricated, so it reports instead of erasing.
    """
    failures = report.failures
    if not failures:
        return answer
    footnote = unverified_footnote(report)
    if len(failures) == report.checked:
        return answer + footnote
    dropped = {v.claim for v in failures}
    kept = [raw.strip() for raw in _SENT_RE.split(answer.strip()) if _normalise(raw) not in dropped]
    return "\n".join(k for k in kept if k) + footnote
