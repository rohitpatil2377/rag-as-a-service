"""The RAG v2 gold-set format and its metrics — Phase 11 (pure logic, no I/O except the loader).

The plan's instruction is "extend the existing harness; don't build a new one", so this sits
BESIDE :mod:`svc_guardrails_eval.evaluation` rather than replacing it. That module is the v1 CI
release gate and its metrics stay exactly as they are: a build that passes it today must keep
passing it. What this adds is the shape the plan's §11 asks for and the v1 format cannot express.

WHAT THE V1 FORMAT CANNOT SAY. ``GoldenItem`` is one question, one expected answer, some expected
doc ids. That is a LOOKUP gold set, and it is the only kind this repository has had. The plan's
Phase-11 format adds three things, each because a v2 answer can fail in a way a lookup item
cannot see:

* ``intent`` — so a suite can report accuracy per intent, and so an aggregate question is not
  graded as though a single sentence should have answered it.
* ``checklist`` — an aggregate answer is judged on WHAT IT COVERS, tier by tier. "Give me the
  compliance details" has no single expected answer; it has twenty items that should be there,
  some of which are optional. Scoring it with a containment check against one expected string
  measures nothing.
* ``supersessions`` — the one case where being right is not enough: an amended document states an
  old value and a new one, and an answer that quotes the old figure is grounded, cited, faithful
  and wrong. Nothing in a lookup gold set can express "and it must NOT say this".

METRICS ARE ARITHMETIC, deliberately, for the reason ``evaluation`` already states: a gate that
needs an LLM to decide whether the build passes inherits that model's variance, and a threshold
against it means nothing. ``faithfulness`` is carried THROUGH from the answer rather than
recomputed here, so this module stays judge-free.

VALUE ACCURACY REUSES PHASE 10. A checklist's ``must_contain`` holds "values/codes", and whether
an answer states the RIGHT value is exactly the question
:func:`rag_core.verification.extract_values` answers — canonicalising "Rs. 50,000" and
"50000", "24.0" and "24", "24-09-2026" and "September 24, 2026" to one token each. Scoring with
raw string containment instead would mark a correct answer wrong for writing a figure differently,
which is the failure mode that gets a gate switched off.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_core.router import ROUTER_INTENTS
from rag_core.verification import extract_values
from rag_schemas import AnswerStatus, GroundedAnswer

__all__ = [
    "INTENTS",
    "SUITE_VERSION",
    "ChecklistItem",
    "GoldQuery",
    "GoldSuite",
    "QueryScore",
    "SuiteFloors",
    "SuiteReport",
    "Supersession",
    "extract_values",
    "fold_text",
    "load_suite",
    "parse_suite",
    "score_query",
    "score_suite",
    "strip_emphasis",
]

# --- markdown emphasis, for the EVALUATION representation only ------------------------------------
#
# WIRED NOWHERE YET, deliberately. `_checklist_scores` still passes the raw answer to
# `extract_values`, so scoring is byte-identical to before this function existed -- verified by
# re-scoring stored answers. Wiring it is a separate change, because a paired benchmark is running
# and a grader that changes between its cycles is not a grader.
#
# WHY IT IS NEEDED. `_checklist_scores` reads the answer two ways on consecutive lines:
# `contains_cue(folded, ...)` and `extract_values(raw_answer)`. `fold_text` was given
# markdown-folding precisely so a bolded figure would still match; its sibling kept reading raw
# text. `extract_values` builds a `qty:<n> <unit>` key from the unit FOLLOWING the number, so
# emphasis between them leaves only a bare `num:`:
#
#     '15 crores'       -> ['num:15', 'qty:15 crore']
#     '₹**15** crores'  -> ['num:15']
#
# And the answer prompt asks for exactly that shape (bold every number, keep the unit), so the
# presentation contract and the grader are in direct conflict. Measured on a real run, this alone
# moved value accuracy 58.8% -> 52.9% on an answer whose value was correct.
#
# WHY NOT `fold_text`, AND WHY NOT A UNION. `extract_values` has a case-sensitive `std:` key that
# types a technical identifier as ONE thing. Folding lowercases, so `IEC 62443-4-2` dissolves into
# `num:62443`, `num:4`, `num:2` -- measured on a real answer, folding destroyed all six standards
# in it while inventing their digits as independent numeric evidence, which is a false POSITIVE
# through the door built to stop a false negative. A raw/folded union keeps the `std:` keys and
# still manufactures those digits, because a union only adds.
#
# ASTERISKS ONLY, AND UNDERSCORES ARE NEVER TOUCHED. This is not a markdown parser: it removes the
# syntax THIS APPLICATION'S PROMPT generates (`**bold**`) and nothing else. A version that also
# handled `_` stripped `__init__` to `init` -- correct about CommonMark, wrong about a Python
# dunder -- so `_` is left alone, which removes the dilemma at no cost. A pair must hug non-space
# content, which keeps `5 * 10` and `a * b * c` intact, and escaped delimiters are protected first.
_ESCAPED_STAR = "\\*"
_STAR_HOLDER = "\x00"
_STRONG_STAR = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.S)
_EM_STAR = re.compile(r"(?<!\*)\*(?=\S)([^*]+?)(?<=\S)\*(?!\*)")


def strip_emphasis(text: str) -> str:
    """``text`` with paired asterisk emphasis removed and every other character preserved.

    The result is an EVALUATION representation. The raw answer stays authoritative for audit and
    for reproducing any score already recorded against it."""
    out = text.replace(_ESCAPED_STAR, _STAR_HOLDER)
    for pattern in (_STRONG_STAR, _EM_STAR):
        prev = None
        while prev != out:  # nested emphasis unwraps outside-in
            prev = out
            out = pattern.sub(r"\1", out)
    return out.replace(_STAR_HOLDER, _ESCAPED_STAR)


# The version of the GOLD-SET CONTRACT — this schema, these metrics. Carried in a suite file and
# on a report so a stored baseline can be told apart from one this build would no longer produce.
# Same role as ``doc_profile.PROFILE_VERSION``.
SUITE_VERSION = 1

# The plan's Phase-7 router contract, which the gold set labels against. IMPORTED FROM THE ROUTER
# rather than restated here: when this module was written Phase 7 did not exist and the set was a
# local literal, which meant a gold file could validate against a vocabulary the router no longer
# emitted. Phase 7 made the router the canonical definition, so adding a seventh intent now
# reaches the gold loader by construction. A label outside the set is a typo and is refused at
# load.
INTENTS = ROUTER_INTENTS
_TIERS = frozenset({"must", "should"})


# Every dash Unicode distinguishes and a reader does not, folded to ASCII "-". NFKC alone does
# NOT do this, which is a thing worth stating because `scripts/eval_gate.py` has claimed since it
# was written that its folding makes "NO₃⁻" match "NO3-" — and it did not. NFKC maps the
# SUPERSCRIPT MINUS (U+207B) to the MINUS SIGN (U+2212), not to the hyphen, so a chemistry gold
# answer typed with an ASCII hyphen scored as a miss against an extraction that preserved the real
# character. Caught by this module's own test asserting the documented behaviour.
_DASHES = str.maketrans(dict.fromkeys("\u2212\u2010\u2011\u2012\u2013\u2014\u2015\u2043\ufe63\uff0d", "-"))


#: Markdown emphasis the model wraps values in. The answers ARE markdown — the console renders
#: them — and the grader compared against the raw source, so a bolded value was invisible to it.
#: Measured: a correct answer reading "minimum **6** months" scored a miss against a gold of
#: "6 (six) months", because `**6** months` does not contain `6 months`. Asterisks and backticks
#: only; underscores are left alone because snake_case identifiers are real content.
_EMPHASIS = re.compile(r"[*`]+")

#: A digit-group separator between digits: the comma in "6,000". Removed before comparison
#: because "6,000" and "6000" are the same number and a grader that disagrees is wrong.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")

#: A parenthetical that merely re-spells the number before it — "6 (six) months", "3 (three)
#: years". Dropped for the same reason: it is a legal-drafting flourish, not a different value,
#: and an answer that omits it has not said anything different.
_NUMBER_GLOSS = re.compile(r"(?<=\d)\s*\((?:[a-z]+(?:[\s-][a-z]+)*)\)")


def fold_text(text: str) -> str:
    """Case-insensitive, NFKC-normalised, dash-unified, whitespace-collapsed, number-normalised.

    "NO₃⁻" matches "NO3-" and "A-4" matches "a-4", because an extraction that lost a subscript or
    kept a typographic minus should not read as a wrong answer. THE ONE DEFINITION for both this
    module and ``scripts/eval_gate.py``, which now calls it — two normalisations would mean a
    question the gate scores one way and the suite scores another, the hazard
    ``rag_core.graph.quote_key`` was promoted to public to avoid.

    NUMBERS ARE FOLDED TOO, and that was added because the grader was failing correct answers.
    Both remaining AEGCL lookup "failures" turned out to be right: traced through the real
    pipeline, the model answered "sustained 6,000 EPS" where the gold said ``6000``, and
    "minimum **6** months" where the gold said ``6 (six) months``. Each was a verbatim-correct
    reading of the document that a substring check could not see. A thousands separator, a
    spelled-out gloss and a pair of bold markers are three renderings of one value, exactly like
    the subscript and the dash this already folded."""
    folded = unicodedata.normalize("NFKC", text).translate(_DASHES).casefold()
    folded = _EMPHASIS.sub("", folded)
    folded = _THOUSANDS.sub("", folded)
    folded = _NUMBER_GLOSS.sub("", folded)
    return " ".join(folded.split())


#: True when a cue carries a figure, which is what makes a bare substring test unsafe for it.
_HAS_DIGIT = re.compile(r"\d")

#: Guards that stop a figure matching INSIDE a longer one. Applied only to a cue containing a
#: digit, and only at the end that is itself a digit — a cue like "75 TB" needs protecting at the
#: front ("175 TB") and not the back, and "UL 62368" the reverse.
#:
#: `(?!\.\d)` rather than `(?!\.)` because a figure at the end of a sentence is followed by a full
#: stop and must still match: "6000." is the value, "6000.5" is not.
_LEFT_GUARD = r"(?<!\d)(?<!\d\.)"
_RIGHT_GUARD = r"(?!\d)(?!\.\d)"


def contains_cue(folded_haystack: str, cue: str) -> bool:
    """Is ``cue`` present in already-folded text, without matching inside a longer figure?

    TEXT CUES STAY SUBSTRING MATCHES. "Video Wall", "PeopleCert" and "Data Diode" are assertions
    that some wording appears, they work correctly that way today, and making them
    boundary-sensitive would change the meaning of every checklist item to fix a problem none of
    them has.

    A CUE WITH A FIGURE IN IT IS DIFFERENT, because substring matching cannot tell "600" from the
    first three characters of "6000", and an answer stating the wrong capacity would score as
    stating the right one. So a numeric cue is matched with boundaries at whichever of its ends
    is a digit:

        "600"           does NOT match inside "6000"
        "19.44"         does NOT match inside "119.44" or "19.447"
        "6000"          DOES match "6,000" (the separator is folded away) and "6000."
        "75 TB"         does NOT match "175 TB" — the number keeps its unit
        "UL 62368"      does NOT match "UL 623689"
        "IEC 62443-4-2" does NOT match "IEC 62443-4-25"

    It deliberately does NOT try to equate "15 crores" with "₹15 crore". That would be a
    semantic equivalence rule written to make one observed phrasing pass, which is how three gold
    entries on this branch came to be wrong in the first place. If singular/plural drift recurs
    across document classes it is evidence for such a rule; one flaky item on one corpus is not.
    """
    needle = fold_text(cue)
    if not needle:
        return False
    if not _HAS_DIGIT.search(needle):
        return needle in folded_haystack
    left = _LEFT_GUARD if needle[0].isdigit() else ""
    right = _RIGHT_GUARD if needle[-1].isdigit() else ""
    return re.search(left + re.escape(needle) + right, folded_haystack) is not None


# --- the format ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChecklistItem:
    """One item an AGGREGATE answer should cover (plan §11).

    ``must_contain`` is every value or code that has to appear for the item to count as covered.
    ``tier`` separates what the answer OWES from what would merely be good: the plan asks for
    must-tier and should-tier recall reported separately, because averaging them lets a build
    trade a required item for two optional ones and call it progress."""

    id: str
    facet: str
    must_contain: tuple[str, ...]
    tier: str = "must"


@dataclass(frozen=True, slots=True)
class Supersession:
    """A field an amended document changed: the answer must state ``new`` and must NOT state ``old``.

    The only assertion in this format that is NEGATIVE, and the reason it exists: an answer
    quoting the superseded figure is grounded, cited and faithful — every other metric passes it.
    Only knowing which value was replaced catches it."""

    field: str
    old: str
    new: str


@dataclass(frozen=True, slots=True)
class GoldQuery:
    """One gold query. Either ``expected_answer`` (lookup) or ``checklist`` (aggregate), or
    ``expect_abstain``.

    ALL THREE MAY BE ABSENT TOGETHER and that is legal: a suite under construction can carry a
    query with its intent labelled and nothing else, which scores nothing and is counted as
    unscored rather than as a failure. That is the same discipline
    ``evaluation.GoldenItem.expected_chunk_ids`` already follows — "an item without it is EXCLUDED
    rather than counted as a miss, so adding a question never makes the score fall"."""

    query: str
    intent: str = "lookup"
    expected_answer: str | None = None
    checklist: tuple[ChecklistItem, ...] = ()
    supersessions: tuple[Supersession, ...] = ()
    expect_abstain: bool = False
    expected_doc_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GoldSuite:
    """A document class's gold set. ``doc_type`` is what the plan's "corpus diversity" counts."""

    name: str
    doc_type: str
    queries: tuple[GoldQuery, ...] = ()
    suite_version: int = SUITE_VERSION
    notes: str = ""


class SuiteFormatError(ValueError):
    """A gold file this module refuses to read. Raised rather than skipped: a silently dropped
    gold query makes a suite smaller and its scores better, which is the one failure mode a
    regression gate must never have."""


def parse_suite(data: Any, *, name: str) -> GoldSuite:
    """Validate a parsed gold document into a :class:`GoldSuite`. Pure; raises on anything
    unusable.

    STRICT, unlike most parsing in this codebase. Elsewhere a malformed entry is dropped and
    reported, because the alternative is taking an ingest or a query down over one bad row. Here
    the opposite holds: this is a REGRESSION GATE, and a dropped query means the suite silently
    got easier. So an unknown intent, an unknown tier, a checklist item with no ``must_contain``
    or a query with both an expected answer and a checklist is an error the author has to fix."""
    if not isinstance(data, dict):
        raise SuiteFormatError(f"{name}: the gold file must be a mapping")
    version = data.get("suite_version", SUITE_VERSION)
    if version != SUITE_VERSION:
        raise SuiteFormatError(f"{name}: suite_version {version!r} is not {SUITE_VERSION}")
    doc_type = str(data.get("doc_type") or "").strip()
    if not doc_type:
        raise SuiteFormatError(f"{name}: doc_type is required — it is what 'corpus diversity' counts")
    raw_queries = data.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise SuiteFormatError(f"{name}: queries must be a non-empty list")
    return GoldSuite(
        name=name,
        doc_type=doc_type,
        notes=str(data.get("notes") or ""),
        queries=tuple(_query(item, name=name, index=i) for i, item in enumerate(raw_queries)),
    )


def _query(item: Any, *, name: str, index: int) -> GoldQuery:
    where = f"{name}: queries[{index}]"
    if not isinstance(item, dict):
        raise SuiteFormatError(f"{where} must be a mapping")
    query = str(item.get("query") or "").strip()
    if not query:
        raise SuiteFormatError(f"{where}: query is required")
    intent = str(item.get("intent") or "lookup").strip().lower()
    if intent not in INTENTS:
        raise SuiteFormatError(f"{where}: intent {intent!r} is not one of {sorted(INTENTS)}")
    expected = item.get("expected_answer")
    checklist = tuple(_checklist_item(c, where=where) for c in item.get("checklist") or ())
    if expected and checklist:
        raise SuiteFormatError(f"{where}: give expected_answer OR checklist, not both")
    return GoldQuery(
        query=query,
        intent=intent,
        expected_answer=str(expected).strip() if expected else None,
        checklist=checklist,
        supersessions=tuple(_supersession(s, where=where) for s in item.get("supersessions") or ()),
        expect_abstain=bool(item.get("expect_abstain", False)),
        expected_doc_ids=tuple(str(d) for d in item.get("expected_doc_ids") or ()),
    )


def _checklist_item(raw: Any, *, where: str) -> ChecklistItem:
    if not isinstance(raw, dict):
        raise SuiteFormatError(f"{where}: each checklist entry must be a mapping")
    must = tuple(str(v).strip() for v in raw.get("must_contain") or () if str(v).strip())
    if not must:
        raise SuiteFormatError(f"{where}: checklist entry {raw.get('id')!r} has no must_contain")
    tier = str(raw.get("tier") or "must").strip().lower()
    if tier not in _TIERS:
        raise SuiteFormatError(f"{where}: tier {tier!r} is not one of {sorted(_TIERS)}")
    return ChecklistItem(
        id=str(raw.get("id") or "").strip() or must[0][:40],
        facet=str(raw.get("facet") or "").strip(),
        must_contain=must,
        tier=tier,
    )


def _supersession(raw: Any, *, where: str) -> Supersession:
    if not isinstance(raw, dict):
        raise SuiteFormatError(f"{where}: each supersession must be a mapping")
    old, new = str(raw.get("old") or "").strip(), str(raw.get("new") or "").strip()
    if not old or not new:
        raise SuiteFormatError(f"{where}: a supersession needs both old and new")
    return Supersession(field=str(raw.get("field") or "").strip(), old=old, new=new)


def load_suite(path: str | Path) -> GoldSuite:
    """Load one gold file. ``.yaml``/``.yml`` or ``.json``.

    The plan writes the path as ``eval/gold/<doc>.yaml`` and YAML is the right format for this
    file in particular: a gold set is hand-edited and reviewed, and a reviewer annotating why an
    item is ``should`` rather than ``must`` needs comments, which JSON has no way to carry. JSON
    is accepted too so the existing sets can be migrated without a rewrite."""
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if file.suffix in {".yaml", ".yml"}:
        import yaml  # noqa: PLC0415 - optional at import time; see the module's dependency note

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return parse_suite(data, name=file.name)


# --- the metrics ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryScore:
    """One gold query's outcome. ``None`` everywhere the gold set did not ask for that metric.

    None rather than 0.0 throughout, because a metric the suite did not specify is not a failure:
    an aggregate query with no checklist yet has no item recall, and scoring it as 0.0 would make
    a half-written suite look like a broken build."""

    query: str
    intent: str
    answered: bool
    abstain_ok: bool
    lookup_hit: bool | None = None
    item_recall_must: float | None = None
    item_recall_should: float | None = None
    missed_items: tuple[str, ...] = ()
    value_accuracy: float | None = None
    supersession_ok: bool | None = None
    stale_values: tuple[str, ...] = ()
    # Amendments whose NEW value the answer states while never mentioning the old one: the
    # current figure given without noting it changed. Neither stale nor a correct report, so
    # it is recorded rather than folded into either.
    unmarked_changes: tuple[str, ...] = ()
    faithfulness: float | None = None
    context_hit: bool | None = None
    latency_ms: dict[str, int] = field(default_factory=dict)


def score_query(
    gold: GoldQuery,
    answer: GroundedAnswer,
    *,
    latency_ms: dict[str, int] | None = None,
) -> QueryScore:
    """Score one answer against one gold query. Pure — no model, no network.

    ``latency_ms`` is the per-stage timeline Phase 12 records
    (:class:`rag_common.timing.StageTimeline`). Carried through rather than measured here: this
    module scores correctness, and the only honest source for a per-stage breakdown is the server
    that ran the stages."""
    answered = answer.status is not AnswerStatus.ABSTAINED
    folded = fold_text(answer.answer)
    score: dict[str, Any] = {
        "query": gold.query,
        "intent": gold.intent,
        "answered": answered,
        "abstain_ok": gold.expect_abstain == (not answered),
        "faithfulness": answer.faithfulness_score,
        "latency_ms": dict(latency_ms or {}),
    }
    if gold.expected_answer:
        # The platform must have STOOD BEHIND the answer as well as stated the fact. An abstention
        # that happens to contain the expected string in its refusal text is not a correct answer,
        # and the v1 gate makes the same distinction.
        score["lookup_hit"] = answered and contains_cue(folded, gold.expected_answer)
    if gold.checklist:
        score.update(_checklist_scores(gold.checklist, folded, answer.answer))
    if gold.supersessions:
        score.update(_supersession_scores(gold.supersessions, folded, answer.answer or ""))
    if not gold.expect_abstain:
        got = {c.doc_id for c in answer.citations}
        score["context_hit"] = bool(not gold.expected_doc_ids or got & set(gold.expected_doc_ids))
    return QueryScore(**score)


def _checklist_scores(checklist: tuple[ChecklistItem, ...], folded: str, raw_answer: str) -> dict[str, Any]:
    """Item recall per tier, the ids that were missed, and value accuracy.

    An item is COVERED when every one of its ``must_contain`` entries appears. All, not any: an
    entry is there because the item is not adequately answered without it, so treating the list
    as alternatives would score a partial item as complete — the same half-answer the Phase-10
    enumeration check exists to catch, one level up."""
    # THE EVALUATION REPRESENTATION, not the raw answer. `contains_cue` below reads `folded`,
    # where `fold_text` strips markdown so a bolded figure still matches; this read the RAW text,
    # so the fix landed at one of two call sites on consecutive lines. `extract_values` builds its
    # `qty:<n> <unit>` key from the unit FOLLOWING the number, so emphasis between them left only
    # a bare `num:` -- and the answer prompt asks for exactly that shape ("put EVERY number in
    # **bold**", "where a value is plainly money, prefix ₹"), which put the presentation contract
    # and the grader in direct conflict. Measured over three paired cycles, this alone cost 3 of
    # 29 value matches on the arm that obeys the contract, and understates every arm that bolds --
    # the incumbent's 622 bold spans included.
    #
    # `strip_emphasis`, NOT `fold_text`: folding lowercases, which destroys the case-sensitive
    # `std:` key and dissolves `IEC 62443-4-2` into `num:62443`/`num:4`/`num:2` -- on a real
    # answer it destroyed all six standards while inventing their digits as independent numeric
    # evidence. The raw answer stays authoritative; this is only what gets compared.
    answer_values = extract_values(strip_emphasis(raw_answer))
    per_tier: dict[str, list[float]] = {}
    missed: list[str] = []
    value_hits: list[float] = []
    for item in checklist:
        covered = all(contains_cue(folded, entry) for entry in item.must_contain)
        per_tier.setdefault(item.tier, []).append(1.0 if covered else 0.0)
        if not covered:
            missed.append(item.id)
        # VALUE accuracy is a different question from recall and is scored separately: an answer
        # that names the facet and states the wrong figure is covered=False here AND
        # value-inaccurate, while one that omits the facet entirely is only the former. Keeping
        # them apart is what distinguishes "did not mention it" from "mentioned it wrongly".
        for entry in item.must_contain:
            wanted = extract_values(entry)
            if wanted:
                value_hits.append(1.0 if wanted <= answer_values else 0.0)
    return {
        "item_recall_must": _mean(per_tier.get("must")),
        "item_recall_should": _mean(per_tier.get("should")),
        "missed_items": tuple(missed),
        "value_accuracy": _mean(value_hits),
    }


#: How far apart the old and the new value may sit and still be read as one statement. Generous,
#: because a real amendment clause restates the whole requirement on each side of the change: the
#: AEGCL corrigendum's DRS amendment puts 200 characters between "letterhead of the respective
#: OEM" and "GTP/Data Sheet". The cap is here only to stop two unrelated bullets pairing up.
_SUPERSESSION_SPAN = 600

#: A digit run, for locating a figure-bearing value one occurrence at a time.
_FIGURE_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _value_positions(value: str, answer: str) -> list[int]:
    """Where ``value`` occurs in ``answer``, matched by VALUE IDENTITY rather than by its wording.

    THE WORDING DIFFERS AND THE VALUE DOES NOT. Measured across all nine stored runs: the gold
    says ``Rs. 19.44 Crores`` and the answers say ``₹19.44 crore`` or ``₹**19.44** crores``.
    ``fold_text`` folds case and separators but not the currency symbol and not the plural, so a
    correctly reported amendment scored as unreported on 9 runs out of 9 -- not one flaky item,
    which is what an earlier decision declined to generalise from.

    NO NEW EQUIVALENCE IS INVENTED HERE. ``extract_values`` ALREADY canonicalises all three of
    those to the same ``qty:19.44 crore`` key; the inconsistency was between that vocabulary and
    the plain substring search this metric used. So a figure-bearing value is located by finding
    its NUMBER and confirming the window around it carries the value's own extracted key, and a
    purely textual value (``GTP/Data Sheet``, ``withdrawn``) is still a folded substring match.
    """
    want = extract_values(value)
    normalised = strip_emphasis(answer)
    folded_answer = fold_text(normalised)
    numeric = {k for k in want if k.startswith(("num:", "qty:"))}
    if not numeric:
        folded_value = fold_text(value)
        if not folded_value:
            return []
        out, i = [], folded_answer.find(folded_value)
        while i != -1:
            out.append(i)
            i = folded_answer.find(folded_value, i + 1)
        return out
    # Figure-bearing: every digit run whose SURROUNDINGS carry the value's own extracted key.
    # The window is what decides, not the digits — `19.44` alone is not `19.44 crore`, and asking
    # `extract_values` about the window is the same question this metric asks about the whole
    # answer, one occurrence at a time.
    out = []
    for m in _FIGURE_RE.finditer(normalised):
        window = normalised[max(0, m.start() - 40) : m.end() + 40]
        if numeric <= extract_values(window):
            out.append(len(fold_text(normalised[: m.start()])))
    return out


def _supersession_scores(supersessions: tuple[Supersession, ...], folded: str, answer: str) -> dict[str, Any]:
    """Was each amendment reported AS A CHANGE — old and new together, as one statement?

    THE PAIR IS THE ASSERTION, and that is a correction. The rule was two independent global
    searches: ``new`` anywhere in the answer satisfied freshness, and ``old`` anywhere without a
    nearby ``new`` was stale. Both halves misfired on the real corpus, in opposite directions.

    It CREDITED AN AMENDMENT THAT WAS NEVER REPORTED. The ``D.3`` entry is "Individual
    Confidentiality Declaration -> withdrawn", and the incumbent's answer passed its freshness
    half because the word "withdrawn" appears in it -- about Appendix D's Vendor Undertaking
    Formats, a different clause entirely. "Individual Confidentiality Declaration" is absent from
    that answer. A metric that passes on a common word landing anywhere in 47,000 characters is
    not measuring supersession, and for a compliance product passing by coincidence is worse than
    failing.

    Tying the values to the gold's ``field`` would have been the obvious fix and does not work:
    the fields are clause identifiers (``BDS ITB 1.3.9.1 item 5``, ``D.3``) and only ``MAAT``
    appears in any answer. Requiring the PAIR needs no such vocabulary and is what this metric's
    own intent already asked for -- a compliance reader has to know a figure CHANGED, not merely
    what it is now, because an answer silently stating ₹19.44 Cr gives a bidder no way to tell
    their ₹35 Cr assumption is stale.

    So each amendment is REPORTED when an occurrence of ``old`` is followed, within
    :data:`_SUPERSESSION_SPAN`, by ``new``. ``stale`` keeps its meaning: ``old`` present and never
    paired. And an answer that states ``new`` while never mentioning ``old`` is recorded as
    ``unmarked_changes`` -- it is not stale (it asserts nothing superseded) and it is not a
    correct report either, so it must be neither silently passed nor counted as staleness.
    """
    stale: list[str] = []
    unmarked: list[str] = []
    reported = 0
    for item in supersessions:
        olds = _value_positions(item.old, answer)
        news = _value_positions(item.new, answer)
        if not olds:
            # The old value is absent. If the new one is present the answer gave the current
            # figure without noting it changed; if neither is there it did not answer at all.
            if news:
                unmarked.append(item.field)
            continue
        # PROXIMITY AND NOT A MARKER WORD. Requiring an arrow or "amended" between them over-fits:
        # "it was Rs. 35 Crores and is now Rs. 19.44 Crores" is a correct report and that rule
        # called it stale.
        if any(0 <= n - o <= _SUPERSESSION_SPAN for o in olds for n in news):
            reported += 1
        else:
            stale.append(item.old)
    return {
        "supersession_ok": bool(supersessions) and reported == len(supersessions),
        "stale_values": tuple(stale),
        "unmarked_changes": tuple(unmarked),
    }


def _mean(values: list[float] | None) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True, slots=True)
class SuiteFloors:
    """The §0 targets a build must not regress below. Every one defaults to None = NOT GATED.

    Defaulting to ungated is deliberate and is the difference between a gate and a guess. The
    plan's §0 criteria are owner-confirmable numbers (discovery §9.2 still lists the latency
    targets as open), and a floor invented here would either be so low it gates nothing or so high
    it fails every build until somebody lowers it — at which point the gate has taught the team to
    edit the threshold instead of the code. An operator sets the floors they have agreed."""

    min_abstain_correctness: float | None = None
    min_lookup_accuracy: float | None = None
    min_item_recall_must: float | None = None
    min_value_accuracy: float | None = None
    min_supersession_correctness: float | None = None
    min_context_precision: float | None = None
    min_faithfulness: float | None = None


@dataclass(frozen=True, slots=True)
class SuiteReport:
    """A suite's aggregate metrics. ``None`` means the suite did not ask for that metric."""

    suite: str
    doc_type: str
    n: int
    scored: int
    abstain_correctness: float | None = None
    lookup_accuracy: float | None = None
    item_recall_must: float | None = None
    item_recall_should: float | None = None
    value_accuracy: float | None = None
    supersession_correctness: float | None = None
    context_precision: float | None = None
    faithfulness_avg: float | None = None
    intent_counts: dict[str, int] = field(default_factory=dict)
    latency_p95: dict[str, int] = field(default_factory=dict)
    suite_version: int = SUITE_VERSION

    def failures(self, floors: SuiteFloors) -> tuple[str, ...]:
        """Which agreed floors this report is below. Empty means the gate passes.

        A metric the suite could not measure NEVER fails a floor. A gold set with no supersession
        items cannot tell you whether supersession handling regressed, and reporting that as a
        regression would make the gate fire on the suite's own incompleteness rather than on the
        build."""
        checks = (
            ("abstain_correctness", self.abstain_correctness, floors.min_abstain_correctness),
            ("lookup_accuracy", self.lookup_accuracy, floors.min_lookup_accuracy),
            ("item_recall_must", self.item_recall_must, floors.min_item_recall_must),
            ("value_accuracy", self.value_accuracy, floors.min_value_accuracy),
            ("supersession_correctness", self.supersession_correctness, floors.min_supersession_correctness),
            ("context_precision", self.context_precision, floors.min_context_precision),
            ("faithfulness_avg", self.faithfulness_avg, floors.min_faithfulness),
        )
        return tuple(
            f"{metric} {value:.3f} < {floor:.3f}"
            for metric, value, floor in checks
            if floor is not None and value is not None and value < floor
        )

    def passes(self, floors: SuiteFloors) -> bool:
        return not self.failures(floors)


def score_suite(suite: GoldSuite, scores: list[QueryScore]) -> SuiteReport:
    """Aggregate per-query scores into the suite report (plan §11's metric list).

    ``latency_p95`` is a per-STAGE p95 across the suite, which is the shape the plan's §0 targets
    are written in ("<= 8 s lookup p95"). Computed per stage rather than only end to end, because
    a suite that is slow tells you nothing and a suite that is slow IN RETRIEVAL tells you where
    to look — which is the whole reason Phase 12's timeline exists."""
    if len(suite.queries) != len(scores):
        raise ValueError("queries/scores length mismatch")
    intents: dict[str, int] = {}
    for gold in suite.queries:
        intents[gold.intent] = intents.get(gold.intent, 0) + 1
    return SuiteReport(
        suite=suite.name,
        doc_type=suite.doc_type,
        n=len(scores),
        # A query that specified NOTHING to check is counted here so a suite cannot look fully
        # measured when half of it is still a list of questions.
        scored=sum(1 for s in scores if _is_scored(s)),
        abstain_correctness=_mean([1.0 if s.abstain_ok else 0.0 for s in scores]),
        lookup_accuracy=_mean([1.0 if s.lookup_hit else 0.0 for s in scores if s.lookup_hit is not None]),
        item_recall_must=_mean([s.item_recall_must for s in scores if s.item_recall_must is not None]),
        item_recall_should=_mean([s.item_recall_should for s in scores if s.item_recall_should is not None]),
        value_accuracy=_mean([s.value_accuracy for s in scores if s.value_accuracy is not None]),
        supersession_correctness=_mean(
            [1.0 if s.supersession_ok else 0.0 for s in scores if s.supersession_ok is not None]
        ),
        context_precision=_mean([1.0 if s.context_hit else 0.0 for s in scores if s.context_hit is not None]),
        faithfulness_avg=_mean([s.faithfulness for s in scores if s.faithfulness is not None]),
        intent_counts=intents,
        latency_p95=_latency_p95(scores),
    )


def _is_scored(score: QueryScore) -> bool:
    return (
        any(
            v is not None
            for v in (score.lookup_hit, score.item_recall_must, score.item_recall_should, score.supersession_ok)
        )
        or score.abstain_ok is not None
    )


def _latency_p95(scores: list[QueryScore]) -> dict[str, int]:
    """Per-stage p95 in milliseconds, over the queries that reported that stage.

    NEAREST-RANK p95 (the value at ``ceil(0.95 n) - 1`` of the sorted samples), not an
    interpolated one: a suite is tens of questions, and interpolating between two samples at that
    size invents a number that no question actually took."""
    stages: dict[str, list[int]] = {}
    for score in scores:
        for stage, ms in score.latency_ms.items():
            stages.setdefault(stage, []).append(ms)
    out: dict[str, int] = {}
    for stage, samples in stages.items():
        ordered = sorted(samples)
        index = max(0, -(-len(ordered) * 95 // 100) - 1)
        out[stage] = ordered[index]
    return out
