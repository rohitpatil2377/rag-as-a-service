"""Domain-profile detection (Slice 6, sub-slice A) — pure logic, no I/O.

The KB-level analogue of :mod:`rag_core.tagging`'s per-document ``auto_tags``: instead of
classifying ONE document, it classifies a whole corpus and names the domain it is drawn from
(general / chemistry / legal / financial / code). The platform then *proposes* — never
auto-applies — the per-KB retrieval knobs (Slice 2a/2b/3/4) that suit that domain.

Same discipline as :mod:`tagging` / :mod:`notation`: deterministic, CPU-only, dependency-free,
CI-safe. Signals are REUSED, not reinvented — the chemistry detector is literally the
battle-tested notation scanner (``notation._iter_notation``) aggregated to a KB verdict; the
legal / financial / code detectors are small regex cue tables shaped exactly like
``tagging._DOC_TYPE_CUES``. Every signal is a rate PER 1000 CHARACTERS so a big KB and a small
KB compare fairly.

It is CONSERVATIVE by mandate (mirrors ``tagging.document_date`` returning ``None`` rather than
guessing): a weak maximum, or a near-tie between two domains, falls back to ``general`` — the
domain-neutral default that changes no knob. A misclassified KB that silently turned on
notation-grounding / exact-boost would shift answers, so "not sure" resolves to "leave the
platform defaults."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_core.notation import _iter_notation

# Domain classes — deliberately small, extensible, domain-neutral default first.
DOMAINS: tuple[str, ...] = ("general", "chemistry", "legal", "financial", "code")
# The domains that carry an actual signal detector. ``general`` is the residual/fallback, never
# scored directly.
_SIGNAL_DOMAINS: tuple[str, ...] = ("chemistry", "legal", "financial", "code")


@dataclass(frozen=True, slots=True)
class DomainSignal:
    """A KB's detected domain + the evidence behind it (auditable — the operator sees WHY)."""

    domain: str  # one of DOMAINS
    score: float  # 0..1 confidence for the winning domain (its share of the signal mass)
    mix: dict[str, float]  # normalized per-domain signal share (sums to ~1 across signal domains)
    evidence: dict[str, int]  # raw hit counts that drove it


@dataclass(frozen=True, slots=True)
class KnobSuggestion:
    """A domain profile's *pending, unconfirmed* knob recommendation, split by COST CLASS.

    ``query_time_knobs`` take effect on the next query (safe to apply immediately);
    ``reindex_knobs`` change what is stored and need a reindex pass, so they are SUGGEST-ONLY —
    the apply endpoint refuses to write them unless the operator explicitly opts in and is warned.
    """

    query_time_knobs: dict[str, float] = field(default_factory=dict)
    reindex_knobs: dict[str, float] = field(default_factory=dict)
    rationale: str = ""


# Which tenant_config keys are safe to apply immediately vs. which require a reindex. These mirror
# the Slice 2a/2b/3/4 keys in ``pipeline._THRESHOLD_KEYS`` and are the guard the apply endpoint
# uses so a reindex-class knob can never be silently written into the query-time bucket.
QUERY_TIME_KNOB_KEYS: frozenset[str] = frozenset(
    {
        "exact_boost",
        "notation_grounding_enabled",
        "parent_child_max_parents",
        "parent_child_context_max_chars",
    }
)
REINDEX_KNOB_KEYS: frozenset[str] = frozenset(
    {
        "parent_child_child_max_chars",
        "parent_child_parent_max_chars",
        "multivector_sparse_enabled",
        "multivector_colbert_enabled",
        "chem_normalization_enabled",
        "domain_relations_enabled",
    }
)


# --- cue tables (shaped like tagging._DOC_TYPE_CUES) -------------------------------------------
# Distinctive surface markers for each non-chemistry domain. Deliberately narrow — a cue that
# fires on ordinary HR/business prose (e.g. a bare "shall") is worse than none, because it would
# tilt a general KB toward a specialised profile. Chemistry is NOT here: it reuses the notation
# scanner instead (the strongest reuse in the slice).
_LEGAL_CUES: tuple[str, ...] = (
    r"\bsection\s+\d+",
    r"\bclause\s+\d+",
    r"\bsub-?section\b",
    r"\bpursuant to\b",
    r"\bhereby\b",
    r"\bherein\b",
    r"\bhereinafter\b",
    r"\bwhereas\b",
    r"\bin witness whereof\b",
    r"\b[A-Z][a-z]+\s+v\.\s+[A-Z]",  # a case citation "Smith v. Jones"
    r"\bAct,?\s+\d{4}\b",
    r"\bnotwithstanding\b",
)
_FINANCIAL_CUES: tuple[str, ...] = (
    r"\bGST(?:IN)?\b",
    r"\binvoice\b",
    r"\bledger\b",
    r"\bfiscal\b",
    r"\bebitda\b",
    r"\bbalance sheet\b",
    r"\baccounts? (?:payable|receivable)\b",
    r"\bnet (?:profit|revenue|income)\b",
    r"\bdepreciation\b",
    r"₹\s?\d",
    r"\$\s?\d",
    r"\b(?:INR|USD)\s?\d",
    r"\b\d+(?:\.\d+)?\s*(?:crore|lakh|million|billion)\b",
)
_CODE_CUES: tuple[str, ...] = (
    r"```",
    r"\bdef\s+\w+\s*\(",
    r"\bfunction\s+\w+\s*\(",
    r"^\s*import\s+\w",
    r"^\s*from\s+\w+\s+import\b",
    r"\bclass\s+\w+\s*[:({]",
    r"\bconst\s+\w+\s*=",
    r"=>",
    r"</?[a-zA-Z][\w-]*>",  # an HTML/XML/JSX tag
)

_LEGAL_RE = [re.compile(p, re.IGNORECASE) for p in _LEGAL_CUES]
_FINANCIAL_RE = [re.compile(p, re.IGNORECASE) for p in _FINANCIAL_CUES]
# Code cues are case-sensitive (keywords are lowercase) and some are anchored per-line.
_CODE_RE = [re.compile(p, re.MULTILINE) for p in _CODE_CUES]
# `{`, `}`, `;` density — punctuation that is dense in code and sparse in prose.
_CODE_PUNCT_RE = re.compile(r"[{};]")

# Winning signal must clear this rate (hits per 1000 chars) or the corpus is called ``general``.
_MIN_RATE = 1.0
# The winner must lead the runner-up by this factor, else the corpus is a near-tie -> ``general``.
_TIE_MARGIN = 1.5


def _cue_hits(text: str, patterns: list[re.Pattern[str]]) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def _code_hits(text: str) -> int:
    # Cue matches plus a discounted contribution from brace/semicolon density (÷4 so a stray
    # sentence with a semicolon doesn't read as code, while a real code block dominates).
    return _cue_hits(text, _CODE_RE) + len(_CODE_PUNCT_RE.findall(text)) // 4


def _evidence(text: str) -> dict[str, int]:
    """Raw per-domain hit counts for one text (auditable; the basis for every rate)."""
    return {
        "chemistry": sum(1 for _ in _iter_notation(text)),
        "legal": _cue_hits(text, _LEGAL_RE),
        "financial": _cue_hits(text, _FINANCIAL_RE),
        "code": _code_hits(text),
    }


def score_domain(text: str) -> dict[str, float]:
    """Per-domain cheap signal for a single text, as a rate per 1000 characters."""
    ev = _evidence(text)
    per_1k = 1000.0 / max(len(text), 1)
    return {d: round(ev[d] * per_1k, 4) for d in _SIGNAL_DOMAINS}


def classify_corpus(samples: list[str]) -> DomainSignal:
    """Aggregate the per-domain signal across a corpus sample into a single :class:`DomainSignal`.

    Sums raw evidence over every sample and divides by the total character count, so the verdict
    is a corpus-wide rate rather than an average of per-sample rates (fair to mixed-length
    samples). Argmax of the rates is the domain — UNLESS the winner is weak (< ``_MIN_RATE``) or
    fails to lead the runner-up by ``_TIE_MARGIN``, in which case the conservative ``general``
    default wins (an incidental formula in HR prose must not flip the whole KB to chemistry).
    """
    ev_total: dict[str, int] = dict.fromkeys(_SIGNAL_DOMAINS, 0)
    chars = 0
    for s in samples:
        for d, n in _evidence(s).items():
            ev_total[d] += n
        chars += len(s)
    per_1k = 1000.0 / max(chars, 1)
    rates = {d: ev_total[d] * per_1k for d in _SIGNAL_DOMAINS}
    total = sum(rates.values())
    mix = {d: round(rates[d] / total, 4) if total else 0.0 for d in _SIGNAL_DOMAINS}

    ranked = sorted(rates.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top_domain, top_rate = ranked[0]
    runner_rate = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_rate < _MIN_RATE or (runner_rate > 0 and top_rate < runner_rate * _TIE_MARGIN):
        # Conservative fallback: confidence in "general" is the share NOT claimed by any one domain.
        return DomainSignal(
            domain="general",
            score=round(1.0 - mix.get(top_domain, 0.0), 4),
            mix=mix,
            evidence=ev_total,
        )
    return DomainSignal(domain=top_domain, score=mix[top_domain], mix=mix, evidence=ev_total)


# --- profile -> knob mapping (additive, reversible, reindex-aware) -----------------------------
def suggest_knobs(signal: DomainSignal) -> KnobSuggestion:
    """The per-KB knobs a detected domain recommends, split into query-time (safe to apply now)
    and reindex-required (SUGGEST ONLY). Every value is the SAME tenant_config key an operator
    edits by hand, so applying one is fully reversible (clear the key -> platform default)."""
    domain = signal.domain
    if domain == "chemistry":
        return KnobSuggestion(
            query_time_knobs={"notation_grounding_enabled": 1.0, "exact_boost": 3.0},
            reindex_knobs={
                "parent_child_child_max_chars": 600.0,
                "multivector_colbert_enabled": 1.0,
                "chem_normalization_enabled": 1.0,
                "domain_relations_enabled": 1.0,
            },
            rationale=(
                "Chemistry corpus: exact formula/SMILES/identifier matches and verbatim notation "
                "grounding matter, so boost exact-term BM25 and turn on notation grounding "
                "(query-time). Denser chunking, ColBERT re-scoring, RDKit canonicalization and "
                "typed domain relations further help but require a reindex — suggested only."
            ),
        )
    if domain == "legal":
        return KnobSuggestion(
            query_time_knobs={"exact_boost": 3.0, "notation_grounding_enabled": 0.0},
            reindex_knobs={"domain_relations_enabled": 1.0},
            rationale=(
                "Legal corpus: citations and section numbers are exact tokens, so boost "
                "exact-term BM25 (query-time). Typed relations (amends/supersedes/cites) improve "
                "the graph but require a reindex — suggested only. Notation grounding stays off."
            ),
        )
    if domain == "financial":
        return KnobSuggestion(
            query_time_knobs={"exact_boost": 3.0},
            reindex_knobs={},
            rationale=(
                "Financial corpus: amounts, GSTINs and identifiers are exact tokens, so boost "
                "exact-term BM25 (query-time). No reindex-class change recommended."
            ),
        )
    if domain == "code":
        return KnobSuggestion(
            query_time_knobs={"exact_boost": 2.5},
            reindex_knobs={},
            rationale=(
                "Code corpus: symbols and identifiers are exact tokens, so a modest exact-term "
                "BM25 boost helps (query-time). No reindex-class change recommended."
            ),
        )
    return KnobSuggestion(
        query_time_knobs={},
        reindex_knobs={},
        rationale="General corpus: leave the platform defaults — no domain-specific knob change recommended.",
    )
