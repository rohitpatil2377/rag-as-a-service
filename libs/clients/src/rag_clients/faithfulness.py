"""Faithfulness checker adapters (implement :class:`FaithfulnessChecker`) — FR-E4.

* :class:`LexicalFaithfulnessChecker` — deterministic, zero-model. Scores each answer
  sentence by content-token overlap with the cited source text; the score is the
  fraction of answer content tokens found in the sources. Used by CI (no model needed).
* :class:`LLMFaithfulnessChecker` — LLM-as-judge via the sovereign gateway, returning a
  structured JSON verdict. Selected in dev/prod when a real model is served.

ADR-0005: on the small dev model the LLM judge is a WEAK signal; prefer a distinct/
stronger judge model once real vLLM lands. Citations stay structural regardless.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from rag_common import ConfigError, get_logger, get_settings
from rag_core import FaithfulnessVerdict, build_faithfulness_prompt
from rag_core.anchoring import anchor_score, content_terms
from rag_core.faithfulness import FaithfulnessChecker, notation_flag_band_score
from rag_core.interfaces import LLMGateway
from rag_core.notation import mask_notation_tokens, notation_tokens_in, unsupported_notation
from rag_core.numeracy import numbers_in, unsupported_numbers
from rag_schemas import RetrievedSpan

_log = get_logger(__name__)

# Word units for overlap. Indic scripts (Hindi/Tamil/Bengali/…) put SPACES between words, so we
# tokenize on whitespace and strip surrounding punctuation — this keeps whole words intact
# ("पर्यावरण", "रक्षा"), unlike a per-character `\w+` which mis-splits Devanagari at combining
# marks/conjuncts. Purpose here is consistent word units for overlap, not linguistic tokenisation
# (for that, an Indic tokenizer like indic-nlp-library / stanza would be the tool).
# Sentence split on Latin AND Devanagari danda (।) / double-danda (॥) terminators.
_SENT_RE = re.compile(r"(?<=[.!?।॥])\s+")
# Minimal stopword set so scoring keys on content words, not filler.
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "these",
        "those",
        "their",
        "they",
        "you",
        "your",
        "our",
        "we",
        "i",
        "he",
        "she",
    ]
)
# Per sentence: the weighted anchor score (rag_core.anchoring) at or above which it is supported.
_SENTENCE_SUPPORT = 0.6


class LexicalFaithfulnessChecker:
    """Deterministic overlap-based faithfulness (no model)."""

    async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
        # `query` unused: this deterministic overlap checker has no per-clause relevance gate
        # (that lives in the NLI backend); accepted only so every checker shares one call shape.
        del query
        # Each sentence is scored against its best ONE span with the weighted anchor
        # (rag_core.anchoring), not against the pooled vocabulary of every span: a pooled bag
        # lets a sentence stitch one word from each source into a claim none of them makes.
        # Measured at the same 0.6: on 138 sentences of real grounded answers the pooled exact-match
        # version wrongly failed 42 and this fails 32; on a labelled set it caught 11 of 15 invented
        # sentences and this catches all 15. (It is the stricter checker by design — the CI default
        # and the fallback when the NLI service is down.)
        texts = [s.text for s in spans]
        length_norm = get_settings().faithfulness_anchor_length_norm
        sentences = [s for s in _SENT_RE.split(answer.strip()) if s.strip()]
        total = 0
        supported = 0.0
        unsupported: list[str] = []
        for sent in sentences:
            words = content_terms(sent, stopwords=_STOP)
            if not words:
                continue
            anchored = anchor_score(sent, texts, stopwords=_STOP, length_norm=length_norm)
            total += len(words)
            supported += anchored * len(words)
            if anchored < _SENTENCE_SUPPORT:
                unsupported.append(sent.strip())
        score = 1.0 if total == 0 else supported / total
        return FaithfulnessVerdict(
            score=round(score, 4), faithful=not unsupported, unsupported_claims=unsupported, method="lexical"
        )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMFaithfulnessChecker:
    """LLM-as-judge faithfulness (structured JSON). Sovereign gateway.

    ``max_tokens`` is generous because a REASONING model (e.g. Qwen3) spends tokens in a
    ``<think>`` block before emitting the JSON verdict — too small a budget and it never
    reaches the JSON. If the verdict is still unparsable, fall back to the deterministic
    lexical checker rather than defaulting to 0.0 (a broken judge must not force good,
    grounded answers to abstain)."""

    def __init__(self, gateway: LLMGateway, *, max_tokens: int = 1024) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens
        self._fallback = LexicalFaithfulnessChecker()

    async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
        # `query` unused by the LLM judge itself; forwarded only to the lexical fallback below
        # for call-shape consistency (it also ignores it today).
        prompt = build_faithfulness_prompt(answer, spans)
        raw = await self._gw.complete(prompt, max_tokens=self._max_tokens, temperature=0.0)
        match = _JSON_RE.search(raw)
        if not match:
            _log.warning("faithfulness_judge_unparsable_fallback_lexical", raw=raw[:160])
            return await self._fallback.check(answer, spans, query)
        try:
            data = json.loads(match.group(0))
            score = float(data.get("score", 0.0))
            claims = [str(c) for c in data.get("unsupported_claims", [])]
        except (ValueError, TypeError):
            _log.warning("faithfulness_judge_bad_json_fallback_lexical", raw=raw[:160])
            return await self._fallback.check(answer, spans, query)
        score = max(0.0, min(1.0, score))
        return FaithfulnessVerdict(
            score=round(score, 4), faithful=not claims, unsupported_claims=claims, method="lexical"
        )


class NumericallyGroundedChecker:
    """Wraps any faithfulness backend so a FABRICATED FIGURE cannot pass as supported.

    Both backends judge prose. The lexical one scores token overlap and drops short tokens by
    design, so a figure has no effect on its score at all; the LLM judge is a small model that
    cannot be trusted with arithmetic. Meanwhile the words around an invented number — "hours",
    "days", "working" — are all in the sources and carry the sentence to "supported".

    A number is the one part of an answer that can be checked without understanding it: it is
    either in the retrieved text or it was invented. Applied HERE, at the seam every backend
    goes through, so a new backend inherits the check instead of having to remember it.

    Slice 4 adds a NOTATION twin alongside the numeric one, gated by ``notation_enabled`` (default
    OFF ⇒ byte-identical to the numeric-only checker). It catches what the ASCII-digit numeric gate
    misses — unit mismatches (5 mg vs 5 g), sub/superscript formula differences (H₂SO₄ vs H₂SO₃)
    and invented chemical/SMILES notation — via the same "is this token in what we retrieved"
    question (see :mod:`rag_core.notation`). Numbers and notation are reported under SEPARATE
    labels so a reviewer sees which fired, and both route to the SAME not-grounded outcome. The
    flag is per-KB adaptive: the wrapper is toggled per KB at the composition seam (see
    ``svc_control_plane.wiring.apply_notation_grounding``)."""

    def __init__(
        self,
        inner: FaithfulnessChecker,
        *,
        notation_enabled: bool = False,
        faithful_threshold: float | None = None,
        flag_threshold: float | None = None,
    ) -> None:
        # Public: "which backend is actually judging this?" is a real question for diagnostics
        # and for tests asserting backend SELECTION, which is not what this wrapper decides.
        self.inner = inner
        # Default OFF: with it off `check` computes nothing extra and returns exactly the
        # numeric-only verdict — the pre-Slice-4 behaviour, byte-for-byte.
        self.notation_enabled = notation_enabled
        # The KB's routing gates, so a NOTATION hit can cap its own score into the FLAGGED band
        # (below `faithful_threshold`, at/above `flag_threshold`). None ⇒ fall back to the platform
        # settings at check time. Only consulted when a notation hit actually fires.
        self._faithful_threshold = faithful_threshold
        self._flag_threshold = flag_threshold

    def with_notation(
        self,
        enabled: bool,
        *,
        faithful_threshold: float | None = None,
        flag_threshold: float | None = None,
    ) -> NumericallyGroundedChecker:
        """A view of this checker with notation grounding toggled + the KB's routing gates bound
        (per-KB adaptive seam). Returns ``self`` when nothing changes so the shared singleton is
        untouched; otherwise a new wrapper over the SAME ``inner`` backend — the numeric check and
        the wrapped judge are unchanged."""
        if (
            enabled == self.notation_enabled
            and faithful_threshold == self._faithful_threshold
            and flag_threshold == self._flag_threshold
        ):
            return self
        return NumericallyGroundedChecker(
            self.inner,
            notation_enabled=enabled,
            faithful_threshold=faithful_threshold,
            flag_threshold=flag_threshold,
        )

    async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
        # Forwarded to `inner` (every real backend is wrapped in this) so a checker that uses
        # `query` -- currently the NLI backend's compound-query relevance gate -- actually sees it.
        verdict = await self.inner.check(answer, spans, query)
        # NOTHING TO ADD TO A VERDICT THAT MADE NO JUDGEMENT. When the inner checker reports
        # `unavailable` there is no score to keep and no claims list to extend, and running the
        # numeric scan anyway would manufacture `faithful=False` plus a claims list on top of a
        # verdict whose whole content is "this was not checked" -- turning an unknown into a
        # finding. The provenance passes straight through.
        if verdict.score is None:
            return verdict
        notation = unsupported_notation(answer, spans) if self.notation_enabled else []
        # When notation grounding is ON, hide detected formula/SMILES tokens from the numeric scan
        # so a formula subscript or a SMILES ring-closure digit is not mis-reported as an invented
        # figure (L1). Numeric-only (notation OFF) sees the answer verbatim — byte-identical to today.
        numeric_answer = mask_notation_tokens(answer) if self.notation_enabled else answer
        invented = unsupported_numbers(numeric_answer, spans)
        if not invented and not notation:
            return verdict
        claims = list(verdict.unsupported_claims)
        if invented:
            claims += _sentences_with(answer, invented)
            _log.warning(
                "answer_states_ungrounded_figures",
                figures=invented,
                detail="numbers not present in any retrieved span — the answer computed or invented them",
            )
        if notation:
            # Separate label from the figures one: a reviewer needs to know whether a UNIT/FORMULA/
            # SMILES token fired, not just that "something numeric" did.
            claims += _sentences_with_notation(answer, notation)
            _log.warning(
                "answer_states_ungrounded_notation",
                notation=notation,
                detail="units/formulae/SMILES not present in any retrieved span — invented or unit-mismatched",
            )
        # De-duplicate the carrying sentences, preserving order (a sentence with BOTH an invented
        # figure and ungrounded notation would otherwise appear two/three times). (L1)
        claims = list(dict.fromkeys(claims))
        # `faithful=False` routes to FLAGGED_FOR_REVIEW through the existing three-way gate.
        #
        # SCOPE — the routed-score cap fires ONLY for a notation hit. `route_faithfulness` reads the
        # SCORE, not `faithful`, so a numeric-only hit keeps the inner judge's score and routes
        # EXACTLY as before (its identical latent no-teeth property is pre-existing and intentionally
        # left unchanged — a separate team decision). A notation hit instead caps the score into the
        # FLAGGED band so a possible false-positive goes to human review — never shipped as GROUNDED,
        # never fully withheld (ABSTAINED). `notation_flagged` lets a reviewer/log tell which fired.
        if notation:
            faithful_gate = self._faithful_threshold
            flag_gate = self._flag_threshold
            if faithful_gate is None or flag_gate is None:
                settings = get_settings()
                faithful_gate = settings.faithful_threshold if faithful_gate is None else faithful_gate
                flag_gate = settings.flag_threshold if flag_gate is None else flag_gate
            score = notation_flag_band_score(faithful_gate, flag_gate)
            return replace(verdict, score=score, faithful=False, unsupported_claims=claims, notation_flagged=True)
        # PROVENANCE SURVIVES THE WRAP. This rebuilds the verdict, and dropping the method here
        # would silently relabel every numerically-guarded answer as `unset` -- so a lexical
        # fallback would stop being visible the moment the guard found a figure.
        return replace(verdict, faithful=False, unsupported_claims=claims)


def _sentences_with(answer: str, figures: list[str]) -> list[str]:
    """The sentences carrying the invented figures — a reviewer needs the claim, not the digit."""
    wanted = set(figures)
    return [s.strip() for s in _SENT_RE.split(answer.strip()) if s.strip() and numbers_in(s) & wanted]


def _sentences_with_notation(answer: str, notation: list[str]) -> list[str]:
    """The sentences carrying the ungrounded notation tokens — a reviewer needs the claim, not the
    bare formula/unit. Mirrors :func:`_sentences_with` for the notation twin."""
    wanted = set(notation)
    return [s.strip() for s in _SENT_RE.split(answer.strip()) if s.strip() and notation_tokens_in(s) & wanted]


def build_faithfulness_checker(
    gateway: LLMGateway | None = None,
    *,
    notation_enabled: bool | None = None,
) -> FaithfulnessChecker:
    """Factory: lexical (deterministic/CI), LLM-judge, or NLI-entailment, per
    ``RAG_FAITHFULNESS_BACKEND``.

    Every backend is wrapped in the numeric check — see :class:`NumericallyGroundedChecker`.
    ``notation_enabled`` seeds the notation twin (Slice 4); when None it falls back to the
    platform-default ``notation_grounding_enabled`` setting, and the per-KB override is applied
    later at the composition seam (``wiring.apply_notation_grounding``)."""
    settings = get_settings()
    notate = settings.notation_grounding_enabled if notation_enabled is None else notation_enabled
    backend = settings.faithfulness_backend
    if backend == "lexical":
        return NumericallyGroundedChecker(LexicalFaithfulnessChecker(), notation_enabled=notate)
    if backend == "llm":
        if gateway is None:
            raise ConfigError("llm faithfulness backend requires an LLM gateway")
        return NumericallyGroundedChecker(LLMFaithfulnessChecker(gateway), notation_enabled=notate)
    if backend == "nli":
        # Imported lazily: the NLI adapter pulls httpx + the TEI scorer, and CI (lexical
        # default) must not pay that import. Same numeric-grounding wrapper as the others.
        from rag_clients.nli_faithfulness import build_nli_faithfulness_checker  # noqa: PLC0415

        return NumericallyGroundedChecker(build_nli_faithfulness_checker(), notation_enabled=notate)
    raise ConfigError(f"unknown faithfulness backend: {backend}")
