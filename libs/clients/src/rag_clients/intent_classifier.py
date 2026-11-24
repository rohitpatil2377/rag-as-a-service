"""Intent classifier adapters (implement :class:`IntentClassifier`).

* :class:`HeuristicIntentClassifier` — wraps the deterministic keyword fallback (default).
* :class:`LLMIntentClassifier` — the real semantic path: one tiny classification call to the
  sovereign gateway, parsed robustly. Fail-safe — any gateway error → ``"rag"`` (never
  mis-route a real question onto a meta-handler).

Mirrors ``routing.py`` (HeuristicRouter / LLMRouter / build_query_router).
"""

from __future__ import annotations

from rag_common import ConfigError, get_logger, get_settings
from rag_core import build_intent_prompt, classify_intent_heuristic, is_formatting_turn, parse_intent
from rag_core.enumeration import enumerated_items
from rag_core.interfaces import LLMGateway

_log = get_logger(__name__)


class HeuristicIntentClassifier:
    """Deterministic, no-LLM intent classification (the default). Fast, sovereign, explainable."""

    async def classify(self, query: str, *, has_history: bool) -> str:
        return classify_intent_heuristic(query)


class LLMIntentClassifier:
    """LLM-classified intent via the sovereign gateway — the real semantic path (one small call)."""

    def __init__(self, gateway: LLMGateway, *, max_tokens: int = 8) -> None:
        self._gw = gateway
        self._max_tokens = max_tokens

    async def classify(self, query: str, *, has_history: bool) -> str:
        try:
            raw = await self._gw.complete(
                build_intent_prompt(query, has_history=has_history), max_tokens=self._max_tokens, temperature=0.0
            )
        except Exception as exc:  # fail-safe: a model error must never mis-route a real question
            _log.warning("intent_classify_failed_fallback_rag", error=str(exc)[:160])
            return "rag"
        return _guard_meta_intent(query, parse_intent(raw, has_history=has_history))


# Meta-intents answer questions ABOUT the knowledge base's files -- their names, their count,
# one summary per file. A question that NAMES identifiers is asking about things INSIDE a file,
# so none of them can be right.
#
# Reported live: "explain PBSOP001, PBSOP002 ... PBSOP010" came back as "Summary of 1 document in
# this knowledge base" plus a list of the three uploaded files. The model read ten identifiers as
# ten documents -- an understandable reading, since each PBSOP really is a procedure document in
# the world; it just is not a document in THIS knowledge base, where all ten live inside one PDF.
#
# The deterministic classifier already refuses this. The guard exists because the LLM backend
# bypasses it, and a model's guess must not override a signal that is not a guess. Identifiers are
# matched by shape, and the threshold is shared with retrieval coverage (enumeration.MIN_ITEMS) so
# the two cannot disagree about what counts as enumerated.
_DOCUMENT_UNIT_INTENTS = frozenset({"catalog", "doc_summary", "count", "filter"})
# Every intent that answers something OTHER than the reader's question from the corpus: the file
# inventory, a count, or a recap of the conversation. A turn asking for a different PRESENTATION
# of the previous answer is none of them.
#
# Reported live: after a table of ten SOPs, "give me the above in bullet point" and "make it
# shorter" both came back as "The user asked for explanations of ten Standard Operating
# Procedures. The assistant responded with a table..." -- a summary OF THE CHAT rather than the
# same ten items re-rendered. The conversation path classifies the RAW turn before condensing it,
# so the model saw two words with no subject and reasonably guessed "summarize"; condensation,
# which would have folded the question back in, never ran.
_META_INTENTS = _DOCUMENT_UNIT_INTENTS | {"summarize"}


def _guard_meta_intent(query: str, intent: str) -> str:
    """Refuse a meta-intent the query's own shape contradicts."""
    if intent not in _META_INTENTS:
        return intent
    # Naming identifiers means asking about things INSIDE a document, never about the documents.
    if intent in _DOCUMENT_UNIT_INTENTS and enumerated_items(query):
        _log.info("intent_override_enumerated_items", model_intent=intent, chosen="rag")
        return "rag"
    # A presentation-only turn re-renders the previous ANSWER; it is never a request to summarise
    # the conversation or to inventory the corpus.
    if is_formatting_turn(query):
        _log.info("intent_override_formatting_turn", model_intent=intent, chosen="rag")
        return "rag"
    return intent


def build_intent_classifier(gateway: LLMGateway | None = None) -> HeuristicIntentClassifier | LLMIntentClassifier:
    """Factory: heuristic (deterministic/default) or llm, per ``RAG_INTENT_BACKEND``."""
    backend = get_settings().intent_backend
    if backend == "heuristic":
        return HeuristicIntentClassifier()
    if backend == "llm":
        if gateway is None:
            raise ConfigError("llm intent backend requires an LLM gateway")
        return LLMIntentClassifier(gateway)
    raise ConfigError(f"unknown intent backend: {backend}")
