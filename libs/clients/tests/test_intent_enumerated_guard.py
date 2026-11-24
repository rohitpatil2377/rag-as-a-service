"""The LLM intent classifier must not re-introduce a bug the heuristic already refuses.

`explain PBSOP001 ... PBSOP010` routes to "rag" under the deterministic classifier. With
RAG_INTENT_BACKEND=llm -- which is what this deployment runs -- the model sees ten
identifiers and reads them as ten DOCUMENTS, returning "doc_summary". The reader then gets an
inventory of the three files in the knowledge base instead of the ten procedures they named.

A model is entitled to be wrong; the classifier is not entitled to pass that through when a
deterministic signal contradicts it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from rag_clients.intent_classifier import LLMIntentClassifier

_TEN = "PBSOP001, PBSOP002, PBSOP003, PBSOP004, PBSOP005, PBSOP006, PBSOP007, PBSOP008, PBSOP009, and PBSOP010"


class _Gateway:
    """Returns whatever label the test wants the model to have produced."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        self.calls += 1
        return self.label

    def stream(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> AsyncIterator[str]:
        raise NotImplementedError  # the classifier never streams

    async def list_models(self) -> list[str]:
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("model_says", ["doc_summary", "catalog", "count", "filter"])
async def test_an_enumerating_question_is_never_routed_to_a_document_inventory(model_says: str) -> None:
    c = LLMIntentClassifier(_Gateway(model_says))
    assert await c.classify(f"explain {_TEN}", has_history=False) == "rag"


@pytest.mark.asyncio
@pytest.mark.parametrize("turn", ["give me the above in bullet point", "make it shorter", "in a table instead"])
@pytest.mark.parametrize("model_says", ["summarize", "doc_summary", "catalog"])
async def test_a_presentation_only_turn_never_becomes_a_recap_of_the_chat(turn: str, model_says: str) -> None:
    """Live: after a table of ten SOPs, "make it shorter" answered "The user asked for
    explanations of ten Standard Operating Procedures. The assistant responded with a table..."
    -- the conversation summarised instead of the answer re-rendered."""
    c = LLMIntentClassifier(_Gateway(model_says))
    assert await c.classify(turn, has_history=True) == "rag"


@pytest.mark.asyncio
async def test_a_genuine_recap_request_still_summarises() -> None:
    """The guard must not swallow the intent it narrows: this names the conversation."""
    c = LLMIntentClassifier(_Gateway("summarize"))
    assert await c.classify("summarise our conversation so far", has_history=True) == "summarize"


@pytest.mark.asyncio
async def test_the_model_is_still_trusted_when_nothing_contradicts_it() -> None:
    """The guard is narrow: it only fires on a question that names identifiers. A genuine
    per-document request must still reach doc_summary."""
    c = LLMIntentClassifier(_Gateway("doc_summary"))
    assert await c.classify("summarize each document in detail", has_history=False) == "doc_summary"


@pytest.mark.asyncio
async def test_an_enumerating_question_the_model_already_got_right_is_untouched() -> None:
    c = LLMIntentClassifier(_Gateway("rag"))
    assert await c.classify(f"explain {_TEN}", has_history=False) == "rag"


@pytest.mark.asyncio
async def test_a_gateway_failure_still_fails_safe_to_rag() -> None:
    class _Broken:
        model_name = "broken"

        async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
            raise RuntimeError("gateway down")

        def stream(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> AsyncIterator[str]:
            raise NotImplementedError

        async def list_models(self) -> list[str]:
            return []

    c = LLMIntentClassifier(_Broken())
    assert await c.classify("what documents do we have?", has_history=False) == "rag"
