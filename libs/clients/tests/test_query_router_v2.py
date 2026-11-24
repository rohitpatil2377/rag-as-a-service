"""``LLMQueryRouter``: the gateway call, its one retry and its fail-safe (RAG v2 Phase 7).

What a routing decision CONTAINS is tested in ``libs/rag-core/tests/test_router.py``. What is
tested here is the adapter's contract, and the one way it differs in posture from every other
"one small JSON call" in this package: those return None on trouble and let the caller carry on
without them, while a router cannot — the pipeline has to route the question somewhere. So the
failure mode is the DETERMINISTIC backend's decision, which is the plan's own stated fallback and
is also exactly the pipeline's behaviour with the router switched off.
"""

from __future__ import annotations

import json
from typing import Any

from rag_clients.query_router_v2 import LLMQueryRouter
from rag_core.router import DocCard, RouterScope

_CARD = DocCard(
    doc_id="corrigendum-v.pdf",
    title="Corrigendum-V",
    doc_type="corrigendum",
    purpose="Amends the bidding document.",
    facet_ids=("financial_qualification", "eligibility"),
)
_SCOPE = RouterScope(cards=(_CARD,), explicit=True)
# An aggregate question, so the heuristic and a good model reply agree on the intent and a test
# asserting the FALLBACK has to distinguish them some other way — which is what `rationale` and
# the call count are for.
_QUERY = "list all the financial qualification requirements"


def _reply(**overrides: object) -> str:
    body: dict[str, object] = {
        "intent": "compare",
        "doc_ids": ["corrigendum-v.pdf"],
        "facets": [{"doc_id": "corrigendum-v.pdf", "facet_id": "eligibility"}],
        "sub_queries": ["what is the turnover floor?", "which certificates are required?"],
        "query_lang": "en",
        "answer_lang": "en",
        "needs_amendment_view": False,
        "rationale": "the model read the cards",
        "confidence": 0.9,
    }
    body.update(overrides)
    return json.dumps(body)


class _Gateway:
    """A scripted gateway: one reply per call, recording the prompts it was given."""

    def __init__(self, *replies: str | Exception) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        reply = self._replies.pop(0) if self._replies else ""
        if isinstance(reply, Exception):
            raise reply
        return reply

    def stream(self, prompt: str, **_: Any) -> Any:  # pragma: no cover — unused by this adapter
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "test-model"

    async def list_models(self) -> list[str]:
        return ["test-model"]


async def _route(gw: _Gateway, query: str = _QUERY, scope: RouterScope = _SCOPE) -> Any:
    return await LLMQueryRouter(gw).route(query, scope, query_lang="en")


async def test_a_usable_reply_is_the_decision() -> None:
    gw = _Gateway(_reply())
    decision = await _route(gw)

    assert len(gw.prompts) == 1
    assert decision.intent == "compare", "the model's reading, not the heuristic's"
    assert decision.facets == (("corrigendum-v.pdf", "eligibility"),)
    assert decision.rationale == "the model read the cards"


async def test_the_prompt_carries_the_cards() -> None:
    gw = _Gateway(_reply())
    await _route(gw)

    prompt = gw.prompts[0]
    assert "corrigendum-v.pdf" in prompt and "financial_qualification" in prompt
    assert _QUERY in prompt


async def test_an_unusable_reply_is_retried_once_with_the_reason() -> None:
    gw = _Gateway("not json at all", _reply())
    decision = await _route(gw)

    assert len(gw.prompts) == 2
    assert "could not be used" in gw.prompts[1] and "no JSON object" in gw.prompts[1]
    assert decision.intent == "compare"


async def test_a_second_unusable_reply_falls_back_to_the_heuristic() -> None:
    """Not to None, and not to an exception: the pipeline has to route the question somewhere."""
    gw = _Gateway("nope", "still nope")
    decision = await _route(gw)

    assert len(gw.prompts) == 2
    assert decision.intent == "aggregate", "the deterministic backend's reading of this query"
    assert decision.rationale.startswith("heuristic")


async def test_a_gateway_error_falls_back_without_a_retry() -> None:
    """There is nothing to retry a parse on, and a second call to a gateway that just raised is
    one more call on the LATENCY PATH for an answer already in hand."""
    gw = _Gateway(RuntimeError("model unavailable"))
    decision = await _route(gw)

    assert len(gw.prompts) == 1
    assert decision.intent == "aggregate" and decision.rationale.startswith("heuristic")


async def test_an_unknown_intent_falls_back_rather_than_being_coerced() -> None:
    gw = _Gateway(_reply(intent="enumerate"), _reply(intent="enumerate"))
    decision = await _route(gw)
    assert decision.rationale.startswith("heuristic")


async def test_targets_outside_the_scope_are_dropped_and_the_decision_still_used() -> None:
    """A model naming a document that is not in scope has made a bad decision in one field, not
    an unusable one — the entry is dropped and the rest of the decision stands."""
    gw = _Gateway(_reply(doc_ids=["corrigendum-v.pdf", "ghost.pdf"], facets=[{"doc_id": "ghost.pdf", "facet_id": "x"}]))
    decision = await _route(gw)

    assert decision.doc_ids == ("corrigendum-v.pdf",)
    assert decision.facets == ()
    assert decision.intent == "compare", "the usable fields survived"


async def test_an_empty_scope_still_routes() -> None:
    """Every KB has no cards until a document is ingested with the v2 flag on."""
    gw = _Gateway(_reply(doc_ids=[], facets=[]))
    decision = await LLMQueryRouter(gw).route(_QUERY, RouterScope(), query_lang="en")
    assert decision.intent == "compare" and decision.doc_ids == ()


async def test_the_callers_language_wins_over_the_replys() -> None:
    gw = _Gateway(_reply(query_lang="english"))
    assert (await _route(gw)).query_lang == "en"
