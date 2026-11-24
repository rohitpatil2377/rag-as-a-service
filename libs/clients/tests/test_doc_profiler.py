"""``LLMDocProfiler``: the gateway call, its retries and its fail-safe (RAG v2 Phase 3).

The card's CONTENT is not tested here and cannot be — see the header of
``libs/rag-core/tests/test_doc_profile.py``. What is tested is the adapter's contract: it never
raises, it retries only when a retry could help, it stops at the configured bound, and each retry
carries a reason rather than re-sending the identical prompt.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from rag_clients.doc_profiler import LLMDocProfiler

_SOURCE = "This corrigendum amends Corrigendum-II dated 31.08.2026."
_GOOD = json.dumps(
    {
        "doc_type": "corrigendum",
        "purpose": "Amends the bidding document.",
        "relations": [{"type": "amends", "target_ref": "Corrigendum-II", "evidence_quote": "amends Corrigendum-II"}],
        "facet_taxonomy": [{"facet_id": "eligibility", "name": "Eligibility", "description": "d"}],
    }
)


class _Gateway:
    """A scripted gateway: one reply per call, and it records the prompts it was given."""

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


@pytest.fixture(autouse=True)
def _two_retries(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the retry budget so these tests assert a contract, not the shipped default."""
    monkeypatch.setenv("RAG_DOC_PROFILE_RETRIES", "2")
    from rag_common import config

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


async def _profile(gateway: _Gateway) -> Any:
    return await LLMDocProfiler(gateway).profile("PROMPT", source_text=_SOURCE, title="Corrigendum-V", doc_id="d1")


async def test_a_usable_reply_is_parsed_on_the_first_call() -> None:
    gw = _Gateway(_GOOD)
    profile = await _profile(gw)
    assert profile is not None and profile.doc_type == "corrigendum"
    assert len(gw.prompts) == 1


async def test_an_unusable_reply_is_retried_and_can_recover() -> None:
    gw = _Gateway("I'm not sure what this document is.", _GOOD)
    profile = await _profile(gw)
    assert profile is not None
    assert len(gw.prompts) == 2, "it must stop as soon as it has a usable card"


async def test_it_keeps_trying_up_to_the_configured_bound() -> None:
    """The reason more than one retry is worth having: these are INDEPENDENT samples.

    The adapter used to allow exactly one, reasoning that a greedy decode re-reading the same
    prompt writes the same bytes. `OpenAICompatGateway._eff_temp` raises a requested 0.0 to 0.6
    whenever reasoning is suppressed — which `RAG_LLM_REASONING_EFFORT=off` does for every
    model — so that premise does not hold and a later attempt really can succeed."""
    gw = _Gateway("no", "still no", _GOOD)
    assert await _profile(gw) is not None
    assert len(gw.prompts) == 3


async def test_it_gives_up_at_the_bound_rather_than_looping() -> None:
    gw = _Gateway("no", "still no", "nope")
    assert await _profile(gw) is None
    assert len(gw.prompts) == 3, "1 initial + 2 retries, and no more"


def _retry_prompt(gw: _Gateway) -> str:
    return gw.prompts[1]


async def test_the_retry_says_what_was_wrong_instead_of_resending_the_same_prompt() -> None:
    """Naming the fault is what makes a retry a different PROMPT as well as a different sample."""
    gw = _Gateway("not json at all", _GOOD)
    await _profile(gw)
    retry = _retry_prompt(gw)
    assert retry.startswith("PROMPT")
    assert retry != "PROMPT"
    assert "no JSON object" in retry


async def test_a_gateway_failure_returns_none_and_is_not_retried() -> None:
    """A parse failure is worth one more call; an unreachable model is not — there is nothing
    different to send it."""
    gw = _Gateway(TimeoutError("model unreachable"), _GOOD)
    assert await _profile(gw) is None
    assert len(gw.prompts) == 1


async def test_a_gateway_failure_never_propagates() -> None:
    """A document with no profile indexes exactly as Phase 5 indexed it; a model outage must not
    become an ingest outage."""
    assert await _profile(_Gateway(RuntimeError("boom"))) is None


async def test_a_relation_the_model_did_not_quote_is_dropped_without_losing_the_card() -> None:
    reply = json.dumps(
        {
            "doc_type": "corrigendum",
            "purpose": "p",
            "relations": [{"type": "amends", "target_ref": "X", "evidence_quote": "text that is not present"}],
            "facet_taxonomy": [],
        }
    )
    profile = await _profile(_Gateway(reply))
    assert profile is not None and profile.relations == ()


class _PolicyGateway(_Gateway):
    """Records the sampling task, like `OpenAICompatGateway` would act on it."""

    honours_sampling_task = True

    def __init__(self, *replies: str | Exception) -> None:
        super().__init__(*replies)
        self.tasks: list[str | None] = []

    async def complete_with_usage(self, prompt: str, **kw: Any) -> tuple[str, dict[str, Any]]:
        self.tasks.append(kw.get("task"))
        return await self.complete(prompt), {"completion_tokens": 300, "finish_reason": "stop"}


async def test_the_profile_call_is_classified_as_structured_extraction() -> None:
    """THE CALL WHERE IT MATTERS MOST. This reply becomes the document's facet vocabulary, and a
    facet id is Phase 8's facet-union join key while its label is a cross-encoder probe -- so when
    this call was being nudged off greedy, three ingests of identical bytes produced three
    10-facet vocabularies sharing ONE id, and section selection moved with them."""
    from rag_core import sampling

    gw = _PolicyGateway(_GOOD)
    parsed = await _profile(gw)

    assert parsed is not None
    assert [f.facet_id for f in parsed.facets] == ["eligibility"], "the vocabulary this call invents"
    assert gw.tasks == [sampling.STRUCTURED_EXTRACTION]
