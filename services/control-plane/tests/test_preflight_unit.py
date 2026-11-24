"""Boot preflight: REAL serving-path probes + model attestation + optional strict mode.

Every HTTP call goes through the module's ``_client`` seam (monkeypatched to an
``httpx.MockTransport``), so no test here ever touches the network. Covers ``_probe``'s
classification (200 ok; 400/422/404/405/5xx = auth passed = ok; 401/403 = bad credential;
connection error = unreachable), that ``_build_probes`` hits each configured REMOTE backend's
serving path and skips local ones, ``attested_models``, strict mode (critical 401 raises;
non-strict returns the problem, no raise), and the REGRESSION this fixes — a backend that
401s on ``/models`` but 200/400s on its serving path must produce NO problem.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from svc_control_plane import preflight
from svc_control_plane.preflight import PreflightStrictError, attested_models

from rag_common.config import get_settings


@pytest.fixture(autouse=True)
def _clean(inmemory: None) -> None:
    get_settings.cache_clear()


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """Route the preflight probe client at an httpx.MockTransport (no network)."""

    def _factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(preflight, "_client", _factory)


def _status(code: int) -> Any:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(code)

    return handler


def _by_path(mapping: dict[str, int], default: int = 200) -> Any:
    """A handler that returns a status per URL-path suffix — used to model a gateway that gates
    ``/models`` differently from its real serving path (the exact live-observed regression)."""

    def handler(request: httpx.Request) -> httpx.Response:
        for suffix, code in mapping.items():
            if request.url.path.endswith(suffix):
                return httpx.Response(code)
        return httpx.Response(default)

    return handler


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("no route to host", request=request)


_CHAT = "/chat/completions"
_BODY: dict[str, object] = {"model": "m", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}


# --- auth header shape (must mirror the runtime adapters EXACTLY) --------------------------


def test_bearer_auth_prefixes_authorization_header() -> None:
    # OpenAICompatGateway/Vision: an Authorization header carries the Bearer scheme.
    assert preflight._bearer_auth("sk-abc", "Authorization") == {"Authorization": "Bearer sk-abc"}
    # An already-prefixed key is not double-prefixed.
    assert preflight._bearer_auth("Bearer sk-abc", "Authorization") == {"Authorization": "Bearer sk-abc"}
    # Any other header name takes the raw key (AIOps apikey), and an empty key = no header.
    assert preflight._bearer_auth("sk-abc", "apikey") == {"apikey": "sk-abc"}
    assert preflight._bearer_auth("", "Authorization") == {}


def test_raw_auth_never_prefixes() -> None:
    # Embedder/reranker/Chandra/Datalab send the key verbatim, even on an Authorization header.
    assert preflight._raw_auth("sk-abc", "Authorization") == {"Authorization": "sk-abc"}
    assert preflight._raw_auth("sk-abc", "X-API-Key") == {"X-API-Key": "sk-abc"}
    assert preflight._raw_auth("", "apikey") == {}


# --- _probe classification (the crux) -----------------------------------------------------


async def test_probe_200_no_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, _status(200))
    ok, detail = await preflight._probe("LLM", "http://llm.example/v1", {}, _CHAT, _BODY)
    assert ok is True and detail == ""


@pytest.mark.parametrize("code", [400, 422, 404, 405, 500, 503])
async def test_probe_non_auth_status_is_not_a_problem(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    # THE fix: a minimal/malformed payload that gets PAST auth (400/422/404/405) proves the
    # credential works — it must NOT be reported as a problem. 5xx (server error, not our
    # credential) is likewise not a credential/unreachable failure per the classification.
    _patch_client(monkeypatch, _status(code))
    ok, detail = await preflight._probe("Reranker", "http://rr.example", {"apikey": "x"}, "/rerank", _BODY)
    assert ok is True and detail == ""


@pytest.mark.parametrize("code", [401, 403])
async def test_probe_bad_credential_is_problem(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    _patch_client(monkeypatch, _status(code))
    ok, detail = await preflight._probe("Embedder", "http://embed.example", {"apikey": "x"}, "/embeddings", _BODY)
    assert ok is False
    assert str(code) in detail and "rejected" in detail


async def test_probe_connection_error_is_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, _connect_error)
    ok, detail = await preflight._probe("OCR", "http://chandra:8000/v1", {}, _CHAT, _BODY)
    assert ok is False and "unreachable" in detail


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("slow endpoint", request=request)


async def test_probe_timeout_is_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    # A slow-but-working VLM/OCR endpoint times out on a minimal probe; a bad credential 401s
    # FAST, never hangs — so a timeout must be inconclusive (reachable), NOT a problem, and must
    # not fail strict-mode boot.
    _patch_client(monkeypatch, _timeout)
    ok, detail = await preflight._probe(
        "OCR (chandra)", "https://models.example.internal/openai/v1", {"Authorization": "x"}, _CHAT, _BODY
    )
    assert ok is True and detail == ""


# --- _build_probes: remote covered on its serving path, local skipped ---------------------


async def test_probes_skipped_for_all_local_defaults() -> None:
    # Defaults: extractive_dev LLM, keyless TEI, cross_encoder rerank, echo vision, noop OCR,
    # lexical faithfulness — every one is local/in-cluster, so nothing is probed.
    probes = preflight._build_probes(get_settings())
    assert probes == []


async def test_probes_cover_each_configured_remote_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_LLM_API_KEY", "llm-key")
    monkeypatch.setenv("RAG_EMBED_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_EMBED_API_KEY", "embed-key")
    monkeypatch.setenv("RAG_RERANK_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_RERANK_API_KEY", "rerank-key")
    monkeypatch.setenv("RAG_RERANK_FALLBACK_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_RERANK_FALLBACK_URL", "http://rerank-fb.example")
    monkeypatch.setenv("RAG_VISION_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_OCR_BACKEND", "chandra")
    monkeypatch.setenv("RAG_FAITHFULNESS_BACKEND", "nli")
    get_settings.cache_clear()

    probes = preflight._build_probes(get_settings())
    names = {p.name for p in probes}
    assert names == {
        "Generation LLM",
        "Embedder",
        "Reranker",
        "Reranker (fallback)",
        "Vision LLM",
        "OCR (chandra)",
        "NLI faithfulness",
    }
    # criticality: generation / embedder / reranker are critical; the rest are not.
    critical = {p.name for p in probes if p.critical}
    assert critical == {"Generation LLM", "Embedder", "Reranker"}
    # Serving paths, never /models: the whole point of the fix.
    paths = {p.name: p.path for p in probes}
    assert paths["Generation LLM"] == "/chat/completions"
    assert paths["Embedder"] == "/embeddings"
    assert paths["Reranker"] == "/rerank"
    assert paths["NLI faithfulness"] == "/predict"
    assert all(p.path != "/models" for p in probes)


async def test_llm_probe_auth_matches_runtime_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    # RAG_LLM_API_KEY_HEADER defaults to "apikey" (raw); an Authorization header must Bearer-prefix.
    monkeypatch.setenv("RAG_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_LLM_API_KEY", "sk-xyz")
    monkeypatch.setenv("RAG_LLM_API_KEY_HEADER", "Authorization")
    get_settings.cache_clear()
    (llm,) = [p for p in preflight._build_probes(get_settings()) if p.name == "Generation LLM"]
    assert llm.headers == {"Authorization": "Bearer sk-xyz"}


async def test_keyless_tei_embedder_not_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A bare TEI with no key is self-hosted in-cluster — nothing hosted to authenticate, skip it.
    monkeypatch.setenv("RAG_EMBED_BACKEND", "tei")
    monkeypatch.delenv("RAG_EMBED_API_KEY", raising=False)
    get_settings.cache_clear()
    assert preflight._build_probes(get_settings()) == []


# --- attested_models ----------------------------------------------------------------------


async def test_attested_models_defaults_map() -> None:
    attested = attested_models(get_settings())
    assert set(attested) == {"generation", "embedding", "reranking", "ocr", "vision", "faithfulness"}
    # Local defaults report "@ local"; the remote (in-cluster TEI) reports its host.
    assert attested["generation"] == "extractive_dev @ local"
    assert attested["embedding"] == "BAAI/bge-m3 @ tei:80"
    assert attested["reranking"] == "cross-encoder/ms-marco-MiniLM-L-6-v2 @ reranker:80"
    assert attested["ocr"] == "noop @ local"
    assert attested["vision"] == "echo @ local"
    assert attested["faithfulness"] == "lexical @ local"


async def test_attested_models_remote_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_LLM_MODEL", "sarvam-105b")
    monkeypatch.setenv("RAG_VLLM_URL", "https://api.sarvam.ai/v1")
    monkeypatch.setenv("RAG_FAITHFULNESS_BACKEND", "nli")
    get_settings.cache_clear()
    attested = attested_models(get_settings())
    assert attested["generation"] == "sarvam-105b @ api.sarvam.ai"
    assert attested["faithfulness"] == "cross-encoder/nli-deberta-v3-base @ nli-checker:80"


# --- run(): strict vs non-strict ----------------------------------------------------------


async def test_run_non_strict_returns_problem_no_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_LLM_API_KEY", "bad-key")
    monkeypatch.setenv("RAG_VLLM_URL", "https://api.sarvam.ai/v1")
    get_settings.cache_clear()
    _patch_client(monkeypatch, _status(401))

    problems = await preflight.run()  # default preflight_strict=False → never raises
    assert any("Generation LLM" in p and "rejected" in p for p in problems)


async def test_run_strict_raises_on_critical_bad_cred(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PREFLIGHT_STRICT", "true")
    monkeypatch.setenv("RAG_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_LLM_API_KEY", "bad-key")
    monkeypatch.setenv("RAG_VLLM_URL", "https://api.sarvam.ai/v1")
    get_settings.cache_clear()
    _patch_client(monkeypatch, _status(401))

    with pytest.raises(PreflightStrictError):
        await preflight.run()


async def test_run_strict_raises_on_critical_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    # A truly unreachable critical backend still fails boot in strict mode.
    monkeypatch.setenv("RAG_PREFLIGHT_STRICT", "true")
    monkeypatch.setenv("RAG_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_LLM_API_KEY", "any-key")
    monkeypatch.setenv("RAG_VLLM_URL", "https://api.sarvam.ai/v1")
    get_settings.cache_clear()
    _patch_client(monkeypatch, _connect_error)

    with pytest.raises(PreflightStrictError):
        await preflight.run()


async def test_run_strict_no_raise_when_only_noncritical_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # OCR is non-critical: even in strict mode a bad OCR credential must not fail boot.
    monkeypatch.setenv("RAG_PREFLIGHT_STRICT", "true")
    monkeypatch.setenv("RAG_OCR_BACKEND", "chandra")
    get_settings.cache_clear()
    _patch_client(monkeypatch, _status(401))

    problems = await preflight.run()
    assert any("OCR (chandra)" in p for p in problems)


async def test_run_no_problems_when_all_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_LLM_API_KEY", "good-key")
    monkeypatch.setenv("RAG_VLLM_URL", "https://api.sarvam.ai/v1")
    get_settings.cache_clear()
    _patch_client(monkeypatch, _status(200))

    assert await preflight.run() == []


# --- REGRESSION: 401 on /models but 200/400 on the serving path → NO problem --------------


async def test_run_gemma_401_on_models_ok_on_serving_path_no_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    # The exact false positive this patch fixes: the live gateway 401s on /models but serves
    # real chat completions fine. The OLD /models probe cried wolf (and would have failed strict
    # boot); the serving-path probe must see 200 and report NO problem — even under strict mode.
    monkeypatch.setenv("RAG_PREFLIGHT_STRICT", "true")
    monkeypatch.setenv("RAG_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_LLM_API_KEY", "gemma-key")
    monkeypatch.setenv("RAG_VLLM_URL", "https://models.example.internal/v1")
    monkeypatch.setenv("RAG_LLM_MODEL", "gemma4-31b")
    get_settings.cache_clear()
    _patch_client(monkeypatch, _by_path({"/models": 401, "/chat/completions": 200}))

    assert await preflight.run() == []  # no raise, no problem


async def test_run_bge_reranker_400_on_serving_path_no_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    # bge-reranker: 401 on /models, but a minimal /rerank payload is refused with 400 (auth
    # passed). A 400 is NOT a problem — the credential works.
    monkeypatch.setenv("RAG_PREFLIGHT_STRICT", "true")
    monkeypatch.setenv("RAG_RERANK_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_RERANK_API_KEY", "bge-key")
    monkeypatch.setenv("RAG_RERANK_URL", "https://models.example.internal/rerank/v1")
    get_settings.cache_clear()
    _patch_client(monkeypatch, _by_path({"/models": 401, "/rerank": 400}))

    assert await preflight.run() == []  # no raise, no problem


async def test_run_gemma_and_bge_both_ok_on_serving_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both critical backends 401 on /models but answer on their serving paths (chat 200, rerank
    # 400). This is the deployed reality — the detector must stay silent.
    monkeypatch.setenv("RAG_PREFLIGHT_STRICT", "true")
    monkeypatch.setenv("RAG_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_LLM_API_KEY", "gemma-key")
    monkeypatch.setenv("RAG_VLLM_URL", "https://models.example.internal/v1")
    monkeypatch.setenv("RAG_RERANK_BACKEND", "openai_compat")
    monkeypatch.setenv("RAG_RERANK_API_KEY", "bge-key")
    monkeypatch.setenv("RAG_RERANK_URL", "https://models.example.internal/rerank/v1")
    get_settings.cache_clear()
    _patch_client(monkeypatch, _by_path({"/models": 401, "/chat/completions": 200, "/rerank": 400}))

    assert await preflight.run() == []
