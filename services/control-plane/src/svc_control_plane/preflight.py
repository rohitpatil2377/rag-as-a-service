"""Startup preflight: report every configured model backend that cannot actually serve.

Three separate outages on one deployment came from the same shape — a backend configured but
not usable, discovered only when a user hit the feature:

* the embedder fell back to the 384-dim default against 1024-dim collections, surfacing as a
  raw Qdrant ``expected dim: 1024, got 384`` inside the chat window;
* the vision gateway had no credential, so every image ingest 401'd while text answers from the
  SAME endpoint worked;
* OCR pointed at ``http://chandra:8000/v1``, a container that does not exist, so scanned PDFs
  silently produced no text;
* the generation model returned a planning preamble instead of JSON, because
  ``RAG_LLM_REASONING_EFFORT`` was unset — 33 of 39 RAG v2 map calls failed, the pipeline
  degraded exactly as designed, and the aggregate answer covered 0 of 40 checklist items while
  reporting success.

None of them announced themselves at boot. This runs once at startup, checks what is REACHABLE
rather than what is merely set, probes EVERY configured remote model endpoint concurrently, and
logs one WARNING per problem naming the exact env var(s) to fix. It also attests the running
model set as one info line (``preflight_models_in_use``) so the ACTUALLY-deployed models are
explicit in the logs — killing docs-vs-deployed drift.

By default it never blocks boot: a degraded platform that says so is more useful than one that
refuses to start, and some backends are legitimately absent on a given profile. When
``preflight_strict`` is set, an unusable CRITICAL backend (generation LLM / embedder / reranker)
raises :class:`PreflightStrictError` so a broken deploy fails loud instead of degrading silently.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from rag_clients.factory import get_vector_store
from rag_common import get_logger, get_settings, model_tls_verify
from rag_common.config import Settings
from rag_core.json_reply import extract_json_object

_log = get_logger(__name__)
_PROBE_TIMEOUT = 10.0
# A minimal payload every serving-path probe sends. The point is only to get PAST auth: the
# server is expected to answer (200) or reject the tiny/partial request on validation (400/422),
# both of which prove the credential works. See :func:`_probe`.
_PING = "ping"

# Backends served locally / in-memory — never a remote endpoint, so nothing to probe.
_LOCAL_BACKENDS = frozenset({"echo", "extractive_dev", "lexical", "noop", "local"})


class PreflightStrictError(RuntimeError):
    """Raised by :func:`run` in strict mode when a CRITICAL backend cannot serve.

    Only raised when ``preflight_strict`` is set; the lifespan lets it propagate to fail boot."""


@dataclass(frozen=True, slots=True)
class _Probe:
    """One remote endpoint to probe at boot, on its REAL serving path (not ``/models``).

    ``path`` + ``body`` describe the cheapest real request for this backend, built to mirror
    the runtime adapter's request shape and — via ``headers`` — its auth EXACTLY, so the probe
    passes or fails auth for the identical reason a live query would."""

    name: str
    # The ROLE this backend serves, spelled exactly as the console's "Models in use" table spells
    # it, so a probe result can be shown against its row without a second mapping table to drift.
    role: str
    url: str
    headers: dict[str, str]
    path: str  # the serving-path endpoint, e.g. "/chat/completions" / "/embeddings" / "/rerank"
    body: dict[str, object]  # a minimal request body the endpoint accepts past auth
    hint: str  # the exact env var(s) to fix, appended to the failure warning
    critical: bool  # a critical backend failing can fail boot in strict mode


def _client() -> httpx.AsyncClient:
    """Seam: the probe HTTP client. Overridden in tests with an ``httpx.MockTransport``.

    Uses the same TLS verification as the runtime model adapters (``model_tls_verify`` honours
    ``RAG_MODEL_CA_BUNDLE`` for private-CA model hosts), so a probe never fails a handshake a
    real query would pass, nor passes one a real query would fail."""
    return httpx.AsyncClient(timeout=_PROBE_TIMEOUT, verify=model_tls_verify())


def _host(url: str) -> str:
    """The bare host of a model URL — non-secret (keys live in headers, never the URL). Falls
    back to 'in-cluster' for internal service names (no dots), which never leave the deployment."""
    host = urlparse(url).netloc or url
    host = host.split("@")[-1]  # defensively drop any user:pass@ if ever present
    return host or "in-cluster"


def _bearer_auth(key: str, header: str) -> dict[str, str]:
    """OpenAI-compatible auth, byte-for-byte as :class:`OpenAICompatGateway` /
    :class:`OpenAICompatVisionGateway` build it: an ``Authorization`` header carries the Bearer
    scheme (a bare token there is 403'd), any other header name (``apikey``, …) takes the raw
    key. Empty key → no header (self-hosted vLLM/llama.cpp need none)."""
    if not key:
        return {}
    if header.lower() == "authorization" and not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {header: key}


def _raw_auth(key: str, header: str) -> dict[str, str]:
    """Raw-key auth: the key travels verbatim in ``header``. Mirrors the embedder, reranker,
    Chandra and Datalab adapters, which never add a Bearer prefix (AIOps ``apikey``, Datalab
    ``X-API-Key``). Empty key → no header (keyless in-cluster TEI)."""
    return {header: key} if key else {}


def _chat_body(model: str) -> dict[str, object]:
    """A one-token chat completion — the cheapest real op on any OpenAI-compatible chat server
    (Generation LLM, and the vision/Chandra VLMs, which accept a text-only message)."""
    return {"model": model, "messages": [{"role": "user", "content": _PING}], "max_tokens": 1}


def _rerank_body(model: str, api_shape: str) -> dict[str, object]:
    """A minimal rerank request in the wire shape the connection speaks (mirrors
    :class:`OpenAICompatReranker`): the ``cohere`` body carries ``model`` + ``documents``,
    the ``tei`` body carries ``texts``. The ``+sigmoid`` suffix is a score transform, not a
    wire difference, so it is ignored here."""
    wire = api_shape.split("+", 1)[0]
    if wire == "cohere":
        return {"model": model, "query": _PING, "documents": [_PING]}
    return {"query": _PING, "texts": [_PING], "return_text": False}


async def _dim_matches_collections() -> tuple[bool, str]:
    """Does the configured embedding width match what the live collections were built at?

    A mismatch is unrecoverable at query time — every search fails — and it is invisible until
    someone asks a question, so it is worth one probe at boot."""
    settings = get_settings()
    store = get_vector_store()
    dims = getattr(store, "collection_dims", None)
    if dims is None:  # in-memory twin, or an adapter that cannot introspect
        return True, ""
    try:
        live = await dims()
    except Exception as exc:  # a probe must never break startup
        return True, f"(could not read collection dims: {str(exc)[:80]})"
    bad = {name: d for name, d in live.items() if d and d != settings.embed_dim}
    if not bad:
        return True, ""
    return False, (
        f"RAG_EMBED_DIM={settings.embed_dim} but {len(bad)} collection(s) were built at a "
        f"different width, e.g. {list(bad.items())[:3]} — every query against those will fail. "
        "Check the model overlay is loaded (make compose-files) or re-index at the new width."
    )


async def _probe(name: str, url: str, headers: dict[str, str], path: str, body: dict[str, object]) -> tuple[bool, str]:
    """Hit a backend's REAL serving path with a minimal request and classify the outcome.

    This replaces the old ``GET /models`` check, which false-positived: on the live gateways
    (poc-aiops, uat-bodhi) ``/models`` is gated differently from the serving path, so it 401'd
    for backends (gemma-31b generation, bge reranker) that answer real queries fine — and would
    have FALSELY failed boot under ``preflight_strict``.

    Classification (the crux):

    * **401 / 403** → the credential is wrong: a real problem (names the env var to fix).
    * **connection refused / DNS** → the host is unreachable: a real problem.
    * **timeout** → INCONCLUSIVE (a slow-but-working VLM/OCR endpoint, not a bad credential —
      those 401 fast): treated as reachable, never a problem.
    * **any other HTTP outcome** — 200, and crucially 400 / 422 / 404 / 405 / 5xx → the request
      got PAST auth. A minimal or malformed payload the server rejects on VALIDATION proves the
      credential works, so the backend is reachable AND authed: NOT a problem. A 400 is not a
      credential failure.

    Never raises — a probe reports, it does not break boot (strict mode decides that in
    :func:`run`, and only on a real 401/403/unreachable, never on a false ``/models`` 401)."""
    try:
        async with _client() as c:
            r = await c.post(f"{url.rstrip('/')}{path}", headers=headers, json=body)
            r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in (401, 403):
            return False, f"{name} at {url} rejected the configured credential (HTTP {code})"
        # 400/422/404/405/5xx: the minimal payload got past auth and was refused on its merits.
        return True, ""
    except httpx.TimeoutException:
        # A timeout is INCONCLUSIVE, not a failure: OCR/VLM endpoints are legitimately slow to a
        # minimal probe, while a bad credential returns a FAST 401 (caught above), never a hang. So
        # a slow-but-working backend must not read as "unusable" — and must not fail strict-mode boot.
        return True, ""
    except Exception as exc:  # connection refused / DNS — genuinely unreachable
        return False, f"{name} at {url} is unreachable ({type(exc).__name__}: {str(exc)[:90]})"
    return True, ""


def _vision_auth(s: Settings) -> dict[str, str]:
    """The vision gateway's exact runtime auth (mirrors :class:`OpenAICompatVisionGateway`),
    including its credential-reuse: a keyless vision endpoint that IS the text endpoint borrows
    ``RAG_LLM_API_KEY`` — the same reason the live vision path answered text but 401'd on images."""
    key, header = s.vision_llm_api_key, s.vision_llm_api_key_header
    if not key and s.vision_llm_url.rstrip("/") == s.vllm_url.rstrip("/") and s.llm_api_key:
        key, header = s.llm_api_key, s.llm_api_key_header
    return _bearer_auth(key, header)


def _build_probes(s: Settings) -> list[_Probe]:
    """Every configured REMOTE model endpoint to probe, each on its REAL serving path with the
    runtime adapter's request + auth. Local/in-memory backends and a bare keyless in-cluster TEI
    are deliberately skipped (nothing hosted to authenticate against)."""
    probes: list[_Probe] = []

    # Generation LLM — any remote OpenAI-compatible / vLLM / llama.cpp server (critical).
    # Real op: a 1-token chat completion, exactly what OpenAICompatGateway.complete sends.
    if s.llm_backend not in _LOCAL_BACKENDS:
        probes.append(
            _Probe(
                name="Generation LLM",
                role="Generation",
                url=s.vllm_url,
                headers=_bearer_auth(s.llm_api_key, s.llm_api_key_header),
                path="/chat/completions",
                body=_chat_body(s.llm_model),
                hint="set RAG_LLM_API_KEY / RAG_VLLM_URL, or RAG_LLM_BACKEND=extractive_dev",
                critical=True,
            )
        )

    # Embedder — openai_compat, or a TEI endpoint that carries a key (hosted) (critical).
    # Real op: embed ["ping"], exactly what OpenAICompatEmbedder.embed sends (raw-key auth).
    if s.embed_backend == "openai_compat" or (s.embed_backend == "tei" and s.embed_api_key):
        probes.append(
            _Probe(
                name="Embedder",
                role="Embedding",
                url=s.tei_url,
                headers=_raw_auth(s.embed_api_key, s.embed_api_key_header),
                path="/embeddings",
                body={"model": s.embed_model, "input": [_PING]},
                hint="set RAG_EMBED_API_KEY / RAG_TEI_URL",
                critical=True,
            )
        )

    # Reranker — the primary sets the final order + abstain confidence (critical). Real op: a
    # one-document /rerank in the connection's wire shape, as OpenAICompatReranker.rerank sends.
    if s.rerank_backend == "openai_compat":
        probes.append(
            _Probe(
                name="Reranker",
                role="Reranking",
                url=s.rerank_url,
                headers=_raw_auth(s.rerank_api_key, s.rerank_api_key_header),
                path="/rerank",
                body=_rerank_body(s.rerank_model, s.rerank_api_shape),
                hint="set RAG_RERANK_API_KEY / RAG_RERANK_URL",
                critical=True,
            )
        )
    # Reranker fallback — a degraded secondary, not like-for-like, so non-critical.
    if s.rerank_fallback_backend == "openai_compat":
        probes.append(
            _Probe(
                name="Reranker (fallback)",
                role="Reranking",
                url=s.rerank_fallback_url,
                headers=_raw_auth(s.rerank_fallback_api_key, s.rerank_fallback_api_key_header),
                path="/rerank",
                body=_rerank_body(s.rerank_fallback_model, s.rerank_fallback_api_shape),
                hint="set RAG_RERANK_FALLBACK_API_KEY / RAG_RERANK_FALLBACK_URL, or clear RAG_RERANK_FALLBACK_BACKEND",
                critical=False,
            )
        )

    # Vision LLM — a remote VLM (non-critical). No image needed to test AUTH: chandra/gemma
    # accept a text-only chat message, so a 1-token text completion on the SAME serving path
    # exercises the credential without a picture. Routed to each backend's real host + auth.
    if s.vision_llm_backend == "openai_compat":
        probes.append(
            _Probe(
                name="Vision LLM",
                role="Vision",
                url=s.vision_llm_url,
                headers=_vision_auth(s),
                path="/chat/completions",
                body=_chat_body(s.vision_llm_model),
                hint="set RAG_VISION_LLM_API_KEY / RAG_VISION_LLM_URL, or RAG_VISION_LLM_BACKEND=echo",
                critical=False,
            )
        )
    elif s.vision_llm_backend == "chandra":
        probes.append(
            _Probe(
                name="Vision LLM",
                role="Vision",
                url=s.chandra_url,
                headers=_raw_auth(s.chandra_api_key, s.chandra_api_key_header),
                path="/chat/completions",
                body=_chat_body(s.chandra_model),
                hint="set RAG_CHANDRA_URL / RAG_CHANDRA_API_KEY, or RAG_VISION_LLM_BACKEND=echo",
                critical=False,
            )
        )
    elif s.vision_llm_backend == "chandra_api":
        # Hosted Datalab: no OpenAI-compatible chat path (multipart /convert only). See the OCR
        # chandra_api note below for why a fileless POST /convert still tests the credential.
        probes.append(
            _Probe(
                name="Vision LLM",
                role="Vision",
                url=s.datalab_url,
                headers=_raw_auth(s.datalab_api_key, s.datalab_api_key_header),
                path="/convert",
                body={},
                hint="set RAG_DATALAB_API_KEY / RAG_DATALAB_URL, or RAG_VISION_LLM_BACKEND=echo",
                critical=False,
            )
        )

    # OCR — a host that does not exist produces silently blank scanned PDFs, the worst failure
    # because ingest still reports success (non-critical).
    if s.ocr_backend == "chandra":
        # Self-hosted Chandra via vLLM: OpenAI-compatible, so a text-only chat completion tests
        # auth on the exact serving path ChandraClient uses (raw-key auth, no Bearer).
        probes.append(
            _Probe(
                name="OCR (chandra)",
                role="OCR / Vision",
                url=s.chandra_url,
                headers=_raw_auth(s.chandra_api_key, s.chandra_api_key_header),
                path="/chat/completions",
                body=_chat_body(s.chandra_model),
                hint="set RAG_CHANDRA_URL / RAG_CHANDRA_API_KEY, or RAG_OCR_BACKEND=noop",
                critical=False,
            )
        )
    elif s.ocr_backend == "chandra_api":
        # Hosted Datalab /convert is NOT OpenAI-compatible and needs a file for a true OCR op, so
        # there is no clean text-only call. A minimal POST /convert with no file still separates
        # the cases we classify on: no/invalid key → 401/403 (credential problem), a valid key
        # with no file → 400/422 (auth passed, request refused on validation → NOT a problem).
        probes.append(
            _Probe(
                name="OCR (chandra)",
                role="OCR / Vision",
                url=s.datalab_url,
                headers=_raw_auth(s.datalab_api_key, s.datalab_api_key_header),
                path="/convert",
                body={},
                hint="set RAG_DATALAB_URL / RAG_DATALAB_API_KEY, or RAG_OCR_BACKEND=noop",
                critical=False,
            )
        )

    # NLI faithfulness checker — no auth today, but still catch an unreachable host (non-critical).
    # Real op: a minimal /predict pair, exactly what TEINLIScorer sends.
    if s.faithfulness_backend == "nli":
        probes.append(
            _Probe(
                name="NLI faithfulness",
                role="Faithfulness (NLI)",
                url=s.nli_url,
                headers={},
                path="/predict",
                body={"inputs": [_PING, _PING]},
                hint="set RAG_NLI_URL, or RAG_FAITHFULNESS_BACKEND=lexical",
                critical=False,
            )
        )

    return probes


@dataclass(frozen=True, slots=True)
class BackendHealth:
    """One probed backend, as the console shows it: is it answering, and how fast."""

    name: str  # the probe's name, e.g. "Generation LLM", "Reranker (fallback)"
    role: str  # the roster role it serves, as the console's table spells it
    endpoint: str  # bare host — non-secret (keys travel in headers, never the URL)
    online: bool
    latency_ms: int | None  # round trip of the probe request; None when it never connected
    detail: str  # "" when healthy, else why it is not
    # The model id the probe actually asked for. The console's roster reads CONFIG; a KB may
    # resolve a different model for the same role (its own attach, or the org-wide default), and a
    # row that named the configured model beside this row's health would be two answers to one
    # question. Whoever probed says which model it reached.
    model: str = ""


async def health(settings: Settings) -> list[BackendHealth]:
    """Probe every configured remote backend NOW and time each one.

    The same probes the boot preflight runs (:func:`_build_probes`) and the same classification
    (:func:`_probe`) — one prober, so the console and the boot log can never disagree about
    whether a backend is usable. Probes run CONCURRENTLY: a dead endpoint costs one timeout, not
    one timeout per backend.

    Why this exists at all: until now a wrong key or an expired credential on embedding,
    reranking, OCR or NLI was invisible until a query or an ingest failed on it. Measured live,
    a reranker fallback sat misconfigured with no indicator anywhere in the product."""

    async def _timed(probe: _Probe) -> BackendHealth:
        started = time.perf_counter()
        ok, detail = await _probe(probe.name, probe.url, probe.headers, probe.path, probe.body)
        elapsed = int((time.perf_counter() - started) * 1000)
        return BackendHealth(
            name=probe.name,
            role=probe.role,
            endpoint=_host(probe.url),
            online=ok,
            # A failed probe's elapsed time is the timeout, not a latency — reporting it as one
            # would put "30000 ms" beside an endpoint that never answered.
            latency_ms=elapsed if ok else None,
            detail=detail,
            model=str(probe.body.get("model", "")),
        )

    probes = _build_probes(settings)
    return list(await asyncio.gather(*(_timed(p) for p in probes))) if probes else []


def attested_models(settings: Settings) -> dict[str, str]:
    """The ACTUAL running model per role as ``"<model> @ <endpoint-host>"`` — logged once at boot
    so deployed reality is explicit in the logs (docs-vs-deployed drift is otherwise invisible)."""
    s = settings

    generation = (
        f"{s.llm_backend} @ local" if s.llm_backend in _LOCAL_BACKENDS else f"{s.llm_model} @ {_host(s.vllm_url)}"
    )
    embedding = (
        f"{s.embed_backend} @ local" if s.embed_backend in _LOCAL_BACKENDS else f"{s.embed_model} @ {_host(s.tei_url)}"
    )
    reranking = (
        f"{s.rerank_backend} @ local"
        if s.rerank_backend in _LOCAL_BACKENDS
        else f"{s.rerank_model} @ {_host(s.rerank_url)}"
    )
    if s.ocr_backend == "chandra_api":
        ocr = f"{s.chandra_model} @ {_host(s.datalab_url)}"
    elif s.ocr_backend == "chandra":
        ocr = f"{s.chandra_model} @ {_host(s.chandra_url)}"
    else:
        ocr = f"{s.ocr_backend} @ local"
    vision = (
        f"{s.vision_llm_backend} @ local"
        if s.vision_llm_backend in _LOCAL_BACKENDS
        else f"{s.vision_llm_model} @ {_host(s.vision_llm_url)}"
    )
    faithfulness = (
        f"{s.nli_model} @ {_host(s.nli_url)}"
        if s.faithfulness_backend == "nli"
        else f"{s.faithfulness_backend} @ local"
    )
    return {
        "generation": generation,
        "embedding": embedding,
        "reranking": reranking,
        "ocr": ocr,
        "vision": vision,
        "faithfulness": faithfulness,
    }


#: Asked of the generation model at boot. Tiny on purpose — the question is not whether the model
#: is clever but whether its reply is MACHINE-READABLE at all under the deployment's own settings.
_JSON_PROBE_PROMPT = 'Return this exact JSON object and nothing else: {"ok": true}'


async def _generation_returns_json() -> tuple[bool, str]:
    """Can the configured generation model return a bare JSON object?

    THE FOURTH OUTAGE IN THIS MODULE'S HEADER, and the only one that is not about reachability:
    the endpoint answers, auth works, the model is correct, and every structured call still fails
    because the reply opens with the model's own deliberation. RAG v2's map, reduce, router,
    doc-profile and section-enrich stages all parse JSON out of a completion; when a preamble
    appears in front of it they each fail SOFT and the answer comes back confident and empty.
    Measured: without ``RAG_LLM_REASONING_EFFORT=off`` on gemma4-31b, 33 of 39 map calls
    returned prose and the aggregate answer covered 0 of 40 gold items.

    THROUGH THE REAL GATEWAY, not an httpx probe like its neighbours, and that is the point. The
    behaviour being tested is produced by the gateway itself — ``_eff_temp`` nudging temperature,
    the reasoning-suppression body, the think-tag stripper — so only the real path reproduces it.
    A raw POST would test the endpoint and miss the defect entirely.

    NOT CRITICAL and never blocks boot: a deployment whose model prefers prose is degraded for
    structured features and perfectly usable for plain answering, which is a judgement for the
    operator reading the warning."""
    settings = get_settings()
    if settings.llm_backend in _LOCAL_BACKENDS:
        return True, ""
    try:
        from rag_clients.factory import get_llm_gateway  # noqa: PLC0415 — cost only when remote

        reply = await get_llm_gateway().complete(_JSON_PROBE_PROMPT, max_tokens=64)
    except Exception as exc:  # reachability is the other probes' job, not this one's
        return True, f"(json shape not checked: {type(exc).__name__})"
    if extract_json_object(reply) is not None:
        return True, ""
    return False, (
        f"generation model {settings.llm_model!r} did not return JSON for a request that asked "
        f"for nothing else (got {reply[:80]!r}...) — every RAG v2 stage parses JSON out of a "
        f"completion and will fail soft, producing confident EMPTY answers"
    )


async def run() -> list[str]:
    """Log one warning per unusable backend + one attestation line. Returns the problems, for
    tests and /system/info. Never blocks boot unless ``preflight_strict`` is set AND a CRITICAL
    backend (generation LLM / embedder / reranker) is unusable — then raises PreflightStrictError."""
    settings = get_settings()
    endpoint_probes = _build_probes(settings)

    # All probes concurrently so total boot latency stays ~one _PROBE_TIMEOUT, not the sum.
    results = await asyncio.gather(
        _dim_matches_collections(),
        _generation_returns_json(),
        *[_probe(p.name, p.url, p.headers, p.path, p.body) for p in endpoint_probes],
    )
    (dim_ok, dim_detail) = results[0]
    (json_ok, json_detail) = results[1]

    problems: list[str] = []
    critical_problems: list[str] = []
    if not dim_ok:
        problems.append(dim_detail)
    if not json_ok:
        problems.append(f"{json_detail} — set RAG_LLM_REASONING_EFFORT=off (or the equivalent for this model)")

    for probe, (ok, detail) in zip(endpoint_probes, results[2:], strict=True):
        if ok:
            continue
        message = f"{detail} — {probe.hint}"
        problems.append(message)
        if probe.critical:
            critical_problems.append(message)

    for problem in problems:
        _log.warning("preflight_backend_unusable", detail=problem)

    _log.info("preflight_models_in_use", **attested_models(settings))

    if not problems:
        _log.info("preflight_ok", embed_dim=settings.embed_dim, ocr_backend=settings.ocr_backend)

    if settings.preflight_strict and critical_problems:
        raise PreflightStrictError(
            f"{len(critical_problems)} critical backend(s) unusable at boot: " + "; ".join(critical_problems)
        )

    return problems
