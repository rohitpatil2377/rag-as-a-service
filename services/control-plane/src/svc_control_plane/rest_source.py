"""Governed REST-API connector — register a read-only search endpoint, ask a question, get back
ONE authoritative evidence span the unified-KB pipeline fuses alongside document spans.

Mirrors :mod:`sql_query`'s shape: per-KB config lives in ``tenant_config`` via
``get_catalog().set_tenant_config``/``get_tenant_config``, an optional bearer token is
envelope-encrypted with :mod:`tenant_crypto` the same way the SQL connector protects its DSN,
and every outbound call is gated by :func:`rag_common.sovereignty.assert_sovereign` before it
leaves the process — both at registration time (so an unreachable/foreign host is rejected up
front) and again at query time (so a later allowlist change is honoured immediately).

Unlike the SQL connector the contract here is deliberately narrow: READ-ONLY (HTTP GET only,
never a mutation), BOUNDED (one request, a hard timeout, and a streamed byte cap so a runaway
API can never blow memory or the prompt), and evidence-span fail-safe by construction —
:func:`rest_evidence_span` returns ``None`` on ANY exception (unconfigured KB, blocked host,
timeout, non-200, oversized/empty body) and never raises or audits. A producer that is silent
on failure is what lets ``unified.gather_evidence`` fuse it alongside every other source without
one bad connector ever denying the whole answer.

The ``httpx.AsyncClient`` is built through a module-level factory seam (``_client_factory``) so
tests can swap in a client backed by ``httpx.MockTransport`` — no network in the test suite.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from rag_clients.factory import get_catalog
from rag_common import get_logger
from rag_common.sovereignty import assert_sovereign
from rag_schemas import RetrievedSpan
from rag_schemas.common import Modality
from svc_control_plane import tenant_crypto

_log = get_logger(__name__)

_BASE_URL_KEY = "rest_base_url"
_PATH_KEY = "rest_path"
_QUERY_PARAM_KEY = "rest_query_param"
_TOKEN_KEY = "rest_token"  # noqa: S105 - a tenant_config KEY name, not a credential; the value is envelope-encrypted

_DEFAULT_PATH = "/search"
_DEFAULT_QUERY_PARAM = "q"

_REQUEST_TIMEOUT_S = 10.0
_MAX_RESPONSE_BYTES = 262_144  # 256 KiB — a bounded read; larger bodies are refused, not truncated-late

# Module-level injection seam: production code calls this with no arguments to get a real
# httpx.AsyncClient. Tests monkeypatch it to a factory that returns a client wired to
# httpx.MockTransport, so `rest_evidence_span` never touches the network.
_client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient


class RestConnectionInfo(BaseModel):
    """Non-secret view of a KB's registered REST connection."""

    configured: bool
    base_url: str = ""
    path: str = ""
    query_param: str = "q"
    has_token: bool = False


async def _sync_graph_best_effort(kb_id: str) -> None:
    """Seed the knowledge graph with this KB's connected sources' structural entities the moment a
    source is registered — additive + best-effort, so a graph hiccup never blocks the connection.
    Local import breaks the graph_sync<->source-module cycle."""
    try:
        from svc_control_plane import graph_sync  # noqa: PLC0415

        await graph_sync.sync_kb_sources(kb_id)
    except Exception as exc:
        _log.warning("source_graph_sync_failed", kb_id=kb_id, error=str(exc)[:160])


async def set_rest_connection(
    kb_id: str,
    base_url: str,
    *,
    path: str = _DEFAULT_PATH,
    query_param: str = _DEFAULT_QUERY_PARAM,
    token: str = "",
) -> RestConnectionInfo:
    """Register/replace a KB's read-only REST source. ``base_url``'s host must be
    egress-allowlisted — checked here so a non-sovereign endpoint is rejected at registration,
    before any query ever reaches it. A blank ``token`` leaves a previously-registered token in
    place (an update to the URL/path shouldn't silently drop existing credentials)."""
    assert_sovereign(base_url)
    catalog = get_catalog()
    await catalog.set_tenant_config(kb_id, _BASE_URL_KEY, base_url)
    await catalog.set_tenant_config(kb_id, _PATH_KEY, path)
    await catalog.set_tenant_config(kb_id, _QUERY_PARAM_KEY, query_param)
    if token:
        await catalog.set_tenant_config(kb_id, _TOKEN_KEY, await tenant_crypto.encrypt(kb_id, token))
    _log.info("rest_connection_registered", kb_id=kb_id, base_url=base_url, path=path)
    await _sync_graph_best_effort(kb_id)
    return await get_rest_connection(kb_id)


async def get_rest_connection(kb_id: str) -> RestConnectionInfo:
    """Non-secret connection info for the console — the token is NEVER decrypted or returned,
    only whether one is set."""
    cfg = await get_catalog().get_tenant_config(kb_id)
    base_url = cfg.get(_BASE_URL_KEY, "")
    if not base_url:
        return RestConnectionInfo(configured=False)
    return RestConnectionInfo(
        configured=True,
        base_url=base_url,
        path=cfg.get(_PATH_KEY) or _DEFAULT_PATH,
        query_param=cfg.get(_QUERY_PARAM_KEY) or _DEFAULT_QUERY_PARAM,
        has_token=bool(cfg.get(_TOKEN_KEY)),
    )


async def clear_rest_connection(kb_id: str) -> None:
    """Disconnect: remove the KB's registered REST connection (URL, path, query param, token)."""
    catalog = get_catalog()
    for key in (_BASE_URL_KEY, _PATH_KEY, _QUERY_PARAM_KEY, _TOKEN_KEY):
        await catalog.set_tenant_config(kb_id, key, "")
    _log.info("rest_connection_cleared", kb_id=kb_id)


def render_rest_evidence(payload: object, *, max_chars: int = 2000) -> str:
    """Compact, model-readable rendering of a JSON payload, truncated at ``max_chars`` so a
    large response never blows the prompt."""
    try:
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


async def _bearer_header(kb_id: str, cfg: dict[str, str]) -> dict[str, str]:
    token_ct = cfg.get(_TOKEN_KEY)
    if not token_ct:
        return {}
    token = await tenant_crypto.decrypt(kb_id, token_ct)
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _fetch_bounded(url: str, *, params: dict[str, str], headers: dict[str, str]) -> object | None:
    """GET ``url`` with a hard timeout, streaming the body so a response over
    ``_MAX_RESPONSE_BYTES`` is refused instead of buffered in full. Returns the parsed JSON body,
    or None for a non-200 status, an oversized body, or an empty/unparseable payload."""
    async with (
        _client_factory() as client,
        client.stream("GET", url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT_S) as resp,
    ):
        if resp.status_code != httpx.codes.OK:
            return None
        body = bytearray()
        async for chunk in resp.aiter_bytes():
            body.extend(chunk)
            if len(body) > _MAX_RESPONSE_BYTES:
                return None  # bounded: refuse an oversized payload rather than keep buffering it
        if not body:
            return None
        payload: object = json.loads(bytes(body))
    return payload


_SPEC_PATHS = ("/openapi.json", "/swagger.json", "/v3/api-docs")  # tried in order, first hit wins
_MAX_INTROSPECT_ENTITIES = 50  # bounded: a huge spec never floods the graph with nodes


def _parse_openapi_entities(paths: dict[str, object]) -> list[str]:
    """De-duplicated, order-preserving entity names from an OpenAPI/Swagger ``paths`` mapping:
    each operation's ``operationId`` when present (else the path's first non-parameter segment),
    plus any ``tags`` — concise names a graph node can carry. Defensive against a malformed spec
    shape (wrong types, missing keys): skips what it can't read instead of raising."""
    seen: set[str] = set()
    names: list[str] = []

    def _add(name: object) -> None:
        if isinstance(name, str) and name.strip() and name.strip() not in seen:
            clean = name.strip()
            seen.add(clean)
            names.append(clean)

    for raw_path, operations in paths.items():
        if len(names) >= _MAX_INTROSPECT_ENTITIES:
            break
        segment = next((p for p in str(raw_path).split("/") if p and not p.startswith("{")), "")
        if not isinstance(operations, dict):
            _add(segment)
            continue
        for op in operations.values():
            if not isinstance(op, dict):
                continue
            _add(op.get("operationId") or segment)
            for tag in op.get("tags") or []:
                _add(tag)
    return names[:_MAX_INTROSPECT_ENTITIES]


async def introspect_entities(kb_id: str) -> list[str]:
    """Best-effort structural entities for a connected REST source, discovered by fetching its
    OpenAPI/Swagger spec at the common conventional paths (:data:`_SPEC_PATHS`, first hit wins)
    and parsing its ``paths`` into a de-duplicated list of concise entity names (operation IDs /
    path segments / tags — see :func:`_parse_openapi_entities`).

    Reuses the same egress guard, injectable ``_client_factory`` seam, bearer token, and bounded
    fetch (:func:`_fetch_bounded`: hard timeout + streamed byte cap) as :func:`rest_evidence_span`.
    READ-ONLY (GET only) and fail-safe by construction: returns ``[]`` when the KB has no REST
    connection, no spec is found at any tried path, or anything errors — never raises. Callers
    (:mod:`graph_sync`) fall back to the host-only entity when this returns ``[]``."""
    try:
        info = await get_rest_connection(kb_id)
        if not info.configured:
            return []
        assert_sovereign(info.base_url)
        cfg = await get_catalog().get_tenant_config(kb_id)
        headers = await _bearer_header(kb_id, cfg)
        base = info.base_url.rstrip("/")
        for spec_path in _SPEC_PATHS:
            try:
                spec = await _fetch_bounded(f"{base}{spec_path}", params={}, headers=headers)
            except Exception as exc:  # a single bad spec path must not block trying the next one
                _log.debug("rest_introspect_spec_path_failed", kb_id=kb_id, path=spec_path, error=str(exc)[:160])
                continue
            if isinstance(spec, dict) and isinstance(spec.get("paths"), dict):
                names = _parse_openapi_entities(spec["paths"])
                if names:
                    return names
        return []
    except Exception as exc:
        _log.info("rest_introspect_skipped", kb_id=kb_id, error=str(exc)[:160])
        return []


async def rest_evidence_span(kb_id: str, question: str, *, actor: str) -> RetrievedSpan | None:
    """A retrieved-span view of the KB's connected REST API, for UNIFIED (documents + API)
    answers. GETs the registered search endpoint with ``question`` as the query param and
    renders the JSON response as ONE authoritative evidence span the generator grounds on
    alongside document spans.

    Fail-safe by construction: returns None when the KB has no REST connection, the sovereignty
    guard rejects the host, the request times out/errors/returns non-200, or the payload is
    empty — in every such case the caller simply falls back to a documents-only (or
    other-source) answer. Never raises and never audits (a read-only structured-evidence
    producer, like :func:`sql_query.sql_evidence_span`). The span is scored 1.0 / ``exact``
    (authoritative) and badged ``Modality.API`` so its citation is flagged as an API source."""
    del actor  # unused: kept for producer-shape parity with the other structured-evidence producers
    try:
        info = await get_rest_connection(kb_id)
        if not info.configured:
            return None
        assert_sovereign(info.base_url)  # re-checked at query time: allowlist may have changed since registration
        cfg = await get_catalog().get_tenant_config(kb_id)
        headers = await _bearer_header(kb_id, cfg)
        url = f"{info.base_url}{info.path}"
        payload = await _fetch_bounded(url, params={info.query_param: question}, headers=headers)
        if not payload:
            return None
        text = render_rest_evidence(payload)
        if not text:
            return None
        host = urlsplit(info.base_url).hostname or info.base_url
        return RetrievedSpan(
            chunk_id=f"rest::{kb_id}",
            doc_id=f"api://{host}",
            text=text,
            score=1.0,
            scorer="exact",
            modality=Modality.API,
            source_uri=info.base_url,
            category="api",
        )
    except Exception as exc:
        _log.info("rest_evidence_skipped", kb_id=kb_id, error=str(exc)[:160])
        return None
