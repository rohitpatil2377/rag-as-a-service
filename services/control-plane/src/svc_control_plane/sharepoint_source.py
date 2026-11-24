"""Governed SharePoint connector — register a site, fold its search hits into a KB's evidence.

Same shape as the SQL connector (:mod:`svc_control_plane.sql_query`): a per-KB config stored via
``get_catalog().set_tenant_config``/``get_tenant_config`` (site URL in the clear — it is not a
secret; the bearer token envelope-encrypted through :mod:`svc_control_plane.tenant_crypto`), the
egress guard (:func:`rag_common.sovereignty.assert_sovereign`) gating every outbound call, and a
single ``RetrievedSpan`` the unified pipeline fuses alongside document spans (FR-D10-adjacent:
docs + connected structured sources, one authoritative span per producer).

READ-ONLY, BOUNDED, SOVEREIGN: this module only ever issues a GET against SharePoint's Search
REST API (``/_api/search/query``), capped at :data:`_MAX_RESULTS` rows and :data:`_TIMEOUT_SECONDS`
seconds, and only after :func:`rag_common.sovereignty.assert_sovereign` clears the site's host.

Evidence-span fail-safe: :func:`sharepoint_evidence_span` returns ``None`` on ANY exception —
not configured, network error, non-200, malformed JSON, an unrecognised response shape, or no
hits — so a SharePoint hiccup degrades to a documents-only answer instead of failing the request.
It never audits: unlike a write/query-execution action, a fold-in read has nothing to attribute
beyond the retrieval log the pipeline already keeps for the fused answer.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel

from rag_clients.factory import get_catalog
from rag_common import get_logger
from rag_common.sovereignty import assert_sovereign
from rag_schemas import RetrievedSpan
from rag_schemas.common import Modality
from svc_control_plane import tenant_crypto

_log = get_logger(__name__)

_SITE_KEY = "sharepoint_site"  # tenant_config key — site URL, stored in the clear (not secret)
_TOKEN_KEY = "sharepoint_token"  # noqa: S105 - a tenant_config KEY name, not a credential; the value is envelope-encrypted

_TIMEOUT_SECONDS = 10.0  # bounded: a slow/hung SharePoint tenant must not stall an answer
_MAX_RESULTS = 10  # bounded: capped both at the query (rowlimit) and at render time

_TAG_RE = re.compile(r"<[^>]+>")  # strips SharePoint's HitHighlightedSummary <c0>/<ddd/> markup


class SharePointConnectionInfo(BaseModel):
    """Non-secret view of a KB's registered SharePoint site. Never carries the token."""

    configured: bool
    site_url: str = ""
    has_token: bool = False


class SharePointHit(BaseModel):
    """One parsed row from a SharePoint Search REST API result."""

    title: str = ""
    snippet: str = ""
    path: str = ""


# Module-level injection seam: production code calls this with no arguments to get a real
# httpx.AsyncClient. Tests monkeypatch it to a factory that returns a client wired to
# httpx.MockTransport, so `sharepoint_evidence_span` never touches the network.
_client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient


async def _sync_graph_best_effort(kb_id: str) -> None:
    """Seed the knowledge graph with this KB's connected sources' structural entities the moment a
    source is registered — additive + best-effort, so a graph hiccup never blocks the connection.
    Local import breaks the graph_sync<->source-module cycle."""
    try:
        from svc_control_plane import graph_sync  # noqa: PLC0415

        await graph_sync.sync_kb_sources(kb_id)
    except Exception as exc:
        _log.warning("source_graph_sync_failed", kb_id=kb_id, error=str(exc)[:160])


async def set_sharepoint_connection(kb_id: str, site_url: str, *, token: str = "") -> SharePointConnectionInfo:
    """Register/replace a KB's SharePoint site. The site host must be egress-allowlisted."""
    assert_sovereign(site_url)  # networked SaaS endpoint is external -> sovereignty-gated
    catalog = get_catalog()
    await catalog.set_tenant_config(kb_id, _SITE_KEY, site_url)
    if token:
        await catalog.set_tenant_config(kb_id, _TOKEN_KEY, await tenant_crypto.encrypt(kb_id, token))
    _log.info("sharepoint_connection_registered", kb_id=kb_id, has_token=bool(token))
    await _sync_graph_best_effort(kb_id)
    return await get_sharepoint_connection(kb_id)


async def get_sharepoint_connection(kb_id: str) -> SharePointConnectionInfo:
    """Non-secret view of the KB's registered site. The token itself is never returned."""
    cfg = await get_catalog().get_tenant_config(kb_id)
    site_url = cfg.get(_SITE_KEY, "")
    if not site_url:
        return SharePointConnectionInfo(configured=False)
    return SharePointConnectionInfo(configured=True, site_url=site_url, has_token=bool(cfg.get(_TOKEN_KEY)))


async def clear_sharepoint_connection(kb_id: str) -> None:
    """Disconnect: remove the KB's registered SharePoint site and token."""
    catalog = get_catalog()
    for key in (_SITE_KEY, _TOKEN_KEY):
        await catalog.set_tenant_config(kb_id, key, "")
    _log.info("sharepoint_connection_cleared", kb_id=kb_id)


# --- Search-response parsing (defensive: an unrecognised shape yields no hits, never raises) ---


def _unwrap(value: Any) -> Any:
    """SharePoint's Search REST API commonly nests collection-valued properties under a
    ``results`` key (OData-light convention) even with ``odata=nometadata`` — unwrap it. A
    plain list (the shape used elsewhere) passes through unchanged."""
    if isinstance(value, dict) and "results" in value:
        return value["results"]
    return value


def _clean(text: str) -> str:
    """Strip SharePoint's hit-highlight markup (``<c0>…</c0>``, ``<ddd/>``) and surrounding
    whitespace so rendered evidence reads as plain text."""
    return _TAG_RE.sub("", text).strip()


def _parse_search_rows(payload: dict[str, Any]) -> list[SharePointHit]:
    """Defensively walk ``PrimaryQueryResult.RelevantResults.Table.Rows``. Any JSON shape this
    doesn't recognise — missing keys, wrong types, an empty table — yields ``[]``, never raises;
    the caller's fail-safe still applies as defence in depth."""
    try:
        rows = _unwrap(payload["PrimaryQueryResult"]["RelevantResults"]["Table"]["Rows"])
    except (KeyError, TypeError):
        return []
    if not isinstance(rows, list):
        return []
    hits: list[SharePointHit] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cells = _unwrap(row.get("Cells"))
        if not isinstance(cells, list):
            continue
        fields: dict[str, str] = {}
        for cell in cells:
            if isinstance(cell, dict) and "Key" in cell and "Value" in cell:
                value = cell["Value"]
                fields[str(cell["Key"])] = "" if value is None else str(value)
        title = _clean(fields.get("Title", ""))
        snippet = _clean(fields.get("HitHighlightedSummary") or fields.get("Summary") or "")
        path = _clean(fields.get("Path", ""))
        if title or snippet or path:
            hits.append(SharePointHit(title=title, snippet=snippet, path=path))
    return hits


def render_sharepoint_evidence(results: list[SharePointHit], *, max_results: int = 10) -> str:
    """Render SharePoint search hits as compact, model-readable evidence text (title — snippet —
    path per hit), bounded at ``max_results`` so a large result set never blows the prompt."""
    lines = [f"SharePoint search result — {len(results)} hit(s):", ""]
    for hit in results[:max_results]:
        parts = [p for p in (hit.title, hit.snippet, hit.path) if p]
        lines.append("- " + " — ".join(parts) if parts else "- (untitled result)")
    if len(results) > max_results:
        lines.append(f"... ({len(results) - max_results} more hit(s))")
    return "\n".join(lines)


# --- Structural introspection: connection-time entities for graph_sync ------------------------

_MAX_INTROSPECT_LISTS = 50  # bounded: a huge site never floods the graph with nodes


def _list_rows(payload: Any) -> Any:
    """SharePoint's REST API returns a lists collection under ``value`` with the
    ``odata=nometadata`` header this module requests, but under ``d.results`` in verbose mode —
    tolerate both, plus the ``results``-wrapper convention (:func:`_unwrap`) and a bare list, so
    a shape difference doesn't turn into a missed introspection rather than raising."""
    if isinstance(payload, dict):
        if "value" in payload:
            return _unwrap(payload["value"])
        if isinstance(payload.get("d"), dict):
            return _unwrap(payload["d"].get("results", payload["d"]))
    return _unwrap(payload)


def _parse_list_titles(payload: Any) -> list[str]:
    """De-duplicated Titles of the non-hidden lists/document-libraries in a
    ``/_api/web/lists`` response, capped at :data:`_MAX_INTROSPECT_LISTS`. Defensive against an
    unrecognised shape — missing/wrong-typed keys yield ``[]``, never raise; the caller's
    fail-safe still applies as defence in depth."""
    rows = _list_rows(payload)
    if not isinstance(rows, list):
        return []
    seen: set[str] = set()
    titles: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("Hidden"):
            continue
        title = row.get("Title")
        if isinstance(title, str) and title and title not in seen:
            seen.add(title)
            titles.append(title)
            if len(titles) >= _MAX_INTROSPECT_LISTS:
                break
    return titles


async def introspect_entities(kb_id: str) -> list[str]:
    """Best-effort structural entities for a connected SharePoint site: the Titles of its
    non-hidden lists and document libraries, via a read-only GET against
    ``{site_url}/_api/web/lists?$select=Title,Hidden``.

    Reuses the same egress guard, injectable ``_client_factory`` seam, and bearer-token handling
    as :func:`sharepoint_evidence_span`; bounded by :data:`_TIMEOUT_SECONDS` and
    :data:`_MAX_INTROSPECT_LISTS`. Fail-safe by construction: returns ``[]`` when the KB has no
    SharePoint connection, the request errors or is non-200, the response is malformed, or
    anything else goes wrong — never raises. Callers (:mod:`graph_sync`) fall back to the
    host-only entity when this returns ``[]``."""
    try:
        info = await get_sharepoint_connection(kb_id)
        if not info.configured:
            return []
        assert_sovereign(info.site_url)
        cfg = await get_catalog().get_tenant_config(kb_id)
        token_ct = cfg.get(_TOKEN_KEY, "")
        token = await tenant_crypto.decrypt(kb_id, token_ct) if token_ct else ""

        base = info.site_url.rstrip("/")
        url = f"{base}/_api/web/lists?$select=Title,Hidden"
        headers = {"Accept": "application/json;odata=nometadata"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with _client_factory() as client:
            response = await client.get(url, headers=headers, timeout=_TIMEOUT_SECONDS)
        if response.status_code != httpx.codes.OK:
            return []
        return _parse_list_titles(response.json())
    except Exception as exc:  # fail-safe: ANY error degrades to the host-only fallback, never raises
        _log.info("sharepoint_introspect_skipped", kb_id=kb_id, error=str(exc)[:160])
        return []


# --- Unified KB: a SharePoint search as a retrieved evidence span -------------------------------


async def sharepoint_evidence_span(kb_id: str, question: str, *, actor: str) -> RetrievedSpan | None:
    """A retrieved-span view of the KB's connected SharePoint site, for UNIFIED (documents +
    structured source) answers. Runs a read-only Search REST API query for ``question`` and
    renders the hits as ONE authoritative evidence span the generator grounds on alongside
    document spans. Returns ``None`` when the KB has no SharePoint connection, the site fails
    the sovereignty check, the request errors or is non-200, the response can't be parsed, or it
    has no hits: in every such case the caller simply falls back to a documents-only answer. The
    span is scored 1.0 / ``exact`` (a search hit is authoritative, not a similarity guess) and
    badged ``Modality.SHAREPOINT`` so its citation is flagged as a SharePoint source.

    ``actor`` is accepted for parity with the SQL connector's evidence-span signature (callers
    pass it uniformly across producers) even though this read path has nothing to audit."""
    del actor  # no audit call on this fail-safe read path (see module docstring)
    try:
        info = await get_sharepoint_connection(kb_id)
        if not info.configured:
            return None
        assert_sovereign(info.site_url)
        cfg = await get_catalog().get_tenant_config(kb_id)
        token_ct = cfg.get(_TOKEN_KEY, "")
        token = await tenant_crypto.decrypt(kb_id, token_ct) if token_ct else ""

        base = info.site_url.rstrip("/")
        url = f"{base}/_api/search/query?querytext='{quote(question, safe='')}'&rowlimit={_MAX_RESULTS}"
        headers = {"Accept": "application/json;odata=nometadata"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with _client_factory() as client:
            response = await client.get(url, headers=headers, timeout=_TIMEOUT_SECONDS)
        if response.status_code != httpx.codes.OK:
            return None
        hits = _parse_search_rows(response.json())
        if not hits:
            return None

        host = urlsplit(info.site_url).hostname or info.site_url
        return RetrievedSpan(
            chunk_id=f"sharepoint::{kb_id}",
            doc_id=f"sharepoint://{host}",
            text=render_sharepoint_evidence(hits, max_results=_MAX_RESULTS),
            score=1.0,
            scorer="exact",
            modality=Modality.SHAREPOINT,
            source_uri=info.site_url,
            category="sharepoint",
        )
    except Exception as exc:  # fail-safe: ANY error degrades to documents-only, never raises
        _log.info("sharepoint_evidence_skipped", kb_id=kb_id, error=str(exc)[:160])
        return None
