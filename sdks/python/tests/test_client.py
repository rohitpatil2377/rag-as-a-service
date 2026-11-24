"""Unit tests for the Python SDK — request shaping + response parsing via a mock transport.

(The sync client is exercised end-to-end against the live server separately; here we prove
it builds correct requests, carries auth, parses typed models, and raises on errors — all
deterministically, with no running app.)
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from rag_sdk import RagClient, RagError, __version__


def _client(handler: httpx.MockTransport, **kw: object) -> RagClient:
    return RagClient("http://rag.test", api_key="ek_abc_secret", transport=handler, **kw)  # type: ignore[arg-type]


def test_requires_a_credential() -> None:
    with pytest.raises(ValueError, match="api_key or token"):
        RagClient("http://rag.test")


def test_query_sends_auth_and_parses_grounded_answer() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["auth"] = req.headers.get("authorization")
        captured["path"] = req.url.path
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "tenant_id": "t",
                "query": "q",
                "status": "grounded",
                "answer": "42 days.",
                "citations": [{"chunk_id": "c1", "doc_id": "leave", "quote": "24 days", "page": 1, "score": 0.9}],
                "confidence": 0.9,
                "suggested_followups": ["What about sick leave?"],
            },
        )

    with _client(httpx.MockTransport(handler)) as c:
        ans = c.query("policies", "How many days?", top_k=3)

    assert captured["auth"] == "Bearer ek_abc_secret"
    assert captured["path"] == "/v1/kb/policies/query"
    assert captured["body"] == {"query": "How many days?", "top_k": 3}
    assert ans.is_grounded and ans.answer == "42 days."
    assert ans.citations[0].doc_id == "leave"
    assert ans.suggested_followups == ["What about sick leave?"]


def test_ingest_base64_encodes_text() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"doc_id": "leave", "chunks": 1})

    with _client(httpx.MockTransport(handler)) as c:
        c.ingest("policies", doc_id="leave", text="Annual leave is 24 days.")

    body = seen["body"]
    assert isinstance(body, dict)
    assert base64.b64decode(body["content_b64"]).decode() == "Annual leave is 24 days."


def test_ingest_rejects_both_or_neither_source() -> None:
    with _client(httpx.MockTransport(lambda r: httpx.Response(200, json={}))) as c:
        with pytest.raises(ValueError, match="exactly one"):
            c.ingest("kb", doc_id="d", text="x", content_b64="y")
        with pytest.raises(ValueError, match="exactly one"):
            c.ingest("kb", doc_id="d")


def test_error_response_raises_ragaas_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"code": "forbidden", "message": "insufficient scope"})

    with _client(httpx.MockTransport(handler)) as c, pytest.raises(RagError) as ei:
        c.query("policies", "q")
    assert ei.value.status_code == 403
    assert ei.value.code == "forbidden"


def test_list_knowledge_bases_parses_models() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"kb_id": "a", "name": "A", "is_default": True}])

    with _client(httpx.MockTransport(handler)) as c:
        kbs = c.list_knowledge_bases()
    assert kbs[0].kb_id == "a" and kbs[0].is_default


def test_sdk_exposes_a_semver_version() -> None:
    """``__version__`` is the single source of truth for the SDK's version."""
    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) == 3, "version must be MAJOR.MINOR.PATCH"
    assert all(p.isdigit() for p in parts)


def test_client_advertises_its_version_on_every_request() -> None:
    """The SDK signals its version to the server via User-Agent and X-Rag-SDK-Version."""
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["user_agent"] = req.headers.get("user-agent")
        seen["sdk_version"] = req.headers.get("x-rag-sdk-version")
        return httpx.Response(200, json=[{"kb_id": "a", "name": "A", "is_default": True}])

    with _client(httpx.MockTransport(handler)) as c:
        c.list_knowledge_bases()

    assert seen["sdk_version"] == __version__
    assert seen["user_agent"] == f"rag-sdk/{__version__}"
