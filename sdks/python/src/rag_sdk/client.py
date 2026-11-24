"""Synchronous RAG as a Service client (httpx). One class, typed methods over the /v1 surface."""

from __future__ import annotations

import base64
from typing import Any

import httpx

from rag_sdk import __version__
from rag_sdk.errors import RagError
from rag_sdk.models import Document, GroundedAnswer, KnowledgeBase

# Advertised to the server on every request so it can negotiate/deprecate by client version
# (see docs/sdk-versioning.md). Single-sourced from ``rag_sdk.__version__``.
USER_AGENT = f"rag-sdk/{__version__}"


class RagClient:
    """A typed, sovereign client for an RAG as a Service deployment.

    Authenticate with either a service-account API key (``api_key="ek_..."``, recommended
    for programmatic access) or a bearer JWT (``token=...``). ``transport`` is an escape
    hatch for tests (e.g. an ``httpx`` ASGI transport).
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        credential = api_key or token
        if not credential:
            raise ValueError("provide either api_key or token")
        headers = {
            "Authorization": f"Bearer {credential}",
            # Version signalling: advertise the SDK version so the server can log, negotiate,
            # or warn on deprecated client releases without any extra round-trip.
            "User-Agent": USER_AGENT,
            "X-Rag-SDK-Version": __version__,
        }
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)

    # --- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> RagClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- low-level -----------------------------------------------------------
    def _request(self, method: str, path: str, **kw: Any) -> Any:
        resp = self._http.request(method, path, **kw)
        if resp.status_code >= 400:  # noqa: PLR2004
            code, message = "http_error", resp.text
            try:
                body = resp.json()
                code = body.get("code", code)
                message = body.get("message") or body.get("detail") or message
            except (ValueError, AttributeError):
                pass
            raise RagError(resp.status_code, str(code), str(message))
        if resp.status_code == 204:  # noqa: PLR2004  (no content)
            return None
        return resp.json()

    # --- knowledge bases -----------------------------------------------------
    def create_knowledge_base(self, display_name: str) -> KnowledgeBase:
        """Create a knowledge base and return it."""
        return KnowledgeBase.model_validate(
            self._request("POST", "/v1/knowledge-bases", json={"display_name": display_name})
        )

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        data = self._request("GET", "/v1/knowledge-bases")
        return [KnowledgeBase.model_validate(k) for k in data]

    # --- documents -----------------------------------------------------------
    def ingest(
        self,
        kb_id: str,
        *,
        doc_id: str,
        text: str | None = None,
        content_b64: str | None = None,
        mime: str = "text/plain",
        strategy: str = "clause",
    ) -> dict[str, Any]:
        """Ingest a document, either from raw ``text`` or pre-encoded ``content_b64``."""
        if (text is None) == (content_b64 is None):
            raise ValueError("provide exactly one of text or content_b64")
        payload = content_b64 or base64.b64encode(text.encode()).decode()  # type: ignore[union-attr]
        result = self._request(
            "POST",
            f"/v1/kb/{kb_id}/ingest",
            json={"doc_id": doc_id, "mime": mime, "content_b64": payload, "strategy": strategy},
        )
        return dict(result)

    def list_documents(self, kb_id: str) -> list[Document]:
        data = self._request("GET", f"/v1/kb/{kb_id}/documents")
        return [Document.model_validate(d) for d in data]

    def delete_document(self, kb_id: str, doc_id: str) -> None:
        self._request("DELETE", f"/v1/kb/{kb_id}/documents/{doc_id}")

    # --- ask -----------------------------------------------------------------
    def query(self, kb_id: str, query: str, *, top_k: int = 5, model: str | None = None) -> GroundedAnswer:
        """Ask a one-shot question; get a grounded-or-abstain answer with citations."""
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if model:
            body["model"] = model
        return GroundedAnswer.model_validate(self._request("POST", f"/v1/kb/{kb_id}/query", json=body))
