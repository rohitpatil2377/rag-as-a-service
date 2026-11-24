"""The interpretive image reading (FR-B6) — the ONE path that is deliberately not
grounded-or-abstain.

An exception to the platform's headline invariant is only safe while it stays an exception, so
what is pinned here is the containment: it is labelled ungrounded, it refuses text documents, it
cannot be used to read around an ACL, and every call lands on the audit chain.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import pytest

pytestmark = pytest.mark.e2e

from rag_schemas import DocumentEnvelope  # noqa: E402
from rag_schemas.common import AuditAction, SourceKind  # noqa: E402

_KB = "visionkb"

# A 1x1 PNG. The echo vision gateway under the in-memory profile never decodes it; what matters
# is that the ingest path classes the document as an image and stores its raw bytes.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def vision(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_MULTIMODAL_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(subject: str, roles: list[str], groups: list[str] | None = None) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt(subject, _KB, roles=roles, attrs={"groups": groups or []})
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _ingest(doc_id: str, *, mime: str, payload: bytes, acl: list[str] | None = None) -> None:
    from svc_control_plane import pipeline

    await pipeline.ingest_document(
        DocumentEnvelope(
            tenant_id=_KB,
            doc_id=doc_id,
            source_id="s",
            source_kind=SourceKind.UPLOAD,
            mime=mime,
            content_b64=base64.b64encode(payload).decode(),
            ingested_at=datetime(2026, 9, 17, tzinfo=UTC),
            source_acl=acl or [],
        ),
        actor="uploader",
    )


async def test_an_interpreted_reading_is_labelled_ungrounded_and_audited(inmemory: None, vision: None) -> None:
    await _ingest("diagram", mime="image/png", payload=_PNG)

    async with _client("reader@acme", ["reader"]) as c:
        r = await c.post(
            f"/v1/kb/{_KB}/documents/diagram/analyse",
            json={"question": "What does this reaction do?", "interpret": True},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    # The whole point of the exception being safe: the response SAYS it is not grounded, so a
    # caller cannot mistake an interpretation for sourced fact.
    assert body["grounded"] is False
    assert body["doc_id"] == "diagram"
    assert body["model"]

    from svc_control_plane import wiring

    events = await wiring.audit_service().list(_KB, limit=50)
    analysis = [e for e in events if e.action is AuditAction.IMAGE_ANALYSIS]
    assert len(analysis) == 1, "an ungrounded, PII-bypassing read must always be attributable"
    assert analysis[0].payload["question"] == "What does this reaction do?"
    assert analysis[0].payload["interpret"] is True


async def test_a_strict_reading_is_reported_as_grounded(inmemory: None, vision: None) -> None:
    await _ingest("strict", mime="image/png", payload=_PNG)
    async with _client("reader@acme", ["reader"]) as c:
        r = await c.post(
            f"/v1/kb/{_KB}/documents/strict/analyse",
            json={"question": "What text is written here?", "interpret": False},
        )
    assert r.status_code == 200, r.text
    assert r.json()["grounded"] is True


async def test_a_text_document_is_refused_rather_than_silently_captioned(inmemory: None, vision: None) -> None:
    """The ordinary grounded ask already sees a text document's full extracted text, so routing
    one through the ungrounded path would trade the gate away for nothing."""
    await _ingest("memo", mime="text/plain", payload=b"The retention period is seven years.")
    async with _client("reader@acme", ["reader"]) as c:
        r = await c.post(f"/v1/kb/{_KB}/documents/memo/analyse", json={"question": "How long?"})
    assert r.status_code == 422, r.text


async def test_it_cannot_be_used_to_read_around_an_acl(inmemory: None, vision: None) -> None:
    """A caller who cannot retrieve any of a document's chunks must not be able to have a model
    read its pixels aloud instead -- that would be a second, unguarded door to the same content."""
    await _ingest("secret", mime="image/png", payload=_PNG, acl=["finance"])

    async with _client("outsider@acme", ["reader"], groups=["marketing"]) as c:
        denied = await c.post(f"/v1/kb/{_KB}/documents/secret/analyse", json={"question": "What is this?"})
    assert denied.status_code == 404, denied.text

    async with _client("insider@acme", ["reader"], groups=["finance"]) as c:
        allowed = await c.post(f"/v1/kb/{_KB}/documents/secret/analyse", json={"question": "What is this?"})
    assert allowed.status_code == 200, allowed.text


async def test_a_missing_document_is_a_404(inmemory: None, vision: None) -> None:
    async with _client("reader@acme", ["reader"]) as c:
        r = await c.post(f"/v1/kb/{_KB}/documents/nope/analyse", json={"question": "What is this?"})
    assert r.status_code == 404
