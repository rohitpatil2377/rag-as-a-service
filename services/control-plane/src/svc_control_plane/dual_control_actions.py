"""Executors for dual-control actions — the "checker approved, now run it" half.

Kept OUT of the engine (dual_control.py) so the engine depends on no services. Each executor
performs one gated action from the request's stored params and returns a short result string.
Imported once at app startup (see app lifespan) to register the executors.
"""

from __future__ import annotations

from svc_control_plane import dual_control
from svc_control_plane.dual_control import ApprovalRequest


async def _exec_purge(req: ApprovalRequest) -> str:
    from svc_control_plane import pipeline  # noqa: PLC0415 — avoid import cycle

    kb_id = req.params["kb_id"]
    result = await pipeline.delete_knowledge_base(req.tenant_id, kb_id, actor=req.decided_by or req.requested_by)
    return f"purged {kb_id}: {result['documents']} docs, {result['chunks']} chunks"


async def _exec_erasure(req: ApprovalRequest) -> str:
    from svc_control_plane import compliance  # noqa: PLC0415

    kb_id = req.params["kb_id"]
    cert = await compliance.erase(
        kb_id,
        actor=req.decided_by or req.requested_by,
        doc_ids=req.params.get("doc_ids"),
        subject=req.params.get("subject"),
        reason=req.params.get("reason", "data_principal_request"),
    )
    return f"erased {len(cert.documents_erased)} docs from {kb_id}"


async def _exec_security_change(req: ApprovalRequest) -> str:
    from svc_control_plane import feature_flags, wiring  # noqa: PLC0415

    name = req.params["name"]
    enabled = bool(req.params["enabled"])
    await feature_flags.set_flag(name, enabled, wiring.feature_flag_store())
    return f"feature '{name}' set to {enabled}"


async def _exec_ingest_classified(req: ApprovalRequest) -> str:
    from datetime import UTC, datetime  # noqa: PLC0415

    from rag_schemas import DocumentEnvelope  # noqa: PLC0415
    from rag_schemas.common import ChunkStrategy, SourceKind  # noqa: PLC0415
    from svc_control_plane import pipeline  # noqa: PLC0415

    p = req.params
    doc = DocumentEnvelope(
        tenant_id=req.tenant_id,
        doc_id=p["doc_id"],
        source_id="api-upload",
        source_kind=SourceKind.UPLOAD,
        mime=p.get("mime", "application/pdf"),
        content_b64=p["content_b64"],
        source_acl=p.get("acl", []),
        classification=int(p.get("classification", 0)),
        metadata=p.get("metadata", {}),
        ingested_at=datetime.now(UTC),
    )
    # The approved request records the strategy the REQUESTER asked for; absent means they
    # did not specify one, so the platform default applies at execution — not a literal here,
    # which is how this path came to differ from every other ingest entry point.
    requested = p.get("strategy")
    outcome = await pipeline.ingest_document(
        doc,
        actor=req.decided_by or req.requested_by,
        strategy=ChunkStrategy(requested) if requested else None,
    )
    return f"ingested {outcome.doc_id}: {outcome.chunk_count} chunks"


async def _exec_publish_kb(req: ApprovalRequest) -> str:
    from svc_control_plane import wiring  # noqa: PLC0415

    kb_id = req.params["kb_id"]
    await wiring.kb_store().set_published(req.tenant_id, kb_id, True)
    return f"published {kb_id}"


def register_executors() -> None:
    """Bind every action executor. Idempotent; called once from the app lifespan."""
    dual_control.register("purge", _exec_purge)
    dual_control.register("erasure", _exec_erasure)
    dual_control.register("security_change", _exec_security_change)
    dual_control.register("ingest_classified", _exec_ingest_classified)
    dual_control.register("publish_kb", _exec_publish_kb)
