"""Embedder-switch state machine (FR-E1 embedding role) — the dangerous half of attach.

A reranker attach takes effect immediately: nothing stored depends on it. An EMBEDDER
attach cannot — every vector in the KB's collection lives in the old model's space (and
usually a different dimension), so the corpus and the query must switch together or
retrieval turns to noise. The invariant, in one line:

    queries and ingests use ``embedder_active``; the connection store only ever
    feeds ``embedder_pending`` — and only the explicit APPLY moves one to the other.

State lives in two ``tenant_config`` keys (no migration; both catalog twins support it):

* ``embedder_active``  — JSON ``{conn_id, endpoint, model, dim, sig}``; absent means the
  platform embedder (true for every KB that predates this feature).
* ``embedder_pending`` — same shape; present means an operator saved a different
  embedding attach and the console shows "pending re-embed" until they apply it.

Apply = recreate-in-place: drop the tenant's collection, recreate at the new dimension,
re-run the write-then-swap reindex per document. During the window dense retrieval
degrades gracefully to BM25+rerank (Qdrant search on a missing collection returns
empty); a shadow-collection alias flip is the documented follow-up for large corpora.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from rag_clients.factory import get_catalog, get_vector_store
from rag_common import ConflictError, get_logger, get_settings
from rag_schemas import AuditAction
from svc_control_plane import wiring
from svc_control_plane.model_connection import StoredModelConnection
from svc_control_plane.semantic_cache import cache as semantic_cache

_log = get_logger(__name__)

ACTIVE_KEY = "embedder_active"
PENDING_KEY = "embedder_pending"


# --- pure helpers (unit-tested) ---------------------------------------------------------------


def record_for(conn: StoredModelConnection, *, scope: str = "kb") -> dict[str, Any]:
    """The tenant-config record describing one embedding attach. ``sig`` is what decides
    whether a later save is a real switch or a no-op edit (name change, key rotation).

    ``scope`` says WHICH store holds the connection: "kb" (this KB's own attach) or "org" (the
    org-wide registry, where one entry serves every KB in the org). Resolution needs it because
    a conn_id alone is ambiguous across the two stores. Absent on records written before org-wide
    embedders existed, which are all KB-scoped — hence the default."""
    return {
        "conn_id": conn.conn_id,
        "endpoint": conn.endpoint,
        "model": conn.model,
        "dim": conn.embed_dim,
        "scope": scope,
        "sig": f"{conn.endpoint}|{conn.model}|{conn.embed_dim}",
    }


def parse_record(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        loaded = json.loads(raw)
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def is_same_embedder(record: dict[str, Any] | None, conn: StoredModelConnection) -> bool:
    """Same VECTOR SPACE, which is endpoint+model+dim — deliberately not the scope or the conn_id.
    An org-wide entry naming the endpoint a KB already embeds with is not a switch: re-embedding
    the corpus into the space it is already in would be an expensive no-op."""
    return record is not None and record.get("sig") == record_for(conn)["sig"]


# --- transitions -------------------------------------------------------------------------------


async def on_embedding_default_change(kb_id: str) -> dict[str, Any]:
    """React to any mutation that may have changed the KB's DEFAULT embedding connection.

    Empty KB (or same embedder as active): activate immediately — there is nothing indexed
    that could disagree. Otherwise: stage as PENDING and change nothing live."""
    catalog = get_catalog()
    cfg = await catalog.get_tenant_config(kb_id)
    active = parse_record(cfg.get(ACTIVE_KEY, ""))
    conn = await wiring.model_connection_store().get_default(kb_id, "embedding")

    if conn is None:
        # No default embedding attach (e.g. the pending one was deleted): drop any pending.
        if cfg.get(PENDING_KEY, ""):
            await catalog.set_tenant_config(kb_id, PENDING_KEY, "")
        return {"state": "platform" if active is None else "active", "active": active, "pending": None}
    return await _transition(kb_id, conn, scope="kb")


async def on_org_embedding_change(org_id: str, conn: StoredModelConnection) -> dict[str, Any]:
    """An ORG-WIDE embedding model was added or made default: decide it for EVERY KB in the org.

    Adding a model org-wide means the whole org uses it — that is what the registry is for, and it
    is immediate for every other role. An embedder is the one role that cannot be immediate,
    because the vectors already stored in each KB were produced by a DIFFERENT model: switching
    without re-embedding does not change the embedder, it makes every stored vector meaningless.

    So the switch is STAGED per KB (the same pending record a per-KB attach writes) and the
    operator applies it once — ``apply_org`` below re-embeds each KB. KBs with nothing indexed, or
    already on that vector space, take it immediately: there is nothing to invalidate."""
    results: dict[str, Any] = {"kbs": [], "pending": 0, "active": 0}
    for kb_id in await org_kb_ids(org_id):
        outcome = await _transition(kb_id, conn, scope="org")
        results["kbs"].append({"kb_id": kb_id, "state": outcome["state"]})
        results["pending" if outcome["state"] == "pending" else "active"] += 1
    _log.info(
        "org_embedding_change",
        org_id=org_id,
        model=conn.model,
        dim=conn.embed_dim,
        pending=results["pending"],
        immediate=results["active"],
    )
    return results


async def org_kb_ids(org_id: str) -> list[str]:
    """Every live KB of an org, plus the org id itself when it is also a KB namespace.

    Older orgs (and every single-KB deployment) index under the bare org id rather than an
    ``{org}__{slug}`` KB, so leaving it out would silently skip the corpus an operator is most
    likely looking at."""
    records = await wiring.kb_store().list(org_id)
    kb_ids = [r.kb_id for r in records]
    if await get_catalog().list_documents(org_id):
        kb_ids.append(org_id)
    return kb_ids


def is_platform_space(conn: StoredModelConnection) -> bool:
    """Does ``conn`` name the SAME model and dimension the platform embedder uses?

    A KB with no attach was embedded by the platform embedder, so a connection naming that same
    model at that same width points at the identical vector space — a different endpoint and
    credential for the same vectors. Adopting it is safe and re-embedding would be an expensive
    no-op. This is the rule ``wiring._org_embedder_or_platform`` applies at resolution time; it
    lives here too so staging and resolution cannot disagree about what counts as a switch (they
    did: a matching org embedder was adopted live AND staged for a re-embed at the same time)."""
    settings = get_settings()
    return (conn.model, conn.embed_dim) == (settings.embed_model, settings.embed_dim)


async def _transition(kb_id: str, conn: StoredModelConnection, *, scope: str) -> dict[str, Any]:
    """Decide ONE KB's embedder state against ``conn``: unchanged, immediate, or staged."""
    catalog = get_catalog()
    cfg = await catalog.get_tenant_config(kb_id)
    active = parse_record(cfg.get(ACTIVE_KEY, ""))

    if active is None and is_platform_space(conn):
        # Same space the platform embedder built: record it and use it now — nothing to re-embed.
        record = record_for(conn, scope=scope)
        await catalog.set_tenant_config(kb_id, ACTIVE_KEY, json.dumps(record))
        await catalog.set_tenant_config(kb_id, PENDING_KEY, "")
        wiring.invalidate_tenant_gateway(kb_id)
        return {"state": "active", "active": record, "pending": None}

    if is_same_embedder(active, conn):
        # Same endpoint/model/dim — an edit, not a switch (key rotation, rename). Refresh the
        # record (conn_id may differ) and clear any stale pending.
        await catalog.set_tenant_config(kb_id, ACTIVE_KEY, json.dumps(record_for(conn, scope=scope)))
        await catalog.set_tenant_config(kb_id, PENDING_KEY, "")
        wiring.invalidate_tenant_gateway(kb_id)
        return {"state": "active", "active": record_for(conn, scope=scope), "pending": None}

    docs = await catalog.list_documents(kb_id)
    if not docs:
        record = record_for(conn, scope=scope)
        await catalog.set_tenant_config(kb_id, ACTIVE_KEY, json.dumps(record))
        await catalog.set_tenant_config(kb_id, PENDING_KEY, "")
        wiring.invalidate_tenant_gateway(kb_id)
        # The collection must exist at the new dimension before the first ingest.
        await (await wiring.indexing_service_for(kb_id)).ensure_namespaces(kb_id)
        semantic_cache.invalidate(kb_id)
        _log.info("embedder_activated_empty_kb", kb_id=kb_id, model=conn.model, dim=conn.embed_dim)
        return {"state": "active", "active": record, "pending": None}

    record = record_for(conn, scope=scope)
    await catalog.set_tenant_config(kb_id, PENDING_KEY, json.dumps(record))
    _log.info("embedder_switch_pending", kb_id=kb_id, model=conn.model, dim=conn.embed_dim, docs=len(docs))
    return {"state": "pending", "active": active, "pending": record}


async def guard_delete(kb_id: str, victim: StoredModelConnection) -> None:
    """Refuse deleting the ACTIVE embedding attach while documents depend on its space —
    the platform embedder cannot answer queries against a collection it did not build."""
    if victim.role != "embedding":
        return
    cfg = await get_catalog().get_tenant_config(kb_id)
    active = parse_record(cfg.get(ACTIVE_KEY, ""))
    if not is_same_embedder(active, victim):
        return
    if await get_catalog().list_documents(kb_id):
        raise ConflictError(
            "this embedding model indexed the KB's documents — attach a replacement and apply "
            "the re-embed (or delete the documents) before removing it",
            details={"conn_id": victim.conn_id, "model": victim.model},
        )
    # Empty KB: releasing the space is safe; fall back to the platform embedder.
    await get_catalog().set_tenant_config(kb_id, ACTIVE_KEY, "")
    await get_catalog().set_tenant_config(kb_id, PENDING_KEY, "")
    wiring.invalidate_tenant_gateway(kb_id)


async def status(kb_id: str) -> dict[str, Any]:
    cfg = await get_catalog().get_tenant_config(kb_id)
    active = parse_record(cfg.get(ACTIVE_KEY, ""))
    pending = parse_record(cfg.get(PENDING_KEY, ""))
    docs = await get_catalog().list_documents(kb_id)
    return {
        "active": active,  # null = platform embedder
        "pending": pending,
        "documents": len(docs),
        "apply_enabled": get_settings().embedder_switch_enabled,
    }


async def apply(kb_id: str, *, actor: str) -> dict[str, Any]:
    """Flip pending → active and rebuild the KB's vector space under the new embedder.

    The flip happens FIRST so every reindex embeds with the new model; each document then
    goes through the existing write-then-swap reindex (per-doc failures are collected,
    not fatal — a straggler stays retrievable via BM25 and can be re-indexed alone)."""
    if not get_settings().embedder_switch_enabled:
        raise ConflictError("the embedder switch is disabled on this deployment (RAG_EMBEDDER_SWITCH_ENABLED)")
    catalog = get_catalog()
    cfg = await catalog.get_tenant_config(kb_id)
    pending = parse_record(cfg.get(PENDING_KEY, ""))
    if pending is None:
        raise ConflictError("no embedder switch is pending for this knowledge base")
    old = parse_record(cfg.get(ACTIVE_KEY, ""))

    await catalog.set_tenant_config(kb_id, ACTIVE_KEY, json.dumps(pending))
    await catalog.set_tenant_config(kb_id, PENDING_KEY, "")
    wiring.invalidate_tenant_gateway(kb_id)

    # Recreate the collection at the new dimension. Queries in this window fall back to
    # BM25+rerank (missing collection = empty dense results), recovering per document.
    await get_vector_store().delete_tenant(kb_id)
    await (await wiring.indexing_service_for(kb_id)).ensure_namespaces(kb_id)

    from svc_control_plane import pipeline  # noqa: PLC0415 — avoid import cycle at module load

    reindexed, failed = 0, []
    for doc in await catalog.list_documents(kb_id):
        try:
            await pipeline.reindex_document(kb_id, doc.doc_id, actor=actor)
            reindexed += 1
        except Exception as exc:
            _log.warning("embedder_switch_doc_failed", kb_id=kb_id, doc_id=doc.doc_id, error=str(exc)[:200])
            failed.append(doc.doc_id)

    # Explicit even though each reindex also invalidates: a zero-doc KB must still drop
    # every cached answer whose key vector lives in the old space.
    semantic_cache.invalidate(kb_id)

    await wiring.audit_service().record(
        tenant_id=kb_id,
        trace_id=str(uuid.uuid4()),
        action=AuditAction.INDEX,
        actor=actor,
        payload={
            "embedder_switch": True,
            "from": (old or {}).get("sig", "platform"),
            "to": pending["sig"],
            "dim": pending.get("dim"),
            "reindexed": reindexed,
            "failed": failed,
        },
    )
    _log.info("embedder_switch_applied", kb_id=kb_id, to=pending["sig"], reindexed=reindexed, failed=len(failed))
    return {"active": pending, "reindexed": reindexed, "failed": failed}


async def org_status(org_id: str) -> dict[str, Any]:
    """What the console's org-wide banner shows: which KBs are waiting to be re-embedded.

    Reported per KB rather than as one number because the work is per KB and so is the risk: an
    operator about to re-embed should see WHICH corpora move and how many documents each holds."""
    pending_record: dict[str, Any] | None = None
    kbs: list[dict[str, Any]] = []
    for kb_id in await org_kb_ids(org_id):
        cfg = await get_catalog().get_tenant_config(kb_id)
        pending = parse_record(cfg.get(PENDING_KEY, ""))
        if pending is None:
            continue
        pending_record = pending_record or pending
        kbs.append({"kb_id": kb_id, "documents": len(await get_catalog().list_documents(kb_id))})
    return {
        "pending": pending_record,
        "kbs": kbs,
        "documents": sum(int(k["documents"]) for k in kbs),
        "apply_enabled": get_settings().embedder_switch_enabled,
    }


async def apply_org(org_id: str, *, actor: str) -> dict[str, Any]:
    """Apply the staged embedder to EVERY KB of the org that is waiting for it.

    One KB failing must not strand the rest: each is applied independently and its error is
    reported against it, because a half-applied org where the failure is invisible is worse than
    a reported failure the operator can retry per KB."""
    applied: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for kb in (await org_status(org_id))["kbs"]:
        kb_id = str(kb["kb_id"])
        try:
            result = await apply(kb_id, actor=actor)
        except Exception as exc:
            _log.warning("org_embedder_apply_failed", org_id=org_id, kb_id=kb_id, error=str(exc)[:200])
            failed.append({"kb_id": kb_id, "error": str(exc)[:200]})
            continue
        applied.append({"kb_id": kb_id, "reindexed": result["reindexed"], "failed": result["failed"]})
    _log.info("org_embedder_applied", org_id=org_id, applied=len(applied), failed=len(failed))
    return {"applied": applied, "failed": failed}
