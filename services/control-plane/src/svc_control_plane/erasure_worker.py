"""In-process retention worker — drives the pure erasure scheduler over every KB on a timer.

The scheduling DECISION (which documents are past retention) lives in the pure
:mod:`erasure_schedule` module. This is the impure half: it reads the catalog for each KB's
documents and calls ``compliance.erase`` (cascade delete + signed certificate + audit-chain
record) for the due ones. Same lifecycle as the eval/sync scheduler — started from the app
lifespan iff ``erasure_schedule_enabled`` (default off).

Heavy dependencies (wiring, compliance, catalog) are imported INSIDE the functions:
``compliance`` imports ``pipeline`` which would otherwise cycle at module load, and an off
deployment should not pay the import cost at all.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

from rag_common import get_logger, get_settings
from svc_control_plane.compliance import ErasureCertificate
from svc_control_plane.erasure_schedule import RetainedDoc, RetentionPolicy, run_scheduled_erasure

_log = get_logger(__name__)

_RETENTION_INTERVAL_SECONDS = 3600.0  # sweep hourly; retention is a days-scale policy


async def _erase(kb_id: str, doc_ids: list[str]) -> ErasureCertificate:
    """The erase_fn seam: cascade-erase due documents + record a signed certificate."""
    from svc_control_plane import compliance  # noqa: PLC0415 (compliance -> pipeline import cycle)

    return await compliance.erase(kb_id, actor="retention-scheduler", doc_ids=doc_ids, reason="retention_expiry")


async def _sweep_once(now: datetime) -> int:
    """One retention pass across every tenant's KBs. Returns the count of documents erased."""
    from rag_clients.factory import get_catalog  # noqa: PLC0415
    from svc_control_plane import wiring  # noqa: PLC0415

    retention_days = get_settings().erasure_retention_days
    catalog = get_catalog()
    erased = 0
    for tenant in await wiring.tenant_registry().list_all():
        for kb in await wiring.kb_store().list(tenant.org_id):
            docs = [
                RetainedDoc(doc_id=d.doc_id, created_at=d.ingested_at) for d in await catalog.list_documents(kb.kb_id)
            ]
            policy = RetentionPolicy(kb_id=kb.kb_id, retention_days=retention_days)
            cert = await run_scheduled_erasure(policy=policy, now=now, docs=docs, erase_fn=_erase)
            if cert and cert.documents_erased:
                erased += len(cert.documents_erased)
                _log.info("retention_erasure", kb_id=kb.kb_id, erased=len(cert.documents_erased))
    return erased


async def worker_loop(stop: asyncio.Event, *, interval_seconds: float = _RETENTION_INTERVAL_SECONDS) -> None:
    """Sweep every ``interval_seconds`` until stopped. A failing tick is logged and swallowed —
    a catalog hiccup must never kill the loop or block shutdown."""
    _log.info("erasure_scheduler_started", retention_days=get_settings().erasure_retention_days)
    while not stop.is_set():
        with contextlib.suppress(Exception):
            await _sweep_once(datetime.now(UTC))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
    _log.info("erasure_scheduler_stopped")
