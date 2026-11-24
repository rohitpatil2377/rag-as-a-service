"""Ingestion service (TDD §4.1 · FR-A1/A2/A3/A7).

Persists the raw document to the object store (lineage-preserving), registers it in
the metadata catalog, and returns the resolved raw bytes for downstream extraction.
Every operation is tenant-scoped via the explicit ``doc.tenant_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_common import get_logger, raw_object_key
from rag_core.interfaces import CatalogAdapter, ExternalObjectFetcher, ObjectStore
from rag_schemas import DocumentEnvelope
from svc_ingestion.connectors import Connector, ConnectorContext, build_connectors

_log = get_logger(__name__)


@dataclass(slots=True)
class IngestResult:
    doc: DocumentEnvelope
    raw: bytes
    object_uri: str


class IngestionService:
    """Normalise -> persist raw -> register in catalog."""

    def __init__(
        self,
        object_store: ObjectStore,
        catalog: CatalogAdapter,
        external_fetcher: ExternalObjectFetcher | None = None,
    ) -> None:
        self._object_store = object_store
        self._catalog = catalog
        # All connectors come from the FR-A5 registry — adding a source is a new module
        # calling register_connector(); nothing here changes.
        self._connectors: dict[str, Connector] = build_connectors(
            ConnectorContext(object_store=object_store, external_fetcher=external_fetcher)
        )

    async def ingest(self, doc: DocumentEnvelope) -> IngestResult:
        connector = self._connectors.get(doc.source_kind.value)
        if connector is None:
            raise NotImplementedError(f"connector for {doc.source_kind} is a Phase-1 stub")
        raw = await connector.fetch(doc)

        # Store raw bytes (WORM tier for originals is a Phase-1 upgrade).
        object_uri = await self._object_store.put(doc.tenant_id, raw_object_key(doc.doc_id), raw, doc.mime)
        # Register document + lineage in the catalog (FR-G7).
        registered = doc if doc.uri else doc.model_copy(update={"uri": object_uri})
        await self._catalog.register_document(registered)
        _log.info("document_ingested", doc_id=doc.doc_id, bytes=len(raw), uri=object_uri)
        return IngestResult(doc=registered, raw=raw, object_uri=object_uri)
