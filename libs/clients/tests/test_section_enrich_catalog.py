"""Section enrichment through the catalog seam: the in-memory twin and the factory (Phase 4).

The Postgres half is ``tests/integration/test_v2_section_enrich.py`` — a per-row UPDATE inside
one advisory-locked transaction cannot be shown by a twin whose store is a dict. What belongs
here is the behaviour both adapters must share, and one consequence worth naming on its own:
Phase 6's graph seeding reads ``sections.amends`` as well as ``doc_relations``, and until now
that half had no producer at all (``PostgresCatalog.document_relations`` said so in its own
docstring). This is its producer, so a SECTION-level amendment becomes a graph edge too.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_clients import factory
from rag_clients.memory import InMemoryCatalog
from rag_core.layout import LayoutBlock, LayoutSection, LayoutTree
from rag_core.section_enrich import (
    EnrichmentManifest,
    KeyValue,
    Proposition,
    SectionAmends,
    SectionEnrichment,
)
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

_TENANT = "acme"
_DOC = "corrigendum-v"
_SECTION = "sec-maat"
_OTHER = "sec-scope"


def _tree() -> LayoutTree:
    def section(sid: str, crumb: str) -> LayoutSection:
        return LayoutSection(
            section_id=sid,
            breadcrumb=crumb,
            title=crumb,
            depth=0,
            ordinal=0,
            text="body",
            text_hash="h",
            blocks=(LayoutBlock(block_id=f"{sid}-b0", block_type="paragraph", text="body", page=1, ordinal=0),),
        )

    return LayoutTree(
        doc_id=_DOC, sections=(section(_SECTION, "4.2 MAAT"), section(_OTHER, "1. Scope")), source="markup"
    )


def _enrichment(section_id: str = _SECTION, *, amends: SectionAmends | None = None) -> SectionEnrichment:
    return SectionEnrichment(
        section_id=section_id,
        summary="Sets the minimum average annual turnover.",
        facet_ids=("financial_qualification",),
        key_values=(KeyValue("turnover", "19.44", "Rs. Crores", "turnover of Rs. 19.44 Crores"),),
        amends=amends,
        propositions=(Proposition("The bidder must meet the turnover floor.", "turnover of Rs. 19.44"),),
    )


async def _catalog() -> InMemoryCatalog:
    catalog = InMemoryCatalog()
    await catalog.register_document(
        DocumentEnvelope(
            tenant_id=_TENANT,
            doc_id=_DOC,
            source_id="s1",
            source_kind=SourceKind.UPLOAD,
            mime="application/pdf",
            ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
        )
    )
    await catalog.replace_document_structure(_TENANT, _tree())
    return catalog


async def test_an_enrichment_is_stored_against_an_existing_section() -> None:
    catalog = await _catalog()
    assert await catalog.replace_section_enrichment(_TENANT, _DOC, {_SECTION: _enrichment()}) == 1

    stored = await catalog.section_enrichment(_TENANT, _DOC)
    assert set(stored) == {_SECTION}
    assert stored[_SECTION].summary.startswith("Sets the minimum")


async def test_an_enrichment_naming_a_section_with_no_row_updates_nothing() -> None:
    """Postgres matches on ``section_id AND doc_id`` and writes nothing when no row matched. A
    twin that stored every key would make a test pass on a shape production cannot produce — and
    the shortfall is the one failure here that is otherwise invisible (it means the tree was
    re-parsed between enriching and persisting)."""
    catalog = await _catalog()
    hit = await catalog.replace_section_enrichment(
        _TENANT, _DOC, {_SECTION: _enrichment(), "ghost": _enrichment("ghost")}
    )
    assert hit == 1
    assert set(await catalog.section_enrichment(_TENANT, _DOC)) == {_SECTION}


async def test_an_empty_map_writes_nothing_and_purges_nothing() -> None:
    """ "The model produced nothing for this document" is not an instruction to erase what is
    stored — enrichment is fail-safe per section, so an empty result is an ordinary outcome."""
    catalog = await _catalog()
    await catalog.replace_section_enrichment(_TENANT, _DOC, {_SECTION: _enrichment()})
    assert await catalog.replace_section_enrichment(_TENANT, _DOC, {}) == 0
    assert set(await catalog.section_enrichment(_TENANT, _DOC)) == {_SECTION}


# --- the second half of Phase 6's graph seeding ----------------------------------------------------


async def test_a_section_amends_becomes_a_declared_relation() -> None:
    catalog = await _catalog()
    amends = SectionAmends("Corrigendum-II", "Rs. 35 Crores", "Rs. 19.44 Crores", "supersedes Rs. 35 Crores")
    await catalog.replace_section_enrichment(_TENANT, _DOC, {_SECTION: _enrichment(amends=amends)})

    relations = await catalog.document_relations(_TENANT, _DOC)
    assert len(relations) == 1
    rel = relations[0]
    assert rel.rel_type == "amends"
    # The SOURCE is the section's breadcrumb, exactly as the Postgres query returns it — a
    # section-level amendment names the clause that declares it, not the document.
    assert rel.source_ref == "4.2 MAAT"
    assert rel.target_ref == "Corrigendum-II" and rel.evidence_quote == "supersedes Rs. 35 Crores"


async def test_a_section_that_amends_nothing_declares_nothing() -> None:
    catalog = await _catalog()
    await catalog.replace_section_enrichment(_TENANT, _DOC, {_SECTION: _enrichment()})
    assert await catalog.document_relations(_TENANT, _DOC) == []


async def test_deleting_a_document_takes_its_enrichment_with_it() -> None:
    """`summary` and the quotes inside `key_values`/`amends` are derived from verbatim document
    text, so this is the twin's half of the DPDP erasure guarantee."""
    catalog = await _catalog()
    await catalog.replace_section_enrichment(_TENANT, _DOC, {_SECTION: _enrichment()})
    await catalog.delete_document(_TENANT, _DOC)
    assert await catalog.section_enrichment(_TENANT, _DOC) == {}


# --- the factory seam --------------------------------------------------------------------------------


async def test_the_factory_routes_to_the_configured_catalog() -> None:
    catalog = await _catalog()
    original = factory.get_catalog
    factory.get_catalog = lambda: catalog  # type: ignore[assignment]
    try:
        assert await factory.persist_section_enrichment(_TENANT, _DOC, {_SECTION: _enrichment()}) == 1
    finally:
        factory.get_catalog = original
        factory.get_catalog.cache_clear()


async def test_an_unknown_catalog_adapter_fails_closed() -> None:
    """A backend that quietly wrote nothing would hide behind this phase's per-section fail-safe
    indefinitely — and would take `sections.amends` with it, which is half of what Phase 6 seeds
    the graph from."""

    class _Other:
        pass

    other = _Other()

    def _unknown() -> _Other:
        return other

    original = factory.get_catalog
    factory.get_catalog = _unknown  # type: ignore[assignment]
    try:
        with pytest.raises(Exception, match="never enriched"):
            await factory.persist_section_enrichment(_TENANT, _DOC, {_SECTION: _enrichment()})
    finally:
        factory.get_catalog = original
        factory.get_catalog.cache_clear()


# --- the enrichment manifest ------------------------------------------------------------------


async def test_the_manifest_is_stored_against_the_document_and_read_back() -> None:
    catalog = await _catalog()
    manifest = EnrichmentManifest(
        content_hash="3dc59d95",
        sections_total=53,
        sections_offered=38,
        sections_enriched=35,
        sections_skipped_policy=15,
        sections_failed=3,
        failure_classes={"TRUNCATED": 3},
        failed_section_ids=(_SECTION,),
    )
    assert await catalog.replace_enrichment_manifest(_TENANT, _DOC, manifest) is True
    stored = await catalog.enrichment_manifests(_TENANT, [_DOC])
    # THE ROW'S CURRENT HASH COMES BACK TOO, because staleness is the caller's judgement and this
    # query is the only cheap place that knows both. The fixture envelope carries no hash, so the
    # pair here is ("", manifest) -- and a manifest whose own hash is non-empty will not match it,
    # which is exactly how a stale manifest reads as UNKNOWN rather than as a clean run.
    assert stored == {_DOC: (manifest, "")}
    assert not stored[_DOC][0].matches(stored[_DOC][1])


async def test_a_document_with_no_manifest_is_absent_rather_than_reported_complete() -> None:
    """The distinction the whole disclosure rests on. A document missing from the result has no
    readable manifest -- never ingested under a version that wrote one, or stored malformed -- and
    a caller must report that as UNKNOWN. Returning a zero-failure manifest here would hand every
    legacy document a completeness guarantee it never earned."""
    catalog = await _catalog()
    assert await catalog.enrichment_manifests(_TENANT, [_DOC]) == {}
    assert await catalog.enrichment_manifests(_TENANT, [_DOC, "never-seen"]) == {}


async def test_a_manifest_for_an_unregistered_document_is_refused() -> None:
    """The Postgres statement is an UPDATE on `documents` and hits nothing for an unknown doc_id.
    A twin that stored it anyway would let a test pass on a shape production cannot produce."""
    catalog = await _catalog()
    assert await catalog.replace_enrichment_manifest(_TENANT, "not-ingested", EnrichmentManifest()) is False
    assert await catalog.enrichment_manifests(_TENANT, ["not-ingested"]) == {}


async def test_a_later_manifest_replaces_an_earlier_one() -> None:
    """A re-ingest that failed must not leave the previous run's success in place."""
    catalog = await _catalog()
    await catalog.replace_enrichment_manifest(_TENANT, _DOC, EnrichmentManifest(content_hash="old"))
    await catalog.replace_enrichment_manifest(_TENANT, _DOC, EnrichmentManifest(content_hash="new", sections_failed=3))
    stored, _current = (await catalog.enrichment_manifests(_TENANT, [_DOC]))[_DOC]
    assert stored.content_hash == "new" and stored.sections_failed == 3
    assert not stored.matches("old"), "and a reader asking about the old bytes is told no"
