"""Section cards through the catalog seam: the in-memory twin and the factory (Phase 8).

The Postgres half is ``tests/integration/test_v2_section_cards.py`` — the real adapter reads
``amends`` as a SQL boolean out of a JSONB column and joins the Phase-2 identity columns to the
Phase-4 derived ones in one row, neither of which a dict-backed twin exercises. What belongs
here is the behaviour both adapters must share, and the one that is easiest to get wrong: the
join is OUTER. A section whose enrichment never ran is the ORDINARY case — the Phase-4 calls are
fail-safe per section — and it must still come back, scoreable on its breadcrumb alone, or the
aggregate strategy would silently see only the sections a model happened to summarise.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag_clients import factory
from rag_clients.memory import InMemoryCatalog
from rag_core.layout import LayoutBlock, LayoutSection, LayoutTree
from rag_core.section_enrich import SectionAmends, SectionEnrichment
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

_TENANT = "aegcl"
_DOC = "corrigendum-v"
_OTHER = "nit"


def _tree(doc_id: str, *crumbs: str) -> LayoutTree:
    return LayoutTree(
        doc_id=doc_id,
        sections=tuple(
            LayoutSection(
                section_id=f"{doc_id}-s{i}",
                breadcrumb=crumb,
                title=crumb,
                depth=0,
                ordinal=i,
                text="body",
                text_hash="h",
                blocks=(
                    LayoutBlock(block_id=f"{doc_id}-s{i}-b0", block_type="paragraph", text="body", page=1, ordinal=0),
                ),
            )
            for i, crumb in enumerate(crumbs)
        ),
        source="markup",
    )


async def _catalog() -> InMemoryCatalog:
    catalog = InMemoryCatalog()
    for doc_id in (_DOC, _OTHER):
        await catalog.register_document(
            DocumentEnvelope(
                tenant_id=_TENANT,
                doc_id=doc_id,
                source_id="s1",
                source_kind=SourceKind.UPLOAD,
                mime="application/pdf",
                ingested_at=datetime(2026, 10, 21, tzinfo=UTC),
            )
        )
    await catalog.replace_document_structure(_TENANT, _tree(_DOC, "Sl.1-9 Amended Clauses", "Attachment-1 > Switches"))
    await catalog.replace_document_structure(_TENANT, _tree(_OTHER, "Notice Inviting Tender"))
    return catalog


async def test_every_section_of_the_scope_comes_back_in_document_order() -> None:
    catalog = await _catalog()
    cards = await catalog.document_sections(_TENANT, [_DOC])
    assert [c.section_id for c in cards] == [f"{_DOC}-s0", f"{_DOC}-s1"]
    assert [c.breadcrumb for c in cards] == ["Sl.1-9 Amended Clauses", "Attachment-1 > Switches"]


async def test_an_unenriched_section_still_comes_back() -> None:
    """The OUTER join. Enrichment is fail-safe per section, so a partial result is ordinary —
    and a selector that only saw summarised sections would miss whatever the model skipped."""
    catalog = await _catalog()
    await catalog.replace_section_enrichment(
        _TENANT, _DOC, {f"{_DOC}-s0": SectionEnrichment(section_id=f"{_DOC}-s0", summary="Amends four clauses.")}
    )
    cards = {c.section_id: c for c in await catalog.document_sections(_TENANT, [_DOC])}
    assert cards[f"{_DOC}-s0"].summary == "Amends four clauses."
    assert cards[f"{_DOC}-s1"].summary == "", "unenriched, but present and still scoreable on its breadcrumb"
    assert cards[f"{_DOC}-s1"].breadcrumb == "Attachment-1 > Switches"


async def test_the_enrichment_facets_ride_on_the_card() -> None:
    catalog = await _catalog()
    await catalog.replace_section_enrichment(
        _TENANT,
        _DOC,
        {f"{_DOC}-s0": SectionEnrichment(section_id=f"{_DOC}-s0", summary="x", facet_ids=("eligibility",))},
    )
    [card] = [c for c in await catalog.document_sections(_TENANT, [_DOC]) if c.facet_ids]
    assert card.facet_ids == ("eligibility",)


async def test_amends_is_a_boolean_not_the_payload() -> None:
    """Step 5 asks only WHETHER the section amends something. The stored object carries amended
    clause text, which is document text and has no business on the selection path."""
    catalog = await _catalog()
    await catalog.replace_section_enrichment(
        _TENANT,
        _DOC,
        {
            f"{_DOC}-s0": SectionEnrichment(
                section_id=f"{_DOC}-s0",
                summary="x",
                amends=SectionAmends(
                    target_ref="Appendix to ITB-2",
                    old="Rs. 35 Crores",
                    new="Rs. 19.44 Crores",
                    quote="is amended as follows",
                ),
            )
        },
    )
    cards = {c.section_id: c for c in await catalog.document_sections(_TENANT, [_DOC])}
    assert cards[f"{_DOC}-s0"].amends is True
    assert cards[f"{_DOC}-s1"].amends is False


async def test_the_scope_is_honoured_and_an_empty_scope_is_not_the_whole_tenant() -> None:
    """A question about no document must not load every section the tenant owns onto the
    latency path — nor answer from documents the asker never named."""
    catalog = await _catalog()
    assert {c.doc_id for c in await catalog.document_sections(_TENANT, [_DOC])} == {_DOC}
    assert {c.doc_id for c in await catalog.document_sections(_TENANT, [_DOC, _OTHER])} == {_DOC, _OTHER}
    assert await catalog.document_sections(_TENANT, []) == []


async def test_a_document_with_no_structure_has_no_cards() -> None:
    catalog = await _catalog()
    assert await catalog.document_sections(_TENANT, ["never-parsed"]) == []


async def test_the_factory_seam_reaches_the_twin(inmemory: None) -> None:
    catalog = factory.get_catalog()
    assert isinstance(catalog, InMemoryCatalog)
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
    await catalog.replace_document_structure(_TENANT, _tree(_DOC, "Sl.1-9 Amended Clauses"))
    assert [c.breadcrumb for c in await factory.document_sections(_TENANT, [_DOC])] == ["Sl.1-9 Amended Clauses"]
