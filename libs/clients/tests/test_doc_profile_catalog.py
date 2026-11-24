"""The profile card through the catalog seam: the in-memory twin and the factory (Phase 3).

The Postgres half of this is ``tests/integration/test_v2_doc_profile.py`` — a DELETE+INSERT in
one transaction cannot be shown by a twin whose "replace" is an assignment. What belongs here is
the behaviour both adapters must share, and one consequence worth naming on its own: Phase 6's
graph seeding shipped COMPLETE but INERT because nothing wrote ``doc_relations`` (discovery
§8.12). This is its producer, so the seeding becomes live under the in-memory profile too.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_clients.memory import InMemoryCatalog
from rag_core.doc_profile import DocProfile, ProfileFacet, ProfileRelation
from rag_core.graph import DeclaredRelation
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

_TENANT = "acme"
_DOC = "corrigendum-v"


def _profile(*, title: str = "Corrigendum-V", facets: tuple[str, ...] = ("eligibility",)) -> DocProfile:
    return DocProfile(
        doc_type="corrigendum",
        purpose="Amends the bidding document.",
        title=title,
        issuer="AEGCL",
        relations=(ProfileRelation("amends", "Corrigendum-II", "amends Corrigendum-II"),),
        facets=tuple(ProfileFacet(f, f.replace("_", " ").title(), "d") for f in facets),
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
    return catalog


async def test_a_card_is_stored_and_read_back() -> None:
    catalog = await _catalog()
    assert await catalog.replace_document_profile(_TENANT, _DOC, _profile()) is True
    stored = await catalog.document_profile(_TENANT, _DOC)
    assert stored is not None and stored.doc_type == "corrigendum" and stored.issuer == "AEGCL"


async def test_a_card_for_a_document_with_no_row_is_refused() -> None:
    """Postgres UPDATEs `documents` and writes nothing when no row matched; the twin must not be
    more permissive, or a test would pass on a shape production never produces."""
    catalog = InMemoryCatalog()
    assert await catalog.replace_document_profile(_TENANT, "never-ingested", _profile()) is False
    assert await catalog.document_profile(_TENANT, "never-ingested") is None


async def test_re_profiling_replaces_the_facet_taxonomy_rather_than_accumulating_it() -> None:
    """A taxonomy is generated per document by a model, so a second run's ids are not the first
    run's. An upsert would leave Phase 4 labelling sections against a union no profile described."""
    catalog = await _catalog()
    await catalog.replace_document_profile(_TENANT, _DOC, _profile(facets=("eligibility", "financial")))
    await catalog.replace_document_profile(_TENANT, _DOC, _profile(facets=("supply_chain",)))
    stored = await catalog.document_profile(_TENANT, _DOC)
    assert stored is not None and [f.facet_id for f in stored.facets] == ["supply_chain"]


async def test_the_declared_relations_become_readable_by_the_phase_six_graph_seeding() -> None:
    catalog = await _catalog()
    await catalog.replace_document_profile(_TENANT, _DOC, _profile())
    relations = await catalog.document_relations(_TENANT, _DOC)
    assert relations == [
        DeclaredRelation(
            rel_type="amends",
            # The SOURCE is the document's own title, as the Postgres query's
            # `COALESCE(d.title, r.doc_id)` returns it — names, not ids (see `DeclaredRelation`).
            source_ref="Corrigendum-V",
            # UNRESOLVED, as the document wrote it: Phase 3 records the reference, it does not
            # resolve it to a document in this KB.
            target_ref="Corrigendum-II",
            evidence_quote="amends Corrigendum-II",
        )
    ]


async def test_a_document_with_no_title_falls_back_to_its_doc_id_as_the_node_name() -> None:
    catalog = await _catalog()
    await catalog.replace_document_profile(_TENANT, _DOC, _profile(title=""))
    assert (await catalog.document_relations(_TENANT, _DOC))[0].source_ref == _DOC


async def test_re_profiling_replaces_the_documents_own_relations() -> None:
    catalog = await _catalog()
    await catalog.replace_document_profile(_TENANT, _DOC, _profile())
    await catalog.replace_document_profile(
        _TENANT, _DOC, DocProfile(doc_type="corrigendum", purpose="p", title="Corrigendum-V")
    )
    assert await catalog.document_relations(_TENANT, _DOC) == []


async def test_erasing_the_document_erases_its_card_and_its_relations() -> None:
    catalog = await _catalog()
    await catalog.replace_document_profile(_TENANT, _DOC, _profile())
    await catalog.delete_document(_TENANT, _DOC)
    assert await catalog.document_profile(_TENANT, _DOC) is None
    assert await catalog.document_relations(_TENANT, _DOC) == []


async def test_the_factory_seam_fails_closed_on_an_adapter_it_does_not_know() -> None:
    """Sharper here than for the structure tables: a silently unstored card is indistinguishable
    from a document the fail-safe profiler declined to profile, which is an ORDINARY outcome."""
    from rag_clients import factory
    from rag_common.errors import ConfigError

    def _unknown_adapter() -> object:
        return object()

    original = factory.get_catalog
    try:
        factory.get_catalog = _unknown_adapter  # type: ignore[assignment]
        with pytest.raises(ConfigError, match="never profiled"):
            await factory.persist_document_profile(_TENANT, _DOC, _profile())
    finally:
        # Swap back by ASSIGNMENT + cache_clear, never monkeypatch: `get_catalog` is `lru_cache`d
        # and the `inmemory` fixture's teardown calls `cache_info()` on it.
        factory.get_catalog = original
        factory.get_catalog.cache_clear()


# --- the facet vocabulary, pinned to a revision -------------------------------------------------
#
# A facet id is Phase 8's facet-union join key AND, de-underscored, its cross-encoder probe text.
# Phase 3 invents the vocabulary per call, so until it was pinned, re-ingesting a document
# re-rolled its own section selection — measured at three 10-facet vocabularies sharing ONE id
# across three ingests of identical bytes.


_REV = "3dc59d958833024d/p1"


async def test_a_card_stored_with_a_revision_reads_back_as_a_reusable_vocabulary() -> None:
    catalog = await _catalog()
    await catalog.replace_document_profile(_TENANT, _DOC, _profile(facets=("eligibility", "financial")), revision=_REV)

    vocab = await catalog.facet_vocabulary(_TENANT, _DOC)

    assert vocab is not None
    assert vocab.revision == _REV and vocab.ids == ("eligibility", "financial")
    assert vocab.matches(_REV) and not vocab.matches("OTHERBYTES/p1")


async def test_a_card_stored_without_a_revision_is_not_reusable() -> None:
    """Every card written before this existed is in that state, and reusing one would attribute a
    vocabulary to bytes nobody checked."""
    catalog = await _catalog()
    await catalog.replace_document_profile(_TENANT, _DOC, _profile())

    assert await catalog.facet_vocabulary(_TENANT, _DOC) is None


async def test_an_unprofiled_document_has_no_vocabulary_rather_than_an_empty_one() -> None:
    """None means "nothing to reuse". An empty vocabulary would read as "this document has no
    facets" and silently disable Phase 8's facet union for it."""
    catalog = await _catalog()

    assert await catalog.facet_vocabulary(_TENANT, _DOC) is None
    assert await catalog.facet_vocabulary(_TENANT, "never-ingested") is None


async def test_approved_aliases_survive_the_rewrite_that_replaces_the_taxonomy() -> None:
    """The rewrite is a DELETE+INSERT, so an alias registry that did not round-trip would be
    silently emptied by the next ingest — and an alias is only ever added by a controlled update,
    so a lost one cannot be re-earned by a model."""
    catalog = await _catalog()
    carded = DocProfile(
        doc_type="corrigendum",
        purpose="Amends the bidding document.",
        title="Corrigendum-V",
        facets=(
            ProfileFacet(
                facet_id="performance_and_scalability",
                name="Performance and scalability",
                description="d",
                aliases=("performance_and_scalability_metrics",),
            ),
        ),
    )
    await catalog.replace_document_profile(_TENANT, _DOC, carded, revision=_REV)

    vocab = await catalog.facet_vocabulary(_TENANT, _DOC)

    assert vocab is not None
    assert vocab.resolve("performance_and_scalability_metrics") == "performance_and_scalability"


async def test_erasing_the_document_erases_its_vocabulary_too() -> None:
    """DPDP erasure must not leave a vocabulary behind that a re-ingest of the same bytes would
    silently inherit from a document the tenant asked us to forget."""
    catalog = await _catalog()
    await catalog.replace_document_profile(_TENANT, _DOC, _profile(), revision=_REV)
    assert await catalog.facet_vocabulary(_TENANT, _DOC) is not None

    await catalog.delete_document(_TENANT, _DOC)

    assert await catalog.facet_vocabulary(_TENANT, _DOC) is None
