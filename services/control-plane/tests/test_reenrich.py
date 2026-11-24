"""Re-running Phases 3-4 only (RAG v2 Phase 12's re-enrichment path).

The two claims worth a test each:

1. **It writes the derived columns and touches nothing else.** That is the whole reason it
   exists instead of ``reindex_document`` — a prompt change does not change the corpus.
2. **It REFUSES when the structure re-derives differently.** Phases 3 and 4 both take the tree
   as input, so re-running them re-derives it; a tree whose section ids moved cannot be
   persisted, because the INDEXED chunks still carry the old ones and the query path would then
   select sections whose chunks it cannot find, with every count reading plausibly.
"""

from __future__ import annotations

import base64
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
from svc_control_plane import pipeline

from rag_clients.factory import get_catalog, stale_profiles
from rag_clients.memory import InMemoryCatalog
from rag_common import get_settings, tenant_scope
from rag_core.doc_profile import PROFILE_VERSION, DocProfile, ProfileFacet
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

_TENANT = "reenrichkb"
_DOC = "corrigendum.txt"
_TEXT = (
    "1. Eligibility\n\nThe bidder shall hold a valid registration certificate.\n\n"
    "2. Financial\n\nThe average annual turnover shall be at least Rs. 19.44 Crores.\n"
)


def _twin() -> InMemoryCatalog:
    catalog = get_catalog()
    assert isinstance(catalog, InMemoryCatalog), "these are unit tests on the in-memory profile"
    return catalog


def _enable(monkeypatch: pytest.MonkeyPatch, *, on: bool = True, **env: str) -> None:
    monkeypatch.setenv("RAG_V2_INGEST", "true" if on else "false")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


async def _ingest() -> None:
    with tenant_scope(_TENANT):
        await pipeline.ingest_document(
            DocumentEnvelope(
                tenant_id=_TENANT,
                doc_id=_DOC,
                source_id="s",
                source_kind=SourceKind.UPLOAD,
                mime="text/plain",
                content_b64=base64.b64encode(_TEXT.encode()).decode(),
                ingested_at=datetime(2026, 10, 22, tzinfo=UTC),
            ),
            actor="loader",
        )


async def _reenrich() -> Any:
    with tenant_scope(_TENANT):
        return await pipeline.reenrich_document(_TENANT, _DOC, actor="test")


# --- the happy path ---------------------------------------------------------------------------------


async def test_a_document_with_structure_is_reenriched(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch)
    await _ingest()

    outcome = await _reenrich()

    assert outcome.status == "reenriched"
    assert outcome.sections > 0


async def test_it_does_not_touch_the_chunks(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole reason this exists rather than a reindex: a prompt change does not change the
    corpus, and re-embedding one to pick up a prompt edit is hours of GPU for two columns."""
    _enable(monkeypatch)
    await _ingest()
    before = {c.chunk_id for c in await _twin().document_chunks(_TENANT, _DOC)}

    await _reenrich()

    assert {c.chunk_id for c in await _twin().document_chunks(_TENANT, _DOC)} == before


async def test_the_structure_rows_survive_with_the_same_ids(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``replace_document_structure`` DELETEs and re-INSERTs, so this is the check that a
    deterministic re-extraction really is a no-op on identity — if it were not, the refusal
    below would fire on every document."""
    _enable(monkeypatch)
    await _ingest()
    before = {c.section_id for c in await _twin().document_sections(_TENANT, [_DOC])}

    await _reenrich()

    assert {c.section_id for c in await _twin().document_sections(_TENANT, [_DOC])} == before


# --- the refusal ------------------------------------------------------------------------------------


async def test_a_changed_structure_is_refused_and_nothing_is_written(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different extractor, a changed layout parser, the Phase-2 heading fix. The indexed
    chunks still carry the old section ids, so enriching here would leave Phase 8 selecting
    sections whose chunks Phase 9 cannot find — and every count would read plausibly."""
    _enable(monkeypatch)
    await _ingest()
    stored = await _twin().document_sections(_TENANT, [_DOC])
    assert stored

    original = pipeline._layout_tree

    def _shifted(doc_id: str, extracted: Any, *, text: str) -> Any:
        tree = original(doc_id, extracted, text=text)
        if tree is None:
            return None
        moved = tuple(replace(s, section_id=f"{s.section_id}-moved") for s in tree.sections)
        return replace(tree, sections=moved)

    monkeypatch.setattr(pipeline, "_layout_tree", _shifted)
    outcome = await _reenrich()

    assert outcome.status == "structure_changed"
    after = {c.section_id for c in await _twin().document_sections(_TENANT, [_DOC])}
    assert after == {c.section_id for c in stored}, "the stored rows were left exactly as they were"


# --- the ordinary non-results ------------------------------------------------------------------------


async def test_a_document_with_no_structure_is_not_an_error(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Most of a real corpus is in this state — the flag was off at ingest, or the extractor
    emitted nothing this build can parse. Failing the command for it would make the CLI
    unusable."""
    _enable(monkeypatch)
    await _ingest()
    monkeypatch.setattr(pipeline, "_layout_tree", lambda *a, **k: None)

    assert (await _reenrich()).status == "no_structure"


async def test_an_unknown_document_reports_no_bytes(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch)
    with tenant_scope(_TENANT):
        outcome = await pipeline.reenrich_document(_TENANT, "never-ingested.pdf", actor="test")
    assert outcome.status == "no_bytes"


async def test_the_flag_being_off_writes_no_card(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both phases are gated on `v2_ingest`, which is why the CLI refuses to run without
    it: the run would report every document re-enriched and write nothing."""
    _enable(monkeypatch, on=False)
    await _ingest()

    outcome = await _reenrich()

    assert outcome.status == "no_structure", "no tree is parsed with the flag off"


# --- what the CLI selects ----------------------------------------------------------------------------


async def test_a_cardless_document_counts_as_stale(
    inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commonest stale case by far — every document ingested before the v2 flag — and a
    query that skipped it would make the CLI useless on exactly the corpus an operator wants to
    bring forward."""
    _enable(monkeypatch, on=False)
    await _ingest()

    assert await stale_profiles(_TENANT, below=PROFILE_VERSION) == [_DOC]


async def test_a_current_card_is_not_stale(inmemory: None, lexical_gate: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    await _ingest()
    await _twin().replace_document_profile(
        _TENANT,
        _DOC,
        DocProfile(doc_type="corrigendum", purpose="p", title=_DOC, facets=(ProfileFacet("f", "F", "d"),)),
    )

    assert await stale_profiles(_TENANT, below=PROFILE_VERSION) == []
    assert await stale_profiles(_TENANT, below=PROFILE_VERSION + 1) == [_DOC], "a bumped version makes it stale"
