"""E2E: RDKit chemical normalization (Slice 6, Phase C) — additive, flag-gated, chemistry-only.

Two SMILES that are the SAME molecule written differently (ethanol ``OCC`` vs ``CCO``) do not match
verbatim, so a query for one under-retrieves a document containing the other. With chemical
normalization on for a chemistry KB, ingest stores the canonical form as an extra searchable token
and the query is augmented with the same canonical form, so they match.

Deterministic in CI WITHOUT rdkit: a tiny FAKE normalizer (mapping OCC/CCO → the same canonical
identity) is wired in place of the real toolkit, so the e2e exercises the full ingest→index→query
path. The assertions are at the SPARSE-RETRIEVAL level with a molecule-ONLY query (no shared prose
words), so they isolate the chem effect itself — and the flag-off / non-chemistry contrasts prove
the feature is doing the work, not incidental word overlap.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator

import httpx
import pytest

from rag_core.chem import CanonicalMolecule
from rag_schemas import RetrievalQuery

pytestmark = pytest.mark.e2e

_ETHANOL = ("CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
_DOC = "The reaction solvent in this protocol is CCO at room temperature."


class _FakeNormalizer:
    """Both ethanol spellings collapse to one canonical identity; everything else is unknown."""

    def canonicalize(self, smiles: str) -> CanonicalMolecule | None:
        if smiles in ("OCC", "CCO", "C(C)O"):
            return CanonicalMolecule(canonical_smiles=_ETHANOL[0], inchikey=_ETHANOL[1])
        return None


@pytest.fixture
def _chem_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Master flag on + a FAKE normalizer wired where the pipeline resolves it (both ingest and
    query go through ``pipeline.get_molecule_normalizer``), so the path is deterministic without
    rdkit installed."""
    from svc_control_plane import pipeline

    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_CHEM_NORMALIZATION_ENABLED", "true")
    get_settings.cache_clear()

    def _fake() -> _FakeNormalizer:
        return _FakeNormalizer()

    monkeypatch.setattr(pipeline, "get_molecule_normalizer", _fake)
    yield
    get_settings.cache_clear()


async def _client(tenant: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    token = issue_dev_jwt("officer", tenant, roles=["admin", "reader", "ingestor"])
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": f"Bearer {token}"})


async def _make_chemistry_kb(tenant: str, *, gate: bool) -> None:
    """Set the KB's ACTIVE domain profile to chemistry and its per-KB chem gate — the two
    tenant_config keys the triple gate reads (the third is the master env flag)."""
    from svc_control_plane.domain_profile import PROFILE_KEY

    from rag_clients.factory import get_catalog

    catalog = get_catalog()
    await catalog.set_tenant_config(tenant, PROFILE_KEY, json.dumps({"domain": "chemistry"}))
    if gate:
        await catalog.set_tenant_config(tenant, "chem_normalization_enabled", "1.0")


async def _ingest(c: httpx.AsyncClient, tenant: str, doc_id: str, text: str) -> None:
    body = base64.b64encode(text.encode()).decode()
    await c.post(f"/v1/kb/{tenant}/ingest", json={"doc_id": doc_id, "mime": "text/plain", "content_b64": body})


async def _augment(tenant: str) -> RetrievalQuery:
    """Run the pipeline's query-side chem step on a molecule-ONLY 'OCC' query for this KB."""
    from svc_control_plane.pipeline import _with_chem_normalization

    return await _with_chem_normalization(RetrievalQuery(tenant_id=tenant, text="OCC", top_k=10))


async def _retrieves_protocol(tenant: str) -> bool:
    """Hit the sparse index with the augmented query — passing the DEDICATED ``chem_terms`` field, so
    the result reflects chemical matching alone, not shared prose words. A hit on the `protocol` doc
    means the CCO chunk was retrieved for OCC."""
    from rag_clients.factory import get_sparse_index

    query = await _augment(tenant)
    hits = await get_sparse_index().search(tenant, query.text, 10, chem_terms=query.chem_terms)
    return any(cid.startswith("protocol") for cid, _ in hits)


async def test_equivalent_smiles_query_retrieves_the_chunk(inmemory: None, _chem_on: None) -> None:
    """A chemistry KB with the gate on: a doc written with CCO is retrieved by a query written with
    OCC — the two collapse to the same canonical token. The canonical terms ride on the dedicated
    ``chem_terms`` field; ``query.text`` is left untouched (so dense/rerank input is not polluted)."""
    tenant = "chemco"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "Chem KB"})
        await _make_chemistry_kb(tenant, gate=True)
        await _ingest(c, tenant, "protocol", _DOC)

        # FIX 1: augmentation carries canonical terms on the dedicated field, NOT in query.text.
        augmented = await _augment(tenant)
        assert augmented.text == "OCC", "query.text was mutated — dense/sparse/rerank input polluted"
        assert _ETHANOL[0] in augmented.chem_terms and _ETHANOL[1] in augmented.chem_terms, augmented.chem_terms

        assert await _retrieves_protocol(tenant), "OCC query did not retrieve the CCO document"


async def test_flag_off_is_byte_identical_no_cross_form_match(inmemory: None) -> None:
    """CONTROL — with the master flag OFF, the query is not rewritten and no canonical tokens are
    indexed, so a molecule-only OCC query does NOT retrieve the CCO document. The byte-identical
    baseline the feature improves on."""
    tenant = "chemoff"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "Chem KB"})
        await _make_chemistry_kb(tenant, gate=True)  # profile + gate set, but the MASTER flag is off
        await _ingest(c, tenant, "protocol", _DOC)

        assert not await _retrieves_protocol(tenant), "OCC matched CCO with the master flag OFF"


async def test_non_chemistry_kb_is_not_normalized(inmemory: None, _chem_on: None) -> None:
    """Even with the master flag on, a NON-chemistry KB (no chemistry profile) is untouched — the
    triple gate requires the domain profile, so a molecule-only OCC query does not cross-match."""
    tenant = "hrkb"
    c = await _client(tenant)
    async with c:
        await c.post("/v1/knowledge-bases", json={"display_name": "HR KB"})
        # deliberately NOT a chemistry KB — no profile set, gate irrelevant
        await _ingest(c, tenant, "protocol", _DOC)

        assert not await _retrieves_protocol(tenant), "a non-chemistry KB was normalized"
