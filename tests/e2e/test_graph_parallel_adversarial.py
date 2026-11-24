"""ADVERSARIAL: parallelized _index_graph + build_communities (asyncio.gather+Semaphore).

Drives pipeline._index_graph and pipeline.build_communities directly against the in-memory
graph store + a stub extractor, proving the parallelization is behaviour-preserving:
  (1) a single failing/timing-out chunk drops ONLY that chunk, not the doc's graph
  (2) graph output byte-identical to a sequential run
  (3) community_id assignment stays 0,1,2... regardless of completion order
  (4) the Semaphore actually bounds concurrency (max in-flight instrumented)
  (5) no concurrent graph-store write corruption
  (6) graph_index_concurrency bounds (1..32) respected
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import pytest

from rag_common.config import get_settings
from rag_core.graph import GraphTriple
from rag_schemas.chunk import Chunk

pytestmark = pytest.mark.e2e


# --------------------------------------------------------------------------- helpers


def _chunk(tenant: str, doc: str, idx: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"{doc}::{idx}", doc_id=doc, tenant_id=tenant, text=text)


class _StubCatalog:
    """Only the two methods _index_graph / build_communities call on the catalog."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    async def document_chunks(self, tenant_id: str, doc_id: str) -> list[Chunk]:
        return [c for c in self._chunks if c.doc_id == doc_id]

    async def get_chunks(self, tenant_id: str, ids: list[str]) -> dict[str, Chunk]:
        by_id = {c.chunk_id: c for c in self._chunks}
        return {i: by_id[i] for i in ids if i in by_id}


@dataclass
class _InstrumentedExtractor:
    """Deterministic entity mapping per chunk-text, with fault + concurrency instrumentation."""

    fail_on: set[str]  # chunk texts (or entity names) that RAISE
    timeout_on: set[str]  # chunk texts that hang past a short timeout
    delay: float = 0.0  # base per-call delay to force overlap
    jitter: bool = False
    mapping: dict[str, GraphTriple] | None = None

    def __post_init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[str] = []
        self._lock = asyncio.Lock()

    async def extract(self, text: str) -> GraphTriple:
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.calls.append(text)
        try:
            d = self.delay + (random.random() * 0.01 if self.jitter else 0.0)
            if text in self.timeout_on:
                await asyncio.wait_for(asyncio.sleep(10), timeout=0.02)  # raises TimeoutError
            if d:
                await asyncio.sleep(d)
            if text in self.fail_on:
                raise RuntimeError(f"boom on {text!r}")
            if self.mapping is not None and text in self.mapping:
                return self.mapping[text]
            # Default: derive one entity + no edges from the text (first token, capitalized).
            ent = text.split(maxsplit=1)[0]
            return GraphTriple(entities=[ent], edges=[])
        finally:
            async with self._lock:
                self.in_flight -= 1


class _StubSummarizer:
    async def summarize(self, top_entities: list[str], sample_texts: list[str], size: int) -> str:
        return f"S:{','.join(top_entities)}|{size}"


def _wire(monkeypatch, *, store, catalog, extractor, summarizer=None) -> None:
    """Patch the names _index_graph / build_communities resolve at call time."""
    import svc_control_plane.pipeline as pl
    from svc_control_plane import wiring

    monkeypatch.setattr(pl, "get_graph_store", lambda: store)
    monkeypatch.setattr(pl, "get_catalog", lambda: catalog)

    async def _resolve_extractor(tenant_id):
        return extractor

    async def _resolve_summarizer(tenant_id):
        return summarizer or _StubSummarizer()

    monkeypatch.setattr(wiring, "resolve_graph_extractor", _resolve_extractor)
    monkeypatch.setattr(wiring, "resolve_community_summarizer", _resolve_summarizer)


def _enable_graphrag(monkeypatch, concurrency: int = 8) -> None:
    monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("RAG_GRAPH_INDEX_CONCURRENCY", str(concurrency))
    get_settings.cache_clear()


def _fresh_store():  # type: ignore[no-untyped-def]
    from rag_clients.graph_store import InMemoryGraphStore

    return InMemoryGraphStore()


# --------------------------------------------------------------------------- (1) fault isolation


async def test_single_failing_chunk_drops_only_that_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_graphrag(monkeypatch, concurrency=4)
    from svc_control_plane.pipeline import _index_graph

    tenant = "zzz-fault"
    n = 10
    chunks = [_chunk(tenant, "d", i, f"Ent{i} rest of text") for i in range(n)]
    fail_text = chunks[3].text
    store = _fresh_store()
    ext = _InstrumentedExtractor(fail_on={fail_text}, timeout_on=set(), delay=0.005)
    _wire(monkeypatch, store=store, catalog=_StubCatalog(chunks), extractor=ext)

    await _index_graph(tenant, "d")

    nodes, _ = await store.graph_data(tenant)
    keys = {node.display for node in nodes}
    # Every chunk except the 3rd contributed its entity; the failing one is absent.
    assert "Ent3" not in keys, f"failing chunk's entity leaked into graph: {keys}"
    for i in [0, 1, 2, 4, 5, 6, 7, 8, 9]:
        assert f"Ent{i}" in keys, f"non-failing chunk {i} was DROPPED (regression!): {keys}"
    assert len(nodes) == n - 1


async def test_timeout_chunk_drops_only_that_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_graphrag(monkeypatch, concurrency=4)
    from svc_control_plane.pipeline import _index_graph

    tenant = "zzz-timeout"
    chunks = [_chunk(tenant, "d", i, f"Ent{i} body") for i in range(6)]
    store = _fresh_store()
    ext = _InstrumentedExtractor(fail_on=set(), timeout_on={chunks[2].text}, delay=0.0)
    _wire(monkeypatch, store=store, catalog=_StubCatalog(chunks), extractor=ext)

    await _index_graph(tenant, "d")
    nodes, _ = await store.graph_data(tenant)
    keys = {node.display for node in nodes}
    assert "Ent2" not in keys
    assert len({f"Ent{i}" for i in [0, 1, 3, 4, 5]} & keys) == 5


async def test_all_chunks_fail_leaves_empty_graph_no_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_graphrag(monkeypatch, concurrency=8)
    from svc_control_plane.pipeline import _index_graph

    tenant = "zzz-allfail"
    chunks = [_chunk(tenant, "d", i, f"Ent{i} x") for i in range(5)]
    store = _fresh_store()
    ext = _InstrumentedExtractor(fail_on={c.text for c in chunks}, timeout_on=set())
    _wire(monkeypatch, store=store, catalog=_StubCatalog(chunks), extractor=ext)
    # Must not raise (graph is additive; ingest must survive).
    await _index_graph(tenant, "d")
    nodes, _ = await store.graph_data(tenant)
    assert nodes == []


# --------------------------------------------------------------------------- (2) byte-identical


async def _sequential_index(chunks: list[Chunk], mapping: dict[str, GraphTriple]):  # type: ignore[no-untyped-def]
    """Reference SEQUENTIAL implementation: identical semantics, no gather/semaphore."""
    from rag_clients.graph_store import InMemoryGraphStore

    store = InMemoryGraphStore()
    for c in chunks:
        t = mapping[c.text]
        if t.entities:
            await store.add_chunk(c.tenant_id, c.chunk_id, t.entities, t.edges)
    return store


async def test_graph_byte_identical_to_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_graphrag(monkeypatch, concurrency=8)
    from svc_control_plane.pipeline import _index_graph

    tenant = "zzz-identical"
    # A fixed corpus with multi-entity chunks + edges + cross-chunk shared entities.
    texts = [
        "Alpha Beta Gamma",
        "Beta Delta",
        "Gamma Epsilon Zeta",
        "Alpha Zeta",
        "Eta Theta Iota Kappa",
        "Alpha Beta",
    ]
    chunks = [_chunk(tenant, "d", i, t) for i, t in enumerate(texts)]

    def triple_for(text: str) -> GraphTriple:
        ents = text.split()
        edges = [(a, b) if a.lower() <= b.lower() else (b, a) for i, a in enumerate(ents) for b in ents[i + 1 :]]
        return GraphTriple(entities=ents, edges=edges)

    mapping = {t: triple_for(t) for t in texts}

    # Parallel run (with jitter so completion order varies).
    store_p = _fresh_store()
    ext = _InstrumentedExtractor(fail_on=set(), timeout_on=set(), delay=0.003, jitter=True, mapping=mapping)
    _wire(monkeypatch, store=store_p, catalog=_StubCatalog(chunks), extractor=ext)
    await _index_graph(tenant, "d")

    # Sequential reference.
    seq_store = await _sequential_index(chunks, mapping)

    pn, pe = await store_p.graph_data(tenant)
    sn, se = await seq_store.graph_data(tenant)

    def norm_nodes(nodes):
        return sorted((n.key, n.display, tuple(sorted(n.chunk_ids))) for n in nodes)

    assert norm_nodes(pn) == norm_nodes(sn), "node set/chunk-membership differs from sequential"
    assert sorted(pe) == sorted(se), "edge set differs from sequential"
    # And run it a few more times: order independence should be stable.
    for _ in range(5):
        store_r = _fresh_store()
        ext_r = _InstrumentedExtractor(fail_on=set(), timeout_on=set(), delay=0.002, jitter=True, mapping=mapping)
        _wire(monkeypatch, store=store_r, catalog=_StubCatalog(chunks), extractor=ext_r)
        await _index_graph(tenant, "d")
        rn, re_ = await store_r.graph_data(tenant)
        assert norm_nodes(rn) == norm_nodes(sn)
        assert sorted(re_) == sorted(se)


# --------------------------------------------------------------------------- (5) concurrent write safety


async def test_no_concurrent_write_corruption_shared_entity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Many chunks all mention the SAME entity concurrently — the accumulated chunk_id set
    on that node must contain EVERY chunk id (no lost update from interleaved writes)."""
    _enable_graphrag(monkeypatch, concurrency=16)
    from svc_control_plane.pipeline import _index_graph

    tenant = "zzz-shared"
    n = 64
    chunks = [_chunk(tenant, "d", i, f"tok{i}") for i in range(n)]
    mapping = {c.text: GraphTriple(entities=["Shared", c.text], edges=[("Shared", c.text)]) for c in chunks}
    store = _fresh_store()
    ext = _InstrumentedExtractor(fail_on=set(), timeout_on=set(), delay=0.001, jitter=True, mapping=mapping)
    _wire(monkeypatch, store=store, catalog=_StubCatalog(chunks), extractor=ext)
    await _index_graph(tenant, "d")

    nodes, edges = await store.graph_data(tenant)
    shared = next(node for node in nodes if node.display == "Shared")
    assert sorted(shared.chunk_ids) == sorted(c.chunk_id for c in chunks), "lost update on shared node"
    assert len(edges) == n  # one distinct edge Shared<->tokI per chunk


# --------------------------------------------------------------------------- (4)+(6) semaphore bound


async def test_semaphore_bounds_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane.pipeline import _index_graph

    tenant = "zzz-sem"
    n = 40
    chunks = [_chunk(tenant, "d", i, f"Ent{i} b") for i in range(n)]
    for limit in (1, 2, 3, 8, 32):
        _enable_graphrag(monkeypatch, concurrency=limit)
        store = _fresh_store()
        # A real delay so many would overlap if unbounded.
        ext = _InstrumentedExtractor(fail_on=set(), timeout_on=set(), delay=0.01)
        _wire(monkeypatch, store=store, catalog=_StubCatalog(chunks), extractor=ext)
        await _index_graph(tenant, "d")
        assert ext.max_in_flight <= limit, f"concurrency {limit}: observed {ext.max_in_flight} in flight"
        # Sanity: with a delay and n>>limit, we should actually reach the bound (proves it ran concurrently).
        if limit > 1:
            assert ext.max_in_flight >= 2, f"concurrency {limit}: never overlapped (max {ext.max_in_flight})"
        assert len(ext.calls) == n


async def test_concurrency_out_of_range_rejected_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """graph_index_concurrency is Field(ge=1, le=32) — out-of-range values fail validation."""
    from pydantic import ValidationError

    for bad in ("0", "-1", "33", "100"):
        monkeypatch.setenv("RAG_GRAPHRAG_ENABLED", "true")
        monkeypatch.setenv("RAG_GRAPH_INDEX_CONCURRENCY", bad)
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            get_settings()
    get_settings.cache_clear()


# --------------------------------------------------------------------------- (3) community_id determinism


async def _make_graph(monkeypatch: pytest.MonkeyPatch, tenant: str):  # type: ignore[no-untyped-def]
    """Build a graph with several clearly-separable communities of DIFFERENT sizes."""
    _enable_graphrag(monkeypatch, concurrency=8)
    from svc_control_plane.pipeline import _index_graph

    chunks: list[Chunk] = []
    mapping: dict[str, GraphTriple] = {}
    specs = [
        ("A1 A2 A3 A4", ["A1", "A2", "A3", "A4"]),
        ("A1 A3", ["A1", "A3"]),
        ("B1 B2 B3", ["B1", "B2", "B3"]),
        ("B1 B3", ["B1", "B3"]),
        ("C1 C2", ["C1", "C2"]),
        ("D1 D2 D3 D4 D5", ["D1", "D2", "D3", "D4", "D5"]),
    ]
    for i, (text, ents) in enumerate(specs):
        c = _chunk(tenant, "d", i, text)
        chunks.append(c)
        edges = [(a, b) if a.lower() <= b.lower() else (b, a) for j, a in enumerate(ents) for b in ents[j + 1 :]]
        mapping[text] = GraphTriple(entities=ents, edges=edges)
    store = _fresh_store()
    ext = _InstrumentedExtractor(fail_on=set(), timeout_on=set(), delay=0.002, jitter=True, mapping=mapping)
    _wire(monkeypatch, store=store, catalog=_StubCatalog(chunks), extractor=ext)
    await _index_graph(tenant, "d")
    return store, chunks


async def test_community_ids_deterministic_regardless_of_order(monkeypatch: pytest.MonkeyPatch) -> None:
    import svc_control_plane.pipeline as pl
    from svc_control_plane import wiring
    from svc_control_plane.pipeline import build_communities

    tenant = "zzz-comm"
    ref_communities = None
    for run in range(8):
        store, chunks = await _make_graph(monkeypatch, tenant)

        async def _resolve_summarizer(tid: str) -> _StubSummarizer:
            return _StubSummarizer()

        monkeypatch.setattr(pl, "get_graph_store", lambda s=store: s)
        monkeypatch.setattr(pl, "get_catalog", lambda c=chunks: _StubCatalog(c))
        monkeypatch.setattr(wiring, "resolve_community_summarizer", _resolve_summarizer)

        await build_communities(tenant)
        coms = await store.list_communities(tenant)
        # ids must be a dense 0,1,2,... prefix, ordered largest-first (tie by min member).
        ids = [c.community_id for c in sorted(coms, key=lambda c: c.community_id)]
        assert ids == list(range(len(coms))), f"community_ids not 0..N: {ids}"
        sizes = [c.size for c in sorted(coms, key=lambda c: c.community_id)]
        assert sizes == sorted(sizes, reverse=True) or all(
            # allow equal-size ties, but never a larger community after a smaller one out of tie-order
            sizes[i] >= sizes[i + 1]
            for i in range(len(sizes) - 1)
        ), f"communities not ordered largest-first: {sizes}"

        snap = [
            (c.community_id, c.size, tuple(c.top_entities), tuple(c.chunk_ids), c.summary, c.label)
            for c in sorted(coms, key=lambda c: c.community_id)
        ]
        if ref_communities is None:
            ref_communities = snap
        else:
            assert snap == ref_communities, f"run {run}: community output differs across runs (non-deterministic)"


async def test_build_communities_summary_isolation_no_write_reorder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-community summarize fan-out completes out of order (jittered), but the SAVED
    list must stay position-keyed: community i keeps its own top_entities/summary."""
    import svc_control_plane.pipeline as pl
    from svc_control_plane import wiring
    from svc_control_plane.pipeline import build_communities

    tenant = "zzz-commsum"
    store, chunks = await _make_graph(monkeypatch, tenant)

    class _SlowSummarizer:
        async def summarize(self, top_entities: list[str], sample_texts: list[str], size: int) -> str:
            # Bigger communities finish LAST -> if the code zipped by completion, ids would scramble.
            await asyncio.sleep(0.001 * size)
            return f"sum:{top_entities[0] if top_entities else '?'}:{size}"

    async def _resolve_summarizer(tid: str) -> _SlowSummarizer:
        return _SlowSummarizer()

    monkeypatch.setattr(pl, "get_graph_store", lambda: store)
    monkeypatch.setattr(pl, "get_catalog", lambda: _StubCatalog(chunks))
    monkeypatch.setattr(wiring, "resolve_community_summarizer", _resolve_summarizer)

    await build_communities(tenant)
    coms = sorted(await store.list_communities(tenant), key=lambda c: c.community_id)
    for c in coms:
        # summary must reference this community's OWN top entity + size (no cross-wiring).
        assert c.summary == f"sum:{c.top_entities[0]}:{c.size}", f"community {c.community_id} summary mismatched"
