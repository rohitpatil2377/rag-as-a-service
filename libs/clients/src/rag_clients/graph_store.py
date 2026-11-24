"""Knowledge-graph store adapters (implement :class:`GraphStore`) — GraphRAG (FR-C6/D4).

Per-KB adjacency tables in the tenant's own Postgres schema (in-memory twin for CI):
* ``graph_nodes(name, chunk_ids)`` — an entity and the chunks that mention it;
* ``graph_edges(source, target, chunk_ids)`` — a co-occurrence relationship + its evidence.

Mentions/weights are derived from the chunk-id sets, so re-adding a chunk is idempotent
(re-index safe). Local graph search resolves query entities case-insensitively, walks one
hop, and returns the chunk ids of the entity + its neighbours. Neo4j sits behind this same
Protocol for graph scale later.
"""

from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings, pg_schema
from rag_core.graph import GraphCommunity, GraphNode, GraphStats, normalize_entity

_log = get_logger(__name__)


def _key(name: str) -> str:
    return normalize_entity(name).lower()


class InMemoryGraphStore:
    def __init__(self) -> None:
        # tenant -> {node_key: (display_name, {chunk_ids})}
        self._nodes: dict[str, dict[str, tuple[str, set[str]]]] = {}
        # tenant -> {(a_key, b_key): {chunk_ids}}
        self._edges: dict[str, dict[tuple[str, str], set[str]]] = {}
        # tenant -> {(a_key, b_key): relation label} — diagram-sourced edges only; absent
        # (co-occurrence) edges simply have no entry here.
        self._edge_relations: dict[str, dict[tuple[str, str], str]] = {}
        # tenant -> [communities]
        self._communities: dict[str, list[GraphCommunity]] = {}

    async def ensure(self, tenant_id: str) -> None:
        return None

    async def add_chunk(self, tenant_id: str, chunk_id: str, entities: list[str], edges: list[tuple[str, str]]) -> None:
        nodes = self._nodes.setdefault(tenant_id, {})
        for name in entities:
            k = _key(name)
            display, chunks = nodes.get(k, (normalize_entity(name), set()))
            chunks.add(chunk_id)
            nodes[k] = (display, chunks)
        emap = self._edges.setdefault(tenant_id, {})
        for a, b in edges:
            ek = tuple(sorted((_key(a), _key(b))))
            emap.setdefault(ek, set()).add(chunk_id)  # type: ignore[arg-type]

    async def add_diagram_edges(
        self, tenant_id: str, chunk_id: str, entities: list[str], edges: list[tuple[str, str, str]]
    ) -> None:
        """Diagram-sourced write path (additive, separate from :meth:`add_chunk`'s
        co-occurrence fan-out): each parsed Mermaid/vision-JSON arrow becomes exactly ONE
        ``(source, target, relation)`` edge with direction preserved as drawn — never every
        pairwise combination of the chunk's entities."""
        nodes = self._nodes.setdefault(tenant_id, {})
        for name in entities:
            k = _key(name)
            display, chunks = nodes.get(k, (normalize_entity(name), set()))
            chunks.add(chunk_id)
            nodes[k] = (display, chunks)
        emap = self._edges.setdefault(tenant_id, {})
        rmap = self._edge_relations.setdefault(tenant_id, {})
        for a, b, rel in edges:
            ek = (_key(a), _key(b))  # NOT sorted — direction is the whole point here
            emap.setdefault(ek, set()).add(chunk_id)
            if rel:
                rmap[ek] = rel

    async def remove_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        drop = set(chunk_ids)
        if not drop:
            return
        nodes = self._nodes.get(tenant_id, {})
        for key, (display, cids) in list(nodes.items()):
            left = cids - drop
            if left:
                nodes[key] = (display, left)
            else:
                nodes.pop(key, None)  # no evidence left: the entity is not in the KB any more
        emap = self._edges.get(tenant_id, {})
        rmap = self._edge_relations.get(tenant_id, {})
        for pair, cids in list(emap.items()):
            left = cids - drop
            if left:
                emap[pair] = left
            else:
                emap.pop(pair, None)
                rmap.pop(pair, None)

    async def linked_chunks(self, tenant_id: str, entities: list[str], hops: int = 1) -> list[str]:
        nodes = self._nodes.get(tenant_id, {})
        emap = self._edges.get(tenant_id, {})
        seeds = {_key(e) for e in entities if _key(e) in nodes}
        if not seeds:
            return []
        reachable = set(seeds)
        frontier = set(seeds)
        for _ in range(max(1, hops)):
            nxt: set[str] = set()
            for a, b in emap:
                if a in frontier and b not in reachable:
                    nxt.add(b)
                elif b in frontier and a not in reachable:
                    nxt.add(a)
            reachable |= nxt
            frontier = nxt
            if not frontier:
                break
        chunks: set[str] = set()
        for k in reachable:
            if k in nodes:
                chunks |= nodes[k][1]
        return sorted(chunks)

    async def stats(self, tenant_id: str, top: int = 10) -> GraphStats:
        nodes = self._nodes.get(tenant_id, {})
        ranked = sorted(((d, len(c)) for d, c in nodes.values()), key=lambda x: x[1], reverse=True)
        return GraphStats(nodes=len(nodes), edges=len(self._edges.get(tenant_id, {})), top_entities=ranked[:top])

    async def clear(self, tenant_id: str) -> None:
        self._nodes.pop(tenant_id, None)
        self._edges.pop(tenant_id, None)
        self._edge_relations.pop(tenant_id, None)
        self._communities.pop(tenant_id, None)

    async def graph_data(self, tenant_id: str) -> tuple[list[GraphNode], list[tuple[str, str]]]:
        nodes = [
            GraphNode(key=k, display=d, chunk_ids=sorted(c)) for k, (d, c) in self._nodes.get(tenant_id, {}).items()
        ]
        edges = list(self._edges.get(tenant_id, {}).keys())
        return nodes, edges

    async def save_communities(self, tenant_id: str, communities: list[GraphCommunity]) -> None:
        self._communities[tenant_id] = list(communities)

    async def list_communities(self, tenant_id: str) -> list[GraphCommunity]:
        return list(self._communities.get(tenant_id, []))


_DDL = (
    """CREATE TABLE IF NOT EXISTS "{schema}".graph_nodes (
        name_key TEXT PRIMARY KEY,
        display TEXT NOT NULL,
        chunk_ids JSONB NOT NULL DEFAULT '[]'
    )""",
    """CREATE TABLE IF NOT EXISTS "{schema}".graph_edges (
        source TEXT NOT NULL,
        target TEXT NOT NULL,
        chunk_ids JSONB NOT NULL DEFAULT '[]',
        relation TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (source, target)
    )""",
    # ADD COLUMN IF NOT EXISTS: this table predates `relation` on already-provisioned tenant
    # schemas, so the CREATE TABLE IF NOT EXISTS above silently no-ops for them and never adds
    # it — mirror the migration-safe pattern used for federation_grants.parent_grant_id /
    # eval_schedules.federate_from. Existing (co-occurrence) rows just get relation = ''.
    """ALTER TABLE "{schema}".graph_edges ADD COLUMN IF NOT EXISTS relation TEXT NOT NULL DEFAULT ''""",
    """CREATE TABLE IF NOT EXISTS "{schema}".graph_communities (
        community_id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        size INTEGER NOT NULL,
        top_entities JSONB NOT NULL DEFAULT '[]',
        chunk_ids JSONB NOT NULL DEFAULT '[]',
        summary TEXT NOT NULL DEFAULT ''
    )""",
    # ALL member node keys per community — lets the subgraph tag every visible node with its real
    # theme (previously only top_entities mapped, so most nodes were unassigned and a theme click
    # matched ~0 nodes). Migration-safe, same pattern as graph_edges.relation above.
    """ALTER TABLE "{schema}".graph_communities ADD COLUMN IF NOT EXISTS members JSONB NOT NULL DEFAULT '[]'""",
)


class PostgresGraphStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self, tenant_id: str) -> None:
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as c:
            await c.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            for stmt in _DDL:
                await c.execute(text(stmt.format(schema=schema)))

    async def add_chunk(self, tenant_id: str, chunk_id: str, entities: list[str], edges: list[tuple[str, str]]) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        # Append chunk_id to the JSONB set only if absent (idempotent for re-index).
        cid = "to_jsonb(CAST(:cid AS text))"
        ncol = f'"{schema}".graph_nodes.chunk_ids'
        ecol = f'"{schema}".graph_edges.chunk_ids'
        node_sql = text(
            f'INSERT INTO "{schema}".graph_nodes (name_key, display, chunk_ids) '
            "VALUES (:k, :d, jsonb_build_array(CAST(:cid AS text))) "
            f"ON CONFLICT (name_key) DO UPDATE SET chunk_ids = "
            f"CASE WHEN {ncol} @> {cid} THEN {ncol} ELSE {ncol} || {cid} END"
        )
        edge_sql = text(
            f'INSERT INTO "{schema}".graph_edges (source, target, chunk_ids) '
            "VALUES (:s, :t, jsonb_build_array(CAST(:cid AS text))) "
            f"ON CONFLICT (source, target) DO UPDATE SET chunk_ids = "
            f"CASE WHEN {ecol} @> {cid} THEN {ecol} ELSE {ecol} || {cid} END"
        )
        async with self._engine.begin() as c:
            for name in entities:
                await c.execute(node_sql, {"k": _key(name), "d": normalize_entity(name), "cid": chunk_id})
            for a, b in edges:
                s, t = sorted((_key(a), _key(b)))
                await c.execute(edge_sql, {"s": s, "t": t, "cid": chunk_id})

    async def add_diagram_edges(
        self, tenant_id: str, chunk_id: str, entities: list[str], edges: list[tuple[str, str, str]]
    ) -> None:
        """Diagram-sourced write path (additive, separate from :meth:`add_chunk`'s
        co-occurrence fan-out): each parsed Mermaid/vision-JSON arrow becomes exactly ONE
        ``(source, target, relation)`` row, direction preserved exactly as drawn (source/target
        are NOT alphabetically sorted, unlike the symmetric co-occurrence edges above) — a
        9-node/14-edge diagram now writes 14 edge rows, not a near-complete pairwise clique."""
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        cid = "to_jsonb(CAST(:cid AS text))"
        ncol = f'"{schema}".graph_nodes.chunk_ids'
        ecol = f'"{schema}".graph_edges.chunk_ids'
        rcol = f'"{schema}".graph_edges.relation'
        node_sql = text(
            f'INSERT INTO "{schema}".graph_nodes (name_key, display, chunk_ids) '
            "VALUES (:k, :d, jsonb_build_array(CAST(:cid AS text))) "
            f"ON CONFLICT (name_key) DO UPDATE SET chunk_ids = "
            f"CASE WHEN {ncol} @> {cid} THEN {ncol} ELSE {ncol} || {cid} END"
        )
        edge_sql = text(
            f'INSERT INTO "{schema}".graph_edges (source, target, chunk_ids, relation) '
            "VALUES (:s, :t, jsonb_build_array(CAST(:cid AS text)), :rel) "
            f"ON CONFLICT (source, target) DO UPDATE SET "
            f"chunk_ids = CASE WHEN {ecol} @> {cid} THEN {ecol} ELSE {ecol} || {cid} END, "
            f"relation = CASE WHEN :rel <> '' THEN :rel ELSE {rcol} END"
        )
        async with self._engine.begin() as c:
            for name in entities:
                await c.execute(node_sql, {"k": _key(name), "d": normalize_entity(name), "cid": chunk_id})
            for a, b, rel in edges:
                s, t = _key(a), _key(b)  # NOT sorted — direction is the whole point here
                await c.execute(edge_sql, {"s": s, "t": t, "cid": chunk_id, "rel": rel or ""})

    async def remove_chunks(self, tenant_id: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        # Subtract the ids from each JSONB set, then drop rows left with no evidence.
        async with self._engine.begin() as c:
            for table in ("graph_nodes", "graph_edges"):
                await c.execute(
                    text(
                        f'UPDATE "{schema}".{table} SET chunk_ids = COALESCE(('
                        f"SELECT jsonb_agg(v) FROM jsonb_array_elements(chunk_ids) AS v "
                        "WHERE NOT (v::text = ANY(:quoted))), '[]'::jsonb)"
                    ),
                    {"quoted": [f'"{cid}"' for cid in chunk_ids]},
                )
                await c.execute(text(f'DELETE FROM "{schema}".{table} WHERE jsonb_array_length(chunk_ids) = 0'))

    async def linked_chunks(self, tenant_id: str, entities: list[str], hops: int = 1) -> list[str]:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        seeds = [_key(e) for e in entities]
        async with self._engine.connect() as c:
            present = {
                str(r.name_key)
                for r in (
                    await c.execute(
                        text(f'SELECT name_key FROM "{schema}".graph_nodes WHERE name_key = ANY(:ks)'), {"ks": seeds}
                    )
                ).all()
            }
            if not present:
                return []
            reachable = set(present)
            frontier = set(present)
            for _ in range(max(1, hops)):
                if not frontier:
                    break
                rows = (
                    await c.execute(
                        text(
                            f'SELECT source, target FROM "{schema}".graph_edges '
                            "WHERE source = ANY(:f) OR target = ANY(:f)"
                        ),
                        {"f": list(frontier)},
                    )
                ).all()
                nxt = {
                    (r.target if r.source in frontier else r.source)
                    for r in rows
                    if (r.source in frontier and r.target not in reachable)
                    or (r.target in frontier and r.source not in reachable)
                }
                reachable |= nxt
                frontier = nxt
            rows = (
                await c.execute(
                    text(f'SELECT chunk_ids FROM "{schema}".graph_nodes WHERE name_key = ANY(:ks)'),
                    {"ks": list(reachable)},
                )
            ).all()
        chunks: set[str] = set()
        for r in rows:
            raw = r.chunk_ids
            chunks |= set(json.loads(raw) if isinstance(raw, str) else raw)
        return sorted(chunks)

    async def stats(self, tenant_id: str, top: int = 10) -> GraphStats:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as c:
            nodes = int((await c.execute(text(f'SELECT COUNT(*) FROM "{schema}".graph_nodes'))).scalar() or 0)
            edges = int((await c.execute(text(f'SELECT COUNT(*) FROM "{schema}".graph_edges'))).scalar() or 0)
            rows = (
                await c.execute(
                    text(
                        f'SELECT display, jsonb_array_length(chunk_ids) AS n FROM "{schema}".graph_nodes '
                        "ORDER BY n DESC LIMIT :top"
                    ),
                    {"top": top},
                )
            ).all()
        return GraphStats(nodes=nodes, edges=edges, top_entities=[(str(r.display), int(r.n)) for r in rows])

    async def clear(self, tenant_id: str) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as c:
            await c.execute(text(f'DELETE FROM "{schema}".graph_edges'))
            await c.execute(text(f'DELETE FROM "{schema}".graph_nodes'))
            await c.execute(text(f'DELETE FROM "{schema}".graph_communities'))

    async def graph_data(self, tenant_id: str) -> tuple[list[GraphNode], list[tuple[str, str]]]:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as c:
            node_rows = (
                await c.execute(text(f'SELECT name_key, display, chunk_ids FROM "{schema}".graph_nodes'))
            ).all()
            edge_rows = (await c.execute(text(f'SELECT source, target FROM "{schema}".graph_edges'))).all()
        nodes = [
            GraphNode(
                key=str(r.name_key),
                display=str(r.display),
                chunk_ids=list(json.loads(r.chunk_ids) if isinstance(r.chunk_ids, str) else r.chunk_ids),
            )
            for r in node_rows
        ]
        edges = [(str(r.source), str(r.target)) for r in edge_rows]
        return nodes, edges

    async def save_communities(self, tenant_id: str, communities: list[GraphCommunity]) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        ins = text(
            f'INSERT INTO "{schema}".graph_communities '
            "(community_id, label, size, top_entities, chunk_ids, summary, members) "
            "VALUES (:cid, :label, :size, CAST(:ents AS jsonb), CAST(:chunks AS jsonb), "
            ":summary, CAST(:members AS jsonb))"
        )
        async with self._engine.begin() as c:
            await c.execute(text(f'DELETE FROM "{schema}".graph_communities'))
            for com in communities:
                await c.execute(
                    ins,
                    {
                        "cid": com.community_id,
                        "label": com.label,
                        "size": com.size,
                        "ents": json.dumps(com.top_entities),
                        "chunks": json.dumps(com.chunk_ids),
                        "summary": com.summary,
                        "members": json.dumps(com.members),
                    },
                )

    async def list_communities(self, tenant_id: str) -> list[GraphCommunity]:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as c:
            rows = (
                await c.execute(
                    text(
                        f"SELECT community_id, label, size, top_entities, chunk_ids, summary, members "
                        f'FROM "{schema}".graph_communities ORDER BY size DESC, community_id'
                    )
                )
            ).all()

        def _list(raw: object) -> list[str]:
            data = json.loads(raw) if isinstance(raw, str) else raw
            return [str(x) for x in data] if isinstance(data, list) else []

        return [
            GraphCommunity(
                community_id=int(r.community_id),
                label=str(r.label),
                size=int(r.size),
                top_entities=_list(r.top_entities),
                chunk_ids=_list(r.chunk_ids),
                summary=str(r.summary),
                members=_list(r.members),
            )
            for r in rows
        ]
