"""Conversation persistence (FR-I1, v1.1) — multi-turn chat sessions per knowledge base.

A conversation is an ordered thread of messages (user + assistant) scoped to one KB and
stored in that tenant's own Postgres schema (in-memory twin for CI) — the same hard
namespace isolation as every other per-tenant store. Assistant messages carry the answer's
``trace_id``, status, citations and faithfulness, so the whole thread reconciles with the
audit trail and the citation viewer (FR-I3) can rehydrate sources for any past turn.

This module is persistence-only (no pipeline/wiring imports) so it can be wired without a
cycle; the turn orchestration lives in :mod:`svc_control_plane.pipeline`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings, pg_schema
from rag_schemas import Citation

_log = get_logger(__name__)

Role = Literal["user", "assistant"]


def _now() -> datetime:
    return datetime.now(UTC)


class Message(BaseModel):
    """One turn in a conversation. Assistant turns carry the grounded-answer metadata."""

    message_id: str
    conversation_id: str
    role: Role
    content: str
    trace_id: str | None = None
    status: str | None = Field(default=None, description="Answer status for assistant turns (grounded/abstained/…).")
    citations: list[Citation] = Field(default_factory=list)
    faithfulness: float | None = None
    created_at: datetime = Field(default_factory=_now)


class Conversation(BaseModel):
    conversation_id: str
    tenant_id: str
    title: str
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


@runtime_checkable
class ConversationStore(Protocol):
    async def ensure(self, tenant_id: str) -> None: ...
    async def create(self, conv: Conversation) -> None: ...
    async def get(self, tenant_id: str, conversation_id: str) -> Conversation | None: ...
    async def recent(self, tenant_id: str, limit: int) -> list[Conversation]: ...
    async def add_message(self, tenant_id: str, msg: Message) -> None: ...
    async def messages(self, tenant_id: str, conversation_id: str, limit: int) -> list[Message]: ...
    async def touch(self, tenant_id: str, conversation_id: str, *, title: str | None = None) -> None: ...
    async def delete_citing(self, tenant_id: str, doc_ids: set[str]) -> int:
        """Delete every conversation that cites any of ``doc_ids`` (DPDP erasure scrub —
        derived chat history quoting erased data must go too). Returns threads removed."""
        ...

    async def purge(self, tenant_id: str) -> None:
        """Delete every conversation + message for a KB (used when the KB is deleted)."""
        ...

    async def delete(self, tenant_id: str, conversation_id: str) -> None:
        """Delete one conversation thread (user removing it from their history)."""
        ...


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._convs: dict[str, dict[str, Conversation]] = {}
        self._msgs: dict[str, list[Message]] = {}

    async def ensure(self, tenant_id: str) -> None:
        return None

    async def create(self, conv: Conversation) -> None:
        self._convs.setdefault(conv.tenant_id, {})[conv.conversation_id] = conv
        self._msgs.setdefault(conv.conversation_id, [])

    async def get(self, tenant_id: str, conversation_id: str) -> Conversation | None:
        return self._convs.get(tenant_id, {}).get(conversation_id)

    async def recent(self, tenant_id: str, limit: int) -> list[Conversation]:
        convs = sorted(self._convs.get(tenant_id, {}).values(), key=lambda c: c.updated_at, reverse=True)
        return convs[:limit]

    async def add_message(self, tenant_id: str, msg: Message) -> None:
        self._msgs.setdefault(msg.conversation_id, []).append(msg)

    async def messages(self, tenant_id: str, conversation_id: str, limit: int) -> list[Message]:
        return self._msgs.get(conversation_id, [])[-limit:]

    async def touch(self, tenant_id: str, conversation_id: str, *, title: str | None = None) -> None:
        conv = self._convs.get(tenant_id, {}).get(conversation_id)
        if conv is None:
            return
        updates: dict[str, object] = {"updated_at": _now()}
        if title is not None:
            updates["title"] = title
        self._convs[tenant_id][conversation_id] = conv.model_copy(update=updates)

    async def delete_citing(self, tenant_id: str, doc_ids: set[str]) -> int:
        convs = self._convs.get(tenant_id, {})
        doomed = {
            cid
            for cid, _ in convs.items()
            for m in self._msgs.get(cid, [])
            if any(c.doc_id in doc_ids for c in m.citations)
        }
        for cid in doomed:
            convs.pop(cid, None)
            self._msgs.pop(cid, None)
        return len(doomed)

    async def purge(self, tenant_id: str) -> None:
        for cid in list(self._convs.get(tenant_id, {})):
            self._msgs.pop(cid, None)
        self._convs.pop(tenant_id, None)

    async def delete(self, tenant_id: str, conversation_id: str) -> None:
        self._convs.get(tenant_id, {}).pop(conversation_id, None)
        self._msgs.pop(conversation_id, None)


_DDL = (
    """CREATE TABLE IF NOT EXISTS "{schema}".conversations (
        conversation_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS "{schema}".conversation_messages (
        message_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        trace_id TEXT,
        status TEXT,
        citations JSONB NOT NULL DEFAULT '[]',
        faithfulness DOUBLE PRECISION,
        created_at TIMESTAMPTZ NOT NULL
    )""",
    'CREATE INDEX IF NOT EXISTS conv_msg_by_conv ON "{schema}".conversation_messages (conversation_id, created_at)',
)


class PostgresConversationStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self, tenant_id: str) -> None:
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            for stmt in _DDL:
                await conn.execute(text(stmt.format(schema=schema)))

    async def create(self, conv: Conversation) -> None:
        await self.ensure(conv.tenant_id)
        schema = pg_schema(conv.tenant_id)
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{schema}".conversations (conversation_id, title, created_at, updated_at) '
                    "VALUES (:cid, :title, :created, :updated) ON CONFLICT (conversation_id) DO NOTHING"
                ),
                {
                    "cid": conv.conversation_id,
                    "title": conv.title,
                    "created": conv.created_at,
                    "updated": conv.updated_at,
                },
            )

    async def get(self, tenant_id: str, conversation_id: str) -> Conversation | None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as c:
            row = (
                (
                    await c.execute(
                        text(f'SELECT * FROM "{schema}".conversations WHERE conversation_id = :cid'),
                        {"cid": conversation_id},
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return Conversation(
            conversation_id=str(row["conversation_id"]),
            tenant_id=tenant_id,
            title=str(row["title"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def recent(self, tenant_id: str, limit: int) -> list[Conversation]:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as c:
            rows = (
                (
                    await c.execute(
                        text(f'SELECT * FROM "{schema}".conversations ORDER BY updated_at DESC LIMIT :limit'),
                        {"limit": limit},
                    )
                )
                .mappings()
                .all()
            )
        return [
            Conversation(
                conversation_id=str(r["conversation_id"]),
                tenant_id=tenant_id,
                title=str(r["title"]),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    async def add_message(self, tenant_id: str, msg: Message) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as c:
            await c.execute(
                text(
                    f'INSERT INTO "{schema}".conversation_messages (message_id, conversation_id, role, '
                    "content, trace_id, status, citations, faithfulness, created_at) VALUES (:mid, :cid, "
                    ":role, :content, :trace, :status, CAST(:citations AS JSONB), :faith, :created) "
                    "ON CONFLICT (message_id) DO NOTHING"
                ),
                {
                    "mid": msg.message_id,
                    "cid": msg.conversation_id,
                    "role": msg.role,
                    "content": msg.content,
                    "trace": msg.trace_id,
                    "status": msg.status,
                    "citations": json.dumps([c.model_dump(mode="json") for c in msg.citations]),
                    "faith": msg.faithfulness,
                    "created": msg.created_at,
                },
            )

    async def messages(self, tenant_id: str, conversation_id: str, limit: int) -> list[Message]:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.connect() as c:
            rows = (
                (
                    await c.execute(
                        text(
                            f'SELECT * FROM "{schema}".conversation_messages WHERE conversation_id = :cid '
                            "ORDER BY created_at ASC LIMIT :limit"
                        ),
                        {"cid": conversation_id, "limit": limit},
                    )
                )
                .mappings()
                .all()
            )
        return [_row_to_message(dict(r)) for r in rows]

    async def touch(self, tenant_id: str, conversation_id: str, *, title: str | None = None) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        clause = ", title = :title" if title is not None else ""
        params: dict[str, object] = {"cid": conversation_id, "updated": _now()}
        if title is not None:
            params["title"] = title
        async with self._engine.begin() as c:
            await c.execute(
                text(f'UPDATE "{schema}".conversations SET updated_at = :updated{clause} WHERE conversation_id = :cid'),
                params,
            )

    async def delete_citing(self, tenant_id: str, doc_ids: set[str]) -> int:
        if not doc_ids:
            return 0
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        ids = list(doc_ids)
        async with self._engine.begin() as c:
            targets = (
                await c.execute(
                    text(
                        f'SELECT DISTINCT conversation_id FROM "{schema}".conversation_messages m '
                        "WHERE EXISTS (SELECT 1 FROM jsonb_array_elements(m.citations) e "
                        "WHERE e->>'doc_id' = ANY(:docs))"
                    ),
                    {"docs": ids},
                )
            ).all()
            cids = [str(r.conversation_id) for r in targets]
            if cids:
                await c.execute(
                    text(f'DELETE FROM "{schema}".conversation_messages WHERE conversation_id = ANY(:cids)'),
                    {"cids": cids},
                )
                await c.execute(
                    text(f'DELETE FROM "{schema}".conversations WHERE conversation_id = ANY(:cids)'),
                    {"cids": cids},
                )
        return len(cids)

    async def purge(self, tenant_id: str) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as c:
            await c.execute(text(f'DELETE FROM "{schema}".conversation_messages'))
            await c.execute(text(f'DELETE FROM "{schema}".conversations'))

    async def delete(self, tenant_id: str, conversation_id: str) -> None:
        await self.ensure(tenant_id)
        schema = pg_schema(tenant_id)
        async with self._engine.begin() as c:
            await c.execute(
                text(f'DELETE FROM "{schema}".conversation_messages WHERE conversation_id = :cid'),
                {"cid": conversation_id},
            )
            await c.execute(
                text(f'DELETE FROM "{schema}".conversations WHERE conversation_id = :cid'),
                {"cid": conversation_id},
            )


def _row_to_message(m: dict[str, object]) -> Message:
    raw = m.get("citations")
    cites: list[object] = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
    faith = m.get("faithfulness")
    return Message(
        message_id=str(m["message_id"]),
        conversation_id=str(m["conversation_id"]),
        role=str(m["role"]),
        content=str(m["content"]),
        trace_id=(str(m["trace_id"]) if m.get("trace_id") else None),
        status=(str(m["status"]) if m.get("status") else None),
        citations=[Citation.model_validate(c) for c in cites],
        faithfulness=(float(faith) if isinstance(faith, (int, float)) else None),
        created_at=m["created_at"],
    )
