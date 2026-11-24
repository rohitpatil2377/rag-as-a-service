"""Tenancy context — the spine of multi-tenant isolation.

``tenant_id`` must flow end-to-end (API -> retrieval -> store). We propagate it via
a :class:`contextvars.ContextVar` so it is available deep in the call stack (and
across ``await`` boundaries) without threading it through every signature, while
still being explicit at trust boundaries.

Store adapters MUST derive their namespace (collection / index / bucket-prefix /
schema) from :func:`current_tenant` — never from a caller-supplied value that was
not authenticated. See ``tests/integration/test_tenant_isolation.py``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from rag_common.errors import TenantIsolationError

_tenant_var: ContextVar[str | None] = ContextVar("rag_tenant_id", default=None)

# Namespacing helpers ---------------------------------------------------------
# Single source of truth for how a tenant_id maps to each store's namespace, so
# isolation logic is not re-implemented (and mis-implemented) per adapter.

_NS_PREFIX = "t_"


def _slug(tenant_id: str) -> str:
    if not tenant_id or not all(c.isalnum() or c in "-_" for c in tenant_id):
        raise TenantIsolationError("invalid tenant_id", details={"tenant_id": tenant_id})
    return tenant_id.replace("-", "_").lower()


def assert_registrable(new_id: str, existing_ids: Iterable[str]) -> None:
    """Fail-closed guard against the non-injective slug collapsing two DISTINCT tenant ids
    into ONE physical namespace.

    ``_slug`` lowercases and maps ``-`` -> ``_``, so ``org-x``/``org_x`` and ``Acme``/``acme``
    derive the SAME Qdrant collection / OpenSearch index / MinIO prefix / Postgres schema —
    yet ``tenant_guard`` compares the RAW strings, so each id passes its own ownership check
    while sharing the store. That is a cross-tenant data leak. Called at every registration
    entry point: reject a NEW id whose derived slug collides with any existing sibling's slug
    (order-independent — whichever id registers second is refused, so the pair can never both
    exist)."""
    target = _slug(new_id)  # also validates the charset (raises on bad chars)
    for existing in existing_ids:
        if existing != new_id and _slug(existing) == target:
            raise TenantIsolationError(
                "tenant_id would collide with an existing namespace (differs only by case or -/_)",
                details={"tenant_id": new_id, "collides_with": existing},
            )


def vector_collection(tenant_id: str) -> str:
    """Qdrant collection name for a tenant."""
    return f"{_NS_PREFIX}{_slug(tenant_id)}_chunks"


def vision_collection(tenant_id: str) -> str:
    """Qdrant collection name for a tenant's CROSS-MODAL (CLIP) image vectors.

    A SEPARATE collection from :func:`vector_collection`: the vision embedder has its own
    dimensionality and vector space, so image vectors cannot share the text collection. Same
    per-tenant isolation discipline — a caller only ever touches the collection derived from
    its explicit ``tenant_id``."""
    return f"{_NS_PREFIX}{_slug(tenant_id)}_vchunks"


def sparse_index(tenant_id: str) -> str:
    """OpenSearch index name for a tenant."""
    return f"{_NS_PREFIX}{_slug(tenant_id)}_bm25"


def object_prefix(tenant_id: str) -> str:
    """Object-store key prefix (within a shared or per-tenant bucket)."""
    return f"{_NS_PREFIX}{_slug(tenant_id)}/"


def pg_schema(tenant_id: str) -> str:
    """Postgres schema name for a tenant (schema-per-tenant discipline)."""
    return f"{_NS_PREFIX}{_slug(tenant_id)}"


def org_of(kb_id: str) -> str:
    """The owning ORG id for a knowledge-base id.

    A KB id is ``{org}__{slug}`` (the org's default KB is simply ``{org}`` with no ``__``),
    so the org is the segment before the first ``__``. For a single-KB org (``kb_id == org``)
    this returns ``kb_id`` unchanged — callers rely on that to detect the "KB schema IS the org
    schema" case (``pg_schema(org_of(kb)) == pg_schema(kb)``) and treat cross-schema logic as a
    no-op. Single source of truth so the split is not re-implemented (and mis-implemented)
    per call site (mirrors the inline ``kb_id.split("__", 1)[0]`` already used in wiring)."""
    return kb_id.split("__", 1)[0]


# A doc id that is safe as an object-store key: alnum/._- segments, single slashes, no scheme.
_OBJECT_SAFE_DOC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9._-]+)*$")


def _object_safe_doc_id(doc_id: str) -> str:
    """A doc id in a form that is legal as an object-store key, deterministically.

    A crawl/web doc id is the page URL (``http://host/path``), whose ``:`` and ``//`` are illegal
    object names in stores like MinIO/S3. Such ids are encoded to a deterministic, collision-free
    safe form (slug + content hash). Ordinary ids (uploads) are object-safe and pass through
    unchanged, so existing objects keep their keys.

    ONE place, shared by every per-document key below: a second implementation of this rule would
    diverge on exactly the ids it exists for, and a crawl document's artifacts would then land
    under keys that do not correspond — or fail to be written and purged at all."""
    if _OBJECT_SAFE_DOC_ID.match(doc_id):
        return doc_id
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_id).strip("_")[:80] or "doc"
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:12]  # noqa: S324 — key uniqueness, not security
    return f"{slug}-{digest}"


def raw_object_key(doc_id: str) -> str:
    """Object-store key for a document's raw bytes."""
    return f"raw/{_object_safe_doc_id(doc_id)}"


def layout_object_key(doc_id: str) -> str:
    """Object-store key for a document's LAYOUT OVERLAY — the extraction engine's raw markup
    (``svc_doc_intelligence.ocr.LayoutOverlay``, stored as JSON).

    Mirrors :func:`raw_object_key` exactly, down to the safe-key encoding it shares, so a crawl
    document's overlay is reachable and purgeable under the same rule as its raw bytes. A
    SEPARATE prefix rather than a suffix on the raw key: the two are different artifacts with
    different content types, and an operator listing ``raw/`` must not find JSON in it.

    Compliance: this artifact is in the SAME class as the raw bytes — see
    ``svc_control_plane.pipeline._persist_layout_overlay``."""
    return f"layout/{_object_safe_doc_id(doc_id)}"


# Context management ----------------------------------------------------------


def set_tenant(tenant_id: str) -> Token[str | None]:
    """Bind the current tenant; returns a token for :func:`reset_tenant`."""
    _slug(tenant_id)  # validate early
    return _tenant_var.set(tenant_id)


def reset_tenant(token: Token[str | None]) -> None:
    _tenant_var.reset(token)


def current_tenant() -> str:
    """Return the bound tenant_id, or raise if none (fail-closed)."""
    tid = _tenant_var.get()
    if tid is None:
        raise TenantIsolationError("no tenant bound to the current context")
    return tid


@contextmanager
def tenant_scope(tenant_id: str) -> Iterator[str]:
    """Scope a block of work to a tenant (used by workers, tests, demo CLI)."""
    token = set_tenant(tenant_id)
    try:
        yield tenant_id
    finally:
        reset_tenant(token)
