"""Federation transport seam — how ``answer_federated`` reaches a peer department.

Two topologies, one contract (see docs/adr/0017):

* **LocalTransport** — the peer's KBs live in THIS deployment (the modular-monolith,
  multi-tenant case). Retrieves in-process. The DEFAULT, and identity-preserving: with it,
  ``answer_federated`` behaves exactly as before this seam existed.
* **MeshTransport** — the peer is a REMOTE, independently-deployed department. Routes the
  grant-authorised query to the LOCAL ``sovereign-node``, which dials the peer over
  noise-encrypted libp2p (``/rag/query/1.0``) and returns spans. This is the true
  decentralised path; the Go node is the on-the-wire carrier.

Grant resolution stays in the caller (``answer_federated`` calls ``grant_store.get_active``):
a department must hold an active grant to even attempt a peer, and for the mesh path that
same grant is the token presented to the remote node, which re-validates it fail-closed.
Defence-in-depth on both paths.

Sovereignty note: ``federation_node_url`` is the operator's OWN local node (localhost /
intranet infrastructure, like the Qdrant/Redis URLs) — it is not egress-guarded. The
cross-organisation hop happens inside the libp2p mesh, where the remote node enforces grants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

from rag_common import current_traceparent, get_logger, get_settings
from rag_schemas import Grant, RetrievalQuery, RetrievedSpan
from svc_control_plane import wiring
from svc_control_plane.grants import GrantStore, verify_grant

_log = get_logger(__name__)

# One "source" is one result set fed to RRF fusion; a peer contributes one per KB it exposes.
SourceSpans = list[RetrievedSpan]


@dataclass(frozen=True)
class PeerFetch:
    """One peer\'s contribution to a federated answer, plus whether we actually reached it.

    ``reachable`` is False ONLY for an availability failure (peer down/slow/unroutable), NOT
    for a grant rejection — that is a reached-and-answered "no". answer_federated uses it to
    report completeness honestly (T1): a down granted peer is named unreachable, never
    silently dropped."""

    sources: list[SourceSpans] = field(default_factory=list)
    used_grants: list[dict[str, Any]] = field(default_factory=list)
    reachable: bool = True
    reason: str = "ok"


def _grant_record(peer_dept_id: str, grant: Grant, *, transport: str | None = None) -> dict[str, Any]:
    """The audit record for a used grant. LocalTransport keeps the exact historical shape;
    MeshTransport adds ``transport`` so a mesh hop is visible in the trail."""
    record: dict[str, Any] = {
        "dept": peer_dept_id,
        "grant_id": grant.grant_id,
        "kb_ids": grant.kb_ids,
        "topics": grant.topics,
    }
    if transport:
        record["transport"] = transport
    return record


def _topic_scoped(spans: list[RetrievedSpan], grant: Grant, *, enabled: bool) -> list[RetrievedSpan]:
    """Fail-closed grant-topic filter (ADR-0017 L1). With enforcement on, keep only spans whose
    primary category is one of the grant's topics (case-insensitive); a span with no category
    cannot be shown to fall under a granted topic, so it is dropped. Off = unchanged."""
    if not enabled:
        return list(spans)
    allowed = {t.strip().lower() for t in grant.topics}
    return [s for s in spans if s.category and s.category.strip().lower() in allowed]


async def _grant_verifies(grant: Grant, grant_store: GrantStore | None) -> bool:
    """Fail-closed use-time grant check — the SAME guarantee the remote node enforces on the
    mesh path, applied locally so a co-located retrieval never serves a peer's spans under a
    forged, expired, or revoked grant (defence-in-depth; the caller already pre-filters, this
    re-checks at the point of use).

    1. Ed25519 SIGNATURE against the issuer's trusted verifying keyring (active + retired-overlap):
       a token signed by an untrusted key, tampered, or missing verifies False -> deny.
    2. Time window + direct revocation via ``Grant.is_active`` (not_before/expires_at/revoked_at).
    3. If the transport has store access, the WHOLE ancestry must be live
       (``is_chain_active``) — a revoked/expired root kills every descendant (transitive
       revocation), matching the caller's chain gate.
    Any failure -> False (no spans from this grant). Never raises to the caller."""
    if not verify_grant(grant):
        _log.warning("federation_local_grant_signature_invalid", grant_id=grant.grant_id)
        return False
    if not grant.is_active:
        _log.warning("federation_local_grant_inactive", grant_id=grant.grant_id)
        return False
    if grant_store is not None:
        try:
            if not await grant_store.is_chain_active(grant.grant_id):
                _log.warning("federation_local_grant_chain_inactive", grant_id=grant.grant_id)
                return False
        except Exception as exc:  # a store hiccup must fail CLOSED, never open
            _log.warning("federation_local_grant_chain_check_failed", grant_id=grant.grant_id, error=str(exc))
            return False
    return True


@runtime_checkable
class FederationTransport(Protocol):
    """Fetch grant-authorised spans from ONE peer department.

    Returns a :class:`PeerFetch` — result sets for RRF fusion, audit records, and whether
    the peer was reachable. Fail-open on availability (unreachable → no spans, reported),
    fail-closed on authorisation (a rejected grant → no spans, reached)."""

    async def fetch(
        self, *, caller_dept: str, peer_dept_id: str, grants: list[Grant], query: str, top_k: int
    ) -> PeerFetch: ...


class LocalTransport:
    """In-process retrieval across peer KBs that live in THIS deployment — the intra-org
    shared-compute case (ADR-0019). Identity-preserving: the exact loop ``answer_federated``
    used before the seam.

    CO-LOCATED ONLY. This is NOT a remote transport and must not be used to reach a
    separately-deployed department — that is ``MeshTransport`` (libp2p), the one
    authoritative remote transport per ADR-0020. The seam is Local (co-located) + Mesh
    (remote), never two competing remote transports.

    Fail-closed authorisation (parity with the mesh path): when wired for verification
    (``verify=True`` + a ``grant_store``, as the registry does in production) each grant is
    re-validated at USE time before ANY of its spans are served — Ed25519 signature against
    the trusted keyring, ``not_before``/``expires_at``/revocation, and (with store access)
    the whole delegation chain. A grant that fails is dropped with no spans; if EVERY grant
    fails the peer is reported reached-and-denied (``grant_rejected``), exactly like a remote
    node rejecting the token. The bare ``LocalTransport()`` (no verification wired) preserves
    the pre-seam in-process behaviour."""

    def __init__(self, *, grant_store: GrantStore | None = None, verify: bool = False) -> None:
        self._grant_store = grant_store
        # Verification is fail-closed once enabled; the registry turns it ON for the real
        # co-located path, while a bare construction stays identity-preserving.
        self._verify = verify

    async def fetch(
        self, *, caller_dept: str, peer_dept_id: str, grants: list[Grant], query: str, top_k: int
    ) -> PeerFetch:
        per_source: list[SourceSpans] = []
        used: list[dict[str, Any]] = []
        denied = 0
        for grant in grants:
            if self._verify and not await _grant_verifies(grant, self._grant_store):
                # Reached-and-refused: this grant serves NO spans (fail closed), like a remote
                # node rejecting the token. Not an availability failure — the peer IS co-located.
                denied += 1
                continue
            cap = min(top_k, grant.max_results_per_query)
            for kb in grant.kb_ids:
                # Least privilege across the boundary: public-ACL content only.
                result = await (await wiring.retrieval_service_for(kb)).retrieve(
                    RetrievalQuery(tenant_id=kb, text=query, top_k=cap, acl=[])
                )
                spans = _topic_scoped(result.spans, grant, enabled=get_settings().federation_topic_enforcement)
                per_source.append(
                    [
                        s.model_copy(update={"chunk_id": f"{kb}::{s.chunk_id}", "doc_id": f"{peer_dept_id}/{s.doc_id}"})
                        for s in spans
                    ]
                )
            used.append(_grant_record(peer_dept_id, grant))
        # In-process: co-located, always reachable. If verification denied EVERY grant, the
        # peer was reached but refused (grant_rejected) — kept out of caller-facing coverage,
        # never reported as an availability gap. Otherwise the historical "ok".
        if denied and not used:
            return PeerFetch(per_source, used, reachable=True, reason="grant_rejected")
        return PeerFetch(per_source, used)


class MeshTransport:
    """Route the query to the local sovereign-node, which dials the remote peer over libp2p.

    The node's outbound control endpoint (``POST {node_url}/federate``) is implemented in
    sovereign-node (``node/control.go`` + ``protocols/query_client.go``): it dials the peer
    over libp2p, ships the QUERY (not a vector, ADR-0019), and returns spans. Unreachable peer
    or an invalid grant → no spans (fail-closed on authorisation, fail-open on availability)."""

    def __init__(self, node_url: str, *, client: httpx.AsyncClient | None = None, timeout: float = 30.0) -> None:
        self._node_url = node_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    async def fetch(
        self, *, caller_dept: str, peer_dept_id: str, grants: list[Grant], query: str, top_k: int
    ) -> PeerFetch:
        per_source: list[SourceSpans] = []
        used: list[dict[str, Any]] = []
        statuses: list[str] = []
        for grant in grants:
            cap = min(top_k, grant.max_results_per_query)
            raw, status = await self._dial(peer_dept_id, grant, query, cap)
            statuses.append(status)
            spans = _topic_scoped(raw, grant, enabled=get_settings().federation_topic_enforcement)
            per_source.append(spans)
            used.append(_grant_record(peer_dept_id, grant, transport="mesh"))
        # Reachable if the node answered at all (ok OR a grant rejection); unreachable only
        # when EVERY dial was an availability failure — that is what is reported as missing.
        reached = any(st != "dial_failed" for st in statuses)
        reason = "ok" if "ok" in statuses else ("grant_rejected" if "grant_rejected" in statuses else "unreachable")
        return PeerFetch(per_source, used, reachable=reached, reason=reason)

    async def _dial(self, peer_dept_id: str, grant: Grant, query: str, top_k: int) -> tuple[SourceSpans, str]:
        payload: dict[str, Any] = {
            "peer_dept_id": peer_dept_id,
            "query_text": query,
            "top_k": top_k,
            "grant": grant.model_dump(mode="json"),
        }
        # Propagate the active trace so the peer's retrieval joins the same distributed trace.
        traceparent = current_traceparent()
        if traceparent:
            payload["traceparent"] = traceparent
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        owns = self._client is None
        try:
            resp = await client.post(f"{self._node_url}/federate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            _log.warning("federation_mesh_dial_failed", peer=peer_dept_id, error=str(exc))
            return [], "dial_failed"
        finally:
            if owns:
                await client.aclose()
        if not data.get("grant_valid", False):
            _log.warning("federation_mesh_grant_rejected", peer=peer_dept_id, node_error=data.get("error"))
            return [], "grant_rejected"
        # Go marshals a nil slice as JSON null (not []) on its Spans:nil paths (e.g.
        # grant_valid:true but the responder's own local retrieval failed) -- .get(key, [])
        # only falls back for a MISSING key, not an explicit null, so guard it here too.
        return [_span_from_wire(sp, peer_dept_id) for sp in (data.get("spans") or [])], "ok"


def _span_from_wire(sp: dict[str, Any], peer_dept_id: str) -> RetrievedSpan:
    """Map a libp2p ``FederatedSpan`` (sovereign-node/protocols/wire.go) to a RetrievedSpan.

    The remote score is a reranked, absolute relevance — clamp to [0,1] and name the scorer
    so the grounded-or-abstain gate reads it on a known scale. score 0 stays unscored (the
    validator forbids a score with no scorer)."""
    score = max(0.0, min(1.0, float(sp.get("score", 0.0) or 0.0)))
    return RetrievedSpan(
        chunk_id=f"{peer_dept_id}::{sp.get('chunk_id', '')}",
        doc_id=f"{peer_dept_id}/{sp.get('doc_id', '')}",
        text=str(sp.get("text", "")),
        score=score,
        scorer="cross_encoder" if score > 0.0 else "unscored",
        acl=[],
        category=sp.get("category"),
    )


class NullTransport:
    """Fail-closed transport: contributes no spans. Used when a peer is explicitly configured
    for the mesh but its node is unreachable/misconfigured — returning nothing is safer than a
    silent in-process fallback that could surface a locally-colliding KB in the peer's place."""

    async def fetch(
        self, *, caller_dept: str, peer_dept_id: str, grants: list[Grant], query: str, top_k: int
    ) -> PeerFetch:
        # A mesh peer we cannot route to is UNREACHABLE (evidence missing), reported as such
        # rather than silently empty (T1 partial-failure honesty).
        return PeerFetch([], [], reachable=False, reason="no_route")


class FederationRegistry:
    """Resolve which transport reaches a given peer, from settings.

    A global default (``RAG_FEDERATION_TRANSPORT``) plus optional per-peer overrides
    (``RAG_FEDERATION_PEERS`` JSON: ``{"<dept>": {"transport": "mesh", "node_url": "..."}}``).
    Default ``local`` everywhere keeps the monolith behaviour."""

    def __init__(
        self, *, default_transport: str, node_url: str, peers: dict[str, dict[str, str]], timeout: float = 8.0
    ) -> None:
        self._default = default_transport
        self._node_url = node_url
        self._peers = peers
        self._timeout = timeout

    @classmethod
    def from_settings(cls) -> FederationRegistry:
        s = get_settings()
        peers: dict[str, dict[str, str]] = {}
        if s.federation_peers.strip():
            try:
                loaded = json.loads(s.federation_peers)
                if isinstance(loaded, dict):
                    # Coerce keys AND values to str: a JSON number for node_url would otherwise
                    # reach MeshTransport and crash on .rstrip (operator-misconfig availability bug).
                    peers = {str(k): {str(kk): str(vv) for kk, vv in dict(v).items()} for k, v in loaded.items()}
            except (ValueError, TypeError) as exc:
                _log.warning("federation_peers_parse_failed", error=str(exc))
        return cls(
            default_transport=s.federation_transport,
            node_url=s.federation_node_url,
            peers=peers,
            timeout=s.federation_peer_timeout_seconds,
        )

    def transport_for(self, peer_dept_id: str) -> FederationTransport:
        entry = self._peers.get(peer_dept_id, {})
        kind = entry.get("transport", self._default)
        if kind == "mesh":
            node_url = entry.get("node_url") or self._node_url
            if not node_url:
                # Explicitly a mesh peer but no node URL to reach it: fail CLOSED (no spans)
                # rather than silently degrade to an in-process lookup, which under a cross-
                # deployment kb_id collision could surface local content in the peer place.
                _log.warning("federation_mesh_no_node_url", peer=peer_dept_id)
                return NullTransport()
            return MeshTransport(node_url, timeout=self._timeout)
        # Fail-closed by default on the co-located path too: wire the grant store so the
        # transport re-validates signature/expiry/chain at use time (mesh-parity). The store
        # is resolved lazily to avoid an import cycle (wiring imports federation).
        return LocalTransport(grant_store=wiring.grant_store(), verify=True)
