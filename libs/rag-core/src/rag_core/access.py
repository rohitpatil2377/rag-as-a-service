"""Permission-aware access control for retrieval (FR-A6/D8).

Source ACLs are captured at ingest (``DocumentEnvelope.source_acl``) and stamped onto
every chunk (``Chunk.acl``); this is the single place that decides, at query time, whether
a caller may see a given span. Kept as a pure function so the rule is trivially testable
and identical across every retrieval path (API, MCP, agents).
"""

from __future__ import annotations

from collections.abc import Iterable


def is_entitled(span_acl: Iterable[str], principal_acl: Iterable[str]) -> bool:
    """True if a caller holding ``principal_acl`` groups may see a span with ``span_acl``.

    Rule: a span with an EMPTY acl is tenant-public (visible to any caller in the tenant);
    otherwise the caller must share at least one group. A caller with no groups therefore
    sees only tenant-public spans (least privilege) — never someone else's restricted docs.
    """
    span_groups = set(span_acl)
    if not span_groups:
        return True
    return not span_groups.isdisjoint(principal_acl)


def is_cleared(span_classification: int, clearance: int) -> bool:
    """True if a caller with ``clearance`` may see a span of ``span_classification`` (banking pack).

    Hierarchical: a span is visible only if its level is at or below the caller's clearance, so a
    Confidential-cleared caller (2) sees Public/Internal/Confidential but never Restricted (3).
    Default clearance 0 (Public) with default classification 0 (Public) = unchanged behaviour.
    Complements :func:`is_entitled`: a span must pass BOTH the level check and the ACL check.
    """
    return span_classification <= clearance
