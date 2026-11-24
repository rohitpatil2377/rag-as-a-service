"""Bridge the request Principal to the rbac access spine — Subject (tier x clearance x scope).

Sub-step B of RBAC wiring: the ONE place a JWT/Principal becomes an access :class:`Subject`, so
every guard (KB-list discovery, access, ingest) derives tier / clearance / unit identically. Keep
this the sole mapping — divergent per-endpoint interpretations of a token are how access leaks.
"""

from __future__ import annotations

from rag_common import get_settings
from rag_common.auth import Principal
from rag_common.rbac import Clearance, Subject, Tier
from rag_schemas.classification import parse_classification

_CLEARANCE_MAX = 3


def is_platform_admin(email: str) -> bool:
    """True if ``email`` is on the platform-operator allowlist (``RAG_PLATFORM_ADMIN_EMAILS``).
    The fleet operator (SUPER_ADMIN) cannot be authorised by any single tenant's RBAC, so it is
    bootstrapped out-of-band by this allowlist."""
    allow = {e.strip().lower() for e in get_settings().platform_admin_emails.split(",") if e.strip()}
    return bool(allow) and email.strip().lower() in allow


def _tier(p: Principal) -> Tier:
    if is_platform_admin(p.subject):
        return Tier.SUPER_ADMIN
    if p.has_role("owner") or p.has_role("admin"):
        # An admin scoped to a specific (non-root) unit governs only that subtree — UNIT_ADMIN.
        unit = p.attrs.get("unit")
        return Tier.UNIT_ADMIN if unit and str(unit) != p.tenant_id else Tier.TENANT_ADMIN
    if p.has_role("ingestor"):
        return Tier.MEMBER  # can write within scope
    return Tier.VIEWER


def _clearance(p: Principal) -> Clearance:
    # parse_classification accepts BOTH the level names ("confidential") and the ints, defaulting to
    # 0 (Public) for anything absent/unparsable — least privilege. The SAME parser the span-level
    # clearance check uses, so the ceiling is identical at every layer.
    return Clearance(max(0, min(_CLEARANCE_MAX, parse_classification(p.attrs.get("clearance")))))


def subject_of(p: Principal) -> Subject:
    """Principal -> access Subject. ``unit`` defaults to the tenant ROOT (tenant-wide) when the
    token carries no unit claim, so a pre-units token behaves exactly as before (back-compatible),
    and ``clearance`` defaults to Public (least privilege) when absent."""
    return Subject(
        tier=_tier(p),
        clearance=_clearance(p),
        tenant_id=p.tenant_id,
        unit_id=str(p.attrs.get("unit") or p.tenant_id),
    )
