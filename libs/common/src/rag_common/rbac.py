"""Hierarchical RBAC — the access spine (pure, I/O-free, exhaustively testable).

The one rule that must never break in production: **a lower level must not read higher-level
data.** This module is the single place that decides it, so every caller (API guard, KB list,
retrieval, assignment) enforces the SAME logic. Three orthogonal axes — deliberately not
conflated, because conflating them is exactly how enterprise access leaks:

* **Tier** — the management hierarchy: VIEWER < MEMBER < UNIT_ADMIN < TENANT_ADMIN < SUPER_ADMIN.
  Governs who may administer whom. A grantor may only assign a tier STRICTLY below its own
  (no privilege escalation).
* **Clearance** — the data-sensitivity ceiling, and it IS the classification ladder
  (PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED, ints 0..3 — identical to the values
  :func:`rag_core.access.is_cleared` checks at span level). A subject reads a resource only
  if ``clearance >= classification``.
* **Scope** — need-to-know over a NESTED org-unit tree. Data at your unit or an ANCESTOR flows
  DOWN to you (company-wide policy is visible to a branch teller); a UNIT_ADMIN additionally
  governs its own SUBTREE (downward). Siblings are never visible without an explicit grant.

Invariants this module guarantees:
  1. No upward read      — ``clearance >= classification`` on every read path.
  2. No escalation       — you may grant only a tier below, and a clearance at/below, your own.
  3. Default-deny        — a fresh subject (VIEWER/PUBLIC/root) sees only root-unit public data.
  4. Provider SoD        — SUPER_ADMIN operates the fleet and NEVER reads tenant data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    """Management hierarchy. Higher = more authority. Ordering is load-bearing (comparisons)."""

    VIEWER = 0  # read/ask within scope
    MEMBER = 1  # + ingest within scope
    UNIT_ADMIN = 2  # administers one org-unit subtree
    TENANT_ADMIN = 3  # administers the whole tenant (org owner)
    SUPER_ADMIN = 4  # platform/Acme operator — fleet lifecycle only, NO tenant data (SoD)


class Clearance(IntEnum):
    """Data-sensitivity ceiling == the classification ladder (must match the classification ints)."""

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


# The smallest tier that may administer anything (grant roles/clearance, manage a unit).
_ADMIN_FLOOR = Tier.UNIT_ADMIN


@dataclass(frozen=True, slots=True)
class Subject:
    """The acting principal, reduced to what access decisions need."""

    tier: Tier
    clearance: Clearance
    tenant_id: str
    unit_id: str  # the org-unit the subject belongs to (the tenant ROOT unit for top-level)
    attrs: Mapping[str, str] = field(default_factory=dict)


# --- Org-unit tree math (parents: unit_id -> parent_id, root's parent is None) --------


def ancestors(parents: Mapping[str, str | None], unit_id: str) -> list[str]:
    """The chain from ``unit_id``'s parent up to the root (exclusive of ``unit_id`` itself).
    Cycle-guarded — a malformed tree yields a finite list, never an infinite loop."""
    chain: list[str] = []
    seen: set[str] = {unit_id}
    cur = parents.get(unit_id)
    while cur is not None and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = parents.get(cur)
    return chain


def is_ancestor_or_self(parents: Mapping[str, str | None], ancestor: str, node: str) -> bool:
    """True if ``ancestor`` is ``node`` itself or lies above it in the tree."""
    return ancestor == node or ancestor in ancestors(parents, node)


def in_subtree(parents: Mapping[str, str | None], root: str, node: str) -> bool:
    """True if ``node`` is ``root`` or a descendant of it (``root`` governs ``node``)."""
    return is_ancestor_or_self(parents, root, node)


# --- No-escalation: who may grant what -------------------------------------------------


def can_grant_role(grantor: Tier, target: Tier) -> bool:
    """A grantor may assign a tier STRICTLY below its own, and only if it is an admin at all.
    So no one ever mints a peer or a superior — the core anti-escalation rule."""
    return grantor >= _ADMIN_FLOOR and grantor > target


def can_grant_clearance(grantor_tier: Tier, grantor_clearance: Clearance, target: Clearance) -> bool:
    """A grantor may assign a clearance AT OR BELOW its own — never above. An admin cannot hand
    out access it does not itself hold."""
    return grantor_tier >= _ADMIN_FLOOR and target <= grantor_clearance


# --- The read + admin decisions (the spine) -------------------------------------------


def can_read_kb(
    subject: Subject, *, kb_classification: Clearance, kb_unit_id: str, parents: Mapping[str, str | None]
) -> bool:
    """May ``subject`` read a KB of ``kb_classification`` living in ``kb_unit_id``?

    Order matters — the ceiling is checked FIRST and independently, so no scope rule can ever
    override it (defence in depth against a future scope bug leaking a Restricted KB)."""
    if subject.tier is Tier.SUPER_ADMIN:
        return False  # provider Separation-of-Duties: fleet ops never read tenant content
    if subject.clearance < kb_classification:
        return False  # the ceiling — a lower level never reads higher-classified data
    if subject.tier is Tier.TENANT_ADMIN:
        return True  # governs the whole tenant, within its clearance
    # Members/viewers: data at their unit or an ANCESTOR flows down (inherited shared context).
    if is_ancestor_or_self(parents, kb_unit_id, subject.unit_id):
        return True
    # A unit admin additionally reads its own SUBTREE (downward management scope).
    return subject.tier is Tier.UNIT_ADMIN and in_subtree(parents, subject.unit_id, kb_unit_id)


def can_ingest_at(
    subject: Subject,
    *,
    kb_floor: Clearance,
    requested: Clearance,
    kb_unit_id: str,
    parents: Mapping[str, str | None],
) -> bool:
    """The INGESTION-LEVEL rule: may ``subject`` ingest a document classified ``requested`` into a
    KB whose classification floor is ``kb_floor``?

    Two independent bounds, both required:
    * write access to the KB (:func:`can_ingest_kb` — MEMBER+, in scope, cleared for the floor), and
    * ``kb_floor <= requested <= subject.clearance`` — you can neither label a document BELOW the
      KB's floor (smuggling low-trust content into a sensitive collection) nor create data MORE
      secret than you are yourself cleared to read (a low user must not plant Restricted content it
      cannot see — an integrity / exfiltration guard).

    The band is non-empty whenever write access holds (that already implies clearance >= floor), so
    a legitimate ingestor always has at least one valid level: its own clearance."""
    if not can_ingest_kb(subject, kb_classification=kb_floor, kb_unit_id=kb_unit_id, parents=parents):
        return False
    return kb_floor <= requested <= subject.clearance


def can_admin_unit(subject: Subject, *, unit_id: str, parents: Mapping[str, str | None]) -> bool:
    """May ``subject`` administer ``unit_id`` (its users, KBs, sub-units)?"""
    if subject.tier is Tier.TENANT_ADMIN:
        return True
    if subject.tier is Tier.UNIT_ADMIN:
        return in_subtree(parents, subject.unit_id, unit_id)
    return False  # SUPER_ADMIN manages TENANTS (not units), members/viewers manage nothing


def can_ingest_kb(
    subject: Subject, *, kb_classification: Clearance, kb_unit_id: str, parents: Mapping[str, str | None]
) -> bool:
    """Ingest needs write scope: at least MEMBER, plus the same clearance + scope as a read. A
    viewer never writes; a member writes only where it can read."""
    if subject.tier < Tier.MEMBER:
        return False
    return can_read_kb(subject, kb_classification=kb_classification, kb_unit_id=kb_unit_id, parents=parents)
