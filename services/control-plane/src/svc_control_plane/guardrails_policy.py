"""Hierarchical guardrail policy (top-down RBAC lock model) — first cut.

A single governance surface over the platform's existing safety controls. Each control is
set at one of three tiers and locked DOWNWARD: a higher tier may SET a value and LOCK it
(every tier below then sees it enforced, read-only), or set it UNLOCKED as a default the
lower tier may override within its own scope. Resolving the effective policy for a KB walks
PLATFORM -> ORG -> KB and stops at the first LOCKED value.

Storage reuses the ``tenant_config`` JSON-string pattern (no migration), exactly like
``kb_roles`` / persona / taxonomy:
  * PLATFORM policy -> the shared KB's tenant_config (always provisioned, fleet-wide).
  * ORG policy      -> the org tenant's tenant_config (``tenant_id``).
  * KB policy       -> the KB's own tenant_config (``kb_id``).

This module is pure model + tenant_config load/save + the resolver + the write-guard. It
does NOT itself enforce guardrails at the pipeline — enforcement call sites read the
resolved value via :func:`resolve_effective`. The first cut catalogs the live controls; new
checks are additive entries.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from rag_core.interfaces import CatalogAdapter

# tenant_config key (JSON string value, no migration — same pattern as kb_roles/persona).
POLICY_KEY = "guardrail_policy"
# Where the fleet-wide platform policy is stored (the shared KB is always provisioned).
PLATFORM_SCOPE = "public__company-wide"

Tier = Literal["platform", "org", "kb"]
Kind = Literal["toggle", "threshold"]

_TIER_RANK = {"platform": 0, "org": 1, "kb": 2}


class ControlDef(BaseModel):
    """A guardrail control's definition (static catalog entry)."""

    id: str
    label: str
    category: str
    kind: Kind = "toggle"
    default_value: bool | float = True
    # constitutional controls are locked ON at the platform tier and can never be opened.
    constitutional: bool = False
    # the tier that typically owns (sets/locks) this control — advisory, shown in the UI.
    default_owner: Tier = "org"
    help: str = ""


# --- The catalog (first cut: the LIVE controls + the constitutional floor) ---------------
CATALOG: list[ControlDef] = [
    # A · input
    ControlDef(
        id="prompt_injection",
        label="Prompt-injection block",
        category="Input",
        constitutional=True,
        default_owner="platform",
        help="Refuse jailbreak / 'ignore your instructions' attempts.",
    ),
    ControlDef(
        id="rate_budget_caps",
        label="Rate & budget caps",
        category="Input",
        default_owner="org",
        help="Per-org query rate limit and token budget ceiling.",
    ),
    # B · grounding
    ControlDef(
        id="cite_or_abstain",
        label="Cite-or-abstain",
        category="Grounding",
        constitutional=True,
        default_owner="platform",
        help="Every answer cites its source or the system refuses. The core promise.",
    ),
    ControlDef(
        id="faithfulness_gate",
        label="Faithfulness gate",
        category="Grounding",
        default_owner="org",
        help="NLI/LLM grounding check on each answer.",
    ),
    ControlDef(
        id="faithful_threshold",
        label="Faithfulness threshold",
        category="Grounding",
        kind="threshold",
        default_value=0.7,
        default_owner="org",
        help="Score at/above which an answer is published (publish >= this).",
    ),
    ControlDef(
        id="clearance_enforcement",
        label="Clearance enforcement (MLS)",
        category="Grounding",
        constitutional=True,
        default_owner="platform",
        help="A user retrieves a doc only when clearance >= classification.",
    ),
    ControlDef(
        id="human_review",
        label="Human-in-the-loop review",
        category="Grounding",
        default_owner="org",
        help="Low-confidence answers held for a reviewer.",
    ),
    # C · output
    ControlDef(
        id="toxicity_filter",
        label="Toxicity filter",
        category="Output",
        default_owner="org",
        help="Block toxic/unsafe generated output.",
    ),
    ControlDef(
        id="reasoning_strip",
        label="Reasoning-strip",
        category="Output",
        constitutional=True,
        default_owner="platform",
        help="Remove chain-of-thought so only the answer is published.",
    ),
    ControlDef(
        id="doc_instruction_defence",
        label="Document-instruction defence",
        category="Output",
        constitutional=True,
        default_owner="platform",
        help="Instructions embedded in a document are treated as text, never obeyed.",
    ),
    ControlDef(
        id="answer_language_lock",
        label="Answer-language lock",
        category="Output",
        default_owner="kb",
        help="Force the answer into the question's language.",
    ),
    # D · data & sovereignty
    ControlDef(
        id="egress_allowlist",
        label="Egress allowlist",
        category="Sovereignty",
        constitutional=True,
        default_owner="platform",
        help="Every model/connector call checked against an approved-host list.",
    ),
    ControlDef(
        id="no_foreign_model",
        label="No foreign model",
        category="Sovereignty",
        constitutional=True,
        default_owner="platform",
        help="A model endpoint off the allowlist cannot be attached.",
    ),
    ControlDef(
        id="pii_masking",
        label="PII masking at ingest",
        category="Sovereignty",
        default_owner="org",
        help="Aadhaar/PAN/phone/email masked before indexing.",
    ),
    ControlDef(
        id="pii_classification_floor",
        label="PII classification floor",
        category="Sovereignty",
        default_owner="org",
        help="Detected sensitive PII raises a doc's classification or blocks it.",
    ),
    ControlDef(
        id="retention_erasure",
        label="Retention & erasure",
        category="Sovereignty",
        default_owner="org",
        help="Per-KB TTL auto-purge + right-to-erasure with certificate.",
    ),
    # E · governance
    ControlDef(
        id="immutable_audit",
        label="Immutable audit",
        category="Governance",
        constitutional=True,
        default_owner="platform",
        help="Every action on a tamper-evident chain; erasure never removes audit rows.",
    ),
    ControlDef(
        id="dual_control",
        label="Dual control (maker-checker)",
        category="Governance",
        default_owner="org",
        help="A second admin approves publish / classified-ingest / purge / erasure.",
    ),
]

_CATALOG_BY_ID = {c.id: c for c in CATALOG}


class PolicyEntry(BaseModel):
    """One tier's setting for one control: a value plus whether it is locked downward."""

    value: bool | float
    locked: bool = False


class ScopePolicy(BaseModel):
    """A tier's policy: ``{control_id: PolicyEntry}`` (only the controls that tier has set)."""

    entries: dict[str, PolicyEntry] = Field(default_factory=dict)


class EffectiveEntry(BaseModel):
    """A resolved control: its effective value, who owns it, and whether the viewer may edit."""

    id: str
    label: str
    category: str
    kind: Kind
    value: bool | float
    owner: Tier | None  # the tier that set the resolved value (None = platform default from the catalog)
    locked_by: Tier | None  # the tier that locked it, if any (below-this-tier viewers cannot edit)
    constitutional: bool
    help: str


# --- tenant_config load / save (pure I/O over the CatalogAdapter seam) --------------------
def _scope_id(tier: Tier, *, org_id: str | None = None, kb_id: str | None = None) -> str:
    if tier == "platform":
        return PLATFORM_SCOPE
    if tier == "org":
        assert org_id is not None
        return org_id
    assert kb_id is not None
    return kb_id


async def load_policy(catalog: CatalogAdapter, scope_id: str) -> ScopePolicy:
    raw = (await catalog.get_tenant_config(scope_id)).get(POLICY_KEY)
    if not raw:
        return ScopePolicy()
    try:
        data = json.loads(raw)
        return ScopePolicy.model_validate(data)
    except (ValueError, TypeError):
        return ScopePolicy()


async def save_policy(catalog: CatalogAdapter, scope_id: str, policy: ScopePolicy) -> None:
    await catalog.set_tenant_config(scope_id, POLICY_KEY, policy.model_dump_json())


# --- the resolver ------------------------------------------------------------------------
async def resolve_effective(catalog: CatalogAdapter, kb_id: str) -> list[EffectiveEntry]:
    """Resolve every catalogued control for ``kb_id``, walking PLATFORM -> ORG -> KB and
    stopping at the first LOCKED value. A constitutional control is always ON and platform-locked.
    ``org_id`` is derived from the ``{org}__{slug}`` kb id (the platform's own convention)."""
    org_id = kb_id.split("__", 1)[0] if "__" in kb_id else kb_id
    plat = await load_policy(catalog, PLATFORM_SCOPE)
    org = await load_policy(catalog, org_id)
    kb = await load_policy(catalog, kb_id)
    layers: list[tuple[Tier, ScopePolicy]] = [("platform", plat), ("org", org), ("kb", kb)]

    out: list[EffectiveEntry] = []
    for c in CATALOG:
        value: bool | float = c.default_value
        owner: Tier | None = None
        locked_by: Tier | None = None
        if c.constitutional:
            # non-negotiable: always ON, platform-locked, regardless of stored policy.
            out.append(
                EffectiveEntry(
                    id=c.id,
                    label=c.label,
                    category=c.category,
                    kind=c.kind,
                    value=True,
                    owner="platform",
                    locked_by="platform",
                    constitutional=True,
                    help=c.help,
                )
            )
            continue
        for tier, pol in layers:
            entry = pol.entries.get(c.id)
            if entry is None:
                continue
            value, owner = entry.value, tier
            if entry.locked:
                locked_by = tier
                break  # a lock at this tier fixes the value for every tier below
        out.append(
            EffectiveEntry(
                id=c.id,
                label=c.label,
                category=c.category,
                kind=c.kind,
                value=value,
                owner=owner,
                locked_by=locked_by,
                constitutional=False,
                help=c.help,
            )
        )
    return out


async def effective_map(catalog: CatalogAdapter, kb_id: str) -> dict[str, bool | float]:
    """The resolved value of every control as a flat ``{control_id: value}`` map — the read side
    that enforcement sites consult. Cheap wrapper over :func:`resolve_effective`."""
    return {e.id: e.value for e in await resolve_effective(catalog, kb_id)}


class PolicyWriteError(Exception):
    """A tier attempted to change a control locked by a higher tier, or an unknown control."""


async def apply_policy_update(
    catalog: CatalogAdapter,
    *,
    tier: Tier,
    org_id: str | None,
    kb_id: str | None,
    updates: dict[str, PolicyEntry],
) -> ScopePolicy:
    """Write ``updates`` into ``tier``'s policy, refusing any control that a HIGHER tier has
    locked (or a constitutional control). Returns the saved scope policy.

    The higher-tier lock check resolves the layers above ``tier`` and rejects the write with
    :class:`PolicyWriteError` if any targeted control is locked above — this is the top-down
    enforcement: a lower tier can never breach a lock set above it."""
    scope_id = _scope_id(tier, org_id=org_id, kb_id=kb_id)
    # Build the set of controls locked by a strictly-higher tier.
    locked_above: dict[str, Tier] = {}
    if tier != "platform":
        plat = await load_policy(catalog, PLATFORM_SCOPE)
        for cid, e in plat.entries.items():
            if e.locked:
                locked_above[cid] = "platform"
    if tier == "kb" and org_id is not None:
        org = await load_policy(catalog, org_id)
        for cid, e in org.entries.items():
            if e.locked:
                locked_above.setdefault(cid, "org")

    for cid, _entry in updates.items():
        c = _CATALOG_BY_ID.get(cid)
        if c is None:
            raise PolicyWriteError(f"unknown guardrail control: {cid}")
        if c.constitutional:
            raise PolicyWriteError(f"'{c.label}' is a constitutional control and cannot be changed")
        if cid in locked_above and _TIER_RANK[locked_above[cid]] < _TIER_RANK[tier]:
            raise PolicyWriteError(f"'{c.label}' is locked by the {locked_above[cid]} tier")
        # A tier may only LOCK downward, never claim a lock it isn't entitled to bypass — the
        # write itself is allowed; the lock simply applies to tiers below this one.

    policy = await load_policy(catalog, scope_id)
    policy.entries.update(updates)
    await save_policy(catalog, scope_id, policy)
    return policy
