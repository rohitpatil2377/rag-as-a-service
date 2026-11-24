"""Runtime feature flags — admin-toggleable capabilities without a restart.

Config is 12-factor (env → ``get_settings()``), which is clean for operators but rigid for
admins. This adds a thin override layer: an admin flips a BEHAVIOURAL capability in the console,
the override is persisted in one small global table, applied to the process environment, and the
cached settings are rebuilt — so every existing ``get_settings().X_enabled`` call site picks it
up live, no restart.

Two tiers, deliberately:
* **TOGGLEABLE** — safe behavioural/quality features an admin may switch at runtime.
* **LOCKED** — security & infrastructure (sovereignty guard, PII masking, encryption/BYOK,
  model/store backends, OIDC/MFA, Redis scale). These are NEVER web-toggleable — they are
  security boundaries or heavy dependencies, changed only by the operator via environment. The
  console shows them read-only so an auditor can see the posture without being able to weaken it.

Platform-wide (single override set per deployment). Multi-replica propagation would ride the
coordination version counter — a follow-up; single-process is correct today.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_common import get_logger, get_settings
from rag_common.errors import ValidationError

_log = get_logger(__name__)
_IDENTITY_SCHEMA = "rag_identity"

# name -> (env var, on-value, off-value, human label)
TOGGLEABLE: dict[str, tuple[str, str, str, str]] = {
    "guardrails": ("RAG_GUARDRAILS_ENABLED", "true", "false", "Guardrails (injection · toxicity)"),
    "faithfulness": ("RAG_FAITHFULNESS_ENABLED", "true", "false", "Faithfulness verification"),
    "semantic_cache": ("RAG_SEMANTIC_CACHE_ENABLED", "true", "false", "Semantic cache"),
    "agentic_retry": ("RAG_AGENTIC_RETRY_ENABLED", "true", "false", "Agentic retry"),
    "contextual_retrieval": ("RAG_CONTEXTUAL_RETRIEVAL_ENABLED", "true", "false", "Contextual retrieval"),
    "adaptive_router": ("RAG_ADAPTIVE_ROUTER_ENABLED", "true", "false", "Adaptive router"),
    "query_transform": ("RAG_QUERY_TRANSFORM_ENABLED", "true", "false", "Query transform"),
    "followups": ("RAG_FOLLOWUPS_ENABLED", "true", "false", "Suggested follow-ups"),
    "auto_metadata": ("RAG_AUTO_METADATA_ENABLED", "true", "false", "Auto-metadata tagging"),
    "field_extraction": ("RAG_FIELD_EXTRACTION_ENABLED", "true", "false", "Field extraction"),
    "language_detection": ("RAG_LANGUAGE_DETECTION_ENABLED", "true", "false", "Language detection"),
    "translation": ("RAG_TRANSLATION_BACKEND", "llm", "noop", "Translation (Indic)"),
    "ocr": ("RAG_OCR_BACKEND", "tesseract", "noop", "OCR (scanned docs)"),
    "nl2sql": ("RAG_NL2SQL_ENABLED", "true", "false", "Governed NL→SQL"),
    "agentic_workflows": ("RAG_AGENTIC_WORKFLOWS_ENABLED", "true", "false", "Agentic workflows"),
    "graphrag": ("RAG_GRAPHRAG_ENABLED", "true", "false", "GraphRAG"),
}

# Operator-managed, shown read-only. name -> (settings attr, label). A truthy/string-match value
# means "on". These are security/infra boundaries and are intentionally NOT web-toggleable.
LOCKED: dict[str, tuple[str, str]] = {
    "sovereign_enforce": ("sovereign_enforce", "Sovereignty egress guard"),
    "pii_masking": ("__always_on__", "PII masking"),
    "oidc_sso": ("auth_mode", "OIDC / SSO"),
    "mfa_required": ("oidc_require_mfa", "MFA required"),
    "multi_replica_scale": ("redis_url", "Multi-replica scale (Redis)"),
    "byok_kms": ("kms_backend", "BYOK / external KMS"),
}


class FlagState(BaseModel):
    name: str
    label: str
    enabled: bool
    locked: bool


class FeatureFlagsView(BaseModel):
    flags: list[FlagState]


def effective(name: str) -> bool:
    """Current effective on/off for a toggleable flag (reads live settings).

    A capability is ON when it is not switched OFF — deliberately not "equal to the enabled value
    named here". Two of these flags select a model BACKEND, and the operator picks which one: with
    an equality test, a deployment running OCR on Chandra read as OFF because this table happens
    to name tesseract. An operator seeing that either switches on something that was never off,
    or believes scans are being skipped when they are not."""
    env_key, _on, off_val, _label = TOGGLEABLE[name]
    attr = env_key[len("RAG_") :].lower()
    val = getattr(get_settings(), attr, None)
    return val if isinstance(val, bool) else str(val) != off_val


def _locked_enabled(attr: str) -> bool:
    if attr == "__always_on__":
        return True
    s = get_settings()
    if attr == "auth_mode":
        return s.auth_mode == "oidc"
    if attr == "kms_backend":
        return s.kms_backend != "local"
    if attr == "redis_url":
        return bool(s.redis_url)
    return bool(getattr(s, attr, False))


def _invalidate() -> None:
    """Rebuild settings + any singleton that baked a toggled flag at construction (OCR / fields)."""
    from svc_control_plane import wiring  # noqa: PLC0415 — avoid import cycle at module load

    get_settings.cache_clear()
    wiring.doc_intel_service.cache_clear()


# What the OPERATOR configured, captured before any override is applied. Enabling a capability
# must put back THEIR value, not a literal chosen here: two of these flags select a model BACKEND
# (ocr, translation), and this module's own contract is that backends belong to the environment.
# Live consequence of the literal: a deployment running self-hosted Chandra on AIOps had its OCR
# repointed to "tesseract" — not installed in the image — by an admin toggling OCR in the console,
# and the Models page then named www.datalab.to, a foreign host the deployment had removed.
_BASELINE: dict[str, str | None] = {}


def capture_baseline() -> None:
    """Record the environment's own value for every toggleable key. Called once at startup,
    BEFORE persisted overrides are applied — after that the environment is no longer the
    operator's statement of intent, it is the last thing the console wrote."""
    _BASELINE.clear()
    _BASELINE.update({meta[0]: os.environ.get(meta[0]) for meta in TOGGLEABLE.values()})


def _apply_env(name: str, on: bool) -> None:
    env_key, on_val, off_val, _label = TOGGLEABLE[name]
    if not on:
        os.environ[env_key] = off_val
        return
    # The baseline is honoured only when it is a real choice. A boolean capability the operator
    # shipped OFF must still turn on when an admin enables it — "enable guardrails" means enable
    # them, and reading the off value back as intent would make the toggle do nothing.
    baseline = _BASELINE.get(env_key)
    os.environ[env_key] = baseline if baseline and baseline != off_val else on_val


def snapshot() -> FeatureFlagsView:
    """The full capability posture for the console: toggleable (live) + locked (read-only)."""
    flags = [FlagState(name=n, label=meta[3], enabled=effective(n), locked=False) for n, meta in TOGGLEABLE.items()]
    flags += [
        FlagState(name=n, label=lbl, enabled=_locked_enabled(attr), locked=True) for n, (attr, lbl) in LOCKED.items()
    ]
    return FeatureFlagsView(flags=flags)


# --- persistence (one small global table; in-memory twin for CI) -------------
@runtime_checkable
class FlagStore(Protocol):
    async def ensure(self) -> None: ...
    async def get_all(self) -> dict[str, bool]: ...
    async def put(self, name: str, enabled: bool) -> None: ...


class InMemoryFlagStore:
    def __init__(self) -> None:
        self._rows: dict[str, bool] = {}

    async def ensure(self) -> None:
        return None

    async def get_all(self) -> dict[str, bool]:
        return dict(self._rows)

    async def put(self, name: str, enabled: bool) -> None:
        self._rows[name] = enabled


_DDL = (
    f'CREATE SCHEMA IF NOT EXISTS "{_IDENTITY_SCHEMA}"',
    f'CREATE TABLE IF NOT EXISTS "{_IDENTITY_SCHEMA}".platform_flags (name TEXT PRIMARY KEY, enabled BOOLEAN NOT NULL)',
)


class PostgresFlagStore:
    def __init__(self, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)

    async def ensure(self) -> None:
        async with self._engine.begin() as c:
            for stmt in _DDL:
                await c.execute(text(stmt))

    async def get_all(self) -> dict[str, bool]:
        await self.ensure()
        async with self._engine.connect() as c:
            rows = (await c.execute(text(f'SELECT name, enabled FROM "{_IDENTITY_SCHEMA}".platform_flags'))).all()
        return {r[0]: bool(r[1]) for r in rows}

    async def put(self, name: str, enabled: bool) -> None:
        await self.ensure()
        sql = (
            f'INSERT INTO "{_IDENTITY_SCHEMA}".platform_flags (name, enabled) VALUES (:n, :e) '
            "ON CONFLICT (name) DO UPDATE SET enabled = :e"
        )
        async with self._engine.begin() as c:
            await c.execute(text(sql), {"n": name, "e": enabled})


async def set_flag(name: str, enabled: bool, store: FlagStore) -> FeatureFlagsView:
    """Toggle a behavioural flag: persist, apply to the environment, rebuild settings."""
    if name not in TOGGLEABLE:
        raise ValidationError(
            "flag is not runtime-toggleable", details={"flag": name, "reason": "operator-managed or unknown"}
        )
    await store.put(name, enabled)
    _apply_env(name, enabled)
    _invalidate()
    _log.info("feature_flag_set", flag=name, enabled=enabled)
    return snapshot()


async def load_overrides(store: FlagStore) -> None:
    """Apply persisted overrides onto the environment at startup (before serving)."""
    capture_baseline()  # the env is the operator's intent only until the first override lands
    overrides = await store.get_all()
    if not overrides:
        return
    for name, enabled in overrides.items():
        if name in TOGGLEABLE:
            _apply_env(name, enabled)
    _invalidate()
    _log.info("feature_flags_loaded", count=len(overrides))
