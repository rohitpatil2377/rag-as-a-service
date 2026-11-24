"""Embeddable chat widget (FR-I6) — per-KB theming + a short-lived session token.

Security model (sovereign, key-never-in-browser):
* the tenant enables the widget on a KB and sets its white-label theming (stored in
  ``tenant_config``);
* the host page's SERVER holds a read-scoped API key (FR-I4) and calls ``POST /v1/widget/session``
  with it, naming the KB. The server never exposes the key to the browser;
* it gets back a SHORT-LIVED JWT bound to exactly that KB (``tenant_id == kb_id`` so
  :func:`tenant_guard` allows only that KB), ``reader`` only, and PUBLIC-ONLY ACL (empty
  groups → no ACL-restricted content ever reaches an anonymous embed);
* the browser widget uses that token to call the normal query API until it expires.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_clients.factory import get_catalog

_WIDGET_KEY = "widget_config"


class WidgetConfig(BaseModel):
    """Per-KB white-label theming for the embeddable widget."""

    enabled: bool = False
    title: str = Field(default="Ask us anything", max_length=80)
    welcome: str = Field(default="Hi! Ask a question about our documents.", max_length=280)
    accent: str = Field(default="#1a3d7c", pattern=r"^#[0-9a-fA-F]{6}$", description="Brand accent colour.")
    placeholder: str = Field(default="Type your question…", max_length=80)
    suggestions: list[str] = Field(default_factory=list, max_length=4, description="Starter questions.")


async def get_widget_config(kb_id: str) -> WidgetConfig:
    """The KB's widget config (defaults, disabled, if never set)."""
    raw = (await get_catalog().get_tenant_config(kb_id)).get(_WIDGET_KEY)
    if not raw:
        return WidgetConfig()
    try:
        return WidgetConfig.model_validate_json(raw)
    except ValueError:
        return WidgetConfig()


async def set_widget_config(kb_id: str, config: WidgetConfig) -> WidgetConfig:
    """Persist the KB's widget config."""
    await get_catalog().set_tenant_config(kb_id, _WIDGET_KEY, config.model_dump_json())
    return config


def owns_kb(tenant_id: str, kb_id: str) -> bool:
    """Same ownership rule as ``tenant_guard`` — a tenant owns its default KB and any KB it
    created (``{tenant_id}__…``). Used by the session endpoint (no path-based guard there)."""
    return kb_id == tenant_id or kb_id.startswith(f"{tenant_id}__")
