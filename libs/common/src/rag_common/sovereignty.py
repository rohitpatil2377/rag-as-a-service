"""Sovereignty egress guard.

Hard constraint: no calls to foreign/managed AI or cloud endpoints. Every outbound
client (embeddings, LLM gateway, object store) validates its target host against the
configured allowlist before connecting. This is defence-in-depth alongside network
egress controls (TDD §9) — the application refuses to be pointed at a non-sovereign
endpoint even if the network would permit it.
"""

from __future__ import annotations

from urllib.parse import urlparse

from rag_common.config import get_settings
from rag_common.errors import SovereigntyViolationError


def assert_sovereign(url: str) -> str:
    """Raise if ``url``'s host is not on the egress allowlist. Returns the url."""
    settings = get_settings()
    if not settings.sovereign_enforce:
        return url
    host = urlparse(url).hostname or ""
    if host not in settings.egress_allowlist:
        raise SovereigntyViolationError(
            "outbound call blocked by sovereignty guard",
            details={"host": host, "allowlist": sorted(settings.egress_allowlist)},
        )
    return url
