"""Secrets access behind a Protocol.

Phase-0 uses an env-backed provider. Phase-1 swaps in a Vault/HSM adapter with NO
call-site changes — code depends on :class:`SecretsProvider`, never on ``os.environ``
directly for sensitive values. This keeps the 12-factor "no secrets in code" rule
enforceable and the sovereign HSM/KMS story (TDD §9) a drop-in.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from rag_common.config import get_settings
from rag_common.errors import ConfigError


@runtime_checkable
class SecretsProvider(Protocol):
    """Resolve named secrets. Implementations must never log secret values."""

    def get(self, name: str, *, default: str | None = None) -> str: ...


class EnvSecretsProvider:
    """Env-backed provider for local/dev. Reads ``RAG_``-prefixed variables."""

    _prefix = "RAG_"

    def get(self, name: str, *, default: str | None = None) -> str:
        key = name if name.startswith(self._prefix) else f"{self._prefix}{name.upper()}"
        val = os.environ.get(key, default)
        if val is None:
            raise ConfigError(f"missing secret: {key}")
        return val


class VaultSecretsProvider:
    """Phase-1 stub: HashiCorp Vault / HSM-backed provider.

    TODO(Phase-1): implement AppRole/Kubernetes auth + KV v2 reads with caching and
    lease renewal; back log-signing keys with the HSM (TDD §9). Interface is frozen.
    """

    def get(self, name: str, *, default: str | None = None) -> str:  # pragma: no cover
        raise NotImplementedError("VaultSecretsProvider is a Phase-1 stub")


def get_secrets_provider() -> SecretsProvider:
    """Factory selecting the provider from configuration."""
    provider = get_settings().secrets_provider
    if provider == "env":
        return EnvSecretsProvider()
    if provider == "vault":
        return VaultSecretsProvider()
    raise ConfigError(f"unknown secrets provider: {provider}")
