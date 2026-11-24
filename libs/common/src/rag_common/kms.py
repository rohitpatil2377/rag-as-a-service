"""External KMS / HSM key wrapper (opt-in, ``RAG_KMS_BACKEND=external``).

The production upgrade path for :class:`rag_common.crypto.KeyWrapper`: the Key Encryption
Key never leaves the KMS/HSM — the platform sends a tenant DEK to be wrapped and receives
the ciphertext, and sends the wrapped DEK back to be unwrapped only in the tenant's request
context. This is what makes crypto-shred real: revoke the KMS key and every wrapped DEK
becomes permanently un-unwrappable.

A concrete **HashiCorp Vault Transit** client is wired below (stdlib HTTP — no new
dependency). It activates ONLY when ``RAG_KMS_ENDPOINT`` is set; the endpoint is put through
``assert_sovereign`` (it is external, so it must be on the egress allowlist), and without an
endpoint the wrapper still fails closed rather than silently downgrading to the local master
key (which would defeat BYOK). Any other KMS (PKCS#11 HSM, a sovereign cloud KMS) slots in by
replacing the transport with the same wrap/unwrap contract.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from collections.abc import Callable
from typing import Any

from rag_common.config import get_settings
from rag_common.errors import ConfigError
from rag_common.sovereignty import assert_sovereign

# A transport takes (path, json_body) and returns the parsed JSON response. Injectable so the
# wrap/unwrap logic is testable without a live Vault.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class ExternalKmsKeyWrapper:
    """Wrap/unwrap a DEK via an external KMS/HSM. ``key_id`` names the KEK (Vault Transit key).

    Wired for Vault Transit; fails closed when no endpoint is configured.
    """

    def __init__(self, key_id: str, *, transport: Transport | None = None) -> None:
        if not key_id:
            raise ConfigError("RAG_KMS_BACKEND=external requires RAG_KMS_KEY_ID (the KEK id)")
        self._key_id = key_id
        settings = get_settings()
        self._endpoint = settings.kms_endpoint.rstrip("/")
        self._token = settings.kms_token
        self._transport = transport
        # A real (network) transport reaches an external host → it must be egress-allowlisted.
        # An injected transport (tests) is not real egress, so the check is skipped there.
        if self._endpoint and transport is None:
            assert_sovereign(self._endpoint)

    def wrap(self, dek: bytes) -> str:
        """Send the DEK to Vault Transit to be encrypted; return the ``vault:v1:…`` ciphertext."""
        self._require_wired()
        resp = self._post(
            f"/v1/transit/encrypt/{self._key_id}",
            {"plaintext": base64.b64encode(dek).decode()},
        )
        ciphertext = resp.get("data", {}).get("ciphertext")
        if not isinstance(ciphertext, str) or not ciphertext:
            raise ConfigError("KMS wrap: no ciphertext in Vault response")
        return ciphertext

    def unwrap(self, wrapped: str) -> bytes:
        """Send the wrapped ciphertext to Vault Transit to be decrypted; return the raw DEK."""
        self._require_wired()
        resp = self._post(f"/v1/transit/decrypt/{self._key_id}", {"ciphertext": wrapped})
        b64 = resp.get("data", {}).get("plaintext")
        if not isinstance(b64, str) or not b64:
            raise ConfigError("KMS unwrap: no plaintext in Vault response")
        return base64.b64decode(b64)

    def _require_wired(self) -> None:
        if not self._endpoint and self._transport is None:
            raise ConfigError(
                "external KMS wrapper is not wired: set RAG_KMS_ENDPOINT (+ RAG_KMS_TOKEN) to a "
                "sovereign Vault Transit, or provide a transport for another KMS/HSM"
            )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(path, body)
        req = urllib.request.Request(  # noqa: S310 — endpoint is operator-set + assert_sovereign'd
            self._endpoint + path,
            data=json.dumps(body).encode(),
            headers={"X-Vault-Token": self._token, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 — see above
            result: dict[str, Any] = json.load(r)
            return result
