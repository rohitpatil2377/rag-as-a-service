"""Per-tenant envelope encryption service (FR-H9 — BYOK / crypto-shred).

Each tenant's secrets (today: their model API keys) are encrypted with a per-tenant Data
Encryption Key. The DEK is stored WRAPPED by the KEK (local master key or an external KMS)
in the tenant's own ``tenant_config`` — no new table, so this is migration-free. Envelope
ciphertext is tagged with an ``env:`` prefix; anything without it is legacy flat ciphertext
and is decrypted with the old platform cipher, so existing stored keys keep working with no
migration.

Rotation is versioned: a new current DEK is prepended and old DEKs are retained so
previously-encrypted secrets still decrypt; a full rotation re-encrypts the secrets and then
prunes the retired DEKs (done by the crypto endpoint, which owns the model-connection store).
"""

from __future__ import annotations

import json

from rag_clients.factory import get_catalog
from rag_common import (
    decrypt_secret,
    decrypt_with_dek,
    encrypt_with_dek,
    generate_dek,
    get_logger,
    get_settings,
    unwrap_dek,
    wrap_dek,
)
from rag_common.errors import ConfigError

_log = get_logger(__name__)

_DEK_KEY = "crypto_deks"  # tenant_config key → JSON list of wrapped DEKs (index 0 = current)
_ENV_PREFIX = "env:"  # marks envelope-encrypted ciphertext (vs legacy flat ciphertext)


async def _wrapped_deks(tenant_id: str) -> list[str]:
    raw = (await get_catalog().get_tenant_config(tenant_id)).get(_DEK_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    return [str(w) for w in data] if isinstance(data, list) else []


async def _save_deks(tenant_id: str, deks: list[str]) -> None:
    await get_catalog().set_tenant_config(tenant_id, _DEK_KEY, json.dumps(deks))


async def _ensure_dek(tenant_id: str) -> list[str]:
    """Return the tenant's wrapped DEKs, provisioning one on first use."""
    deks = await _wrapped_deks(tenant_id)
    if not deks:
        deks = [wrap_dek(generate_dek())]
        await _save_deks(tenant_id, deks)
        _log.info("tenant_dek_provisioned", tenant_id=tenant_id)
    return deks


async def encrypt(tenant_id: str, plaintext: str) -> str:
    """Encrypt a tenant secret with the tenant's current DEK (envelope-tagged)."""
    if not plaintext:
        return ""
    deks = await _ensure_dek(tenant_id)
    return _ENV_PREFIX + encrypt_with_dek(unwrap_dek(deks[0]), plaintext)


async def decrypt(tenant_id: str, ciphertext: str) -> str:
    """Decrypt a tenant secret. Legacy (untagged) ciphertext falls back to the platform key."""
    if not ciphertext:
        return ""
    if not ciphertext.startswith(_ENV_PREFIX):
        return decrypt_secret(ciphertext)  # backward compat — pre-BYOK flat ciphertext
    body = ciphertext[len(_ENV_PREFIX) :]
    for wrapped in await _wrapped_deks(tenant_id):  # current DEK, then retired ones
        try:
            return decrypt_with_dek(unwrap_dek(wrapped), body)
        except ConfigError:
            continue
    raise ConfigError("cannot decrypt tenant secret with any known DEK")


async def rotate(tenant_id: str) -> int:
    """Prepend a fresh current DEK (old ones kept for decrypt). Returns the DEK version count."""
    deks = await _ensure_dek(tenant_id)
    deks.insert(0, wrap_dek(generate_dek()))
    await _save_deks(tenant_id, deks)
    _log.info("tenant_dek_rotated", tenant_id=tenant_id, versions=len(deks))
    return len(deks)


async def prune(tenant_id: str) -> None:
    """Drop retired DEKs, keeping only the current one (call AFTER re-encrypting secrets)."""
    deks = await _wrapped_deks(tenant_id)
    if len(deks) > 1:
        await _save_deks(tenant_id, deks[:1])


async def status(tenant_id: str) -> dict[str, object]:
    """Non-secret crypto posture for the console / attestation."""
    deks = await _wrapped_deks(tenant_id)
    s = get_settings()
    return {
        "envelope_encryption": True,
        "kms_backend": s.kms_backend,
        "byok": s.kms_backend == "external",
        "kms_key_id_configured": bool(s.kms_key_id),
        "dek_versions": len(deks),
        "provisioned": bool(deks),
    }
