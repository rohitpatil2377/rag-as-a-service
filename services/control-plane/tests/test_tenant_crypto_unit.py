"""Per-tenant envelope encryption service tests (FR-H9 BYOK) against the in-memory catalog."""

from __future__ import annotations

import pytest
from svc_control_plane import tenant_crypto

from rag_common import encrypt_secret
from rag_common.crypto import reset_crypto_cache
from rag_common.errors import ConfigError


@pytest.fixture(autouse=True)
async def _clean(inmemory: None) -> None:
    reset_crypto_cache()


async def test_envelope_roundtrip_and_tagging() -> None:
    ct = await tenant_crypto.encrypt("kbA", "sk-tenant-a")
    assert ct.startswith("env:")  # tagged as envelope ciphertext
    assert await tenant_crypto.decrypt("kbA", ct) == "sk-tenant-a"


async def test_per_tenant_isolation() -> None:
    ct_a = await tenant_crypto.encrypt("kbA", "secret-a")
    # kbB has its own DEK — it must NOT be able to decrypt kbA's ciphertext.
    await tenant_crypto.encrypt("kbB", "secret-b")
    with pytest.raises(ConfigError):
        await tenant_crypto.decrypt("kbB", ct_a)


async def test_legacy_flat_ciphertext_still_decrypts() -> None:
    """Pre-BYOK ciphertext (no env: tag) falls back to the platform master key."""
    legacy = encrypt_secret("old-key")
    assert not legacy.startswith("env:")
    assert await tenant_crypto.decrypt("kbLegacy", legacy) == "old-key"


async def test_rotation_keeps_old_ciphertext_readable() -> None:
    ct_v1 = await tenant_crypto.encrypt("kbR", "secret-v1")
    versions = await tenant_crypto.rotate("kbR")
    assert versions == 2  # new current DEK + retired old DEK
    # old ciphertext still decrypts (retired DEK retained) ...
    assert await tenant_crypto.decrypt("kbR", ct_v1) == "secret-v1"
    # ... and new writes use the new current DEK.
    ct_v2 = await tenant_crypto.encrypt("kbR", "secret-v2")
    assert await tenant_crypto.decrypt("kbR", ct_v2) == "secret-v2"


async def test_prune_after_reencrypt() -> None:
    ct_v1 = await tenant_crypto.encrypt("kbP", "secret")
    await tenant_crypto.rotate("kbP")
    # simulate the endpoint re-encrypting the secret with the new DEK, then pruning
    plaintext = await tenant_crypto.decrypt("kbP", ct_v1)
    ct_v2 = await tenant_crypto.encrypt("kbP", plaintext)
    await tenant_crypto.prune("kbP")
    st = await tenant_crypto.status("kbP")
    assert st["dek_versions"] == 1
    assert await tenant_crypto.decrypt("kbP", ct_v2) == "secret"


async def test_status_shape() -> None:
    await tenant_crypto.encrypt("kbS", "x")
    st = await tenant_crypto.status("kbS")
    assert st["envelope_encryption"] is True
    assert st["kms_backend"] == "local"
    assert st["byok"] is False
    assert st["provisioned"] is True
