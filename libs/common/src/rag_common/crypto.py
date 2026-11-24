"""Symmetric encryption for secrets at rest (TDD §9 — sovereign KMS/HSM).

Used to store client-supplied secrets — notably per-tenant model API keys entered in the
console (FR-E1 "bring your own model") — as ciphertext, never plaintext, in the metadata
store. A Fernet (AES-128-CBC + HMAC) key is derived from the configured master key, so:

* dev derives it from ``RAG_SECRETS_MASTER_KEY`` (an insecure default out of the box), and
* prod supplies a strong master key from the HSM/KMS with **no call-site change**.

Rotating the master key invalidates existing ciphertexts (re-encrypt on rotation).
"""

from __future__ import annotations

import base64
import hashlib
from functools import lru_cache
from typing import Protocol, runtime_checkable

from cryptography.fernet import Fernet, InvalidToken

from rag_common.config import get_settings
from rag_common.errors import ConfigError


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    master = get_settings().secrets_master_key
    if not master:
        raise ConfigError("secrets_master_key is not configured; cannot encrypt/decrypt secrets")
    # Derive a stable 32-byte urlsafe key from any master string (KMS-issued in prod).
    key = base64.urlsafe_b64encode(hashlib.sha256(master.encode()).digest())
    return Fernet(key)


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret for storage at rest. Returns urlsafe ciphertext text."""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    """Decrypt a secret produced by :func:`encrypt_secret` (same master key)."""
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise ConfigError("cannot decrypt secret (wrong or rotated master key)") from exc


def reset_crypto_cache() -> None:
    """Drop the cached cipher (call after the master key changes; used in tests)."""
    _fernet.cache_clear()
    _key_wrapper.cache_clear()


# --- Envelope encryption / BYOK (FR-H9 — per-tenant keys, sovereign KMS) ------
# Instead of encrypting every tenant's secrets directly with one platform key, each tenant
# gets its own random Data Encryption Key (DEK). The DEK is stored WRAPPED (encrypted) by a
# Key Encryption Key (KEK); tenant secrets are encrypted with the unwrapped DEK. This gives:
#   * per-tenant crypto isolation (one tenant's DEK never touches another's data);
#   * rotation of a tenant's DEK without re-keying anyone else (old DEKs kept for decrypt);
#   * BYOK / crypto-shred — with an external KMS holding the KEK, revoking the tenant's KMS
#     key renders their wrapped DEK (and therefore their data) permanently unreadable.
# The KEK lives behind the :class:`KeyWrapper` seam: LocalKeyWrapper derives it from the
# master key (dev default, same trust model as before); an external KMS/HSM plugs in via
# ``RAG_KMS_BACKEND=external`` with NO call-site change.


@runtime_checkable
class KeyWrapper(Protocol):
    """Wraps/unwraps a per-tenant DEK with the platform (or tenant BYOK) KEK."""

    def wrap(self, dek: bytes) -> str: ...
    def unwrap(self, wrapped: str) -> bytes: ...


class LocalKeyWrapper:
    """Default KEK: derived from the configured master key (HSM/KMS-issued in prod)."""

    def wrap(self, dek: bytes) -> str:
        return _fernet().encrypt(dek).decode()

    def unwrap(self, wrapped: str) -> bytes:
        try:
            return _fernet().decrypt(wrapped.encode())
        except InvalidToken as exc:
            raise ConfigError("cannot unwrap tenant DEK (wrong or rotated master key)") from exc


@lru_cache(maxsize=1)
def _key_wrapper() -> KeyWrapper:
    backend = get_settings().kms_backend
    if backend == "external":
        from rag_common.kms import ExternalKmsKeyWrapper  # noqa: PLC0415 — optional heavy dep

        return ExternalKmsKeyWrapper(get_settings().kms_key_id)
    return LocalKeyWrapper()


def get_key_wrapper() -> KeyWrapper:
    """The active KEK wrapper (local master key, or an external KMS/HSM when configured)."""
    return _key_wrapper()


def generate_dek() -> bytes:
    """Mint a fresh per-tenant Data Encryption Key (a Fernet key)."""
    return Fernet.generate_key()


def wrap_dek(dek: bytes) -> str:
    """Wrap (encrypt) a DEK with the KEK for storage at rest."""
    return get_key_wrapper().wrap(dek)


def unwrap_dek(wrapped: str) -> bytes:
    """Unwrap a DEK produced by :func:`wrap_dek`."""
    return get_key_wrapper().unwrap(wrapped)


def encrypt_with_dek(dek: bytes, plaintext: str) -> str:
    """Encrypt a secret with a tenant's (unwrapped) DEK."""
    return Fernet(dek).encrypt(plaintext.encode()).decode()


def decrypt_with_dek(dek: bytes, ciphertext: str) -> str:
    """Decrypt a secret with a tenant's (unwrapped) DEK."""
    try:
        return Fernet(dek).decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise ConfigError("cannot decrypt secret with this DEK (rotated key?)") from exc
