"""Unit tests for envelope-encryption primitives (FR-H9 BYOK)."""

from __future__ import annotations

import pytest

from rag_common.crypto import (
    LocalKeyWrapper,
    decrypt_with_dek,
    encrypt_with_dek,
    generate_dek,
    reset_crypto_cache,
    unwrap_dek,
    wrap_dek,
)
from rag_common.errors import ConfigError


def test_dek_roundtrip() -> None:
    dek = generate_dek()
    ct = encrypt_with_dek(dek, "sk-secret-key")
    assert ct != "sk-secret-key"
    assert decrypt_with_dek(dek, ct) == "sk-secret-key"


def test_wrap_unwrap_dek() -> None:
    reset_crypto_cache()
    dek = generate_dek()
    wrapped = wrap_dek(dek)
    assert wrapped != dek.decode()
    assert unwrap_dek(wrapped) == dek


def test_wrong_dek_cannot_decrypt() -> None:
    ct = encrypt_with_dek(generate_dek(), "top secret")
    with pytest.raises(ConfigError):
        decrypt_with_dek(generate_dek(), ct)  # different DEK


def test_local_wrapper_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common import crypto

    reset_crypto_cache()
    assert isinstance(crypto.get_key_wrapper(), LocalKeyWrapper)


def test_external_kms_fails_closed_without_key_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings
    from rag_common.kms import ExternalKmsKeyWrapper

    with pytest.raises(ConfigError, match="RAG_KMS_KEY_ID"):
        ExternalKmsKeyWrapper("")
    # backend=external without wiring must not silently downgrade to the local key
    monkeypatch.setenv("RAG_KMS_BACKEND", "external")
    monkeypatch.setenv("RAG_KMS_KEY_ID", "sovereign-kek-1")
    get_settings.cache_clear()
    reset_crypto_cache()
    with pytest.raises(ConfigError, match="not wired"):
        wrap_dek(generate_dek())
    get_settings.cache_clear()
    reset_crypto_cache()
