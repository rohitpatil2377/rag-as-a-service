"""Unit tests for secrets-at-rest encryption (FR-E1 BYO-model key storage)."""

from __future__ import annotations

import pytest

from rag_common import decrypt_secret, encrypt_secret
from rag_common.crypto import reset_crypto_cache
from rag_common.errors import ConfigError


def test_encrypt_decrypt_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_SECRETS_MASTER_KEY", "unit-test-master-key")
    get_settings.cache_clear()
    reset_crypto_cache()

    secret = "sk-aiops-super-secret-key"
    ct = encrypt_secret(secret)
    assert ct != secret  # never stored in the clear
    assert secret not in ct
    assert decrypt_secret(ct) == secret


def test_wrong_master_key_cannot_decrypt(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_common.config import get_settings

    monkeypatch.setenv("RAG_SECRETS_MASTER_KEY", "key-A")
    get_settings.cache_clear()
    reset_crypto_cache()
    ct = encrypt_secret("payload")

    monkeypatch.setenv("RAG_SECRETS_MASTER_KEY", "key-B")
    get_settings.cache_clear()
    reset_crypto_cache()
    with pytest.raises(ConfigError):
        decrypt_secret(ct)
