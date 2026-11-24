"""Unit tests for the external KMS (Vault Transit) key wrapper — wrap/unwrap logic without a
live Vault, via an injected transport."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from rag_common.errors import ConfigError
from rag_common.kms import ExternalKmsKeyWrapper


def _fake_vault() -> Any:
    """A stand-in Vault Transit: encrypt tags the base64 plaintext; decrypt reverses it."""

    def transport(path: str, body: dict[str, Any]) -> dict[str, Any]:
        if "/encrypt/" in path:
            return {"data": {"ciphertext": "vault:v1:" + body["plaintext"]}}
        if "/decrypt/" in path:
            b64 = body["ciphertext"].removeprefix("vault:v1:")
            return {"data": {"plaintext": b64}}
        raise AssertionError(f"unexpected path {path}")

    return transport


def test_wrap_unwrap_round_trip() -> None:
    kms = ExternalKmsKeyWrapper("rag-kek", transport=_fake_vault())
    dek = b"\x00\x01\x02 a 32-byte-ish data key value!"
    wrapped = kms.wrap(dek)
    assert wrapped.startswith("vault:v1:")
    assert base64.b64decode(wrapped.removeprefix("vault:v1:")) == dek  # plaintext was base64'd
    assert kms.unwrap(wrapped) == dek  # round-trips back to the exact DEK


def test_requires_key_id() -> None:
    with pytest.raises(ConfigError, match="KEK id"):
        ExternalKmsKeyWrapper("", transport=_fake_vault())


def test_fails_closed_without_endpoint_or_transport() -> None:
    # No endpoint configured and no transport injected → fail closed, never downgrade.
    kms = ExternalKmsKeyWrapper("rag-kek")  # endpoint empty by default in tests
    with pytest.raises(ConfigError, match="not wired"):
        kms.wrap(b"dek")
    with pytest.raises(ConfigError, match="not wired"):
        kms.unwrap("vault:v1:x")


def test_malformed_vault_response_fails_closed() -> None:
    def bad(path: str, body: dict[str, Any]) -> dict[str, Any]:
        return {"data": {}}  # no ciphertext / plaintext

    kms = ExternalKmsKeyWrapper("rag-kek", transport=bad)
    with pytest.raises(ConfigError, match="no ciphertext"):
        kms.wrap(b"dek")
    with pytest.raises(ConfigError, match="no plaintext"):
        kms.unwrap("vault:v1:x")
