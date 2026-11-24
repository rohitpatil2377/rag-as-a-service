"""Grant signing-key ROTATION — a rotation never invalidates already-signed, unexpired grants.

Signing always uses the ACTIVE key; verification tries the active key PLUS retired keys still
in their rotation overlap. The signed canonical form is untouched, so Go byte-interop holds.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from svc_control_plane.grants import current_key_id, sign_grant, verify_grant

from rag_common.config import get_settings
from rag_schemas import Grant


def _keypair() -> tuple[str, str, Ed25519PrivateKey]:
    k = Ed25519PrivateKey.generate()
    priv = base64.b64encode(k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode()
    pub = base64.b64encode(k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
    return priv, pub, k


def _grant(gid: str = "g1") -> Grant:
    now = datetime.now(UTC)
    return Grant(
        grant_id=gid,
        issuer_dept_id="hr",
        subject_dept_id="fin",
        topics=["t"],
        kb_ids=["hr__k"],
        max_results_per_query=5,
        not_before=now,
        expires_at=now + timedelta(days=30),
        signed_token="",
        created_at=now,
    )


def _activate(
    monkeypatch: pytest.MonkeyPatch, priv_b64: str, *, kid: str = "k", retired: list[str] | None = None
) -> None:
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY", priv_b64)
    monkeypatch.setenv("RAG_GRANT_SIGNING_KEY_ID", kid)
    monkeypatch.setenv("RAG_GRANT_RETIRED_VERIFY_KEYS", json.dumps(retired or []))
    get_settings.cache_clear()


def test_active_key_signs_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    a_priv, _a_pub, _ = _keypair()
    _activate(monkeypatch, a_priv, kid="2026-a")
    g = _grant()
    g = g.model_copy(update={"signed_token": sign_grant(g)})
    assert verify_grant(g)
    assert current_key_id() == "2026-a"
    get_settings.cache_clear()


def test_rotation_keeps_old_grants_valid_and_signs_new_with_the_new_key(monkeypatch: pytest.MonkeyPatch) -> None:
    a_priv, a_pub, _ = _keypair()
    b_priv, _b_pub, _ = _keypair()

    # Key A is active: sign an "old" grant with it.
    _activate(monkeypatch, a_priv, kid="A")
    old = _grant("old")
    old = old.model_copy(update={"signed_token": sign_grant(old)})
    assert verify_grant(old)

    # ROTATE: B becomes active, A's PUBLIC key is retired (still trusted during overlap).
    _activate(monkeypatch, b_priv, kid="B", retired=[a_pub])
    assert verify_grant(old), "a grant signed before the rotation must still verify via the retired key"
    new = _grant("new")
    new = new.model_copy(update={"signed_token": sign_grant(new)})
    assert verify_grant(new), "a new grant is signed by the active key B"
    assert current_key_id() == "B"

    # Once A is dropped from the retired set, the old grant no longer verifies (overlap ended).
    _activate(monkeypatch, b_priv, kid="B", retired=[])
    assert not verify_grant(old)
    assert verify_grant(new)
    get_settings.cache_clear()


def test_a_grant_signed_by_an_untrusted_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    a_priv, _a_pub, _ = _keypair()
    c_priv, _c_pub, _ = _keypair()
    _activate(monkeypatch, c_priv, kid="C")
    forged = _grant()
    forged = forged.model_copy(update={"signed_token": sign_grant(forged)})  # signed by C
    _activate(monkeypatch, a_priv, kid="A", retired=[])  # A active, C not trusted
    assert not verify_grant(forged)
    get_settings.cache_clear()


def test_explicit_peer_key_path_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mesh path passes the issuer peer's known key explicitly — it must check ONLY that
    key, independent of the local keyring."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    a_priv, a_pub, _ = _keypair()
    _b_priv, b_pub, _ = _keypair()
    _activate(monkeypatch, a_priv, kid="A")
    g = _grant()
    g = g.model_copy(update={"signed_token": sign_grant(g)})  # signed by A
    a_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(a_pub))
    b_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(b_pub))
    assert verify_grant(g, a_key) is True
    assert verify_grant(g, b_key) is False
    get_settings.cache_clear()
