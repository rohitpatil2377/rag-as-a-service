"""Unit tests for OIDC/SSO + MFA verification (real-IdP path)."""

from __future__ import annotations

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from rag_common import auth
from rag_common.config import get_settings
from rag_common.errors import UnauthorizedError

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PRIV = _KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)


def _oidc(monkeypatch: pytest.MonkeyPatch, **over: str) -> None:
    monkeypatch.setenv("RAG_AUTH_MODE", "oidc")
    monkeypatch.setenv("RAG_OIDC_JWKS_URL", "https://idp.sovereign.local/jwks")
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,idp.sovereign.local")
    for k, v in over.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    # Bypass the network JWKS fetch with our public key.
    monkeypatch.setattr(auth, "_oidc_signing_key", lambda token, url: _KEY.public_key())


def _token(**claims: object) -> str:
    return jwt.encode(claims, _PRIV, algorithm="RS256")


def test_oidc_maps_claims_to_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    _oidc(monkeypatch)
    tok = _token(sub="alice@gov.in", org="cag", roles=["reader", "ingestor"], groups=["audit"])
    p = auth.authenticate(f"Bearer {tok}")
    assert p.subject == "alice@gov.in" and p.tenant_id == "cag"
    assert "ingestor" in p.roles and p.attrs["groups"] == ["audit"]


def test_oidc_requires_mfa_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _oidc(monkeypatch, RAG_OIDC_REQUIRE_MFA="true")
    no_mfa = _token(sub="bob", org="cag", amr=["pwd"])
    with pytest.raises(UnauthorizedError, match="multi-factor"):
        auth.authenticate(f"Bearer {no_mfa}")
    with_mfa = _token(sub="bob", org="cag", amr=["pwd", "otp"])
    assert auth.authenticate(f"Bearer {with_mfa}").subject == "bob"


def test_oidc_rejects_tampered_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _oidc(monkeypatch)
    with pytest.raises(UnauthorizedError):
        auth.authenticate("Bearer not.a.jwt")
    get_settings.cache_clear()
