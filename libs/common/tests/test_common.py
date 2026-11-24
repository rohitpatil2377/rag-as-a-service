"""Unit tests for the cross-cutting foundation (no external stores)."""

from __future__ import annotations

import pytest

from rag_common import (
    SovereigntyViolationError,
    TenantIsolationError,
    current_tenant,
    tenant_scope,
    vector_collection,
)
from rag_common.auth import authenticate, issue_dev_jwt
from rag_common.errors import UnauthorizedError
from rag_common.sovereignty import assert_sovereign


def test_tenant_scope_binds_and_resets() -> None:
    with tenant_scope("acme-bank"):
        assert current_tenant() == "acme-bank"
    with pytest.raises(TenantIsolationError):
        current_tenant()  # fail-closed outside a scope


def test_namespacing_is_tenant_specific() -> None:
    assert vector_collection("tenant-a") != vector_collection("tenant-b")
    assert vector_collection("Tenant-A") == "t_tenant_a_chunks"


def test_invalid_tenant_id_rejected() -> None:
    with pytest.raises(TenantIsolationError):
        vector_collection("../etc/passwd")


def test_dev_jwt_roundtrip() -> None:
    token = issue_dev_jwt("officer-1", "cag", roles=["reader"], attrs={"grade": "AAO"})
    principal = authenticate(f"Bearer {token}")
    assert principal.tenant_id == "cag"
    assert principal.has_role("reader")
    assert principal.attrs["grade"] == "AAO"


def test_missing_token_rejected() -> None:
    with pytest.raises(UnauthorizedError):
        authenticate(None)


def test_sovereignty_guard_blocks_foreign_host() -> None:
    with pytest.raises(SovereigntyViolationError):
        assert_sovereign("https://api.openai.com/v1/chat")


def test_sovereignty_guard_allows_localhost() -> None:
    assert assert_sovereign("http://localhost:8000/v1") == "http://localhost:8000/v1"


def test_model_tls_verify_defaults_to_real_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """No CA bundle configured -> verify=True. With one -> the path itself. There is no
    configuration that yields False: a private-CA model endpoint gets its CA mounted,
    never verification disabled."""
    from rag_common import model_tls_verify
    from rag_common.config import get_settings

    monkeypatch.delenv("RAG_MODEL_CA_BUNDLE", raising=False)
    get_settings.cache_clear()
    assert model_tls_verify() is True

    monkeypatch.setenv("RAG_MODEL_CA_BUNDLE", "/etc/rag/ca.pem")
    get_settings.cache_clear()
    assert model_tls_verify() == "/etc/rag/ca.pem"
    get_settings.cache_clear()
