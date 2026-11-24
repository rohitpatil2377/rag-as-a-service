"""Air-gap preflight checks: violation detection, assertion, and report shape."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rag_common.airgap import AirGapViolation, airgap_preflight, airgap_violations, assert_air_gap


def _settings(**overrides: Any) -> SimpleNamespace:
    """A clean, fully-offline-capable settings stub; override fields per test."""
    base = {
        "air_gap_enabled": True,
        "llm_backend": "extractive_dev",
        "vllm_url": "http://vllm:8000/v1",
        "embed_backend": "tei",
        "tei_url": "http://tei:80",
        "rerank_backend": "cross_encoder",
        "rerank_url": "http://reranker:80",
        "ocr_backend": "noop",
        "vision_llm_backend": "echo",
        "vision_llm_url": "http://vllm-vision:8000/v1",
        "chandra_url": "http://chandra:8000/v1",
        "datalab_url": "https://www.datalab.to/api/v1",
        "smtp_host": "",
        "oidc_jwks_url": "",
        "siem_enabled": False,
        "siem_endpoint": "",
        "update_release_public_key": "ed25519:deadbeefcafe",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# --- air_gap_enabled False: never raises, no matter how dirty the config ---


def test_disabled_never_raises_even_when_dirty() -> None:
    settings = _settings(
        air_gap_enabled=False,
        smtp_host="smtp.gmail.com",
        oidc_jwks_url="https://accounts.google.com/.well-known/jwks.json",
        update_release_public_key="",
        llm_backend="openai_compat",
        vllm_url="https://api.openai.com/v1",
    )
    assert_air_gap(settings)  # must not raise


def test_disabled_default_missing_attr_never_raises() -> None:
    # air_gap_enabled absent entirely -> defaults to False, still a no-op.
    settings = SimpleNamespace(smtp_host="smtp.gmail.com")
    assert_air_gap(settings)


# --- air_gap_enabled True + clean offline config: no violations, no raise ---


def test_clean_offline_config_has_no_violations() -> None:
    settings = _settings()
    assert airgap_violations(settings) == []
    assert_air_gap(settings)  # must not raise


def test_openai_compat_pointed_at_intranet_is_not_a_violation() -> None:
    # openai_compat is self-hostable; an internal target is fine.
    settings = _settings(llm_backend="openai_compat", vllm_url="http://vllm-gpu.internal:8000/v1")
    assert airgap_violations(settings) == []


# --- individual violations, each reported and each raising ---


def test_smtp_host_set_is_reported_and_raises() -> None:
    settings = _settings(smtp_host="smtp.gmail.com")
    violations = airgap_violations(settings)
    assert len(violations) == 1
    assert violations[0].setting == "smtp_host"
    assert violations[0].value == "smtp.gmail.com"
    with pytest.raises(ValueError, match="smtp_host"):
        assert_air_gap(settings)


def test_oidc_jwks_url_set_is_reported_and_raises() -> None:
    settings = _settings(oidc_jwks_url="https://idp.example.com/.well-known/jwks.json")
    violations = airgap_violations(settings)
    assert len(violations) == 1
    assert violations[0].setting == "oidc_jwks_url"
    with pytest.raises(ValueError, match="oidc_jwks_url"):
        assert_air_gap(settings)


def test_empty_update_release_public_key_is_reported_and_raises() -> None:
    settings = _settings(update_release_public_key="")
    violations = airgap_violations(settings)
    assert len(violations) == 1
    assert violations[0].setting == "update_release_public_key"
    with pytest.raises(ValueError, match="update_release_public_key"):
        assert_air_gap(settings)


def test_siem_remote_endpoint_is_reported_and_raises() -> None:
    settings = _settings(siem_enabled=True, siem_endpoint="https://splunk.saas-vendor.com/hec")
    violations = airgap_violations(settings)
    assert len(violations) == 1
    assert violations[0].setting == "siem_endpoint"
    with pytest.raises(ValueError, match="siem_endpoint"):
        assert_air_gap(settings)


def test_siem_local_endpoint_is_not_a_violation() -> None:
    settings = _settings(siem_enabled=True, siem_endpoint="http://siem.internal:8088")
    assert airgap_violations(settings) == []


def test_siem_disabled_ignores_remote_endpoint() -> None:
    # A stale/unused siem_endpoint doesn't matter if forwarding itself is off.
    settings = _settings(siem_enabled=False, siem_endpoint="https://splunk.saas-vendor.com/hec")
    assert airgap_violations(settings) == []


def test_hosted_llm_backend_off_box_is_reported_and_raises() -> None:
    settings = _settings(llm_backend="openai_compat", vllm_url="https://api.openai.com/v1")
    violations = airgap_violations(settings)
    assert len(violations) == 1
    assert violations[0].setting == "vllm_url"
    with pytest.raises(ValueError, match="vllm_url"):
        assert_air_gap(settings)


def test_chandra_api_ocr_backend_is_reported() -> None:
    # chandra_api always resolves to Datalab's hosted API by default -> off-box.
    settings = _settings(ocr_backend="chandra_api")
    violations = airgap_violations(settings)
    assert len(violations) == 1
    assert violations[0].setting == "datalab_url"


def test_multiple_violations_all_named_in_the_raised_error() -> None:
    settings = _settings(smtp_host="smtp.gmail.com", oidc_jwks_url="https://idp.example.com/jwks.json")
    violations = airgap_violations(settings)
    assert {v.setting for v in violations} == {"smtp_host", "oidc_jwks_url"}
    with pytest.raises(ValueError, match="smtp_host") as exc_info:
        assert_air_gap(settings)
    assert "oidc_jwks_url" in str(exc_info.value)


# --- violation type ---


def test_violations_are_airgap_violation_instances() -> None:
    settings = _settings(smtp_host="smtp.gmail.com")
    violations = airgap_violations(settings)
    assert isinstance(violations[0], AirGapViolation)


# --- preflight report shape ---


def test_preflight_report_shape_clean() -> None:
    report = airgap_preflight(_settings())
    assert report == {"air_gap_enabled": True, "clean": True, "violations": []}


def test_preflight_report_shape_dirty() -> None:
    report = airgap_preflight(_settings(smtp_host="smtp.gmail.com"))
    assert report["air_gap_enabled"] is True
    assert report["clean"] is False
    assert len(report["violations"]) == 1
    assert report["violations"][0]["setting"] == "smtp_host"
    assert report["violations"][0]["value"] == "smtp.gmail.com"
    assert "reason" in report["violations"][0]


def test_preflight_computes_violations_even_when_disabled() -> None:
    # Preview mode: shows what would break air-gap before the flag is flipped on.
    report = airgap_preflight(_settings(air_gap_enabled=False, smtp_host="smtp.gmail.com"))
    assert report["air_gap_enabled"] is False
    assert report["clean"] is False
    assert len(report["violations"]) == 1
