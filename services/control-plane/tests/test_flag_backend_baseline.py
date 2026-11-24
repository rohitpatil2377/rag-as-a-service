"""Enabling a capability must not overwrite the operator's choice of backend.

Found live: Acme runs OCR on self-hosted Chandra (RAG_OCR_BACKEND=chandra, on AIOps). The OCR
flag declared its "on" value as the literal "tesseract", so an admin toggling OCR in the console
silently repointed the platform at a local engine that is not installed in the image — and the
Models page then displayed the endpoint as www.datalab.to, a foreign host this deployment had
deliberately removed. The module's own contract says model backends are the operator's, not the
console's.

A boolean capability is different: an admin enabling guardrails means "on", whatever the env
said. So the baseline is honoured only when it is a real choice — not when it is the off value.
"""

from __future__ import annotations

import os

import pytest
from svc_control_plane import feature_flags


def test_enabling_ocr_restores_the_operator_s_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_OCR_BACKEND", "chandra")
    feature_flags.capture_baseline()

    feature_flags._apply_env("ocr", False)
    assert os.environ["RAG_OCR_BACKEND"] == "noop"

    feature_flags._apply_env("ocr", True)
    assert os.environ["RAG_OCR_BACKEND"] == "chandra", "the console chose the backend, not the operator"


def test_enabling_ocr_with_no_operator_setting_falls_back_to_the_declared_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_OCR_BACKEND", raising=False)
    feature_flags.capture_baseline()
    feature_flags._apply_env("ocr", True)
    assert os.environ["RAG_OCR_BACKEND"] == "tesseract"


def test_a_boolean_capability_still_turns_on_even_if_the_operator_shipped_it_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The baseline is not a veto: 'enable guardrails' means enable them."""
    monkeypatch.setenv("RAG_GUARDRAILS_ENABLED", "false")
    feature_flags.capture_baseline()
    feature_flags._apply_env("guardrails", True)
    assert os.environ["RAG_GUARDRAILS_ENABLED"] == "true"


def test_translation_keeps_the_operator_s_backend_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TRANSLATION_BACKEND", "sarvam")
    feature_flags.capture_baseline()
    feature_flags._apply_env("translation", True)
    assert os.environ["RAG_TRANSLATION_BACKEND"] == "sarvam"


def test_a_backend_flag_reads_as_enabled_for_any_backend_the_operator_chose() -> None:
    """The console's toggle must reflect whether the capability is ON, not whether the operator
    happens to have picked the same backend this table names. Reported live: OCR ran on Chandra,
    probed online, and the switch showed OFF — an operator reading that would turn it 'on' to fix
    something that was never off, or worse, trust it and think scans were being skipped."""
    from rag_common.config import get_settings

    for backend, expected in (("chandra", True), ("tesseract", True), ("paddleocr", True), ("noop", False)):
        os.environ["RAG_OCR_BACKEND"] = backend
        get_settings.cache_clear()
        assert feature_flags.effective("ocr") is expected, backend


def test_a_boolean_capability_still_reads_from_its_own_value() -> None:
    from rag_common.config import get_settings

    os.environ["RAG_GUARDRAILS_ENABLED"] = "false"
    get_settings.cache_clear()
    assert feature_flags.effective("guardrails") is False
    os.environ["RAG_GUARDRAILS_ENABLED"] = "true"
    get_settings.cache_clear()
    assert feature_flags.effective("guardrails") is True
