"""IRDAI framework builder: control evaluation reacts to live signals, stays honest."""

from __future__ import annotations

from svc_control_plane.responsible_ai import (
    _FRAMEWORK_BUILDERS,
    Control,
    ControlStatus,
    Signals,
    _irdai,
)

_STRONG = Signals(
    sovereignty_enforced=True,
    guardrails=True,
    faithfulness=True,
    continuous_eval=True,
    oidc=True,
    mfa=True,
    rate_limits=True,
    byok=True,
    audit_chain_ok=True,
    audit_events=42,
    retention_days=90,
    multilingual=True,
)
_WEAK = Signals(
    sovereignty_enforced=False,
    guardrails=False,
    faithfulness=False,
    continuous_eval=False,
    oidc=False,
    mfa=False,
    rate_limits=False,
    byok=False,
    audit_chain_ok=True,
    audit_events=0,
    retention_days=0,
    multilingual=False,
)

_VALID = {ControlStatus.SATISFIED, ControlStatus.PARTIAL, ControlStatus.MANUAL, ControlStatus.NOT_APPLICABLE}


def test_irdai_registered_in_dispatch() -> None:
    assert "irdai" in _FRAMEWORK_BUILDERS
    name, version, builder = _FRAMEWORK_BUILDERS["irdai"]
    assert builder is _irdai
    assert "IRDAI" in name
    assert version == "2023"


def test_irdai_returns_controls_with_valid_statuses() -> None:
    for sig in (_STRONG, _WEAK):
        controls = _irdai(sig)
        assert controls, "IRDAI builder must return at least one control"
        assert all(isinstance(c, Control) for c in controls)
        assert all(c.status in _VALID for c in controls)
        assert all(c.id.startswith("irdai.") for c in controls)
        assert all(c.evidence for c in controls), "every control cites at least one evidence line"
        assert len({c.id for c in controls}) == len(controls), "control ids are unique"


def test_irdai_flagged_control_reacts_to_posture() -> None:
    # Access control is flag-driven: strong posture satisfies, weak posture downgrades.
    assert next(c for c in _irdai(_STRONG) if c.id == "irdai.access_control").status == ControlStatus.SATISFIED
    assert next(c for c in _irdai(_WEAK) if c.id == "irdai.access_control").status == ControlStatus.PARTIAL
    # Localization tracks sovereignty enforcement.
    assert next(c for c in _irdai(_STRONG) if c.id == "irdai.localization").status == ControlStatus.SATISFIED
    assert next(c for c in _irdai(_WEAK) if c.id == "irdai.localization").status == ControlStatus.PARTIAL


def test_irdai_architectural_guarantees_always_satisfied() -> None:
    # Encryption at rest/in transit + PII-DLP + outsourcing inspectability hold regardless of flags.
    for sig in (_STRONG, _WEAK):
        assert next(c for c in _irdai(sig) if c.id == "irdai.encryption").status == ControlStatus.SATISFIED
        assert next(c for c in _irdai(sig) if c.id == "irdai.dlp").status == ControlStatus.SATISFIED
        assert next(c for c in _irdai(sig) if c.id == "irdai.outsourcing").status == ControlStatus.SATISFIED


def test_irdai_incident_response_is_manual() -> None:
    # An organisational process is never over-claimed as satisfied.
    for sig in (_STRONG, _WEAK):
        assert next(c for c in _irdai(sig) if c.id == "irdai.incident_response").status == ControlStatus.MANUAL
