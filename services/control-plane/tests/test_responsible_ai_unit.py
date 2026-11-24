"""Responsible-AI compliance pack: control evaluation reacts to live signals."""

from __future__ import annotations

from svc_control_plane.responsible_ai import (
    _FRAMEWORK_BUILDERS,
    ControlStatus,
    Signals,
    _dpdp,
    _niti,
    _owasp,
    _rbi,
    _sebi,
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


def test_all_frameworks_present() -> None:
    assert set(_FRAMEWORK_BUILDERS) == {"niti", "dpdp", "rbi", "sebi", "irdai", "iso42001", "owasp"}


def test_rbi_reacts_to_posture_and_keeps_processes_honest() -> None:
    # Flag-driven controls satisfy under a strong posture, downgrade under a weak one.
    assert next(c for c in _rbi(_STRONG) if c.id == "rbi.access_control").status == ControlStatus.SATISFIED
    assert next(c for c in _rbi(_WEAK) if c.id == "rbi.access_control").status == ControlStatus.PARTIAL
    # Architectural DLP (banking PII masking) holds regardless of flags.
    assert next(c for c in _rbi(_WEAK) if c.id == "rbi.dlp").status == ControlStatus.SATISFIED
    # Periodic VAPT + BCP drills are never over-claimed as fully satisfied.
    assert next(c for c in _rbi(_STRONG) if c.id == "rbi.vapt").status == ControlStatus.PARTIAL


def test_sebi_governance_is_manual() -> None:
    assert next(c for c in _sebi(_STRONG) if c.id == "sebi.governance").status == ControlStatus.MANUAL
    assert next(c for c in _sebi(_STRONG) if c.id == "sebi.respond_audit").status == ControlStatus.SATISFIED


def test_strong_posture_satisfies_flagged_controls() -> None:
    safety = next(c for c in _niti(_STRONG) if c.id == "niti.safety")
    assert safety.status == ControlStatus.SATISFIED
    robustness = next(c for c in _niti(_STRONG) if c.id == "niti.robustness")
    assert robustness.status == ControlStatus.SATISFIED


def test_weak_posture_downgrades_to_partial() -> None:
    safety = next(c for c in _niti(_WEAK) if c.id == "niti.safety")
    assert safety.status == ControlStatus.PARTIAL
    injection = next(c for c in _owasp(_WEAK) if c.id == "owasp.llm01")
    assert injection.status == ControlStatus.PARTIAL


def test_process_controls_are_manual_not_satisfied() -> None:
    consent = next(c for c in _dpdp(_STRONG) if c.id == "dpdp.consent")
    assert consent.status == ControlStatus.MANUAL  # never over-claims an org process


def test_retention_control_reflects_setting() -> None:
    assert next(c for c in _dpdp(_STRONG) if c.id == "dpdp.retention").status == ControlStatus.SATISFIED
    assert next(c for c in _dpdp(_WEAK) if c.id == "dpdp.retention").status == ControlStatus.PARTIAL


def test_architectural_guarantees_always_satisfied() -> None:
    # Grounded-or-abstain / transparency hold regardless of flags.
    for sig in (_STRONG, _WEAK):
        transparency = next(c for c in _niti(sig) if c.id == "niti.transparency")
        assert transparency.status == ControlStatus.SATISFIED
        disclosure = next(c for c in _owasp(sig) if c.id == "owasp.llm06")
        assert disclosure.status == ControlStatus.SATISFIED
