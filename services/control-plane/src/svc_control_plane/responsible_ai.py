"""Responsible-AI compliance pack (FR-H8+, v1.1) — framework attestation.

Regulated Indian buyers (govt, BFSI, PSU) put "map your controls to <framework>" in every
RFP. This module answers it as a LIVE report instead of a static PDF: it maps the platform's
actual controls to the requirements of the frameworks those buyers cite, and computes each
control's status from real configuration + runtime signals (audit-chain integrity, enabled
guardrails, encryption posture, …). What is an architectural guarantee (sovereignty,
grounded-or-abstain, tenant isolation, hash-chained audit) reports SATISFIED; what depends on
a flag reports SATISFIED / PARTIAL from that flag; what is inherently an organisational process
(consent, breach notification, bias audits) reports MANUAL so the report never over-claims.

Frameworks covered: NITI Aayog "Responsible AI for All" principles, the DPDP Act 2023, the
RBI IT-Governance/Cyber-Security/outsourcing expectations and the SEBI CSCRF (for BFSI
buyers), the IRDAI Information & Cyber Security Guidelines (for insurers), ISO/IEC 42001 (AI
management system), and the OWASP LLM Top-10. The catalog is declarative — adding a framework
or control is data, not code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from rag_common import get_settings
from svc_control_plane import compliance, tenant_crypto


class ControlStatus(StrEnum):
    SATISFIED = "satisfied"  # enforced by the platform (architecture or an enabled control)
    PARTIAL = "partial"  # available but not fully enabled/configured in this deployment
    MANUAL = "manual"  # an organisational process the platform supports but cannot enforce
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class Signals:
    """Live posture the controls are evaluated against."""

    sovereignty_enforced: bool
    guardrails: bool
    faithfulness: bool
    continuous_eval: bool
    oidc: bool
    mfa: bool
    rate_limits: bool
    byok: bool
    audit_chain_ok: bool
    audit_events: int
    retention_days: int
    multilingual: bool


class Control(BaseModel):
    id: str
    title: str
    requirement: str
    status: ControlStatus
    evidence: list[str] = Field(default_factory=list)


class Framework(BaseModel):
    id: str
    name: str
    version: str
    controls: list[Control]
    satisfied: int
    total: int


class ComplianceReport(BaseModel):
    kb_id: str
    generated_at: datetime
    frameworks: list[Framework]
    summary: dict[str, int]
    notes: list[str] = Field(default_factory=list)


def _c(cid: str, title: str, requirement: str, status: ControlStatus, evidence: list[str]) -> Control:
    return Control(id=cid, title=title, requirement=requirement, status=status, evidence=evidence)


def _niti(s: Signals) -> list[Control]:
    sat, part, man = ControlStatus.SATISFIED, ControlStatus.PARTIAL, ControlStatus.MANUAL
    return [
        _c(
            "niti.safety",
            "Safety & Reliability",
            "AI must be reliable and behave as intended.",
            sat if (s.faithfulness and s.continuous_eval) else part,
            [
                "Grounded-or-abstain (type-enforced): no citation ⇒ no answer",
                f"Faithfulness verification {'ON' if s.faithfulness else 'available'} (FR-E4)",
                f"Continuous evaluation {'ON' if s.continuous_eval else 'available'} (FR-G9)",
            ],
        ),
        _c(
            "niti.privacy",
            "Privacy & Security",
            "Protect personal data through the lifecycle.",
            sat,
            [
                "PII detection + masking at ingest (checksum-aware, FR-B6/G2)",
                "Per-tenant envelope encryption at rest" + (" + BYOK/KMS" if s.byok else ""),
                "Namespace tenant isolation across every store",
                f"Sovereignty egress guard {'enforced' if s.sovereignty_enforced else 'DISABLED (non-prod)'}",
            ],
        ),
        _c(
            "niti.equality",
            "Equality & Inclusivity",
            "Non-discrimination; serve India's linguistic diversity.",
            part if s.multilingual else man,
            [
                "Indic language detection + translation seam (FR-B1)",
                "Bias/fairness audit of a tenant's own corpus is an organisational process",
            ],
        ),
        _c(
            "niti.transparency",
            "Transparency",
            "Decisions must be explainable and traceable.",
            sat,
            [
                "Every answer carries citations to source spans",
                "In-document citation viewer (FR-I3)",
                f"Hash-chained audit trail ({'verified' if s.audit_chain_ok else 'UNVERIFIED'})",
            ],
        ),
        _c(
            "niti.accountability",
            "Accountability",
            "Clear responsibility and an audit record.",
            sat if s.audit_chain_ok else part,
            [
                f"Tamper-evident SHA-256 audit chain ({s.audit_events} events)",
                "RBAC + ABAC on every action",
                "HITL review queue (FR-F6)",
            ],
        ),
        _c(
            "niti.human_oversight",
            "Human Oversight",
            "A human can review and override AI outputs.",
            sat,
            [
                "Human-in-the-loop review of flagged answers (FR-F6)",
                "Faithfulness-flagged answers route to a reviewer, not the end user",
            ],
        ),
        _c(
            "niti.robustness",
            "Robustness / Security of the system",
            "Resist adversarial and abusive inputs.",
            sat if s.guardrails else part,
            [
                f"Prompt-injection + toxicity guardrails {'ON' if s.guardrails else 'available'} (FR-G1/G3)",
                f"Per-org rate limiting {'ON' if s.rate_limits else 'available'} (FR-H1)",
            ],
        ),
    ]


def _dpdp(s: Signals) -> list[Control]:
    sat, part, man = ControlStatus.SATISFIED, ControlStatus.PARTIAL, ControlStatus.MANUAL
    return [
        _c(
            "dpdp.localization",
            "Data localization / residency",
            "Personal data processed within India on sovereign infrastructure.",
            sat if s.sovereignty_enforced else part,
            [
                "Fully self-hostable; no foreign/managed AI or cloud calls at runtime",
                "Egress allowlist enforced in code (assert_sovereign)",
            ],
        ),
        _c(
            "dpdp.erasure",
            "Right to erasure",
            "Erase a data principal's personal data on request.",
            sat,
            [
                "/kb/{id}/erase cascades across catalog, dense+sparse indexes, object bytes, cache, and chat history",
                "Emits a tamper-evident erasure certificate (AuditAction.ERASURE)",
            ],
        ),
        _c(
            "dpdp.retention",
            "Storage / retention limitation",
            "Retain personal data only as long as necessary.",
            sat if s.retention_days > 0 else part,
            [
                f"Per-KB retention policy ({s.retention_days} days)"
                if s.retention_days > 0
                else "Retention policy available but not set (keeps until erased)"
            ],
        ),
        _c(
            "dpdp.security",
            "Reasonable security safeguards",
            "Protect personal data against breach.",
            sat,
            [
                "Encryption at rest (per-tenant envelope) + in transit",
                "RBAC/ABAC + permission-aware retrieval (FR-A6)",
                ("Federated SSO via OIDC" + (" with enforced MFA" if s.mfa else ""))
                if s.oidc
                else "Federated SSO (OIDC/MFA) available (RAG_AUTH_MODE=oidc)",
            ],
        ),
        _c(
            "dpdp.consent",
            "Consent & purpose limitation",
            "Process data only for consented, specified purposes.",
            man,
            ["Platform records purpose in the audit trail; obtaining/managing consent is a data-fiduciary process"],
        ),
        _c(
            "dpdp.breach",
            "Breach notification",
            "Detect and report personal-data breaches.",
            man,
            ["Audit chain + observability provide the evidence trail; notification workflow is an operator process"],
        ),
    ]


def _iso42001(s: Signals) -> list[Control]:
    sat, part, man = ControlStatus.SATISFIED, ControlStatus.PARTIAL, ControlStatus.MANUAL
    return [
        _c(
            "iso42001.risk",
            "AI risk assessment & monitoring",
            "Continuously assess AI system risk and performance.",
            sat if s.continuous_eval else part,
            [
                f"Continuous evaluation with drift alerting {'ON' if s.continuous_eval else 'available'} (FR-G9)",
                "Faithfulness scoring per answer (FR-E4)",
            ],
        ),
        _c(
            "iso42001.data_governance",
            "Data governance",
            "Govern data quality, lineage, and protection.",
            sat,
            ["PII masking, per-doc metadata/lineage, retention, near-dup detection"],
        ),
        _c(
            "iso42001.transparency",
            "Transparency & information to users",
            "Users are informed of AI outputs and their basis.",
            sat,
            ["Citations + faithfulness meter + suggested-follow-ups surfaced in the console"],
        ),
        _c(
            "iso42001.human_oversight",
            "Human oversight",
            "Humans oversee and can intervene.",
            sat,
            ["HITL review queue; flagged answers held for review (FR-F6)"],
        ),
        _c(
            "iso42001.security",
            "Information security of the AI system",
            "Protect models, data, and keys.",
            sat,
            [
                "Per-tenant envelope encryption" + (" with external KMS/BYOK" if s.byok else ""),
                "Key rotation with audit (AuditAction.KEY_ROTATION)",
            ],
        ),
        _c(
            "iso42001.logging",
            "Logging & traceability",
            "Record system events for accountability.",
            sat if s.audit_chain_ok else part,
            [f"Hash-chained audit ({s.audit_events} events), OpenTelemetry tracing"],
        ),
        _c(
            "iso42001.mgmt_system",
            "AI management system governance",
            "Documented roles, policies, and lifecycle management.",
            man,
            ["Platform provides the controls + evidence; the AIMS policy set is an organisational artefact"],
        ),
    ]


def _owasp(s: Signals) -> list[Control]:
    sat, part = ControlStatus.SATISFIED, ControlStatus.PARTIAL
    return [
        _c(
            "owasp.llm01",
            "LLM01 Prompt Injection",
            "Mitigate malicious prompt manipulation.",
            sat if s.guardrails else part,
            # Two surfaces, and the second is the one an auditor asks about: the QUERY is guarded
            # (heuristic injection guard), and RETRIEVED DOCUMENT TEXT is fenced and declared data
            # rather than instructions (P32). Reporting SATISFIED on the input guard alone was an
            # over-claim while the untrusted corpus went into the prompt unfenced.
            [
                f"Input prompt-injection guardrail {'ON' if s.guardrails else 'available'}; blocked ⇒ ABSTAINED",
                "Retrieved source text fenced and declared data-not-instructions in the prompt (P32)",
            ],
        ),
        _c(
            "owasp.llm02",
            "LLM02 Insecure Output Handling",
            "Screen model output before use.",
            sat if s.guardrails else part,
            [f"Output toxicity guardrail {'ON' if s.guardrails else 'available'} (FR-G3)"],
        ),
        _c(
            "owasp.llm06",
            "LLM06 Sensitive Information Disclosure",
            "Prevent leakage of sensitive data.",
            sat,
            [
                "PII masked before indexing; permission-aware retrieval (FR-A6)",
                "Answers grounded only in retrieved spans (never parametric knowledge)",
            ],
        ),
        _c(
            "owasp.llm08",
            "LLM08 Excessive Agency",
            "Constrain what the system/agents may do.",
            sat,
            ["Agents/MCP default to least-privilege (public-ACL, reader); RBAC + HITL gate sensitive actions"],
        ),
        _c(
            "owasp.llm09",
            "LLM09 Overreliance",
            "Guard against unwarranted trust in outputs.",
            sat if s.faithfulness else part,
            ["Grounded-or-abstain + citations; faithfulness flags unsupported answers to a human"],
        ),
        _c(
            "owasp.llm10",
            "LLM10 Model Theft",
            "Protect proprietary models and keys.",
            sat,
            [
                "Tenant model keys envelope-encrypted" + (" (BYOK/KMS)" if s.byok else ""),
                "Access controlled + audited; sovereignty guard blocks exfiltration endpoints",
            ],
        ),
    ]


def _rbi(s: Signals) -> list[Control]:
    """RBI expectations for a bank running an internal IT/AI system: the IT-Governance Master
    Direction (2023), the Cyber-Security Framework, and the IT-outsourcing guidelines."""
    sat, part = ControlStatus.SATISFIED, ControlStatus.PARTIAL
    return [
        _c(
            "rbi.localization",
            "Data localization & residency",
            "Financial/customer data stored and processed within India.",
            sat if s.sovereignty_enforced else part,
            [
                "Fully self-hostable in the bank's own DC; no foreign/managed AI or cloud calls at runtime",
                "Egress allowlist enforced in code (assert_sovereign)",
            ],
        ),
        _c(
            "rbi.access_control",
            "Access control & authentication",
            "Role-based, least-privilege access with strong authentication (Cyber-Security Framework).",
            sat if (s.oidc and s.mfa) else part,
            [
                "RBAC + ABAC on every action; permission-aware retrieval (need-to-know, FR-A6)",
                ("Federated SSO via OIDC" + (" with enforced MFA" if s.mfa else " (MFA not enforced)"))
                if s.oidc
                else "OIDC/MFA available (RAG_AUTH_MODE=oidc); console currently dev-JWT",
            ],
        ),
        _c(
            "rbi.audit_trail",
            "Immutable audit trail",
            "Tamper-evident logging of access and actions for supervisory review.",
            sat if s.audit_chain_ok else part,
            [
                f"SHA-256 hash-chained audit ({s.audit_events} events), integrity-verifiable",
                "Every ingest/query/answer/deletion recorded with actor + purpose",
            ],
        ),
        _c(
            "rbi.encryption",
            "Encryption & key management",
            "Encrypt sensitive data at rest and in transit; control keys.",
            sat,
            [
                "Per-tenant envelope encryption at rest" + (" + external KMS/HSM (BYOK)" if s.byok else ""),
                "Key rotation with audit (AuditAction.KEY_ROTATION); crypto-shred on key revocation",
            ],
        ),
        _c(
            "rbi.dlp",
            "Data-leak prevention",
            "Prevent leakage of customer / account / card data.",
            sat,
            [
                "Banking-aware PII masking at ingest (account/MICR/CVV/CIF + Aadhaar/PAN/card, FR-B6)",
                "Answers grounded only in retrieved spans; permission-aware retrieval blocks cross-scope leakage",
            ],
        ),
        _c(
            "rbi.monitoring",
            "Continuous monitoring & anomaly detection",
            "Monitor the system for degradation and abuse.",
            sat if (s.continuous_eval and s.guardrails) else part,
            [
                f"Continuous evaluation with drift alerting {'ON' if s.continuous_eval else 'available'} (FR-G9)",
                f"Prompt-injection/abuse guardrails {'ON' if s.guardrails else 'available'}; OpenTelemetry tracing",
            ],
        ),
        _c(
            "rbi.vapt",
            "Security testing (VAPT)",
            "Periodic vulnerability assessment & penetration testing.",
            part,
            [
                "Platform hardened per a VAPT (security headers, port binding, banner suppression — "
                "see VAPT response doc)",
                "Periodic independent VAPT of the deployed instance remains an operator/assurance process",
            ],
        ),
        _c(
            "rbi.model_risk",
            "Model risk & explainability",
            "Understand, validate, and explain model-driven outputs.",
            sat if s.faithfulness else part,
            [
                "Grounded-or-abstain + per-answer citations + faithfulness scoring (explainable by construction)",
                "Formal model-risk validation & sign-off is a bank governance process the evidence supports",
            ],
        ),
        _c(
            "rbi.bcp_dr",
            "Business continuity & DR",
            "Recover the service and data within defined objectives.",
            part,
            [
                "Backup/restore scripts + DR runbook (authoritative vs derived stores, RPO/RTO); multi-replica HA seam",
                "Tested BCP/DR drills + RTO/RPO sign-off are an operator process",
            ],
        ),
        _c(
            "rbi.outsourcing_audit",
            "Right to audit / inspect",
            "Retain the ability to inspect and audit the system (IT-outsourcing guidelines).",
            sat,
            [
                "Runs on-prem in the bank's DC — full inspection of code, data, logs and models",
                "Complete audit-chain export; no opaque third-party runtime dependency",
            ],
        ),
    ]


def _sebi(s: Signals) -> list[Control]:
    """SEBI Cyber-Security & Cyber-Resilience Framework (CSCRF, 2024) — for a bank's SEBI-
    regulated arms (broking/AMC/depository). Mapped to the NIST-style Identify→Recover pillars."""
    sat, part, man = ControlStatus.SATISFIED, ControlStatus.PARTIAL, ControlStatus.MANUAL
    return [
        _c(
            "sebi.governance",
            "Governance & risk management",
            "Board-approved cyber governance and periodic risk assessment.",
            man,
            ["Platform supplies the controls + evidence; the governance policy set is an organisational artefact"],
        ),
        _c(
            "sebi.protect_access",
            "Protect — access & data security",
            "Least-privilege access, encryption, and data protection.",
            sat if (s.oidc and s.mfa) else part,
            [
                "RBAC/ABAC + permission-aware retrieval; envelope encryption" + (" + BYOK/KMS" if s.byok else ""),
                "PII/financial-data masking at ingest (FR-B6)",
            ],
        ),
        _c(
            "sebi.detect",
            "Detect — continuous monitoring",
            "Detect anomalies and security events.",
            sat if s.continuous_eval else part,
            [
                f"Continuous evaluation/drift alerting {'ON' if s.continuous_eval else 'available'}; OTel telemetry",
                f"Guardrails {'ON' if s.guardrails else 'available'} flag abusive inputs/outputs",
            ],
        ),
        _c(
            "sebi.respond_audit",
            "Respond — audit & traceability",
            "Tamper-evident records to investigate incidents.",
            sat if s.audit_chain_ok else part,
            [f"Hash-chained audit trail ({s.audit_events} events), integrity-verifiable"],
        ),
        _c(
            "sebi.recover",
            "Recover — backup & continuity",
            "Restore data and service after an incident.",
            part,
            ["Backup/restore + DR runbook; HA seam. Tested recovery drills are an operator process"],
        ),
        _c(
            "sebi.data_localization",
            "Data localization",
            "Regulated data stored within India.",
            sat if s.sovereignty_enforced else part,
            ["Self-hosted, sovereign; egress-guarded in code"],
        ),
    ]


def _irdai(s: Signals) -> list[Control]:
    """IRDAI Information & Cyber Security Guidelines (2023) expectations for an INSURER running
    an internal AI/IT system: data localization/residency, RBAC/least-privilege access, an
    immutable audit trail with log retention, encryption of sensitive/PII data, data-loss
    prevention / PII handling, incident & breach response (organisational), and outsourcing /
    third-party governance. Same honesty rule as _rbi/_sebi: architectural guarantees SATISFIED,
    flag-gated controls SATISFIED/PARTIAL by the real signal, organisational processes MANUAL."""
    sat, part, man = ControlStatus.SATISFIED, ControlStatus.PARTIAL, ControlStatus.MANUAL
    return [
        _c(
            "irdai.localization",
            "Data localization & residency",
            "Policyholder / sensitive personal data stored and processed within India.",
            sat if s.sovereignty_enforced else part,
            [
                "Fully self-hostable in the insurer's own DC; no foreign/managed AI or cloud calls at runtime",
                "Egress allowlist enforced in code (assert_sovereign)",
            ],
        ),
        _c(
            "irdai.access_control",
            "Access control & authentication",
            "Role-based, least-privilege access with strong authentication to policyholder data.",
            sat if (s.oidc and s.mfa) else part,
            [
                "RBAC + ABAC on every action; permission-aware retrieval (need-to-know, FR-A6)",
                ("Federated SSO via OIDC" + (" with enforced MFA" if s.mfa else " (MFA not enforced)"))
                if s.oidc
                else "OIDC/MFA available (RAG_AUTH_MODE=oidc); console currently dev-JWT",
            ],
        ),
        _c(
            "irdai.audit_trail",
            "Audit trail & log retention",
            "Tamper-evident logging of access and actions, retained for supervisory review.",
            sat if s.audit_chain_ok else part,
            [
                f"SHA-256 hash-chained audit ({s.audit_events} events), integrity-verifiable",
                "Every ingest/query/answer/deletion recorded with actor + purpose",
                f"Per-KB retention policy ({s.retention_days} days)"
                if s.retention_days > 0
                else "Log/data retention policy available but not set (keeps until erased)",
            ],
        ),
        _c(
            "irdai.encryption",
            "Encryption of sensitive data",
            "Encrypt sensitive / PII data at rest and in transit; control the keys.",
            sat,
            [
                "Per-tenant envelope encryption at rest" + (" + external KMS/HSM (BYOK)" if s.byok else ""),
                "Encryption in transit (TLS); key rotation with audit (AuditAction.KEY_ROTATION)",
            ],
        ),
        _c(
            "irdai.dlp",
            "Data-loss prevention & PII handling",
            "Prevent leakage of policyholder / claims / health / financial PII.",
            sat,
            [
                "PII masking at ingest (Aadhaar Verhoeff, card Luhn, PAN/passport/voter/phone, FR-B6/G2)",
                "Answers grounded only in retrieved spans; permission-aware retrieval blocks cross-scope leakage",
            ],
        ),
        _c(
            "irdai.incident_response",
            "Incident & breach response",
            "Detect, respond to, and report cyber / data-breach incidents to IRDAI.",
            man,
            [
                "Audit chain + continuous-eval alerting + OpenTelemetry provide the detection/evidence trail",
                "Incident-response plan, CERT-In/IRDAI reporting timelines and drills are an organisational process",
            ],
        ),
        _c(
            "irdai.outsourcing",
            "Outsourcing & third-party governance",
            "Retain oversight, inspection and audit rights over outsourced / third-party systems.",
            sat,
            [
                "Runs on-prem in the insurer's DC — full inspection of code, data, logs and models",
                "Complete audit-chain export; no opaque third-party runtime dependency (sovereign by design)",
            ],
        ),
    ]


_FRAMEWORK_BUILDERS = {
    "niti": ("NITI Aayog — Responsible AI for All", "2021 principles", _niti),
    "dpdp": ("Digital Personal Data Protection Act", "2023", _dpdp),
    "rbi": ("RBI — IT Governance, Cyber-Security & Outsourcing", "2023", _rbi),
    "sebi": ("SEBI — Cyber-Security & Cyber-Resilience Framework", "2024", _sebi),
    "irdai": ("IRDAI — Information & Cyber Security Guidelines", "2023", _irdai),
    "iso42001": ("ISO/IEC 42001 — AI Management System", "2023", _iso42001),
    "owasp": ("OWASP Top 10 for LLM Applications", "2025", _owasp),
}


async def _gather_signals(kb_id: str) -> Signals:
    s = get_settings()
    chain = await compliance._verify_audit_chain(kb_id)
    retention = await compliance.get_retention(kb_id)
    crypto = await tenant_crypto.status(kb_id)
    return Signals(
        sovereignty_enforced=s.sovereign_enforce,
        guardrails=s.guardrails_enabled,
        faithfulness=s.faithfulness_enabled,
        continuous_eval=s.continuous_eval_enabled,
        oidc=s.auth_mode == "oidc",
        mfa=s.oidc_require_mfa,
        rate_limits=s.quota_enabled,
        byok=bool(crypto.get("byok")),
        audit_chain_ok=chain.ok,
        audit_events=chain.events,
        retention_days=retention.retention_days,
        multilingual=s.language_detection_enabled or s.translation_backend != "noop",
    )


async def report(kb_id: str, *, only: str | None = None, now: datetime | None = None) -> ComplianceReport:
    """Compile the Responsible-AI framework attestation for a KB (optionally one framework)."""
    signals = await _gather_signals(kb_id)
    frameworks: list[Framework] = []
    for fid, (name, version, builder) in _FRAMEWORK_BUILDERS.items():
        if only and only != fid:
            continue
        controls = builder(signals)
        satisfied = sum(1 for c in controls if c.status == ControlStatus.SATISFIED)
        frameworks.append(
            Framework(id=fid, name=name, version=version, controls=controls, satisfied=satisfied, total=len(controls))
        )
    total_controls = sum(f.total for f in frameworks)
    total_satisfied = sum(f.satisfied for f in frameworks)
    notes: list[str] = []
    if not signals.sovereignty_enforced:
        notes.append("Sovereignty enforcement is DISABLED in this environment — not production-representative.")
    notes.append("MANUAL controls are organisational processes the platform supports with evidence but cannot enforce.")
    return ComplianceReport(
        kb_id=kb_id,
        generated_at=now or datetime.now(UTC),
        frameworks=frameworks,
        summary={"frameworks": len(frameworks), "controls": total_controls, "satisfied": total_satisfied},
        notes=notes,
    )
