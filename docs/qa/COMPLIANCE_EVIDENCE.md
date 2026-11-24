# QA runbook — Compliance evidence pack

This runbook explains how QA (and, downstream, a regulator or an RFP reviewer) generates and
reads the **compliance evidence pack**: a single Markdown document that maps the platform's
**live** controls to every framework a regulated Indian buyer cites, backed by the signed
residency attestation and the audit-chain integrity check.

The pack is not a static PDF — every status is computed at generation time from real
configuration and runtime signals of the deployment it is run against. Regenerate it and it
re-reads the live posture.

## What produces it

- **Endpoint (per-framework catalog):** `GET /v1/kb/{kb}/compliance/frameworks` →
  `svc_control_plane/responsible_ai.py`. Optional `?framework=niti|dpdp|iso42001|owasp|rbi|sebi|irdai`.
- **Endpoint (residency attestation):** `GET /v1/kb/{kb}/attestation` →
  `svc_control_plane/compliance.py` (residency, sovereignty, encryption, retention, and the
  **audit-chain integrity** result computed by walking the SHA-256 hash chain).
- **Generator:** `scripts/compliance_evidence.py` — pulls both endpoints for a KB and writes
  `compliance_evidence.md` plus a `compliance_evidence.json` sibling.

## How to generate the pack

Stack up (`make up`), then:

```bash
uv run python scripts/compliance_evidence.py --kb public__company-wide
```

Flags:

- `--base-url` (default `http://localhost:8000`) — the live API to attest.
- `--kb` (default `public__company-wide`) — the knowledge base id (`org__slug`). The org
  segment is used to mint a QA dev token; the KB segment is what gets attested.
- `--out` (default `compliance_evidence.md`) — the Markdown path; the `.json` sibling is
  written alongside it.
- `--token` — supply a bearer token instead of minting a dev token (e.g. to attest as a real
  OIDC principal).

The script mints a dev token itself (`POST /v1/dev/token`, `subject=qa`, `roles=[owner,admin]`,
`clearance=3`), reads the two endpoints read-only, writes the two files, prints a one-line
summary to stderr, and exits 0. It changes **no** platform state.

Example summary line (live run against this dev stack, KB `public__company-wide`):

```
wrote compliance_evidence.md + compliance_evidence.json — 7 frameworks, 49 controls:
36 satisfied / 8 partial / 5 manual; audit_chain_verified=True (75 events)
```

## What each framework covers

| Framework | id | Who cites it | What it maps |
| --- | --- | --- | --- |
| NITI Aayog — Responsible AI for All | `niti` | Govt / PSU | Safety, privacy, equality, transparency, accountability, human oversight, robustness. |
| Digital Personal Data Protection Act 2023 | `dpdp` | Every buyer | Localization, erasure, retention, security safeguards, consent, breach notification. |
| RBI — IT Governance / Cyber-Security / Outsourcing | `rbi` | Banks | Localization, access control, immutable audit, encryption/KMS, DLP, monitoring, VAPT, model risk, BCP/DR, right-to-audit. |
| SEBI — Cyber-Security & Cyber-Resilience Framework | `sebi` | Broking / AMC / depository | Governance, protect (access/data), detect, respond (audit), recover, localization. |
| **IRDAI — Information & Cyber Security Guidelines 2023** | `irdai` | **Insurers** | **Localization, RBAC/least-privilege, audit trail + log retention, encryption of sensitive/PII data, DLP/PII handling, incident & breach response, outsourcing/third-party governance.** |
| ISO/IEC 42001 — AI Management System | `iso42001` | Enterprise / assurance | AI risk, data governance, transparency, human oversight, security, logging, AIMS governance. |
| OWASP Top 10 for LLM Applications | `owasp` | Security review | Prompt injection, insecure output, sensitive-info disclosure, excessive agency, overreliance, model theft. |

Adding a framework or a control is **data, not code** — a new builder function plus a
`_FRAMEWORK_BUILDERS` registry entry.

## How statuses are derived (SATISFIED vs PARTIAL vs MANUAL)

Every control resolves to one of three statuses, computed from the live `Signals` the
generator's target deployment reports (`responsible_ai._gather_signals`):

- **SATISFIED** — enforced by the platform. Either an **architectural guarantee** that holds
  regardless of configuration (tenant isolation, grounded-or-abstain, hash-chained audit,
  envelope encryption, on-prem inspectability) **or** a flag-gated control that is currently
  **enabled** (guardrails, faithfulness, continuous-eval, sovereignty enforcement, OIDC+MFA).
- **PARTIAL** — the capability exists but is **not fully enabled/configured** in this
  deployment. Example: on this dev stack `irdai.access_control` is PARTIAL because the console
  authenticates with dev-JWT (OIDC/MFA are available via `RAG_AUTH_MODE=oidc` but not enforced
  here); `dpdp.retention` is PARTIAL when no retention window is set.
- **MANUAL** — an **organisational process** the platform supports with evidence but **cannot
  enforce** on the operator's behalf: consent management, breach/incident notification
  (`irdai.incident_response`), bias audits, the AIMS/governance policy set. These are reported
  MANUAL by design so the pack **never over-claims** a process as satisfied.

The rule of thumb for a reviewer: **SATISFIED** = the platform does it and QA can prove it with
an automated test (see the appendix); **PARTIAL** = flip the flag / finish the config to reach
SATISFIED; **MANUAL** = the buyer's own team owns the process, the platform gives them the
evidence trail.

Because status is computed from live signals, the same KB can move a control from PARTIAL to
SATISFIED simply by enabling the gating flag (e.g. `RAG_AUTH_MODE=oidc` with MFA) — regenerate
the pack and it reflects the new posture.

## How audit-chain integrity + the signed attestation back the claims

Two independent, cryptographic mechanisms underpin the sovereignty and immutability claims in
the pack, and both surface directly in it:

- **Audit-chain integrity** — every ingest/query/answer/deletion is recorded as an `AuditEvent`
  in a **SHA-256 hash chain** (each event carries `prev_hash` + a recomputable `event_hash`).
  `compliance._verify_audit_chain` walks the whole chain for the KB and confirms it is intact;
  the result appears as **`audit_chain_verified`** (with the event count) in the attestation
  block. A single altered or dropped event breaks the chain and flips this to `False` — that is
  the tamper-evidence behind `*.audit_trail`, `sebi.respond_audit`, and `niti.accountability`.
- **Residency attestation** — the `GET .../attestation` report states data residency
  (India, self-hosted), sovereignty enforcement, the egress allowlist, and encryption posture.
  The **per-node** counterpart (`GET /v1/platform/residency/node`) is additionally **Ed25519
  signed** (`signer_kid` + `signed_token`), so a regulator holding the public key can
  independently verify the node's residency claim was not forged or altered after issuance.

Together these mean the pack's sovereignty/immutability claims are not assertions — a reviewer
can verify the audit chain and the signature themselves.

## Appendix — control → proving test

The pack's final section maps key controls to the automated tests that enforce them, so a
reviewer can go from a claim straight to the test and run it (`uv run pytest <path> -q`). The
current mappings:

- **Tenant isolation** → `tests/integration/test_tenant_isolation.py::test_beta_cannot_read_alpha_across_all_stores`
- **Audit hash-chain integrity** → `tests/e2e/test_compliance.py::test_attestation_reports_sovereignty_and_chain`,
  `tests/e2e/test_erasure_adversarial.py::test_certificate_on_verifying_audit_chain`
- **Grounded-or-abstain invariant** → `tests/e2e/test_grounded_or_abstain_invariant.py`
- **RBAC / least-privilege** → `tests/e2e/test_rbac_enforcement.py`
- **Right-to-erasure cascade** → `tests/e2e/test_compliance.py::test_right_to_erasure_by_subject_cascades`,
  `tests/e2e/test_erasure_adversarial.py`
- **Framework catalog & dispatch (incl. IRDAI)** →
  `services/control-plane/tests/test_responsible_ai_unit.py`,
  `services/control-plane/tests/test_responsible_ai_irdai.py`,
  `tests/e2e/test_responsible_ai.py::test_frameworks_report`

## Verifying the IRDAI addition in isolation

```bash
uv run pytest services/control-plane/tests/test_responsible_ai_irdai.py -q
```

(If `uv` is not on PATH, use the project venv: `.venv/bin/python -m pytest <path> -q` from the repository root.)
