"""Compliance evidence-pack generator (item 4) — QA/regulator-facing.

Pulls the platform's LIVE compliance posture for a KB over HTTP (same style as
``scripts/benchmark.py``: httpx + argparse + a dev token) and emits a single, well-structured
Markdown evidence pack (plus a ``.json`` sibling) an auditor or QA lead can read end to end:

  * an EXECUTIVE SUMMARY — per-framework counts of SATISFIED / PARTIAL / MANUAL controls;
  * a PER-FRAMEWORK CONTROL TABLE — control id, requirement, status, and the live
    evidence/signal each status was computed from;
  * the RESIDENCY ATTESTATION block — data residency, sovereignty enforcement, egress
    allowlist, encryption posture, and the audit-chain integrity result that backs the
    immutability/sovereignty claims;
  * a CONTROL -> PROVING TEST appendix — maps key controls to the automated tests in this
    repo that actually prove them (tenant isolation, audit hash-chain, grounded-or-abstain,
    RBAC, erasure), so a reviewer can go from a claim straight to the test that enforces it.

It reads three LIVE endpoints, all served by the control plane:
  GET /v1/kb/{kb}/compliance/frameworks   (every framework: niti/dpdp/iso42001/owasp/rbi/sebi/irdai)
  GET /v1/kb/{kb}/attestation             (residency + protection + audit-chain integrity)

The script writes ONLY the two report files; it changes no platform state.

Usage (stack up):
    uv run python scripts/compliance_evidence.py --kb public__company-wide
    uv run python scripts/compliance_evidence.py --base-url http://localhost:8000 \
        --kb public__company-wide --out compliance_evidence.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# Control -> proving test map: the automated tests in this repo that enforce a key control, so a
# reviewer can jump from a compliance claim straight to the test that proves it. Kept here (not
# fetched) because it is repo-structure knowledge, and printed verbatim into the appendix.
_PROVING_TESTS: dict[str, list[str]] = {
    "Tenant isolation (multi-tenant namespace separation; *.localization, *.access_control, *.dlp)": [
        "tests/integration/test_tenant_isolation.py::test_beta_cannot_read_alpha_across_all_stores",
        "tests/integration/test_tenant_isolation_adversarial.py",
    ],
    "Audit hash-chain integrity (tamper-evidence; *.audit_trail, *.respond_audit, niti.accountability)": [
        "tests/e2e/test_compliance.py::test_attestation_reports_sovereignty_and_chain",
        "tests/e2e/test_erasure_adversarial.py::test_certificate_on_verifying_audit_chain",
    ],
    "Grounded-or-abstain invariant (transparency/overreliance; niti.transparency, owasp.llm09, irdai.dlp)": [
        "tests/e2e/test_grounded_or_abstain_invariant.py::test_type_grounded_zero_citations_raises",
        "tests/e2e/test_grounded_or_abstain_invariant.py::test_empty_kb_abstains",
    ],
    "RBAC / least-privilege access (*.access_control, owasp.llm08)": [
        "tests/e2e/test_rbac_enforcement.py::test_a_confidential_kb_is_hidden_and_blocked_below_clearance",
        "tests/e2e/test_rbac_enforcement.py::test_cannot_ingest_above_your_clearance",
    ],
    "Right-to-erasure cascade (dpdp.erasure, retention; *.dlp)": [
        "tests/e2e/test_compliance.py::test_right_to_erasure_by_subject_cascades",
        "tests/e2e/test_erasure_adversarial.py::test_erasure_by_docid_cascade_every_store",
        "tests/e2e/test_erasure_adversarial.py::test_erasure_leaves_no_graph_shard",
    ],
    "Framework catalog & dispatch (all frameworks incl. irdai)": [
        "services/control-plane/tests/test_responsible_ai_unit.py",
        "services/control-plane/tests/test_responsible_ai_irdai.py",
        "tests/e2e/test_responsible_ai.py::test_frameworks_report",
    ],
}

_STATUSES = ("satisfied", "partial", "manual", "not_applicable")


def _dev_token(base_url: str, org: str) -> str:
    with httpx.Client(base_url=base_url, timeout=30.0) as c:
        r = c.post(
            "/v1/dev/token",
            json={"subject": "qa", "tenant_id": org, "roles": ["owner", "admin"], "clearance": 3},
        )
        r.raise_for_status()
        return str(r.json()["access_token"])


def _fetch(base_url: str, token: str, kb: str) -> tuple[dict[str, Any], dict[str, Any]]:
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=base_url, timeout=120.0, headers=headers) as c:
        fr = c.get(f"/v1/kb/{kb}/compliance/frameworks")
        fr.raise_for_status()
        at = c.get(f"/v1/kb/{kb}/attestation")
        at.raise_for_status()
        return fr.json(), at.json()


def _counts(controls: list[dict[str, Any]]) -> dict[str, int]:
    out = dict.fromkeys(_STATUSES, 0)
    for c in controls:
        out[c["status"]] = out.get(c["status"], 0) + 1
    return out


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _build_markdown(kb: str, base_url: str, report: dict[str, Any], att: dict[str, Any]) -> str:
    generated = datetime.now(UTC).isoformat()
    lines: list[str] = [
        "# Compliance Evidence Pack",
        "",
        f"- **Knowledge base:** `{kb}`",
        f"- **Source:** `{base_url}` (live platform, read-only)",
        f"- **Generated:** {generated}",
        f"- **Platform report generated_at:** {report.get('generated_at', 'n/a')}",
        "",
        "> Status semantics — **SATISFIED**: enforced by the platform (architecture or an enabled",
        "> control). **PARTIAL**: available but not fully enabled/configured in this deployment.",
        "> **MANUAL**: an organisational process the platform supports with evidence but cannot",
        "> enforce. The pack never over-claims a MANUAL process as satisfied.",
        "",
        "## Executive summary",
        "",
        "| Framework | Version | Satisfied | Partial | Manual | Total |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    tot = dict.fromkeys(_STATUSES, 0)
    total_controls = 0
    for fw in report["frameworks"]:
        cc = _counts(fw["controls"])
        for s in _STATUSES:
            tot[s] += cc[s]
        total_controls += len(fw["controls"])
        lines.append(
            f"| {_md_escape(fw['name'])} | {fw['version']} | {cc['satisfied']} | "
            f"{cc['partial']} | {cc['manual']} | {len(fw['controls'])} |"
        )
    lines.append(
        f"| **All frameworks** | | **{tot['satisfied']}** | **{tot['partial']}** | "
        f"**{tot['manual']}** | **{total_controls}** |"
    )
    lines += ["", "## Per-framework control mapping", ""]
    for fw in report["frameworks"]:
        lines += [
            f"### {fw['name']} ({fw['version']})",
            "",
            f"`{fw['id']}` — {fw['satisfied']}/{fw['total']} controls satisfied.",
            "",
            "| Control | Requirement | Status | Evidence / signal |",
            "| --- | --- | --- | --- |",
        ]
        for c in fw["controls"]:
            evidence = "<br>".join(_md_escape(e) for e in c.get("evidence", []))
            lines.append(
                f"| `{c['id']}`<br>{_md_escape(c['title'])} | {_md_escape(c['requirement'])} | "
                f"**{c['status'].upper()}** | {evidence} |"
            )
        lines.append("")

    lines += [
        "## Residency & protection attestation",
        "",
        "The signed residency + protection attestation (`GET /v1/kb/{kb}/attestation`) — the",
        "sovereignty, encryption, and audit-chain-integrity evidence that backs the immutability",
        "and data-residency claims above.",
        "",
        "| Attribute | Value |",
        "| --- | --- |",
        f"| Data residency | {_md_escape(str(att.get('data_residency', 'n/a')))} |",
        f"| Sovereignty enforced | {att.get('sovereignty_enforced')} |",
        f"| Encryption at rest | {att.get('encryption_at_rest')} |",
        f"| Encryption in transit | {att.get('encryption_in_transit')} |",
        f"| **Audit-chain verified** | **{att.get('audit_chain_verified')}** |",
        f"| Audit events (chain length) | {att.get('audit_events')} |",
        f"| Retention (days) | {att.get('retention_days')} |",
        f"| Documents | {att.get('documents')} |",
        f"| Egress allowlist | {_md_escape(', '.join(att.get('egress_allowlist', [])) or '(none)')} |",
        "",
    ]
    fed = att.get("federation") or {}
    lines += [
        "**Federation posture** (the one mechanism that opens tenant isolation — shown for the auditor):",
        "",
        f"- enabled: `{fed.get('enabled')}`; grants issued (active): "
        f"{fed.get('grants_issued', 0)} ({fed.get('grants_issued_active', 0)}); "
        f"received (active): {fed.get('grants_received_active', 0)}; "
        f"delegated: {fed.get('delegated_grants_issued', 0)}; "
        f"max delegation depth: {fed.get('max_delegation_depth', 0)}; "
        f"revoked: {fed.get('revoked_grants', 0)}",
        "",
    ]
    if att.get("notes"):
        lines.append("**Attestation notes:**")
        lines.append("")
        for n in att["notes"]:
            lines.append(f"- {_md_escape(n)}")
        lines.append("")
    if report.get("notes"):
        lines.append("**Framework-report notes:**")
        lines.append("")
        for n in report["notes"]:
            lines.append(f"- {_md_escape(n)}")
        lines.append("")

    lines += [
        "## Appendix: control -> proving test",
        "",
        "Key controls above are enforced by automated tests in this repository. A reviewer can run",
        "any of these to independently confirm the claim (`uv run pytest <path> -q`).",
        "",
        "| Control area | Proving test(s) |",
        "| --- | --- |",
    ]
    for area, tests in _PROVING_TESTS.items():
        joined = "<br>".join(f"`{t}`" for t in tests)
        lines.append(f"| {_md_escape(area)} | {joined} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--kb", default="public__company-wide", help="knowledge base id (org__slug)")
    p.add_argument("--out", default="compliance_evidence.md", help="markdown output path (.json sibling written too)")
    p.add_argument("--token", default=None, help="use this bearer token instead of minting a dev token")
    args = p.parse_args()

    org = args.kb.split("__", 1)[0]
    token = args.token or _dev_token(args.base_url, org)
    report, att = _fetch(args.base_url, token, args.kb)

    md = _build_markdown(args.kb, args.base_url, report, att)
    md_path = Path(args.out)
    md_path.write_text(md)
    json_path = md_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "kb_id": args.kb,
                "base_url": args.base_url,
                "generated_at": datetime.now(UTC).isoformat(),
                "frameworks": report,
                "attestation": att,
                "proving_tests": _PROVING_TESTS,
            },
            indent=2,
            default=str,
        )
    )

    tot = dict.fromkeys(_STATUSES, 0)
    for fw in report["frameworks"]:
        for s, n in _counts(fw["controls"]).items():
            tot[s] += n
    print(
        f"wrote {md_path} + {json_path} — {len(report['frameworks'])} frameworks, "
        f"{sum(tot.values())} controls: {tot['satisfied']} satisfied / {tot['partial']} partial / "
        f"{tot['manual']} manual; audit_chain_verified={att.get('audit_chain_verified')} "
        f"({att.get('audit_events')} events)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
