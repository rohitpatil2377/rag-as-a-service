#!/usr/bin/env python
"""Federated RAG federation QA — proves the cross-institution federation lifecycle end-to-end
against a LIVE stack: issue a signed grant -> real cross-department retrieval -> a
tampered/forged grant is rejected -> revoking the grant fails the SAME query closed.

This is the platform's differentiating claim (decentralised, cryptographically-authorised,
revocable P2P federation — "Federated RAG") made QA-runnable: every assertion is checked
against the live HTTP API (and, where available, the sovereign-node's own control API), with
a clear PASS/FAIL/SKIP per step and a non-zero exit on any real failure. See
docs/qa/FEDERATION_QA.md for the full write-up (what each step proves, and how to bring up a
two-node mesh if this deployment doesn't already have one).

Style follows scripts/benchmark.py: httpx + argparse, dev-token auth, no framework.

Steps:
  (a) issue_grant()     — an Ed25519-signed grant, issuer dept -> subject dept, scoped to a
                          throwaway KB + doc created for this run (never touches existing KBs).
  (b) federated_ask()   — the subject department asks a question the grant should be able to
                          answer; asserts a GROUNDED answer with a citation whose doc_id is
                          prefixed by the issuer department and whose quote carries this run's
                          unique marker.
  (c) tamper_and_forge() — a tampered claim (same signature) and a forged signature (different
                          key) must BOTH fail Ed25519 verification — the exact function
                          svc_control_plane.grants.verify_grant the live API and the Go
                          sovereign-node both use. Where a sovereign-node container is
                          reachable via `docker exec`, the SAME three grants are also replayed
                          against its real local control API (POST /federate) for a genuine,
                          live, cross-process, cross-language (Python-signed, Go-verified)
                          wire-level proof — otherwise this half is honestly reported SKIPPED,
                          not faked.
  (d) revoke_and_reask() — revoke the grant, re-run the exact same question, and assert the
                          issuer's evidence is now GONE (no citation, no marker) — fail-closed,
                          immediately, no re-login/propagation delay.

Usage (stack up):
    uv run python scripts/federation_qa.py
    uv run python scripts/federation_qa.py --base-url http://localhost:8000 --keep
    uv run python scripts/federation_qa.py --issuer-dept dept-c --subject-dept dept-a
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

HTTP_OK = 200
HTTP_NO_CONTENT = 204
HTTP_FORBIDDEN = 403
ASK_RETRIES = 5
ASK_RETRY_DELAY_S = 4.0
INGEST_SETTLE_S = 2.0  # dense/sparse indexing after /ingest returns is not guaranteed instantaneous
QUESTION = "What is the confirmation phrase for this federation QA run?"

# A plain-English "Adjective Noun 1234" marker, not a hex code: the platform's own NLI
# faithfulness checker (E4, an uncalibrated small multilingual model — see docs/adr/0005)
# scores short alphanumeric-code sentences unreliably (a known, documented, out-of-scope
# limitation), which has nothing to do with what this script proves. A natural-language
# phrase is unambiguously verifiable and keeps the test on-target: federation, not NLI.
_ADJECTIVES = ["Golden", "Silver", "Crimson", "Azure", "Emerald", "Violet", "Amber", "Coral", "Ivory", "Obsidian"]
_NOUNS = ["Falcon", "Harbor", "Lagoon", "Meadow", "Comet", "Beacon", "Orchard", "Summit", "Anchor", "Horizon"]


def marker_phrase(run_id: str) -> str:
    n = int(run_id, 16)
    adj = _ADJECTIVES[n % len(_ADJECTIVES)]
    noun = _NOUNS[(n // len(_ADJECTIVES)) % len(_NOUNS)]
    return f"{adj} {noun} {n % 10000}"


# --- pretty PASS/FAIL/SKIP reporting -----------------------------------------------------


@dataclass
class StepResult:
    name: str
    ok: bool
    skipped: bool = False
    detail: str = ""


class Report:
    def __init__(self) -> None:
        self.results: list[StepResult] = []

    def record(self, name: str, ok: bool, detail: str = "", *, skipped: bool = False) -> None:
        self.results.append(StepResult(name, ok, skipped, detail))
        mark = "SKIP" if skipped else ("PASS" if ok else "FAIL")
        symbol = "✓" if (ok or skipped) else "✗"
        line = f"  [{mark}] {symbol} {name}"
        if detail:
            line += f" — {detail}"
        print(line)

    @property
    def hard_failed(self) -> bool:
        return any((not r.ok) and (not r.skipped) for r in self.results)

    def summary(self) -> None:
        header("SUMMARY")
        for r in self.results:
            mark = "SKIP" if r.skipped else ("PASS" if r.ok else "FAIL")
            print(f"  {mark:5s}  {r.name}")
        n_pass = sum(1 for r in self.results if r.ok and not r.skipped)
        n_fail = sum(1 for r in self.results if not r.ok and not r.skipped)
        n_skip = sum(1 for r in self.results if r.skipped)
        print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped.")
        if self.hard_failed:
            print("\n  RESULT: FAIL — the federation lifecycle is NOT fully proven on this deployment.")
        else:
            print("\n  RESULT: PASS — the Federated RAG federation lifecycle is proven end-to-end, live.")


def header(title: str) -> None:
    print()
    print("=" * 82)
    print(title)
    print("=" * 82)


class FederationDisabledError(RuntimeError):
    """Raised when the deployment has RAG_FEDERATION_ENABLED off (surface closed at 403)."""


# --- HTTP plumbing (retried: this box has been observed to bounce the api container) ------


def request(client: httpx.Client, method: str, url: str, **kw: Any) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(4):
        try:
            return client.request(method, url, **kw)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise last


def dev_token(client: httpx.Client, base_url: str, *, subject: str, tenant_id: str, roles: list[str]) -> str:
    r = request(
        client,
        "POST",
        f"{base_url}/v1/dev/token",
        json={"subject": subject, "tenant_id": tenant_id, "roles": roles, "groups": [], "clearance": 3},
    )
    r.raise_for_status()
    return str(r.json()["access_token"])


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def raise_if_federation_disabled(r: httpx.Response) -> None:
    if r.status_code == HTTP_FORBIDDEN and "federation" in r.text.lower():
        raise FederationDisabledError(r.text)


# --- crypto-level grant construction (the exact code the live API + Go node both run) ------
# Imported lazily so a clear ImportError points at `uv sync --all-packages`, matching the
# other scripts/*.py that import rag_* packages straight from the workspace.


def build_signed_grant(
    *,
    issuer_dept: str,
    subject_dept: str,
    kb_ids: list[str],
    topics: list[str],
    grant_id: str,
    max_results_per_query: int = 5,
) -> Any:
    from svc_control_plane.grants import sign_grant

    from rag_schemas import Grant

    now = datetime.now(UTC).replace(microsecond=0)  # whole-second: Go/Python interop (see grants.py)
    g = Grant(
        grant_id=grant_id,
        issuer_dept_id=issuer_dept,
        subject_dept_id=subject_dept,
        topics=topics,
        kb_ids=kb_ids,
        max_results_per_query=max_results_per_query,
        not_before=now,
        expires_at=now + timedelta(days=1),
        signed_token="",
    )
    return g.model_copy(update={"signed_token": sign_grant(g)})


# --- sovereign-node control-API wire test (best-effort; needs docker + the mesh containers) -


def find_node_container(dept: str) -> str | None:
    """Best-effort discovery of the dept's sovereign-node container (compose service
    ``node-<letter>``, e.g. dept-a -> a node container name containing ``node-a``)."""
    if shutil.which("docker") is None:
        return None
    suffix = dept.rsplit("-", 1)[-1]
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    candidates = [n for n in out.splitlines() if f"node-{suffix}" in n]
    return candidates[0] if candidates else None


def control_federate(container: str, *, peer_dept_id: str, query_text: str, top_k: int, grant: Any) -> dict[str, Any]:
    """POST the node's LOCAL /federate control endpoint via `docker exec` (it binds loopback
    inside the container only, by design — see sovereign-node/node/control.go). Returns the
    parsed FederatedQueryResponse: {"spans": [...] | null, "peer_dept_id": ..., "grant_valid":
    bool, "error": "..."?}."""
    payload = json.dumps(
        {
            "peer_dept_id": peer_dept_id,
            "query_text": query_text,
            "top_k": top_k,
            "grant": grant.model_dump(mode="json"),
        }
    )
    cmd = [
        "docker",
        "exec",
        "-i",
        container,
        "curl",
        "-s",
        "-X",
        "POST",
        "http://127.0.0.1:7070/federate",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-",
    ]
    proc = subprocess.run(cmd, input=payload, capture_output=True, text=True, timeout=20, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"control API call failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}")
    return dict(json.loads(proc.stdout))


# --- step (a): issue -----------------------------------------------------------------------


def step_issue_grant(
    client: httpx.Client,
    args: argparse.Namespace,
    *,
    marker: str,
    topic: str,
    run_id: str,
    issuer_token: str,
    report: Report,
) -> tuple[str | None, str | None, str]:
    """Provision a throwaway KB + marker document under the issuer, then issue a signed grant
    to the subject. Returns (temp_kb_id, grant_id, temp_kb_name) — both None on failure."""
    name = "grant issued (POST /v1/grants, 201, Ed25519-signed by the issuer)"
    temp_kb_name = f"FedQA Temp {run_id}"
    try:
        r = request(
            client,
            "POST",
            f"{args.base_url}/v1/knowledge-bases",
            headers=auth(issuer_token),
            json={"display_name": temp_kb_name},
        )
        raise_if_federation_disabled(r)
        r.raise_for_status()
        temp_kb_id = str(r.json()["kb_id"])

        doc_text = f"For this federation QA run, the confirmation phrase is {marker}."
        r = request(
            client,
            "POST",
            f"{args.base_url}/v1/kb/{temp_kb_id}/ingest",
            headers=auth(issuer_token),
            json={
                "doc_id": f"fedqa-marker-{run_id}",
                "mime": "text/plain",
                "content_b64": base64.b64encode(doc_text.encode()).decode(),
            },
        )
        r.raise_for_status()
        chunk_count = r.json().get("chunk_count", 0)

        r = request(
            client,
            "POST",
            f"{args.base_url}/v1/grants",
            headers=auth(issuer_token),
            json={
                "subject_dept_id": args.subject_dept,
                "topics": [topic],
                "kb_ids": [temp_kb_id],
                "max_results_per_query": 5,
                "ttl_days": 1,
            },
        )
        raise_if_federation_disabled(r)
        r.raise_for_status()
        grant = r.json()
        grant_id = str(grant["grant_id"])
        report.record(
            name,
            True,
            f"grant_id={grant_id} {args.issuer_dept}->{args.subject_dept} kb_ids={grant['kb_ids']} "
            f"expires_at={grant['expires_at']} (temp KB {temp_kb_id}, {chunk_count} chunk ingested)",
        )
    except FederationDisabledError as exc:
        report.record(name, False, f"FEDERATION IS DISABLED on this deployment (RAG_FEDERATION_ENABLED not set): {exc}")
        return None, None, temp_kb_name
    except httpx.HTTPError as exc:
        report.record(name, False, str(exc))
        return None, None, temp_kb_name
    return temp_kb_id, grant_id, temp_kb_name


# --- step (b): federated ask -----------------------------------------------------------------


def ask_body(*, query: str, subject_kb_ids: list[str], issuer_dept: str) -> dict[str, Any]:
    """The subject's own scope is pinned to ITS OWN KB(s) discovered in Discovery — never the
    default "ALL owned KBs", which silently pulls in the shared public__company-wide KB.
    That KB's unrelated multi-doc content dilutes the LLM's source numbering on a federated
    (multi-source) prompt: live-observed on this stack, the model would occasionally attach its
    citation to the WRONG numbered source, and the faithfulness checker (correctly) then scores
    that mis-citation ~0 — an own-KB-selection artifact of this run's setup, not a federation
    defect. Pinning kb_ids keeps the test on-target."""
    body: dict[str, Any] = {"query": query, "federate_from": [issuer_dept]}
    if subject_kb_ids:
        body["kb_ids"] = subject_kb_ids
    return body


def step_federated_ask(
    client: httpx.Client,
    args: argparse.Namespace,
    *,
    marker: str,
    subject_token: str,
    subject_kb_ids: list[str],
    report: Report,
) -> None:
    name = f"federated ask ({args.subject_dept} asks, federate_from=[{args.issuer_dept}])"
    answer: dict[str, Any] = {}
    time.sleep(INGEST_SETTLE_S)  # let the issuer's just-ingested doc finish dense/sparse indexing
    for _ in range(ASK_RETRIES):
        r = request(
            client,
            "POST",
            f"{args.base_url}/v1/ask",
            headers=auth(subject_token),
            json=ask_body(query=QUESTION, subject_kb_ids=subject_kb_ids, issuer_dept=args.issuer_dept),
        )
        r.raise_for_status()
        answer = r.json()
        if answer.get("status") == "grounded" and marker in json.dumps(answer.get("citations", [])):
            break
        time.sleep(ASK_RETRY_DELAY_S)
    citations = answer.get("citations", [])
    prefixed = [c for c in citations if c.get("doc_id", "").startswith(f"{args.issuer_dept}/")]
    reached = (answer.get("federation_coverage") or {}).get("reached", [])
    ok = (
        answer.get("status") == "grounded"
        and bool(prefixed)
        and any(marker in c.get("quote", "") for c in prefixed)
        and args.issuer_dept in reached
    )
    report.record(
        name,
        ok,
        f"status={answer.get('status')} reached={reached} "
        f"citation_doc_ids={[c.get('doc_id') for c in citations]} answer={answer.get('answer', '')[:120]!r}",
    )


# --- step (c): tamper / forge -----------------------------------------------------------------


def step_tamper_and_forge(
    args: argparse.Namespace, *, marker: str, topic: str, run_id: str, temp_kb_id: str | None, report: Report
) -> tuple[Any, Any, Any]:
    """Crypto-level proof (always runs): a genuine signature verifies; a tampered claim and a
    forged signature both fail. Returns (valid, tampered, forged) grant objects for reuse by
    the wire-level check, or (None, None, None) on failure."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from svc_control_plane.grants import _canonical_claims, verify_grant

        kb_for_wire = temp_kb_id or f"{args.issuer_dept}__nonexistent-fedqa-probe"
        valid = build_signed_grant(
            issuer_dept=args.issuer_dept,
            subject_dept=args.subject_dept,
            kb_ids=[kb_for_wire],
            topics=[topic],
            grant_id=f"grant_fedqa_ctrl_{run_id}",
        )
        tampered = valid.model_copy(update={"max_results_per_query": 20})
        forged_sig = base64.b64encode(Ed25519PrivateKey.generate().sign(_canonical_claims(valid))).decode()
        forged = valid.model_copy(update={"signed_token": forged_sig})

        report.record(
            "crypto-level: a genuinely-signed grant verifies (control)",
            verify_grant(valid),
            "svc_control_plane.grants.verify_grant(valid_grant) == True",
        )
        report.record(
            "crypto-level: a tampered claim (same signature) is rejected",
            not verify_grant(tampered),
            "widened max_results_per_query 5->20, kept the original signature -> verify_grant() == False",
        )
        report.record(
            "crypto-level: a forged signature (attacker's own key) is rejected",
            not verify_grant(forged),
            "identical claims, signed with a fresh unrelated Ed25519 key -> verify_grant() == False",
        )
    except Exception as exc:
        report.record("crypto-level tamper/forge verification", False, f"{type(exc).__name__}: {exc}")
        return None, None, None
    return valid, tampered, forged


def _skip_wire_check(report: Report, reason: str) -> None:
    report.record(
        "wire-level: sovereign-node control API replay (real libp2p dial, Go-side verify)",
        False,
        reason,
        skipped=True,
    )


def step_wire_check(
    args: argparse.Namespace, *, marker: str, valid: Any, tampered: Any, forged: Any, report: Report
) -> None:
    if args.skip_wire_check:
        _skip_wire_check(report, "SKIPPED (--skip-wire-check)")
        return
    if valid is None or tampered is None or forged is None:
        _skip_wire_check(report, "crypto-level grant construction failed above; nothing to replay")
        return
    container = args.node_container or find_node_container(args.subject_dept)
    if container is None:
        _skip_wire_check(
            report,
            f"no reachable sovereign-node container for {args.subject_dept} (needs `docker` + the mesh's "
            "node-* containers up — see docs/qa/FEDERATION_QA.md § bringing up the mesh)",
        )
        return
    try:
        query_text = "federation QA marker code"
        v = control_federate(container, peer_dept_id=args.issuer_dept, query_text=query_text, top_k=3, grant=valid)
        t = control_federate(container, peer_dept_id=args.issuer_dept, query_text=query_text, top_k=3, grant=tampered)
        f_ = control_federate(container, peer_dept_id=args.issuer_dept, query_text=query_text, top_k=3, grant=forged)
    except Exception as exc:
        report.record(
            "wire-level: sovereign-node control API replay (real libp2p dial, Go-side verify)",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return
    wire_valid_ok = bool(v.get("grant_valid")) and any(marker in (s.get("text") or "") for s in (v.get("spans") or []))
    report.record(
        f"wire-level: valid grant accepted by {container} -> real dial to {args.issuer_dept}, real spans",
        wire_valid_ok,
        f"grant_valid={v.get('grant_valid')} spans={[s.get('chunk_id') for s in (v.get('spans') or [])]}",
    )
    report.record(
        "wire-level: tampered grant rejected by the remote responder",
        not t.get("grant_valid"),
        f"grant_valid={t.get('grant_valid')}",
    )
    report.record(
        "wire-level: forged grant rejected by the remote responder",
        not f_.get("grant_valid"),
        f"grant_valid={f_.get('grant_valid')}",
    )


# --- step (d): revoke + reask -----------------------------------------------------------------


def step_revoke_and_reask(
    client: httpx.Client,
    args: argparse.Namespace,
    *,
    marker: str,
    grant_id: str | None,
    issuer_token: str,
    subject_token: str,
    subject_kb_ids: list[str],
    report: Report,
) -> None:
    revoke_name = "grant revoked (DELETE /v1/grants/{id} -> 204)"
    leak_name = "post-revocation: no peer data leaks (fail-closed, immediate, no re-login)"
    if grant_id is None:
        report.record(revoke_name, False, "SKIPPED", skipped=True)
        report.record(leak_name, False, "SKIPPED", skipped=True)
        return

    r = request(client, "DELETE", f"{args.base_url}/v1/grants/{grant_id}", headers=auth(issuer_token))
    revoked_ok = r.status_code == HTTP_NO_CONTENT
    report.record(revoke_name, revoked_ok, f"status={r.status_code}")
    if not revoked_ok:
        report.record(leak_name, False, "SKIPPED (revoke itself failed above)", skipped=True)
        return

    r = request(
        client,
        "POST",
        f"{args.base_url}/v1/ask",
        headers=auth(subject_token),
        json=ask_body(query=QUESTION, subject_kb_ids=subject_kb_ids, issuer_dept=args.issuer_dept),
    )
    r.raise_for_status()
    answer = r.json()
    citations = answer.get("citations", [])
    reached = (answer.get("federation_coverage") or {}).get("reached", [])
    no_leak = (
        args.issuer_dept not in reached
        and marker not in json.dumps(citations)
        and marker not in answer.get("answer", "")
    )
    report.record(
        leak_name,
        no_leak,
        f"status={answer.get('status')} reached={reached} citation_doc_ids={[c.get('doc_id') for c in citations]}",
    )


# --- discovery + cleanup -----------------------------------------------------------------


def owned_kbs(client: httpx.Client, base_url: str, token: str) -> list[str]:
    r = request(client, "GET", f"{base_url}/v1/knowledge-bases", headers=auth(token))
    r.raise_for_status()
    return [k["kb_id"] for k in r.json() if k["kb_id"] != "public__company-wide"]


def cleanup(
    client: httpx.Client, args: argparse.Namespace, *, temp_kb_id: str | None, temp_kb_name: str, issuer_token: str
) -> None:
    header("Cleanup")
    if args.keep:
        print(f"  --keep set: leaving {temp_kb_id or '(nothing was created)'} and the (now revoked) grant in place.")
        return
    if temp_kb_id is None:
        print("  nothing to clean up.")
        return
    r = request(
        client,
        "POST",
        f"{args.base_url}/v1/knowledge-bases/{temp_kb_id}/purge",
        headers=auth(issuer_token),
        json={"confirm_name": temp_kb_name},
    )
    if r.status_code == HTTP_OK:
        print(f"  purged temp KB {temp_kb_id} (permanently deleted, {r.json()})")
        return
    # Fall back to a soft-delete (trash) so nothing lingers as an ACTIVE KB even if purge is
    # gated (e.g. dual-control) on this deployment.
    r2 = request(client, "DELETE", f"{args.base_url}/v1/knowledge-bases/{temp_kb_id}", headers=auth(issuer_token))
    print(
        f"  purge returned {r.status_code} ({r.text[:150]}); trashed instead "
        f"(DELETE -> {r2.status_code}) — restorable, but no longer active."
    )


# --- entrypoint -----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000", help="The SUBJECT department's own api instance.")
    p.add_argument("--issuer-dept", default="dept-b", help="Department that owns the KB and issues the grant.")
    p.add_argument("--subject-dept", default="dept-a", help="Department granted access; the one that asks.")
    p.add_argument("--keep", action="store_true", help="Do not clean up the temp KB/doc this run creates.")
    p.add_argument(
        "--skip-wire-check",
        action="store_true",
        help="Skip the docker-exec sovereign-node control-API wire test even if a node container is reachable.",
    )
    p.add_argument(
        "--node-container", default=None, help="Override auto-detection of the subject dept's node container."
    )
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout (seconds).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = uuid.uuid4().hex[:8]
    marker = marker_phrase(run_id)
    topic = f"fedqa-topic-{run_id}"
    report = Report()

    header("Federated RAG federation QA")
    print(f"  base-url:     {args.base_url}")
    print(f"  issuer dept:  {args.issuer_dept}  (owns the KB, issues/revokes the grant)")
    print(f"  subject dept: {args.subject_dept}  (receives the grant, asks the question)")
    print(f"  run marker:   {marker}")

    client = httpx.Client(timeout=args.timeout)

    header("Discovery")
    try:
        subject_token = dev_token(
            client, args.base_url, subject="fedqa", tenant_id=args.subject_dept, roles=["owner", "admin", "reader"]
        )
        issuer_token = dev_token(
            client, args.base_url, subject="fedqa", tenant_id=args.issuer_dept, roles=["owner", "admin", "ingestor"]
        )
    except httpx.HTTPError as exc:
        print(f"  could not mint dev tokens from {args.base_url}: {exc}")
        print("  (this script needs a dev-JWT-mode deployment; see docs/qa/FEDERATION_QA.md)")
        return 2
    print(f"  minted dev tokens for {args.subject_dept} and {args.issuer_dept} against {args.base_url}")

    subject_kbs = owned_kbs(client, args.base_url, subject_token)
    issuer_kbs = owned_kbs(client, args.base_url, issuer_token)
    print(f"  {args.subject_dept} owns: {subject_kbs or '(none — only the shared KB)'}")
    print(f"  {args.issuer_dept} owns:  {issuer_kbs or '(none — only the shared KB)'}")
    print(
        "  -> this run creates its OWN throwaway KB + document under "
        f"{args.issuer_dept} for the grant, and never touches the KBs listed above."
    )

    header("(a) Issue an Ed25519-signed grant")
    temp_kb_id, grant_id, temp_kb_name = step_issue_grant(
        client, args, marker=marker, topic=topic, run_id=run_id, issuer_token=issuer_token, report=report
    )

    header("(b) Cross-department federated ask returns real citations")
    if grant_id and temp_kb_id:
        step_federated_ask(
            client, args, marker=marker, subject_token=subject_token, subject_kb_ids=subject_kbs, report=report
        )
    else:
        report.record(
            f"federated ask ({args.subject_dept} asks, federate_from=[{args.issuer_dept}])",
            False,
            "no grant to test (step (a) did not produce one — see above)",
            skipped=True,
        )

    header("(c) A tampered or forged grant is REJECTED (fail-closed signature enforcement)")
    valid, tampered, forged = step_tamper_and_forge(
        args, marker=marker, topic=topic, run_id=run_id, temp_kb_id=temp_kb_id, report=report
    )
    step_wire_check(args, marker=marker, valid=valid, tampered=tampered, forged=forged, report=report)

    header("(d) Revoke the grant -> the SAME federated ask now fails closed")
    step_revoke_and_reask(
        client,
        args,
        marker=marker,
        grant_id=grant_id,
        issuer_token=issuer_token,
        subject_token=subject_token,
        subject_kb_ids=subject_kbs,
        report=report,
    )

    cleanup(client, args, temp_kb_id=temp_kb_id, temp_kb_name=temp_kb_name, issuer_token=issuer_token)

    report.summary()
    return 1 if report.hard_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
