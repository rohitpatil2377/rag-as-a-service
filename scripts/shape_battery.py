#!/usr/bin/env python3
"""Shape-test battery — cheap per-KB probes that catch whole CLASSES of silent quality
regressions (the kinds that shipped unnoticed: synonym-intent gaps, multi-point answer
truncation, and unanswerable questions that fail to abstain).

Run against any KB; it exercises canonical "shapes", not specific facts, so one small suite
guards many features. Exit code is non-zero if any shape fails, so it drops into CI or a
pre-release check.

    python scripts/shape_battery.py --kb public__company-wide [--base http://localhost:8000/v1]

Each shape asserts a BEHAVIOUR, not an exact answer:
  * catalog-synonyms: "documents"/"files"/"artifacts"/"records" must all resolve to the KB
    inventory (grounded), never out-of-scope — the synonym-intent gap we fixed.
  * multi-point: a question whose answer is an enumerated set must return the WHOLE set
    (all expected markers present) — the 6->3 truncation class.
  * unanswerable: an out-of-domain question must ABSTAIN, not fabricate — the grounding gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# (name, question, kind, spec)
#   kind="not_abstain": passes if the answer is grounded (intent resolved, not out-of-scope)
#   kind="contains_all": spec = markers that must ALL appear (case-insensitive) — completeness
#   kind="abstain":      passes if the answer abstains
_SHAPES = [
    ("catalog:documents", "what documents do we have in this knowledge base", "not_abstain", None),
    ("catalog:files", "list the files in this KB", "not_abstain", None),
    ("catalog:artifacts", "list all the artifacts we currently have here", "not_abstain", None),
    ("catalog:records", "what records are stored in this knowledge base", "not_abstain", None),
    # multi-point completeness — override MARKERS via --markers for a different KB.
    ("multipoint:core-values", "What are the six core values of Acme?", "contains_all", None),
    ("unanswerable:out-of-domain", "What is the boiling point of the planet Neptune's core?", "abstain", None),
]
_DEFAULT_MARKERS = [
    "exuberant",
    "proactive",
    "continuous knowledge",
    "organization above self",
    "respect and care",
    "love for environment",
]


def _post(base: str, path: str, token: str, body: dict) -> tuple[int, dict | str]:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - operator-run against own API
            return r.status, json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
    except Exception as e:
        return 0, str(e)


def _token(base: str, kb: str) -> str:
    st, body = _post(
        base,
        "/dev/token",
        "",
        {"subject": "shape-battery", "tenant_id": kb, "roles": ["owner", "admin", "reader"], "clearance": 3},
    )
    if st != 200 or not isinstance(body, dict):
        raise SystemExit(f"could not mint token ({st}): {body}")
    return body["access_token"]


def _judge(kind: str, spec: object, status: str, answer: str) -> tuple[bool, str]:
    a = answer.lower()
    if kind == "abstain":
        return status == "abstained", f"status={status}"
    if kind == "not_abstain":
        return status == "grounded", f"status={status}"
    if kind == "contains_all":
        markers = spec if isinstance(spec, list) else _DEFAULT_MARKERS
        missing = [m for m in markers if m.lower() not in a]
        return not missing, (
            f"{len(markers) - len(missing)}/{len(markers)} markers" + (f"; missing {missing}" if missing else "")
        )
    return False, "unknown kind"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="public__company-wide")
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--markers", help="comma-separated completeness markers for the multipoint shape")
    args = ap.parse_args()
    markers = [m.strip() for m in args.markers.split(",")] if args.markers else _DEFAULT_MARKERS

    token = _token(args.base, args.kb)
    print(f"shape battery vs kb={args.kb}")
    failed = 0
    for name, question, kind, spec in _SHAPES:
        st, resp = _post(args.base, f"/kb/{args.kb}/query", token, {"query": question})
        if not isinstance(resp, dict):
            print(f"  FAIL  {name}: HTTP {st}: {resp}")
            failed += 1
            continue
        ok, detail = _judge(
            kind, markers if kind == "contains_all" else spec, resp.get("status", ""), resp.get("answer", "")
        )
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        failed += 0 if ok else 1
    print(f"{'ALL PASS' if not failed else str(failed) + ' FAILED'} ({len(_SHAPES)} shapes)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
