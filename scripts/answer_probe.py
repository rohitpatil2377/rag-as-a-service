"""Eyeball answer quality: ask a list of questions and show what was retrieved.

Runs each question through the REAL pipeline over HTTP (`/v1/kb/{kb}/retrieve` and
`/v1/kb/{kb}/query` on a running stack) — nothing about retrieval, ranking, grounding or
generation is reimplemented here. This script only drives the API and formats what comes
back, so what you read is exactly what the platform did.

For each question it prints:
  * the question
  * the final answer + status / confidence / faithfulness / abstain reason
  * RETRIEVED  — ranked spans from plain hybrid retrieval (visible even when it abstains)
  * CITED      — the chunks the answer is actually built from (the ones the LLM saw)
  * the pipeline's own agent trace (route / retry / graph / verify steps)

Read the two chunk lists as different things. `/retrieve` is plain hybrid retrieval only:
it does NOT run the adaptive router, query transformation, agentic retry, graph augment or
the recency filter, so on a heavy-path question the query may have used a different (wider)
set. CITED is authoritative for "what the LLM saw"; RETRIEVED is the honest view of what
hybrid search surfaces, and the only view you get when the answer abstains.

Usage (stack already up — see README "Quickstart"):

    # ingest a folder, then ask every question in QUESTIONS below
    uv run python scripts/answer_probe.py --folder /path/to/test-docs --fresh

    # reuse an existing KB, ask only
    uv run python scripts/answer_probe.py --kb myorg__my-kb

No metrics, no scoring — this is for reading with your own eyes.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# EDIT ME — the questions to run. One per line.
# ---------------------------------------------------------------------------
QUESTIONS: list[str] = [
    "What is the guaranteed uptime target?",
    "How quickly does priority support respond?",
    "What are the maintenance requirements?",
    "What is the boiling point of helium on Jupiter?",  # deliberately unanswerable — proves the abstain gate
]

# ---------------------------------------------------------------------------

# Extensions the ingest path knows how to parse; anything else is sent as PDF and will be
# classified by magic bytes (and routed to OCR if it turns out to be an image/scan).
_MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_SAFFRON = "\033[38;5;208m"
_GREEN = "\033[38;5;28m"
_RED = "\033[38;5;160m"


def _rule(text: str) -> None:
    print(f"\n{_SAFFRON}{'─' * 78}\n  {text}\n{'─' * 78}{_RESET}")


def _colour_for(status: str) -> str:
    return {"grounded": _GREEN, "flagged_for_review": _SAFFRON}.get(status, _RED)


def _snippet(text: str, chars: int) -> str:
    """One-line preview of a chunk — newlines collapsed so the ranking stays readable."""
    flat = " ".join(text.split())
    return flat[:chars] + ("…" if len(flat) > chars else "")


class Probe:
    """Thin driver over the running `/v1` API."""

    def __init__(self, base_url: str, token: str, timeout: float) -> None:
        self._c = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._c.close()

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        r = self._c.post(path, json=payload or {})
        if r.status_code >= httpx.codes.BAD_REQUEST:
            raise SystemExit(f"{_RED}HTTP {r.status_code} on {path}: {r.text[:400]}{_RESET}")
        return r.json() if r.content else None

    def create_kb(self, display_name: str) -> str:
        return str(self._post("/v1/knowledge-bases", {"display_name": display_name})["kb_id"])

    def ingest_folder(self, kb_id: str, folder: Path, strategy: str) -> None:
        files = sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.startswith("."))
        if not files:
            raise SystemExit(f"{_RED}no files found under {folder}{_RESET}")
        _rule(f"Ingesting {len(files)} file(s) from {folder}")
        for path in files:
            mime = _MIME_BY_EXT.get(path.suffix.lower(), "application/pdf")
            payload = {
                "doc_id": str(path.relative_to(folder)),
                "mime": mime,
                "content_b64": base64.b64encode(path.read_bytes()).decode(),
                "strategy": strategy,
            }
            try:
                res = self._post(f"/v1/kb/{kb_id}/ingest", payload)
            except SystemExit as exc:  # one unparseable file must not kill the run
                print(f"  {_RED}✗{_RESET} {path.name:<40} {exc}")
                continue
            dup = f"  duplicate_of={res['duplicate_of']}" if res.get("duplicate_of") else ""
            print(
                f"  {_GREEN}✓{_RESET} {path.name:<40} chunks={res['chunk_count']:<4} "
                f"pii_masked={res['pii_masked']:<3} {res['outcome']}{dup}"
            )

    def retrieve(self, kb_id: str, question: str, top_k: int) -> dict[str, Any]:
        return dict(self._post(f"/v1/kb/{kb_id}/retrieve", {"query": question, "top_k": top_k}))

    def query(self, kb_id: str, question: str, top_k: int) -> dict[str, Any]:
        return dict(self._post(f"/v1/kb/{kb_id}/query", {"query": question, "top_k": top_k}))


def mint_dev_token(base_url: str, tenant: str, timeout: float) -> str:
    """Dev JWT for the run (dev_jwt mode only — the endpoint is disabled in prod)."""
    r = httpx.post(
        f"{base_url.rstrip('/')}/v1/dev/token",
        json={
            "subject": "answer-probe",
            "tenant_id": tenant,
            "roles": ["owner", "admin", "reader", "ingestor", "reviewer"],
        },
        timeout=timeout,
    )
    if r.status_code != httpx.codes.OK:
        raise SystemExit(
            f"{_RED}could not mint a dev token ({r.status_code}). Pass --token explicitly, "
            f"or check RAG_AUTH_MODE=dev_jwt and RAG_ENV != prod.{_RESET}"
        )
    return str(r.json()["access_token"])


def _print_answer(ans: dict[str, Any]) -> None:
    status = str(ans["status"])
    colour = _colour_for(status)
    bits = [f"status={colour}{status.upper()}{_RESET}", f"confidence={ans['confidence']:.3f}"]
    if ans.get("faithfulness_score") is not None:
        bits.append(f"faithfulness={ans['faithfulness_score']:.2f}")
    if ans.get("model"):
        bits.append(f"model={ans['model']}")
    if ans.get("review_id"):
        bits.append(f"review_id={ans['review_id']}")
    print("  " + "  ".join(bits))
    if ans.get("abstain_reason"):
        print(f"  abstain_reason={_RED}{ans['abstain_reason']}{_RESET}")
    print(f"\n  {_BOLD}ANSWER{_RESET}")
    for line in str(ans["answer"]).splitlines() or [""]:
        print(f"    {line}")


def _print_spans(spans: list[dict[str, Any]], cited_ids: set[str], snippet_chars: int) -> None:
    if not spans:
        print(f"    {_DIM}(nothing retrieved){_RESET}")
        return
    for i, s in enumerate(spans, start=1):
        mark = f"{_GREEN}●{_RESET}" if s["chunk_id"] in cited_ids else f"{_DIM}○{_RESET}"
        page = f" p.{s['page']}" if s.get("page") else ""
        print(f"    {mark} [{i}] {_BOLD}{s['doc_id']}{page}{_RESET}  score={s['score']:.4f}  ({s['chunk_id']})")
        print(f"        {_DIM}{_snippet(s['text'], snippet_chars)}{_RESET}")


def _print_citations(citations: list[dict[str, Any]], snippet_chars: int) -> None:
    if not citations:
        print(f"    {_DIM}(no citations — the answer is not grounded){_RESET}")
        return
    for i, c in enumerate(citations, start=1):
        page = f" p.{c['page']}" if c.get("page") else ""
        print(f"    [{i}] {_BOLD}{c['doc_id']}{page}{_RESET}  score={c['score']:.4f}  ({c['chunk_id']})")
        print(f"        {_DIM}{_snippet(c['quote'], snippet_chars)}{_RESET}")


def run(args: argparse.Namespace) -> int:
    token = args.token or mint_dev_token(args.base_url, args.tenant, args.timeout)
    probe = Probe(args.base_url, token, args.timeout)
    try:
        kb_id = args.kb
        if args.fresh or kb_id is None:
            kb_id = probe.create_kb(args.kb_name)
            print(f"{_DIM}knowledge base: {kb_id}{_RESET}")
        if args.folder:
            probe.ingest_folder(kb_id, Path(args.folder).expanduser().resolve(), args.strategy)

        for question in QUESTIONS:
            _rule(question)
            retrieval = probe.retrieve(kb_id, question, args.top_k)
            answer = probe.query(kb_id, question, args.top_k)
            _print_answer(answer)

            cited = answer.get("citations") or []
            cited_ids = {c["chunk_id"] for c in cited}
            print(f"\n  {_BOLD}RETRIEVED{_RESET} {_DIM}(plain hybrid; ● = also cited in the answer){_RESET}")
            print(f"    {_DIM}mode={retrieval['mode_used']}  max_score={retrieval['max_score']:.4f}{_RESET}")
            _print_spans(retrieval.get("spans") or [], cited_ids, args.snippet)

            print(f"\n  {_BOLD}CITED{_RESET} {_DIM}(the chunks the answer was built from){_RESET}")
            _print_citations(cited, args.snippet)

            trace = answer.get("agent_trace") or []
            if trace:
                print(f"\n  {_BOLD}PIPELINE{_RESET}")
                for step in trace:
                    print(f"    {_DIM}{step['step']:<10}{_RESET} {step['detail']}")
            followups = answer.get("suggested_followups") or []
            if followups and args.show_followups:
                print(f"\n  {_BOLD}FOLLOW-UPS{_RESET}")
                for f in followups:
                    print(f"    {_DIM}{_snippet(f, 120)}{_RESET}")
        print()
        return 0
    finally:
        probe.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000", help="running API (default: %(default)s)")
    p.add_argument("--kb", default=None, help="existing knowledge-base id; omit to create one")
    p.add_argument("--fresh", action="store_true", help="always create a new KB (ignore --kb)")
    p.add_argument("--kb-name", default="Answer Probe", help="display name when creating a KB")
    p.add_argument("--tenant", default="probe", help="org/tenant for the dev token (default: %(default)s)")
    p.add_argument("--token", default=None, help="use this JWT / API key instead of minting a dev token")
    p.add_argument("--folder", default=None, help="ingest every file under this folder first (recursive)")
    p.add_argument("--strategy", default="clause", choices=["fixed", "clause"], help="chunking strategy")
    p.add_argument("--top-k", type=int, default=5, help="chunks requested per question (default: %(default)s)")
    p.add_argument("--snippet", type=int, default=220, help="chunk preview length in characters")
    p.add_argument("--timeout", type=float, default=300.0, help="per-request timeout (reasoning models are slow)")
    p.add_argument("--show-followups", action="store_true", help="also print suggested follow-up questions")
    args = p.parse_args()
    if not QUESTIONS:
        sys.exit("QUESTIONS is empty — edit the list at the top of this file.")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
