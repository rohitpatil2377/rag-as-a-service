"""Walking-skeleton demo (`make demo`) — the whole architecture in one run.

Drives the REAL modular-monolith API in-process via an ASGI transport (no network,
no docker): mint dev JWT -> create KB -> ingest a sample PDF (parse + PII-mask +
chunk + embed + index) -> ask a grounded question (hybrid retrieve + rerank +
grounded-or-abstain + citations) -> then ask an unanswerable question to show the
abstain gate.

Backends: in-memory by default (`--in-memory`) so this runs anywhere; the SAME code
path runs against the sovereign stack (Qdrant/OpenSearch/MinIO/Postgres/TEI) when
`make up` is running and `--in-memory` is omitted.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
from pathlib import Path
from typing import Any

_SAMPLE_MD = Path(__file__).resolve().parents[4] / "sample-data" / "platform_brief.md"


def _build_sample_pdf(md_path: Path) -> bytes:
    """Render the markdown brief to a real PDF so the pypdf parse path is exercised."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for raw in md_path.read_text(encoding="utf-8").splitlines():
        line = raw.lstrip("# ").strip()
        if not line:
            pdf.ln(4)
            continue
        pdf.multi_cell(
            w=pdf.epw,
            h=8,
            text=line.encode("latin-1", "replace").decode("latin-1"),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
    return bytes(pdf.output())


def _banner(text: str) -> None:
    print(f"\n\033[38;5;208m{'─' * 68}\n  {text}\n{'─' * 68}\033[0m")  # Acme saffron rule


async def _run(tenant: str, question: str) -> int:
    # Import AFTER env is set so cached settings pick up the in-memory profile.
    import httpx

    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    app = create_app()
    # The demo drives the WHOLE lifecycle, which starts by creating a knowledge base — and KB
    # creation is owner/admin-only (api.py `create_kb`). With only reader+ingestor this script
    # died on its first call with a bare 403, so the roles have to cover every step it performs.
    token = issue_dev_jwt(
        "demo-officer",
        tenant,
        roles=["owner", "admin", "ingestor", "reviewer", "reader"],
        attrs={"grade": "SAO"},
    )
    headers = {"Authorization": f"Bearer {token}"}
    pdf_b64 = base64.b64encode(_build_sample_pdf(_SAMPLE_MD)).decode()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://demo", headers=headers, timeout=60) as c:
        _banner("1 · Create knowledge base (tenant namespace)")
        r = await c.post("/v1/knowledge-bases", json={"display_name": "Demo KB"})
        r.raise_for_status()
        print(f"  KB ready: {r.json()}")

        _banner("2 · Ingest sample PDF  (parse → PII-mask → chunk → embed → index)")
        r = await c.post(
            f"/v1/kb/{tenant}/ingest",
            json={
                "doc_id": "platform_brief",
                "mime": "application/pdf",
                "content_b64": pdf_b64,
                "strategy": "clause",
            },
        )
        r.raise_for_status()
        ing = r.json()
        print(f"  chunks={ing['chunk_count']}  pii_masked={ing['pii_masked']}  trace={ing['trace_id'][:8]}")

        _banner(f"3 · Grounded query:  “{question}”")
        r = await c.post(f"/v1/kb/{tenant}/query", json={"query": question, "top_k": 3})
        r.raise_for_status()
        _print_answer(r.json())

        _banner("4 · Abstain gate:  “What is the boiling point of helium on Jupiter?”")
        r = await c.post(f"/v1/kb/{tenant}/query", json={"query": "What is the boiling point of helium on Jupiter?"})
        r.raise_for_status()
        _print_answer(r.json())

    print("\n\033[38;5;28m✓ walking skeleton complete — grounded, cited, and abstaining as designed.\033[0m\n")
    return 0


def _print_answer(ans: dict[str, Any]) -> None:
    status = ans["status"].upper()
    colour = "28" if status == "GROUNDED" else "208"
    print(f"  status={status}  confidence={ans['confidence']:.3f}")
    print(f"  answer: {ans['answer']}")
    for i, cit in enumerate(ans.get("citations", []), start=1):
        page = f" p.{cit['page']}" if cit.get("page") else ""
        print(f"    \033[38;5;{colour}m[{i}]\033[0m {cit['doc_id']}{page}  “{cit['quote'][:80]}…”")
    if ans.get("abstain_reason"):
        print(f"  abstain_reason: {ans['abstain_reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG as a Service walking-skeleton demo")
    parser.add_argument("--ask", default="What is the platform's uptime target?", help="question to ask")
    parser.add_argument("--tenant", default="demo", help="tenant / knowledge-base id")
    parser.add_argument("--seed", action="store_true", help="(kept for Makefile compatibility)")
    parser.add_argument("--stores", action="store_true", help="use the real sovereign stack instead of in-memory")
    args = parser.parse_args()

    # 12-factor: configure the process via env before settings are first read.
    os.environ.setdefault("RAG_ENV", "dev")
    os.environ.setdefault("RAG_OTEL_ENABLED", "false")
    os.environ.setdefault("RAG_LOG_JSON", "false")
    if not args.stores:
        os.environ["RAG_INMEMORY"] = "true"

    raise SystemExit(asyncio.run(_run(args.tenant, args.ask)))


if __name__ == "__main__":
    main()
