"""E2E: the section that scopes a rule survives ingest → retrieval → the prompt (P15).

QA Q10 applied a rule from "6.2.3 Social Media Content" to a sales deck. Read on its own the
rule IS general — the limit lives in the heading, which is a different chunk. This proves the
heading now travels with the rule all the way to the generator.
"""

from __future__ import annotations

import base64

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _body(subject: str, n: int) -> str:
    """Section bodies long enough that a chunk lands INSIDE one, as in a real handbook. A
    toy-length section makes the whole document one chunk, which genuinely belongs to no single
    heading and would prove nothing either way."""
    return " ".join(
        f"Paragraph {i} of this clause sets out the procedure for {subject} and the records "
        f"that must be retained by the responsible team for audit purposes."
        for i in range(n)
    )


_HANDBOOK = (
    "Acme Content Governance Master SOP\n\n"
    "6.2.3 Social Media Content\n"
    "All content must be approved by the Communications team before it is released. "
    + _body("social media posts", 14)
    + "\n\n6.5.1 Customer Collateral\n"
    "Sales decks may be shared with a customer directly by the account owner. "
    + _body("customer collateral", 14)
    + "\n"
)


def _client(kb: str) -> httpx.AsyncClient:
    from rag_api.app import create_app
    from rag_common import issue_dev_jwt

    token = issue_dev_jwt("o", kb, roles=["reader", "ingestor"])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://t",
        headers={"Authorization": f"Bearer {token}"},
    )


async def _ingest(c: httpx.AsyncClient, kb: str) -> None:
    await c.post("/v1/knowledge-bases", json={"display_name": "KB"})
    body = base64.b64encode(_HANDBOOK.encode()).decode()
    r = await c.post(
        f"/v1/kb/{kb}/ingest", json={"doc_id": "sop", "mime": "text/plain", "content_b64": body, "strategy": "clause"}
    )
    assert r.status_code == 200, r.text


async def test_a_retrieved_rule_carries_the_section_that_governs_it(inmemory: None) -> None:
    kb = "scopeco"
    async with _client(kb) as c:
        await _ingest(c, kb)
        spans = (
            await c.post(f"/v1/kb/{kb}/retrieve", json={"query": "procedure and records retained", "top_k": 20})
        ).json()["spans"]
        labelled = [s for s in spans if s["section"]]
        assert labelled, "no retrieved span carried a section at all"

        # Every label must be TRUE of the span carrying it: a span labelled 6.2.3 talks about
        # social media, one labelled 6.5.1 about collateral. A label that is merely present is
        # worthless; a label that is wrong is the QA Q10 failure with extra confidence.
        for s in labelled:
            subject = "social media posts" if s["section"].startswith("6.2.3") else "customer collateral"
            assert subject in s["text"], f"{s['section']} stamped on other text: {s['text'][:120]}"

        assert {s["section"] for s in labelled} == {"6.2.3 Social Media Content", "6.5.1 Customer Collateral"}


async def test_a_chunk_that_straddles_two_headings_claims_neither(inmemory: None) -> None:
    """The honest half. A chunk holding the tail of one section and the head of the next is
    governed by no single heading, and stamping it with the one it began under would authorise
    exactly the over-application P15 exists to stop. Absent, not guessed."""
    kb = "scopeco2"
    async with _client(kb) as c:
        await _ingest(c, kb)
        chunks = (await c.get(f"/v1/kb/{kb}/documents/sop/content")).json()["chunks"]
        straddling = [
            ch for ch in chunks if "6.5.1 Customer Collateral" in ch["text"] and "social media posts" in ch["text"]
        ]
        assert straddling, "fixture no longer produces a straddling chunk — the assertion is vacuous"
        for ch in straddling:
            assert not ch.get("metadata", {}).get("section")
