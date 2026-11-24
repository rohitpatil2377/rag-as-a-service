"""A terse request is re-framed as an explicit instruction before a second generation attempt.

Measured live against sarvam-105b on a real SOP corpus, retrieval confidence 0.92 in every case:

    "detailed summary analysis for all the 10 SOPs"          -> "I don't know."  (abstain)
    "Provide a detailed summary analysis for all the 10 SOPs" -> grounded, 2 citations
    "Give a detailed summary analysis for all the 10 SOPs"    -> grounded, 2 citations

The sources were identical. What changed was that the query read as an instruction rather than
a bare noun phrase. This module only decides HOW to re-frame; whether a re-frame is attempted at
all is the generation service's call, and it only does so when retrieval was strong -- a refusal
on weak retrieval is correct and must stand.
"""

from __future__ import annotations

import pytest

from rag_core.grounding import as_explicit_request


@pytest.mark.parametrize(
    "query",
    [
        "detailed summary analysis for all the 10 SOPs",
        "the escalation matrix for priority incidents",
        "revision history of PBSOP006",
    ],
)
def test_a_bare_noun_phrase_becomes_an_instruction(query: str) -> None:
    out = as_explicit_request(query)
    assert out is not None
    assert out.lower().startswith("provide")
    # The subject must survive verbatim -- re-framing may not quietly change what was asked.
    assert query.rstrip("?.") in out


@pytest.mark.parametrize(
    "query",
    [
        "What is the retention period?",
        "how do I file an incident report",
        "list the SOPs",
        "explain PBSOP006",
        "summarize the escalation matrix",
        "Provide a detailed summary of the SOPs",
        "compare PBSOP001 and PBSOP002",
        "is the policy still current?",
    ],
)
def test_a_question_or_instruction_is_left_alone(query: str) -> None:
    """Already explicit: re-framing these would add nothing and risk changing their meaning."""
    assert as_explicit_request(query) is None


def test_an_empty_query_is_left_alone() -> None:
    assert as_explicit_request("   ") is None
