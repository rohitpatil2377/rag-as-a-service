"""The prompt-file loader (RAG v2; discovery §8.5 scopes it as its own work).

Small, but it guards a failure mode no other test here can see: a prompt that is read off the
filesystem relative to the source checkout works in CI and is simply absent from the wheel the
API ships as, so the first profile call in a deployment fails on a missing file. These tests go
through ``importlib.resources``, which is the same mechanism the packaged build uses.
"""

from __future__ import annotations

import pytest

from rag_core.prompts import load_prompt, render_prompt


def test_the_doc_profile_prompt_is_packaged_with_the_code_that_reads_it() -> None:
    text = load_prompt("doc_profile")
    assert "facet_taxonomy" in text
    assert "$head_text" in text  # still a template; substitution is render_prompt's job


def test_an_unknown_prompt_raises_rather_than_returning_an_empty_instruction() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("no_such_prompt")


def test_a_placeholder_the_caller_forgot_raises_here_not_at_the_model() -> None:
    with pytest.raises(KeyError):
        render_prompt("doc_profile", title="t")


def test_the_loader_is_cached_so_a_corpus_reads_the_file_once() -> None:
    assert load_prompt("doc_profile") is load_prompt("doc_profile")


def test_the_section_enrich_prompt_is_packaged_too() -> None:
    """Phase 4's prompt ships from inside the package for the same packaging reason — see the
    module docstring. One call per SECTION means a missing file fails a whole corpus, not one
    document."""
    text = load_prompt("section_enrich")
    assert "propositions" in text and "$section_text" in text


def test_a_dollar_sign_in_a_document_is_not_read_as_a_placeholder() -> None:
    """`string.Template` scans the TEMPLATE for placeholders, never the substituted values, so a
    section that happens to contain "$100" or "$facet_ids" cannot inject one."""
    rendered = render_prompt(
        "section_enrich",
        profile_json="{}",
        facet_ids="- a",
        breadcrumb="1. Costs",
        section_text="The fee is $100 and the $facet_ids column is blank.",
    )
    assert "$100" in rendered and "the $facet_ids column" in rendered
