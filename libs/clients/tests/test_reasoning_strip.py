"""Unit test: reasoning-model chain-of-thought is stripped from gateway output."""

from __future__ import annotations

from rag_clients.llm_gateway import (
    _answer_text,
    _StreamReasoningFilter,
    _strip_reasoning,
    _strip_reasoning_tags,
)


def test_strip_think_block_keeps_only_final_answer() -> None:
    raw = "<think>\nThe user asks X. Source [1] says 3 seconds.\n</think>\nThe P95 latency target is 3 seconds [1]."
    assert _strip_reasoning(raw) == "The P95 latency target is 3 seconds [1]."


def test_plain_answer_is_unchanged() -> None:
    assert _strip_reasoning("  The target is 3 seconds [1].  ") == "The target is 3 seconds [1]."


def test_an_unclosed_think_block_yields_no_answer() -> None:
    """The docstring claimed this was handled; the code returned the whole block, so a model
    that ran out of budget mid-thought published its WORKING as the answer. Measured live: a
    reformulation call with a 200-token budget came back as 900 characters of reasoning and
    zero answer.

    Empty is the honest result — the caller sees no answer rather than a plausible one."""
    assert _strip_reasoning("<think>\nOkay, the user is asking whether the CMD can") == ""


def test_text_without_any_think_block_is_untouched() -> None:
    assert _strip_reasoning("A plain answer with no reasoning.") == "A plain answer with no reasoning."


# --- _strip_reasoning_tags: the reasoning-tag family remover ----------------------------------


def test_strip_tags_scratch_then_answer() -> None:
    assert _strip_reasoning_tags("<think>scratch</think>Answer") == "Answer"


def test_strip_tags_reasoning_only_is_empty() -> None:
    assert _strip_reasoning_tags("<think>only</think>") == ""


def test_strip_tags_plain_answer_unchanged() -> None:
    # A plain answer with no reasoning tags passes through byte-identical (no whitespace to trim).
    plain = "The P95 latency target is 3 seconds [1]."
    assert _strip_reasoning_tags(plain) == plain


def test_strip_tags_covers_thinking_and_reasoning_variants_case_insensitive() -> None:
    assert _strip_reasoning_tags("<THINKING>x</THINKING>Y") == "Y"
    assert _strip_reasoning_tags("<Reasoning>\nmulti\nline\n</Reasoning>Done") == "Done"


def test_strip_tags_multiple_blocks_and_answer_between() -> None:
    raw = "<think>a</think>Real [1]<reasoning>b</reasoning> answer [2]"
    assert _strip_reasoning_tags(raw) == "Real [1] answer [2]"


def test_strip_tags_answer_before_and_after_block() -> None:
    assert _strip_reasoning_tags("Before <think>mid</think> after") == "Before  after"


def test_answer_text_strips_inline_reasoning() -> None:
    assert _answer_text({"content": "<think>scratch</think>Answer [1]."}) == "Answer [1]."


def test_answer_text_reasoning_only_is_empty() -> None:
    assert _answer_text({"content": "<think>only</think>"}) == ""


def test_answer_text_null_content_is_empty() -> None:
    assert _answer_text({"content": None}) == ""


def test_answer_text_ignores_separate_reasoning_content_field() -> None:
    # reasoning_content is chain-of-thought, NOT the answer — never substituted.
    assert _answer_text({"content": "The answer [1].", "reasoning_content": "long cot"}) == "The answer [1]."


# --- _StreamReasoningFilter: reasoning spans never stream to the user --------------------------


def _run_stream(filt: _StreamReasoningFilter, chunks: list[str]) -> str:
    out = "".join(filt.feed(c) for c in chunks)
    return out + filt.flush()


def test_stream_plain_answer_passes_through() -> None:
    filt = _StreamReasoningFilter()
    assert _run_stream(filt, ["Hello ", "world", " [1]."]) == "Hello world [1]."


def test_stream_drops_inline_reasoning_block() -> None:
    filt = _StreamReasoningFilter()
    assert _run_stream(filt, ["<think>scratch</think>", "Answer [1]."]) == "Answer [1]."


def test_stream_reasoning_only_yields_nothing() -> None:
    filt = _StreamReasoningFilter()
    assert _run_stream(filt, ["<think>", "only reasoning", "</think>"]) == ""


def test_stream_open_tag_split_across_chunks() -> None:
    filt = _StreamReasoningFilter()
    assert _run_stream(filt, ["<th", "ink>scratch</th", "ink>Real answer"]) == "Real answer"


def test_stream_answer_around_reasoning() -> None:
    filt = _StreamReasoningFilter()
    got = _run_stream(filt, ["A ", "<think>", "cot", "</think>", " B"])
    assert got == "A  B"


def test_stream_unclosed_reasoning_yields_nothing() -> None:
    filt = _StreamReasoningFilter()
    assert _run_stream(filt, ["<think>", "still thinking when budget ran out"]) == ""


# --- a bare reasoning dump is never published as the answer --------------------------------------

_CONVERSATIONAL_LEAK = (
    'thought Role: Sovereign document-assistant. User input: "hi" (conversational). '
    "Constraint 1: Reply to the conversational part in ONE short, warm sentence. "
    "Constraint 2: Invite them to ask about their documents. Constraint 3: Do NOT state facts "
    "about documents. Constraint 4: Do NOT ask them to repeat themselves. Constraint 5: Reply "
    'with the sentence only. Conversational part: "Hello'
)


def test_a_truncated_bare_thought_dump_is_not_published_as_the_answer() -> None:
    """Observed live on Gemma-31B: the user typed "hi" and the console rendered the model's whole
    planning preamble — the system prompt's role and all five constraints — as the answer. That
    is prompt disclosure, not merely untidy output.

    The extractor recognised the dump but isolates the answer by its [n] citation, and a
    CONVERSATIONAL reply carries none by design; this one was truncated mid-thought as well, so
    there was no answer to find. It then returned the text unchanged. Empty is the honest result
    — the rule the tagged <think> path already follows — and it lets the caller's
    empty-generation guard retry or abstain instead of publishing the model's working."""
    assert _strip_reasoning(_CONVERSATIONAL_LEAK) == ""


def test_a_cited_answer_after_a_thought_dump_is_still_recovered() -> None:
    """The dump usually ends with the real, cited answer — that must keep working."""
    text = "thought Role: assistant. Constraint 1: cite every sentence. The uptime target is 99.5 percent. [1]"
    assert _strip_reasoning(text) == "The uptime target is 99.5 percent. [1]"


def test_prose_that_merely_mentions_thoughts_is_untouched() -> None:
    """The guard must not blank a real answer that happens to open with the word."""
    text = "Thoughts on the policy were invited from every department. [1]"
    assert _strip_reasoning(text) == text
