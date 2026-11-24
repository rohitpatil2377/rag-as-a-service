"""One presentation contract, two prompts — the parity Phase 9 did not have.

The gap these tests close was MEASURED, not suspected. Asked the same aggregate question over
the same corpus, the single-pass answer carried 587 bold spans and the map-reduce answer carried
0; 9.4 words per sentence against 16.8; no duplicate lines against 2.8. The evidence behind the
map-reduce answer was better (92.5% must-tier recall against 45.8%) — the rules for writing it
down simply were not in its prompt, because they were written inside ``build_prompt``.

So the real assertion here is the LAST one: every line rule reaches BOTH prompts. The rest guard
the two things that make sharing the text safe — that the single-pass prompt did not shift while
the rules were lifted out of it (it is the control arm of a comparison that is still running),
and that the DOCUMENT-level rules did not come along for the ride into a prompt that writes one
fragment of an answer and cannot see the rest of it.
"""

from __future__ import annotations

from rag_core.articulation import LINE_RULES, line_rules
from rag_core.grounding import build_continuation_prompt, build_coverage_reprompt, build_prompt
from rag_core.mapreduce import Coverage, MappedItem, ReduceInput, build_reduce_prompt
from rag_schemas import RetrievedSpan


def _spans() -> list[RetrievedSpan]:
    return [RetrievedSpan(chunk_id="c1", doc_id="d1", text="The uptime target is 99.5 percent.")]


def _reduce_prompt(facet: str = "power") -> str:
    item = MappedItem(
        item_id="m1",
        section_id="s1",
        doc_id="d1",
        facet_id=facet,
        statement="The UPS shall be rated 20 kVA",
        quote="UPS shall be rated 20 kVA",
    )
    data = ReduceInput(
        query="list every requirement",
        profiles="- corrigendum-v.pdf",
        items=[item],
        coverage=Coverage(1, 1, 1),
        answer_lang="English",
    )
    return build_reduce_prompt(data, facet, (item,), {"s1": "6.2 Power"})


def test_the_rules_continue_the_callers_own_numbering() -> None:
    """A shared list that restarts at 1 halfway down a prompt reads as a second, competing list,
    so the caller says where its own numbering has got to."""
    rendered = line_rules(start=6)

    assert rendered.startswith("6. NEVER write a multi-sentence")
    assert "\n11. NOTATION:" in rendered
    assert "\n1. " not in rendered


def test_the_notation_rules_cross_reference_follows_the_numbering() -> None:
    """The NOTATION rule points at the NUMBERS rule BY NUMBER ("that one is about readability,
    not formulas"). Hard-coded, it would name a different rule in each prompt — which is worse
    than no cross-reference, because it names one that exists and says something else."""
    assert "Rule 9 is about the readability" in line_rules(start=6)
    assert "Rule 6 is about the readability" in line_rules(start=3)
    assert "Rule 4 is about the readability" in line_rules(start=1)


def test_the_evidence_is_named_in_the_callers_own_words() -> None:
    """The reduce prompt has no SOURCES section. Telling it to copy notation "as the SOURCES
    write them" points at nothing in the prompt it is in."""
    assert "as the SOURCES write them" in line_rules()
    assert "as the items write them" in line_rules(sources="the items")


def test_the_single_pass_prompt_still_numbers_these_rules_three_to_eight() -> None:
    """The control arm must not shift. ``build_prompt`` is the deployed answer path AND the
    control in a paired comparison that is still being run, so lifting these rules out of it had
    to leave its rendered text alone — including the numbering gap at 2, which predates the
    extraction and is a quirk of this prompt, not something to tidy while moving house."""
    prompt = build_prompt("what is the uptime target", _spans())

    assert "  1. Whatever the container, the FIRST line is the direct answer" in prompt
    assert "  2. " not in prompt
    assert "  3. NEVER write a multi-sentence prose paragraph" in prompt
    assert "  8. NOTATION: copy superscripts" in prompt
    # Still inside STEP 2's indented block, between the container choice and the examples.
    assert prompt.index("  d. ONE FACT") < prompt.index("  4. Put EVERY entity name")
    assert prompt.index("  8. NOTATION") < prompt.index("Two shapes, for contrast")


def test_the_reduce_prompt_keeps_grounding_and_citation_rules_first() -> None:
    """Presentation never outranks grounding. A model that reads "bold every number" before it
    reads "use ONLY the items" has been handed the two in the wrong order."""
    prompt = _reduce_prompt()

    assert prompt.index("Use ONLY the items.") < prompt.index("HOW EACH LINE IS WRITTEN")
    assert prompt.index("End EVERY bullet with its citation markers") < prompt.index("7. NEVER write")
    # And the rule that keeps a marker out of the bold rule's reach precedes the bold rule.
    assert prompt.index("write [m3], never [**m3**]") < prompt.index("EVERY number in **bold**")


def test_the_reduce_prompt_does_not_receive_the_document_level_rules() -> None:
    """A reduce call writes ONE facet's bullets into an answer it cannot see, and rule 1 already
    forbids it a heading. Handing it the container choice asks it to decide the shape of
    something it was shown one topic of; ``assemble_answer`` owns that, in code."""
    prompt = _reduce_prompt()

    assert "STEP 1" not in prompt and "STEP 2" not in prompt
    assert "Headings ('## Title')" not in prompt
    assert "GitHub-flavoured Markdown table" not in prompt
    assert "the FIRST line is the direct answer" not in prompt


def test_every_line_rule_reaches_both_prompts() -> None:
    """THE EXIT CONDITION. Not "the shared module exists" — that every rule in it is actually in
    both prompts, so a rule added here can never again land in one path only."""
    single_pass = build_prompt("list every requirement", _spans())
    reduce = _reduce_prompt()

    for rule in LINE_RULES:
        # The two differ only where the text names the evidence; compare the stable head.
        head = rule.split("{")[0].strip()
        assert head, "a rule whose text begins with a placeholder cannot be matched this way"
        assert head in single_pass, f"missing from the single-pass prompt: {head[:60]}"
        assert head in reduce, f"missing from the reduce prompt: {head[:60]}"


def test_the_bold_rule_is_present_verbatim_in_the_reduce_prompt() -> None:
    """Called out on its own because it is the single largest measured divergence: 587 bold spans
    against 0. Everything else in the gap is a matter of degree; this one was total."""
    assert "Put EVERY entity name and EVERY number in **bold**." in _reduce_prompt()


def test_the_coverage_rewrite_states_the_line_rules_too() -> None:
    """The same defect, one prompt over: the whole-document path's coverage loop REWRITES the
    answer, and its prompt carries no presentation rule at all. So an answer that was written to
    the contract can be rewritten out of it -- the rewrite is told to keep what is correct and add
    what is missing, and nothing about how either is written down.

    That was worse than absent, because adherence then depended on how many times the loop ran,
    which is invisible in any mean. Held until the paired comparison finished, because this prompt
    is on the control arm's path and a control whose prompt moves is not a control."""
    spans = _spans()
    prompt = build_coverage_reprompt("list every requirement", "The target is **99.5%**. [1]", spans, spans)

    for rule in LINE_RULES:
        head = rule.split("{")[0].strip()
        assert head in prompt, f"missing from the coverage rewrite: {head[:60]}"


def test_the_truncation_continuation_states_the_line_rules_too() -> None:
    """The continuation prompt said "keep the SAME format as the partial answer", which is format
    by imitation: it carries the shape the model can see in the text it is continuing, and cannot
    carry the rules it cannot infer from a sample -- bold every entity, give an Indian amount its
    readable scale, keep the subscripts. Those are stated now."""
    prompt = build_continuation_prompt("list every requirement", "The target is **99.5%**. [1]", _spans())

    for rule in LINE_RULES:
        head = rule.split("{")[0].strip()
        assert head in prompt, f"missing from the continuation prompt: {head[:60]}"
