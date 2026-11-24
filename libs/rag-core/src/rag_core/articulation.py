"""The presentation contract an answer is articulated under — one text, every prose-writing call.

WHY THIS IS ITS OWN MODULE. Every rule below was added to ``grounding.build_prompt`` because a
real answer read badly, and each was then measured. None of that work reached Phase 9: the
map-reduce path writes its prose in :func:`~rag_core.mapreduce.build_reduce_prompt`, which
never saw ``build_prompt``, so the two paths articulated the same corpus to two different
standards. Measured on the AEGCL corrigendum, incumbent vs Phase 9: **587 bold spans vs 0**,
9.4 vs 16.8 words per sentence, 0.0 vs 2.8 duplicate lines. The rules were not weaker in Phase
9 — they were absent. A rule that lives inside one prompt builder is a rule only that builder
has.

TWO TIERS, AND THE SPLIT IS NOT COSMETIC.

*Line-level* rules (:func:`line_rules`) govern how ONE line is written: what is bolded, how a
figure is printed, that notation survives, that the line says something new. They hold wherever
prose is written, however the answer is later assembled, so they are shared.

*Document-level* rules — which container the answer takes (a sequence is numbered, parallel
facts are bulleted, shared attributes are a table), where a heading goes, that the first line is
the direct answer — govern the WHOLE answer. They can only be obeyed by something that can see
the whole answer. ``build_prompt`` can, because it makes one call that writes everything, so it
states them, and they stay there next to the table override that is decided in code. Phase 9
cannot: each reduce call writes ONE facet's bullets, is explicitly told not to write a heading,
and the frame is then assembled in code (``mapreduce.assemble_answer``). Handing a fragment the
whole answer's shape rules asks it to decide something it cannot see, so Phase 9 owns that tier
in code and takes only the line tier from here.

WHAT THIS DOES NOT CLOSE. Phase 9's container is fixed to bullets by the reduce prompt, so a
facet whose content is a SEQUENCE still renders unordered there, where a single-pass answer
would number it. That is a document-level divergence, and this module is deliberately not where
it gets fixed.

THE TEXT IS PARAMETERISED, NOT REWRITTEN. ``build_prompt``'s rendering of these rules is
byte-identical to what it emitted before they were extracted — deliberately, because it is the
control arm of a paired comparison that is still running, and a control whose prompt moved is
not a control. Two things vary by caller and nothing else does: what the evidence is called
(numbered SOURCES in one prompt, extracted items in the other) and where the caller's own
numbering has got to.
"""

from __future__ import annotations

__all__ = ["LINE_RULES", "NUMBERS_RULE_INDEX", "line_rules"]

#: The line-level presentation contract, one entry per rule, in the order they are stated.
#:
#: Rendered rather than stored as one blob so a caller can number them from wherever its own
#: rules stop and indent them to match its own list — the TEXT is shared, the list position is
#: the caller's. Each rule carries the observation that produced it; none of them is a style
#: preference someone liked the look of. ``{sources}`` and ``{numbers_rule}`` are filled by
#: :func:`line_rules`.
LINE_RULES: tuple[str, ...] = (
    # A multi-sentence paragraph is where a preamble and a restatement of the question hide.
    "NEVER write a multi-sentence prose paragraph. NEVER open with a preamble or restate the question.\n",
    "Put EVERY entity name and EVERY number in **bold**.\n",
    # Faithfulness is scored per sentence, so a restatement is not free: it adds a sentence
    # that must independently survive entailment. Observed live — the model answered one
    # fact three ways and scored 0.33 for content that scored 1.00 as a single sentence.
    "Each line must add NEW information. NEVER restate the opening sentence in "
    "different words, and never repeat a fact already given.\n",
    # Figures arriving from a database are raw floats. Printed verbatim they read as
    # 22844717.08470588 — unreadable, and it makes a correct answer look unfinished.
    "NUMBERS: write them the way a person would. Group thousands (1,234,567), keep at "
    "most two decimals, and never print a long floating-point tail. For Indian amounts also "
    "give the readable scale in brackets — e.g. **1,25,49,23,246** (₹125.5 crore). "
    "Where a value is plainly money, prefix ₹. Never alter the value itself.\n",
    "THREE OR MORE figures belong on separate lines, one per line — not run together in a single sentence.\n",
    # The NUMBERS rule tells the model to make figures readable, which invites it to "tidy" a
    # formula by flattening a-squared to a2, or H2SO4 out of its subscripts. Those are different
    # statements: "a2 - b2" is not the identity, and H2SO4 is not the same substance as the
    # subscripted form. Extraction preserves the raised and lowered characters (see the Chandra
    # OCR prompt), so the answer must not undo that on the way out.
    "NOTATION: copy superscripts, subscripts and symbols EXACTLY as {sources} write "
    "them — a², x³, H₂SO₄, NO₃⁻, 25°C. Never flatten them to "
    "ordinary digits, and never rewrite them as a^2 or H2SO4. Rule {numbers_rule} is about the "
    "readability of quantities, not about formulas.\n",
)

#: Position of the NUMBERS rule within :data:`LINE_RULES`. The NOTATION rule refers to it BY
#: NUMBER ("Rule 6 is about readability, not formulas"), and that number depends on where the
#: caller started counting — so it is derived, never written down twice.
NUMBERS_RULE_INDEX = 3


def line_rules(*, start: int = 1, indent: str = "", sources: str = "the SOURCES") -> str:
    """:data:`LINE_RULES` as a numbered list starting at ``start``, each line prefixed ``indent``.

    ``start`` exists because these rules are never the whole list: they continue a caller's own
    rules, and a list that restarts at 1 halfway down reads as a second, competing list. The
    caller says where its numbering has got to, and the cross-reference inside the NOTATION rule
    follows from it.

    ``sources`` names the evidence in the caller's own words — a prompt that never uses the word
    SOURCES must not be told to copy notation from them.
    """
    numbers_rule = start + NUMBERS_RULE_INDEX
    return "".join(
        f"{indent}{n}. " + rule.format(sources=sources, numbers_rule=numbers_rule)
        for n, rule in enumerate(LINE_RULES, start=start)
    )
