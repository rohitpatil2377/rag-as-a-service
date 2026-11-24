"""What a call is FOR, so a gateway can apply the right sampling policy to it.

WHY THIS EXISTS. ``OpenAICompatGateway`` nudges an explicitly requested ``temperature=0.0`` up to
0.6 whenever reasoning is suppressed, because a model decoding greedily with no reasoning pass
loops on tables and echoes the prompt's own examples back as content. That is a real, measured
generation failure and the nudge stays.

It was ALSO silently applied to structured extraction, where it is the wrong trade. Measured on
`corrigendum-5.pdf` (2026-11-02): Phase 4 asks for 0.0 and the request carried 0.6, and three
ingests of identical bytes produced three near-disjoint facet vocabularies (1 shared id of 10),
0.43 median summary similarity, and three distinct amendments of which each run found a different
two. Those labels are not decoration -- a facet id is the join key of Phase 8's facet union and
its label is a cross-encoder probe -- so re-rolling them re-rolls section selection.

A TASK, NOT A BYPASS FLAG. ``honour_exact_temperature=True`` would read as "ignore the nudge" at
every future call site and would say nothing about why. A task says what the call is, and the
policy for it lives here where it can be reasoned about once.

UNCLASSIFIED CALLERS KEEP THEIR BEHAVIOUR. :data:`GENERATION` is the default everywhere, so
nothing changes for a caller that has not been classified -- including Phase 9's map/reduce, whose
sampling behaviour at 0.6 is a MEASURED baseline (see the window-size experiment in
``docs/rag_v2/goal_scorecard.md``, 18 runs, every request verified on the wire at
``temperature=0.6``). Reclassifying it would invalidate that baseline and change answers, so it is
a separate decision and not a side effect of this one.

EXACT SAMPLING IS NOT DETERMINISM, and this module must not be read as promising it. Measured on
one section, six calls, posted directly to the endpoint: at 0.0 two of three extractions were
distinct; at 0.6 three of three were. Greedy narrows the variation; the endpoint is nondeterministic
without it. What a task guarantees is that the sampling policy the caller ASKED FOR is the one the
endpoint is given -- the remaining variation is the model's and is measured, not assumed away.
"""

from __future__ import annotations

from typing import Final

#: Answer generation, condensation, rewriting -- anything whose output a person reads as prose.
#: Keeps the existing nudge.
GENERATION: Final = "generation"

#: A call whose reply must satisfy a schema and be stored as metadata: the Phase-3 document
#: profile, Phase-4 section enrichment. The caller's temperature is passed through unchanged,
#: because what is wanted is the same answer for the same bytes rather than a readable one.
STRUCTURED_EXTRACTION: Final = "structured_extraction"

#: Every task this module defines. A value outside it is treated as :data:`GENERATION` rather
#: than raising: a sampling policy must never be the reason an ingest fails.
TASKS: Final = frozenset({GENERATION, STRUCTURED_EXTRACTION})


def honours_exact_temperature(task: str) -> bool:
    """True when the caller's temperature must reach the endpoint unmodified.

    An unknown task answers False -- the conservative direction, since False is exactly today's
    behaviour for every caller that has not been classified."""
    return task == STRUCTURED_EXTRACTION
