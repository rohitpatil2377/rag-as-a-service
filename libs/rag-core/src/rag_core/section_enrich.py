"""Section enrichment — RAG v2 Phase 4 (pure logic; the gateway calls are in rag_clients).

One model call per section produces four things the rest of the plan reads: a SUMMARY (the
aggregate grain Phase 5 built and withdrew for want of one), FACET LABELS drawn from the
document's own Phase-3 taxonomy, extracted KEY-VALUES, and standalone PROPOSITIONS. A section
that states it changes something stated elsewhere also yields an AMENDS record.

Where each of those goes:

* ``sections.summary`` / ``sections.facet_ids`` / ``sections.key_values`` / ``sections.amends`` —
  the Phase-1 columns, written by ``rag_clients.postgres_catalog.replace_section_enrichment``.
* ``sections.amends`` is additionally the second input to Phase 6's graph seeding:
  ``document_relations`` already reads it and its docstring says so ("still Phase 4's and is still
  empty"). This module is what fills it, so a SECTION-level amendment becomes a graph edge the
  same way a document-level one does.
* ``chunks`` at ``level='proposition'`` and ``level='section'`` — the two levels
  :class:`rag_schemas.common.ChunkLevel` names as waiting on this phase. The chunker emits them
  from a :class:`SectionEnrichment`; see ``rag_core.layout_chunking``.

EVERY QUOTED ITEM IS VALIDATED AGAINST THE TEXT THE MODEL WAS SHOWN, through
:func:`rag_core.graph.quote_key` — the plan's own rule ("discard any item whose quote is not
a substring, after whitespace normalisation, of the section text") and the same normalisation
Phase 3 validates a relation's quote with and Phase 6 anchors it with. A key-value, a proposition
or an amends whose quote is not there is DROPPED and REPORTED, never kept with a weaker quote: a
proposition is indexed as its own retrievable chunk, so an unquoted one is a sentence the corpus
now asserts and no page of the document says.

THE SUMMARY NEEDS NO QUOTE, and that is the one asymmetry here. A summary is by construction not
verbatim; requiring a quote for it would mean requiring a quote for a paraphrase. What bounds it
instead is that it is the ONLY unquoted thing this module keeps, and it reaches the index as a
``section``-level chunk whose citation points at the section it summarises.

THE TEXT THIS MODULE SEES IS MASKED, which is a deliberate deviation from the plan's "section text
(tables as HTML)" and the one worth stating twice. ``blocks.text`` and the text behind
``sections.text_hash`` are UNMASKED (they are the extractor's own output); ``Chunk.text`` is
masked (FR-G2). Propositions BECOME chunk text and summaries become section-chunk text, so an
unmasked enrichment pass would route document PII straight into the index through a path that
looks like derived metadata. The caller therefore masks before building the prompt, quotes
validate against the masked text, and ``blocks.html`` is not used at all — it is unmasked raw
markup and is None on the whole ``layout-json`` extraction path anyway, so "tables as HTML" would
have been available for a minority of documents and unmasked for all of them. Tables reach the
prompt as :func:`section_prompt_text` renders them: the table block's own pipe-delimited lines.

NOTHING HERE CALLS A MODEL. :func:`build_section_enrich_prompt` renders the reviewable template in
``prompts/section_enrich.md`` and :func:`parse_section_enrichment` validates a reply; the gateway
call, its one retry, its windowing and its fail-safe live in ``rag_clients.section_enricher`` —
the same split as ``doc_profile`` / ``rag_clients.doc_profiler``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_core.doc_profile import snake_id
from rag_core.graph import quote_key
from rag_core.json_reply import scan_json_object
from rag_core.layout import LayoutSection
from rag_core.prompts import render_prompt

__all__ = [
    "ENRICH_VERSION",
    "EnrichParse",
    "EnrichReport",
    "KeyValue",
    "Proposition",
    "SectionAmends",
    "SectionEnrichment",
    "build_section_enrich_prompt",
    "merge_enrichments",
    "parse_section_enrichment",
    "section_prompt_text",
    "section_windows",
]

# The version of the ENRICHMENT CONTRACT — this prompt, this parser, this schema. Read by
# `svc_indexing.service.pipeline_fingerprint`, so bumping it marks every enriched document stale
# and a reindex re-enriches it. Same role, and the same reasoning, as
# `rag_core.doc_profile.PROFILE_VERSION`: it answers "which documents hold an enrichment this
# build would no longer produce", which selects a corpus to act on.
ENRICH_VERSION = 1

# Block types rendered into the prompt. `table_row` is deliberately absent — see
# `section_prompt_text`.
_PROMPT_BLOCK_TYPES = frozenset({"heading", "paragraph", "list_item", "table"})
_PROMPT_BLOCK_SEP = "\n\n"

# Bounds. Guards against a model reply, not tuning knobs: everything here lands in a Postgres row
# and most of it lands in the index as chunk text.
_MAX_SUMMARY_CHARS = 1500
_MAX_FACET_IDS = 12
_MAX_KEY_VALUES = 40
_MAX_PROPOSITIONS = 30
_MAX_NAME_CHARS = 160
_MAX_VALUE_CHARS = 400
_MAX_UNIT_CHARS = 40
_MAX_QUOTE_CHARS = 1000
_MAX_PROPOSITION_CHARS = 800
_MAX_TARGET_REF_CHARS = 300


@dataclass(frozen=True, slots=True)
class KeyValue:
    """One extracted figure or named value, with the span of section text that states it.

    ``unit`` is None where the section states none. It is NOT inferred: "19.44" beside "Crores" in
    a sentence the model quoted is a unit the document gave, and "19.44" alone is a number whose
    unit this module declines to guess, because the guess would be stored as if the document had
    said it."""

    name: str
    value: str
    unit: str | None
    quote: str


@dataclass(frozen=True, slots=True)
class SectionAmends:
    """What this section says it changes, and where the section says it.

    ``old`` is None where the section states the new value without naming the old one, which is
    the common shape of a corrigendum clause that simply restates a requirement. ``new`` and
    ``target_ref`` are both required — an amendment with no target names nothing, and one with no
    new value asserts no change."""

    target_ref: str
    old: str | None
    new: str
    quote: str


@dataclass(frozen=True, slots=True)
class Proposition:
    """One standalone requirement or fact, plus the span that states it.

    Self-contained by instruction: the prompt asks for the subject to be NAMED, because a
    proposition is indexed as its own retrievable chunk and "it must be submitted within 30 days"
    retrieved on its own supports no answer at all."""

    text: str
    quote: str


@dataclass(frozen=True, slots=True)
class SectionEnrichment:
    """One section's validated enrichment (plan §1.1 ``sections`` + §4).

    ``section_id`` is the tree's own, UNMASKED — it is identity, and it must equal the persisted
    ``sections`` row that this enrichment updates and the ``Chunk.section_id`` of every unit built
    from it."""

    section_id: str
    summary: str
    facet_ids: tuple[str, ...] = ()
    key_values: tuple[KeyValue, ...] = ()
    amends: SectionAmends | None = None
    propositions: tuple[Proposition, ...] = ()
    enrich_version: int = ENRICH_VERSION

    def key_values_json(self) -> str:
        """``sections.key_values`` as stored JSONB: a LIST, each item carrying its own quote.

        A list rather than a ``{name: value}`` object because a section routinely states the same
        name twice (a bid value before and after revision), and an object would silently keep one
        of them — exactly the loss this phase exists to prevent."""
        return json.dumps(
            [{"name": kv.name, "value": kv.value, "unit": kv.unit, "quote": kv.quote} for kv in self.key_values],
            ensure_ascii=False,
        )

    def amends_json(self) -> str | None:
        """``sections.amends`` as stored JSONB, or None for a section that amends nothing.

        The key names are the ones ``PostgresCatalog.document_relations`` already reads out of this
        column (``amends->>'target_ref'``, ``amends->>'quote'``) — that query was written in Phase
        6 against a column nothing wrote, and this is the write it was written for."""
        if self.amends is None:
            return None
        return json.dumps(
            {
                "target_ref": self.amends.target_ref,
                "old": self.amends.old,
                "new": self.amends.new,
                "quote": self.amends.quote,
            },
            ensure_ascii=False,
        )


@dataclass(frozen=True, slots=True)
class EnrichParse:
    """What :func:`parse_section_enrichment` made of a reply, INCLUDING what it threw away.

    Same discipline as :class:`rag_core.doc_profile.ProfileParse`: a dropped item is reported,
    never silently discarded. The one that matters most is ``amends_unquoted`` — a section-level
    amendment lost to a misquoted span is the difference between a clause that supersedes another
    and one that does not, and with no report it is indistinguishable from a section that simply
    amends nothing."""

    enrichment: SectionEnrichment | None = None
    error: str = ""
    #: WHY there is no enrichment, as a stable code rather than a sentence. `error` stays the
    #: human string; this is what a dashboard groups by and what an operator's fix is chosen from,
    #: and the four causes do not share a fix:
    #:
    #:   TRUNCATED        a JSON object opened and the reply ended inside it -> output budget,
    #:                    or a smaller input. ANOTHER ATTEMPT ONLY RE-ROLLS THE LENGTH.
    #:   UNUSABLE_OUTPUT  no JSON object at all: prose -> prompt, or another attempt
    #:   INVALID_JSON     a syntactically malformed object -> prompt or schema
    #:   INVALID_SCHEMA   valid JSON, wrong shape or a required field missing -> schema
    #:
    #: Empty on success. Conflating the first two cost one wrong committed diagnosis on
    #: 2026-11-02; see :func:`~rag_core.json_reply.scan_json_object`.
    reason_code: str = ""
    unquoted_key_values: tuple[str, ...] = ()  # dropped: the quote is not in the section text
    unquoted_propositions: tuple[str, ...] = ()  # dropped: same
    amends_unquoted: bool = False  # an amends was declared and dropped for an unfindable quote
    unknown_facets: tuple[str, ...] = ()  # dropped: not in this document's taxonomy
    quotes_seen: int = 0  # every quoted item the reply offered, kept or dropped
    quotes_kept: int = 0  # ...and how many validated. The plan's ">= 95% of quotes validate".


def section_prompt_text(section: LayoutSection) -> str:
    """The section's own text as the enrichment prompt should see it. UNMASKED — the caller masks.

    Built from the blocks rather than taken from ``LayoutSection.text`` because that attribute
    concatenates EVERY block, and a table is present twice over: once as the ``table`` block's
    pipe-delimited lines and again as one ``table_row`` block per row (Phase 2 emits both, which is
    what lets :class:`~rag_core.layout_chunking.LayoutAwareChunker` index rows separately).
    Sending both doubles a wide table's share of the prompt and costs the surrounding prose the
    attention it needs.

    THE TABLE BLOCK IS THE ONE KEPT, not the rows, because the table carries its header line and a
    row's linearised form repeats the breadcrumb on every row. A ``table_row`` whose parent table is
    NOT in this section's blocks is kept — that is the same orphan case ``_body_level`` guards, and
    dropping it would hide the table from the prompt entirely."""
    present = {b.block_id for b in section.blocks}
    parts = [
        block.text
        for block in section.blocks
        if block.text.strip()
        and (
            block.block_type in _PROMPT_BLOCK_TYPES
            or (block.block_type == "table_row" and block.parent_block_id not in present)
        )
    ]
    return _PROMPT_BLOCK_SEP.join(parts)


def section_windows(text: str, window_chars: int, max_windows: int) -> list[str]:
    """Split an over-long section into windows for separate calls (plan §4: "very long sections:
    split into windows and merge").

    Splits on the BLOCK SEPARATOR that :func:`section_prompt_text` joined with, and only falls back
    to a hard character cut for a single block that is itself longer than a window. A window that
    begins mid-sentence costs the model the subject of the sentence, and a quote spanning the cut
    validates against neither window — so the split is placed where the document already had one
    wherever the document had one.

    TRUNCATES at ``max_windows`` rather than enriching a 400-page appendix in sixty calls. The
    caller logs the drop; nothing silently claims to have read the whole section.
    """
    if window_chars <= 0 or max_windows <= 0:
        return []
    if len(text) <= window_chars:
        return [text] if text.strip() else []
    out: list[str] = []
    current = ""
    for block in text.split(_PROMPT_BLOCK_SEP):
        for piece in _hard_pieces(block, window_chars):
            if not current:
                current = piece
            elif len(current) + len(_PROMPT_BLOCK_SEP) + len(piece) <= window_chars:
                current = f"{current}{_PROMPT_BLOCK_SEP}{piece}"
            else:
                out.append(current)
                current = piece
            if len(out) >= max_windows:
                return out[:max_windows]
    if current.strip():
        out.append(current)
    return out[:max_windows]


def _hard_pieces(block: str, window_chars: int) -> list[str]:
    """One block, cut to the window size only if the block alone exceeds it."""
    if len(block) <= window_chars:
        return [block]
    return [block[at : at + window_chars] for at in range(0, len(block), window_chars)]


def build_section_enrich_prompt(
    *, profile_json: str, facet_ids: Sequence[str], breadcrumb: str, section_text: str
) -> str:
    """Render ``prompts/section_enrich.md`` with one section's inputs (plan §4's three inputs,
    plus the taxonomy spelled out).

    EVERY ARGUMENT MUST ALREADY BE MASKED. ``breadcrumb`` and ``section_text`` come off the
    structure tree, which is captured PRE-masking; ``profile_json`` carries the document's purpose
    and title, which Phase 3 masked before storing. What comes back is written to ``sections`` and
    indexed as chunk text, so an identifier that reaches this prompt reaches both.

    THE FACET IDS ARE LISTED SEPARATELY even though ``profile_json`` already contains the whole
    ``facet_taxonomy``, which is a deviation from the plan's prompt. "From the provided
    facet_taxonomy only" is a hard validation rule — a label outside it is discarded by
    :func:`parse_section_enrichment` — and burying the permitted values inside a JSON blob that
    also holds relations, dates and a purpose makes the model work to find the one list it is not
    allowed to depart from. The blob stays because Phase 8 and the plan both want the model to see
    the document's own framing, not just its vocabulary."""
    listing = "\n".join(f"- {fid}" for fid in facet_ids)
    return render_prompt(
        "section_enrich",
        profile_json=profile_json.strip() or "(this document has no profile card)",
        facet_ids=listing or "(this document has no facet taxonomy; return an empty facet_ids list)",
        breadcrumb=breadcrumb.strip() or "(this section has no heading path)",
        section_text=section_text.strip(),
    )


def parse_section_enrichment(
    reply: str, *, section_id: str, source_text: str, facet_ids: Iterable[str] = ()
) -> EnrichParse:
    """Validate a model reply into a :class:`SectionEnrichment`, or explain why there isn't one.

    ``source_text`` is the text every quote must be found in — the SAME text the prompt was given,
    which for a windowed section is THAT WINDOW and not the whole section. Passing the whole
    section would let a window's reply cite text that window never saw, which is the fabrication
    the check exists to catch.

    ``facet_ids`` is this document's Phase-3 taxonomy. A label outside it is dropped and reported
    rather than created: ``facets`` is keyed ``(doc_id, facet_id)`` and the profile call is what
    defines the vocabulary, so a section inventing ``penalty_clauses`` would put a label in
    ``sections.facet_ids`` that joins to no facet row and that Phase 8 would group an aggregate
    answer by with nothing to name it.

    THE VERDICT IS A SUMMARY. A reply with no usable ``summary`` yields no enrichment at all, even
    if its key-values and propositions are perfect — the same all-or-nothing rule
    :func:`~rag_core.doc_profile.parse_doc_profile` applies to ``doc_type``/``purpose``, and
    for the same reason: a row with a NULL summary and populated facets reads downstream as a
    section that was enriched and has nothing to say about itself, which is not what happened.

    Never raises.
    """
    blob, scan = scan_json_object(reply)
    if blob is None:
        # TWO DIFFERENT FAILURES, and the old message called both of them "no JSON object in the
        # reply". A reply that opens an object and ends inside it was answering correctly and ran
        # out of room; a reply with no brace at all answered in prose. See `scan_json_object`.
        if scan == "unclosed":
            return EnrichParse(
                error="the reply opened a JSON object and ended before closing it (truncated)",
                reason_code="TRUNCATED",
            )
        return EnrichParse(error="no JSON object in the reply", reason_code="UNUSABLE_OUTPUT")
    try:
        data = json.loads(blob)
    except (ValueError, TypeError) as exc:
        return EnrichParse(error=f"malformed JSON: {str(exc)[:120]}", reason_code="INVALID_JSON")
    if not isinstance(data, dict):
        return EnrichParse(error="the JSON value is not an object", reason_code="INVALID_SCHEMA")

    summary = _text(data.get("summary"), _MAX_SUMMARY_CHARS)
    if not summary:
        return EnrichParse(error="missing summary", reason_code="INVALID_SCHEMA")

    haystack = quote_key(source_text)
    allowed = {snake_id(f) for f in facet_ids if snake_id(f)}
    facets, unknown = _facet_labels(data.get("facet_ids"), allowed)
    key_values, kv_dropped = _key_values(data.get("key_values"), haystack)
    propositions, prop_dropped = _propositions(data.get("propositions"), haystack)
    amends, amends_unquoted = _amends(data.get("amends"), haystack)

    kept = len(key_values) + len(propositions) + (1 if amends is not None else 0)
    seen = kept + len(kv_dropped) + len(prop_dropped) + (1 if amends_unquoted else 0)
    return EnrichParse(
        enrichment=SectionEnrichment(
            section_id=section_id,
            summary=summary,
            facet_ids=facets,
            key_values=key_values,
            amends=amends,
            propositions=propositions,
        ),
        unquoted_key_values=kv_dropped,
        unquoted_propositions=prop_dropped,
        amends_unquoted=amends_unquoted,
        unknown_facets=unknown,
        quotes_seen=seen,
        quotes_kept=kept,
    )


def merge_enrichments(parts: Sequence[SectionEnrichment]) -> SectionEnrichment | None:
    """Merge one section's per-window enrichments into one (plan §4: "split into windows and
    merge"). None for no parts.

    Each field merges by the rule that loses the least:

    * ``summary`` — the window summaries JOINED, in window order, bounded. Not "the first", which
      would describe the opening of a section and claim to describe the whole; not a re-summary,
      which would need another model call over text this module has already paid to read.
    * ``facet_ids`` — union, first-seen order. A facet the section raises anywhere it raises.
    * ``key_values`` / ``propositions`` — concatenated in window order, de-duplicated on the pair
      that makes an item the same item (name+value; the proposition's own text). A section whose
      windows overlap nothing still repeats a figure stated in two places, and storing it twice
      would put two identical proposition chunks in the index.
    * ``amends`` — the FIRST window that declared one. Two windows of one section declaring two
      different amendments is a shape the column cannot hold (it is one JSONB object, per plan
      §1.1), and picking the first is the only choice that is stable across re-runs.
    """
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    summary = " ".join(p.summary for p in parts if p.summary)[:_MAX_SUMMARY_CHARS]
    facets: dict[str, None] = {}
    for part in parts:
        for fid in part.facet_ids:
            facets.setdefault(fid, None)
    key_values: dict[tuple[str, str], KeyValue] = {}
    propositions: dict[str, Proposition] = {}
    for part in parts:
        for kv in part.key_values:
            key_values.setdefault((kv.name, kv.value), kv)
        for prop in part.propositions:
            propositions.setdefault(prop.text, prop)
    amends = next((p.amends for p in parts if p.amends is not None), None)
    return SectionEnrichment(
        section_id=parts[0].section_id,
        summary=summary,
        facet_ids=tuple(list(facets)[:_MAX_FACET_IDS]),
        key_values=tuple(list(key_values.values())[:_MAX_KEY_VALUES]),
        amends=amends,
        propositions=tuple(list(propositions.values())[:_MAX_PROPOSITIONS]),
    )


def merge_windows(parts: Sequence[SectionEnrichment], *, windows: int) -> tuple[SectionEnrichment | None, bool]:
    """One section's windows merged ATOMICALLY. ``(merged, partial)``.

    A MULTI-WINDOW SECTION IS NOT ENRICHED UNLESS EVERY WINDOW PARSED. ``merge_enrichments``
    merges whatever it is handed, which is the right contract for a merge and the wrong one for a
    publication decision: handed two of a section's three windows it returns a perfectly valid
    enrichment, and that enrichment is then stored, counted as a success and used by Phase 8 as
    the section's selection metadata -- with a third of the section's facets, key values and
    summary missing and NOTHING saying so. A clean failure is disclosed (the manifest names the
    section, the answer carries a count); this is not, which makes a partial merge worse for a
    reader than the failure it hides.

    MEASURED, not hypothetical. The rejected windowing arm of the 2026-11-02 output-budget
    experiment truncated its final window in all three runs, and the merged result's facets
    collapsed from 5-6 to 2 and its key values swung 2-11 across runs while every run reported
    success.

    ``partial`` is True only when SOME window produced an enrichment and some did not -- the state
    worth a warning, because the model demonstrably could read the section and the result is still
    being thrown away. A section where nothing parsed is an ordinary failure, not a partial, and
    the caller already classifies it by reason code.

    ``windows`` is the number of windows the section was ACTUALLY SPLIT INTO, i.e. what
    :func:`section_windows` returned after its own cap -- not the number the whole section would
    need. Text past ``max_windows`` is a disclosed, deliberate truncation of the INPUT and is not
    a parse failure; conflating the two would report every over-long section as incomplete.
    """
    if windows <= 0:
        return None, False
    if len(parts) < windows:
        return None, bool(parts)
    return merge_enrichments(parts), False


# --- field helpers ------------------------------------------------------------------------------


def _text(value: object, limit: int) -> str:
    """A model's string field, trimmed and capped; "" for anything that is not a usable string."""
    return value.strip()[:limit] if isinstance(value, str) else ""


def _facet_labels(value: object, allowed: set[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The labels that exist in this document's taxonomy, plus the ones that do not."""
    if not isinstance(value, list):
        return (), ()
    kept: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for item in value[: _MAX_FACET_IDS * 2]:
        raw = _text(item, _MAX_NAME_CHARS)
        fid = snake_id(raw)
        if not fid or fid in seen:
            continue
        seen.add(fid)
        if fid in allowed:
            kept.append(fid)
        else:
            unknown.append(fid)
    return tuple(kept[:_MAX_FACET_IDS]), tuple(unknown)


def _key_values(value: object, haystack: str) -> tuple[tuple[KeyValue, ...], tuple[str, ...]]:
    """Validated key-values, plus the quotes of the ones dropped as unfindable."""
    if not isinstance(value, list):
        return (), ()
    kept: list[KeyValue] = []
    dropped: list[str] = []
    for item in value[:_MAX_KEY_VALUES]:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"), _MAX_NAME_CHARS)
        raw_value = _text(item.get("value"), _MAX_VALUE_CHARS)
        quote = _text(item.get("quote"), _MAX_QUOTE_CHARS)
        if not name or not raw_value or not quote:
            continue
        if quote_key(quote) not in haystack:
            dropped.append(quote[:120])
            continue
        kept.append(
            KeyValue(name=name, value=raw_value, unit=_text(item.get("unit"), _MAX_UNIT_CHARS) or None, quote=quote)
        )
    return tuple(kept), tuple(dropped)


def _propositions(value: object, haystack: str) -> tuple[tuple[Proposition, ...], tuple[str, ...]]:
    """Validated propositions, plus the quotes of the ones dropped as unfindable."""
    if not isinstance(value, list):
        return (), ()
    kept: list[Proposition] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for item in value[:_MAX_PROPOSITIONS]:
        if not isinstance(item, dict):
            continue
        body = _text(item.get("text"), _MAX_PROPOSITION_CHARS)
        quote = _text(item.get("quote"), _MAX_QUOTE_CHARS)
        if not body or not quote or body in seen:
            continue
        if quote_key(quote) not in haystack:
            dropped.append(quote[:120])
            continue
        seen.add(body)
        kept.append(Proposition(text=body, quote=quote))
    return tuple(kept), tuple(dropped)


def _amends(value: object, haystack: str) -> tuple[SectionAmends | None, bool]:
    """``(amends, dropped_for_an_unfindable_quote)``.

    The flag exists so a caller can tell "the section amends nothing" from "the section said it
    amends something and could not prove it" — the second is worth a WARNING and the first is the
    normal case for almost every section of almost every document. An object with no
    ``target_ref`` or no ``new`` is neither: it says nothing about what changed, so there is no
    amendment to lose and nothing to report."""
    if not isinstance(value, dict):
        return None, False
    target_ref = _text(value.get("target_ref"), _MAX_TARGET_REF_CHARS)
    new = _text(value.get("new"), _MAX_VALUE_CHARS)
    quote = _text(value.get("quote"), _MAX_QUOTE_CHARS)
    if not target_ref or not new or not quote:
        return None, False
    if quote_key(quote) not in haystack:
        return None, True
    return (
        SectionAmends(
            target_ref=target_ref, old=_text(value.get("old"), _MAX_VALUE_CHARS) or None, new=new, quote=quote
        ),
        False,
    )


@dataclass(frozen=True, slots=True)
class EnrichReport:
    """Per-document totals, for the one log line a caller emits after enriching a whole document.

    Separate from :class:`EnrichParse`, which is per-call: the question an operator asks is "did
    this document's enrichment hold up", and that is answered by a quote-validation RATE over the
    document, which the plan states as an acceptance threshold (">= 95% of quotes validate")."""

    sections_enriched: int = 0
    sections_failed: int = 0
    windows: int = 0
    quotes_seen: int = 0
    quotes_kept: int = 0
    unknown_facets: tuple[str, ...] = field(default_factory=tuple)
    amends_found: int = 0
    amends_unquoted: int = 0
    #: WHICH sections failed, and WHY, because `sections_failed` is a number and the question an
    #: operator has is which part of the document is now unreachable. A section that failed
    #: enrichment is written with no summary, no `facet_ids` and no `amends`, which closes all
    #: three Phase-8 selection paths at once -- so this is not a diagnostic nicety, it is the
    #: only record of what an aggregate answer can no longer see.
    failed_section_ids: tuple[str, ...] = field(default_factory=tuple)
    #: `reason_code` -> count, over the failed sections. Kept apart from the total because the
    #: codes do not share a fix: TRUNCATED wants output budget, UNUSABLE_OUTPUT wants a prompt,
    #: and reading one as the other is the mistake of 2026-11-02. See :class:`EnrichParse`.
    failure_classes: Mapping[str, int] = field(default_factory=dict)
    #: Sections where SOME window parsed and some did not, so the whole section was refused
    #: (:func:`merge_windows`). A SUBSET of `sections_failed`, not an addition to it: a partial is
    #: a failure, and the only reason it is counted apart is that it is the one failure the model
    #: half-answered -- which says the section is readable and points at the window, not the
    #: prompt. Zero on every single-window document.
    sections_partial: int = 0
    partial_section_ids: tuple[str, ...] = field(default_factory=tuple)
    #: What the document's enrichment ACTUALLY SPENT, summed over every call including retries and
    #: failed ones -- server-reported where the gateway reports usage, estimated from reply length
    #: where it does not. Raising `section_enrich_max_tokens` raises a CEILING; this is the only
    #: number that says what the ceiling cost, and the two are not the same.
    completion_tokens: int = 0
    #: Calls the model ended by running out of output budget (`finish_reason="length"`). NOT a
    #: count of failures: a call can stop at the cap having already closed its JSON. It is the
    #: early warning that the current ceiling is becoming the next 1,200.
    calls_at_cap: int = 0
    #: Sections that lost at least one item to an unfindable quote. `quotes_seen - quotes_kept`
    #: says HOW MANY claims the document could not support; this says WHERE, which is the first
    #: thing anyone asks when the rate lands under the floor. Measured need: three of fourteen
    #: 3,000-token ingests came in below 0.95, and the dropped claims were unrecoverable after
    #: the fact -- `EnrichParse` carried them and nothing logged or stored them.
    sections_with_unquoted: tuple[str, ...] = field(default_factory=tuple)

    @property
    def quote_rate(self) -> float | None:
        """Kept / seen, or None when the document offered no quoted item at all.

        None rather than 1.0 for an empty document: a rate of 1.0 over zero quotes reads as a
        perfect score and is not one, and the plan's threshold is a claim about evidence that
        exists."""
        return self.quotes_kept / self.quotes_seen if self.quotes_seen else None


#: Bumped when the MEANING of a stored manifest changes, so a reader can refuse one it cannot
#: interpret rather than misread it. Not bumped for a new optional field.
MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class EnrichmentManifest:
    """What Phase 4 actually managed to do to one document, persisted with that document.

    WHY THIS IS STORED RATHER THAN LOGGED. A section whose enrichment failed is written with no
    summary, no ``facet_ids`` and no ``amends``, which closes all three Phase-8 selection paths
    into it: the document keeps the section and an aggregate answer can no longer see it. Until
    this existed the only record was one log line at ingest, so an answer served a day later
    could not know, and neither could anyone reading the answer. Reconstructing it afterwards is
    not possible either -- a row with no summary cannot say whether the minimum-length rule
    DECLINED the section or a model call FAILED on it, and only one of those is a defect.

    ``content_hash`` BINDS IT TO THE BYTES. A manifest that does not match the document's current
    hash describes a previous ingest, and a stale success must never outlive a re-ingest that
    failed. A reader compares before trusting.

    POLICY SKIPS ARE NOT FAILURES and are counted separately on purpose. 15 of this corpus's 18
    unenriched sections are 13-157 characters of document furniture and table fragments that the
    ``section_enrich_min_chars`` rule correctly declined; 3 were offered, called and lost. Folding
    them together would report a 34% processing failure for a document with a 5.7% one.

    ``failure_classes`` carries the PARSER's codes verbatim -- ``TRUNCATED``,
    ``UNUSABLE_OUTPUT``, ``INVALID_JSON``, ``INVALID_SCHEMA``, ``GATEWAY_FAILURE`` -- and not a
    coarser pair. They do not share a fix, and collapsing five into two at the persistence layer
    would restore exactly the ambiguity :class:`EnrichParse` was given reason codes to remove."""

    content_hash: str = ""
    sections_total: int = 0
    sections_offered: int = 0
    sections_enriched: int = 0
    sections_skipped_policy: int = 0
    sections_failed: int = 0
    #: Of the failures, how many were refused for a PARTIALLY parsed set of windows
    #: (:func:`merge_windows`). Stored and not merely logged for the same reason the rest of this
    #: is: the distinction is "this section is readable and one window ran out of room" versus
    #: "this section could not be read at all", the fixes differ, and the ingest log is gone by
    #: the time anyone reads an answer. Always <= `sections_failed`.
    sections_partial: int = 0
    failure_classes: Mapping[str, int] = field(default_factory=dict)
    failed_section_ids: tuple[str, ...] = field(default_factory=tuple)
    #: True when more sections were enrichable than `section_enrich_max_sections` allowed. Those
    #: were never offered to a model at all, which is neither a policy skip nor a failure.
    capped: bool = False
    manifest_version: int = MANIFEST_VERSION

    @property
    def complete(self) -> bool:
        """True when nothing was lost to a failure or a cap.

        Policy skips do not make a document incomplete — they are intentional exclusions. A CAP
        does: those sections were enrichable and were never read."""
        return self.sections_failed == 0 and not self.capped

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "content_hash": self.content_hash,
            "sections_total": self.sections_total,
            "sections_offered": self.sections_offered,
            "sections_enriched": self.sections_enriched,
            "sections_skipped_policy": self.sections_skipped_policy,
            "sections_failed": self.sections_failed,
            "sections_partial": self.sections_partial,
            "failure_classes": dict(self.failure_classes),
            "failed_section_ids": list(self.failed_section_ids),
            "capped": self.capped,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> EnrichmentManifest | None:
        """A stored manifest, or None when there isn't a readable one.

        None for absent, for a non-mapping, and for a version this code does not understand. NOT
        a default-valued manifest: an empty one would read as a document with zero sections and
        nothing failed, which is indistinguishable from a successful ingest of an empty document
        — and a missing manifest must never be presented as a completeness guarantee."""
        if not isinstance(data, Mapping):
            return None
        version = data.get("manifest_version")
        if version != MANIFEST_VERSION:
            return None
        classes = data.get("failure_classes")
        ids = data.get("failed_section_ids")
        return cls(
            content_hash=str(data.get("content_hash") or ""),
            sections_total=int(data.get("sections_total") or 0),
            sections_offered=int(data.get("sections_offered") or 0),
            sections_enriched=int(data.get("sections_enriched") or 0),
            sections_skipped_policy=int(data.get("sections_skipped_policy") or 0),
            sections_failed=int(data.get("sections_failed") or 0),
            # Absent in a manifest written before partial-window refusal existed. 0 is the honest
            # read of that: such a run could not have refused a partial, it MERGED it -- so the
            # section is counted in `sections_enriched` and this says nothing about it either way.
            sections_partial=int(data.get("sections_partial") or 0),
            failure_classes={str(k): int(v) for k, v in classes.items()} if isinstance(classes, Mapping) else {},
            failed_section_ids=tuple(str(x) for x in ids) if isinstance(ids, (list, tuple)) else (),
            capped=bool(data.get("capped")),
        )

    def matches(self, content_hash: str) -> bool:
        """True when this manifest describes the bytes the caller holds.

        An empty stored hash never matches: a manifest that cannot say which bytes it describes
        is not evidence about the document in front of you."""
        return bool(self.content_hash) and self.content_hash == content_hash


@dataclass(frozen=True, slots=True)
class ProcessingCompleteness:
    """Whether the source material behind an answer was successfully processed. A FOURTH
    dimension, independent of the three that already exist.

    Grounding asks whether claims are tied to evidence. Faithfulness asks how verification judged
    them. Verification degradation asks whether the intended checker was available. THIS asks
    whether the document was readable in the first place -- and an answer can be grounded in every
    citation it carries and still be incomplete because three other sections failed at ingest.
    So this must never move ``AnswerStatus``, a faithfulness score, a verification method or a
    degradation flag, and none of those may move this.

    ``status`` is a SCALAR on purpose. A per-document composite on the answer contract would
    repeat the mistake verification nearly made: the per-document detail belongs in diagnostics,
    and the answer carries one status plus aggregate counts."""

    status: str = "UNKNOWN"
    #: Where the assessed scope came from, so the limitation is inspectable: `resolved_doc_ids`,
    #: `cited_documents`, `mixed`, or `none`.
    scope_basis: str = "none"
    documents_assessed: int = 0
    documents_partial: int = 0
    #: Assessed documents with no readable, CURRENT manifest. Kept even when the status is
    #: PARTIAL, because "one document failed and two others we cannot speak for" is a different
    #: situation from "one document failed and the rest are clean".
    documents_unknown: int = 0
    sections_offered: int = 0
    sections_failed: int = 0
    failure_classes: Mapping[str, int] = field(default_factory=dict)
    failed_section_ids: tuple[str, ...] = field(default_factory=tuple)

    def disclosure(self) -> str:
        """The count-only sentence an answer may carry, or "" when there is nothing to disclose.

        NAMES NO SECTION AND NO DOCUMENT. The platform knows which section RECORDS failed; it does
        not have successfully extracted evidence about what those sections CONTAIN, and putting a
        heading in the answer body would be a claim about missing content drawn from the absence
        of that content. The names belong in the diagnostic view, labelled as source-derived
        section metadata whose content was never enriched."""
        if self.status == "PARTIAL":
            return (
                f"Section enrichment failed for {self.documents_partial} of "
                f"{self.documents_assessed} documents in scope. {self.sections_failed} of "
                f"{self.sections_offered} sections submitted for enrichment could not be "
                f"processed successfully. This answer may omit information contained in those "
                f"sections."
            )
        if self.status == "UNKNOWN":
            return (
                "The processing history of this answer's source documents could not be "
                "established, so its completeness cannot be confirmed."
            )
        return ""


def assess_processing(
    *,
    scoped_doc_ids: Iterable[str] = (),
    cited_doc_ids: Iterable[str] = (),
    manifests: Mapping[str, tuple[EnrichmentManifest, str]] | None = None,
) -> ProcessingCompleteness:
    """Assess the documents behind one answer. Pure; no I/O.

    THE SCOPE IS THE UNION of the resolved ``doc_ids`` and the documents the answer actually
    cited, deduplicated. Neither alone is right. Selection cannot be the basis -- a scoped
    document whose every section failed enrichment contributes no evidence and would vanish from
    its own assessment, which is exactly the failure this dimension exists to surface -- and the
    whole knowledge base cannot be either, because one partial document would then mark every
    answer the KB ever gives, which trains a reader to ignore the warning.

    THE LIMIT OF THAT CHOICE, stated rather than hidden. For an UNSCOPED question the cited set is
    not the universe of documents that could have contributed, so ``COMPLETE`` means "every
    assessed source document has a complete manifest" and never "nothing in this KB was missed".
    ``scope_basis`` records which basis was used so the claim can be read correctly. Closing that
    gap needs the whole-document reader to report the inventory it considered, including what it
    could not process -- a different change on a different path.

    A MANIFEST MUST BE CURRENT TO COUNT. One whose ``content_hash`` does not match the document's
    is a previous ingest's result, and a stale success must not outlive a re-ingest that failed;
    it reads as unknown, not as clean.

    KNOWN FAILURE BEATS UNKNOWN. A mixture reports PARTIAL -- there is a definite problem to
    disclose -- and keeps the unknown count so the mixture is still visible."""
    scoped = {d for d in scoped_doc_ids if d}
    cited = {d for d in cited_doc_ids if d}
    assessed = sorted(scoped | cited)
    basis = "mixed" if scoped and cited else "resolved_doc_ids" if scoped else "cited_documents" if cited else "none"
    if not assessed:
        return ProcessingCompleteness(status="UNKNOWN", scope_basis=basis)

    stored = manifests or {}
    partial = unknown = offered = failed = 0
    classes: dict[str, int] = {}
    failed_ids: list[str] = []
    for doc_id in assessed:
        entry = stored.get(doc_id)
        if entry is None or not entry[0].matches(entry[1]):
            unknown += 1
            continue
        manifest = entry[0]
        offered += manifest.sections_offered
        if not manifest.complete:
            partial += 1
            failed += manifest.sections_failed
            failed_ids.extend(manifest.failed_section_ids)
            for code, n in manifest.failure_classes.items():
                classes[code] = classes.get(code, 0) + n
    status = "PARTIAL" if partial else "UNKNOWN" if unknown else "COMPLETE"
    return ProcessingCompleteness(
        status=status,
        scope_basis=basis,
        documents_assessed=len(assessed),
        documents_partial=partial,
        documents_unknown=unknown,
        sections_offered=offered,
        sections_failed=failed,
        failure_classes=classes,
        failed_section_ids=tuple(failed_ids),
    )
