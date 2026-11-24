"""Generation by map then reduce — RAG v2 Phase 9 (pure logic; the gateway calls are in rag_clients).

Phase 8 decides WHICH sections an aggregate answer must be built from. This decides how to read
them. One extraction call per selected section (the MAP), then one call that writes the answer
from the extracted items and nothing else (the REDUCE).

WHY NOT ONE CALL OVER THE SELECTED SECTIONS. The plan's §0 opens on it and the AEGCL corrigendum
is the case: 38 selected sections of an 80-page tender are far past any context window worth
trusting, and a model given all of them at once does not fail loudly — it answers from the first
few and the last few, and the requirement in the middle is simply absent from an answer that
reads complete. Map-reduce turns "the model did not read it" into "no item came back from that
section", which is a COUNT, and a count is what the coverage statement reports.

THE QUOTE IS THE CONTRACT. Every item carries a verbatim span of the section it came from,
checked with :func:`~rag_core.graph.quote_key` — the same normalisation Phase 3 validates a
relation with, Phase 4 validates an enrichment with and Phase 6 anchors with. An item whose quote
is not in the section is DROPPED, not repaired. That is what makes the reduce step's "use ONLY
the items" enforceable rather than hopeful, and it is also how a citation is resolved: the quote
locates the CHUNK the item came from (:func:`resolve_item_chunk`), so a map-reduce answer cites
real chunk ids through the same machinery every other answer uses.

THE SECTION TEXT IS CHUNK TEXT, which is a deliberate deviation from the plan's "section text
(tables as HTML)". ``blocks.text`` and the text behind ``sections.text_hash`` are UNMASKED (they
are captured pre-masking, by design — see ``pipeline._section_enrichment``), while ``Chunk.text``
is masked at ingest under FR-G2. A map prompt is a model call over document text, so it gets the
masked form, exactly as the Phase-3 and Phase-4 prompts do. The cost is that an HTML table
arrives as its chunked text rather than as markup; the alternative is sending unmasked PII to a
model on the query path, which is not a trade this codebase makes anywhere else.

NOTHING HERE CALLS A MODEL. Prompts in, replies in, validated items out — so the parse, the quote
rule, the coverage arithmetic and the citation resolution are all assertable without a gateway.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rag_core.articulation import line_rules
from rag_core.graph import quote_key
from rag_core.json_reply import extract_json_array
from rag_core.prompts import render_prompt
from rag_core.section_enrich import KeyValue

__all__ = [
    "MAP_VERSION",
    "Coverage",
    "MapCacheEntry",
    "MapParse",
    "MappedItem",
    "ReduceInput",
    "amendment_lines",
    "assemble_answer",
    "build_map_prompt",
    "build_reduce_prompt",
    "group_by_facet",
    "items_block",
    "map_cache_key",
    "parse_map_items",
    "resolve_item_chunk",
]

# The version of the EXTRACTION CONTRACT — this prompt, this parser, this item shape. It is part
# of the `map_cache` key, so a build whose extraction changed cannot read the previous build's
# cached items back and present them as its own.
MAP_VERSION = 1

# Bounds. Guards against a model reply, not tuning knobs: every one of these lands in a cache row
# or a prompt on the latency path.
# PER WINDOW REPLY, not per section: `parse_map_items` validates one window's reply, so a
# section split into five windows may keep up to five times this. Named per-section until
# 2026-10-27, when a cap experiment measured 20 of 35 sections hitting it and the misreading
# mattered: at a 6,000-token map cap the bound discarded 325 distinct evidence spans that no
# other window recovered — more than the larger cap gained. It is the binding constraint now.
_MAX_ITEMS_PER_WINDOW = 40
_MAX_STATEMENT_CHARS = 600
_MAX_QUOTE_CHARS = 800
# How many rejected quotes one section keeps for diagnosis. A sample, not a log: these ride in a
# cache row and the point is to see the SHAPE of the failures, which a handful shows as well as all.
_MAX_DROPPED_SAMPLE = 5
_MAX_VALUES_PER_ITEM = 12
_MAX_VALUE_CHARS = 200
_MAX_CLAUSE_REF_CHARS = 60


@dataclass(frozen=True, slots=True)
class MappedItem:
    """One extracted item: something the section says that the question asked about.

    ``item_id`` is what the reduce prompt cites and what maps back to a chunk. It is short and
    opaque (``m1``, ``m2``) rather than a section id, because the model has to copy it into the
    answer verbatim and a 64-character hash is a thing models truncate.

    ``quote`` is verbatim from the section and has already been checked — an item that reaches
    this type is one whose quote was found. ``statement`` is the model's prose and is NOT
    verbatim; it is what the answer is written from, and the quote is what makes it checkable.

    ``old_value``/``new_value`` carry the plan's "old → new" for an amendment. ``old_value`` is
    None where the section states the new value without naming the old, which is the ordinary
    shape of a corrigendum clause that simply restates a requirement."""

    item_id: str
    section_id: str
    doc_id: str
    facet_id: str
    statement: str
    quote: str
    values: tuple[KeyValue, ...] = ()
    clause_ref: str | None = None
    is_amendment: bool = False
    old_value: str | None = None
    new_value: str | None = None

    def as_json(self) -> dict[str, object]:
        """The shape stored in ``map_cache.items``."""
        return {
            "item_id": self.item_id,
            "section_id": self.section_id,
            "doc_id": self.doc_id,
            "facet_id": self.facet_id,
            "statement": self.statement,
            "quote": self.quote,
            "values": [{"name": v.name, "value": v.value, "unit": v.unit} for v in self.values],
            "clause_ref": self.clause_ref,
            "is_amendment": self.is_amendment,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> MappedItem:
        """Rebuild a cached item. The inverse of :meth:`as_json`, and the reason a cache row can
        be read back at all — the quote was validated when it was written, so a cached item does
        not need re-validating against text that has not changed (the cache key includes the
        section's own id, and ``replace_document_structure`` purges the cache for a section whose
        text it replaces)."""
        stored = raw.get("values", [])
        values = tuple(
            KeyValue(name=str(v.get("name", "")), value=str(v.get("value", "")), unit=_opt(v.get("unit")), quote="")
            for v in (stored if isinstance(stored, list) else [])
            if isinstance(v, Mapping)
        )
        return cls(
            item_id=str(raw.get("item_id", "")),
            section_id=str(raw.get("section_id", "")),
            doc_id=str(raw.get("doc_id", "")),
            facet_id=str(raw.get("facet_id", "")),
            statement=str(raw.get("statement", "")),
            quote=str(raw.get("quote", "")),
            values=values,
            clause_ref=_opt(raw.get("clause_ref")),
            is_amendment=bool(raw.get("is_amendment", False)),
            old_value=_opt(raw.get("old_value")),
            new_value=_opt(raw.get("new_value")),
        )


@dataclass(frozen=True, slots=True)
class MapCacheEntry:
    """One ``map_cache`` row, as the catalog writes it (plan §9.1).

    A dataclass rather than a tuple because the five fields are easy to transpose — two of them
    are opaque hex strings — and a silently swapped ``cache_key`` and ``query_sig`` would make
    every lookup a miss with nothing failing."""

    cache_key: str
    section_id: str
    query_sig: str
    model_version: str
    items: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class MapParse:
    """What one section's extraction produced, and what it cost.

    The counts are the point. ``dropped_unquoted`` is the plan's own validation rule firing, and
    a section where it is high is a section the model paraphrased instead of quoting — which is
    invisible in the answer and obvious here.

    ``dropped_over_cap`` is the same idea for :data:`_MAX_ITEMS_PER_WINDOW`. That bound exists to
    stop a runaway reply, and until it was measured nobody knew whether it ever fired: it does.
    On a 3,000-character window of a dense specification table this model returns 40, 40, 40 items
    across three runs — a figure that round and that repeatable is a ceiling being hit, not a model
    deciding it is finished. Every item past it was discarded with no trace at all, which is the
    one thing a bound on the latency path must never do quietly. HOW HARD IT BINDS DEPENDS ON THE
    OUTPUT BUDGET: at the shipped 4,000-token map cap it cost 13 distinct spans over two runs, and
    at 6,000 it cost 373 — raising the cap moved evidence from one ceiling to another rather than
    into the answer."""

    items: tuple[MappedItem, ...] = ()
    error: str = ""
    seen: int = 0
    dropped_unquoted: int = 0
    dropped_unknown_facet: int = 0
    dropped_over_cap: int = 0
    unknown_facets: tuple[str, ...] = ()
    #: A bounded SAMPLE of the quotes the validator rejected, kept so the rate can be diagnosed.
    #: A count cannot tell a paraphrase (the rule working) from a verbatim quote lost to a
    #: punctuation difference (the rule misfiring), and those have opposite fixes — one is a
    #: prompt problem and the other costs real evidence. Measured on a real tender at 7.9% of
    #: items, and the kind is NOT an even split: every rejection judged against a whole section
    #: was text that section held — elided with `...`, or a list stem welded to its bullet — and
    #: none was a paraphrase. Keep the text: a count alone cannot show that.
    dropped_quotes: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """Whether the reply parsed at all. An EMPTY list is usable — "nothing in this section is
        relevant" is the expected answer for most sections of a long document, and treating it as
        a failure would make the coverage statement count a correct extraction as a loss."""
        return not self.error


@dataclass(frozen=True, slots=True)
class Coverage:
    """The plan's §9.2 coverage statement, as numbers before it is a sentence.

    WHAT IT IS FOR: an aggregate answer over part of a document must not read like an aggregate
    answer over all of it. These three numbers are the difference, and the reduce prompt is
    required to end with them."""

    used: int = 0
    selected: int = 0
    total: int = 0
    partial: int = 0
    #: Selected sections whose extraction produced nothing usable — a timeout, an unreachable
    #: endpoint, or a reply that would not parse twice running. They are already absent from
    #: ``used``, which is where the arithmetic stops being enough: a section that FAILED and a
    #: section that was read and held nothing relevant subtract from ``used`` identically, and
    #: the reader cannot tell "this document has nothing more to say on it" from "we did not
    #: manage to ask". Those two deserve different trust in the same answer.
    failed: int = 0

    def line(self) -> str:
        """The sentence the answer ends with.

        ``partial`` is an ADDITION to the plan's wording, and it closes a gap the AEGCL run
        exposed in this implementation rather than in the plan. A section longer than the map's
        window bound is read only as far as the bound allows, and it still counts as having
        CONTRIBUTED — which is true, and on its own overstates the answer. Three of the
        corrigendum's thirty-eight selected sections were in that state, and what went unread
        were real requirements (the ITSM certification, the threat-feed clauses). Saying
        "contributed" about a section the model saw half of, with no qualifier, is the same
        class of silence this whole phase exists to remove.

        ``failed`` is the same addition for the same reason, one step further out. A degraded
        endpoint cost 28 of 63 map calls across 11 sections during the scheduler benchmark, and
        the answer that came back read exactly like an answer over a document with less in it.
        An INCOMPLETE result has to say so in the sentence the reader actually sees, not only in
        a log line the reader never will."""
        line = (
            f"Coverage: {self.used} of {self.selected} selected sections contributed; {self.total} sections in scope."
        )
        if self.partial:
            line += f" {self.partial} of the selected sections were too long to read in full."
        if self.failed:
            line += (
                f" This answer is INCOMPLETE: {self.failed} of the selected sections could not be "
                "read at all, so anything they contain is missing here."
            )
        return line


@dataclass(slots=True)
class ReduceInput:
    """Everything the reduce call needs, assembled from the map results."""

    query: str
    profiles: str
    items: list[MappedItem] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    answer_lang: str = "en"


def _opt(value: object) -> str | None:
    """A model's ``null``, ``""`` and missing key all mean "not stated"."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def map_cache_key(section_id: str, query_sig: str, model_version: str) -> str:
    """The plan's `(section_id, hash(query + facets), model_version)` cache key, as one string.

    HASHED, not concatenated: ``section_id`` is already a hash and ``query_sig`` is derived from
    free text, so a delimiter-joined key would be unbounded and would collide the moment a query
    contained the delimiter. ``MAP_VERSION`` is mixed in so a build whose extraction contract
    changed cannot read the previous build's items and present them as its own."""
    raw = f"{MAP_VERSION}\x00{section_id}\x00{query_sig}\x00{model_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def query_signature(query: str, facets: Sequence[str]) -> str:
    """The plan's `hash(query + facets)`.

    The FACETS are in it because they change what the map call asks for: the same question
    against a document whose facet vocabulary has been re-derived is a different extraction, and
    a cache that ignored them would serve items grouped under labels that no longer exist."""
    joined = "\x00".join(sorted(set(facets)))
    return hashlib.sha256(f"{query.strip().casefold()}\x00{joined}".encode()).hexdigest()[:32]


def build_map_prompt(*, profile: str, breadcrumb: str, section_text: str, query: str, facet_ids: Sequence[str]) -> str:
    """The plan's §9.1 map prompt, rendered from ``prompts/map_extract.md``."""
    return render_prompt(
        "map_extract",
        profile=profile or "(none)",
        breadcrumb=breadcrumb or "(untitled section)",
        section_text=section_text,
        query=query,
        facet_ids=", ".join(facet_ids) or "(none)",
    )


def parse_map_items(
    reply: str,
    *,
    section_id: str,
    doc_id: str,
    section_text: str,
    facet_ids: Sequence[str],
    start_index: int = 1,
) -> MapParse:
    """Validate one map reply into items (plan §9.1: "validate quotes, drop invalid items and log").

    THE QUOTE RULE IS THE WHOLE GATE. An item whose ``quote`` is not in ``section_text`` after
    :func:`~rag_core.graph.quote_key` normalisation is dropped. Not repaired, not kept with a
    weaker quote, not kept with the quote blanked: the reduce step writes the answer from these
    items and claims to use nothing else, so an item the section does not actually say is a
    sentence the answer will assert and no page of the document supports.

    AN UNKNOWN FACET DROPS THE FACET, NOT THE ITEM. A label the document's taxonomy does not
    define cannot be a grouping heading, but the item's statement and quote are still evidence —
    so the facet is cleared to "" and the item survives, where Phase 3 and Phase 4 drop the whole
    entry. The asymmetry is deliberate: there, a bad facet would be WRITTEN to the catalog and
    joined against later; here it only decides which heading a bullet sits under.

    AN EMPTY LIST IS A VALID REPLY and the expected one for most sections.
    """
    body = extract_json_array(reply)
    if body is None:
        return MapParse(error="no JSON array in the reply")
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        return MapParse(error=f"unparseable JSON: {str(exc)[:80]}")
    if not isinstance(raw, list):
        return MapParse(error="the reply is not a list")

    haystack = quote_key(section_text)
    known = set(facet_ids)
    items: list[MappedItem] = []
    unquoted = unknown_facet = 0
    rejected: list[str] = []
    unknown: dict[str, None] = {}
    over_cap = max(0, len(raw) - _MAX_ITEMS_PER_WINDOW)
    for entry in raw[:_MAX_ITEMS_PER_WINDOW]:
        if not isinstance(entry, Mapping):
            continue
        quote = str(entry.get("quote", "")).strip()[:_MAX_QUOTE_CHARS]
        statement = str(entry.get("statement", "")).strip()[:_MAX_STATEMENT_CHARS]
        if not statement:
            continue
        if not quote or quote_key(quote) not in haystack:
            unquoted += 1
            if len(rejected) < _MAX_DROPPED_SAMPLE:
                rejected.append(quote)
            continue
        facet = str(entry.get("facet_id") or "").strip()
        if facet and facet not in known:
            unknown.setdefault(facet, None)
            unknown_facet += 1
            facet = ""
        items.append(
            MappedItem(
                item_id=f"m{start_index + len(items)}",
                section_id=section_id,
                doc_id=doc_id,
                facet_id=facet,
                statement=statement,
                quote=quote,
                values=_values(entry.get("values")),
                clause_ref=(_opt(entry.get("clause_ref")) or "")[:_MAX_CLAUSE_REF_CHARS] or None,
                is_amendment=bool(entry.get("is_amendment", False)),
                old_value=_opt(entry.get("old_value")),
                new_value=_opt(entry.get("new_value")),
            )
        )
    return MapParse(
        items=tuple(items),
        seen=len(raw),
        dropped_unquoted=unquoted,
        dropped_unknown_facet=unknown_facet,
        dropped_over_cap=over_cap,
        unknown_facets=tuple(unknown),
        dropped_quotes=tuple(rejected),
    )


def _values(raw: object) -> tuple[KeyValue, ...]:
    """The item's figures. ``quote`` is empty on these: the ITEM carries the quote, and a value
    is a field of it rather than an independently evidenced claim — unlike Phase 4's
    :class:`KeyValue`, where each one is quoted because each is stored on its own."""
    if not isinstance(raw, list):
        return ()
    out: list[KeyValue] = []
    for entry in raw[:_MAX_VALUES_PER_ITEM]:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "")).strip()[:_MAX_VALUE_CHARS]
        value = str(entry.get("value", "")).strip()[:_MAX_VALUE_CHARS]
        if not name or not value:
            continue
        unit = _opt(entry.get("unit"))
        out.append(KeyValue(name=name, value=value, unit=unit[:_MAX_VALUE_CHARS] if unit else None, quote=""))
    return tuple(out)


def items_block(items: Sequence[MappedItem], breadcrumbs: Mapping[str, str] | None = None) -> str:
    """The extracted items as the reduce prompt sees them.

    ONE LINE PER ITEM, leading with the ``item_id`` the model must cite. The breadcrumb is on the
    line because the plan's reduce rules ask the answer to say where a merged requirement
    applies, and the model cannot do that from an opaque id."""
    crumbs = breadcrumbs or {}
    lines: list[str] = []
    for item in items:
        parts = [f"[{item.item_id}]"]
        if crumbs.get(item.section_id):
            parts.append(f"({crumbs[item.section_id]})")
        if item.facet_id:
            parts.append(f"facet={item.facet_id}")
        if item.clause_ref:
            parts.append(f"clause={item.clause_ref}")
        parts.append(item.statement)
        if item.values:
            rendered = "; ".join(f"{v.name}={v.value}{' ' + v.unit if v.unit else ''}" for v in item.values)
            parts.append(f"values: {rendered}")
        if item.is_amendment:
            parts.append(f"AMENDMENT: {item.old_value or '(unstated)'} -> {item.new_value or '(unstated)'}")
        parts.append(f'quote: "{item.quote}"')
        lines.append(" ".join(parts))
    return "\n".join(lines)


#: The heading an item with no facet is written under. A map call may legitimately return an item
#: it could not label (``parse_map_items`` clears an unknown facet rather than dropping the item),
#: and those are evidence like any other — they just have no topic of their own.
UNFACETED = "other"


def group_by_facet(items: Sequence[MappedItem]) -> list[tuple[str, list[MappedItem]]]:
    """Items grouped under their facet, facets in first-appearance order, unfaceted last.

    FIRST-APPEARANCE, not alphabetical: the items arrive in document order, so the facets come
    out in the order the document introduces them, and the assembled answer reads in roughly the
    order a reader of the document would meet the topics."""
    groups: dict[str, list[MappedItem]] = {}
    for item in items:
        groups.setdefault(item.facet_id or UNFACETED, []).append(item)
    ordered = [(facet, got) for facet, got in groups.items() if facet != UNFACETED]
    if UNFACETED in groups:
        ordered.append((UNFACETED, groups[UNFACETED]))
    return ordered


def build_reduce_prompt(
    data: ReduceInput, facet: str, items: Sequence[MappedItem], breadcrumbs: Mapping[str, str] | None = None
) -> str:
    """The reduce prompt for ONE facet, rendered from ``prompts/reduce.md``.

    ONE CALL PER FACET, which is a deliberate deviation from the plan's §9.2 single reduce and
    was forced by measuring it. On the real AEGCL corrigendum the map pass returns 322 items
    from 34 sections; rendered into one prompt that is 107,000 characters, and the answer the
    model wrote from it was cut off mid-word with the coverage statement — the last line the
    plan's own rules ask for — never reached. Half the document's requirements were simply
    absent from an answer that read complete, which is the exact failure map-reduce exists to
    prevent, reintroduced at the reduce step.

    Per facet, each prompt is bounded by how much the document says about ONE topic, the calls
    run in parallel, and the plan's own rule 2 already asked for the output to be grouped by
    facet — so this produces the same document shape with a bound on each piece.

    THE LINE-LEVEL PRESENTATION CONTRACT COMES FROM ``articulation``, the same text
    ``grounding.build_prompt`` renders. It was missing here, and the cost was measured on the
    AEGCL corrigendum: against the single-pass answer over the same corpus this path produced
    **0 bold spans against 587**, 16.8 words per sentence against 9.4, and 2.8 duplicate lines
    against none. Nothing about the evidence was worse — the rules simply were not in the prompt.

    Only the LINE tier is injected. The DOCUMENT tier (which container the answer takes, where a
    heading goes) belongs to whoever can see the whole answer, which here is
    :func:`assemble_answer` in code, not this call: rule 1 already forbids this call a heading,
    and a facet cannot choose a shape for an answer it sees one topic of."""
    return render_prompt(
        "reduce",
        profiles=data.profiles or "(none)",
        query=data.query,
        facet=facet.replace("_", " "),
        items=items_block(items, breadcrumbs) or "(no items)",
        answer_lang=data.answer_lang or "en",
        # Numbered from 7 because the prompt's own rules stop at 6 and this is one contract, not
        # a second list; "the items" because this prompt has no SOURCES section to copy from.
        presentation=line_rules(start=7, sources="the items"),
    )


#: How a disposition reads when the amendment withdrew a requirement rather than replacing it.
_WITHDRAWN = "withdrawn"

#: What is said when the amendment's OTHER side is absent from the extracted items. It describes
#: the EVIDENCE, not the document: see `amendment_lines`. Phrased so it reads the same whether the
#: missing side is the thing replaced or the thing withdrawn, and without the word "amended",
#: which the block's own heading already supplies.
_UNCAPTURED = " (the previous wording is not in the extracted evidence)"


def amendment_lines(items: Sequence[MappedItem]) -> list[str]:
    """One line per amendment, built from the ITEMS rather than asked of the model.

    WHY THIS IS CODE. The relationship is already structured data -- `is_amendment`, `old_value`,
    `new_value` on each item -- and asking a model to carry it through a summarisation step is the
    weaker guarantee, which is the same argument :func:`assemble_answer` already makes about the
    coverage line. Measured, twice, on the real corrigendum: the reduce call is handed
    `AMENDMENT: <old> -> <new>` verbatim in its prompt and rule 3 tells it to show the change, and
    for the two amendments whose sides are SHORT it does -- "₹35 Crores → ₹19.44 Crores", "... is
    withdrawn". For the one whose sides are two ~200-character competing requirements it reports
    the replacement and silently drops what it replaced, in both attempts. The current
    requirement is right; the fact that it superseded something is gone, which for a bidder
    reading a tender is the part that matters.

    NOTHING IS INVENTED HERE, AND THAT INCLUDES THE ABSENCES. An item with no `old_value` is
    reported as missing EVIDENCE, not as a missing requirement -- "previous wording not captured"
    rather than "previously unstated". Those are different claims: the second is a statement about
    the source document, and an empty `old_value` does not establish it. It means the map did not
    capture the previous wording, which happens both when the document never stated one and when
    extraction simply missed it, and this renderer cannot tell those apart. Asserting the stronger
    one would be the deterministic path inventing history while trying to stop the model inventing
    it -- the same error one layer down. Filling the value in would be worse still.

    Every line carries its own `item_id` markers, so these resolve to real citations through the
    same `_MarkerResolver` the body uses and the grounded-or-abstain invariant is untouched."""
    # SEVERAL ITEMS CAN DESCRIBE ONE AMENDMENT. The map returns one item per distinct requirement,
    # so a long amendment arrives as two or three items carrying the SAME old -> new pair; they
    # are one change to a reader and must not become three bullets saying it three times.
    grouped: dict[tuple[str, str], list[str]] = {}
    for item in items:
        if not item.is_amendment:
            continue
        old = (item.old_value or "").strip()
        new = (item.new_value or "").strip()
        if not new:
            continue  # nothing to report: no replacement and no disposition
        grouped.setdefault((old, new), []).append(item.item_id)
    lines = []
    for (old, new), ids in grouped.items():
        marker = "[" + ",".join(ids) + "]"
        if new.lower() == _WITHDRAWN:
            body = f"**{old}** — withdrawn" if old else f"A requirement was withdrawn{_UNCAPTURED}"
        elif old:
            body = f"**{old}** → **{new}**"
        else:
            body = f"**{new}**{_UNCAPTURED}"
        lines.append(f"- {body} {marker}")
    return lines


def assemble_answer(data: ReduceInput, sections: Sequence[tuple[str, str]]) -> str:
    """The per-facet sections, assembled into the answer the plan's §9.2 describes.

    THE FRAME IS CODE, NOT INSTRUCTION, and that is the point of assembling rather than asking
    one call to do everything. The plan's rule 1 (what the document is) and rule 7 (the coverage
    statement) are both FACTS the caller already holds — making the model restate them is a
    weaker guarantee than printing them, and when the single-call reduce ran out of budget the
    coverage line was the first thing lost. Printed here, it cannot be."""
    merged: list[tuple[str, list[str]]] = []
    for facet, text in sections:
        if not text.strip():
            continue
        # SEVERAL CALLS CAN SHARE A FACET. A facet whose items do not fit one prompt is reduced
        # in batches (see the caller), and each batch comes back as its own section — so the
        # bullets are concatenated under ONE heading rather than repeating the heading per
        # batch, which would tell the reader about the token budget instead of the document.
        if merged and merged[-1][0] == facet:
            merged[-1][1].append(text.strip())
        else:
            merged.append((facet, [text.strip()]))
    body = "\n\n".join(f"### {facet.replace('_', ' ')}\n" + "\n".join(parts) for facet, parts in merged)
    opening = data.profiles.strip()
    if not body:
        return ""
    # THE AMENDMENTS SECTION IS PRINTED, NOT REQUESTED -- see `amendment_lines`. It is its own
    # block rather than woven into the facet sections because an amendment belongs to the
    # DOCUMENT's history, not to one topic: the same change can bear on several facets, and a
    # reader checking whether their assumption is stale should not have to find the right heading
    # first.
    amendments = amendment_lines(data.items)
    change_log = "\n".join(["### amendments in this document", *amendments]) if amendments else ""
    parts = [opening, body, change_log, data.coverage.line()]
    return "\n\n".join(p for p in parts if p)


def resolve_item_chunk(item: MappedItem, chunks: Sequence[tuple[str, str]]) -> str | None:
    """The chunk id whose text contains this item's quote, or None (plan §9.2's "map citation
    markers back to chunk IDs/pages via existing quote-based citation resolution").

    ``chunks`` is ``[(chunk_id, text), ...]`` for the item's own section, in document order, and
    the FIRST containing chunk wins — a quote spanning a chunk boundary is attributed to where it
    starts, which is where a reader clicking the citation should land.

    None is a real outcome, not a failure to handle elsewhere: the quote was validated against
    the SECTION text, which is the concatenation of its chunks, so a quote that straddles the
    join between two chunks is in the section and in neither chunk. The caller drops the
    citation, not the item — see ``pipeline._map_citations``."""
    needle = quote_key(item.quote)
    if not needle:
        return None
    for chunk_id, text in chunks:
        if needle in quote_key(text):
            return chunk_id
    return None
