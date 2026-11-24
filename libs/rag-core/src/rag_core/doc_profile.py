"""The document profile card — RAG v2 Phase 3 (pure logic; the gateway call is in rag_clients).

One model call per document produces a CARD: what kind of document this is, why it exists, who
issued it, what other documents it declares a relationship to, and a facet taxonomy drawn from its
own content. Three consumers depend on it and they want different halves:

* ``doc_relations`` — the declared relations, which Phase 6 already turns into directed graph
  edges anchored to the chunk that proves each one. That consumer landed first and is INERT until
  this module writes its rows (discovery §8.12); this is its producer.
* ``facets`` — the per-document taxonomy Phase 4 assigns sections to and Phase 8 groups an
  aggregate answer by. Facets are per-document BY DESIGN (plan §14), so they are a join key within
  a document, never across a corpus.
* the contextualize prefix — ``doc_type``/``title``/``purpose`` in front of every v2 chunk's
  ``embed_text`` (:func:`profile_card`), which is the slot Phase 5 deliberately left empty rather
  than filling with a placeholder it would later have to re-embed back out.

WHAT THIS MODULE REFUSES TO STORE is as much of the contract as what it keeps. A relation whose
``evidence_quote`` is not in the document is DROPPED, not stored with a weaker quote or anchored
to the nearest text: an "amends" edge is an instruction to a later phase to prefer one clause over
another, and one resting on a quote the document does not contain would make an answer cite text
that does not say it. The match is :func:`rag_core.graph.quote_key` — the same normalisation
the graph seeding uses to resolve the quote to a chunk, shared so that a relation cannot pass
validation here and silently fail to anchor there.

THE VERDICT IS NEVER PARTIAL. ``doc_type`` and ``purpose`` are the card; a reply missing either
yields no profile at all rather than a row with NULLs that reads downstream as "this document was
profiled and has no type". Everything else is optional and absent means absent.

NOTHING HERE CALLS A MODEL. :func:`build_doc_profile_prompt` renders the reviewable template in
``prompts/doc_profile.md`` and :func:`parse_doc_profile` validates a reply; the gateway call, its
one retry and its fail-safe live in ``rag_clients.doc_profiler``, matching how every other
"one small JSON call" in this codebase is split (``query_plan`` / ``LLMQueryPlanner``).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from rag_core.graph import RELATION_TYPES, quote_key
from rag_core.json_reply import extract_json_object
from rag_core.prompts import render_prompt

__all__ = [
    "PROFILE_VERSION",
    "DocProfile",
    "ProfileFacet",
    "ProfileParse",
    "ProfileRelation",
    "build_doc_profile_prompt",
    "parse_doc_profile",
    "profile_card",
    "snake_id",
]

# The version of the CARD CONTRACT — this prompt, this parser, this schema — stored in
# `documents.profile_version`.
#
# NOT a per-document counter of how many times the document has been profiled. The plan asks for a
# version "so re-profiling is traceable", and the question an operator actually has to answer is
# "which documents hold a card this build would no longer produce" — `WHERE profile_version < N`
# answers it and selects exactly the corpus to re-profile. A per-document counter answers "how many
# times has this been re-profiled", which selects nothing and triggers no action. Bump this when
# the prompt or the parser changes in a way that would change a stored card.
PROFILE_VERSION = 1

# Bounds. Every one of these is a GUARD against a model reply, not a tuning knob: the reply is
# untrusted input that lands in a Postgres row, in a prompt for Phase 4, and (for the card) inside
# every v2 vector in the document.
_MAX_DOC_TYPE_CHARS = 60
_MAX_PURPOSE_CHARS = 400
_MAX_ISSUER_CHARS = 200
_MAX_AUDIENCE_CHARS = 200
_MAX_LANG_CHARS = 16
_MAX_TARGET_REF_CHARS = 300
_MAX_QUOTE_CHARS = 1000
_MAX_FACET_ID_CHARS = 60
_MAX_FACET_NAME_CHARS = 120
_MAX_FACET_DESC_CHARS = 400
# A hard ceiling on how many of each are kept, well above the prompt's own "6-15 facets", so a
# model that answers with a hundred cannot turn one document into a hundred `facets` rows.
_MAX_FACETS = 30
_MAX_RELATIONS = 30
# The band the prompt asks for. OUTSIDE IT IS REPORTED, NOT REJECTED: the count is prompt guidance
# about usefulness (too few facets group an aggregate answer coarsely), not a correctness
# invariant, and discarding an otherwise-valid card over it would lose the relations with it.
FACET_BAND = (6, 15)

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


@dataclass(frozen=True, slots=True)
class ProfileRelation:
    """One relationship the document declares about itself, validated.

    ``rel_type`` is always one of :data:`rag_core.graph.RELATION_TYPES`; ``evidence_quote`` is
    always present and always found in the document's text. A row that reaches here has already
    passed both checks, so no consumer has to repeat them."""

    rel_type: str
    target_ref: str
    evidence_quote: str


@dataclass(frozen=True, slots=True)
class ProfileFacet:
    """One content area of THIS document. ``facet_id`` is normalised snake_case and unique within
    the profile — it is the key Phase 4 labels sections with and Phase 8 groups by."""

    facet_id: str
    name: str
    description: str
    #: APPROVED aliases for this facet id, carried when the card's taxonomy came from a stored
    #: vocabulary (`rag_core.facet_vocab`). NEVER parsed from a model reply -- an alias is
    #: added only by a controlled vocabulary update, because it is an instruction to JOIN two
    #: labels and a model proposing its own would reintroduce exactly the churn the vocabulary
    #: exists to stop. Empty on every freshly proposed taxonomy.
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DocProfile:
    """A document's profile card (plan §1.1 ``documents`` + §3).

    ``title`` is the PLATFORM's title for the document, not a model output: it is given to the
    prompt, never asked for, so that ``documents.title`` — which `document_relations` reads as the
    source node's name when seeding the graph — can never be something the model invented."""

    doc_type: str
    purpose: str
    title: str
    issuer: str | None = None
    issue_date: date | None = None
    audience: str | None = None
    lang: str | None = None
    relations: tuple[ProfileRelation, ...] = ()
    facets: tuple[ProfileFacet, ...] = ()
    profile_version: int = PROFILE_VERSION

    def as_json(self) -> str:
        """The whole card as the JSON stored in ``documents.profile_json``.

        Stored WHOLE, including the relations and facets that also have their own tables, because
        Phase 4's section-enrichment prompt is handed ``{profile_json}`` as one blob and would
        otherwise have to re-join three tables to rebuild what this call already had in hand. The
        tables remain the queryable form; this is the readable one."""
        return json.dumps(
            {
                "doc_type": self.doc_type,
                "purpose": self.purpose,
                "title": self.title,
                "issuer": self.issuer,
                "issue_date": self.issue_date.isoformat() if self.issue_date else None,
                "audience": self.audience,
                "lang": self.lang,
                "relations": [
                    {"type": r.rel_type, "target_ref": r.target_ref, "evidence_quote": r.evidence_quote}
                    for r in self.relations
                ],
                "facet_taxonomy": [
                    {
                        "facet_id": f.facet_id,
                        "name": f.name,
                        "description": f.description,
                        # Only when there are any, so a card with a freshly proposed taxonomy
                        # serialises exactly as it did before this field existed.
                        **({"aliases": list(f.aliases)} if f.aliases else {}),
                    }
                    for f in self.facets
                ],
                "profile_version": self.profile_version,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class ProfileParse:
    """What :func:`parse_doc_profile` made of a reply, INCLUDING what it threw away.

    Same discipline as :class:`rag_core.graph.RelationSeed`: a dropped relation is reported,
    never silently discarded. A model that declares an "amends" and loses it to a misquoted
    evidence span is the single most consequential failure this phase has — it is the difference
    between a corrigendum that supersedes a clause and one that does not — and it must be visible
    in a log line rather than inferable only from an absent row."""

    profile: DocProfile | None = None
    error: str = ""
    unquoted: tuple[str, ...] = ()  # relations dropped: the evidence quote is not in the document
    unknown_types: tuple[str, ...] = ()  # relations dropped: `type` is not one of the four
    dropped_facets: tuple[str, ...] = ()  # facets dropped: unusable id, or a duplicate of an earlier one
    facet_count_off_band: bool = False  # kept, but outside the prompt's 6-15


def build_doc_profile_prompt(*, title: str, metadata: str, breadcrumbs: str, head_text: str) -> str:
    """Render ``prompts/doc_profile.md`` with this document's inputs (plan §3's four inputs).

    EVERY ARGUMENT MUST ALREADY BE MASKED. The caller assembles these from the document's title,
    its structure tree's breadcrumbs and its opening pages, and the tree is captured PRE-masking
    (see ``rag_core.layout_chunking``'s module docstring) — so the breadcrumbs in particular
    are unmasked unless the caller ran them through the document's own masker. What comes back is
    stored in `documents.purpose`/`profile_json` and embedded into every v2 vector via
    :func:`profile_card`, so an identifier that reaches this prompt reaches all three."""
    return render_prompt(
        "doc_profile",
        title=title.strip() or "(untitled)",
        metadata=metadata.strip() or "(none)",
        breadcrumbs=breadcrumbs.strip() or "(this document has no detected section headings)",
        head_text=head_text.strip(),
    )


def parse_doc_profile(reply: str, *, source_text: str, title: str, lang: str | None = None) -> ProfileParse:
    """Validate a model reply into a :class:`DocProfile`, or explain why there isn't one.

    ``source_text`` is the text an ``evidence_quote`` must be found in — the SAME text the prompt
    was given, not the whole document: a quote the model produced from the opening pages it was
    shown is verifiable, and one it produced from pages it never saw is a fabrication whether or
    not those pages happen to contain it. Passing the full document here would make the check pass
    for exactly the replies it exists to catch.

    ``lang`` is the CALLER's detection and it WINS over the model's, for the same reason ``title``
    is never read off the reply: the platform already answers this question deterministically
    (``rag_core.language.detect_language``) in ISO-639-1, and the model does not — measured
    against this deployment's gemma4-31b on a real document, the reply's ``lang`` came back
    ``"english"``. Storing that in ``documents.lang`` would put a second vocabulary in a column the
    rest of the platform writes codes into, and a filter joining them would silently match nothing.
    The field stays in the prompt because it is in the plan's schema and costs nothing; it is used
    only when the caller supplies no detection of its own.

    Never raises. A reply that is not JSON, not an object, or missing ``doc_type``/``purpose``
    returns a parse with ``profile=None`` and a populated ``error``.
    """
    blob = extract_json_object(reply)
    if blob is None:
        return ProfileParse(error="no JSON object in the reply")
    try:
        data = json.loads(blob)
    except (ValueError, TypeError) as exc:
        return ProfileParse(error=f"malformed JSON: {str(exc)[:120]}")
    if not isinstance(data, dict):
        return ProfileParse(error="the JSON value is not an object")

    doc_type = _text(data.get("doc_type"), _MAX_DOC_TYPE_CHARS)
    purpose = _text(data.get("purpose"), _MAX_PURPOSE_CHARS)
    if not doc_type or not purpose:
        # Both halves of the card, so neither is optional — see the module docstring.
        missing = " and ".join(n for n, v in (("doc_type", doc_type), ("purpose", purpose)) if not v)
        return ProfileParse(error=f"missing {missing}")

    relations, unquoted, unknown = _relations(data.get("relations"), source_text)
    facets, dropped = _facets(data.get("facet_taxonomy"))
    profile = DocProfile(
        # Lowercased: `doc_type` is a FILTER value (plan §1.2 puts a Qdrant payload index on it) and
        # a per-doc_type chunking key, so "Corrigendum" and "corrigendum" must not be two classes.
        doc_type=doc_type.lower(),
        purpose=purpose,
        title=title.strip(),
        issuer=_text(data.get("issuer"), _MAX_ISSUER_CHARS) or None,
        issue_date=_date(data.get("issue_date")),
        audience=_text(data.get("audience"), _MAX_AUDIENCE_CHARS) or None,
        lang=(lang or "").strip().lower() or _text(data.get("lang"), _MAX_LANG_CHARS).lower() or None,
        relations=relations,
        facets=facets,
    )
    return ProfileParse(
        profile=profile,
        unquoted=unquoted,
        unknown_types=unknown,
        dropped_facets=dropped,
        facet_count_off_band=not FACET_BAND[0] <= len(facets) <= FACET_BAND[1],
    )


# The plan's contextualize prefix, whole: "[{doc_type}: {title} — {purpose}] [{breadcrumb}] ".
# This function owns the first bracket; `layout_chunking.embed_prefix` owns the second.
_MAX_CARD_CHARS = 200


def profile_card(profile: DocProfile | None) -> str:
    """The plan §5 card prefix for this document's chunks: ``"[{doc_type}: {title} — {purpose}] "``.

    Empty string for no profile, which is what keeps the v2 chunking path byte-identical to Phase 5
    whenever the profile call did not produce a card — the fail-safe in ``rag_clients.doc_profiler``
    returns None far more often than it should be allowed to change what gets indexed.

    BOUNDED AT :data:`_MAX_CARD_CHARS`, by truncating the purpose. This string is prepended to every
    one of the document's v2 chunks before embedding, so its length is subtracted from every chunk's
    usable budget against an embedder that truncates over-long input SILENTLY (§8.10 — a 512-token
    bge-small answers HTTP 200 and a normal-looking vector). An unbounded one-sentence purpose from
    a model that wrote a paragraph would eat the budget of every chunk in the document at once. The
    caller subtracts this length from its character cap; the cap cannot subtract what has no ceiling.

    ``profile.title`` must already be MASKED, for the same reason ``embed_prefix``'s breadcrumb must
    be: this string is embedded and BM25-indexed."""
    if profile is None:
        return ""
    head = f"{profile.doc_type}: {profile.title}" if profile.title else profile.doc_type
    room = _MAX_CARD_CHARS - len(head) - len("[ — ] ")
    purpose = profile.purpose if len(profile.purpose) <= room else profile.purpose[: max(0, room - 1)].rstrip() + "…"
    if room <= 0 or not purpose:
        return f"[{head[: _MAX_CARD_CHARS - 3]}] "
    return f"[{head} — {purpose}] "


# --- field helpers ------------------------------------------------------------------------------


def _text(value: object, limit: int) -> str:
    """A model's string field, trimmed and capped; "" for anything that is not a usable string."""
    return value.strip()[:limit] if isinstance(value, str) else ""


def _date(value: object) -> date | None:
    """``"YYYY-MM-DD"`` or nothing.

    STRICT, and deliberately not a tolerant date parser. `documents.issue_date` is a DATE column
    that recency filters and supersession ordering read; "24-09-2026" is ambiguous between two real
    calendars and the document that motivated this phase is Indian, where the day comes first and
    the model's training data mostly does not. A wrong date is worse than no date here, because a
    null is visible and a plausible wrong one is not."""
    if not isinstance(value, str) or not (m := _DATE_RE.match(value.strip())):
        return None
    try:
        return date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:  # a syntactically fine but impossible date, e.g. 2026-02-30
        return None


def _relations(value: object, source_text: str) -> tuple[tuple[ProfileRelation, ...], tuple[str, ...], tuple[str, ...]]:
    """Validated relations, plus the ones dropped for an unfindable quote and for an unknown type."""
    if not isinstance(value, list):
        return (), (), ()
    haystack = quote_key(source_text)
    kept: list[ProfileRelation] = []
    unquoted: list[str] = []
    unknown: list[str] = []
    for item in value[:_MAX_RELATIONS]:
        if not isinstance(item, dict):
            continue
        rel_type = _text(item.get("type"), _MAX_DOC_TYPE_CHARS).lower()
        target_ref = _text(item.get("target_ref"), _MAX_TARGET_REF_CHARS)
        quote = _text(item.get("evidence_quote"), _MAX_QUOTE_CHARS)
        if not target_ref or not quote:
            continue
        if rel_type not in RELATION_TYPES:
            # Reported rather than coerced to the nearest type. `linked_chunks` walks edges
            # regardless of label, so a guessed label would become a real graph edge nothing can
            # interpret — the same reasoning `graph.RELATION_TYPES` states for the seeding side.
            unknown.append(rel_type or "(blank)")
            continue
        if quote_key(quote) not in haystack:
            unquoted.append(quote[:120])
            continue
        kept.append(ProfileRelation(rel_type=rel_type, target_ref=target_ref, evidence_quote=quote))
    return tuple(kept), tuple(unquoted), tuple(unknown)


def _facets(value: object) -> tuple[tuple[ProfileFacet, ...], tuple[str, ...]]:
    """Validated facets with normalised ids, plus the ones dropped as unusable or duplicate."""
    if not isinstance(value, list):
        return (), ()
    kept: list[ProfileFacet] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for item in value[:_MAX_FACETS]:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"), _MAX_FACET_NAME_CHARS)
        facet_id = snake_id(_text(item.get("facet_id"), _MAX_FACET_ID_CHARS) or name)
        if not facet_id:
            dropped.append(name or "(unnamed)")
            continue
        if facet_id in seen:
            # FIRST WINS. The id is a primary key (`facets(doc_id, facet_id)`) and the join key
            # Phase 4 labels sections with, so two facets normalising onto one id is a collision,
            # not a merge — keeping the later one would silently redefine what the earlier
            # description said the facet means.
            dropped.append(facet_id)
            continue
        seen.add(facet_id)
        kept.append(
            ProfileFacet(
                facet_id=facet_id,
                name=name or facet_id.replace("_", " "),
                description=_text(item.get("description"), _MAX_FACET_DESC_CHARS),
            )
        )
    return tuple(kept), tuple(dropped)


def snake_id(value: str) -> str:
    r"""``"Financial Qualification"`` -> ``"financial_qualification"``.

    The prompt asks for snake_case and models mostly comply, but "Supply Chain Security" and
    "supply-chain-security" both arrive in practice and both mean the same facet. Normalising here
    rather than trusting the reply is what makes the id usable as a primary key.

    PUBLIC because Phase 4 normalises the facet ids a section is labelled with through the SAME
    function, for the reason :func:`rag_core.graph.quote_key` is shared: a section labelled
    ``supply-chain-security`` must land on the facet the profile stored as
    ``supply_chain_security``, and two normalisations would mean a label that validates here and
    joins to nothing there.

    CHARACTER-WISE RATHER THAN A REGEX, because of one script this corpus actually has. An
    ASCII-only class reduces a Devanagari facet name to the empty string, which would drop the
    whole taxonomy of a Hindi document while reporting each facet as merely "unusable"; and `\w`
    is no better, because it matches letters but NOT the combining marks (`Mn`) that carry a
    Devanagari word's vowels, so "वित्तीय" comes out as "व_त_त_य" — mangled rather than dropped,
    which is worse, since a mangled id is a usable-looking primary key. Keeping alphanumerics AND
    combining marks is the rule that works in every script here; both are legal in a Postgres TEXT
    key and a JSON object key."""
    out: list[str] = []
    for ch in value.strip().lower():
        if ch.isalnum() or unicodedata.category(ch).startswith("M"):
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")[:_MAX_FACET_ID_CHARS].strip("_")
