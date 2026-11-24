"""Knowledge-graph extraction for GraphRAG (FR-C6/D4) — pure logic, no framework.

At ingest each chunk yields ENTITIES (named things) and co-occurrence EDGES (two entities
in the same chunk are related). Accumulated across a KB this is a knowledge graph; local
graph search (FR-D4) then answers multi-hop questions by linking the query's entities and
pulling in the chunks of their graph neighbours — recall a passage-similarity search misses.

Two backends behind one seam (the platform's heuristic-default / LLM-opt-in pattern):
* :class:`HeuristicGraphExtractor` — deterministic, zero-cost, CI-safe: proper-noun phrases
  as entities, co-occurrence as edges. The default; works on the CPU dev tier.
* the LLM backend (in ``rag_clients``) prompts the KB's own model for a richer typed graph;
  this module supplies its prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # `Chunk` is only a type here — importing it at runtime would make rag-core's
    from rag_schemas import Chunk  # graph module depend on the schemas package for no behaviour.

_MAX_ENTITIES_PER_CHUNK = 12
_MIN_ENTITY_LEN = 3
_MAX_ENTITY_WORDS = 5

# Capitalised word runs (proper nouns / titles): "Acme", "Exuberant Customer Support".
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4}\b")

# Capitalised-but-not-an-entity words (sentence openers, headings, filler).
_STOP = frozenset(
    {
        "the",
        "this",
        "that",
        "these",
        "those",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "when",
        "in",
        "on",
        "for",
        "to",
        "of",
        "at",
        "by",
        "with",
        "as",
        "it",
        "we",
        "you",
        "they",
        "he",
        "she",
        "i",
        "is",
        "are",
        "was",
        "were",
        "be",
        "will",
        "shall",
        "may",
        "can",
        "answer",
        "question",
        "section",
        "note",
        "sources",
        "source",
        "policy",
        "please",
        "objective",
        "eligibility",
        "definition",
        "example",
        "summary",
        "introduction",
        # more aux verbs / determiners / connectives that open sentences in dense prose
        "been",
        "being",
        "could",
        "would",
        "should",
        "must",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "yes",
        "any",
        "all",
        "each",
        "every",
        "some",
        "such",
        "both",
        "either",
        "neither",
        "per",
        "upon",
        "during",
        "before",
        "after",
        "above",
        "below",
        "under",
        "over",
        "between",
        "within",
        "without",
        "into",
        "from",
        "than",
        "then",
        "there",
        "here",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "how",
        "why",
        "however",
        "therefore",
        "hence",
        "thus",
        "also",
        "only",
        "provided",
        "following",
        "aforesaid",
        "including",
        "otherwise",
        "accordingly",
        "subject",
        "applicable",
        "respective",
        "relevant",
        # document scaffolding (structural, domain-neutral)
        "notes",
        "table",
        "figure",
        "chapter",
        "part",
        "page",
    }
)
# NOTE: keep this set strictly domain-NEUTRAL — grammar + document-structure words only.
# Never add content nouns (e.g. "employee", "salary", "act", "invoice", "patient"): the
# platform serves every domain, and one domain's filler is another's key entity ("the
# Companies Act"). The LLM extraction backend is the quality path for dense docs.


@dataclass(frozen=True, slots=True)
class GraphTriple:
    """Entities + undirected co-occurrence edges extracted from ONE chunk.

    ``diagram_edges`` is populated ONLY by the diagram-structure parsers below
    (:func:`parse_mermaid_diagram` / :func:`parse_diagram_structure`) — real directed,
    labeled arrows ``(source, target, relation_label)`` read off a flowchart/architecture
    image. It is empty for ordinary text extraction (``edges`` covers that path, unchanged).
    """

    entities: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    diagram_edges: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GraphStats:
    """A KB's knowledge-graph size + its most-connected entities (for the console)."""

    nodes: int
    edges: int
    top_entities: list[tuple[str, int]] = field(default_factory=list)  # (name, mentions)


@runtime_checkable
class GraphExtractor(Protocol):
    async def extract(self, text: str) -> GraphTriple: ...


def normalize_entity(name: str) -> str:
    """Canonical form for matching/storage: collapse whitespace, strip surrounding punctuation."""
    return re.sub(r"\s+", " ", name).strip(" .,;:'\"")


def extract_entities(text: str) -> list[str]:
    """Proper-noun phrases in ``text``, de-duplicated in first-seen order."""
    seen: dict[str, str] = {}
    for m in _ENTITY_RE.finditer(text):
        phrase = normalize_entity(m.group(0))
        words = phrase.split()
        if len(phrase) < _MIN_ENTITY_LEN or len(words) > _MAX_ENTITY_WORDS:
            continue
        if len(words) == 1 and phrase.lower() in _STOP:
            continue
        # Drop leading stopwords ("The Acme Policy" -> "Acme Policy").
        while words and words[0].lower() in _STOP:
            words = words[1:]
        if not words:
            continue
        phrase = " ".join(words)
        key = phrase.lower()
        if key not in seen:
            seen[key] = phrase
    return list(seen.values())


def cooccurrence_edges(entities: list[str]) -> list[tuple[str, str]]:
    """Every unordered pair of distinct entities (they share a chunk → related)."""
    edges: list[tuple[str, str]] = []
    for i, a in enumerate(entities):
        for b in entities[i + 1 :]:
            edges.append((a, b) if a.lower() <= b.lower() else (b, a))
    return edges


class HeuristicGraphExtractor:
    """Deterministic proper-noun + co-occurrence extractor (default; no model, CI-safe)."""

    async def extract(self, text: str) -> GraphTriple:
        entities = extract_entities(text)[:_MAX_ENTITIES_PER_CHUNK]
        return GraphTriple(entities=entities, edges=cooccurrence_edges(entities))


def build_graph_extract_prompt(text: str) -> str:
    """Prompt for the LLM extraction backend — a typed entity/relationship graph as JSON."""
    return (
        "Extract the named entities and their relationships from the TEXT as JSON: "
        '{"entities": ["Name", ...], "edges": [["A", "B"], ...]}. Entities are specific '
        "named things (people, organisations, policies, systems, places, defined terms) — "
        "not generic words. Edges connect entities that are directly related in the text. "
        "Return only the JSON.\n\n"
        f"TEXT:\n{text.strip()[:2000]}\n\nJSON:"
    )


# ---------------------------------------------------------------------------
# Diagram-structure extraction (additive) — turn a vision model's reading of an image
# document (architecture diagram, flowchart) into real ARROW edges, not co-occurrence.
# ---------------------------------------------------------------------------

_MAX_DIAGRAM_ENTITIES = 24
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
# A Mermaid node: an id, optionally followed by a `[Bracket Label]`.
_MERMAID_NODE_RE = r"([A-Za-z_][\w-]*)\s*(?:\[([^\]\n]{1,80})\])?"
# `A[Label] -- "edge text" --> B[Label]` — the shape Chandra (this deployment's OCR/vision
# model) emits for a flowchart, quoted-label variant.
_MERMAID_LABELED_EDGE_RE = re.compile(rf'{_MERMAID_NODE_RE}\s*--\s*"([^"\n]{{0,80}})"\s*-->\s*{_MERMAID_NODE_RE}')
# `A[Label] --> B[Label]` — the unlabeled-arrow variant.
_MERMAID_PLAIN_EDGE_RE = re.compile(rf"{_MERMAID_NODE_RE}\s*-->\s*{_MERMAID_NODE_RE}")
# Mermaid declares a node's `[Bracket Label]` at its FIRST mention only; later edges reference
# it by bare id (`APIServer -- "SQL query" --> Database[Database]`). A whole-text pre-pass
# collects id -> label so every mention of the same id resolves to the SAME entity.
_MERMAID_DECL_RE = re.compile(r"\b([A-Za-z_][\w-]*)\s*\[([^\]\n]{1,80})\]")

# Single-word caption/scaffolding fragments Chandra's surrounding prose puts right next to the
# Mermaid block (image alt-text, section headings) — never a real diagram box, so they must not
# become graph nodes just because they look like a capitalised phrase.
_DIAGRAM_JUNK = frozenset(
    {
        "architecture",
        "diagram",
        "diagrams",
        "image",
        "images",
        "figure",
        "figures",
        "chart",
        "flowchart",
        "overview",
        "picture",
        "screenshot",
        "graph",
        "drawing",
        "schema",
        "schematic",
        "photo",
        "illustration",
        "sketch",
        "legend",
        "title",
    }
)
# A short ALL-CAPS(+digits) label — the kind of abbreviation Mermaid ids often are when Chandra
# doesn't declare them with a bracketed label ("AS1", "LB", "DB2").
_ACRONYM_RE = re.compile(r"^[A-Z]{1,6}\d{0,3}$")


def _is_diagram_junk(label: str) -> bool:
    """A caption-fragment word, or a mangled multi-sentence run (garbled OCR bleed like
    'Primary Database. The Primary Database') rather than a real node label."""
    words = label.split()
    if len(words) == 1:
        return label.lower() in _DIAGRAM_JUNK or len(label) < 2  # noqa: PLR2004
    return "." in label and len(words) > 6  # noqa: PLR2004 — run-on sentence fragment, not a box caption


def _acronym_of(label: str) -> str:
    """The initials of a multi-word label + its trailing number, e.g. 'API Server 1' -> 'AS1' —
    the convention Mermaid ids that abbreviate a full label tend to follow."""
    words = label.split()
    digits = ""
    if words and words[-1].isdigit():
        digits = words[-1]
        words = words[:-1]
    initials = "".join(w[0] for w in words if w and w[0].isalpha()).upper()
    return initials + digits


def _dedupe_diagram_entities(entities: dict[str, str]) -> dict[str, str]:
    """Best-effort alias merge: a short acronym-shaped label (e.g. 'AS1') whose initials+digit
    match a longer label declared elsewhere in the SAME diagram ('API Server 1') is folded into
    that longer label, which becomes the canonical display name.

    Heuristic, not exhaustive — it only catches the initials+trailing-number convention. An
    alias that doesn't follow that convention (e.g. a synonym like 'DB' for 'Primary Database'
    with no shared digit, or a typo'd variant) still slips through as a separate node."""
    long_labels = [v for v in entities.values() if len(v.split()) > 1]
    remap: dict[str, str] = {}
    for key, label in entities.items():
        if len(label.split()) > 1 or not _ACRONYM_RE.match(label):
            continue
        for long in long_labels:
            if _acronym_of(long) == label.upper():
                remap[key] = long
                break
    return remap


def _finalize_diagram_triple(entities: dict[str, str], edges: list[tuple[str, str, str]]) -> GraphTriple:
    """Shared last step for both diagram parsers: drop junk caption-fragment nodes, merge
    obvious short/long-form aliases, then rebuild entities + labeled directed edges from
    whatever survives. An edge whose endpoint was pure junk is dropped with it. Isolated
    (edge-less) declared nodes are kept — only junk/merged-away ones are removed — so the
    JSON-backend path (which declares nodes independent of edges) doesn't silently lose them."""
    kept = {k: v for k, v in entities.items() if not _is_diagram_junk(v)}
    remap = _dedupe_diagram_entities(kept)

    def _canon(label: str) -> str | None:
        key = label.lower()
        if key in remap:
            return remap[key]
        return kept.get(key)

    # Every surviving, non-merged-away node — including ones with no edge at all.
    final_entities: dict[str, str] = {k: v for k, v in kept.items() if k not in remap}
    for target in remap.values():
        final_entities.setdefault(target.lower(), target)

    final_edges: list[tuple[str, str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for a, b, rel in edges:
        ca, cb = _canon(a), _canon(b)
        if not ca or not cb or ca.lower() == cb.lower():
            continue
        ek = (ca.lower(), cb.lower())
        if ek in seen_edges:
            continue
        seen_edges.add(ek)
        final_edges.append((ca, cb, rel))
    return GraphTriple(
        entities=list(final_entities.values())[:_MAX_DIAGRAM_ENTITIES],
        edges=[(a, b) for a, b, _ in final_edges],
        diagram_edges=final_edges,
    )


def build_diagram_structure_prompt() -> str:
    """Prompt for a PROMPTABLE vision LLM to read a diagram's structure as JSON.

    Written for a real vision Q&A backend (self-hosted VLM / ``openai_compat``). The backend
    actually configured in this deployment (``chandra_api``, Datalab's hosted OCR) ignores the
    prompt and transcribes instead — but for a flowchart image, that transcription itself comes
    back as a Mermaid graph (Chandra is a real document VLM, not OCR software), which
    :func:`parse_diagram_structure` also parses. Forward-compatible with either backend."""
    return (
        "Read this image as a DIAGRAM (architecture diagram, flowchart, or similar). Identify "
        "every labeled box/node and every arrow connecting them. Return ONLY JSON of this "
        'shape: {"nodes": ["Name", ...], "edges": [{"from": "Name", "to": "Name", "label": '
        '"arrow text or empty"}, ...]}. Use each box\'s own visible label as its name. Include '
        "one edge per arrow you see, in the arrow's direction. If the image is not a diagram, "
        'return {"nodes": [], "edges": []}. Return only the JSON.'
    )


def parse_mermaid_diagram(text: str) -> GraphTriple:
    """Parse a Mermaid flowchart (```` ```graph LR; A -- "label" --> B; ``` ````-style, or plain
    ``A --> B``) into a :class:`GraphTriple` of real DIRECTED-ARROW edges — not co-occurrence.
    Entities are the boxes' own labels; an edge exists only where the diagram drew an arrow, so
    two boxes that merely share the image are NOT linked (unlike the co-occurrence heuristic
    ``_index_graph`` already runs on the same chunk's OCR/caption text)."""
    declared: dict[str, str] = {}
    for m in _MERMAID_DECL_RE.finditer(text):
        declared.setdefault(m.group(1), normalize_entity(m.group(2)))

    def _resolve(node_id: str, inline_label: str | None) -> str:
        if inline_label:
            return normalize_entity(inline_label)
        if node_id in declared:
            return declared[node_id]
        return normalize_entity(node_id)

    entities: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(a_id: str, a_label: str | None, b_id: str, b_label: str | None, rel: str = "") -> None:
        a, b = _resolve(a_id, a_label), _resolve(b_id, b_label)
        if not a or not b or a.lower() == b.lower():
            return
        entities.setdefault(a.lower(), a)
        entities.setdefault(b.lower(), b)
        ek = (a.lower(), b.lower())
        if ek not in seen:
            seen.add(ek)
            edges.append((a, b, normalize_entity(rel) if rel else ""))

    # Labeled edges first, then blank out their spans so the plain-arrow pass below doesn't
    # re-match the same arrowhead a second time (both regexes end on `--> NODE`).
    consumed = list(text)
    for m in _MERMAID_LABELED_EDGE_RE.finditer(text):
        _add(m.group(1), m.group(2), m.group(4), m.group(5), m.group(3))
        for i in range(m.start(), m.end()):
            consumed[i] = " "
    for m in _MERMAID_PLAIN_EDGE_RE.finditer("".join(consumed)):
        _add(m.group(1), m.group(2), m.group(3), m.group(4))

    return _finalize_diagram_triple(entities, edges)


def _parse_diagram_json(text: str) -> GraphTriple | None:
    """The JSON shape :func:`build_diagram_structure_prompt` asks for. None if not present."""
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or ("nodes" not in data and "entities" not in data):
        return None
    entities: dict[str, str] = {}
    for n in data.get("nodes") or data.get("entities") or []:
        raw_name = n.get("id") or n.get("name") if isinstance(n, dict) else n
        name = normalize_entity(str(raw_name)) if raw_name else ""
        if name:
            entities.setdefault(name.lower(), name)
    edges: list[tuple[str, str, str]] = []
    for e in data.get("edges", []):
        rel = ""
        if isinstance(e, dict):
            a, b = str(e.get("from", "")), str(e.get("to", ""))
            rel = str(e.get("label", "") or "")
        elif isinstance(e, list) and len(e) >= 2:  # noqa: PLR2004 - [source, target] or [source, target, label]
            a, b = str(e[0]), str(e[1])
            rel = str(e[2]) if len(e) > 2 else ""  # noqa: PLR2004
        else:
            continue
        a, b = normalize_entity(a), normalize_entity(b)
        if a and b and a.lower() != b.lower():
            entities.setdefault(a.lower(), a)
            entities.setdefault(b.lower(), b)
            edges.append((a, b, normalize_entity(rel) if rel else ""))
    if not entities:
        return None
    return _finalize_diagram_triple(entities, edges)


def parse_diagram_structure(text: str) -> GraphTriple:
    """Turn a vision model's reading of a diagram into a :class:`GraphTriple` of real nodes +
    arrow edges. Tries, in order: (1) the JSON shape a PROMPTABLE vision backend returns for
    :func:`build_diagram_structure_prompt`; (2) a Mermaid flowchart, which is what THIS
    deployment's backend (Chandra via Datalab, OCR-only — prompt ignored) actually emits for a
    diagram; (3) empty, if the response is neither — the caller then adds nothing extra and the
    document's OCR/caption text still indexes and searches exactly as it did before this."""
    return _parse_diagram_json(text) or parse_mermaid_diagram(text)


# ---------------------------------------------------------------------------
# Image-document noise filter for the ORDINARY heuristic extractor (additive, image-only).
#
# An image document's OCR/caption text still feeds HeuristicGraphExtractor unchanged (hybrid
# BM25 needs that text regardless — see _index_diagram_graph's docstring), but that text is
# Mermaid syntax plus captions/headings, not prose: it reads nothing like the sentences
# extract_entities() was tuned for. Single scaffolding fragments ("Architecture", "SQL",
# "Queue") surface as standalone entities, and bare category words ("API Servers",
# "Databases" — the class of thing, not a specific named box) pair up into a near-complete
# co-occurrence clique alongside the diagram parser's own precise, directed edges above.
# This filter is applied ONLY when the source document is an image (see pipeline.py's
# ``is_image`` plumbing into ``_index_graph``); a real text/PDF document's extraction is
# completely untouched — this module exposes the filter, the caller decides when to use it.
# ---------------------------------------------------------------------------

# Single-word entities to drop outright when the source is an image: reuses the diagram
# parser's own junk vocabulary above (same scaffolding words apply here) plus a few more
# generic infra nouns that show up as isolated OCR fragments but never as a diagram's own
# junk-node text (so they don't belong in _DIAGRAM_JUNK itself).
_IMAGE_TEXT_JUNK = _DIAGRAM_JUNK | frozenset(
    {
        "sql",
        "queue",
        "queues",
        "cache",
        "caches",
        "client",
        "clients",
        "network",
        "networks",
        "node",
        "nodes",
        "instance",
        "instances",
        "cluster",
        "clusters",
    }
)

# Generic infra-category words: an entity is dropped only when EVERY one of its words is in
# this set — a bare category ("API Servers", "Databases") rather than a specific named
# component ("Load Balancer", "Primary Database", "API Server 1"), which keeps at least one
# word outside this set and so survives.
_GENERIC_INFRA_WORDS = frozenset(
    {
        "server",
        "servers",
        "database",
        "databases",
        "queue",
        "queues",
        "cache",
        "caches",
        "client",
        "clients",
        "api",
        "apis",
        "node",
        "nodes",
        "service",
        "services",
        "instance",
        "instances",
        "cluster",
        "clusters",
    }
)


def filter_image_text_entities(entities: list[str]) -> list[str]:
    """Drop OCR/caption-artifact entities from an IMAGE document's ordinary heuristic
    extraction, before co-occurrence edges are built from what survives. Order-preserving.
    Two checks, both deliberately narrow (see the module comment above for why): a
    single-word scaffolding fragment (``_IMAGE_TEXT_JUNK``), or a phrase built entirely out
    of generic infra-category words (``_GENERIC_INFRA_WORDS``) with no specific modifier."""
    kept: list[str] = []
    for entity in entities:
        words = entity.split()
        if len(words) == 1 and entity.lower() in _IMAGE_TEXT_JUNK:
            continue
        if words and all(w.lower() in _GENERIC_INFRA_WORDS for w in words):
            continue
        kept.append(entity)
    return kept


# ---------------------------------------------------------------------------
# Slice 2 — communities + global/thematic search (GraphRAG map-reduce, FR-C6)
# ---------------------------------------------------------------------------

_GLOBAL_CUES = frozenset(
    {
        "theme",
        "themes",
        "thematic",
        "topic",
        "topics",
        "overview",
        "overall",
        "summary",
        "summarise",
        "summarize",
        "summar",
        "main",
        "key",
        "broad",
        "across",
        "overarching",
        "high-level",
        "high",
        "categories",
        "areas",
        "cover",
        "covers",
        "covered",
        "gist",
        "everything",
        "entire",
        "whole",
    }
)
_GLOBAL_MAX_WORDS = 12  # a global question is short; longer ones are usually specific
_MIN_TERM_LEN = 3  # ignore 1-2 char tokens when scoring community relevance
_GLOBAL_PHRASES = (
    "main themes",
    "key themes",
    "what topics",
    "what themes",
    "main topics",
    "overall",
    "in general",
    "high level",
    "high-level",
    "summarise the",
    "summarize the",
    "give me an overview",
    "what is this",
    "what are the main",
    "what does this cover",
    "across the",
    "key areas",
    "main points",
    # Enumeration / "list-everything" questions want KB-wide breadth even when they are long,
    # so they match on phrase (no word-count cap) rather than the short-question cue path.
    "list all",
    "list every",
    "list each",
    "list the ",
    "list of all",
    "list out",
    "complete list",
    "full list",
    "comprehensive list",
    "exhaustive list",
    "all the ",
    "every ",
    "each of the",
    "all of the",
    "name all",
    "name every",
)


def is_global_query(text: str) -> bool:
    """True when a question is broad/aggregate ("what are the main themes?") rather than
    specific — such questions are answered from community summaries (KB-wide breadth) instead
    of a narrow top-k that lands in one cluster. Heuristic; deliberately conservative."""
    low = text.lower()
    if any(p in low for p in _GLOBAL_PHRASES):
        return True
    words = {w.strip(" .,;:?'\"") for w in low.split()}
    return len(_GLOBAL_CUES & words) >= 1 and len(words) <= _GLOBAL_MAX_WORDS


def detect_communities(node_keys: list[str], edges: list[tuple[str, str]], *, max_iters: int = 12) -> dict[str, int]:
    """Partition the graph into communities by **deterministic label propagation**.

    Co-occurrence graphs collapse into one giant connected component, so connected-components
    alone is useless — label propagation finds the thematic sub-clusters within it. Each node
    adopts the most frequent label among its neighbours; ties (and the seed order) break by
    the lowest label so the result is fully deterministic (test-stable, no RNG). Returns
    ``node_key -> community_id`` with ids densely re-numbered from 0 by descending size."""
    adj: dict[str, set[str]] = {k: set() for k in node_keys}
    for a, b in edges:
        if a in adj and b in adj and a != b:
            adj[a].add(b)
            adj[b].add(a)
    order = sorted(node_keys)
    label = {k: k for k in order}  # seed: every node its own label (its key)
    for _ in range(max_iters):
        changed = False
        for k in order:
            if not adj[k]:
                continue
            counts: dict[str, int] = {}
            for nb in adj[k]:
                counts[label[nb]] = counts.get(label[nb], 0) + 1
            # most frequent neighbour label; tie -> lexicographically smallest (deterministic)
            best = min(counts, key=lambda lab: (-counts[lab], lab))
            if best != label[k]:
                label[k] = best
                changed = True
        if not changed:
            break
    # Re-number labels to dense ids 0..n, largest community first.
    groups: dict[str, list[str]] = {}
    for k, lab in label.items():
        groups.setdefault(lab, []).append(k)
    ranked = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    out: dict[str, int] = {}
    for cid, members in enumerate(ranked):
        for k in members:
            out[k] = cid
    return out


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One entity in the KB graph + the chunks that mention it (mentions = len(chunk_ids))."""

    key: str
    display: str
    chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GraphCommunity:
    """A cluster of related entities = one theme in the KB, with its evidence chunks."""

    community_id: int
    label: str  # short human label (its top entities), e.g. "Acme · Central Government"
    size: int  # number of member entities
    top_entities: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)  # representative evidence
    summary: str = ""
    members: list[str] = field(default_factory=list)  # ALL member node keys — per-node theme lookup


def community_label(top_entities: list[str]) -> str:
    """A compact human label for a community: its two or three strongest entities."""
    return " · ".join(top_entities[:3]) if top_entities else "(unlabelled)"


def summarize_community_heuristic(top_entities: list[str], size: int) -> str:
    """Zero-cost theme summary (default backend): the community's principal entities.
    The LLM backend (rag_clients) writes real prose from representative chunks instead."""
    head = ", ".join(top_entities[:6])
    return f"Theme around {head} ({size} related entities)." if head else f"A cluster of {size} entities."


def build_community_summary_prompt(top_entities: list[str], sample_texts: list[str]) -> str:
    """Prompt for the LLM community-summary backend — one sentence naming the theme."""
    ents = ", ".join(top_entities[:12])
    joined = "\n---\n".join(t.strip()[:600] for t in sample_texts[:4])
    return (
        "You are naming a THEME in a document collection. These entities recur together: "
        f"{ents}.\nRepresentative passages:\n{joined}\n\n"
        "In ONE sentence (max 30 words), state the theme these share. Return only the sentence."
    )


@runtime_checkable
class CommunitySummarizer(Protocol):
    async def summarize(self, top_entities: list[str], sample_texts: list[str], size: int) -> str: ...


class HeuristicCommunitySummarizer:
    """Zero-cost default: names the theme by its principal entities (no model)."""

    async def summarize(self, top_entities: list[str], sample_texts: list[str], size: int) -> str:
        return summarize_community_heuristic(top_entities, size)


def rank_communities(query: str, communities: list[GraphCommunity]) -> list[GraphCommunity]:
    """Order communities by relevance to a global query: lexical overlap of the query's
    terms against each community's entities + summary. A generic query ("summarise the KB")
    has no distinctive terms, so ties fall back to community size (biggest themes first)."""
    terms = {w.strip(" .,;:?'\"") for w in query.lower().split() if len(w) >= _MIN_TERM_LEN} - _STOP - _GLOBAL_CUES

    def score(c: GraphCommunity) -> tuple[int, int]:
        hay = (" ".join(c.top_entities) + " " + c.summary).lower()
        overlap = sum(1 for t in terms if t in hay)
        return (overlap, c.size)

    return sorted(communities, key=score, reverse=True)


# ---------------------------------------------------------------------------
# --- RAG v2 Phase 6: seeding the graph from DECLARED relations ----------------------------------
#
# Phase 3 writes `doc_relations` (document -> document: amends / supersedes / attaches /
# references) and Phase 4 writes `sections.amends` (section -> section). Both are the document
# stating a relationship about itself, in its own words, with a verbatim quote to prove it — which
# is strictly better evidence than anything an extractor infers from prose. Phase 6 turns those
# rows into DIRECTED, LABELED graph edges so "what amends this clause?" is a graph walk.
#
# This consumer deliberately landed BEFORE its producers. That is the plan's own rollout order
# (§13: phases 5 and 6, then 3 and 4) and what its Phase 6 line means by "before or alongside its
# own extraction". Phase 3 has since landed, so `doc_relations` HAS a producer and this path now
# carries real edges; `sections.amends` is still Phase 4's and is still empty. The two sides share
# one quote normalisation by construction — `quote_key` below — so a relation cannot pass Phase 3's
# write-time validation and then silently fail to anchor here.
#
# Edges go out through the SAME `GraphStore.add_diagram_edges` path the diagram parser and the
# Slice 6-B domain-relation extractor already use: that method is the directed-labeled-edge writer
# (see `GraphTriple.diagram_edges`, "it is literally typed directed edges"), not a diagram-specific
# one, so a third producer needs no new store surface.

# The four relation types the plan's Phase 3 prompt can emit. A row carrying anything else is
# SKIPPED rather than written with an unrecognised label: `linked_chunks` walks edges regardless of
# label, so a typo would silently become a real graph edge that nothing can interpret.
RELATION_TYPES: frozenset[str] = frozenset({"amends", "supersedes", "attaches", "references"})


@dataclass(frozen=True, slots=True)
class DeclaredRelation:
    """One relationship a document declares, as stored by Phase 3/4.

    ``source_ref``/``target_ref`` are NAMES, not ids: the document's own title and the reference as
    the document wrote it ("Corrigendum-II", "Clause 14.2 of the bidding document"). Names are what
    the graph is keyed on and what a question uses, so an id here would make the edge unreachable
    from any query. ``evidence_quote`` is verbatim text from the source document — the plan requires
    one for every relation, and it is what anchors the edge to a citable chunk."""

    rel_type: str
    source_ref: str
    target_ref: str
    evidence_quote: str


@dataclass(frozen=True, slots=True)
class RelationSeed:
    """What :func:`relation_edges` resolved: edges ready for the store, their node names, and the
    relations it had to drop (reported, never silently discarded)."""

    edges: list[tuple[str, str, str, str]] = field(default_factory=list)  # (chunk_id, src, tgt, rel)
    entities: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)  # quotes that matched no chunk
    unknown_types: list[str] = field(default_factory=list)


def quote_key(text: str) -> str:
    """Whitespace-normalised, case-folded text for quote matching.

    Whitespace because a quote's line breaks never survive extraction identically — the plan's own
    Phase-4 validation rule is "a substring after whitespace normalisation". Case-folded on top of
    that, because a model asked for a verbatim quote routinely returns one with the first letter
    capitalised or an acronym normalised, and refusing those would drop real evidence for a
    cosmetic difference.

    PUBLIC, and deliberately the ONLY definition of quote matching in the codebase. Phase 3
    validates a declared relation's quote against the document text before storing it
    (:func:`rag_core.doc_profile.parse_doc_profile`) and this module resolves the same quote
    to a chunk when seeding the graph. Two normalisations would mean a relation that passes
    validation at write time and silently fails to anchor at seed time — an amendment edge that
    exists in `doc_relations` and nowhere in the graph, with nothing reporting the gap."""
    return " ".join(text.split()).casefold()


def relation_edges(relations: Sequence[DeclaredRelation], chunks: Sequence[Chunk]) -> RelationSeed:
    """Resolve declared relations into graph edges anchored to the chunk that proves each one.

    A relation is attached to the chunk whose text CONTAINS its evidence quote, so the edge carries
    real, citable evidence — the same discipline Phase 4 applies to a key-value or a proposition,
    and the reason a relation whose quote resolves to nothing is dropped instead of being hung off
    an arbitrary chunk. Where several chunks contain the quote the FIRST in document order wins,
    which makes the result deterministic.

    Pure: no store, no I/O, so the resolution rule is testable on its own."""
    seed = RelationSeed()
    keyed = [(quote_key(c.text), c.chunk_id) for c in chunks]
    for relation in relations:
        rel_type = relation.rel_type.strip().lower()
        if rel_type not in RELATION_TYPES:
            seed.unknown_types.append(relation.rel_type)
            continue
        if not (relation.source_ref.strip() and relation.target_ref.strip()):
            continue  # a relation missing an end cannot be an edge
        quote = quote_key(relation.evidence_quote)
        anchor = next((cid for text, cid in keyed if quote and quote in text), None)
        if anchor is None:
            seed.unresolved.append(relation.evidence_quote[:120])
            continue
        seed.edges.append((anchor, relation.source_ref.strip(), relation.target_ref.strip(), rel_type))
        seed.entities.extend((relation.source_ref.strip(), relation.target_ref.strip()))
    return seed


# Slice 6, sub-slice B — typed domain-graph edges (additive, no new dependency)
#
# The ordinary text path (``pipeline._index_graph``) writes only UNTYPED co-occurrence edges.
# For a KB whose domain profile declares a relation vocabulary, ALSO extract real DIRECTED,
# LABELED edges — "Aspirin inhibits COX-2", "the 2013 Act amends the 1956 Act" — as
# ``GraphTriple.diagram_edges`` (the same field the diagram parsers populate — it is literally
# "typed directed edges"), written via the existing ``GraphStore.add_diagram_edges`` path.
#
# HONEST SCOPE (see the design's §3.3): ``linked_chunks`` walks edges regardless of relation
# label, so typed edges improve graph recall/quality and are auditable/visible, but they do NOT
# yet give relation-FILTERED precision. That query-path filter is DEFERRED as B2.
# ---------------------------------------------------------------------------

# Per-domain relation vocabulary. Domain-neutral core stays empty (general KBs are unaffected —
# no vocab -> the extractor is never even constructed). Extensible per domain.
DOMAIN_RELATIONS: dict[str, tuple[str, ...]] = {
    "chemistry": (
        "inhibits",
        "activates",
        "binds",
        "binds to",
        "metabolizes",
        "reacts with",
        "catalyzes",
        "is a metabolite of",
        "is an agonist of",
        "is an antagonist of",
    ),
    "legal": (
        "amends",
        "supersedes",
        "repeals",
        "cites",
        "is governed by",
        "incorporates",
    ),
}

# The subject / object of a relation: a proper-noun phrase (capitalised head + capitalised
# continuation words), matched WITHOUT case-folding so an arbitrary lowercase word is not mistaken
# for an entity (the relation phrase itself is matched case-insensitively via an inline
# ``(?i:...)`` group instead). A sentence-ending period is deliberately EXCLUDED from the token so
# an entity never spans a sentence boundary — "…COX-2. Warfarin binds…" captures "COX-2" and
# "Warfarin" as two entities, not one run.
_REL_ENTITY = r"([A-Z][A-Za-z0-9&\-]*(?:\s+[A-Z][A-Za-z0-9&\-]*){0,4})"


def relation_vocab(domain: str) -> tuple[str, ...]:
    """The relation vocabulary a domain declares, or an empty tuple (general / unknown domains)."""
    return DOMAIN_RELATIONS.get(domain, ())


def _clean_relation_entity(phrase: str) -> str | None:
    """Normalise a captured subject/object and reject non-entities (too short, all-stopword,
    too many words) — the same discipline :func:`extract_entities` applies."""
    words = normalize_entity(phrase).split()
    while words and words[0].lower() in _STOP:
        words = words[1:]
    while words and words[-1].lower() in _STOP:
        words = words[:-1]
    if not words or len(words) > _MAX_ENTITY_WORDS:
        return None
    cleaned = " ".join(words)
    if len(cleaned) < _MIN_ENTITY_LEN:
        return None
    if len(words) == 1 and cleaned.lower() in _STOP:
        return None
    return cleaned


def _relation_pattern(relation: str) -> re.Pattern[str]:
    """``ENTITY <relation-phrase> [article] ENTITY`` — entities case-sensitive, the phrase and the
    optional intervening article (the/a/an) case-insensitive. The article is skipped so "the 2013
    Act amends the 1956 Act" links the two Acts rather than failing on the lowercase determiner."""
    return re.compile(_REL_ENTITY + r"\s+(?i:" + re.escape(relation) + r")\s+(?i:the\s+|an?\s+)?" + _REL_ENTITY)


class HeuristicDomainRelationExtractor:
    """Deterministic verb-pattern domain-relation extractor (default; no model, CI-safe).

    For each relation phrase in the KB's domain vocabulary, finds ``ENTITY <phrase> ENTITY`` and
    emits a directed, labeled ``(source, target, relation)`` triple. Implements the same
    ``extract(text) -> GraphTriple`` shape as :class:`GraphExtractor`, populating
    ``diagram_edges`` (real typed edges) — ``edges`` mirrors them undirected for the co-occurrence
    node bookkeeping ``add_diagram_edges`` shares with ``add_chunk``."""

    def __init__(self, vocab: tuple[str, ...]) -> None:
        # Longest phrases first so "binds to" is tried before "binds" and wins the span.
        self._patterns = [(rel, _relation_pattern(rel)) for rel in sorted(vocab, key=len, reverse=True)]

    async def extract(self, text: str) -> GraphTriple:
        entities: dict[str, str] = {}
        edges: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for rel, pat in self._patterns:
            for m in pat.finditer(text):
                a = _clean_relation_entity(m.group(1))
                b = _clean_relation_entity(m.group(2))
                if not a or not b or a.lower() == b.lower():
                    continue
                key = (a.lower(), b.lower(), rel.lower())
                if key in seen:
                    continue
                seen.add(key)
                entities.setdefault(a.lower(), a)
                entities.setdefault(b.lower(), b)
                edges.append((a, b, rel))
        return GraphTriple(
            entities=list(entities.values()),
            edges=[(a, b) for a, b, _ in edges],
            diagram_edges=edges,
        )


def build_domain_relation_prompt(text: str, vocab: tuple[str, ...]) -> str:
    """Prompt for the LLM domain-relation backend — typed edges CONSTRAINED to the profile's
    relation vocabulary, so the extracted graph stays typed rather than free-text (mirrors
    :func:`build_graph_extract_prompt`)."""
    rels = ", ".join(vocab)
    return (
        "Extract DIRECTED, TYPED relationships between named entities in the TEXT as JSON: "
        '{"edges": [["Subject", "relation", "Object"], ...]}. The "relation" MUST be one of '
        f"exactly these labels: {rels}. Subject and Object are specific named things (compounds, "
        "proteins, organisations, statutes, defined terms) — not generic words. Include an edge "
        "only when the text states that relationship in that direction. If none apply, return "
        '{"edges": []}. Return only the JSON.\n\n'
        f"TEXT:\n{text.strip()[:2000]}\n\nJSON:"
    )
