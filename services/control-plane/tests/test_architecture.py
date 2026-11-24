"""Architecture extraction core — schema, parse, Mermaid/PlantUML renderers, heuristic summary."""

from __future__ import annotations

import pytest
from svc_control_plane.architecture import (
    ArchComponent,
    ArchEdge,
    Architecture,
    answer_architecture,
    architecture_from_image,
    architecture_grounded_answer,
    attach_diagram,
    build_architecture_prompt,
    build_explain_prompt,
    extract_architecture,
    has_architecture_potential,
    heuristic_summary,
    is_architecture_query,
    parse_architecture,
    sanitize_architecture,
    slug_id,
    to_mermaid,
    to_plantuml,
    truncate_architecture,
)

from rag_schemas import Citation, GroundedAnswer, RetrievalMode, RetrievalQuery, RetrievalResult, RetrievedSpan
from rag_schemas.common import AnswerStatus


def _arch() -> Architecture:
    return Architecture(
        title="Orders Platform",
        components=[
            ArchComponent(id="web", name="Web App", kind="frontend"),
            ArchComponent(id="api", name="Orders API", kind="api"),
            ArchComponent(id="db", name="Orders DB", kind="database"),
            ArchComponent(id="pay", name="Payments", kind="external"),
        ],
        edges=[
            ArchEdge(source="web", target="api", relation="calls"),
            ArchEdge(source="api", target="db", relation="reads"),
            ArchEdge(source="api", target="pay", relation="integrates"),
        ],
    )


def test_slug_id() -> None:
    assert slug_id("API Gateway") == "api_gateway"
    assert slug_id("  Orders-DB!! ") == "orders_db"
    assert slug_id("***") == "node"  # never empty


def test_parse_assigns_ids_maps_unknown_kind_and_drops_dangling_edges() -> None:
    data = {
        "title": "Shop",
        "components": [
            {"name": "Storefront", "kind": "frontend"},
            {"name": "Catalog Service", "kind": "microservice"},  # unknown kind -> service
            {"name": "Storefront", "kind": "frontend"},  # duplicate -> deduped
        ],
        "edges": [
            {"source": "Storefront", "target": "Catalog Service", "relation": "calls"},  # by NAME
            {"source": "Storefront", "target": "Ghost", "relation": "calls"},  # dangling -> dropped
        ],
    }
    arch = parse_architecture(data)
    ids = {c.id: c.kind for c in arch.components}
    assert ids == {"storefront": "frontend", "catalog_service": "service"}  # deduped + unknown->service
    assert len(arch.edges) == 1  # the dangling edge to "Ghost" was dropped
    assert arch.edges[0].source == "storefront" and arch.edges[0].target == "catalog_service"


def test_parse_accepts_list_form_edges() -> None:
    data = {"components": [{"name": "A"}, {"name": "B"}], "edges": [["A", "B"]]}
    arch = parse_architecture(data)
    assert len(arch.edges) == 1
    assert arch.edges[0].relation == "depends_on"


def test_to_mermaid_shapes_and_labels() -> None:
    out = to_mermaid(_arch())
    assert out.startswith("flowchart LR")
    assert 'db[("Orders DB")]' in out  # database -> cylinder
    assert 'pay{{"Payments"}}' in out  # external -> hexagon
    assert "web -->|calls| api" in out  # labelled edge
    assert to_mermaid(_arch()) == out  # deterministic


def test_to_plantuml_declarations_and_edges() -> None:
    out = to_plantuml(_arch())
    assert out.startswith("@startuml") and out.rstrip().endswith("@enduml")
    assert 'database "Orders DB" as db' in out
    assert 'interface "Orders API" as api' in out
    assert "api --> pay : integrates" in out


def test_heuristic_summary_counts_and_flows() -> None:
    s = heuristic_summary(_arch())
    assert "4 components" in s
    assert "1 database" in s
    assert "Web App calls Orders API" in s
    assert heuristic_summary(Architecture()) == "No architecture components were identified."


def test_prompt_builders() -> None:
    p = build_architecture_prompt("Service A calls Service B.")
    assert "STRICT, COMPACT JSON" in p and "kind MUST be one of" in p
    plain = build_explain_prompt(_arch(), audience="non-technical")
    assert "NO technical background" in plain
    tech = build_explain_prompt(_arch(), audience="technical")
    assert "technical reader" in tech


# --- extractor (fake gateways) -------------------------------------------------------


class _Gateway:
    def __init__(self, out: str) -> None:
        self._out = out

    async def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        return self._out


async def test_extract_architecture_parses_fenced_json() -> None:
    gw = _Gateway(
        '```json\n{"title":"T","components":[{"name":"Orders API","kind":"api"},'
        '{"name":"Orders DB","kind":"database"}],"edges":[{"source":"Orders API","target":"Orders DB",'
        '"relation":"reads"}]}\n```'
    )
    arch = await extract_architecture("some technical text", gateway=gw)
    assert arch.title == "T"
    assert {c.id for c in arch.components} == {"orders_api", "orders_db"}
    assert arch.edges[0].relation == "reads"


async def test_extract_architecture_failsafe_on_garbage() -> None:
    arch = await extract_architecture("x", gateway=_Gateway("sorry, no JSON here"))
    assert arch.components == []  # empty, no raise


async def test_extract_architecture_failsafe_on_gateway_error() -> None:
    class _Boom:
        async def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
            raise RuntimeError("model down")

    arch = await extract_architecture("x", gateway=_Boom())
    assert arch.components == []


async def test_architecture_from_image() -> None:
    class _Vision:
        async def complete_with_images(self, prompt: str, images: list[bytes]) -> str:
            return "A web frontend calls an orders API."

    gw = _Gateway(
        '{"components":[{"name":"Web","kind":"frontend"},{"name":"Orders API","kind":"api"}],'
        '"edges":[{"source":"Web","target":"Orders API","relation":"calls"}]}'
    )
    arch = await architecture_from_image(b"fakeimg", vision_gateway=_Vision(), gateway=gw)
    assert {c.id for c in arch.components} == {"web", "orders_api"}
    assert to_mermaid(arch).startswith("flowchart LR")  # renders end-to-end from an image


async def test_answer_architecture_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    span = RetrievedSpan(
        chunk_id="c1",
        doc_id="d1",
        text="The Web App calls the Orders API which reads the Orders DB.",
        score=0.9,
        scorer="cross_encoder",
    )

    class _Svc:
        async def retrieve(self, query: object) -> RetrievalResult:
            return RetrievalResult(query=query, spans=[span], max_score=0.9, mode_used=RetrievalMode.HYBRID)

    gw = _Gateway(
        '{"components":[{"name":"Web App","kind":"frontend"},{"name":"Orders API","kind":"api"},'
        '{"name":"Orders DB","kind":"database"}],"edges":[{"source":"Web App","target":"Orders API",'
        '"relation":"calls"},{"source":"Orders API","target":"Orders DB","relation":"reads"}]}'
    )

    async def _svc_for(_kb: str) -> _Svc:
        return _Svc()

    async def _gw_for(_kb: str, _model: str | None = None) -> _Gateway:
        return gw

    monkeypatch.setattr(wiring, "retrieval_service_for", _svc_for)
    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _gw_for)
    ans = await answer_architecture("bank__deals", "Orders system", audience="non-technical", actor="u")
    assert {c.id for c in ans.architecture.components} == {"web_app", "orders_api", "orders_db"}
    assert "flowchart LR" in ans.mermaid
    assert ans.plantuml.startswith("@startuml")
    assert ans.explanation  # audience-tuned prose (non-empty)
    assert len(ans.citations) == 1  # cited back to the retrieved chunk


# --- auto-intent (route architecture questions through /query) -----------------------

_ARCH_JSON = (
    '{"components":[{"name":"Web App","kind":"frontend"},{"name":"Orders API","kind":"api"}],'
    '"edges":[{"source":"Web App","target":"Orders API","relation":"calls"}]}'
)


def _svc_with(spans: list[RetrievedSpan]) -> object:
    class _Svc:
        async def retrieve(self, query: object) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                spans=spans,
                max_score=max((s.score for s in spans), default=0.0),
                mode_used=RetrievalMode.HYBRID,
            )

    return _Svc()


def test_is_architecture_query() -> None:
    """ONLY the literal word 'architecture' triggers the diagram flow.

    Deliberately narrowed (per the owner): a broader trigger fired on ordinary questions that
    merely said 'services' or 'diagram' and hijacked them into an architecture answer. These
    assertions pin the narrow contract so it cannot quietly widen again.
    """
    assert is_architecture_query("Explain the architecture of the orders system")
    assert is_architecture_query("show me the architecture")
    assert is_architecture_query("ARCHITECTURE overview please")  # case-insensitive
    # Adjacent phrasings that must NOT trigger it — they do not contain the word.
    assert not is_architecture_query("generate a system design diagram")
    assert not is_architecture_query("which microservices make up the platform")
    assert not is_architecture_query("what is the refund policy")
    assert not is_architecture_query("who approved invoice 42")


def test_has_architecture_potential_is_disabled() -> None:
    """The weak-signal 'offer a diagram chip' path is switched off, and must stay off.

    It used to fire on services / components / data-flow / topology, pestering ordinary questions
    with a diagram offer. The strong signal (:func:`is_architecture_query`) already auto-renders
    the case that matters, so no separate weak signal is wanted.
    """
    for q in (
        "how do the payments service and the ledger connect together",
        "walk me through the data flow between components",
        "which microservices make up the platform",
        "what does the deployment topology look like end-to-end",
        "what is the refund policy",
        "who approved invoice 42",
    ):
        assert not has_architecture_potential(q), f"the weak-signal path must stay disabled: {q!r}"


def _grounded(answer: str = "RICH PROSE") -> GroundedAnswer:
    return GroundedAnswer(
        tenant_id="k",
        query="how do the orders components fit together",
        status=AnswerStatus.GROUNDED,
        answer=answer,
        citations=[Citation(chunk_id="c1", doc_id="d1", quote="Web App calls Orders API.", score=0.8)],
        confidence=0.8,
        model="sarvam-105b",
    )


async def test_attach_diagram_layers_onto_grounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    span = RetrievedSpan(
        chunk_id="c1", doc_id="d1", text="Web App calls Orders API.", score=0.8, scorer="cross_encoder"
    )

    async def _svc_for(_kb: str) -> object:
        return _svc_with([span])

    async def _gw_for(_kb: str, _model: str | None = None) -> _Gateway:
        return _Gateway(_ARCH_JSON)

    monkeypatch.setattr(wiring, "retrieval_service_for", _svc_for)
    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _gw_for)
    out = await attach_diagram(_grounded(), "k", "orders system")
    # Prose is PRESERVED (not swapped for the heuristic caption); the diagram RIDES in structured.
    assert out.answer == "RICH PROSE"
    assert out.structured is not None and "mermaid" in out.structured and "plantuml" in out.structured
    assert "flowchart LR" in out.structured["mermaid"]
    assert out.diagram_available is False  # rendered inline, not merely offered


async def test_attach_diagram_failsafe_offers_chip(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    async def _svc_for(_kb: str) -> object:
        return _svc_with([RetrievedSpan(chunk_id="c1", doc_id="d1", text="noise", score=0.5, scorer="x")])

    async def _gw_for(_kb: str, _model: str | None = None) -> _Gateway:
        return _Gateway("sorry, no JSON here")  # extraction yields no components

    monkeypatch.setattr(wiring, "retrieval_service_for", _svc_for)
    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _gw_for)
    out = await attach_diagram(_grounded(), "k", "orders system")
    # Couldn't ground a diagram -> offer the chip, leave the prose + structured untouched.
    assert out.diagram_available is True
    assert out.answer == "RICH PROSE"
    assert (out.structured or {}).get("mermaid") is None


async def test_architecture_grounded_answer_converts(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    span = RetrievedSpan(
        chunk_id="c1", doc_id="d1", text="Web App calls Orders API.", score=0.8, scorer="cross_encoder"
    )

    async def _svc_for(_kb: str) -> object:
        return _svc_with([span])

    async def _gw_for(_kb: str, _model: str | None = None) -> _Gateway:
        return _Gateway(_ARCH_JSON)

    monkeypatch.setattr(wiring, "retrieval_service_for", _svc_for)
    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _gw_for)
    ga = await architecture_grounded_answer(
        RetrievalQuery(tenant_id="k", text="explain the architecture of orders", top_k=12, acl=[]), actor="u"
    )
    assert ga.status is AnswerStatus.GROUNDED
    assert ga.structured is not None and "mermaid" in ga.structured and "plantuml" in ga.structured
    assert ga.citations and ga.model == "architecture"


async def test_architecture_grounded_answer_abstains_when_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from svc_control_plane import wiring

    async def _svc_for(_kb: str) -> object:
        return _svc_with([])  # no chunks -> no citations -> abstain

    async def _gw_for(_kb: str, _model: str | None = None) -> _Gateway:
        return _Gateway(_ARCH_JSON)

    monkeypatch.setattr(wiring, "retrieval_service_for", _svc_for)
    monkeypatch.setattr(wiring, "resolve_tenant_gateway", _gw_for)
    ga = await architecture_grounded_answer(
        RetrievalQuery(tenant_id="k", text="explain the architecture", top_k=12, acl=[]), actor="u"
    )
    assert ga.status is AnswerStatus.ABSTAINED  # never hallucinates a diagram


# --- styling / direction -------------------------------------------------------------


def test_to_mermaid_styles_and_classes() -> None:
    out = to_mermaid(_arch())
    # per-kind classDef + class assignment for colour-coding
    assert "classDef database" in out
    assert "class db database" in out
    assert "classDef api" in out and "class api api" in out


def test_to_mermaid_direction_switches_by_size() -> None:
    small = to_mermaid(_arch())  # 4 components
    assert small.startswith("flowchart LR")
    big = Architecture(
        title="Big",
        components=[ArchComponent(id=f"n{i}", name=f"N{i}", kind="service") for i in range(10)],
        edges=[ArchEdge(source="n0", target=f"n{i}", relation="calls") for i in range(1, 10)],
    )
    assert to_mermaid(big).startswith("flowchart TB")  # >8 components -> top-down


def test_to_mermaid_merges_duplicate_edges_joining_labels() -> None:
    arch = Architecture(
        components=[ArchComponent(id="a", name="A"), ArchComponent(id="b", name="B")],
        edges=[
            ArchEdge(source="a", target="b", relation="reads"),
            ArchEdge(source="a", target="b", relation="writes"),
        ],
    )
    out = to_mermaid(arch)
    assert "a -->|reads / writes| b" in out
    assert out.count("a -->") == 1  # collapsed into a single arrow


# --- validate / repair (adversarial) -------------------------------------------------


def test_parse_suffixes_colliding_slugs_from_distinct_names() -> None:
    data = {"components": [{"name": "Web App"}, {"name": "Web-App"}]}  # distinct names, same slug
    arch = parse_architecture(data)
    assert {c.id for c in arch.components} == {"web_app", "web_app_2"}  # both kept


def test_sanitize_neutralises_punctuation_and_empty_labels() -> None:
    arch = Architecture(
        components=[
            ArchComponent(id="a", name='Bad|Node "quoted" [x]'),
            ArchComponent(id="b", name="   "),  # empty after clean -> falls back to id
        ],
        edges=[ArchEdge(source="a", target="b", relation="calls")],
    )
    clean = sanitize_architecture(arch)
    labels = {c.id: c.name for c in clean.components}
    assert "|" not in labels["a"] and '"' not in labels["a"] and "[" not in labels["a"]
    assert labels["b"] == "b"  # empty label falls back to the id
    out = to_mermaid(arch)  # renders without raising
    assert 'a["Bad/Node' in out and 'b["b"]' in out


def test_reserved_word_ids_are_suffixed_and_consistent() -> None:
    # A component named "End" slugs to the Mermaid keyword `end`, which would break the flowchart
    # grammar; it (and other reserved words) must be suffixed and referenced consistently in edges.
    data = {
        "components": [{"name": "End"}, {"name": "End User"}, {"name": "Start"}],
        "edges": [{"source": "Start", "target": "End", "relation": "flows"}],
    }
    arch = parse_architecture(data)
    ids = {c.id for c in arch.components}
    assert "end" not in ids  # bare reserved id would corrupt the diagram
    assert "end_1" in ids  # routed through the numeric-suffix logic
    # the edge to "End" resolves to the suffixed id, and the rendered diagram references it
    edge = arch.edges[0]
    assert edge.target == "end_1"
    out = to_mermaid(arch)
    assert "end_1" in out and " end[" not in out and " end(" not in out
    # sanitize is also reserved-word safe for a hand-built architecture
    clean = sanitize_architecture(Architecture(components=[ArchComponent(id="end", name="End")], edges=[]))
    assert clean.components[0].id == "end_1"


def test_sanitize_drops_dangling_and_self_loop_edges() -> None:
    arch = Architecture(
        components=[ArchComponent(id="a", name="A"), ArchComponent(id="b", name="B")],
        edges=[
            ArchEdge(source="a", target="ghost", relation="calls"),  # dangling
            ArchEdge(source="a", target="a", relation="loops"),  # self-loop
            ArchEdge(source="a", target="b", relation="calls"),  # valid
        ],
    )
    clean = sanitize_architecture(arch)
    assert len(clean.edges) == 1
    assert clean.edges[0].source == "a" and clean.edges[0].target == "b"


def test_truncate_keeps_most_connected_and_notes() -> None:
    comps = [ArchComponent(id=f"n{i}", name=f"N{i}", kind="service") for i in range(5)]
    # n0 is a hub (degree 4); n1..n4 have degree 1 each
    edges = [ArchEdge(source="n0", target=f"n{i}", relation="calls") for i in range(1, 5)]
    arch = Architecture(components=comps, edges=edges)
    out = truncate_architecture(arch, 3)
    assert len(out.components) == 3
    assert "n0" in {c.id for c in out.components}  # the hub is kept
    assert "3 of 5" in out.note
    # edges only reference kept nodes
    kept = {c.id for c in out.components}
    assert all(e.source in kept and e.target in kept for e in out.edges)


def test_to_mermaid_is_deterministic_after_upgrade() -> None:
    assert to_mermaid(_arch()) == to_mermaid(_arch())
    assert to_plantuml(_arch()) == to_plantuml(_arch())


def test_to_mermaid_clamps_long_labels() -> None:
    long = "X" * 80
    arch = Architecture(components=[ArchComponent(id="a", name=long)], edges=[])
    out = to_mermaid(arch)
    assert "…" in out  # ellipsis
    assert ("X" * 80) not in out  # full label not rendered inside the node
