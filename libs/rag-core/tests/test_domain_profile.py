"""Unit tests for domain-profile detection + typed domain-relation extraction (Slice 6) — pure."""

from __future__ import annotations

from rag_core.domain_profile import (
    QUERY_TIME_KNOB_KEYS,
    REINDEX_KNOB_KEYS,
    DomainSignal,
    classify_corpus,
    suggest_knobs,
)
from rag_core.graph import (
    DOMAIN_RELATIONS,
    HeuristicDomainRelationExtractor,
    build_domain_relation_prompt,
    relation_vocab,
)

_CHEM = (
    "The reaction of H2SO4 with NaOH is exothermic. The recommended dose is 5 mg/kg. "
    "The active compound has SMILES CC(=O)Oc1ccccc1C(=O)O and 10 mmol was added at 25 degrees Celsius."
)
_LEGAL = (
    "Section 12 hereby applies pursuant to the Companies Act, 2013. Whereas the parties agree, "
    "Clause 4 shall govern herein. Notwithstanding Section 3, the Act, 1956 is repealed."
)
_FINANCIAL = (
    "Invoice total is ₹5,000 and the GSTIN is recorded. The fiscal year net profit was 3.2 crore. "
    "The ledger shows accounts payable and depreciation of USD 1,200 in the balance sheet."
)
_CODE = (
    "```python\n"
    "import os\n"
    "from sys import argv\n"
    "def parse(x):\n"
    "    return {a: b};\n"
    "```\n"
    "function render() { const y = 2; return y; }\n"
)
_GENERAL = (
    "The annual company picnic will be held next Saturday at the riverside park. Everyone is "
    "warmly invited to join the celebration with food, music and games for the whole family."
)


def test_classify_chemistry() -> None:
    assert classify_corpus([_CHEM]).domain == "chemistry"


def test_classify_legal() -> None:
    assert classify_corpus([_LEGAL]).domain == "legal"


def test_classify_financial() -> None:
    assert classify_corpus([_FINANCIAL]).domain == "financial"


def test_classify_code() -> None:
    assert classify_corpus([_CODE]).domain == "code"


def test_classify_general_prose() -> None:
    assert classify_corpus([_GENERAL]).domain == "general"


def test_empty_corpus_is_general() -> None:
    sig = classify_corpus([])
    assert sig.domain == "general"
    assert sig.score == 1.0  # nothing detected -> fully confident it is general


def test_mixed_corpus_falls_back_to_general() -> None:
    """Two comparable domain signals in one corpus is a near-tie -> the conservative ``general``
    default wins (a misclassified KB that flipped knobs is worse than 'leave the defaults')."""
    chem = "H2SO4 reacts with NaOH. The dose is 5 mg."  # 2 notation hits
    legal = "Section 1 hereby applies."  # 2 legal cues (section 1, hereby)
    sig = classify_corpus([chem + " " + legal])
    assert sig.domain == "general", sig.mix


def test_signal_is_auditable() -> None:
    sig = classify_corpus([_CHEM])
    assert sig.evidence["chemistry"] > 0
    assert 0.0 <= sig.score <= 1.0
    assert abs(sum(sig.mix.values()) - 1.0) < 1e-6 or sum(sig.mix.values()) == 0.0


def test_suggest_knobs_splits_query_time_from_reindex() -> None:
    """The suggestion separates safe query-time knobs from reindex-required ones — the guarantee
    the apply endpoint relies on to keep destructive knobs opt-in."""
    chem = suggest_knobs(DomainSignal(domain="chemistry", score=0.9, mix={}, evidence={}))
    # Query-time bucket: only ever query-time knobs; reindex bucket: only ever reindex knobs.
    assert set(chem.query_time_knobs) <= QUERY_TIME_KNOB_KEYS
    assert set(chem.reindex_knobs) <= REINDEX_KNOB_KEYS
    assert "notation_grounding_enabled" in chem.query_time_knobs
    assert "exact_boost" in chem.query_time_knobs
    # Reindex-class capabilities are SUGGEST-ONLY (never in the immediately-appliable bucket).
    assert "chem_normalization_enabled" in chem.reindex_knobs
    assert "domain_relations_enabled" in chem.reindex_knobs
    assert chem.rationale


def test_suggest_knobs_general_recommends_no_change() -> None:
    gen = suggest_knobs(DomainSignal(domain="general", score=1.0, mix={}, evidence={}))
    assert gen.query_time_knobs == {}
    assert gen.reindex_knobs == {}


def test_legal_suggestion_keeps_notation_off() -> None:
    legal = suggest_knobs(DomainSignal(domain="legal", score=0.9, mix={}, evidence={}))
    assert legal.query_time_knobs.get("notation_grounding_enabled") == 0.0
    assert legal.query_time_knobs.get("exact_boost", 0) > 0


# --- typed domain-relation extraction (sub-slice B) --------------------------
def test_relation_vocab_is_domain_scoped() -> None:
    assert relation_vocab("chemistry")  # non-empty
    assert relation_vocab("legal")
    assert relation_vocab("general") == ()  # general KBs declare no vocab -> extractor never runs
    assert relation_vocab("financial") == ()


async def test_heuristic_chemistry_relations_are_directed_and_typed() -> None:
    ex = HeuristicDomainRelationExtractor(relation_vocab("chemistry"))
    triple = await ex.extract("Aspirin inhibits COX-2. Warfarin binds to Albumin.")
    assert ("Aspirin", "COX-2", "inhibits") in triple.diagram_edges
    assert ("Warfarin", "Albumin", "binds to") in triple.diagram_edges
    # Entities are populated so the graph-store node bookkeeping has them.
    assert {"Aspirin", "COX-2", "Warfarin", "Albumin"} <= set(triple.entities)


async def test_heuristic_legal_relations() -> None:
    ex = HeuristicDomainRelationExtractor(relation_vocab("legal"))
    triple = await ex.extract("The Companies Act amends the Old Act. This Statute cites Article 21.")
    rels = {(a, b, r) for a, b, r in triple.diagram_edges}
    assert ("Companies Act", "Old Act", "amends") in rels
    assert ("Statute", "Article", "cites") in rels


async def test_heuristic_relations_do_not_span_sentences() -> None:
    """An entity never runs across a sentence boundary — a full stop stops the phrase."""
    ex = HeuristicDomainRelationExtractor(relation_vocab("chemistry"))
    triple = await ex.extract("Aspirin inhibits COX-2. Warfarin is unrelated here.")
    for a, b, _ in triple.diagram_edges:
        assert "." not in a and "." not in b


async def test_no_vocab_extracts_nothing() -> None:
    ex = HeuristicDomainRelationExtractor(())
    triple = await ex.extract("Aspirin inhibits COX-2.")
    assert triple.diagram_edges == []


def test_relation_prompt_constrains_to_vocab() -> None:
    prompt = build_domain_relation_prompt("some text", relation_vocab("chemistry"))
    assert "inhibits" in prompt
    assert "JSON" in prompt


def test_domain_relations_table_is_only_specialised_domains() -> None:
    # General/financial/code declare no relation vocabulary (they are unaffected by sub-slice B).
    assert set(DOMAIN_RELATIONS) == {"chemistry", "legal"}
