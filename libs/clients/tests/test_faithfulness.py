"""Unit tests for the deterministic lexical faithfulness checker (FR-E4)."""

from __future__ import annotations

from rag_clients.faithfulness import LexicalFaithfulnessChecker
from rag_core import route_faithfulness
from rag_schemas import RetrievedSpan
from rag_schemas.common import AnswerStatus

_SPANS = [
    RetrievedSpan(
        chunk_id="c1",
        doc_id="d1",
        score=0.9,
        text="The platform uptime target is 99.5 percent baseline and 99.9 percent at peak.",
        scorer="cross_encoder",
    )
]


async def test_fully_supported_scores_high() -> None:
    v = await LexicalFaithfulnessChecker().check("The uptime target is 99.5 percent baseline.", _SPANS)
    assert v.score is not None, "the lexical checker always judges; None would mean it did not run"
    assert v.score >= 0.7
    assert v.faithful
    assert v.unsupported_claims == []


async def test_fabricated_answer_scores_low() -> None:
    v = await LexicalFaithfulnessChecker().check("Mars has two moons named Phobos and Deimos orbiting quickly.", _SPANS)
    assert v.score is not None
    assert v.score < 0.4
    assert not v.faithful
    assert v.unsupported_claims


async def test_partial_support_is_borderline() -> None:
    # One supported sentence + one fabricated -> score lands in the flag band.
    answer = "The uptime target is 99.5 percent baseline. Refunds are processed within twelve weeks."
    v = await LexicalFaithfulnessChecker().check(answer, _SPANS)
    assert v.score is not None
    assert 0.4 <= v.score < 0.7
    assert any("Refunds" in c for c in v.unsupported_claims)


def test_route_three_way() -> None:
    assert route_faithfulness(0.9, 0.7, 0.4) is AnswerStatus.GROUNDED
    assert route_faithfulness(0.55, 0.7, 0.4) is AnswerStatus.FLAGGED_FOR_REVIEW
    assert route_faithfulness(0.2, 0.7, 0.4) is AnswerStatus.ABSTAINED


# --- fabricated figures (P29) -----------------------------------------------------------------


def test_a_wrong_figure_in_a_near_verbatim_quote_scores_perfectly_faithful() -> None:
    """The exact shape the guard exists for, measured rather than assumed.

    The more closely an answer quotes its source, the higher token overlap scores it — and the
    lexical checker drops the figure before scoring at all (`_MIN_TOKEN_LEN = 3`, commented
    "ignore 1-2 char tokens (numbers/particles)"). So an answer that reproduces a sentence
    verbatim and changes only the NUMBER scores 1.00: perfectly faithful, and wrong in the one
    way a compliance reader would act on.

    A heavily paraphrased fabrication (the QA run's "18 working days") is caught by overlap
    anyway, at 0.25. It is the near-quote that gets through, which is the more dangerous one."""
    import asyncio

    from rag_clients.faithfulness import LexicalFaithfulnessChecker, NumericallyGroundedChecker

    spans = [
        RetrievedSpan(chunk_id="c1", doc_id="d", text="Step 5 Content Curator Full Review: 48 hours."),
        RetrievedSpan(chunk_id="c2", doc_id="d", text="Step 6 Legal and Compliance Review for Category B: 72 hours."),
        RetrievedSpan(chunk_id="c3", doc_id="d", text="Step 7 Final Sign-Off: 24 hours."),
    ]
    answer = "Step 6 Legal and Compliance Review for Category B: 96 hours."  # source says 72

    bare = asyncio.run(LexicalFaithfulnessChecker().check(answer, spans))
    assert bare.faithful and bare.score == 1.0, "pinned: overlap rates this PERFECT"

    wrapped = asyncio.run(NumericallyGroundedChecker(LexicalFaithfulnessChecker()).check(answer, spans))
    assert not wrapped.faithful
    assert wrapped.unsupported_claims == [answer]  # the claim, not the bare digit


def test_a_grounded_answer_is_not_disturbed() -> None:
    import asyncio

    from rag_clients.faithfulness import LexicalFaithfulnessChecker, NumericallyGroundedChecker

    spans = [RetrievedSpan(chunk_id="c", doc_id="d", text="The night shift allowance is 400 rupees per shift.")]
    verdict = asyncio.run(
        NumericallyGroundedChecker(LexicalFaithfulnessChecker()).check(
            "The night shift allowance is 400 rupees per shift. [1]", spans
        )
    )
    assert verdict.faithful and verdict.unsupported_claims == []


def test_every_backend_gets_the_check_by_construction() -> None:
    """Wrapped at the factory, not inside one backend, so a new backend cannot forget it."""
    from rag_clients.faithfulness import NumericallyGroundedChecker, build_faithfulness_checker

    assert isinstance(build_faithfulness_checker(), NumericallyGroundedChecker)


# --- notation grounding (Slice 4) -------------------------------------------------------------


def test_notation_off_is_byte_identical_to_numeric_only() -> None:
    """Default OFF ⇒ the wrapper computes nothing extra: a unit mismatch the notation check would
    catch passes exactly as it does today (numeric-only). "5" IS in the source, so numeracy is
    silent; without notation the whole verdict is undisturbed."""
    import asyncio

    from rag_clients.faithfulness import LexicalFaithfulnessChecker, NumericallyGroundedChecker

    spans = [RetrievedSpan(chunk_id="c", doc_id="d", text="The dose is 5 mg once daily.")]
    answer = "The dose is 5 mg once daily."  # grounded; "5 g" is NOT what is asked here

    off = asyncio.run(NumericallyGroundedChecker(LexicalFaithfulnessChecker()).check(answer, spans))
    bare = asyncio.run(LexicalFaithfulnessChecker().check(answer, spans))
    assert off.faithful == bare.faithful
    assert off.unsupported_claims == bare.unsupported_claims


def test_notation_on_flags_a_unit_mismatch_the_numeric_check_misses() -> None:
    """ "5" is in the source (numeracy passes it), but "5 g" is not — the source says 5 mg. Only
    the notation twin catches this, and only when enabled."""
    import asyncio

    from rag_clients.faithfulness import LexicalFaithfulnessChecker, NumericallyGroundedChecker

    spans = [RetrievedSpan(chunk_id="c", doc_id="d", text="The dose is 5 mg once daily.")]
    answer = "The dose is 5 g once daily."

    off = asyncio.run(NumericallyGroundedChecker(LexicalFaithfulnessChecker()).check(answer, spans))
    assert off.faithful, "numeric-only does not see the unit"

    on = asyncio.run(
        NumericallyGroundedChecker(LexicalFaithfulnessChecker(), notation_enabled=True).check(answer, spans)
    )
    assert not on.faithful
    assert on.unsupported_claims == [answer]  # the carrying sentence, not the bare token


def test_notation_on_does_not_disturb_a_correct_preserved_formula() -> None:
    """The whole ethos: a genuinely-correct answer must pass even with notation ON. H₂SO₄ (as the
    generator preserves it) matches an ASCII source H2SO4."""
    import asyncio

    from rag_clients.faithfulness import LexicalFaithfulnessChecker, NumericallyGroundedChecker

    spans = [RetrievedSpan(chunk_id="c", doc_id="d", text="Add H2SO4 slowly, dose 5 mg per mL.")]
    answer = "Add H₂SO₄ slowly, dose 5 mg per mL."
    on = asyncio.run(
        NumericallyGroundedChecker(LexicalFaithfulnessChecker(), notation_enabled=True).check(answer, spans)
    )
    assert on.faithful and on.unsupported_claims == []


def test_numbers_and_notation_are_separate_signals() -> None:
    """A fabricated FIGURE and an ungrounded FORMULA both fire, and both carrying sentences are
    surfaced — a reviewer sees which is which (distinct log labels)."""
    import asyncio

    from rag_clients.faithfulness import LexicalFaithfulnessChecker, NumericallyGroundedChecker

    spans = [RetrievedSpan(chunk_id="c", doc_id="d", text="Sulfuric acid is H2SO4. Step 5 takes 48 hours.")]
    answer = "The reagent is H₂SO₃. The total is 96 hours."  # wrong formula + invented figure
    on = asyncio.run(
        NumericallyGroundedChecker(LexicalFaithfulnessChecker(), notation_enabled=True).check(answer, spans)
    )
    assert not on.faithful
    assert any("H₂SO₃" in c for c in on.unsupported_claims)
    assert any("96 hours" in c for c in on.unsupported_claims)


def test_with_notation_returns_self_when_unchanged() -> None:
    """The per-KB toggle seam must not clone the shared singleton when the flag doesn't change."""
    from rag_clients.faithfulness import LexicalFaithfulnessChecker, NumericallyGroundedChecker

    checker = NumericallyGroundedChecker(LexicalFaithfulnessChecker())
    assert checker.with_notation(False) is checker
    toggled = checker.with_notation(True)
    assert toggled is not checker and toggled.inner is checker.inner and toggled.notation_enabled


def test_factory_seeds_notation_from_setting(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from rag_clients.faithfulness import build_faithfulness_checker
    from rag_common import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("RAG_NOTATION_GROUNDING_ENABLED", "true")
    checker = build_faithfulness_checker()
    assert checker.notation_enabled  # type: ignore[attr-defined]
    get_settings.cache_clear()


def test_notation_hit_routes_to_flagged_for_review_but_off_stays_grounded() -> None:
    """T1 / H1 — the routed STATUS, not just `faithful`. With a stub judge that would score the
    answer GROUNDED (1.0), a unit mismatch ("5 g" vs source "5 mg") and an invented formula (SO₃
    vs SO₄) each route to FLAGGED_FOR_REVIEW when notation is ON (the notation hit caps the score
    into the flag band), and stay GROUNDED when notation is OFF (numeric-only routing unchanged —
    "5"/the unicode formula contribute no ASCII figure, so the stub's score is untouched)."""
    import asyncio

    from rag_clients.faithfulness import NumericallyGroundedChecker
    from rag_core import FaithfulnessVerdict, route_faithfulness

    class _GroundedStub:
        async def check(self, answer: str, spans: list[RetrievedSpan], query: str = "") -> FaithfulnessVerdict:
            return FaithfulnessVerdict(score=1.0, faithful=True, unsupported_claims=[])

    faithful, flag = 0.7, 0.4
    on = NumericallyGroundedChecker(
        _GroundedStub(), notation_enabled=True, faithful_threshold=faithful, flag_threshold=flag
    )
    off = NumericallyGroundedChecker(_GroundedStub())

    for answer, source in (
        ("The dose is 5 g once daily.", "The dose is 5 mg once daily."),  # unit mismatch
        ("The reagent is H₂SO₃.", "The reagent is H2SO4."),  # invented formula (SO3 vs SO4)
    ):
        spans = [RetrievedSpan(chunk_id="c", doc_id="d", text=source)]
        v_on = asyncio.run(on.check(answer, spans))
        assert v_on.notation_flagged
        assert v_on.score is not None
        assert route_faithfulness(v_on.score, faithful, flag) is AnswerStatus.FLAGGED_FOR_REVIEW

        v_off = asyncio.run(off.check(answer, spans))
        assert not v_off.notation_flagged
        assert v_off.score is not None
        assert route_faithfulness(v_off.score, faithful, flag) is AnswerStatus.GROUNDED


def test_numeric_only_routing_is_byte_identical_when_notation_off() -> None:
    """A numeric-only hit (invented FIGURE, notation OFF) keeps the inner judge's score and its
    `faithful=False` — routing reads the score, so it is unchanged from pre-Slice-4 (the numeric
    checker's own latent no-teeth property is intentionally left as-is)."""
    import asyncio

    from rag_clients.faithfulness import LexicalFaithfulnessChecker, NumericallyGroundedChecker

    spans = [RetrievedSpan(chunk_id="c", doc_id="d", text="Step 6 review takes 72 hours.")]
    answer = "Step 6 review takes 96 hours."  # 96 invented; near-verbatim, inner scores 1.0
    v = asyncio.run(NumericallyGroundedChecker(LexicalFaithfulnessChecker()).check(answer, spans))
    assert not v.notation_flagged
    assert v.score == 1.0 and not v.faithful  # score untouched (no cap); routing sees the raw score


def test_a_correct_pharma_answer_with_several_notations_is_not_flagged() -> None:
    """The acceptance case, at the wrapper: preserved formulae (H₂SO₄, C₉H₈O₄) + a dose + a
    temperature that ARE grounded (against ASCII / spelled-out sources) must all pass with
    notation ON — a guard that fires on a correct answer gets turned off."""
    import asyncio

    from rag_clients.faithfulness import LexicalFaithfulnessChecker, NumericallyGroundedChecker

    spans = [
        RetrievedSpan(chunk_id="c1", doc_id="d", text="Dissolve the API in H2SO4 at 5 mg/mL."),
        RetrievedSpan(chunk_id="c2", doc_id="d", text="Store at 25 degrees Celsius. Aspirin is C9H8O4."),
    ]
    answer = "Dissolve in H₂SO₄ at 5 mg/mL, store at 25°C; the active ingredient is C₉H₈O₄."
    # Notation contributes NOTHING: the wrapper's claims equal the bare inner checker's claims, so
    # every preserved formula / dose / temperature was recognised as grounded (the lexical judge's
    # own prose scoring is unchanged — that is a separate signal, not the notation guard firing).
    wrapped = asyncio.run(
        NumericallyGroundedChecker(LexicalFaithfulnessChecker(), notation_enabled=True).check(answer, spans)
    )
    bare = asyncio.run(LexicalFaithfulnessChecker().check(answer, spans))
    assert wrapped.unsupported_claims == bare.unsupported_claims


def test_notation_default_off_at_the_factory() -> None:
    from rag_clients.faithfulness import NumericallyGroundedChecker, build_faithfulness_checker

    checker = build_faithfulness_checker()
    assert isinstance(checker, NumericallyGroundedChecker)
    assert checker.notation_enabled is False  # platform default is OFF
