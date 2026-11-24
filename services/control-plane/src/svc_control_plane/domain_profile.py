"""Domain-profile orchestration (Slice 6, sub-slice A) — control-plane logic over the pure
:mod:`rag_core.domain_profile` helpers (mirrors :mod:`golden_gen` / :mod:`eval_schedule`).

Samples a KB's already-indexed chunks, classifies the corpus (heuristic default, optional LLM
refinement), and stores THREE ``tenant_config`` JSON keys — no new table, no migration:

* ``domain_profile``            — the ACTIVE profile: what the KB *is*.
* ``domain_profile_suggestion`` — the PENDING, unconfirmed knob recommendation (the crux of
  detect-and-suggest — detection writes this but moves NO knob).
* ``domain_profile_applied``    — an AUDIT of which knobs an operator confirmed and when, so an
  override is reversible and traceable.

The single most important rule: **detect-and-suggest, never auto-apply.** ``detect`` writes the
profile + the suggestion and touches no retrieval knob; ``apply_suggestion`` moves only the
query-time knobs an operator confirmed and REFUSES the reindex-class knobs unless the operator
explicitly opts in (so the profile can never silently trigger a reindex or a config/index drift).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rag_clients.factory import get_catalog
from rag_common import get_logger, get_settings
from rag_core.domain_profile import (
    DOMAINS,
    QUERY_TIME_KNOB_KEYS,
    REINDEX_KNOB_KEYS,
    DomainSignal,
    KnobSuggestion,
    classify_corpus,
    suggest_knobs,
)
from rag_core.parent_child import not_parent
from svc_control_plane import wiring
from svc_control_plane.continuous_eval import select_sample
from svc_control_plane.semantic_cache import cache as semantic_cache

_log = get_logger(__name__)

# tenant_config keys (JSON values). No migration — these ride the existing generic key/value store.
PROFILE_KEY = "domain_profile"
SUGGESTION_KEY = "domain_profile_suggestion"
APPLIED_KEY = "domain_profile_applied"


@dataclass(frozen=True, slots=True)
class AppliedKnobs:
    """The outcome of an operator confirming a knob suggestion (typed for the API layer)."""

    applied: dict[str, float] = field(default_factory=dict)
    refused_reindex: dict[str, float] = field(default_factory=dict)
    allow_reindex: bool = False
    applied_at: str = ""
    actor: str = ""
    reindex_warning: str | None = None


_MIN_CHUNK_LEN = 80  # skip tiny/boilerplate chunks — too little signal (mirrors golden_gen)
_SAMPLE_RATE = 0.5  # re-use the recency-sampler's fraction; deterministic, reproducible re-detection
_SAMPLE_CAP = 60  # bound the classification work on a large KB
# Stop gathering once we hold a few times the sample cap of candidates: `select_sample` takes at most
# `_SAMPLE_CAP`, so scanning the ENTIRE corpus is wasted work + memory on a large KB. This bounds a
# detect call to a fixed slice regardless of KB size.
_MAX_CANDIDATES = _SAMPLE_CAP * 3
_PER_DOC_CAP = 20  # cap per-doc reads so one huge document can't dominate the sample or the memory


async def _sample_texts(kb_id: str) -> list[str]:
    """Deterministic corpus sample: substantive child-chunk text spread across the KB's documents
    (round-robin so one big doc doesn't dominate), then the recency-sampler's reproducible cut.

    Bounded — caps per-doc reads and stops scanning documents once ``_MAX_CANDIDATES`` candidates
    are gathered, so a detect call never materialises the whole corpus on a large KB.

    FR-C2: parents are a non-searchable context bank — never sample them (``not_parent``)."""
    docs = await get_catalog().list_documents(kb_id)
    per_doc: list[list[str]] = []
    gathered = 0
    for d in docs:
        raw = not_parent(await get_catalog().document_chunks(kb_id, d.doc_id))
        texts = [c.text for c in raw if len(c.text) >= _MIN_CHUNK_LEN][:_PER_DOC_CAP]
        if texts:
            per_doc.append(texts)
            gathered += len(texts)
        if gathered >= _MAX_CANDIDATES:
            break  # enough spread of documents scanned — don't read the rest of a large KB
    ordered: list[str] = []
    idx = 0
    while per_doc and len(ordered) < _MAX_CANDIDATES:
        made_progress = False
        for texts in per_doc:
            if idx < len(texts):
                ordered.append(texts[idx])
                made_progress = True
                if len(ordered) >= _MAX_CANDIDATES:
                    break
        if not made_progress:
            break
        idx += 1
    # select_sample dedupes (first-seen order) and takes a reproducible fraction, capped.
    return select_sample(ordered, _SAMPLE_RATE, _SAMPLE_CAP)


async def _llm_refine(kb_id: str, signal: DomainSignal, samples: list[str]) -> DomainSignal:
    """Optional one-shot LLM label refinement (``RAG_DOMAIN_PROFILE_BACKEND=llm``), with the
    heuristic verdict as prior and as the fallback on ANY failure — mirrors resolve_graph_extractor.
    Only the DOMAIN label may change; the heuristic mix/evidence are kept as the auditable basis."""
    joined = "\n---\n".join(s.strip()[:600] for s in samples[:4])
    prompt = (
        "Classify the DOMAIN of this document collection as exactly one of: "
        f"{', '.join(DOMAINS)}. A heuristic scan suggests '{signal.domain}'. "
        "Reply with ONLY the single best label from that list, nothing else.\n\n"
        f"SAMPLES:\n{joined}\n\nLABEL:"
    )
    try:
        gateway = await wiring.resolve_tenant_gateway(kb_id)
        # A one-word label needs almost no budget; a low cap also stops a reasoning model from
        # spilling a <think> paragraph that a substring match could misread.
        raw = (await gateway.complete(prompt, max_tokens=16, temperature=0.0)).strip().lower()
        # Word-boundary match, NOT a substring: `d in raw` would resolve "encoded"/"encodes" to the
        # `code` domain. Tokenise on non-word chars and match a whole token.
        tokens = set(re.split(r"[^a-z]+", raw))
        label = next((d for d in DOMAINS if d in tokens), None)
        if label is None:
            _log.warning("domain_profile_llm_unparsable_fallback", kb_id=kb_id, raw=raw[:120])
            return signal
        if label != signal.domain:
            _log.info("domain_profile_llm_refined", kb_id=kb_id, heuristic=signal.domain, llm=label)
        return DomainSignal(domain=label, score=signal.score, mix=signal.mix, evidence=signal.evidence)
    except Exception as exc:  # a flaky model never blocks detection — keep the heuristic verdict
        _log.warning("domain_profile_llm_failed_fallback", kb_id=kb_id, error=str(exc)[:160])
        return signal


async def detect(kb_id: str) -> tuple[DomainSignal, KnobSuggestion]:
    """Classify the KB's corpus and compute a knob SUGGESTION — writing ``domain_profile`` +
    ``domain_profile_suggestion`` to tenant_config. Moves NO knob (that is ``apply_suggestion``).
    Returns the active signal + the pending suggestion."""
    settings = get_settings()
    samples = await _sample_texts(kb_id)
    signal = classify_corpus(samples)
    backend = settings.domain_profile_backend
    if backend == "llm" and samples:
        signal = await _llm_refine(kb_id, signal, samples)
    suggestion = suggest_knobs(signal)

    catalog = get_catalog()
    now = datetime.now(UTC).isoformat()
    profile_doc = {
        "domain": signal.domain,
        "score": signal.score,
        "mix": signal.mix,
        "evidence": signal.evidence,
        "detected_at": now,
        "backend": backend,
        "sampled_chunks": len(samples),
    }
    suggestion_doc = {
        "knobs": suggestion.query_time_knobs,
        "reindex_knobs": suggestion.reindex_knobs,
        "rationale": suggestion.rationale,
        "detected_at": now,
    }
    await catalog.set_tenant_config(kb_id, PROFILE_KEY, json.dumps(profile_doc))
    await catalog.set_tenant_config(kb_id, SUGGESTION_KEY, json.dumps(suggestion_doc))
    _log.info(
        "domain_profile_detected",
        kb_id=kb_id,
        domain=signal.domain,
        score=signal.score,
        backend=backend,
        sampled=len(samples),
    )
    return signal, suggestion


async def _load_json(kb_id: str, key: str) -> dict[str, object]:
    raw = (await get_catalog().get_tenant_config(kb_id)).get(key)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


async def load_profile(kb_id: str) -> dict[str, object]:
    """The KB's active domain profile (empty dict if never detected)."""
    return await _load_json(kb_id, PROFILE_KEY)


async def load_suggestion(kb_id: str) -> dict[str, object]:
    """The KB's pending knob suggestion (empty dict if never detected)."""
    return await _load_json(kb_id, SUGGESTION_KEY)


async def load_applied(kb_id: str) -> dict[str, object]:
    """The audit of which knobs an operator confirmed (empty dict if none applied)."""
    return await _load_json(kb_id, APPLIED_KEY)


async def relation_vocab_domain(kb_id: str) -> str:
    """The active profile's domain label ('' if never detected) — used by the ingest path to
    decide whether the KB declares a domain-relation vocabulary (sub-slice B)."""
    profile = await load_profile(kb_id)
    return str(profile.get("domain", ""))


async def apply_suggestion(
    kb_id: str,
    *,
    query_knobs: dict[str, float],
    reindex_knobs: dict[str, float] | None = None,
    allow_reindex: bool = False,
    actor: str,
) -> AppliedKnobs:
    """Operator confirmation: write ONLY the accepted query-time knobs via the SAME
    ``set_tenant_config`` call ``set_kb_settings`` uses, and record ``domain_profile_applied``.

    Reindex-class knobs are REFUSED unless ``allow_reindex`` is true (the operator explicitly
    opted in AND was warned): the profile never silently triggers a reindex or flips
    multivector/parent_child sizing. A key that is not a recognised knob of its class is rejected,
    so a reindex-only key can never be smuggled through the query-time bucket."""
    reindex_knobs = reindex_knobs or {}
    catalog = get_catalog()

    # VALIDATE EVERY KEY (both buckets) BEFORE writing anything, so a bad reindex key raises 422
    # with NO partial apply — a rejected request never leaves a query-time knob persisted.
    for key in query_knobs:
        if key not in QUERY_TIME_KNOB_KEYS:
            raise _bad_knob(key, "query-time")
    for key in reindex_knobs:
        if key not in REINDEX_KNOB_KEYS:
            raise _bad_knob(key, "reindex")

    applied: dict[str, float] = {}
    for key, value in query_knobs.items():
        await catalog.set_tenant_config(kb_id, key, str(float(value)))
        applied[key] = float(value)

    refused: dict[str, float] = {}
    for key, value in reindex_knobs.items():
        if not allow_reindex:
            refused[key] = float(value)  # SUGGEST-ONLY: not written without explicit opt-in
            continue
        await catalog.set_tenant_config(kb_id, key, str(float(value)))
        applied[key] = float(value)

    # Query-time knobs (exact_boost, notation_grounding_enabled, …) change how answers are produced
    # RIGHT NOW; a stale semantic-cache entry would keep serving the pre-apply behaviour. Invalidate
    # so the change takes effect immediately (mirrors ingest/reindex/delete invalidation).
    if any(k in QUERY_TIME_KNOB_KEYS for k in applied):
        semantic_cache.invalidate(kb_id)

    warning = (
        "these knobs change what is indexed and require a reindex_document pass; they were "
        "NOT applied (re-send with allow_reindex=true to apply and then reindex the KB)."
        if refused
        else None
    )
    result = AppliedKnobs(
        applied=applied,
        refused_reindex=refused,
        allow_reindex=allow_reindex,
        applied_at=datetime.now(UTC).isoformat(),
        actor=actor,
        reindex_warning=warning,
    )
    record: dict[str, object] = {
        "applied": applied,
        "refused_reindex": refused,
        "allow_reindex": allow_reindex,
        "applied_at": result.applied_at,
        "actor": actor,
    }
    if warning:
        record["reindex_warning"] = warning
    await catalog.set_tenant_config(kb_id, APPLIED_KEY, json.dumps(record))
    _log.info(
        "domain_profile_applied",
        kb_id=kb_id,
        applied=sorted(applied),
        refused=sorted(refused),
        allow_reindex=allow_reindex,
        actor=actor,
    )
    return result


def _bad_knob(key: str, expected_class: str) -> Exception:
    from rag_common import ValidationError  # noqa: PLC0415 - avoid import cycle at module load

    return ValidationError(
        f"'{key}' is not a recognised {expected_class} knob",
        details={"key": key, "expected_class": expected_class},
    )
