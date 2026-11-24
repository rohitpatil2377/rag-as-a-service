"""Runtime feature-flag toggles: safe set is live-toggleable; security set is locked."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from svc_control_plane import feature_flags
from svc_control_plane.feature_flags import InMemoryFlagStore, effective, set_flag, snapshot

from rag_common.config import get_settings
from rag_common.errors import ValidationError


@pytest.fixture(autouse=True)
def _clean(inmemory: None) -> Iterator[None]:
    """Restore the environment afterwards.

    ``set_flag``/``load_overrides`` write ``os.environ`` DIRECTLY — that is the mechanism, not
    an accident: a runtime toggle has to change what ``get_settings`` reads. But nothing here
    was putting it back, so every flag these tests turned on stayed on for the whole session.
    Measured: ``RAG_QUERY_TRANSFORM_ENABLED=true`` leaked out of
    ``test_load_overrides_applies_persisted`` and put an extra gateway call (the FR-D6 query
    transformer) in front of every later pipeline test in the run."""
    before = dict(os.environ)
    get_settings.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(before)
    get_settings.cache_clear()


async def test_snapshot_has_toggleable_and_locked() -> None:
    view = snapshot()
    names = {f.name for f in view.flags}
    assert "guardrails" in names and "ocr" in names  # toggleable
    assert "sovereign_enforce" in names and "pii_masking" in names  # locked
    locked = {f.name for f in view.flags if f.locked}
    assert "sovereign_enforce" in locked and "byok_kms" in locked
    assert "guardrails" not in locked


async def test_toggle_changes_effective_state() -> None:
    store = InMemoryFlagStore()
    before = effective("adaptive_router")
    await set_flag("adaptive_router", not before, store)
    assert effective("adaptive_router") is (not before)
    # persisted
    assert (await store.get_all())["adaptive_router"] is (not before)


async def test_string_backed_flag_ocr_toggles_backend() -> None:
    store = InMemoryFlagStore()
    await set_flag("ocr", True, store)
    assert get_settings().ocr_backend == "tesseract"
    await set_flag("ocr", False, store)
    assert get_settings().ocr_backend == "noop"


async def test_locked_or_unknown_flag_refused() -> None:
    store = InMemoryFlagStore()
    with pytest.raises(ValidationError):
        await set_flag("sovereign_enforce", False, store)  # security boundary — not toggleable
    with pytest.raises(ValidationError):
        await set_flag("does_not_exist", True, store)


async def test_pii_masking_reports_always_on() -> None:
    pii = next(f for f in snapshot().flags if f.name == "pii_masking")
    assert pii.locked is True and pii.enabled is True


async def test_load_overrides_applies_persisted() -> None:
    store = InMemoryFlagStore()
    await store.put("query_transform", True)
    await store.put("nl2sql", True)
    await feature_flags.load_overrides(store)
    assert effective("query_transform") is True
    assert effective("nl2sql") is True
