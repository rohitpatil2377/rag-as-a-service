"""Building the PaddleOCR reader must not quietly abandon the air-gap (P35).

``model_dir`` points at pre-mirrored det/rec/cls models so an air-gapped deployment never
reaches the internet for weights — a sovereignty guarantee the docstring makes explicitly.
Newer PaddleOCR releases dropped some constructor kwargs, and the retry for that dropped
*everything* except ``lang``, including the model directories. On such a release, a sovereign
deployment silently fell back to PaddleOCR resolving its own models over the network, and the
degraded reader was then cached for the process lifetime.

Not all kwargs are equal, so the retry must not treat them as if they were: rotation handling
and log verbosity are tuning, the model directories are the guarantee.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest
import structlog
from svc_doc_intelligence import ocr as ocr_mod
from svc_doc_intelligence.ocr import REASON_OCR_TUNING_DROPPED, _paddle_reader

from rag_common import ConfigError


class _Paddle:
    """A PaddleOCR stand-in that rejects whichever kwargs the release does not know."""

    def __init__(self, rejects: set[str]) -> None:
        self.rejects = rejects
        self.built_with: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> Any:
        bad = self.rejects & set(kwargs)
        if bad:
            raise TypeError(f"__init__() got an unexpected keyword argument {sorted(bad)[0]!r}")
        self.built_with = kwargs
        return self


@pytest.fixture(autouse=True)
def _clean_reader_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    ocr_mod._PADDLE_READERS.clear()
    yield
    ocr_mod._PADDLE_READERS.clear()


def _install(paddle: _Paddle, monkeypatch: pytest.MonkeyPatch) -> None:
    module = type(sys)("paddleocr")
    module.PaddleOCR = paddle  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)


def _mirror(tmp_path: Any) -> str:
    for sub in ("det", "rec", "cls"):
        (tmp_path / sub).mkdir()
    return str(tmp_path)


def test_dropping_tuning_kwargs_keeps_the_mirrored_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    paddle = _Paddle(rejects={"show_log", "use_angle_cls"})
    _install(paddle, monkeypatch)

    with structlog.testing.capture_logs() as logs:
        built = _paddle_reader("en", _mirror(tmp_path))

    assert built.reader is paddle
    assert paddle.built_with["det_model_dir"].endswith("/det")  # the guarantee survived
    assert "show_log" not in paddle.built_with  # the tuning did not
    assert REASON_OCR_TUNING_DROPPED in built.degradations
    assert [e for e in logs if e.get("event") == "paddle_tuning_kwargs_dropped"]


def test_a_release_that_cannot_take_mirrored_models_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Resolving models over the network is not a lesser outcome for an air-gapped install —
    it is the thing model_dir exists to prevent. Fail closed and say why."""
    _install(_Paddle(rejects={"show_log", "det_model_dir"}), monkeypatch)

    with pytest.raises(ConfigError, match="model"):
        _paddle_reader("en", _mirror(tmp_path))


def test_without_mirrored_models_the_retry_is_a_plain_degradation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """No model_dir configured means the operator already accepted PaddleOCR's own bootstrap,
    so there is no guarantee to break — only tuning to report."""
    paddle = _Paddle(rejects={"show_log", "use_angle_cls"})
    _install(paddle, monkeypatch)

    built = _paddle_reader("en", "")
    assert built.reader is paddle
    assert built.degradations == (REASON_OCR_TUNING_DROPPED,)


def test_the_degradation_is_cached_with_the_reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The reader is cached for the process lifetime, so the reason has to be too — otherwise
    only the first document of the run reports what happened to every document of the run."""
    _install(_Paddle(rejects={"show_log", "use_angle_cls"}), monkeypatch)
    first = _paddle_reader("en", "")
    second = _paddle_reader("en", "")
    assert second.degradations == first.degradations == (REASON_OCR_TUNING_DROPPED,)
