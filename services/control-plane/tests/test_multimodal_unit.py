"""Unit tests: vision twins (EchoVisionGateway, LocalHashVisionEmbedder) + caption helper."""

from __future__ import annotations

import pytest

from rag_clients.vision import EchoVisionGateway, LocalHashVisionEmbedder

_IMG_A = b"\x89PNG\r\n\x1a\n fake image bytes A"
_IMG_B = b"\xff\xd8\xff\xe0 fake jpeg bytes B totally different"


async def test_echo_gateway_is_deterministic_and_content_derived() -> None:
    gw = EchoVisionGateway()
    a1 = await gw.complete_with_images("describe", [_IMG_A])
    a2 = await gw.complete_with_images("describe", [_IMG_A])
    b = await gw.complete_with_images("describe", [_IMG_B])
    assert a1 == a2  # deterministic
    assert a1 != b  # different bytes → different output
    assert "1 image" in a1 and "describe" in a1


async def test_echo_gateway_reports_model_name() -> None:
    assert EchoVisionGateway().model_name == "echo-vision-dev"


async def test_vision_embedder_same_space_image_and_text() -> None:
    emb = LocalHashVisionEmbedder(dim=64)
    [iv] = await emb.embed_image([_IMG_A])
    [tv] = await emb.embed_text(["invoice total amount"])
    assert len(iv) == 64 and len(tv) == 64
    # Both are unit-normalised vectors in the same space.
    assert abs(sum(x * x for x in iv) - 1.0) < 1e-6
    assert abs(sum(x * x for x in tv) - 1.0) < 1e-6


async def test_vision_embedder_is_deterministic() -> None:
    emb = LocalHashVisionEmbedder(dim=32)
    v1 = await emb.embed_image([_IMG_A])
    v2 = await emb.embed_image([_IMG_A])
    assert v1 == v2


async def test_caption_helper_uses_the_vision_gateway(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MULTIMODAL_ENABLED", "true")
    from rag_common.config import get_settings

    get_settings.cache_clear()
    from svc_control_plane.multimodal import caption_image

    caption = await caption_image(_IMG_A, mime="image/png")
    assert caption  # non-empty description from the echo twin
    assert "image" in caption.lower()
