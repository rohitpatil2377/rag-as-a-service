"""Multimodal (vision) Protocols — image-aware RAG (additive, default-OFF).

Two seams, mirroring the text side so a vision backend is a pure adapter swap:

* :class:`VisionLLMGateway` — a vision-capable LLM (Qwen2.5-VL / LLaVA / InternVL via
  vLLM) that answers a prompt while SEEING image bytes. This is the genuinely-new
  capability: it reads a diagram/scan/photo, not just OCR'd text.
* :class:`VisionEmbedder` — a CLIP-style encoder that maps images AND text into ONE
  vector space, so a text query can retrieve an image by visual similarity. Wired
  (behind ``vision_recall_enabled``): at ingest an image CLIP vector is indexed into a
  SEPARATE per-tenant vision collection, and the hybrid retriever adds a THIRD leg that
  searches it — so an image is retrievable by VISUAL similarity, not only its caption.

Both are sovereign: implementations call self-hosted, egress-allowlisted endpoints
only. Every image chunk is ALSO indexed by its caption on the ordinary text path, so a
text query still finds it even with cross-modal recall off; the cross-modal leg is an
additive bonus (a vision-backend fault degrades to text-only). The vision gateway is
consulted only at generation when a retrieved chunk carries image bytes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionLLMGateway(Protocol):
    """A vision-capable sovereign LLM: answers a prompt while seeing image bytes."""

    async def complete_with_images(
        self, prompt: str, images: list[bytes], *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> str: ...

    @property
    def model_name(self) -> str: ...


@runtime_checkable
class VisionEmbedder(Protocol):
    """CLIP-style encoder: images and text share ONE vector space (cross-modal retrieval)."""

    @property
    def dim(self) -> int: ...

    async def embed_image(self, images: list[bytes]) -> list[list[float]]: ...

    async def embed_text(self, texts: list[str]) -> list[list[float]]: ...
