"""Chandra (Datalab) — ONE HTTP client, two roles: OCR engine + vision gateway.

Chandra is a VLM-based document model: strong OCR (tables, forms, handwriting, reading
order, Indic scripts) AND open image question-answering. Rather than run a separate
general VLM, we point BOTH the OCR backend (``RAG_OCR_BACKEND=chandra``) and the
multimodal vision gateway (``vision_llm_backend=chandra``) at it — configured once via the
``chandra_*`` settings.

The transport (:class:`ChandraClient`) speaks the OpenAI-compatible vision chat API — the
shape a self-hosted Chandra (vLLM) or a hosted OpenAI-compatible endpoint exposes — with the
image sent as a base64 ``image_url`` content part. It is API-key ready (for a hosted Chandra
endpoint) and sovereign (the host must be egress-allowlisted; the guard is never loosened).

The OCR-engine adapter that turns Chandra's output into a doc-intelligence ``Extraction``
lives in the doc-intelligence service (it constructs service-owned types); this library holds
only the shared transport + the :class:`ChandraVisionGateway`.
"""

from __future__ import annotations

import base64
import re

import httpx

from rag_common import ModelGatewayError, assert_sovereign, get_logger, get_settings, model_tls_verify

_log = get_logger(__name__)


# C0 control chars EXCEPT tab/newline/CR — Postgres text/JSON cannot store NUL (see datalab.py).
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize(text: str) -> str:
    """Strip NUL + other C0 control chars from OCR/vision text before it reaches Postgres."""
    return _CTRL_RE.sub("", text).strip()


def _data_uri(image: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"


class ChandraClient:
    """Shared OpenAI-compatible vision transport to a Chandra endpoint (OCR + Q&A both use it)."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout: float = 180.0,
        *,
        api_key: str | None = None,
        api_key_header: str | None = None,
    ) -> None:
        settings = get_settings()
        self._url = (url or settings.chandra_url).rstrip("/")
        self._model = model or settings.chandra_model
        # Self-hosted vLLM OR a sanctioned hosted Chandra API — either way the host must be on
        # RAG_ALLOWED_EGRESS_HOSTS. Sovereignty is enforced here, not bypassed by a key.
        assert_sovereign(self._url)
        self._max_tokens = settings.llm_max_tokens
        # An org-wide OCR/vision connection carries its OWN endpoint AND credential; the env key
        # belongs to the env endpoint, so sending it to a different host would be both wrong and a
        # credential leak. `api_key=""` therefore means "no key", not "fall back to the env key".
        key = settings.chandra_api_key if api_key is None else api_key
        headers = {(api_key_header or settings.chandra_api_key_header): key} if key else {}
        self._client = httpx.AsyncClient(
            base_url=self._url, timeout=timeout, headers=headers, verify=model_tls_verify()
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def run(
        self, prompt: str, images: list[bytes], *, mime: str = "image/jpeg", max_tokens: int | None = None
    ) -> str:
        """One vision chat turn: a text instruction + one or more images → the model's text."""
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": _data_uri(img, mime)}} for img in images)
        try:
            resp = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelGatewayError(
                f"Chandra at {self._url} failed ({type(exc).__name__}: {exc})",
                details={"url": self._url, "model": self._model},
            ) from exc
        return _sanitize(str(resp.json()["choices"][0]["message"]["content"]))


class ChandraVisionGateway:
    """Chandra as the multimodal vision gateway (implements :class:`VisionLLMGateway`)."""

    def __init__(self, client: ChandraClient | None = None) -> None:
        self._client = client or ChandraClient()

    @property
    def model_name(self) -> str:
        return self._client.model_name

    async def complete_with_images(
        self, prompt: str, images: list[bytes], *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> str:
        return await self._client.run(prompt, images, max_tokens=max_tokens)
