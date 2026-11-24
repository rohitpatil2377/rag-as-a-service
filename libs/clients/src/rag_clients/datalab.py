"""Datalab (hosted Chandra) — async /convert client, two roles: OCR engine + vision gateway.

Datalab's HOSTED API (the free/paid cloud) is NOT OpenAI-compatible: you POST a document to
``/api/v1/convert`` (multipart), get a ``request_check_url``, and poll it until
``status == "complete"`` for the transcribed markdown. This is the shape to use when testing
against the hosted Chandra API before a GPU host exists.

(The SELF-HOSTED Chandra path — Chandra served via vLLM — speaks the OpenAI-compatible vision
chat API instead; that is :mod:`rag_clients.chandra`. Same model, two serving shapes:
``RAG_OCR_BACKEND=datalab`` for the hosted API, ``=chandra`` for self-hosted vLLM.)

The transport is sovereign — the Datalab host must be egress-allowlisted. For OCR the ``run``
result IS the transcription; used as the vision gateway it returns that transcription as the
image's description (Datalab /convert is OCR, not open Q&A — fine for captioning at ingest).
"""

from __future__ import annotations

import asyncio
import re

import httpx

from rag_common import ModelGatewayError, assert_sovereign, get_logger, get_settings, model_tls_verify

_log = get_logger(__name__)


# C0 control chars EXCEPT tab (\x09), newline (\x0a), CR (\x0d).
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize(text: str) -> str:
    """Strip NUL (\\x00) and other C0 control chars from model text.

    PostgreSQL text/JSON columns cannot store NUL — an OCR transcription that contains one
    (they occasionally do) would fail the insert with UntranslatableCharacter. Stripping here,
    at the text source, keeps every downstream store (catalog, audit JSON) safe."""
    return _CTRL_RE.sub("", text).strip()


class DatalabClient:
    """Async submit→poll client for the hosted Datalab /convert endpoint (OCR to markdown)."""

    def __init__(
        self, url: str | None = None, *, timeout: float = 60.0, poll_seconds: float = 3.0, max_polls: int = 40
    ) -> None:
        settings = get_settings()
        self._base = (url or settings.datalab_url).rstrip("/")
        assert_sovereign(self._base)  # sovereignty: the Datalab host must be egress-allowlisted
        self._key = settings.datalab_api_key
        self._header = settings.datalab_api_key_header
        self._poll = poll_seconds
        self._max_polls = max_polls
        self._client = httpx.AsyncClient(timeout=timeout, verify=model_tls_verify())

    @property
    def model_name(self) -> str:
        return "datalab-chandra"

    async def convert(self, data: bytes, mime: str, *, output_format: str = "markdown") -> str:
        """Submit one document/image, poll to completion, return the transcription."""
        headers = {self._header: self._key} if self._key else {}
        try:
            resp = await self._client.post(
                f"{self._base}/convert",
                headers=headers,
                files={"file": ("doc", data, mime or "image/png")},
                data={"output_format": output_format},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelGatewayError(
                f"Datalab submit failed ({type(exc).__name__}: {exc})", details={"url": self._base}
            ) from exc
        body = resp.json()
        if not body.get("success"):
            raise ModelGatewayError("Datalab rejected the document", details={"error": body.get("error")})
        check_url = str(body["request_check_url"])

        for _ in range(self._max_polls):
            await asyncio.sleep(self._poll)
            try:
                s = (await self._client.get(check_url, headers=headers)).json()
            except httpx.HTTPError as exc:
                raise ModelGatewayError(
                    f"Datalab poll failed ({type(exc).__name__}: {exc})", details={"url": check_url}
                ) from exc
            status = s.get("status")
            if status == "complete":
                return _sanitize(str(s.get(output_format) or s.get("markdown") or s.get("output") or ""))
            if status == "failed" or s.get("error"):
                raise ModelGatewayError("Datalab conversion failed", details={"error": s.get("error")})
        raise ModelGatewayError("Datalab conversion timed out", details={"polls": self._max_polls})


class DatalabVisionGateway:
    """Datalab as the vision gateway (implements VisionLLMGateway).

    /convert is OCR, not open Q&A — so this returns the image's TRANSCRIPTION regardless of the
    prompt. That is exactly what the caption-at-ingest path needs; for open visual reasoning at
    answer time, use a self-hosted VLM (the ``chandra``/``openai_compat`` backend) instead.
    """

    def __init__(self, client: DatalabClient | None = None) -> None:
        self._client = client or DatalabClient()

    @property
    def model_name(self) -> str:
        return self._client.model_name

    async def complete_with_images(
        self, prompt: str, images: list[bytes], *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> str:
        if not images:
            return ""
        return await self._client.convert(images[0], "image/png")
