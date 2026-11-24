"""Vision adapters — implement the multimodal Protocols (additive, default-OFF).

* :class:`OpenAICompatVisionGateway` — a self-hosted vision LLM (Qwen2.5-VL / LLaVA via
  vLLM) over the OpenAI chat API's multimodal ``image_url`` content parts. Sovereign.
* :class:`EchoVisionGateway` — deterministic zero-model twin (CI/demo): returns a stable
  description derived from the prompt + image byte-length, so the multimodal path runs
  end-to-end with no GPU/model, exactly like ``ExtractiveDevGateway`` on the text side.
* :class:`TEIVisionEmbedder` / :class:`LocalHashVisionEmbedder` — the CLIP-style
  cross-modal embedder (real TEI + deterministic twin) for Phase-2 visual retrieval.

The working ingest path retrieves images by their auto-caption via the ordinary text
embedder; the vision gateway is what reads the image at generation time.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re

import httpx

from rag_clients.llm_gateway import _NO_REASONING_SYSTEM, _answer_text, _strip_reasoning
from rag_common import ModelGatewayError, assert_sovereign, get_logger, get_settings, model_tls_verify

_log = get_logger(__name__)


def _messages(content: object) -> list[dict[str, object]]:
    """The chat turns for a vision call, with reasoning suppressed when it is switched off."""
    out: list[dict[str, object]] = []
    if get_settings().llm_reasoning_effort.lower() in ("off", "none", "disabled", "false", "0"):
        out.append({"role": "system", "content": _NO_REASONING_SYSTEM})
    out.append({"role": "user", "content": content})
    return out


def _data_uri(image: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"


class OpenAICompatVisionGateway:
    """Vision LLM over the OpenAI-compatible chat API (multimodal content parts)."""

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
        self._url = (url or settings.vision_llm_url).rstrip("/")
        self._model = model or settings.vision_llm_model
        # Sovereignty: the endpoint host must be egress-allowlisted. Having a key never bypasses it.
        assert_sovereign(self._url)
        self._max_tokens = settings.llm_max_tokens
        key = api_key if api_key is not None else settings.vision_llm_api_key
        header = api_key_header or settings.vision_llm_api_key_header
        # A vision endpoint that IS the text endpoint shares its credential. Reported live: the
        # same host served text answers fine and 401'd on every image, because the key had been
        # configured once (RAG_LLM_API_KEY) and the vision path reads its own variable. Making
        # an operator paste the same secret twice is a footgun that fails only on the image path,
        # long after the model looks proven. Scoped to an EXACT endpoint match, so one provider's
        # credential is never sent to another host.
        if not key and self._url == settings.vllm_url.rstrip("/") and settings.llm_api_key:
            key, header = settings.llm_api_key, settings.llm_api_key_header
            _log.info("vision_reusing_llm_key", url=self._url, detail="same endpoint as the text model")
        # Same convention as the text gateway: an Authorization header carries the Bearer scheme
        # (a bare token there is rejected); other header names take the raw key unchanged.
        if key and header.lower() == "authorization" and not key.lower().startswith("bearer "):
            key = f"Bearer {key}"
        headers = {header: key} if key else {}
        if not headers:
            _log.warning(
                "vision_gateway_unauthenticated",
                url=self._url,
                detail="no vision API key configured and the endpoint is not the text model's — "
                "a hosted endpoint will answer 401. Set RAG_VISION_LLM_API_KEY.",
            )
        self._client = httpx.AsyncClient(
            base_url=self._url, timeout=timeout, headers=headers, verify=model_tls_verify()
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def complete_with_images(
        self, prompt: str, images: list[bytes], *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": _data_uri(img)}} for img in images)
        try:
            resp = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    # Suppress the planning preamble AT THE SOURCE, exactly as the text gateway
                    # does. Stripping it afterwards does not give back the budget it consumed:
                    # the same page captioned fully on one call and returned six characters on
                    # the next, because the preamble ate max_tokens before the caption was
                    # written. This is the one path where the model's output IS the document's
                    # indexed text, so a truncated answer is a silently incomplete document.
                    "messages": _messages(content),
                    "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelGatewayError(
                f"vision gateway at {self._url} failed ({type(exc).__name__}: {exc})",
                details={"url": self._url, "model": self._model},
            ) from exc
        # The SAME hygiene the text gateway applies. A caption is indexed as the document's
        # searchable text and quoted in citations, so a leaked reasoning preamble ("thought A
        # plain white image containing three lines…" — observed live on Gemma) becomes part of
        # the corpus, and a null content would be indexed as the literal string "None".
        message = resp.json()["choices"][0]["message"]
        return _strip_reasoning(_answer_text(message))


class EchoVisionGateway:
    """Deterministic, zero-model vision gateway (CI/demo twin).

    Returns a stable, content-derived description so the multimodal pipeline is exercised
    end-to-end without a GPU. NOT a real model — the sovereign path is a self-hosted VLM.
    """

    @property
    def model_name(self) -> str:
        return "echo-vision-dev"

    async def complete_with_images(
        self, prompt: str, images: list[bytes], *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> str:
        # Deterministic: a short digest of each image + the prompt, so tests can assert on it.
        sizes = ", ".join(f"{len(img)}B/{hashlib.sha1(img).hexdigest()[:8]}" for img in images)  # noqa: S324
        return f"[vision:{len(images)} image(s) {sizes}] {prompt.strip()[:200]}"


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LocalHashVisionEmbedder:
    """Deterministic CLIP-style twin: image bytes and text hash into ONE space.

    An image is embedded from a stable digest of its bytes; text from its tokens. This is
    a placeholder for cosine geometry in tests — NOT semantic. The real path is TEI CLIP.
    """

    def __init__(self, dim: int | None = None) -> None:
        self._dim = dim or min(get_settings().vision_embed_dim, 256)

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_image(self, images: list[bytes]) -> list[list[float]]:
        return [self._embed_tokens(hashlib.sha1(img).hexdigest()) for img in images]  # noqa: S324

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_tokens(" ".join(_tokens(t))) for t in texts]

    def _embed_tokens(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in text.split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # noqa: S324
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class TEIVisionEmbedder:
    """Real CLIP-style embedder over a self-hosted TEI server (image + text endpoints)."""

    def __init__(self, url: str | None = None, model: str | None = None, timeout: float = 60.0) -> None:
        settings = get_settings()
        self._url = (url or settings.vision_tei_url).rstrip("/")
        self._model = model or settings.vision_embed_model
        self._dim = settings.vision_embed_dim
        assert_sovereign(self._url)
        self._client = httpx.AsyncClient(base_url=self._url, timeout=timeout, verify=model_tls_verify())

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return await self._embed({"inputs": texts})

    async def embed_image(self, images: list[bytes]) -> list[list[float]]:
        return await self._embed({"inputs": [_data_uri(img) for img in images]})

    async def _embed(self, payload: dict[str, object]) -> list[list[float]]:
        try:
            resp = await self._client.post("/embed", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelGatewayError(
                f"vision embedder at {self._url} failed ({type(exc).__name__}: {exc})",
                details={"url": self._url, "model": self._model},
            ) from exc
        return [[float(x) for x in row] for row in resp.json()]
