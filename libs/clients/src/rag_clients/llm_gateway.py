"""LLM gateway adapters (implement :class:`LLMGateway`) — FR-E1.

Backends:

* :class:`OpenAICompatGateway` — production/dev path for any self-hosted, OpenAI-
  compatible server. Uses the **chat completions** API (`/chat/completions`), which is
  what both **vLLM** (GPU) and **llama.cpp**'s server (CPU) expose, so the same adapter
  serves both — the only difference is the endpoint URL and which model is loaded.
  Sovereign: the endpoint is validated against the egress allowlist on construction.
  ``VLLMGateway`` remains as a backward-compatible alias.
* :class:`ExtractiveDevGateway` — zero-GPU/zero-model fallback (approved). Composes a
  grounded answer extractively from the numbered SOURCES in the prompt, preserving the
  [n] citation contract. Never selected in prod.

Selection is by ``RAG_LLM_BACKEND`` via :func:`build_llm_gateway`.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import cast

import httpx

from rag_common import (
    ConfigError,
    ModelGatewayError,
    assert_sovereign,
    get_logger,
    get_settings,
    model_tls_verify,
)
from rag_core import sampling
from rag_core.grounding import SOURCE_CLOSE, SOURCE_OPEN

_log = get_logger(__name__)

# Reasoning tags a chain-of-thought model may emit INLINE inside the answer content.
# Kept case-insensitive / DOTALL so a multi-line block is matched whole. The set is the
# common family: <think> (Qwen-thinking / QwQ / DeepSeek-R1), <thinking>, <reasoning>.
_REASONING_TAGS = ("think", "thinking", "reasoning")
_REASONING_BLOCK_RE = re.compile(
    r"<(?P<tag>think|thinking|reasoning)\b[^>]*>.*?</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
# An opening reasoning tag on its own — used to detect an UNCLOSED block (budget ran out
# mid-thought) after complete blocks have been removed.
_REASONING_OPEN_RE = re.compile(r"<(?:think|thinking|reasoning)\b[^>]*>", re.IGNORECASE)
# Longest reasoning-tag token, used to size the tail we hold back on the streaming path so a
# tag split across two chunks is never mistaken for answer text.
_MAX_TAG_LEN = max(len(t) for t in _REASONING_TAGS) + len("</>")

# Some models (Gemma-class in particular) emit a free-text "thought"/planning preamble as ORDINARY
# content on complex prompts — not a <think> block — so enable_thinking / reasoning_effort don't
# catch it and there is no tag delimiter for the strip to key on. When reasoning is meant to be OFF
# we add this system turn to suppress the behaviour at the source. It is harmless for models that
# were not going to think, and it does not weaken the grounding prompt (which still arrives as the
# user turn, unchanged).
_NO_REASONING_SYSTEM = (
    "Output only the final answer. Do not output any planning, analysis, a constraint list, a draft, "
    "self-checks, or any section labelled 'thought', 'thinking', or 'reasoning'. Respond with the "
    "answer text only, obeying the user's formatting and citation instructions."
)


def _strip_reasoning_tags(text: str) -> str:
    """Remove any complete ``<think>…</think>`` / ``<thinking>…</thinking>`` /
    ``<reasoning>…</reasoning>`` block from ``text`` (case-insensitive, DOTALL) and return the
    remainder, stripped.

    A reasoning model may return its chain-of-thought INLINE, polluting the published answer.
    Removal is repeated until stable so adjacent or nested blocks are all cleared. If nothing but
    reasoning was returned, the result is ``""`` — the caller then treats it as an empty
    generation and abstains, rather than publishing the model's working. Text with no reasoning
    tag passes through unchanged apart from surrounding whitespace."""
    prev = None
    cur = text
    # Non-greedy repeated removal: each pass strips the innermost complete blocks; loop until the
    # string stops changing so nested/back-to-back blocks are fully cleared.
    while prev != cur:
        prev = cur
        cur = _REASONING_BLOCK_RE.sub("", cur)
    return cur.strip()


def _failure_kind(exc: Exception) -> str:
    """Classify a raw httpx failure. A timeout is NOT an unreachable endpoint — conflating the
    two sent one investigation chasing endpoint instability that was really our own configured
    client limit, so every timeout flavour keeps its own name here."""
    if isinstance(exc, httpx.HTTPStatusError):
        return "http_status"
    for cls, kind in (
        (httpx.ConnectTimeout, "connect_timeout"),
        (httpx.ReadTimeout, "read_timeout"),
        (httpx.WriteTimeout, "write_timeout"),
        (httpx.PoolTimeout, "pool_timeout"),
        (httpx.TimeoutException, "timeout"),
        (httpx.ConnectError, "unreachable"),
    ):
        if isinstance(exc, cls):
            return kind
    return "transport_error"


def _raise_clean(
    exc: Exception,
    endpoint: str,
    model: str,
    *,
    timeout: float | None = None,
    elapsed: float | None = None,
) -> ModelGatewayError:
    """Turn a raw httpx failure into an actionable model-connection error (never leaks the
    full URL/creds; points the operator at the Models page).

    A read timeout gets its own message naming the elapsed time and the configured limit:
    the fix for "the model took longer than we allowed" is a larger
    ``RAG_LLM_TIMEOUT_SECONDS`` (or a smaller ``max_tokens``), not a URL check."""
    kind = _failure_kind(exc)
    if kind == "http_status":
        status = cast(httpx.HTTPStatusError, exc).response.status_code
        hint = {
            404: "the endpoint URL or model name is wrong",
            401: "the API key is missing or invalid",
            403: "the endpoint rejected the key",
        }.get(status, "the endpoint returned an error")
        msg = f"Model endpoint returned {status} — {hint}. Check this KB's model connection in Models."
    elif kind in {"read_timeout", "write_timeout", "timeout"}:
        took = f"{elapsed:.1f}s" if elapsed is not None else "too long"
        limit = f"{timeout:.0f}s" if timeout is not None else "the configured limit"
        msg = (
            f"The model endpoint did not finish responding within {limit} (gave up after {took}). "
            "The endpoint is reachable — the generation was too slow or too long for the timeout. "
            "Raise RAG_LLM_TIMEOUT_SECONDS or lower the model's max_tokens."
        )
    elif kind in {"connect_timeout", "pool_timeout"}:
        msg = (
            f"Could not open a connection to the model endpoint within the timeout ({kind}). "
            "The endpoint may be overloaded or the client's connection pool exhausted."
        )
    else:
        msg = f"Could not reach the model endpoint ({endpoint}). Check the URL and that it is reachable."
    _log.warning(
        "model_gateway_error",
        endpoint=endpoint,
        model=model,
        failure_kind=kind,
        timeout_s=timeout,
        elapsed_s=round(elapsed, 2) if elapsed is not None else None,
        exc_type=type(exc).__name__,
        error=str(exc)[:200],
    )
    return ModelGatewayError(
        msg,
        details={
            "endpoint": endpoint,
            "model": model,
            "failure_kind": kind,
            "timeout_s": timeout,
            "elapsed_s": round(elapsed, 2) if elapsed is not None else None,
            "exc_type": type(exc).__name__,
        },
    )


def _estimate_usage(prompt: str, completion: str) -> dict[str, int | bool | str]:
    """Rough usage when the server doesn't report it (~4 chars/token heuristic)."""
    p, c = max(1, len(prompt) // 4), max(1, len(completion) // 4)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c, "estimated": True}


class _StreamReasoningFilter:
    """Incremental filter that drops reasoning spans from a token stream.

    A reasoning model may interleave ``<think>…</think>`` (or ``<thinking>`` / ``<reasoning>``)
    working with the answer. We must not stream tokens that fall INSIDE such a span. The filter
    is a small state machine fed one delta at a time; it buffers while inside an open reasoning
    tag and only emits text that is outside every reasoning block.

    Robustness to chunk boundaries: an open/close tag may be split across two deltas (``"<th"`` +
    ``"ink>"``). So the filter never emits its trailing few characters until it has seen enough to
    know they are not the start of a tag — a short tail (``_MAX_TAG_LEN``) is always held back and
    flushed at the end. This keeps a genuine ``<`` that begins a tag from leaking as answer text."""

    def __init__(self) -> None:
        self._buf = ""
        self._in_reasoning = False
        # Bare "thought" dump handling (Gemma-class, untagged): decided once from the stream's
        # opening. If the stream opens with "thought" we cannot strip incrementally (the answer is
        # only known at the end), so we HOLD the whole stream and emit the stripped answer at flush.
        self._decided = False
        self._bare_dump = False

    def _decide(self) -> None:
        # Wait for enough opening bytes to recognise a bare "thought" preamble, then lock the mode.
        if self._decided or len(self._buf) < 16:  # noqa: PLR2004  # min opening bytes to sniff a preamble
            return
        self._decided = True
        self._bare_dump = bool(_BARE_THOUGHT_RE.match(self._buf))

    def feed(self, delta: str) -> str:
        """Consume a content delta, return the answer text safe to emit now (possibly empty)."""
        self._buf += delta
        self._decide()
        if not self._decided or self._bare_dump:
            return ""  # still sniffing the opening, or holding a whole reasoning dump for flush
        return self._drain(final=False)

    def flush(self) -> str:
        """Emit whatever remains once the stream ends. A held reasoning dump is stripped to its
        final answer; text still inside an unclosed <think> block is discarded (budget ran out
        mid-thought → nothing was written)."""
        if not self._decided:  # a very short stream that never reached the sniff threshold
            self._decided = True
            self._bare_dump = bool(_BARE_THOUGHT_RE.match(self._buf))
        if self._bare_dump:
            out = _strip_bare_reasoning(self._buf)
            self._buf = ""
            return out
        if self._in_reasoning:
            self._buf = ""
            return ""
        out, self._buf = self._buf, ""
        return out

    def _drain(self, *, final: bool) -> str:
        out = ""
        while self._buf:
            if self._in_reasoning:
                close = self._next_close(self._buf)
                if close is None:
                    # No close yet. Might be a close tag split across chunks — keep a tail, drop
                    # the rest (it is reasoning). Nothing emitted while inside a block.
                    self._buf = self._buf[-_MAX_TAG_LEN:] if not final else ""
                    return out
                self._buf = self._buf[close:]
                self._in_reasoning = False
                continue
            open_at = self._next_open(self._buf)
            if open_at is None:
                # No reasoning tag opens in the buffer. Emit everything except a held-back tail
                # that could be the start of a tag straddling the next chunk.
                if final:
                    out += self._buf
                    self._buf = ""
                    return out
                lt = self._buf.rfind("<")
                if lt == -1 or len(self._buf) - lt > _MAX_TAG_LEN:
                    # No pending "<", or it is already too long to be a tag opener — safe to emit
                    # all but a minimal tail.
                    keep = min(_MAX_TAG_LEN, len(self._buf))
                    out += self._buf[: len(self._buf) - keep]
                    self._buf = self._buf[len(self._buf) - keep :]
                else:
                    out += self._buf[:lt]
                    self._buf = self._buf[lt:]
                return out
            start, after_open = open_at
            out += self._buf[:start]
            self._buf = self._buf[after_open:]
            self._in_reasoning = True
        return out

    @staticmethod
    def _next_open(buf: str) -> tuple[int, int] | None:
        """(start_index, index_after_the_open_tag) of the earliest reasoning open tag, or None."""
        best: tuple[int, int] | None = None
        for tag in _REASONING_TAGS:
            m = re.search(rf"<{tag}\b[^>]*>", buf, re.IGNORECASE)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), m.end())
        return best

    @staticmethod
    def _next_close(buf: str) -> int | None:
        """Index just past the earliest reasoning close tag, or None if not present yet."""
        best: int | None = None
        for tag in _REASONING_TAGS:
            m = re.search(rf"</{tag}\s*>", buf, re.IGNORECASE)
            if m and (best is None or m.end() < best):
                best = m.end()
        return best


def _strip_reasoning(text: str) -> str:
    """Drop a reasoning model's chain-of-thought (``<think>…</think>``) — keep only the final
    answer, so citations, grounding and the faithfulness check see clean text. Qwen-thinking /
    QwQ / DeepSeek-R1 style.

    An UNCLOSED block means the token budget ran out mid-thought and the answer was never
    written. There is nothing to keep: returning the reasoning would publish the model's
    working as if it were the answer — which is how chain-of-thought reached the console's
    follow-up chips. The docstring used to claim this case was handled while the code returned
    the whole block.

    Empty is the honest result, and it is loud: the caller sees no answer instead of a
    plausible-looking one."""
    # Complete blocks (any of the reasoning-tag family) are removed wherever they sit — inline
    # working before/after the answer no longer leaks into the published text.
    cleaned = _REASONING_BLOCK_RE.sub("", text)
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _REASONING_BLOCK_RE.sub("", cleaned)
    # An opening reasoning tag surviving that pass had no close: the budget ran out mid-thought
    # and no answer was ever written. Keeping it would publish the model's working as the answer,
    # so the honest result is empty — and loud, via an actionable operator log.
    if _REASONING_OPEN_RE.search(cleaned):
        _log.warning(
            "reasoning_block_never_closed",
            chars=len(text),
            detail="the model was still thinking when its token budget ran out, so it never "
            "produced an answer — raise max_tokens for this call",
        )
        return ""
    return _strip_bare_reasoning(cleaned.strip())


# Some models (Gemma-class) emit a free-text reasoning DUMP as ordinary content: it opens with a
# bare "thought" line and a constraint / self-check checklist, and repeats the real answer at the
# very end — there is no tag for _strip_reasoning to key on. Detect that specific shape and keep
# only the final answer (the last citation-bearing prose sentence that is not itself a reasoning
# label). Conservative: it fires ONLY on the reasoning-dump signature, so an ordinary answer — even
# one that happens to mention "thoughts" — is returned untouched.
_BARE_THOUGHT_RE = re.compile(r"^\s*thought\b", re.IGNORECASE)
_REASONING_LABEL_RE = re.compile(
    r"^\s*[*\-]?\s*(user question|constraint\b|source \[|check\b|direct answer|format\b|output language|answer:)",
    re.IGNORECASE,
)


def _strip_bare_reasoning(text: str) -> str:
    if not _BARE_THOUGHT_RE.match(text):
        return text
    low = text.lower()
    if "constraint" not in low[:600] and "user question" not in low[:300]:
        return text  # opens with "thought" but not the reasoning-dump shape — leave it
    # Split on sentence boundaries, newlines, AND a period glued directly to a capital
    # ("English.The" — the model runs its last self-check into the final answer with no space).
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?<=[a-z])\.(?=[A-Z])", text)
    clean = [re.sub(r"^\s*[*\-]\s*", "", p).strip() for p in parts]
    keep = [i for i, p in enumerate(clean) if p and re.search(r"\[\d+\]", p) and not _REASONING_LABEL_RE.match(p)]
    if not keep:
        # Recognised as a reasoning dump, but no answer could be isolated. A CONVERSATIONAL reply
        # carries no citation by design, and a dump truncated mid-thought never reached its answer
        # at all — observed live on Gemma-31B, where "hi" rendered the system prompt's role and
        # all five constraints to the end user. Returning the text published the model's working
        # AND disclosed the prompt. Empty is the honest result, and it is the same rule the tagged
        # <think> path already follows: the caller's empty-generation guard then retries or
        # abstains instead of showing reasoning as an answer.
        _log.warning(
            "bare_reasoning_dump_without_answer",
            chars=len(text),
            detail="the model returned planning text with no isolatable answer; emitting nothing",
        )
        return ""
    last = keep[-1]
    answer = clean[last]
    # The platform's own citation style puts the marker after the sentence's full stop
    # ("... 99.5 percent. [1]"), which the sentence split leaves as a lone "[1]". Re-attach the
    # sentence it belongs to, or the "answer" would be a bare citation marker.
    if re.fullmatch(r"(?:\[\d+\][\s,]*)+", answer) and last > 0:
        prev = clean[last - 1]
        if prev and not _REASONING_LABEL_RE.match(prev):
            answer = f"{prev} {answer}".strip()
    _log.info("stripped_bare_reasoning_preamble", chars=len(text))
    return answer


def _answer_text(message: dict[str, object]) -> str:
    """The final answer from a chat message, robust to a REASONING model that returns
    ``content: null`` and puts its working in ``reasoning_content`` (Sarvam-105b, DeepSeek-R1).

    ``str(message["content"])`` on a null content yields the literal string ``"None"`` — a
    non-empty, plausible-looking 'answer' that then flows into grounding and is scored as an
    unfaithful claim (observed live: Sarvam intermittently abstained with a "None" answer). Treat
    a null/absent content as EMPTY so the caller's empty-generation guard retries or abstains
    honestly, instead of publishing 'None'. A separate ``reasoning_content`` is chain-of-thought,
    NOT the answer, so it is deliberately NOT substituted."""
    content = message.get("content")
    if content is None:
        _log.warning(
            "model_returned_null_content",
            detail="reasoning model produced no final answer (content=null) — treating as empty",
        )
        return ""
    return _strip_reasoning(str(content))


class OpenAICompatGateway:
    """Gateway to any self-hosted OpenAI-compatible LLM server (vLLM or llama.cpp)."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        *,
        api_key: str | None = None,
        api_key_header: str | None = None,
        skip_reasoning: bool = False,
        display_name: str | None = None,
    ) -> None:
        settings = get_settings()
        # ``vllm_url`` is the OpenAI-compatible base (e.g. http://vllm:8000/v1 or
        # http://llama:8080/v1). CPU inference is slow, hence the generous timeout —
        # operator-tunable via RAG_LLM_TIMEOUT_SECONDS, because it caps how large a
        # max_tokens budget can actually finish (see the settings field's note).
        # url/model/api_key may be passed explicitly for a per-tenant connection (FR-E1),
        # otherwise they fall back to the process-wide settings.
        self._url = (url or settings.vllm_url).rstrip("/")
        self._model = model or settings.llm_model
        # Sovereignty: the endpoint host must be egress-allowlisted (self-hosted OR a
        # sanctioned sovereign endpoint like the AIOps platform). The guard is never loosened here.
        assert_sovereign(self._url)
        # Optional API key for a managed endpoint (self-hosted vLLM/llama.cpp need none).
        # Injected via env/SecretsProvider or a per-tenant connection — never committed;
        # httpx keeps it out of logs.
        key = api_key if api_key is not None else settings.llm_api_key
        header = api_key_header or settings.llm_api_key_header
        # An "Authorization" header uses the Bearer scheme (OpenAI/Sarvam-compat); a bare token there
        # is rejected with 403. Auto-prefix so a raw key pasted into a connection just works. Other
        # header names (apikey, api-subscription-key, …) take the raw key unchanged.
        if key and header.lower() == "authorization" and not key.lower().startswith("bearer "):
            key = f"Bearer {key}"
        headers = {header: key} if key else {}
        self._max_tokens = settings.llm_max_tokens
        self._reasoning_effort = settings.llm_reasoning_effort
        # Some self-hosted reasoning models (e.g. Sarvam-30B) IGNORE reasoning_effort and always emit
        # a long <think> block — slow, and empty when the budget runs out mid-thought. Reasoning is now
        # suppressed at the SOURCE: gpu1's chat template primes a closed <think></think> in the
        # generation prompt (see the model dir's SKIPTHINK_PATCH_NOTE.txt), so no client-side prefill is
        # needed. This flag now only marks such a model so we nudge its sampling temperature (a
        # no-reasoning greedy decode LOOPS on tables — see _eff_temp).
        self._skip_reasoning = skip_reasoning
        # Friendly label for display/citation badges (the connection's name, e.g. "Sarvam-30B
        # (local GPU)"). API calls always use self._model; this is display-only. Falls back to the
        # raw model id when unset, so a bare deployment id like "/model" never reaches the badge.
        self._display_name = display_name
        self._timeout = timeout if timeout is not None else settings.llm_timeout_seconds
        self._client = httpx.AsyncClient(
            base_url=self._url, timeout=self._timeout, headers=headers, verify=model_tls_verify()
        )

    @property
    def url(self) -> str:
        """The endpoint this gateway calls. Non-secret: credentials travel in headers, never the
        URL — so a status surface can name the host a query actually goes to."""
        return self._url

    @property
    def model_name(self) -> str:
        return self._display_name or self._model

    def _reasoning_body(self) -> dict[str, object]:
        """Sarvam reasoning control: "off" -> reasoning_effort=null (thinking DISABLED); low/medium/
        high -> that depth; anything else -> omit (server default). Reasoning tokens count against
        max_tokens, so "off" stops a short-budget call (e.g. NL→SQL) from spending the whole budget
        inside <think> and returning an empty completion."""
        effort = self._reasoning_effort.lower()
        if effort in ("off", "none", "disabled", "false", "0"):
            # TWO switches so reasoning is actually off across servers: OpenAI-style reasoning_effort=null
            # AND the vLLM chat-template flag enable_thinking=false. Qwen3 / Sarvam-M reasoning models
            # IGNORE reasoning_effort and only obey enable_thinking — without it they spiral (measured
            # qwen-32b: 33s / <think> vs 17s / no-think). A server that doesn't know the kwarg ignores it.
            return {"reasoning_effort": None, "chat_template_kwargs": {"enable_thinking": False}}
        if effort in ("low", "medium", "high"):
            return {"reasoning_effort": effort}
        return {}

    def _chat_body(self, prompt: str) -> dict[str, object]:
        """Messages + reasoning controls: a single user turn + reasoning_effort. Skip-reasoning
        models (Sarvam-30B on gpu1) suppress their <think> pass via the SERVER chat template (it
        primes a closed <think></think> in the generation prompt), so NO client-side prefill is sent.
        A prefill here would be redundant and, on continue_final_message, yields a messier completion
        — the model answers, then spuriously re-opens <think> and rambles. One source of truth (the
        template) keeps the output clean for every client, not just this app."""
        messages: list[dict[str, str]] = []
        if self._reasoning_effort.lower() in ("off", "none", "disabled", "false", "0"):
            messages.append({"role": "system", "content": _NO_REASONING_SYSTEM})
        messages.append({"role": "user", "content": prompt})
        return {"messages": messages, **self._reasoning_body()}

    def _eff_temp(self, temperature: float, task: str = sampling.GENERATION) -> float:
        """A model decoding greedily with NO reasoning pass loops — it repeats a row, a phrase, or
        a single word forever, and it echoes the prompt's own examples back as if they were
        content. A small temperature breaks the loop without penalizing structural tokens
        (repetition_penalty kills a table's '|' delimiters). Only nudge when the caller asked for
        greedy, so an explicit temperature is still honoured.

        The trigger is the CONDITION, not one model id. It used to be `skip_reasoning`, which
        wiring sets only for the local "/model" deployment -- while RAG_LLM_REASONING_EFFORT=off
        suppresses reasoning for EVERY model. So gemma4-31b ran greedy-with-no-reasoning and was
        never nudged: observed live producing a page of its own deliberation, a repeated-word run,
        and this prompt's own "1,25,49,23,246" formatting example presented as an answer.

        A STRUCTURED-EXTRACTION TASK IS EXEMPT, because for that caller the nudge trades away the
        only property it wants. The loop it prevents was observed on GENERATION; whether a
        schema-constrained extraction prompt loops at 0.0 is a separate question this exemption
        makes measurable instead of unknowable. What the nudge COST there is already measured --
        see `rag_core.sampling`. Unclassified callers keep the nudge."""
        if sampling.honours_exact_temperature(task):
            return temperature
        if temperature == 0.0 and (self._skip_reasoning or self._reasoning_suppressed):
            return 0.6
        return temperature

    @property
    def _reasoning_suppressed(self) -> bool:
        """Whether this gateway tells the model not to reason (see :meth:`_reasoning_body`)."""
        return self._reasoning_effort.lower() in ("off", "none", "disabled", "false", "0")

    #: This adapter applies `rag_core.sampling`'s per-task policy, so `complete_for_task`
    #: may pass `task=` to it. A gateway WITHOUT this attribute is called exactly as before, which
    #: is why adding the policy needed no change to the `LLMGateway` Protocol and none to any of
    #: its other implementations or stubs.
    honours_sampling_task = True

    async def complete(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0, task: str = sampling.GENERATION
    ) -> str:
        """Single-turn chat completion. The grounding prompt is sent as the user turn."""
        _t0 = time.perf_counter()
        try:
            resp = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    **self._chat_body(prompt),
                    "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
                    "temperature": self._eff_temp(temperature, task),
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise _raise_clean(
                exc,
                self._url,
                self._model,
                timeout=self._timeout,
                elapsed=time.perf_counter() - _t0,
            ) from exc
        return _answer_text(resp.json()["choices"][0]["message"])

    async def complete_with_usage(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        task: str = sampling.GENERATION,
    ) -> tuple[str, dict[str, int | bool | str]]:
        """Complete + the server-reported token usage (estimated only if absent).

        The usage dict also carries ``finish_reason`` when the server reports it — ``"length"`` is
        the authoritative "cut off by the token budget" signal the completeness guard keys on, so it
        is threaded out here rather than discarded. A server that omits it leaves the key absent
        (the guard then falls back to its structural truncation heuristic)."""
        _t0 = time.perf_counter()
        try:
            resp = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    **self._chat_body(prompt),
                    "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
                    "temperature": self._eff_temp(temperature, task),
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise _raise_clean(
                exc,
                self._url,
                self._model,
                timeout=self._timeout,
                elapsed=time.perf_counter() - _t0,
            ) from exc
        data = resp.json()
        choice = data["choices"][0]
        text = _answer_text(choice["message"])
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}
        if usage.get("total_tokens"):
            metered: dict[str, int | bool | str] = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "estimated": False,
            }
        else:
            metered = _estimate_usage(prompt, text)
        if isinstance(finish_reason, str) and finish_reason:
            metered["finish_reason"] = finish_reason
        return text, metered

    async def stream(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token (reasoning <think> hidden until it closes)."""
        payload = {
            "model": self._model,
            **self._chat_body(prompt),
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "temperature": self._eff_temp(temperature),
            "stream": True,
        }
        reasoning = _StreamReasoningFilter()
        _t0 = time.perf_counter()
        try:
            stream_cm = self._client.stream("POST", "/chat/completions", json=payload)
            resp = await stream_cm.__aenter__()
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise _raise_clean(
                exc,
                self._url,
                self._model,
                timeout=self._timeout,
                elapsed=time.perf_counter() - _t0,
            ) from exc
        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    choices = json.loads(data).get("choices") or [{}]
                except json.JSONDecodeError:
                    continue
                # ``reasoning_content`` is a SEPARATE chain-of-thought field; only ``content`` is
                # the answer, and it is filtered for any INLINE reasoning tags before emitting.
                delta = choices[0].get("delta", {}).get("content")
                if not delta:
                    continue
                out = reasoning.feed(delta)
                if out:
                    yield out
            tail = reasoning.flush()
            if tail:
                yield tail
        finally:
            await stream_cm.__aexit__(None, None, None)

    async def list_models(self) -> list[str]:
        """List the deployed models the endpoint exposes (OpenAI ``/models``)."""
        _t0 = time.perf_counter()
        try:
            resp = await self._client.get("/models")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise _raise_clean(
                exc,
                self._url,
                self._model,
                timeout=self._timeout,
                elapsed=time.perf_counter() - _t0,
            ) from exc
        data = resp.json().get("data", [])
        return [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]

    async def aclose(self) -> None:
        await self._client.aclose()


# Backward-compatible alias (Phase-0 referenced VLLMGateway; behaviour now chat-based).
VLLMGateway = OpenAICompatGateway


_SOURCE_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)


class ExtractiveDevGateway:
    """Zero-GPU/zero-model fallback. Extractive, deterministic, citation-preserving.

    Returns the top source snippet from the grounding prompt tagged with its [n] marker,
    proving the end-to-end citation path without a model. NOT for production (guarded in
    :func:`build_llm_gateway`).
    """

    async def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        blocks = _split_sources(prompt)
        if not blocks:
            return "I don't know based on the provided sources."
        top = blocks[0][1].strip().replace("\n", " ")
        return f"{top[:400]} [1]"

    @property
    def model_name(self) -> str:
        return "extractive_dev"

    async def complete_with_usage(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> tuple[str, dict[str, int | bool | str]]:
        text = await self.complete(prompt)
        return text, _estimate_usage(prompt, text)

    async def stream(
        self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        yield await self.complete(prompt)

    async def list_models(self) -> list[str]:
        return []


# The source block's leading "(doc=..., page=...)" attribution line — prompt metadata, not document.
_ATTRIBUTION_LINE = re.compile(r"^[ \t]*\([^)\n]*\)[ \t]*\n")


def _split_sources(prompt: str) -> list[tuple[int, str]]:
    """Parse the numbered SOURCES section of a grounding prompt into (n, text).

    Source text is FENCED (P32), so the markers are stripped here: this parser stands in for a
    model reading the sources, and a model is told everything between them is the document. Left
    in, the delimiters ended up quoted verbatim in the extractive answer, which then read as
    unsupported text to the faithfulness judge."""
    if "SOURCES:" not in prompt:
        return []
    body = prompt.split("SOURCES:", 1)[1].split("QUESTION:", 1)[0]
    matches = list(_SOURCE_RE.finditer(body))
    out: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.end()
        # Bound each block at ITS OWN closing fence, not the next [n] marker or end-of-body --
        # anything build_prompt() inserts after the last source's fence (e.g. a language
        # reminder placed for recency right before QUESTION:) must never leak into what an
        # extractive/dev-fallback reader treats as source text.
        close_idx = body.find(SOURCE_CLOSE, start)
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        end = close_idx if close_idx != -1 and close_idx < next_start else next_start
        block = body[start:end].replace(SOURCE_OPEN, "").replace(SOURCE_CLOSE, "")
        # Drop the attribution line "(doc=..., section=..., page=...)" for the same reason the
        # fences are dropped above: it is the PROMPT's formatting, not the document's text. Echoed
        # into the extractive answer it reads as words no span contains, and the lexical anchor
        # scores it down -- a grounded answer flagged for a header the twin invented.
        out.append((n, _ATTRIBUTION_LINE.sub("", block, count=1)))
    return out


def build_llm_gateway() -> OpenAICompatGateway | ExtractiveDevGateway:
    """Factory selecting the LLM backend from configuration (fail-closed in prod).

    ``vllm`` and ``llamacpp`` both map to :class:`OpenAICompatGateway` (same wire API);
    they differ only in the configured ``RAG_VLLM_URL`` and served model.
    """
    settings = get_settings()
    backend = settings.llm_backend
    if backend in {"vllm", "llamacpp", "openai_compat"}:
        return OpenAICompatGateway()
    if backend == "extractive_dev":
        if settings.env == "prod":
            raise ConfigError("extractive_dev LLM backend is forbidden in prod")
        _log.warning("using_extractive_dev_llm", note="zero-model dev fallback; not for prod")
        return ExtractiveDevGateway()
    raise ConfigError(f"unknown llm backend: {backend}")
