"""Call a gateway for a stated TASK, so its sampling policy is the one the caller asked for.

ONE SEAM, NOT A GETATTR AT EVERY CALL SITE. ``rag_core.sampling`` defines what a task means
and ``OpenAICompatGateway`` applies it; this is how a stage says which task it is without every
stage re-deriving what the gateway in front of it can do.

WHY NOT A PARAMETER ON THE PROTOCOL. ``LLMGateway.complete`` is implemented by two adapters, the
in-memory twin and ~32 test stubs. Adding a keyword to the Protocol would require every one of
them to accept it to stay conformant, to buy a policy only one adapter can act on. So the
capability is advertised instead — an adapter sets ``honours_sampling_task = True`` — and anything
without it is called exactly as it is called today. Same discipline as ``complete_with_usage``,
which is a capability rather than a Protocol method for the same reason.

USAGE COMES BACK TOO, because every current caller of this wants both: the per-section token spend
`EnrichReport.completion_tokens` records, and ``finish_reason``, which is the only authoritative
"ran out of output budget" signal. ``{}`` from a gateway that cannot report it.
"""

from __future__ import annotations

from typing import Any

from rag_core import sampling
from rag_core.interfaces import LLMGateway


async def complete_for_task(
    gateway: LLMGateway,
    prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    task: str = sampling.GENERATION,
) -> tuple[str, dict[str, Any]]:
    """``(reply, usage)`` from ``gateway``, honouring ``task`` where the gateway supports it.

    Raises whatever the gateway raises — a sampling policy is not a fail-safe, and the stages that
    call this already have their own (the enricher scopes a failure to one section, the profiler to
    one document). Swallowing a gateway error here would hide it from both.
    """
    kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
    if getattr(gateway, "honours_sampling_task", False):
        kwargs["task"] = task
    with_usage = getattr(gateway, "complete_with_usage", None)
    if with_usage is not None:
        reply, metered = await with_usage(prompt, **kwargs)
        return reply, dict(metered)
    return await gateway.complete(prompt, **kwargs), {}
