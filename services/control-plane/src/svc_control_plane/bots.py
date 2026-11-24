"""Chat-platform bots (FR-I7) — thin adapters over the query pipeline.

Self-hosted **Mattermost** first (sovereign). The integration is a single inbound webhook:
Mattermost's *outgoing webhook* POSTs a channel message to ``/v1/bots/mattermost/{kb_id}``
and we answer **synchronously in the response body** — so the platform never makes an
outbound call, no egress-allowlist entry is needed, and the whole exchange stays on-prem.

Slack/Teams slash commands use the same request/response shape; because they are foreign
SaaS, exposing the endpoint to them is a deployment/network-policy decision (not this
module's egress). The answer is grounded-or-abstain with citations, public-ACL (least
privilege for a shared channel), just like the embeddable widget (FR-I6).
"""

from __future__ import annotations

import hashlib
import json
import secrets

from pydantic import BaseModel, Field

from rag_clients.factory import get_catalog
from rag_common import get_logger
from rag_schemas import GroundedAnswer
from rag_schemas.common import AnswerStatus

_log = get_logger(__name__)
_BOT_KEY = "bot_config"
_MAX_CITED = 5


class BotConfig(BaseModel):
    """A KB's chat-bot binding (the shared token is stored hashed, never returned)."""

    enabled: bool = False
    platform: str = Field(default="mattermost", pattern="^(mattermost|slack|teams)$")
    trigger: str = Field(default="", max_length=40, description="Optional trigger word to strip from the message.")


class BotConfigView(BaseModel):
    """What the console sees — config plus whether a token has been set (never the token)."""

    enabled: bool
    platform: str
    trigger: str
    configured: bool
    webhook_path: str


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def _raw_config(kb_id: str) -> dict[str, str]:
    raw = (await get_catalog().get_tenant_config(kb_id)).get(_BOT_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


async def get_bot_config(kb_id: str) -> BotConfigView:
    data = await _raw_config(kb_id)
    return BotConfigView(
        enabled=bool(data.get("enabled", False)),
        platform=str(data.get("platform", "mattermost")),
        trigger=str(data.get("trigger", "")),
        configured=bool(data.get("token_hash")),
        webhook_path=f"/v1/bots/mattermost/{kb_id}",
    )


async def _save(kb_id: str, data: dict[str, str]) -> None:
    await get_catalog().set_tenant_config(kb_id, _BOT_KEY, json.dumps(data))


async def set_bot_config(kb_id: str, config: BotConfig) -> BotConfigView:
    data = await _raw_config(kb_id)
    data.update({"enabled": config.enabled, "platform": config.platform, "trigger": config.trigger})  # type: ignore[dict-item]
    await _save(kb_id, data)
    return await get_bot_config(kb_id)


async def rotate_token(kb_id: str) -> str:
    """Generate + store (hashed) a new shared token, enable the bot, and return it ONCE."""
    token = secrets.token_urlsafe(24)
    data = await _raw_config(kb_id)
    data["token_hash"] = _hash(token)
    data["enabled"] = True  # type: ignore[assignment]
    data.setdefault("platform", "mattermost")
    await _save(kb_id, data)
    return token


async def verify_token(kb_id: str, presented: str) -> bool:
    """Constant-time check of the inbound webhook token against the stored hash."""
    data = await _raw_config(kb_id)
    stored = str(data.get("token_hash", ""))
    if not stored or not data.get("enabled") or not presented:
        return False
    return secrets.compare_digest(stored, _hash(presented))


def strip_trigger(text: str, trigger: str) -> str:
    """Drop a leading trigger word / bot mention so the model sees a clean question."""
    q = text.strip()
    if trigger and q.lower().startswith(trigger.lower()):
        q = q[len(trigger) :].strip()
    return q


def format_answer(answer: GroundedAnswer) -> str:
    """Render a grounded answer as Mattermost/Slack markdown (answer + cited sources)."""
    if answer.status is AnswerStatus.ABSTAINED:
        return f":warning: {answer.answer}"
    body = answer.answer
    docs: list[str] = []
    for c in answer.citations[:_MAX_CITED]:
        if c.doc_id not in docs:
            docs.append(c.doc_id)
    if docs:
        body += "\n\n_Sources: " + ", ".join(f"`{d}`" for d in docs) + "_"
    if answer.suggested_followups:
        body += "\n\n**You could also ask:** " + " · ".join(answer.suggested_followups[:3])
    return body


class MattermostInbound(BaseModel):
    """The subset of a Mattermost outgoing-webhook payload we use (extra fields ignored)."""

    model_config = {"extra": "ignore"}

    token: str = ""
    text: str = ""
    trigger_word: str = ""
    user_name: str = ""
    channel_name: str = ""


class MattermostResponse(BaseModel):
    """Mattermost posts this back into the channel (synchronous, no outbound call from us)."""

    response_type: str = "comment"
    text: str
