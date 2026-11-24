"""SIEM audit forwarding (banking pack) — ship the tamper-evident audit chain to the bank's SOC.

Banks aggregate security events in a SIEM (Splunk / IBM QRadar / ArcSight). This forwards every
sealed audit event to that SIEM in near-real-time, so the platform's activity shows up alongside
the rest of the estate for correlation and alerting. Two transports: an HTTP webhook (e.g. Splunk
HEC) and RFC-5424 syslog (UDP/TCP). Two formats: JSON and CEF (Common Event Format, what most
SIEMs parse natively).

Design guarantees:
* **Never blocks or breaks auditing** — forwarding is fire-and-forget (a background task) and every
  error is swallowed + logged; a SIEM outage can never fail a request or drop the local audit row.
* **Sovereign** — the SIEM endpoint is put through ``assert_sovereign`` (operator-allowlisted host),
  so this cannot become a covert exfiltration channel. Off by default (``siem_enabled``).
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from rag_common.logging import get_logger
from rag_common.sovereignty import assert_sovereign

_log = get_logger(__name__)

Sender = Callable[[str], Awaitable[None]]

# CEF severity per audit action — most are informational; destructive/security events rank higher.
_CEF_SEVERITY = {
    "erasure": 8,
    "delete": 7,
    "approval": 6,
    "key_rotation": 6,
    "egress": 6,
    "guardrail": 5,
    "faithfulness": 4,
}
_CEF_ESCAPE = str.maketrans({"\\": "\\\\", "|": "\\|", "\n": " "})


class SiemForwarder:
    """Forwards sealed audit events to a SIEM. Construct via :func:`build_siem_forwarder`."""

    def __init__(
        self, *, endpoint: str, transport: str, fmt: str, token: str = "", sender: Sender | None = None
    ) -> None:
        self._endpoint = endpoint
        self._transport = transport  # "webhook" | "syslog"
        self._fmt = fmt  # "json" | "cef"
        self._token = token
        self._send: Sender = sender or self._default_send
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, event: dict[str, Any], event_hash: str) -> None:
        """Fire-and-forget forward — non-blocking; safe to call from the audit hot path."""
        try:
            task = asyncio.create_task(self._forward(event, event_hash))
        except RuntimeError:  # no running loop (e.g. sync context) — skip rather than raise
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _forward(self, event: dict[str, Any], event_hash: str) -> None:
        try:
            assert_sovereign(self._endpoint)  # SIEM host must be operator-allowlisted (sovereign)
            line = self._format(event, event_hash)
            await self._send(line)
        except Exception as exc:
            _log.warning("siem_forward_failed", transport=self._transport, error=str(exc)[:180])

    def _format(self, event: dict[str, Any], event_hash: str) -> str:
        if self._fmt == "cef":
            return self._to_cef(event, event_hash)
        return json.dumps({**event, "event_hash": event_hash, "source": "ragaas"}, default=str)

    def _to_cef(self, event: dict[str, Any], event_hash: str) -> str:
        action = str(event.get("action", "event"))
        sev = _CEF_SEVERITY.get(action, 3)
        ext = (
            f"act={action} suser={_esc(event.get('actor'))} "
            f"cs1Label=tenant cs1={_esc(event.get('tenant_id'))} "
            f"cs2Label=trace cs2={_esc(event.get('trace_id'))} "
            f"cs3Label=eventHash cs3={event_hash[:16]}"
        )
        return f"CEF:0|Acme|RagaaS|1.0|{_esc(action)}|{_esc(action)}|{sev}|{ext}"

    async def _default_send(self, line: str) -> None:
        if self._transport == "syslog":
            await self._send_syslog(line)
        else:
            await self._send_webhook(line)

    async def _send_webhook(self, line: str) -> None:
        import httpx  # noqa: PLC0415 — only needed when the webhook transport is active

        headers = {"Content-Type": "application/json"}
        if self._token:  # Splunk HEC style
            headers["Authorization"] = f"Splunk {self._token}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(self._endpoint, content=line, headers=headers)
            resp.raise_for_status()

    async def _send_syslog(self, line: str) -> None:
        parsed = urlsplit(self._endpoint if "://" in self._endpoint else f"//{self._endpoint}")
        host, port = parsed.hostname or "localhost", parsed.port or 514
        # RFC-5424 <134> = local0.info; keep it simple and non-blocking via a thread.
        msg = f"<134>1 - ragaas - - - - {line}".encode()

        def _emit() -> None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(msg, (host, port))

        await asyncio.to_thread(_emit)


def _esc(value: Any) -> str:
    return str(value if value is not None else "").translate(_CEF_ESCAPE)


def build_siem_forwarder(sender: Sender | None = None) -> SiemForwarder | None:
    """Construct the forwarder from config, or None when SIEM export is disabled (the default)."""
    from rag_common.config import get_settings  # noqa: PLC0415 — avoid import cycle at module load

    s = get_settings()
    if not s.siem_enabled or not s.siem_endpoint:
        return None
    return SiemForwarder(
        endpoint=s.siem_endpoint, transport=s.siem_transport, fmt=s.siem_format, token=s.siem_token, sender=sender
    )
