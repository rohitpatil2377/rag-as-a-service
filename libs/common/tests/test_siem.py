"""SIEM audit forwarding (banking pack): formats, egress gate, fire-and-forget safety."""

from __future__ import annotations

import pytest

from rag_common import get_settings
from rag_common.siem import SiemForwarder, build_siem_forwarder

_EVENT = {
    "action": "erasure",
    "actor": "officer@bank",
    "tenant_id": "bank",
    "trace_id": "t-123",
    "payload": {"doc": "x"},
}


def test_json_format() -> None:
    fwd = SiemForwarder(endpoint="http://localhost:8088", transport="webhook", fmt="json")
    line = fwd._format(_EVENT, "abcdef1234")
    assert '"action": "erasure"' in line and '"event_hash": "abcdef1234"' in line
    assert '"source": "ragaas"' in line


def test_cef_format_escapes_and_severity() -> None:
    fwd = SiemForwarder(endpoint="http://localhost:8088", transport="syslog", fmt="cef")
    line = fwd._format({**_EVENT, "actor": "a|b"}, "hash1234567890ab")
    assert line.startswith("CEF:0|Acme|RagaaS|1.0|erasure|erasure|8|")  # erasure severity 8
    assert "suser=a\\|b" in line  # pipe escaped for CEF
    assert "cs3=hash1234567890" in line  # event-hash carried


async def test_forward_delivers_when_host_allowlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost")
    get_settings.cache_clear()
    sent: list[str] = []

    async def _capture(line: str) -> None:
        sent.append(line)

    fwd = SiemForwarder(endpoint="http://localhost:8088", transport="webhook", fmt="json", sender=_capture)
    await fwd._forward(_EVENT, "hash")
    assert len(sent) == 1 and "erasure" in sent[0]
    get_settings.cache_clear()


async def test_forward_blocked_for_non_allowlisted_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost")
    monkeypatch.setenv("RAG_ENV", "prod")  # sovereign_enforce on
    get_settings.cache_clear()
    sent: list[str] = []

    async def _capture(line: str) -> None:
        sent.append(line)

    fwd = SiemForwarder(endpoint="http://exfil.example.com/collect", transport="webhook", fmt="json", sender=_capture)
    await fwd._forward(_EVENT, "hash")  # sovereignty guard blocks; error swallowed, nothing sent
    assert sent == []
    get_settings.cache_clear()


def test_build_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_SIEM_ENABLED", raising=False)
    get_settings.cache_clear()
    assert build_siem_forwarder() is None

    monkeypatch.setenv("RAG_SIEM_ENABLED", "true")
    monkeypatch.setenv("RAG_SIEM_ENDPOINT", "http://localhost:8088/services/collector")
    get_settings.cache_clear()
    fwd = build_siem_forwarder()
    assert fwd is not None
    get_settings.cache_clear()
