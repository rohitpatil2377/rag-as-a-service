"""Track B3 lock — the agent has no exfiltration channel.

Injection assumes the model can be told to "send the context somewhere". Sealed egress
(NetworkPolicy) removes the route; this test removes the TOOL — the agent's tool catalog must
stay a fixed set of in-process, read/grounded operations with no network or shell primitive.
A future PR that adds a web-fetch / URL / code-exec tool fails here on purpose."""

from __future__ import annotations

import inspect

_ALLOWED_TOOLS = {"list_kbs", "retrieve", "answer", "sql_query", "ingest_document"}

# Primitives that could open an exfil channel from inside a tool. Adding any of these to a tool
# must be a deliberate, reviewed change — not something that slips in unnoticed.
_EXFIL_PRIMITIVES = (
    "requests.",
    "httpx.",
    "aiohttp",
    "urllib",
    "socket.",
    "http.client",
    "subprocess",
    "os.system",
    "os.popen",
    "pty.",
    "smtplib",
    "ftplib",
)


def test_agent_tool_catalog_is_the_safe_fixed_set() -> None:
    from svc_control_plane.agent_orchestrator import default_tools

    assert set(default_tools()) <= _ALLOWED_TOOLS, (
        "a new agent tool appeared — review it for an exfil channel and add it to _ALLOWED_TOOLS"
    )


def test_no_agent_tool_has_an_exfil_primitive() -> None:
    from svc_control_plane.agent_orchestrator import default_tools

    for name, spec in default_tools().items():
        src = inspect.getsource(spec.fn)
        for bad in _EXFIL_PRIMITIVES:
            assert bad not in src, f"agent tool '{name}' contains a potential exfil primitive: {bad!r}"
