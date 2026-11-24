"""MCP tool-plugin framework — a registry seam so tools are contributed, not hardcoded.

The server used to carry a fixed ``_TOOLS`` tuple. This registry lets tools be added
from three places without touching ``server.py``:

* **core** — the built-in pipeline tools (``server.register_core_tools``);
* **built-in packs** — e.g. the governance pack (``tools_governance.register``);
* **operator/third-party plugins** — modules named in ``RAG_MCP_PLUGINS`` (comma-
  separated import paths), each exposing a module-level ``register(registry)``.

Sovereign + safe by default: ``RAG_MCP_PLUGINS`` is empty out of the box, so the tool
surface is exactly the built-ins — no dynamic code is imported unless an operator opts in.
Every tool, whatever its origin, is invoked through the SAME auth/tenancy context
(``rag_mcp.context``), so a plugin tool cannot escape grounding, tenancy, or audit.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from rag_common import get_logger

_log = get_logger(__name__)

ToolFn = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    """One registered MCP tool: its wire ``name``, human ``description``, and callable."""

    name: str
    description: str
    fn: ToolFn


class ToolRegistry:
    """An ordered, name-unique collection of MCP tools.

    Registration order is preserved so the discovery UI and the server list tools
    deterministically (core first, then packs, then operator plugins)."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, fn: ToolFn) -> None:
        """Add a tool. Duplicate names fail loudly — two tools answering to the same
        name would make dispatch ambiguous, so a collision is a configuration error."""
        if name in self._tools:
            raise ValueError(f"duplicate MCP tool name: {name!r}")
        self._tools[name] = ToolSpec(name=name, description=description, fn=fn)

    def tool(self, name: str, description: str) -> Callable[[ToolFn], ToolFn]:
        """Decorator form of :meth:`register`, returning the function unchanged."""

        def decorator(fn: ToolFn) -> ToolFn:
            self.register(name, description, fn)
            return fn

        return decorator

    def specs(self) -> list[ToolSpec]:
        """Every registered tool, in registration order."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Registered tool names, in registration order."""
        return list(self._tools)

    def load_plugins(self, module_paths: Iterable[str]) -> list[str]:
        """Import each module path and call its ``register(registry)`` entrypoint.

        Blank entries are skipped (so a trailing comma in the env var is harmless). A
        module that is missing or lacks ``register`` raises ``ImportError`` — an operator
        who opted a plugin in should hear about a broken one, not get a silently smaller
        tool surface. Returns the paths successfully loaded."""
        loaded: list[str] = []
        for raw in module_paths:
            path = raw.strip()
            if not path:
                continue
            module = importlib.import_module(path)
            register = getattr(module, "register", None)
            if register is None or not callable(register):
                raise ImportError(f"MCP plugin {path!r} has no register(registry) entrypoint")
            register(self)
            loaded.append(path)
            _log.info("mcp_plugin_loaded", plugin=path)
        return loaded


def plugin_paths(setting: str) -> list[str]:
    """Split an ``RAG_MCP_PLUGINS`` value into clean, non-empty import paths."""
    return [p.strip() for p in setting.split(",") if p.strip()]
