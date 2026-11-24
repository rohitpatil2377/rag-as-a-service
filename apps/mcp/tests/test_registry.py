"""Unit tests for the MCP tool-plugin framework (registry + plugin loading)."""

from __future__ import annotations

import sys
import types

import pytest

from rag_mcp.registry import ToolRegistry, plugin_paths


async def _noop() -> dict[str, str]:
    return {}


def test_register_and_specs_preserve_order() -> None:
    reg = ToolRegistry()
    reg.register("a", "first", _noop)
    reg.register("b", "second", _noop)
    assert reg.names() == ["a", "b"]
    assert [s.description for s in reg.specs()] == ["first", "second"]


def test_duplicate_name_fails_loudly() -> None:
    reg = ToolRegistry()
    reg.register("dup", "one", _noop)
    with pytest.raises(ValueError, match="duplicate MCP tool name"):
        reg.register("dup", "two", _noop)


def test_decorator_registers() -> None:
    reg = ToolRegistry()

    @reg.tool("deco", "via decorator")
    async def _tool() -> dict[str, str]:
        return {"ok": "1"}

    assert reg.names() == ["deco"]
    # Decorator returns the function unchanged.
    assert _tool.__name__ == "_tool"


def test_plugin_paths_strips_and_drops_blanks() -> None:
    assert plugin_paths("  a.b , , c.d ,") == ["a.b", "c.d"]
    assert plugin_paths("") == []


def test_load_plugins_invokes_register_entrypoint() -> None:
    reg = ToolRegistry()
    mod = types.ModuleType("fake_mcp_plugin_ok")

    def register(registry: ToolRegistry) -> None:
        registry.register("rag.extra", "from a plugin", _noop)

    mod.register = register  # type: ignore[attr-defined]
    sys.modules["fake_mcp_plugin_ok"] = mod
    try:
        loaded = reg.load_plugins(["fake_mcp_plugin_ok", "  "])
    finally:
        del sys.modules["fake_mcp_plugin_ok"]
    assert loaded == ["fake_mcp_plugin_ok"]
    assert "rag.extra" in reg.names()


def test_load_plugins_without_entrypoint_raises() -> None:
    reg = ToolRegistry()
    mod = types.ModuleType("fake_mcp_plugin_bad")  # no register attr
    sys.modules["fake_mcp_plugin_bad"] = mod
    try:
        with pytest.raises(ImportError, match="no register"):
            reg.load_plugins(["fake_mcp_plugin_bad"])
    finally:
        del sys.modules["fake_mcp_plugin_bad"]


def test_default_registry_has_core_then_governance() -> None:
    from rag_mcp.server import default_registry

    names = default_registry().names()
    # Core pipeline tools present.
    for core in ("rag.list_kbs", "rag.retrieve", "rag.query", "rag.federated_query"):
        assert core in names
    # Governance pack present and AFTER the core tools (deterministic order).
    for gov in ("rag.attestation", "rag.usage", "rag.list_grants"):
        assert gov in names
    assert names.index("rag.attestation") > names.index("rag.query")


def test_tool_catalog_matches_registry() -> None:
    from rag_mcp.server import default_registry, tool_catalog

    catalog = tool_catalog()
    assert [c["name"] for c in catalog] == default_registry().names()
    assert all(c["description"] for c in catalog)
