"""Tests for cook_node: a forced cook reported per node, never a bare success."""

from __future__ import annotations

# Built-in
import os
import sys
from unittest.mock import MagicMock

# Third-party
import pytest

sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

from fxhoudinimcp_server.handlers import cook_handlers as ch  # noqa: E402


def _node(path, errors=(), fails=None, points=8, prims=6):
    node = MagicMock()
    node.path.return_value = path
    node.type.return_value.name.return_value = "python"
    node.errors.return_value = tuple(errors)
    node.warnings.return_value = ()
    if fails:
        node.cook.side_effect = RuntimeError(fails)
    geo = MagicMock()
    geo.intrinsicValue.side_effect = lambda name: {"pointcount": points, "primitivecount": prims}[name]
    node.geometry.return_value = geo
    return node


@pytest.fixture
def scene(monkeypatch):
    nodes = {}
    monkeypatch.setattr(ch.hou, "node", lambda path: nodes.get(path))
    monkeypatch.setattr(ch, "update_mode_warning", lambda: None)
    return nodes


class TestCookNode:
    def test_a_forced_cook_reports_what_it_produced(self, scene):
        scene["/obj/geo1/gen"] = _node("/obj/geo1/gen")
        result = ch.cook_node(["/obj/geo1/gen"])
        assert result["success"] is True
        entry = result["nodes"][0]
        assert entry["cooked"] is True and entry["points"] == 8 and entry["prims"] == 6
        scene["/obj/geo1/gen"].cook.assert_called_once_with(force=True)

    def test_a_missing_node_and_a_failed_cook_are_failures(self, scene):
        scene["/obj/geo1/bad"] = _node("/obj/geo1/bad", fails="Python error: boom")
        result = ch.cook_node(["/obj/geo1/nope", "/obj/geo1/bad"])
        assert result["success"] is False and result["failed"] == 2
        assert result["nodes"][0]["error"] == "node not found"
        assert "boom" in result["nodes"][1]["cook_error"]

    def test_node_errors_after_the_cook_fail_it(self, scene):
        scene["/obj/geo1/err"] = _node("/obj/geo1/err", errors=["Invalid source"])
        result = ch.cook_node("/obj/geo1/err")
        assert result["success"] is False
        assert result["nodes"][0]["errors"] == ["Invalid source"]

    def test_manual_update_mode_makes_it_unverified(self, scene, monkeypatch):
        scene["/obj/geo1/gen"] = _node("/obj/geo1/gen")
        monkeypatch.setattr(ch, "update_mode_warning", lambda: "Update mode is Manual")
        result = ch.cook_node(["/obj/geo1/gen"])
        assert result["success"] is False and "Manual" in result["warning"]

    def test_bad_arguments_are_named(self, scene):
        with pytest.raises(ValueError, match="non-empty"):
            ch.cook_node([])
        with pytest.raises(ValueError, match="force"):
            ch.cook_node(["/obj/geo1/gen"], force="yes")


@pytest.mark.asyncio
async def test_the_tool_passes_paths_and_force(mock_ctx, mock_bridge):
    from fxhoudinimcp.tools.cook import cook_node

    await cook_node(mock_ctx, ["/obj/geo1/gen"], force=False)
    command, params = mock_bridge.execute.call_args.args
    assert command == "cook.cook_node"
    assert params == {"node_paths": ["/obj/geo1/gen"], "force": False}
