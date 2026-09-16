"""Tests for change_node_type and press_button.

Two Houdini gestures had no tool: swapping a node's type in place (Type
Properties "change type", or moving an asset instance to an installed newer
version without losing its edits) and pressing a button parameter ("Stash
Input", "Reload Geometry", an asset's Build button). Both were being done
through execute_python, where the swap silently dropped parameter values and
a pressed button reported nothing about the node afterwards.

hou is mocked here; the live path was checked on Houdini 22.0.429.
"""

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

# Internal
import fxhoudinimcp_server.handlers.node_handlers as nodes  # noqa: E402


def _node(path, type_name="xform"):
    node = MagicMock()
    node.path.return_value = path
    node.name.return_value = path.rsplit("/", 1)[-1]
    node.type.return_value.name.return_value = type_name
    return node


class TestChangeNodeType:
    def _setup(self, monkeypatch, old="xform", new="null"):
        node = _node("/obj/geo1/keepme", old)
        category = node.type.return_value.category.return_value
        category.name.return_value = "Sop"
        category.nodeTypes.return_value = {"xform": None, "null": None}
        kept = MagicMock()
        kept.name.return_value = "tx"
        kept.isAtDefault.return_value = False
        kept.isSpare.return_value = False
        node.parms.return_value = [kept]
        changed = _node("/obj/geo1/keepme", new)
        changed.parms.return_value = []
        changed.inputs.return_value = [_node("/obj/geo1/box1", "box")]
        changed.outputs.return_value = []
        changed.children.return_value = []
        changed.type.return_value.definition.return_value = None
        node.changeNodeType.return_value = changed
        resolved = MagicMock()
        resolved.name.return_value = new
        monkeypatch.setattr(nodes, "_get_node", lambda path: node)
        monkeypatch.setattr(nodes, "_resolve_node_type", lambda cat, name: resolved)
        monkeypatch.setattr(nodes, "_focus_network_editor", lambda *a, **k: None)
        return node, changed

    def test_the_swap_keeps_contents_by_default_and_names_dropped_values(self, monkeypatch):
        node, changed = self._setup(monkeypatch)
        result = nodes.change_node_type("/obj/geo1/keepme", "null")
        node.changeNodeType.assert_called_once_with(
            "null", keep_name=True, keep_parms=True, keep_network_contents=True
        )
        assert result["changed"] is True
        assert result["new_type"] == "null"
        assert result["parms_dropped"] == ["tx"]
        assert result["inputs"] == ["/obj/geo1/box1"]

    def test_the_same_type_is_a_noop(self, monkeypatch):
        node, _ = self._setup(monkeypatch, old="null", new="null")
        result = nodes.change_node_type("/obj/geo1/keepme", "null")
        assert result["changed"] is False
        node.changeNodeType.assert_not_called()

    def test_an_unknown_type_is_refused_with_a_hint(self, monkeypatch):
        self._setup(monkeypatch)
        monkeypatch.setattr(nodes, "_resolve_node_type", lambda *a, **k: None)
        with pytest.raises(ValueError, match="does not exist in Sop"):
            nodes.change_node_type("/obj/geo1/keepme", "nul")

    def test_hom_refusal_is_readable(self, monkeypatch):
        node, _ = self._setup(monkeypatch)
        node.changeNodeType.side_effect = RuntimeError("locked asset")
        with pytest.raises(ValueError, match="Could not change"):
            nodes.change_node_type("/obj/geo1/keepme", "null")


class TestResolveNodeType:
    def test_preferred_version_wins_then_exact_then_newest(self):
        hou = nodes.hou
        category = MagicMock()
        category.name.return_value = "Sop"
        category.nodeTypes.return_value = {"curve": "classic", "curve::2.0": "new"}
        hou.preferredNodeType.return_value = None
        assert nodes._resolve_node_type(category, "curve") == "classic"
        assert nodes._resolve_node_type(category, "curve::2.0") == "new"
        category.nodeTypes.return_value = {"copytopoints::2.0": "v2", "copytopoints::3.0": "v3"}
        assert nodes._resolve_node_type(category, "copytopoints") == "v3"
        assert nodes._resolve_node_type(category, "nope") is None
        hou.preferredNodeType.return_value = "preferred"
        assert nodes._resolve_node_type(category, "anything") == "preferred"


class TestPressButton:
    def _node_with_button(self, monkeypatch, kind="Button"):
        node = _node("/obj/geo1/stash1", "stash")
        parm = MagicMock()
        parm.name.return_value = "stashinput"
        parm.parmTemplate.return_value.type.return_value.name.return_value = kind
        parm.parmTemplate.return_value.scriptCallback.return_value = "..."
        node.parm.side_effect = lambda name: parm if name == "stashinput" else None
        node.parms.return_value = [parm]
        node.errors.return_value = []
        node.warnings.return_value = ["w"]
        monkeypatch.setattr(nodes, "_get_node", lambda path: node)
        return node, parm

    def test_the_button_is_pressed_and_the_node_state_read_back(self, monkeypatch):
        _, parm = self._node_with_button(monkeypatch)
        result = nodes.press_button("/obj/geo1/stash1", "stashinput")
        parm.pressButton.assert_called_once_with()
        assert result["warnings"] == ["w"]
        assert result["callback_present"] is True
        assert "duration_ms" in result
        assert "note" not in result

    def test_arguments_reach_the_callback(self, monkeypatch):
        _, parm = self._node_with_button(monkeypatch)
        nodes.press_button("/obj/geo1/stash1", "stashinput", arguments={"mode": 1})
        parm.pressButton.assert_called_once_with({"mode": 1})

    def test_a_non_button_is_pressed_but_noted(self, monkeypatch):
        self._node_with_button(monkeypatch, kind="Toggle")
        result = nodes.press_button("/obj/geo1/stash1", "stashinput")
        assert "not a Button" in result["note"]

    def test_a_missing_parm_lists_the_buttons(self, monkeypatch):
        self._node_with_button(monkeypatch)
        with pytest.raises(ValueError, match="Buttons on this node: \\['stashinput'\\]"):
            nodes.press_button("/obj/geo1/stash1", "stash_input")

    def test_a_failing_callback_is_readable(self, monkeypatch):
        _, parm = self._node_with_button(monkeypatch)
        parm.pressButton.side_effect = RuntimeError("script error")
        with pytest.raises(ValueError, match="Callback of"):
            nodes.press_button("/obj/geo1/stash1", "stashinput")
