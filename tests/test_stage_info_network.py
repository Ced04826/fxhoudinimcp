"""get_stage_info refused a LOP network: "Node is not a LOP node (no stage())".

`/stage` is the network the viewport shows, and "what is it showing" had to
be assembled in execute_python from displayNode(). A network now answers
through its display node: `display_node`, `render_node`, `resolved_from`,
plus `frame` and the Scene Viewer's `viewport_delegate`. A network with
nothing to show still refuses, and says why.

hou and pxr are mocked here; the live check ran on Houdini 22.0.429.
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
import fxhoudinimcp_server.handlers.lops_handlers as lops  # noqa: E402


def _lop(path):
    node = MagicMock(spec=["path", "stage"])
    node.path.return_value = path
    return node


def _network(path, display=None, render=None):
    node = MagicMock(spec=["path", "displayNode", "renderNode"])
    node.path.return_value = path
    node.displayNode.return_value = display
    node.renderNode.return_value = render
    return node


@pytest.fixture
def refusals(monkeypatch):
    monkeypatch.setattr(lops.hou, "OperationFailed", RuntimeError)


class TestResolveStageNode:
    def test_a_network_answers_through_its_display_node(self, monkeypatch):
        display = _lop("/stage/karmarendersettings1")
        network = _network("/stage", display, _lop("/stage/usdrender_rop1"))
        monkeypatch.setattr(lops.hou, "node", lambda path: network)
        node, context = lops._resolve_stage_node("/stage")
        assert node is display
        assert context == {
            "requested_path": "/stage",
            "resolved_from": "display_node",
            "display_node": "/stage/karmarendersettings1",
            "render_node": "/stage/usdrender_rop1",
        }

    def test_a_lop_answers_for_itself(self, monkeypatch):
        lop = _lop("/stage/sopimport1")
        monkeypatch.setattr(lops.hou, "node", lambda path: lop)
        node, context = lops._resolve_stage_node("/stage/sopimport1")
        assert node is lop
        assert context == {}

    def test_a_network_with_no_display_node_says_so(self, monkeypatch, refusals):
        monkeypatch.setattr(lops.hou, "node", lambda path: _network("/stage"))
        with pytest.raises(RuntimeError, match="no display node"):
            lops._resolve_stage_node("/stage")

    def test_a_display_node_that_is_not_a_lop_is_named(self, monkeypatch, refusals):
        geo = MagicMock(spec=["path"])
        geo.path.return_value = "/obj/geo1"
        monkeypatch.setattr(lops.hou, "node", lambda path: _network("/obj", geo))
        with pytest.raises(RuntimeError, match="/obj/geo1 is not a LOP"):
            lops._resolve_stage_node("/obj")

    def test_a_missing_node_is_still_not_found(self, monkeypatch, refusals):
        monkeypatch.setattr(lops.hou, "node", lambda path: None)
        with pytest.raises(RuntimeError, match="Node not found"):
            lops._resolve_stage_node("/nope")


class TestGetStageInfo:
    def test_the_reply_names_what_the_network_shows(self, monkeypatch):
        display = _lop("/stage/OUT")
        network = _network("/stage", display)
        stage = MagicMock()
        stage.Traverse.return_value = [object(), object()]
        stage.GetUsedLayers.return_value = []
        stage.GetDefaultPrim.return_value = None
        seen = []
        monkeypatch.setattr(lops.hou, "node", lambda path: network)
        monkeypatch.setattr(lops.hou, "frame", lambda: 12.0)
        monkeypatch.setattr(lops, "_get_lop_stage", lambda path: seen.append(path) or stage)
        monkeypatch.setattr(lops, "_viewport_delegate", lambda: "Karma CPU")
        monkeypatch.setattr(lops, "UsdGeom", MagicMock(), raising=False)
        monkeypatch.setattr(lops.UsdGeom, "GetStageMetersPerUnit", lambda s: 1.0, raising=False)

        reply = lops._get_stage_info(node_path="/stage")

        assert seen == ["/stage/OUT"]
        assert reply["node_path"] == "/stage/OUT"
        assert reply["display_node"] == "/stage/OUT"
        assert reply["resolved_from"] == "display_node"
        assert reply["prim_count"] == 2
        assert reply["frame"] == 12.0
        assert reply["viewport_delegate"] == "Karma CPU"

    def test_the_default_is_the_stage_network(self, monkeypatch, refusals):
        asked = []
        monkeypatch.setattr(lops.hou, "node", lambda path: asked.append(path))
        with pytest.raises(RuntimeError):
            lops._get_stage_info()
        assert asked == ["/stage"]


class TestViewportDelegate:
    def test_the_scene_viewer_names_its_hydra_delegate(self, monkeypatch):
        network_editor = MagicMock()
        network_editor.type.return_value = "NetworkEditor"
        viewer = MagicMock()
        viewer.type.return_value = "SceneViewer"
        viewer.currentHydraRenderer.return_value = "Houdini VK"
        monkeypatch.setattr(lops.hou.ui, "paneTabs", lambda: [network_editor, viewer])
        monkeypatch.setattr(lops.hou.paneTabType, "SceneViewer", "SceneViewer")
        assert lops._viewport_delegate() == "Houdini VK"

    def test_no_ui_is_no_delegate_not_an_error(self, monkeypatch):
        def _no_ui():
            raise RuntimeError("hou.ui is not available")

        monkeypatch.setattr(lops.hou.ui, "paneTabs", _no_ui)
        assert lops._viewport_delegate() is None
