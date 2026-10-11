"""capture_network_editor frames at once, and says what it framed.

``homeToSelection()`` animates the editor's view over the next event-loop
ticks, while the capture runs in the same tick, so the image came out
mid-flight. The view is set with ``setVisibleBounds(transition_time=0)`` --
with ``set_center_when_scale_rejected``, or a rect at the zoom the editor
already has moves nothing -- and the reply carries the rect the image shows,
so the frame can be checked without opening it.

The fork frames a node with its direct inputs and outputs, a whole network,
or given bounds, grown to the pane's aspect, and puts the pane back after.
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
import fxhoudinimcp_server.handlers.viewport_handlers as viewport  # noqa: E402


class _Node:
    """A network item with a place: what _items_rect reads."""

    def __init__(self, path, x, y, w=1.0, h=0.3, parent=None):
        self._path, self._pos, self._size, self._parent = path, (x, y), (w, h), parent

    def path(self):
        return self._path

    def name(self):
        return self._path.rsplit("/", 1)[-1]

    def position(self):
        return self._pos

    def size(self):
        return self._size

    def parent(self):
        return self._parent

    def inputs(self):
        return ()

    def outputs(self):
        return ()


def _rect(xmin, ymin, xmax, ymax):
    rect = MagicMock()
    rect.min.return_value = (xmin, ymin)
    rect.max.return_value = (xmax, ymax)
    return rect


@pytest.fixture
def stage():
    net = MagicMock()
    net.path.return_value = "/stage"
    return net


@pytest.fixture
def editor(monkeypatch, stage, tmp_path):
    """A 1600x900 Network Editor pane in /stage; hou.BoundingRect hands back its numbers."""
    pane = MagicMock()
    pane.type.return_value = "NetworkEditor"
    pane.qtScreenGeometry.return_value.width.return_value = 1600
    pane.qtScreenGeometry.return_value.height.return_value = 900
    pane.pwd.return_value = stage
    pane.visibleBounds.return_value = _rect(20.0, -70.0, 40.0, -50.0)
    monkeypatch.setattr(viewport.hou.ui, "paneTabs", lambda: [pane])
    monkeypatch.setattr(viewport.hou.paneTabType, "NetworkEditor", "NetworkEditor")
    monkeypatch.setattr(viewport.hou, "BoundingRect", lambda *box: box)
    monkeypatch.setattr(viewport.hou, "Node", _Node)
    monkeypatch.setattr(viewport.hou, "SubnetIndirectInput", type("Indirect", (), {}))
    monkeypatch.setattr(viewport, "require_ui", lambda *a, **k: None)
    monkeypatch.setattr(viewport, "_let_qt_repaint", lambda: None)
    monkeypatch.setattr(viewport, "_image_spread", lambda path: {"pixels": [8, 8]})

    def grab(pane_tab, path):
        with open(path, "wb") as handle:
            handle.write(b"png")

    monkeypatch.setattr(viewport, "_capture_pane_tab_qt", grab)
    return pane


def _use_nodes(monkeypatch, stage, places):
    nodes = {path: _Node(path, *xy, parent=stage) for path, xy in places.items()}
    stage.children.return_value = list(nodes.values())
    monkeypatch.setattr(viewport.hou, "node", lambda path: nodes.get(path))
    return nodes


def _centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


class TestNetworkEditorIsFramedAtOnce:
    def test_the_view_is_set_without_an_animated_home(self, editor, stage, monkeypatch, tmp_path):
        _use_nodes(monkeypatch, stage, {"/stage/TARGET": (30.0, -60.0)})

        viewport.capture_network_editor(str(tmp_path / "ne.png"), "/stage/TARGET")

        editor.homeToSelection.assert_not_called()
        (box,), kwargs = editor.setVisibleBounds.call_args_list[0]
        assert kwargs == {"transition_time": 0.0, "set_center_when_scale_rejected": True}
        # The node sits inside the requested rect, with room around it.
        assert box[0] < 30.0 and box[1] < -60.0 and box[2] > 31.0 and box[3] > -59.7

    def test_each_capture_is_centred_on_its_own_node(self, editor, stage, monkeypatch, tmp_path):
        # Two nodes of one size give two rects at one zoom: the case Houdini
        # ignores without set_center_when_scale_rejected.
        _use_nodes(monkeypatch, stage, {"/stage/far": (30.0, -60.0), "/stage/near": (0.0, 0.0)})
        out = str(tmp_path / "ne.png")

        viewport.capture_network_editor(out, "/stage/far")
        viewport.capture_network_editor(out, "/stage/near")

        # Each capture frames, then puts the old view back: calls 0 and 2 frame.
        (first,), _ = editor.setVisibleBounds.call_args_list[0]
        (second,), kwargs = editor.setVisibleBounds.call_args_list[2]
        assert _centre(first)[1] == pytest.approx(-59.85)
        assert _centre(second)[1] == pytest.approx(0.15)
        assert _centre(first)[0] > _centre(second)[0] + 20
        assert kwargs["set_center_when_scale_rejected"] is True

    def test_the_old_view_is_put_back(self, editor, stage, monkeypatch, tmp_path):
        _use_nodes(monkeypatch, stage, {"/stage/TARGET": (30.0, -60.0)})
        before = editor.visibleBounds.return_value

        reply = viewport.capture_network_editor(str(tmp_path / "ne.png"), "/stage/TARGET")

        (box,), _ = editor.setVisibleBounds.call_args_list[-1]
        assert box is before
        assert reply["pane_restored"] is True and reply["success"] is True


class TestTheReplySaysWhatIsInTheFrame:
    def test_a_framed_node_is_reported_in_view(self, editor, stage, monkeypatch, tmp_path):
        _use_nodes(monkeypatch, stage, {"/stage/TARGET": (30.0, -60.0)})

        reply = viewport.capture_network_editor(str(tmp_path / "ne.png"), "/stage/TARGET")

        assert reply["node_path"] == "/stage/TARGET"
        assert reply["node_bounds"] == [30.0, -60.0, 31.0, -59.7]
        assert reply["visible_bounds"] == [20.0, -70.0, 40.0, -50.0]
        assert reply["node_in_view"] is True

    def test_a_view_that_did_not_reach_the_node_is_reported(
        self, editor, stage, monkeypatch, tmp_path
    ):
        # The editor still shows the previous stretch of the network.
        editor.visibleBounds.return_value = _rect(-10.0, -5.0, 10.0, 5.0)
        _use_nodes(monkeypatch, stage, {"/stage/TARGET": (30.0, -60.0)})

        reply = viewport.capture_network_editor(str(tmp_path / "ne.png"), "/stage/TARGET")

        assert reply["node_in_view"] is False

    def test_without_a_node_the_whole_network_is_framed(self, editor, stage, monkeypatch, tmp_path):
        _use_nodes(monkeypatch, stage, {"/stage/a": (0.0, 0.0), "/stage/b": (30.0, -60.0)})

        reply = viewport.capture_network_editor(str(tmp_path / "ne.png"))

        assert reply["framed"] == "network" and reply["items_framed"] == 2
        assert reply["visible_bounds"] == [20.0, -70.0, 40.0, -50.0]
        assert "node_bounds" not in reply
        assert "node_in_view" not in reply

    def test_an_unreadable_view_leaves_the_answer_open(self, editor, stage, monkeypatch, tmp_path):
        # Readable when the capture starts (it is put back after), not when framed.
        editor.visibleBounds.side_effect = [_rect(0, 0, 1, 1), RuntimeError("no view")]
        _use_nodes(monkeypatch, stage, {"/stage/TARGET": (30.0, -60.0)})

        reply = viewport.capture_network_editor(str(tmp_path / "ne.png"), "/stage/TARGET")

        assert reply["success"] is True
        assert reply["visible_bounds"] is None
        assert reply["node_in_view"] is None
