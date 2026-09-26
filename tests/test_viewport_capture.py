"""Tests for capture_viewport's framing maths and the two capture tool wrappers.

The maths is what decides whether a part ends up whole in the image, and it
can be checked without Houdini: a box projected by the same camera model the
handler uses must land inside the frame with the requested margin. That the
model matches Houdini's viewport was measured live on 22.0.368 (predicted
rectangles against drawn pixels, 1-2 px on axis views); these tests keep the
maths from drifting away from what was measured.
"""

from __future__ import annotations

# Built-in
import math
import os
import sys
from unittest.mock import MagicMock

# Third-party
import pytest

sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

from fxhoudinimcp_server.handlers import viewport_capture_handlers as vc  # noqa: E402

W, H = 1219.0, 1915.0


def _box(lo, hi):
    return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def _camera(axes, ortho=False):
    return {
        "axes": axes,
        "pivot": [0.0, 0.0, 0.0],
        "t": [0.0, 0.0, 1.0],
        "ortho": ortho,
        "ortho_width": 1.0,
        "focal": 50.0,
        "aperture": 41.4213562373095,
    }


def _image_rect(points, params, out_w, out_h):
    xs, ys = [], []
    for point in points:
        screen = vc.project(point, params, W, H)
        assert screen is not None
        x, y = vc.to_image(screen[0], screen[1], W, H, out_w, out_h)
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


class TestOutputSize:
    def test_a_tall_target_gets_a_tall_image(self):
        assert vc.output_size(100, 400, 1200) == (300, 1200)

    def test_a_wide_target_gets_a_wide_image(self):
        assert vc.output_size(400, 100, 1600) == (1600, 400)

    def test_a_sliver_is_clamped_to_eight_to_one(self):
        assert vc.output_size(1, 1000, 1600) == (200, 1600)


class TestZoomFactor:
    def test_an_exact_fit_needs_no_zoom(self):
        # Frame 1219 viewport px wide; out 600x300 => 609.5 px tall half-frame 304.75.
        factor = vc.zoom_factor(609.5 * 0.9, 304.75 * 0.9, W, 600, 300, 0.05)
        assert factor == pytest.approx(1.0)

    def test_the_short_frame_side_governs_when_it_is_tighter(self):
        assert vc.zoom_factor(10, 609.5, W, 600, 600, 0.0) == pytest.approx(1.0)


class TestToImage:
    def test_the_viewport_centre_is_the_image_centre(self):
        assert vc.to_image(W / 2, H / 2, W, H, 400, 900) == (200.0, 450.0)

    def test_up_on_screen_is_up_in_the_image(self):
        _, y = vc.to_image(W / 2, H / 2 + 100, W, H, 400, 900)
        assert y < 450.0


class TestViewRotation:
    def test_azimuth_zero_is_the_front_view(self):
        axes = vc.view_rotation(0, 0)
        for row, expected in zip(axes, ([1, 0, 0], [0, 1, 0], [0, 0, 1]), strict=True):
            assert row == pytest.approx(expected, abs=1e-12)

    def test_azimuth_ninety_looks_from_plus_x(self):
        right, up, back = vc.view_rotation(90, 0)
        assert back == pytest.approx([1, 0, 0], abs=1e-9)
        assert right == pytest.approx([0, 0, -1], abs=1e-9)

    def test_straight_down_is_oriented_like_the_top_view(self):
        right, up, back = vc.view_rotation(0, 90)
        assert back == pytest.approx([0, 1, 0], abs=1e-9)
        assert right == pytest.approx([1, 0, 0], abs=1e-9)
        assert up == pytest.approx([0, 0, -1], abs=1e-9)


class TestProjection:
    def test_the_pivot_projects_to_the_centre(self):
        params = _camera(vc.view_rotation(35, 25))
        params["pivot"] = [1.0, 2.0, 3.0]
        params["t"] = [1.0, 2.0, 8.0]
        assert vc.project((1.0, 2.0, 3.0), params, W, H) == pytest.approx((W / 2, H / 2))

    def test_ortho_width_is_the_horizontal_extent(self):
        params = _camera(vc.view_rotation(0, 0), ortho=True)
        params["ortho_width"] = 2.0
        x, _ = vc.project((1.0, 0.0, 0.0), params, W, H)
        assert x == pytest.approx(W)

    def test_a_point_behind_the_camera_has_no_projection(self):
        assert vc.project((0.0, 0.0, 5.0), _camera(vc.view_rotation(0, 0)), W, H) is None


class TestFit:
    @pytest.mark.parametrize("ortho", [False, True])
    @pytest.mark.parametrize("angles", [(0, 0), (35, 25), (-120, -30), (90, 0)])
    def test_the_target_ends_up_whole_and_filling_the_frame(self, ortho, angles):
        box = _box([-0.49, 0.0, 0.0], [0.0, 1.23, 1.26])
        center, radius = vc._center_radius(box)
        start = vc.centred_on(_camera(vc.view_rotation(*angles), ortho), center, radius)
        params, out_w, out_h, _ = vc.fit(box, start, W, H, 1200, 0.05, min_distance=radius * 1.05)
        x0, y0, x1, y1 = _image_rect(box, params, out_w, out_h)
        assert x0 >= -0.5 and y0 >= -0.5 and x1 <= out_w + 0.5 and y1 <= out_h + 0.5
        # Some edge reaches the margin: the target fills the frame, not a
        # corner of it. Measured from the centre, because a perspective
        # projection of a box is not symmetric about it.
        reach_x = max(abs(x0 - out_w / 2), abs(x1 - out_w / 2)) / (out_w / 2)
        reach_y = max(abs(y0 - out_h / 2), abs(y1 - out_h / 2)) / (out_h / 2)
        assert max(reach_x, reach_y) >= 0.85
        assert max(out_w, out_h) == 1200

    def test_a_tall_perspective_frame_narrows_the_lens(self):
        box = _box([-0.01, 0.0, -0.01], [0.01, 2.0, 0.01])
        center, radius = vc._center_radius(box)
        start = vc.centred_on(_camera(vc.view_rotation(0, 0)), center, radius)
        params, out_w, out_h, _ = vc.fit(box, start, W, H, 1200, 0.05, min_distance=radius * 1.05)
        assert out_h > out_w
        assert params["aperture"] < start["aperture"]
        # The long side keeps a sane field of view instead of 100+ degrees.
        fov_y = 2 * math.degrees(
            math.atan(params["aperture"] / 2 / params["focal"] * out_h / out_w)
        )
        assert fov_y < 60

    def test_the_camera_never_enters_the_target(self):
        box = _box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        center, radius = vc._center_radius(box)
        start = vc.centred_on(_camera(vc.view_rotation(20, 10)), center, radius)
        params, *_ = vc.fit(box, start, W, H, 1200, 0.05, min_distance=radius * 1.05)
        assert params["t"][2] - params["pivot"][2] >= radius * 1.05 - 1e-9


class TestParseView:
    def test_a_string_is_a_direction(self):
        assert vc.parse_view("front", 0)["direction"] == "front"

    def test_angles_default_to_perspective_and_get_a_name(self):
        view = vc.parse_view({"azimuth": 35, "elevation": 25}, 0)
        assert view["projection"] == "persp"
        assert view["name"] == "az35_el25"

    @pytest.mark.parametrize(
        "spec, message",
        [
            ({"direction": "front", "azimuth": 10}, "not both"),
            ({"direction": "sideways"}, "unknown direction"),
            ({"direction": "top", "projection": "persp"}, "orthographic"),
            ({"azimuth": 0, "elevation": 120}, "elevation"),
            ({"direction": "front", "zoom": 2}, "unknown keys"),
        ],
    )
    def test_mistakes_are_named(self, spec, message):
        with pytest.raises(ValueError, match=message):
            vc.parse_view(spec, 0)


class TestHandlerRefusesBeforeTouchingTheViewer:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"max_size": 100}, "max_size"),
            ({"margin": 0.6}, "margin"),
            ({"shading": "glossy"}, "unknown shading"),
            ({"views": ["front", {"direction": "front"}]}, "unique"),
            ({"bbox": [0, 0, 0, 1, 1]}, "bbox"),
        ],
    )
    def test_bad_arguments(self, kwargs, message, monkeypatch):
        touched = MagicMock()
        monkeypatch.setattr(vc, "_find_scene_viewer", touched)
        with pytest.raises(ValueError, match=message):
            vc.capture_viewport(output_dir="unused", **kwargs)
        touched.assert_not_called()


class TestToolWrappers:
    @pytest.mark.asyncio
    async def test_capture_viewport_sends_only_what_was_given(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.viewport_capture import capture_viewport

        await capture_viewport(mock_ctx, "C:/tmp/shots", views=["front"], shading="smooth_wire")
        command, params = mock_bridge.execute.call_args.args
        assert command == "viewport.capture_viewport"
        assert params["views"] == ["front"]
        assert params["shading"] == "smooth_wire"
        assert "targets" not in params and "bbox" not in params

    @pytest.mark.asyncio
    async def test_capture_network_editor_passes_the_framing(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.viewport import capture_network_editor

        await capture_network_editor(mock_ctx, "C:/tmp/net.png", network_path="/obj/geo1")
        command, params = mock_bridge.execute.call_args.args
        assert command == "viewport.capture_network_editor"
        assert params["network_path"] == "/obj/geo1"
        assert "node_path" not in params


class TestTargetsAreDrawn:
    """A framed target that the viewer does not draw is a problem, not a picture."""

    def _sop(self, path, parent):
        node = MagicMock()
        node.path.return_value = path
        node.type.return_value.category.return_value.name.return_value = "Sop"
        node.parent.return_value = parent
        node.setDisplayFlag.side_effect = lambda on: setattr(parent, "_display", node)
        return node

    def _network(self, path):
        net = MagicMock()
        net.path.return_value = path
        net._display = None
        net.displayNode.side_effect = lambda: net._display
        return net

    def test_the_viewer_follows_the_target_and_a_non_display_target_is_reported(self, monkeypatch):
        other, net = self._network("/obj/geo1/other"), self._network("/obj/geo1/net")
        shown = self._sop("/obj/geo1/net/out", net)
        target = self._sop("/obj/geo1/net/mid", net)
        net._display = shown
        nodes = {"/obj/geo1/net/mid": target, "/obj/geo1/net": net, "/obj/geo1/other": other}
        monkeypatch.setattr(vc.hou, "node", lambda path: nodes.get(path))
        viewer = MagicMock()
        viewer._pwd = other
        viewer.pwd.side_effect = lambda: viewer._pwd
        viewer.setPwd.side_effect = lambda node: setattr(viewer, "_pwd", node)

        facts = vc._show_targets(viewer, ["/obj/geo1/net/mid"], True, False, {})
        assert facts["network"] == "/obj/geo1/net"
        assert facts["targets_drawn"] == {"/obj/geo1/net/mid": False}

        moved = {}
        facts = vc._show_targets(viewer, ["/obj/geo1/net/mid"], True, True, moved)
        assert facts["targets_drawn"] == {"/obj/geo1/net/mid": True}
        assert moved == {"/obj/geo1/net": "/obj/geo1/net/out"}

    def test_a_non_display_target_is_drawn_through_a_proxy(self, monkeypatch):
        other, net = self._network("/obj/geo1/other"), self._network("/obj/geo1/net")
        net._display = self._sop("/obj/geo1/net/out", net)
        target = self._sop("/obj/geo1/net/mid", net)
        obj = self._network("/obj")
        nodes = {"/obj/geo1/net/mid": target, "/obj/geo1/net": net, "/obj": obj}
        monkeypatch.setattr(vc.hou, "node", lambda path: nodes.get(path))
        proxy = MagicMock()
        proxy.path.return_value = "/obj/__fxmcp_capture_proxy"
        made = []

        def make(sops):
            made.append([n.path() for n in sops])
            return proxy, {"proxy": proxy.path(), "proxy_points": 8, "target_points": 8}

        monkeypatch.setattr(vc, "_make_proxy", make)
        viewer = MagicMock()
        viewer._pwd = other
        viewer.pwd.side_effect = lambda: viewer._pwd
        viewer.setPwd.side_effect = lambda node: setattr(viewer, "_pwd", node)
        moved, proxies = {}, []

        facts = vc._show_targets(viewer, ["/obj/geo1/net/mid"], True, False, moved, proxies)
        assert made == [["/obj/geo1/net/mid"]]
        assert proxies == ["/obj/__fxmcp_capture_proxy"]
        assert facts["via"] == "proxy" and facts["network"] == "/obj"
        assert facts["targets_drawn"] == {"/obj/geo1/net/mid": True}
        assert moved == {}, "no display flag is moved"

    def test_no_targets_need_no_drawing_check(self):
        facts = vc._show_targets(MagicMock(), [], True, False, {})
        assert facts["targets_drawn"] == {}


class TestTargetsDrawBboxFrames:
    """targets decide what is drawn, bbox decides the framing."""

    @pytest.fixture
    def scene(self, monkeypatch):
        node = MagicMock()
        node.path.return_value = "/obj/geo1/sub/part"
        box = MagicMock()
        box.minvec.return_value = (0.0, 0.0, 0.0)
        box.maxvec.return_value = (10.0, 1.0, 1.0)
        geometry = MagicMock()
        geometry.intrinsicValue.return_value = 100
        geometry.boundingBox.return_value = box
        monkeypatch.setattr(vc.hou, "node", lambda path: node)
        monkeypatch.setattr(vc, "_geometry_of", lambda n: (geometry, None))
        monkeypatch.setattr(vc.hou, "Vector3", lambda x, y, z: (x, y, z))

    def test_bbox_alone_decides_the_framing(self, scene):
        corners, drawn, framed = vc._target_corners(["/obj/geo1/sub/part"], [0, 0, 0, 1, 1, 1])
        assert drawn == ["/obj/geo1/sub/part"]
        assert framed == "bbox"
        assert vc._frame_box(corners) == [0, 0, 0, 1, 1, 1]

    def test_without_bbox_the_targets_are_framed(self, scene):
        corners, drawn, framed = vc._target_corners(["/obj/geo1/sub/part"], None)
        assert framed == ["/obj/geo1/sub/part"]
        assert vc._frame_box(corners) == [0, 0, 0, 10, 1, 1]

    def test_nothing_given_frames_nothing(self, scene):
        assert vc._target_corners(None, None) == ([], [], None)


class TestIsolate:
    """isolate maps node paths to the objects that hold them."""

    @staticmethod
    def _scene(monkeypatch):
        from types import SimpleNamespace

        nodes = {}

        def make(path, category, parent):
            node = SimpleNamespace(
                path=lambda: path,
                parent=lambda: nodes.get(parent),
                type=lambda: SimpleNamespace(
                    category=lambda: SimpleNamespace(name=lambda: category)
                ),
            )
            nodes[path] = node

        make("/", "Manager", None)
        make("/obj", "Manager", "/")
        make("/obj/a", "Object", "/obj")
        make("/obj/a/box1", "Sop", "/obj/a")
        make("/obj/sub", "Object", "/obj")
        make("/obj/sub/c", "Object", "/obj/sub")
        make("/obj/sub/c/tube1", "Sop", "/obj/sub/c")
        monkeypatch.setattr(vc.hou, "node", lambda path: nodes.get(path))

    def test_paths_map_to_their_objects(self, monkeypatch):
        self._scene(monkeypatch)
        assert vc._parse_isolate(True) is True
        assert vc._parse_isolate(False) is False
        assert vc._parse_isolate(["/obj/sub/c/tube1", "/obj/a/box1", "/obj/a"]) == [
            "/obj/a",
            "/obj/sub/c",
        ]
        assert vc._parse_isolate("/obj/a") == ["/obj/a"]

    @pytest.mark.parametrize("bad", [[], ["/obj/nope"], ["/obj"], 3, [None]])
    def test_bad_isolate_is_refused(self, monkeypatch, bad):
        self._scene(monkeypatch)
        with pytest.raises(ValueError):
            vc._parse_isolate(bad)

    def test_true_isolates_the_targets_objects_only(self, monkeypatch):
        self._scene(monkeypatch)
        described = ["/obj/a/box1", "/obj/sub/c/tube1"]
        assert vc._isolated_objects(True, described) == ["/obj/a", "/obj/sub/c"]
        assert vc._isolated_objects(True, []) is None
        assert vc._isolated_objects(False, described) is None
        assert vc._isolated_objects(["/obj/a"], []) == ["/obj/a"]
