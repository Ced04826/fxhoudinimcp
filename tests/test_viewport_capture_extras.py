"""Tests for capture_viewport's close-up, overlay, sheet and shot-list helpers.

What they cover without Houdini: region and overlay validation, the orbit
preset, the angles a camera reports (the inverse of the handler's own view
rotation), the "facing" direction on a thin part whose two sides cancel, the
sheet grid staying under its size limit, the on-screen edge length, the clip
VEX, and the shot list round trip. The node chain, the clip and overlays as
drawn, and pixel-identical replays were checked live on 22.0.368.
"""

from __future__ import annotations

# Built-in
import importlib.util
import json
import math
import os
import sys
from unittest.mock import MagicMock

# Third-party
import pytest

sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

from fxhoudinimcp_server.handlers import viewport_capture_extras as ex  # noqa: E402
from fxhoudinimcp_server.handlers import viewport_capture_handlers as vc  # noqa: E402

# numpy is in Houdini's Python; the MCP venv may not have it.
needs_numpy = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None, reason="numpy is not installed"
)


class TestRegions:
    @pytest.mark.parametrize(
        "spec, kind",
        [
            ({"bbox": [0, 0, 0, 1, 1, 1]}, "bbox"),
            ({"center": [0, 1, 2], "radius": 0.5}, "center"),
            ({"group": "hinge"}, "group"),
            ({"prims": [5, 3, 3]}, "prims"),
        ],
    )
    def test_each_kind_parses(self, spec, kind):
        region = ex.parse_region(spec)
        assert region["kind"] == kind
        if kind == "prims":
            assert region["prims"] == [3, 5]

    @pytest.mark.parametrize(
        "spec, message",
        [
            ({"bbox": [0, 0, 0, 1, 1]}, "bbox"),
            ({"center": [0, 0, 0]}, "radius"),
            ({"center": [0, 0, 0], "radius": 0}, "radius"),
            ({"group": "a", "prims": [1]}, "exactly one"),
            ({"prims": []}, "prims"),
            ({"prims": [-1]}, "prims"),
            ({"bbox": [0, 0, 0, 1, 1, 1], "colour": 1}, "unknown"),
            ("hinge", "dict"),
        ],
    )
    def test_mistakes_are_named(self, spec, message):
        with pytest.raises(ValueError, match=message):
            ex.parse_region(spec)

    def test_a_sphere_region_clips_to_its_sphere_and_the_rest_to_the_box(self):
        sphere = ex.parse_region({"center": [1, 2, 3], "radius": 0.5})
        assert ex.clip_shape(sphere, [0] * 6) == {
            "shape": "sphere",
            "center": [1.0, 2.0, 3.0],
            "radius": 0.5,
        }
        group = ex.parse_region({"group": "g"})
        assert ex.clip_shape(group, [0, 0, 0, 1, 1, 1]) == {
            "shape": "box",
            "bbox": [0, 0, 0, 1, 1, 1],
        }


class TestOverlays:
    def test_known_names_in_a_fixed_order(self):
        assert ex.parse_overlays(["zebra", "triangles"]) == ["triangles", "zebra"]
        assert ex.parse_overlays(None) == []
        assert ex.parse_overlays("poles") == ["poles"]

    def test_unknown_names_are_refused(self):
        with pytest.raises(ValueError, match="curvature"):
            ex.parse_overlays(["curvature"])

    def test_face_vex_only_holds_what_was_asked(self):
        code = ex.vertex_vex(["triangles"])
        assert "nv == 3" in code and "nv > 4" not in code and "aspect" not in code
        assert "aspect" in ex.vertex_vex(["stretch"])

    def test_zebra_is_not_in_the_node_chain(self):
        # Zebra is a per-pixel matcap now: no subdivided copy, no VEX bands.
        assert not hasattr(ex, "zebra_vex")
        assert "zebra" not in ex.vertex_vex(["triangles", "ngons"])

    def test_clip_vex_keeps_a_face_by_any_point_or_its_centroid(self):
        code = ex.clip_vex({"shape": "box", "bbox": [0, 0, 0, 1, 2, 3]})
        assert code.count("keep = 1") == 2 and "primpoints" in code and "removeprim" in code
        assert "q.z <= 3" in code
        sphere = ex.clip_vex({"shape": "sphere", "center": [1, 0, 0], "radius": 0.5})
        assert "distance(q, set(1, 0, 0)) <= 0.5" in sphere


class TestOrbitAndAngles:
    def test_default_orbit_is_two_rings_of_eight(self):
        views = ex.orbit_views(True)
        assert len(views) == 16
        assert {v["elevation"] for v in views} == {30.0, -20.0}
        assert views[1]["azimuth"] == 45.0 and views[1]["name"] == "az45_el30"

    def test_orbit_options_and_limits(self):
        views = ex.orbit_views({"elevations": [0], "azimuths": [10, 100]})
        assert [v["name"] for v in views] == ["az10_el0", "az100_el0"]
        with pytest.raises(ValueError, match="36"):
            ex.orbit_views({"elevations": [0, 10, 20], "azimuths": 13})
        with pytest.raises(ValueError, match="unknown"):
            ex.orbit_views({"rings": 2})
        assert ex.orbit_views(None) == []

    @pytest.mark.parametrize(
        "azimuth, elevation", [(0, 0), (90, 0), (-135, 30), (45, -60), (0, 90)]
    )
    def test_angles_invert_the_view_rotation(self, azimuth, elevation):
        axes = vc.view_rotation(azimuth, elevation)
        az, el = ex.angles_of(axes)
        assert el == pytest.approx(elevation, abs=1e-6)
        if abs(elevation) < 89:
            assert az == pytest.approx(azimuth, abs=1e-6)

    def test_facing_looks_against_the_normal(self):
        assert ex.facing_angles([0, 0, 1]) == (0.0, 0.0)
        assert ex.facing_angles([1, 0, 0]) == (90.0, 0.0)
        assert ex.facing_angles([0, 2, 0])[1] == 90.0
        with pytest.raises(ValueError, match="normal"):
            ex.facing_angles([0, 0, 0])


@needs_numpy
class TestDominantNormal:
    def test_a_thin_plate_does_not_cancel_out(self):
        # Top and bottom of a plate, the top a little larger, plus a hole wall.
        areas = [1.0, 0.9] + [0.05] * 8
        normals = [[0, 1, 0], [0, -1, 0]] + [
            [math.cos(a), 0, math.sin(a)] for a in (i * math.pi / 4 for i in range(8))
        ]
        assert ex.dominant_normal(areas, normals) == pytest.approx([0, 1, 0], abs=1e-6)

    def test_a_curved_patch_gives_its_mean_direction(self):
        normals = [[math.sin(t), math.cos(t), 0] for t in (-0.3, 0.0, 0.3)]
        assert ex.dominant_normal([1, 1, 1], normals) == pytest.approx([0, 1, 0], abs=1e-6)

    def test_nothing_to_face(self):
        assert ex.dominant_normal([], []) is None


class TestCropAndDirection:
    def test_the_crop_holds_the_content_with_a_border(self):
        crop = ex.crop_box([200, 100, 800, 500], (1000, 600))
        assert crop == [170, 70, 830, 530]
        assert crop[0] <= 200 and crop[1] <= 100 and crop[2] >= 800 and crop[3] >= 500

    def test_the_crop_stays_inside_the_image(self):
        assert ex.crop_box([2, 0, 999, 600], (1000, 600)) == [0, 0, 1000, 600]
        assert ex.crop_box(None, (1000, 600)) is None

    def test_world_direction_in_camera_space(self):
        front = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert ex.camera_direction(front, [0, 2, 0]) == pytest.approx([0, 1, 0])
        # Looking from +X (the Right view): world +X points back at the camera.
        right = vc.view_rotation(90, 0)
        assert ex.camera_direction(right, [1, 0, 0]) == pytest.approx([0, 0, 1], abs=1e-9)


@needs_numpy
class TestMatcaps:
    def test_headlight_is_brightest_facing_the_camera_and_never_dark(self):
        values = ex.headlight_matcap(64)
        centre = values[32, 32]
        rim = values[32, 0], values[32, 63], values[0, 32], values[63, 32]
        assert centre > max(rim)
        assert values.min() >= ex.HEADLIGHT["low"] - 1e-9
        assert values.max() <= ex.HEADLIGHT["high"] + 1e-9
        # The light is left of the camera: the left rim is lit more than the right.
        assert values[32, 8] > values[32, 55]

    def test_zebra_bands_follow_the_direction_in_camera_space(self):
        up = ex.zebra_matcap([0, 1, 0], 8, 128)
        # Bands of N . up are horizontal in the matcap: constant along a row.
        assert (up[40] == up[40, 64]).all()
        assert set(up.ravel().tolist()) == set(ex.ZEBRA_BANDS)
        side = ex.zebra_matcap([1, 0, 0], 8, 128)
        assert (side[:, 40] == side[64, 40]).all()

    def test_a_face_square_to_the_direction_sits_mid_band(self):
        # The centre texel is N = (0, 0, 1): N . d = 1 for d along the view.
        # Its neighbours stay in the same band, so a flat face does not flicker.
        values = ex.zebra_matcap([0, 0, 1], 16, 256)
        assert (values[120:136, 120:136] == values[128, 128]).all()


class TestSheetLayout:
    @pytest.mark.parametrize("count", [1, 2, 4, 6, 16, 36])
    def test_the_sheet_stays_within_its_long_side(self, count):
        sizes = [(600, 400)] * count
        layout = ex.sheet_layout(sizes, 2000)
        assert max(layout["size"]) <= 2000
        assert layout["columns"] * layout["rows"] >= count
        for scale, (w, h) in zip(layout["scales"], sizes, strict=True):
            assert w * scale <= layout["cell_w"] + 1e-9 and h * scale <= layout["cell_h"] + 1e-9

    def test_columns_can_be_set(self):
        layout = ex.sheet_layout([(500, 500)] * 4, 2000, columns=2)
        assert (layout["columns"], layout["rows"]) == (2, 2)

    def test_mixed_aspects_fit_their_cells(self):
        layout = ex.sheet_layout([(1000, 200), (300, 900)], 1200)
        assert max(layout["size"]) <= 1200
        assert all(s > 0 for s in layout["scales"])


@needs_numpy
class TestEdgePixels:
    def _front(self, ortho_width=2.0):
        return {
            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "pivot": [0, 0, 0],
            "t": [0, 0, 5],
            "ortho": True,
            "ortho_width": ortho_width,
            "focal": 50.0,
            "aperture": 41.4214,
        }

    def test_ortho_edge_length_in_image_pixels(self):
        segments = [[[0, 0, 0], [0.1, 0, 0]], [[0, 0, 0], [0, 0.2, 0]]]
        # 1000 px viewport shows 2 units; the image is 500 px wide: 250 px per unit.
        stats = ex.edge_pixels(segments, self._front(), 1000.0, 800.0, 500)
        assert stats["edges"] == 2
        assert stats["mean"] == pytest.approx(37.5)
        assert stats["median"] == pytest.approx(37.5)

    def test_edges_behind_a_perspective_camera_are_skipped(self):
        params = dict(self._front(), ortho=False)
        segments = [[[0, 0, 0], [0.1, 0, 0]], [[0, 0, 9], [0.1, 0, 9]]]
        stats = ex.edge_pixels(segments, params, 1000.0, 800.0, 1000)
        assert stats["edges"] == 1

    def test_readability_warning_uses_the_sheet_value(self):
        result = {
            "views": [
                {
                    "name": "a",
                    "azimuth": 10,
                    "elevation": 20,
                    "edge_px": {"median": 9.0},
                    "sheet_edge_px": 4.0,
                },
                {"name": "b", "azimuth": 0, "elevation": 0, "edge_px": {"median": 9.0}},
            ],
            "warnings": [],
        }
        vc._readability(result, 6.0)
        assert len(result["warnings"]) == 1
        assert "a: median edge 4px in the sheet" in result["warnings"][0]
        assert "az 10, el 20" in result["warnings"][0]


class TestShotList:
    def test_round_trip_and_refusals(self, tmp_path):
        camera = {
            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "pivot": [0, 0, 0],
            "t": [0, 0, 5],
            "ortho": False,
            "ortho_width": 1.0,
            "focal": 50.0,
            "aperture": 41.4,
        }
        record = {
            "kind": "fxhoudinimcp.capture_viewport.shots",
            "version": 1,
            "shots": [{"name": "a", "camera": camera, "pixels": [800, 600]}],
        }
        path = ex.write_shot_list(str(tmp_path / "s.json"), record)
        assert ex.read_shot_list(path)["shots"][0]["camera"] == camera

        bad = dict(record, shots=[{"name": "a", "camera": {"axes": []}, "pixels": [1, 1]}])
        (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="camera fields"):
            ex.read_shot_list(str(tmp_path / "bad.json"))
        (tmp_path / "other.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="not a capture_viewport shot list"):
            ex.read_shot_list(str(tmp_path / "other.json"))


class TestHandlerRefusals:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"overlays": ["glow"]}, "unknown overlays"),
            ({"orbit": {"rings": 2}}, "unknown"),
            ({"sheet_max": 100}, "sheet_max"),
            ({"zebra_stripes": 0}, "zebra_stripes"),
            ({"zebra_direction": [0, 0, 0]}, "zebra_direction"),
            ({"lighting": "studio"}, "lighting"),
            ({"region": {"prims": []}}, "prims"),
            ({"views": [{"direction": "facing", "region": "x"}]}, "dict"),
            ({"replay": "x.json", "views": ["front"]}, "replay"),
        ],
    )
    def test_bad_arguments_before_the_viewer(self, kwargs, message, monkeypatch):
        touched = MagicMock()
        monkeypatch.setattr(vc, "_find_scene_viewer", touched)
        with pytest.raises(ValueError, match=message):
            vc.capture_viewport(output_dir="unused", **kwargs)
        touched.assert_not_called()


class TestWrapper:
    @pytest.mark.asyncio
    async def test_new_options_are_passed_through(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.viewport_capture import capture_viewport

        await capture_viewport(
            mock_ctx,
            "C:/tmp/shots",
            region={"center": [0, 0, 0], "radius": 1},
            clip=True,
            compare="/obj/a/SRC",
            overlays=["zebra"],
            orbit=True,
            replay=None,
        )
        _, params = mock_bridge.execute.call_args.args
        assert params["region"] == {"center": [0, 0, 0], "radius": 1}
        assert params["clip"] is True and params["compare"] == "/obj/a/SRC"
        assert params["overlays"] == ["zebra"] and params["orbit"] is True
        assert "replay" not in params and params["sheet_max"] == 2000
        assert "lighting" not in params

    @pytest.mark.asyncio
    async def test_lighting_is_passed_through(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.viewport_capture import capture_viewport

        await capture_viewport(mock_ctx, "C:/tmp/shots", lighting="viewport")
        _, params = mock_bridge.execute.call_args.args
        assert params["lighting"] == "viewport"
