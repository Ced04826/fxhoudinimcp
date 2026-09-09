"""Tests for the modeling tools and their Houdini-free maths and validation.

That half is the reason this file matters: every number in a mesh report, and
every refusal edit_points issues, comes out of functions that never touch
``hou``, so a wrong pole count or a move that half-applies is caught here rather
than in front of a live Houdini.
"""

from __future__ import annotations

# Built-in
import os
import sys
from unittest.mock import MagicMock

# Third-party
import pytest
from support import tool_input_schema

# Internal
from fxhoudinimcp.tools.modeling import (
    compare_geometry,
    edit_points,
    get_mesh_report,
)

# The handler module imports hou at module scope; the maths below does not use
# it, so a stub is enough to get at the functions. Same prelude as
# tests/test_dispatcher.py.
sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

from fxhoudinimcp_server.handlers.modeling_handlers import (  # noqa: E402
    COMMENT_MARKER,
    apply_moves,
    boundary_report,
    build_edge_faces,
    classify_provenance,
    compare_meshes,
    connected_pieces,
    degenerate_faces,
    edit_comment_line,
    extract_faces,
    find_poles,
    folded_quads,
    geometry_fingerprint,
    position_report,
    provenance_report,
    quad_fold_flags,
    replace_comment_line,
    topology_report,
    valence_map,
    validate_moves,
)

###### Fixtures in plain data
#
#   6 - 7 - 8
#   |   |   |
#   3 - 4 - 5     a 3x3 point grid: 4 quads, 8 boundary edges, 1 loop, no poles
#   |   |   |
#   0 - 1 - 2

GRID_POSITIONS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [2.0, 0.0, 1.0],
    [0.0, 0.0, 2.0],
    [1.0, 0.0, 2.0],
    [2.0, 0.0, 2.0],
]

GRID_FACES = [
    (0, [0, 1, 4, 3]),
    (1, [1, 2, 5, 4]),
    (2, [3, 4, 7, 6]),
    (3, [4, 5, 8, 7]),
]


class TestGrid:
    """The clean case: everything a healthy quad sheet should report."""

    def test_boundary_and_pieces(self):
        edge_faces = build_edge_faces(GRID_FACES)
        assert len(edge_faces) == 12  # 4 interior + 8 boundary

        count, sizes = connected_pieces(len(GRID_FACES), edge_faces)
        assert (count, sizes) == (1, [4])

        boundary, loops, chains = boundary_report(edge_faces)
        assert len(boundary) == 8
        assert loops == [8]
        assert chains == 0

    def test_no_poles_and_no_folds(self):
        edge_faces = build_edge_faces(GRID_FACES)
        boundary, _loops, _chains = boundary_report(edge_faces)
        valence = valence_map(edge_faces)

        # The centre point is the only interior one, and it has four edges.
        assert valence[4] == 4
        assert find_poles(valence, boundary) == []

        diag02, diag13 = folded_quads(GRID_FACES, GRID_POSITIONS)
        assert (diag02, diag13) == ([], [])

    def test_nothing_degenerate(self):
        assert degenerate_faces(GRID_FACES, GRID_POSITIONS, 1e-12) == []


class TestDefects:
    def test_two_disjoint_quads_are_two_pieces_with_two_loops(self):
        positions = GRID_POSITIONS + [
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [11.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
        ]
        faces = [(0, [0, 1, 4, 3]), (1, [9, 10, 11, 12])]
        edge_faces = build_edge_faces(faces)

        assert connected_pieces(len(faces), edge_faces) == (2, [1, 1])
        boundary, loops, chains = boundary_report(edge_faces)
        assert len(boundary) == 8
        assert loops == [4, 4]
        assert chains == 0
        assert degenerate_faces(faces, positions, 1e-12) == []

    def test_bow_tie_quad_is_folded(self):
        """Corners ordered 0,1,3,2 cross the quad over itself."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        bowtie = [(0, [0, 1, 3, 2])]
        diag02, diag13 = folded_quads(bowtie, positions)
        assert diag02 or diag13, "a bow tie must fold on at least one diagonal"

        # The same four points in order are not folded on either diagonal.
        assert folded_quads([(0, [0, 1, 2, 3])], positions) == ([], [])

    def test_a_quad_can_fold_on_one_diagonal_only(self):
        """A planar dart: reflex at corner 2, so only diagonal 1-3 goes outside.

        This is why both diagonals are tested. Checking one would call this quad
        clean, and a triangulator that happens to pick 1-3 would flip a face.
        """
        positions = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 0.0, 2.0],
        ]
        f02, f13 = quad_fold_flags(*positions)
        assert (f02, f13) == (False, True)

    def test_three_faces_on_one_edge_is_non_manifold(self):
        faces = [
            (0, [0, 1, 4, 3]),
            (1, [1, 0, 3, 4]),  # shares edge 0-1 (and more) with face 0
            (2, [0, 1, 9, 10]),  # a third face on edge 0-1
        ]
        edge_faces = build_edge_faces(faces)
        non_manifold = [edge for edge, users in edge_faces.items() if len(users) > 2]
        assert (0, 1) in non_manifold

    def test_repeated_point_is_degenerate(self):
        faces = [(7, [0, 1, 1, 4])]
        assert degenerate_faces(faces, GRID_POSITIONS, 1e-12) == [7]

    def test_zero_area_face_is_degenerate(self):
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        assert degenerate_faces([(3, [0, 1, 2])], positions, 1e-9) == [3]

    def test_interior_points_off_valence_four_are_poles(self):
        """A 5-valence point from a split quad, a 3-valence one from a tri fan.

        Boundary points are excluded on purpose: the rim of an open sheet sits
        at valence 2 and 3 and is not a defect.
        """
        # The 3x3 sheet with its lower-left quad split along 0-4.
        faces = [
            (0, [0, 1, 4]),
            (1, [0, 4, 3]),
            (2, [1, 2, 5, 4]),
            (3, [3, 4, 7, 6]),
            (4, [4, 5, 8, 7]),
        ]
        edge_faces = build_edge_faces(faces)
        boundary, _loops, _chains = boundary_report(edge_faces)
        valence = valence_map(edge_faces)
        assert valence[4] == 5  # the diagonal 0-4 is an extra edge at point 4
        poles = find_poles(valence, boundary)
        assert poles == [(4, 5)], f"expected only point 4 to be a pole, got {poles}"

        # A four-triangle fan: the centre has four edges, so it is not a pole.
        fan = [(0, [0, 1, 4]), (1, [1, 2, 4]), (2, [2, 3, 4]), (3, [3, 0, 4])]
        fan_edges = build_edge_faces(fan)
        fan_boundary, _l, _c = boundary_report(fan_edges)
        assert find_poles(valence_map(fan_edges), fan_boundary) == []

        # Three triangles round one point: valence 3, interior, a pole.
        triangle_fan = [(0, [0, 1, 4]), (1, [1, 2, 4]), (2, [2, 0, 4])]
        tri_edges = build_edge_faces(triangle_fan)
        tri_boundary, _l2, _c2 = boundary_report(tri_edges)
        assert find_poles(valence_map(tri_edges), tri_boundary) == [(4, 3)]


class TestGeoDocumentParsing:
    """Layouts taken verbatim from Houdini 22.0.368 .geo JSON."""

    def _document(self, primitives, pointref, pointcount, primcount):
        return [
            "fileversion",
            "22.0.368",
            "pointcount",
            pointcount,
            "vertexcount",
            len(pointref),
            "primitivecount",
            primcount,
            "topology",
            ["pointref", ["indices", pointref]],
            "primitives",
            primitives,
        ]

    def test_polygon_run_with_rle(self):
        document = self._document(
            [
                [
                    ["type", "Polygon_run"],
                    ["startvertex", 0, "nprimitives", 4, "nvertices_rle", [4, 4]],
                ]
            ],
            [0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7],
            9,
            4,
        )
        parsed = extract_faces(document)
        assert parsed["faces"] == GRID_FACES
        assert parsed["open_polylines"] == []
        assert parsed["other_prims"] == []

    def test_mixed_run_open_curve_and_non_polygon(self):
        """The merge of a grid, a line and a primitive sphere, as observed."""
        document = self._document(
            [
                [
                    ["type", "Polygon_run"],
                    ["startvertex", 0, "nprimitives", 4, "nvertices_rle", [4, 4]],
                ],
                [
                    ["type", "PolygonCurve_run"],
                    ["startvertex", 16, "nprimitives", 1, "nvertices", [2]],
                ],
                [["type", "Sphere"], ["vertex", 18, "transform", [1, 0, 0, 0, 0, -1, 0, 1, 0]]],
            ],
            [0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7, 9, 10, 11],
            12,
            6,
        )
        parsed = extract_faces(document)
        assert [prim_id for prim_id, _points in parsed["faces"]] == [0, 1, 2, 3]
        assert parsed["open_polylines"] == [4]
        assert parsed["other_prims"] == [5]
        assert parsed["prim_count"] == parsed["declared_prim_count"] == 6

    def test_mixed_side_counts_from_rle(self):
        """A poly sphere writes 24 tris, 240 quads, 24 tris in one run."""
        pointref = list(range(3 * 2 + 4 * 1))
        document = self._document(
            [
                [
                    ["type", "Polygon_run"],
                    ["startvertex", 0, "nprimitives", 3, "nvertices_rle", [3, 2, 4, 1]],
                ]
            ],
            pointref,
            10,
            3,
        )
        parsed = extract_faces(document)
        assert [len(points) for _prim_id, points in parsed["faces"]] == [3, 3, 4]

    def test_single_polygon_entry_with_closed_flag(self):
        """Not seen on 22.0.368, but part of the format, so it is handled."""
        document = self._document(
            [
                [["type", "Polygon"], ["vertex", [0, 1, 2], "closed", True]],
                [["type", "Polygon"], ["vertex", [0, 1, 2], "closed", False]],
            ],
            [5, 6, 7],
            8,
            2,
        )
        parsed = extract_faces(document)
        assert parsed["faces"] == [(0, [5, 6, 7])]
        assert parsed["open_polylines"] == [1]

    def test_a_run_that_cannot_be_sized_is_refused(self):
        document = self._document(
            [[["type", "Polygon_run"], ["startvertex", 0, "nprimitives", 4]]], [0, 1, 2], 3, 4
        )
        with pytest.raises(ValueError, match="lists sizes for"):
            extract_faces(document)


###### The MCP wrapper


class TestGetMeshReportTool:
    @pytest.mark.asyncio
    async def test_delegates_with_defaults(self, mock_ctx, mock_bridge):
        mock_bridge.execute.return_value = {"counts": {"quads": 10000}}
        result = await get_mesh_report(mock_ctx, node_path="/obj/geo1/grid1")
        mock_bridge.execute.assert_called_once_with(
            "modeling.get_mesh_report",
            {"node_path": "/obj/geo1/grid1", "max_list": 20},
        )
        assert result == {"counts": {"quads": 10000}}

    @pytest.mark.asyncio
    async def test_passes_group_and_dump_path(self, mock_ctx, mock_bridge):
        await get_mesh_report(
            mock_ctx,
            node_path="/obj/geo1/grid1",
            group="cage",
            max_list=5,
            dump_path="/tmp/mesh.json",
        )
        mock_bridge.execute.assert_called_once_with(
            "modeling.get_mesh_report",
            {
                "node_path": "/obj/geo1/grid1",
                "max_list": 5,
                "group": "cage",
                "dump_path": "/tmp/mesh.json",
            },
        )

    @pytest.mark.asyncio
    async def test_optional_arguments_stay_out_of_the_payload(self, mock_ctx, mock_bridge):
        """Sending group=None would ask the handler to resolve a group named None."""
        await get_mesh_report(mock_ctx, node_path="/obj/geo1/grid1", group=None, dump_path=None)
        _command, params = mock_bridge.execute.call_args.args
        assert "group" not in params
        assert "dump_path" not in params

    @pytest.mark.asyncio
    async def test_schema_types_every_parameter(self):
        from fxhoudinimcp.server import mcp

        tools = {tool.name: tool for tool in await mcp.list_tools()}
        schema = tool_input_schema(tools["get_mesh_report"])
        properties = schema["properties"]
        assert set(properties) >= {"node_path", "group", "max_list", "dump_path"}
        assert properties["node_path"]["type"] == "string"
        assert properties["max_list"]["type"] == "integer"


###### edit_points: validating an entry list before anything is written


class TestValidateMoves:
    """One bad entry has to stop the whole call, so each reason is checked."""

    def _bad(self, moves, match, point_count=9, groups=("top",)):
        with pytest.raises(ValueError, match=match):
            validate_moves(moves, point_count, groups)

    def test_accepts_and_normalises_the_three_shapes(self):
        entries = validate_moves(
            [
                {"point": 0, "to": [1, 2, 3]},
                {"point": 8, "delta": [0, 0.5, 0]},
                {"group": "top", "delta": [0, 1, 0]},
            ],
            9,
            ["top"],
        )
        assert [(e["point"], e["group"], e["mode"]) for e in entries] == [
            (0, None, "to"),
            (8, None, "delta"),
            (None, "top", "delta"),
        ]
        # Integers arrive from JSON; the arithmetic downstream wants floats.
        assert entries[0]["vector"] == (1.0, 2.0, 3.0)

    def test_moves_must_be_a_non_empty_list(self):
        self._bad({"point": 0}, "moves must be a list")
        self._bad([], "nothing to move")

    def test_entry_must_be_a_dict(self):
        self._bad([[0, 1, 2]], r"moves\[0\] must be a dict")

    def test_point_and_group_are_exclusive(self):
        self._bad([{"point": 0, "group": "top", "delta": [0, 1, 0]}], r"moves\[0\] names both")
        self._bad([{"delta": [0, 1, 0]}], r"moves\[0\] names neither")

    def test_to_and_delta_are_exclusive(self):
        self._bad([{"point": 0, "to": [0, 1, 0], "delta": [0, 1, 0]}], r"moves\[0\] gives both")
        self._bad([{"point": 0}], r"moves\[0\] gives neither")

    def test_to_on_a_group_is_refused(self):
        self._bad([{"group": "top", "to": [0, 1, 0]}], "stack every point on the same spot")

    def test_point_must_be_a_whole_number_in_range(self):
        self._bad([{"point": 99, "delta": [0, 1, 0]}], "point 99 is out of range")
        self._bad([{"point": -1, "delta": [0, 1, 0]}], "out of range")
        self._bad([{"point": 1.5, "delta": [0, 1, 0]}], "must be a whole number")
        self._bad([{"point": "0", "delta": [0, 1, 0]}], "must be a whole number")
        # True is an int to Python, which is not a defensible point number.
        self._bad([{"point": True, "delta": [0, 1, 0]}], "must be a whole number")

    def test_the_index_of_the_bad_entry_is_named(self):
        self._bad(
            [{"point": 0, "delta": [0, 1, 0]}, {"point": 0, "delta": [0, 1, 0]}, {"point": 99}],
            r"moves\[2\]",
        )

    def test_unknown_group_lists_the_real_ones(self):
        with pytest.raises(ValueError, match="'nope'") as caught:
            validate_moves([{"group": "nope", "delta": [0, 1, 0]}], 9, ["base", "top"])
        assert "['base', 'top']" in str(caught.value)

    def test_group_error_says_none_when_there_are_none(self):
        with pytest.raises(ValueError, match="none") as caught:
            validate_moves([{"group": "top", "delta": [0, 1, 0]}], 9, [])
        assert "none" in str(caught.value)

    def test_vectors_need_three_finite_numbers(self):
        self._bad([{"point": 0, "delta": [0, 1]}], "three numbers")
        self._bad([{"point": 0, "delta": "up"}], "three numbers")
        self._bad([{"point": 0, "delta": [0, "1", 0]}], "y must be a number")
        self._bad([{"point": 0, "delta": [0, float("nan"), 0]}], "finite")
        self._bad([{"point": 0, "to": [float("inf"), 0, 0]}], "finite")


class TestApplyMoves:
    """Nine points on a line at x = 0..8, so a wrong index is obvious."""

    def _positions(self):
        return [float(value) for point in range(9) for value in (point, 0.0, 0.0)]

    def _entries(self, moves, groups=("top",)):
        return validate_moves(moves, 9, groups)

    def test_absolute_move(self):
        updated, touched, requested = apply_moves(
            self._positions(), self._entries([{"point": 2, "to": [0, 5, 0]}]), {}
        )
        assert updated[6:9] == [0.0, 5.0, 0.0]
        assert touched == [2]
        assert requested == {2: (0.0, 5.0, 0.0)}
        assert updated[0:3] == [0.0, 0.0, 0.0]  # nothing else moved

    def test_relative_move(self):
        updated, touched, requested = apply_moves(
            self._positions(), self._entries([{"point": 2, "delta": [0, 5, 0]}]), {}
        )
        assert updated[6:9] == [2.0, 5.0, 0.0]
        assert touched == [2]
        assert requested == {}  # a delta asks for no particular position

    def test_group_delta_touches_every_member(self):
        updated, touched, _requested = apply_moves(
            self._positions(),
            self._entries([{"group": "top", "delta": [0, 1, 0]}]),
            {"top": [6, 7, 8]},
        )
        assert touched == [6, 7, 8]
        assert [updated[point * 3 + 1] for point in (6, 7, 8)] == [1.0, 1.0, 1.0]
        assert updated[5 * 3 + 1] == 0.0

    def test_an_empty_group_touches_nothing(self):
        updated, touched, _requested = apply_moves(
            self._positions(), self._entries([{"group": "top", "delta": [0, 1, 0]}]), {"top": []}
        )
        assert touched == []
        assert updated == self._positions()

    def test_touched_keeps_first_touch_order_without_repeats(self):
        _updated, touched, _requested = apply_moves(
            self._positions(),
            self._entries(
                [
                    {"point": 5, "delta": [0, 1, 0]},
                    {"point": 1, "delta": [0, 1, 0]},
                    {"point": 5, "delta": [0, 1, 0]},
                ]
            ),
            {},
        )
        assert touched == [5, 1]

    def test_a_delta_after_a_to_carries_the_request_with_it(self):
        """Otherwise the read-back check would compare against a stale target."""
        updated, _touched, requested = apply_moves(
            self._positions(),
            self._entries([{"point": 3, "to": [0, 1, 0]}, {"point": 3, "delta": [0, 2, 0]}]),
            {},
        )
        assert updated[9:12] == [0.0, 3.0, 0.0]
        assert requested == {3: (0.0, 3.0, 0.0)}

    def test_the_input_list_is_not_modified(self):
        original = self._positions()
        apply_moves(original, self._entries([{"point": 0, "to": [9, 9, 9]}]), {})
        assert original == self._positions()


class TestFingerprintAndComment:
    def test_fingerprint_is_count_and_digest(self):
        stamp = geometry_fingerprint(9, b"\x00" * 108)
        count, at, digest = stamp.partition("@")
        assert (count, at) == ("9", "@")
        assert len(digest) == 8  # blake2b with digest_size=4

    def test_fingerprint_follows_both_kinds_of_change(self):
        base = geometry_fingerprint(9, b"\x00" * 108)
        assert geometry_fingerprint(10, b"\x00" * 108) != base  # count changed
        assert geometry_fingerprint(9, b"\x01" + b"\x00" * 107) != base  # a point moved

    def test_comment_line_reads_as_a_sentence(self):
        line = edit_comment_line(3, 0.5, "9@a1b2c3d4")
        assert line == f"{COMMENT_MARKER} 3 points moved, max 0.5000, input 9 pts @a1b2c3d4"

    def test_comment_replaces_the_previous_marked_line(self):
        first = edit_comment_line(1, 0.25, "9@aaaaaaaa")
        second = edit_comment_line(3, 0.5, "9@bbbbbbbb")
        after_first = replace_comment_line("", first)
        after_second = replace_comment_line(after_first, second)
        assert after_second == second
        assert after_second.count(COMMENT_MARKER) == 1

    def test_a_human_comment_survives(self):
        existing = "control cage for the north facade"
        result = replace_comment_line(existing, edit_comment_line(2, 1.0, "9@aaaaaaaa"))
        assert result.splitlines()[0] == existing
        assert result.splitlines()[1].startswith(COMMENT_MARKER)


class TestEditPointsTool:
    @pytest.mark.asyncio
    async def test_create_mode_delegates(self, mock_ctx, mock_bridge):
        mock_bridge.execute.return_value = {"created": True}
        moves = [{"point": 0, "to": [0, 1, 0]}]
        result = await edit_points(
            mock_ctx, moves=moves, after="/obj/geo1/grid1", name="lift_corner"
        )
        mock_bridge.execute.assert_called_once_with(
            "modeling.edit_points",
            {
                "moves": moves,
                "force": False,
                "after": "/obj/geo1/grid1",
                "name": "lift_corner",
            },
        )
        assert result == {"created": True}

    @pytest.mark.asyncio
    async def test_update_mode_passes_every_given_argument(self, mock_ctx, mock_bridge):
        moves = [{"group": "top", "delta": [0, 0.25, 0]}]
        await edit_points(
            mock_ctx,
            moves=moves,
            edit_node="/obj/geo1/edit1",
            expect_points=9,
            force=True,
        )
        mock_bridge.execute.assert_called_once_with(
            "modeling.edit_points",
            {
                "moves": moves,
                "force": True,
                "edit_node": "/obj/geo1/edit1",
                "expect_points": 9,
            },
        )

    @pytest.mark.asyncio
    async def test_optional_arguments_stay_out_of_the_payload(self, mock_ctx, mock_bridge):
        """A literal None for 'after' would read as a node path to the handler."""
        await edit_points(mock_ctx, moves=[{"point": 0, "delta": [0, 1, 0]}], after="/obj/geo1/g")
        _command, params = mock_bridge.execute.call_args.args
        assert set(params) == {"moves", "force", "after"}

    @pytest.mark.asyncio
    async def test_schema_types_every_parameter(self):
        from fxhoudinimcp.server import mcp

        tools = {tool.name: tool for tool in await mcp.list_tools()}
        schema = tool_input_schema(tools["edit_points"])
        properties = schema["properties"]
        assert set(properties) >= {
            "moves",
            "after",
            "edit_node",
            "name",
            "expect_points",
            "force",
        }
        assert properties["moves"]["type"] == "array"
        assert properties["force"]["type"] == "boolean"


###### compare / reload / views: hou-free maths and wrappers


def _flat(positions):
    return [float(value) for point in positions for value in point]


class TestTopologyAndPositions:
    def test_identical_grid_is_same(self):
        faces = list(GRID_FACES)
        pos = _flat(GRID_POSITIONS)
        report = compare_meshes(
            a_faces=faces,
            b_faces=faces,
            a_point_count=9,
            b_point_count=9,
            a_prim_count=4,
            b_prim_count=4,
            a_positions=pos,
            b_positions=pos,
        )
        assert report["same"] is True
        assert report["topology"] == {"identical": True, "points": 9, "prims": 4}
        assert report["positions"]["max_delta"] == 0.0
        assert report["positions"]["over_tolerance"] == 0
        assert "worst" not in report["positions"]

    def test_count_mismatch_names_both_sides(self):
        report = topology_report(GRID_FACES, GRID_FACES[:1], 9, 4, 4, 1)
        assert report["identical"] is False
        assert report["points"] == {"a": 9, "b": 4}
        assert report["prims"] == {"a": 4, "b": 1}
        assert report["first_mismatch"]["prim"] == 1

    def test_face_mismatch_names_the_first_prim(self):
        other = [(0, [0, 1, 4, 3]), (1, [1, 2, 4, 5]), (2, [3, 4, 7, 6]), (3, [4, 5, 8, 7])]
        report = topology_report(GRID_FACES, other, 9, 9, 4, 4)
        assert report["first_mismatch"] == {"prim": 1, "a": [1, 2, 5, 4], "b": [1, 2, 4, 5]}

    def test_ty_shift_is_over_tolerance(self):
        a = _flat(GRID_POSITIONS)
        b = list(a)
        for point in range(9):
            b[point * 3 + 1] += 0.001
        report = position_report(a, b, 1e-5, 20)
        assert report["over_tolerance"] == 9
        assert report["max_delta"] == pytest.approx(0.001, abs=1e-9)
        assert report["worst"][0]["delta"] == pytest.approx(0.001, abs=1e-9)

    def test_within_tolerance_is_not_over(self):
        a = _flat(GRID_POSITIONS)
        b = list(a)
        b[1] += 1e-6
        report = position_report(a, b, 1e-5, 20)
        assert report["over_tolerance"] == 0
        assert "worst" not in report


class TestProvenance:
    def test_identity_tag_keeps_everyone(self):
        kept, new, removed, new_ids, pairs = classify_provenance(list(range(9)), 9)
        assert (kept, new, removed, new_ids) == (9, 0, 0, [])
        assert pairs[0] == (0, 0)

    def test_minus_one_or_out_of_range_is_new(self):
        kept, new, removed, new_ids, _pairs = classify_provenance([0, 1, -1, 99], 3)
        assert (kept, new, removed) == (2, 2, 1)
        assert new_ids == [2, 3]

    def test_duplicate_source_from_extrude_is_new(self):
        """PolyExtrude copies sourcept from the extruded vertices onto new points."""
        sourcept = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 4, 3]
        kept, new, removed, new_ids, _pairs = classify_provenance(sourcept, 9)
        assert (kept, new, removed) == (9, 4, 0)
        assert new_ids == [9, 10, 11, 12]

    def test_default_zero_on_new_points_is_a_duplicate_of_point_zero(self):
        sourcept = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]
        kept, new, removed, new_ids, _pairs = classify_provenance(sourcept, 9)
        assert new_ids == [9, 10, 11, 12]
        assert (kept, new, removed) == (9, 4, 0)

    def test_provenance_report_includes_prims_and_kept_delta(self):
        a = _flat(GRID_POSITIONS)
        b = list(a) + [0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5, 1.0]
        sourcept = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 4, 3]
        sourceprim = [1, 2, 3, 0, 0, 0, 0, 0]
        report = provenance_report(
            sourcept=sourcept,
            a_point_count=9,
            sourceprim=sourceprim,
            a_prim_count=4,
            a_positions=a,
            b_positions=b,
            max_list=20,
        )
        assert report["points"] == {"kept": 9, "new": 4, "removed": 0}
        assert report["new_points"] == [9, 10, 11, 12]
        assert report["prims"] == {"kept": 4, "new": 4, "removed": 0}
        assert report["new_prims"] == [4, 5, 6, 7]
        assert report["kept_max_delta"] == 0.0


class TestCompareGeometryTool:
    @pytest.mark.asyncio
    async def test_delegates_with_defaults(self, mock_ctx, mock_bridge):
        mock_bridge.execute.return_value = {"same": True}
        result = await compare_geometry(mock_ctx, a="/obj/geo1/a", b="/obj/geo1/b")
        mock_bridge.execute.assert_called_once_with(
            "modeling.compare_geometry",
            {"a": "/obj/geo1/a", "b": "/obj/geo1/b", "tolerance": 1e-5, "max_list": 20},
        )
        assert result == {"same": True}

    @pytest.mark.asyncio
    async def test_optional_dump_path_stays_out(self, mock_ctx, mock_bridge):
        await compare_geometry(mock_ctx, a="/obj/geo1/a", b="/obj/geo1/b", dump_path=None)
        _command, params = mock_bridge.execute.call_args.args
        assert "dump_path" not in params

    @pytest.mark.asyncio
    async def test_schema_types_every_parameter(self):
        from fxhoudinimcp.server import mcp

        tools = {tool.name: tool for tool in await mcp.list_tools()}
        schema = tool_input_schema(tools["compare_geometry"])
        properties = schema["properties"]
        assert set(properties) >= {"a", "b", "tolerance", "max_list", "dump_path"}
        assert properties["a"]["type"] == "string"
        assert properties["tolerance"]["type"] == "number"
