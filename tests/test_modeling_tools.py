"""Tests for the modeling tool wrapper and its Houdini-free topology maths.

The maths half is the reason this file matters: every number in a mesh report
comes out of functions that never touch ``hou``, so a wrong pole count or a
missed bow tie is caught here rather than in front of a live Houdini.
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
from fxhoudinimcp.tools.modeling import get_mesh_report

# The handler module imports hou at module scope; the maths below does not use
# it, so a stub is enough to get at the functions. Same prelude as
# tests/test_dispatcher.py.
sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

from fxhoudinimcp_server.handlers.modeling_handlers import (  # noqa: E402
    boundary_report,
    build_edge_faces,
    connected_pieces,
    degenerate_faces,
    extract_faces,
    find_poles,
    folded_quads,
    quad_fold_flags,
    valence_map,
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
