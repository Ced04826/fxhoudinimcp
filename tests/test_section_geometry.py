"""Tests for the plane-section maths behind modeling.section_geometry.

Meshes are built by hand as (prim_id, [point ids]) faces over a point list,
the same shape extract_faces hands the handler.
"""

from __future__ import annotations

# Built-in
import math
import os
import sys

# Third-party
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

from fxhoudinimcp_server import geometry_math as gm  # noqa: E402

UP = (0.0, 1.0, 0.0)


def _cut(faces, positions, level, normal=UP, box=None):
    unit = gm.normalize3(normal)
    heights = [gm.dot3(p, unit) for p in positions]
    segments = gm.section_segments(faces, positions, heights, level, unit)
    if box is not None:
        segments = gm.clip_segments(segments, box[0], box[1])
    return gm.chain_segments(segments)


def _cube(lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    positions = [
        (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
        (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1),
    ]  # fmt: skip
    quads = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return list(enumerate(quads)), positions


def _prism(sides, radius, height, centre=(0.0, 0.0), offset=0, prim_offset=0):
    """An open-ended n-gon tube around the y axis: side quads only."""
    positions = []
    for y in (0.0, height):
        for k in range(sides):
            a = 2.0 * math.pi * k / sides
            positions.append(
                (centre[0] + radius * math.cos(a), y, centre[1] + radius * math.sin(a))
            )
    faces = [
        (prim_offset + k, [offset + k, offset + (k + 1) % sides, offset + sides + (k + 1) % sides, offset + sides + k])
        for k in range(sides)
    ]  # fmt: skip
    return faces, positions


def test_a_cube_cut_through_the_middle_is_one_closed_square():
    faces, positions = _cube()
    chains, branches = _cut(faces, positions, 0.5)
    assert branches == 0
    assert len(chains) == 1
    chain = chains[0]
    assert chain.closed and len(chain.points) == 4
    assert gm.chain_length(chain.points, True) == pytest.approx(4.0)
    assert gm.polygon_area_3d(chain.points) == pytest.approx(1.0)
    assert all(p[1] == pytest.approx(0.5) for p in chain.points)


def test_a_cut_exactly_at_a_face_joins_through_the_vertices():
    faces, positions = _cube()
    chains, _ = _cut(faces, positions, 1.0)
    assert len(chains) == 1 and chains[0].closed
    assert sorted(chains[0].points) == sorted(positions[4:])


def test_shared_edges_give_identical_points():
    faces, positions = _cube(hi=(1.0, 3.0, 1.0))
    heights = [p[1] for p in positions]
    segments = gm.section_segments(faces, positions, heights, 1.3, UP)
    by_key = {}
    for key_a, p, key_b, q, _prim in segments:
        for key, point in ((key_a, p), (key_b, q)):
            assert by_key.setdefault(key, point) == point


def test_a_level_outside_the_mesh_cuts_nothing():
    faces, positions = _cube()
    assert _cut(faces, positions, 2.0) == ([], 0)
    assert _cut(faces, positions, -0.1) == ([], 0)


def test_a_polygonal_tube_fits_its_circle():
    faces, positions = _prism(32, 2.5, 1.0, centre=(10.0, -4.0))
    chains, _ = _cut(faces, positions, 0.4)
    assert len(chains) == 1 and chains[0].closed and len(chains[0].points) == 32
    n, u, v = gm.plane_basis(UP)
    (cu, cv), radius, rms, worst = gm.fit_circle_2d(gm.to_plane(chains[0].points, u, v))
    centre = [cu * u[i] + cv * v[i] + 0.4 * n[i] for i in range(3)]
    assert centre == pytest.approx([10.0, 0.4, -4.0])
    assert radius == pytest.approx(2.5)
    assert rms < 1e-9 and worst < 1e-9


def test_an_outline_with_a_hole_is_two_closed_chains():
    outer, outer_positions = _prism(4, 5.0, 1.0)
    hole, hole_positions = _prism(16, 1.0, 1.0, offset=len(outer_positions), prim_offset=len(outer))
    chains, _ = _cut(outer + hole, outer_positions + hole_positions, 0.5)
    closed = [chain for chain in chains if chain.closed]
    assert len(closed) == 2
    _n, u, v = gm.plane_basis(UP)
    small, large = sorted(closed, key=lambda c: gm.polygon_area_3d(c.points))
    assert gm.point_in_polygon_2d(
        gm.to_plane(small.points, u, v)[0], gm.to_plane(large.points, u, v)
    )
    assert not gm.point_in_polygon_2d(
        gm.to_plane(large.points, u, v)[0], gm.to_plane(small.points, u, v)
    )


def test_an_open_strip_gives_one_open_chain_walked_end_to_end():
    positions = [(float(i), 0.0, 0.0) for i in range(4)] + [(float(i), 1.0, 0.0) for i in range(4)]
    faces = [(i, [i, i + 1, i + 5, i + 4]) for i in range(3)]
    chains, _ = _cut(faces, positions, 0.5)
    assert len(chains) == 1
    chain = chains[0]
    assert not chain.closed and len(chain.points) == 4
    assert [p[0] for p in chain.points] in ([0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0])
    assert gm.chain_length(chain.points, False) == pytest.approx(3.0)
    assert len(chain.prims) == 3


def test_a_concave_face_crossed_four_times_gives_two_segments():
    # A U-shaped face in the XY plane; y = 1.5 crosses both arms.
    positions = [
        (0, 0, 0),
        (3, 0, 0),
        (3, 2, 0),
        (2, 2, 0),
        (2, 1, 0),
        (1, 1, 0),
        (1, 2, 0),
        (0, 2, 0),
    ]
    positions = [tuple(float(c) for c in p) for p in positions]
    faces = [(0, list(range(8)))]
    chains, _ = _cut(faces, positions, 1.5)
    assert len(chains) == 2
    spans = sorted(sorted(p[0] for p in chain.points) for chain in chains)
    assert spans == [[0.0, 1.0], [2.0, 3.0]]


def test_a_box_clips_chains_open_at_its_faces():
    faces, positions = _cube(hi=(2.0, 1.0, 2.0))
    chains, _ = _cut(faces, positions, 0.5, box=((-1.0, -1.0, -1.0), (1.0, 1.0, 3.0)))
    assert len(chains) == 1
    chain = chains[0]
    assert not chain.closed
    assert max(p[0] for p in chain.points) == pytest.approx(1.0)
    assert gm.chain_length(chain.points, False) == pytest.approx(1.0 + 2.0 + 1.0)


def test_a_tilted_plane_cuts_where_it_should():
    faces, positions = _cube()
    normal = gm.normalize3((1.0, 1.0, 0.0))
    chains, _ = _cut(faces, positions, gm.dot3((0.5, 0.5, 0.0), normal), normal=normal)
    assert len(chains) == 1 and chains[0].closed
    assert all(
        gm.dot3(p, normal) == pytest.approx(gm.dot3((0.5, 0.5, 0.0), normal))
        for p in chains[0].points
    )


def test_plane_basis_is_right_handed():
    for normal in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.3, -0.2, 0.9)):
        n, u, v = gm.plane_basis(normal)
        assert gm.cross3(u, v) == pytest.approx(n)
        assert gm.dot3(u, n) == pytest.approx(0.0, abs=1e-12)


def test_circle_fit_refuses_what_is_not_a_circle():
    assert gm.fit_circle_2d([(0.0, 0.0), (1.0, 1.0)]) is None
    assert gm.fit_circle_2d([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]) is None
    fitted = gm.fit_circle_2d([(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
    assert fitted[0] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert fitted[1] == pytest.approx(1.0)


def test_a_square_is_not_reported_as_round():
    square = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
    centre, radius, rms, corners = gm.fit_circle_2d(square)
    assert rms == pytest.approx(0.0, abs=1e-12) and corners == pytest.approx(0.0, abs=1e-12)
    assert gm.chord_deviation(square, True, centre, radius) == pytest.approx(math.sqrt(2.0) - 1.0)
    ring = [(math.cos(2 * math.pi * k / 32), math.sin(2 * math.pi * k / 32)) for k in range(32)]
    centre, radius, _rms, _corners = gm.fit_circle_2d(ring)
    assert gm.chord_deviation(ring, True, centre, radius) == pytest.approx(
        1.0 - math.cos(math.pi / 32)
    )
