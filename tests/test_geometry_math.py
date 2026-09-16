"""Tests for the measurement maths behind the modeling acceptance tools.

This is where the acceptance semantics are pinned down: what a distance means,
what counts as a UV overlap, when two faces are in the same island, which
corner is too sharp, and the difference between a mapping that is denser than
its neighbours and one that is actually squashed.

Triangulation and closest-point search are Houdini's -- the module under test
consumes triangles rather than making them, and takes its distances from an
oracle. The oracles here are stated in closed form (the distance to a plane,
to a rectangle) rather than computed by a second geometry engine, so a test
failure means the statistics are wrong and not that two implementations of the
same maths disagree. The native side is exercised by the live probe.
"""

from __future__ import annotations

# Built-in
import math
import os
import sys

# Third-party
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

from fxhoudinimcp_server.geometry_math import (  # noqa: E402
    SurfaceSet,
    UVTriangle,
    compare_surface_sets,
    corner_angle_check,
    corner_angles_deg,
    coverage_table,
    percentile,
    planarity_check,
    plane_deviation,
    sample_triangles,
    significant,
    triangle_overlap_area,
    triangles_are_stacked,
    triangulation_quality,
    udim_tiles,
    uv_distortion_stats,
    uv_island_labels,
    uv_jacobian_singular_values,
    uv_overlaps,
    weighted_percentile,
)

###### Fixtures in plain data


def sheet(
    y: float = 0.0,
    x0: float = 0.0,
    x1: float = 2.0,
    z0: float = 0.0,
    z1: float = 2.0,
    divisions: int = 4,
):
    """A flat quad sheet in the XZ plane at height *y*, already triangulated.

    Two triangles per quad, the way the native Divide hands them over: the
    module under test never sees a polygon with more than three corners.
    """
    triangles = []
    prim_ids = []
    step_x = (x1 - x0) / divisions
    step_z = (z1 - z0) / divisions
    for i in range(divisions):
        for j in range(divisions):
            ax, az = x0 + i * step_x, z0 + j * step_z
            bx, bz = ax + step_x, az + step_z
            corners = [(ax, y, az), (bx, y, az), (bx, y, bz), (ax, y, bz)]
            prim = i * divisions + j
            triangles.append((corners[0], corners[1], corners[2]))
            triangles.append((corners[0], corners[2], corners[3]))
            prim_ids.extend([prim, prim])
    return triangles, prim_ids


def surface(*sheets) -> SurfaceSet:
    triangles: list = []
    prim_ids: list = []
    offset = 0
    for tris, prims in sheets:
        triangles.extend(tris)
        prim_ids.extend(prim + offset for prim in prims)
        offset += max(prims) + 1 if prims else 0
    return SurfaceSet(triangles=triangles, prim_ids=prim_ids)


def rectangle_oracle(y: float, x0: float, x1: float, z0: float, z1: float, prim: int = 0):
    """Distance to an axis-aligned rectangle in the plane ``Y = y``.

    Closed form, so the expected answer of every comparison below can be
    written down rather than computed by the code under test. It takes the
    whole batch of points, the same signature the handler's native pass has.
    """

    def one(point):
        dx = max(x0 - point[0], 0.0, point[0] - x1)
        dz = max(z0 - point[2], 0.0, point[2] - z1)
        dy = point[1] - y
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def measure(points):
        return ([one(point) for point in points], [prim] * len(points))

    return measure


def nearest_of(*oracles):
    """The closest answer among several surfaces, per point."""

    def measure(points):
        answers = [oracle(points) for oracle in oracles]
        distances = []
        prims = []
        for index in range(len(points)):
            best = min(range(len(answers)), key=lambda which: answers[which][0][index])
            distances.append(answers[best][0][index])
            prims.append(answers[best][1][index])
        return (distances, prims)

    return measure


###### Distances, percentiles, precision


def test_percentile_is_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0.95) == 95.0
    assert percentile(values, 1.0) == 100.0
    assert percentile([], 0.95) == 0.0


class TestSignificant:
    """Distances are in whatever units the SOP works in, so precision has to
    travel with the number rather than sit at a fixed decimal place."""

    def test_a_submicron_error_survives(self):
        assert significant(4.2e-8) == 4.2e-8
        assert round(4.2e-8, 6) == 0.0  # what the receipt used to do

    def test_a_large_value_keeps_six_figures(self):
        assert significant(1234.56789) == 1234.57

    def test_zero_and_infinities_pass_through(self):
        assert significant(0.0) == 0.0
        assert significant(math.inf) == math.inf


###### Sampling


class TestSampling:
    def test_same_seed_same_points(self):
        triangles, _prims = sheet()
        first = sample_triangles(triangles, 64, seed=7)
        again = sample_triangles(triangles, 64, seed=7)
        assert first == again
        assert sample_triangles(triangles, 64, seed=8) != first

    def test_every_sample_lands_on_the_sheet_it_came_from(self):
        triangles, _prims = sheet(y=1.5)
        for position, index in sample_triangles(triangles, 200, seed=3):
            assert position[1] == pytest.approx(1.5)
            assert 0.0 <= position[0] <= 2.0
            assert 0.0 <= position[2] <= 2.0
            assert 0 <= index < len(triangles)

    def test_samples_follow_area_not_triangle_count(self):
        """One big triangle and one small one: the big one takes its share."""
        big = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 10.0))
        small = ((100.0, 0.0, 0.0), (101.0, 0.0, 0.0), (100.0, 0.0, 1.0))
        samples = sample_triangles([big, small], 1000, seed=1)
        on_big = sum(1 for _position, index in samples if index == 0)
        assert on_big / 1000 == pytest.approx(100 / 101, abs=0.02)

    def test_zero_area_input_yields_nothing_rather_than_dividing_by_zero(self):
        flat = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        assert sample_triangles([flat], 10, seed=0) == []


###### compare_surface_sets: the acceptance cases named in the brief


def compare(a, b, a_measure, b_measure, **kwargs):
    defaults = {"samples": 400, "seed": 5, "tolerances": [0.01], "max_list": 3}
    defaults.update(kwargs)
    return compare_surface_sets(a, b, a_measure=a_measure, b_measure=b_measure, **defaults)


class TestSurfaceComparison:
    def test_the_same_surface_measures_zero_both_ways(self):
        plane = rectangle_oracle(0.0, 0.0, 2.0, 0.0, 2.0)
        report, samples = compare(surface(sheet()), surface(sheet()), plane, plane)
        for direction in ("a_to_b", "b_to_a"):
            assert report[direction]["max"] == pytest.approx(0.0, abs=1e-12)
            assert report[direction]["mean"] == pytest.approx(0.0, abs=1e-12)
        assert report["max_both"] == pytest.approx(0.0, abs=1e-12)
        assert report["coverage"] == [{"tolerance": 0.01, "a_to_b": 1.0, "b_to_a": 1.0}]
        assert len(samples["a_to_b"].distances) == 400
        assert len(samples["a_to_b"].records()) == 400

    def test_an_offset_plane_reports_the_offset(self):
        report, _samples = compare(
            surface(sheet(y=0.0)),
            surface(sheet(y=0.5)),
            rectangle_oracle(0.0, 0.0, 2.0, 0.0, 2.0),
            rectangle_oracle(0.5, 0.0, 2.0, 0.0, 2.0),
        )
        for direction in ("a_to_b", "b_to_a"):
            assert report[direction]["mean"] == pytest.approx(0.5, abs=1e-9)
            assert report[direction]["max"] == pytest.approx(0.5, abs=1e-9)
        assert report["coverage"][0]["a_to_b"] == 0.0

    def test_a_missing_region_is_only_visible_from_one_side(self):
        """The reason both directions are measured, in one test.

        B is the left half of A. Every sample of B lands on A, so b_to_a alone
        would call this a match; a_to_b is what reports the missing half.
        """
        report, _samples = compare(
            surface(sheet(x0=0.0, x1=2.0)),
            surface(sheet(x0=0.0, x1=1.0)),
            rectangle_oracle(0.0, 0.0, 2.0, 0.0, 2.0),
            rectangle_oracle(0.0, 0.0, 1.0, 0.0, 2.0),
            tolerances=[0.001],
        )
        assert report["b_to_a"]["max"] == pytest.approx(0.0, abs=1e-12)
        assert report["a_to_b"]["max"] == pytest.approx(1.0, abs=0.05)
        assert report["coverage"][0]["b_to_a"] == 1.0
        assert report["coverage"][0]["a_to_b"] == pytest.approx(0.5, abs=0.05)
        assert report["max_both"] == report["a_to_b"]["max"]

    def test_worst_points_name_where_and_which_prims(self):
        report, _samples = compare(
            surface(sheet(x0=0.0, x1=2.0)),
            surface(sheet(x0=0.0, x1=1.0)),
            rectangle_oracle(0.0, 0.0, 2.0, 0.0, 2.0, prim=11),
            rectangle_oracle(0.0, 0.0, 1.0, 0.0, 2.0, prim=7),
        )
        worst = report["a_to_b"]["worst"][0]
        assert worst["P"][0] > 1.5  # in the half B does not have
        assert worst["to_prim"] == 7
        assert worst["from_prim"] is not None
        assert worst["distance"] > 0.5

    def test_thin_parallel_surfaces_are_not_confused_for_one(self):
        """A 2 mm plate against the sheet down its middle: both sides sampled."""
        plate = surface(sheet(y=0.0), sheet(y=0.002))
        middle = surface(sheet(y=0.001))
        report, _samples = compare(
            plate,
            middle,
            nearest_of(
                rectangle_oracle(0.0, 0.0, 2.0, 0.0, 2.0),
                rectangle_oracle(0.002, 0.0, 2.0, 0.0, 2.0),
            ),
            rectangle_oracle(0.001, 0.0, 2.0, 0.0, 2.0),
            tolerances=[0.0005],
        )
        assert report["a_to_b"]["max"] == pytest.approx(0.001, abs=1e-9)
        assert report["a_to_b"]["mean"] == pytest.approx(0.001, abs=1e-9)
        assert report["b_to_a"]["max"] == pytest.approx(0.001, abs=1e-9)
        assert report["coverage"][0]["a_to_b"] == 0.0

    def test_several_tolerances_each_get_a_row(self):
        report, _samples = compare(
            surface(sheet(y=0.0)),
            surface(sheet(y=0.05)),
            rectangle_oracle(0.0, 0.0, 2.0, 0.0, 2.0),
            rectangle_oracle(0.05, 0.0, 2.0, 0.0, 2.0),
            tolerances=[0.01, 0.1],
        )
        assert [row["tolerance"] for row in report["coverage"]] == [0.01, 0.1]
        assert report["coverage"][0]["a_to_b"] == 0.0
        assert report["coverage"][1]["a_to_b"] == 1.0


class TestCoverageTable:
    def test_close_tolerances_stay_two_rows(self):
        """Formatted as keys these two would collide into one and lose a row."""
        distances = [0.0001, 0.0002]
        table = coverage_table(distances, distances, [0.00011234567, 0.00011234568])
        assert len(table) == 2
        assert table[0]["tolerance"] != table[1]["tolerance"]

    def test_a_direction_with_no_samples_is_left_out_rather_than_reported_as_zero(self):
        table = coverage_table([0.0, 1.0], [], [0.5])
        assert table[0]["a_to_b"] == 0.5
        assert "b_to_a" not in table[0]


###### Mixed-topology quality checks


class TestCornerAngles:
    def test_a_square_is_four_right_angles(self):
        square = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]
        assert corner_angles_deg(square) == pytest.approx([90.0] * 4)

    def test_a_zero_length_edge_has_no_angle_rather_than_an_angle_of_zero(self):
        degenerate = [(0, 0, 0), (0, 0, 0), (1, 0, 0)]
        assert corner_angles_deg(degenerate)[0] is None

    def test_a_nearly_flat_corner_is_reported_where_degeneracy_is_not(self):
        """The gap this check fills: this quad has area and no repeated point."""
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.01], [1.0, 0.0, 1.0]]
        report = corner_angle_check(
            [(6, [0, 1, 2, 3])], positions, min_deg=10.0, max_deg=170.0, max_list=5
        )
        assert report["above_max"] == 1
        assert report["flat"][0]["prim"] == 6
        assert report["flat"][0]["corner"] == 1
        assert report["flat"][0]["angle_deg"] > 170.0

    def test_an_ngon_is_not_a_defect_for_being_an_ngon(self):
        hexagon = [
            (math.cos(math.radians(60 * i)), 0.0, math.sin(math.radians(60 * i))) for i in range(6)
        ]
        positions = [list(corner) for corner in hexagon]
        report = corner_angle_check(
            [(0, list(range(6)))], positions, min_deg=10.0, max_deg=170.0, max_list=5
        )
        assert (report["below_min"], report["above_max"]) == (0, 0)
        assert report["min_angle_deg"] == pytest.approx(120.0, abs=1e-6)

    def test_a_sliver_triangle_is_reported_as_sharp(self):
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.001]]
        report = corner_angle_check(
            [(3, [0, 1, 2])], positions, min_deg=10.0, max_deg=170.0, max_list=5
        )
        assert report["below_min"] == 2
        assert report["sharp"][0]["angle_deg"] < 10.0


class TestPlanarity:
    def test_a_flat_ngon_passes(self):
        positions = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0],
            [1.0, 0.0, 0.8],
            [0.0, 0.0, 2.0],
        ]
        report = planarity_check(
            [(0, [0, 1, 2, 3, 4])], positions, planarity_ratio=0.01, max_list=5
        )
        assert report["faces_checked"] == 1
        assert report["nonplanar"] == 0
        assert report["max_planarity"] == 0.0

    def test_a_bent_quad_is_reported_with_how_bent(self):
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 1.0], [0.0, 0.0, 1.0]]
        report = planarity_check([(0, [0, 1, 2, 3])], positions, planarity_ratio=0.01, max_list=5)
        assert report["nonplanar"] == 1
        assert report["max_planarity"] > 0.01
        assert report["nonplanar_faces"][0]["prim"] == 0

    def test_a_triangle_is_never_nonplanar(self):
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
        report = planarity_check([(0, [0, 1, 2])], positions, planarity_ratio=1e-9, max_list=5)
        assert report["faces_checked"] == 0
        assert report["nonplanar"] == 0
        assert plane_deviation([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0)]) == 0.0


###### UV analysis


class TestUVIslands:
    def test_two_faces_sharing_an_edge_with_matching_uvs_are_one_island(self):
        faces = [(0, [0, 1, 2, 3]), (1, [3, 2, 4, 5])]
        uvs = [
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            [(0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)],
        ]
        labels, count = uv_island_labels(faces, uvs)
        assert count == 1
        assert labels == [0, 0]

    def test_a_seam_splits_the_island_though_the_edge_is_still_shared(self):
        faces = [(0, [0, 1, 2, 3]), (1, [3, 2, 4, 5])]
        uvs = [
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            [(5.0, 1.0), (6.0, 1.0), (6.0, 2.0), (5.0, 2.0)],  # cut: different UVs
        ]
        labels, count = uv_island_labels(faces, uvs)
        assert count == 2
        assert labels[0] != labels[1]

    def test_faces_that_share_no_edge_are_separate_islands(self):
        faces = [(0, [0, 1, 2, 3]), (1, [4, 5, 6, 7])]
        uvs = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]] * 2
        _labels, count = uv_island_labels(faces, uvs)
        assert count == 2


def uv_tri(uv, prim=0, face_index=0, world=None):
    return UVTriangle(
        prim=prim,
        face_index=face_index,
        uv=uv,
        world=world or ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )


class TestUVOverlap:
    TRI_A = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

    def test_identical_triangles_overlap_by_their_whole_area(self):
        assert triangle_overlap_area(self.TRI_A, self.TRI_A) == pytest.approx(0.5)

    def test_mirrored_triangles_overlap_the_same_as_positive_ones(self):
        """Winding must not change an area: a flipped island still covers UV."""
        flipped = (self.TRI_A[0], self.TRI_A[2], self.TRI_A[1])
        assert triangle_overlap_area(self.TRI_A, flipped) == pytest.approx(0.5)

    def test_separated_triangles_do_not_overlap(self):
        far = ((5.0, 5.0), (6.0, 5.0), (5.0, 6.0))
        assert triangle_overlap_area(self.TRI_A, far) == 0.0

    def test_triangles_sharing_an_edge_do_not_overlap(self):
        neighbour = ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        assert triangle_overlap_area(self.TRI_A, neighbour) == pytest.approx(0.0, abs=1e-18)

    def test_a_known_partial_overlap_reports_its_area(self):
        shifted = ((0.5, 0.0), (1.5, 0.0), (0.5, 1.0))
        # The two right triangles meet in a triangle of legs 0.5: area 0.125.
        assert triangle_overlap_area(self.TRI_A, shifted) == pytest.approx(0.125)

    def test_stacked_is_the_same_three_corners_in_any_order(self):
        rotated = (self.TRI_A[1], self.TRI_A[2], self.TRI_A[0])
        assert triangles_are_stacked(self.TRI_A, rotated, 1e-6)
        assert triangles_are_stacked(self.TRI_A, self.TRI_A, 1e-6)

    def test_a_partial_overlap_is_never_stacked(self):
        shifted = ((0.5, 0.0), (1.5, 0.0), (0.5, 1.0))
        assert not triangles_are_stacked(self.TRI_A, shifted, 1e-6)

    def test_allow_stacking_cannot_hide_an_accidental_overlap(self):
        """The requirement in one test: stacked shells forgiven, a fold is not."""
        triangles = [
            uv_tri(self.TRI_A, prim=0),
            uv_tri(self.TRI_A, prim=1),  # a deliberate stack
            uv_tri(((0.5, 0.0), (1.5, 0.0), (0.5, 1.0)), prim=2),  # a partial overlap
        ]
        found = uv_overlaps(triangles, area_tolerance=1e-12, stack_tolerance=1e-6, max_pairs=1000)
        assert found["stacked_count"] == 1
        assert {tuple(entry["prims"]) for entry in found["overlapping"]} == {(0, 2), (1, 2)}
        assert found["overlapping_count"] == 2
        assert found["overlap_area"] == pytest.approx(0.25)
        assert found["stacked_area"] == pytest.approx(0.5)
        assert found["records_truncated"] is False

    def test_a_tiny_overlap_is_found_and_a_tolerance_can_ignore_it(self):
        tiny = ((0.999999, 0.0), (1.999999, 0.0), (0.999999, 1.0))
        triangles = [uv_tri(self.TRI_A), uv_tri(tiny, prim=1)]
        strict = uv_overlaps(triangles, area_tolerance=1e-14, stack_tolerance=1e-6, max_pairs=1000)
        assert len(strict["overlapping"]) == 1
        assert strict["overlap_area"] == pytest.approx(5e-13, rel=0.1)

        loose = uv_overlaps(triangles, area_tolerance=1e-9, stack_tolerance=1e-6, max_pairs=1000)
        assert loose["overlapping"] == []

    def test_the_counts_stay_exact_when_the_record_list_is_capped(self):
        """A badly mapped mesh can overlap a million times over; the receipt
        still has to say how many, and which are biggest, without holding them
        all in memory."""
        triangles = [uv_tri(self.TRI_A, prim=i) for i in range(10)]
        found = uv_overlaps(
            triangles,
            area_tolerance=1e-12,
            stack_tolerance=1e-6,
            max_pairs=1000,
            max_records=4,
        )
        assert found["stacked_count"] == 45
        assert len(found["stacked"]) == 4
        assert found["records_truncated"] is True
        assert found["stacked_area"] == pytest.approx(45 * 0.5)

    def test_the_records_kept_are_the_largest(self):
        triangles = [
            uv_tri(((0.0, 0.0), (scale, 0.0), (0.0, scale)), prim=scale) for scale in (1, 2, 3, 4)
        ]
        found = uv_overlaps(
            triangles,
            area_tolerance=1e-12,
            stack_tolerance=1e-6,
            max_pairs=1000,
            max_records=1,
        )
        # The 3-4 pair overlaps by the whole of the smaller: 4.5, the largest.
        assert found["overlapping"][0]["area"] == pytest.approx(4.5)
        assert found["overlapping"][0]["prims"] == [3, 4]

    def test_a_budget_that_bites_is_reported_as_truncated(self):
        triangles = [uv_tri(self.TRI_A, prim=i) for i in range(20)]
        found = uv_overlaps(triangles, area_tolerance=1e-12, stack_tolerance=1e-6, max_pairs=5)
        assert found["truncated"] is True

    def test_many_separate_triangles_stay_cheap(self):
        """The broadphase must not degenerate into every pair."""
        triangles = [uv_tri(((x, 0.0), (x + 0.5, 0.0), (x, 0.5)), prim=x) for x in range(200)]
        found = uv_overlaps(
            triangles, area_tolerance=1e-12, stack_tolerance=1e-6, max_pairs=1_000_000
        )
        assert found["overlapping"] == []
        assert found["candidate_pairs"] < 200 * 199 / 2


###### UV distortion: density variation and anisotropy are different faults

UNIT_TRIANGLE = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


class TestJacobian:
    def test_an_identity_mapping_is_one_and_one(self):
        sigma1, sigma2 = uv_jacobian_singular_values(
            UNIT_TRIANGLE, ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        )
        assert sigma1 == pytest.approx(1.0)
        assert sigma2 == pytest.approx(1.0)

    def test_a_uniform_scale_moves_both_together(self):
        sigma1, sigma2 = uv_jacobian_singular_values(
            UNIT_TRIANGLE, ((0.0, 0.0), (3.0, 0.0), (0.0, 3.0))
        )
        assert (sigma1, sigma2) == pytest.approx((3.0, 3.0))

    def test_an_area_preserving_squash_separates_them(self):
        sigma1, sigma2 = uv_jacobian_singular_values(
            UNIT_TRIANGLE, ((0.0, 0.0), (100.0, 0.0), (0.0, 0.01))
        )
        assert (sigma1, sigma2) == pytest.approx((100.0, 0.01))
        assert sigma1 * sigma2 == pytest.approx(1.0)  # area unchanged

    def test_a_rotation_is_still_one_and_one(self):
        sigma1, sigma2 = uv_jacobian_singular_values(
            UNIT_TRIANGLE, ((0.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
        )
        assert (sigma1, sigma2) == pytest.approx((1.0, 1.0))

    def test_a_degenerate_world_triangle_has_no_mapping(self):
        flat = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        assert uv_jacobian_singular_values(flat, ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))) is None

    def test_a_uv_collapsed_to_a_line_has_a_zero_second_value(self):
        sigma1, sigma2 = uv_jacobian_singular_values(
            UNIT_TRIANGLE, ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
        )
        assert sigma1 > 0.0
        assert sigma2 == pytest.approx(0.0, abs=1e-15)


class TestExtremeAnisotropyStability:
    """The smaller singular value must come from the determinant.

    Taking it from ``half_trace - gap`` subtracts two numbers that agree to
    eleven digits once the stretch passes about 1e5:1, so the answer is
    rounding noise -- and when the noise lands below zero the triangle gets
    called collapsed, which is the opposite of what it is.
    """

    def _squashed(self, scale: float):
        return uv_jacobian_singular_values(
            UNIT_TRIANGLE, ((0.0, 0.0), (scale, 0.0), (0.0, 1.0 / scale))
        )

    def test_a_million_to_one_keeps_both_values(self):
        sigma1, sigma2 = self._squashed(1e6)
        assert sigma1 == pytest.approx(1e6, rel=1e-9)
        assert sigma2 == pytest.approx(1e-6, rel=1e-6)

    def test_a_hundred_million_to_one_is_still_not_collapsed(self):
        sigma1, sigma2 = self._squashed(1e8)
        assert sigma2 > 0.0
        assert sigma1 * sigma2 == pytest.approx(1.0, rel=1e-6)  # area preserved
        stats = uv_distortion_stats(
            [
                uv_tri(
                    ((0.0, 0.0), (1e8, 0.0), (0.0, 1e-8)),
                    world=UNIT_TRIANGLE,
                )
            ]
        )
        assert "collapsed_uv" not in stats
        assert stats["anisotropy"]["max"] == pytest.approx(1e16, rel=1e-6)

    def test_the_naive_form_really_would_have_failed(self):
        """Guards the two tests above: the cancellation is real, not theoretical."""
        sigma1 = 1e8
        m00, m11 = sigma1**2, (1.0 / sigma1) ** 2
        half_trace = 0.5 * (m00 + m11)
        gap = math.sqrt(max(0.0, half_trace * half_trace - m00 * m11))
        assert math.sqrt(max(0.0, half_trace - gap)) != pytest.approx(1e-8, rel=0.5)


class TestWeightedPercentile:
    def test_it_follows_weight_not_count(self):
        # One tenth of the values are bad, but they carry nine tenths of the area.
        pairs = [(1.0, 1.0)] * 9 + [(100.0, 81.0)]
        assert percentile(sorted(value for value, _w in pairs), 0.95) == 100.0
        assert weighted_percentile(pairs, 0.95) == 100.0
        # And the reverse: many tiny bad triangles must not dominate.
        pairs = [(100.0, 0.001)] * 50 + [(1.0, 100.0)]
        assert weighted_percentile(pairs, 0.95) == 1.0

    def test_splitting_a_triangle_does_not_move_it(self):
        """Retessellation independence, which is the reason for weighting."""
        coarse = [(1.0, 4.0), (3.0, 1.0)]
        fine = [(1.0, 1.0)] * 4 + [(3.0, 0.25)] * 4
        assert weighted_percentile(coarse, 0.9) == weighted_percentile(fine, 0.9)

    def test_edges(self):
        assert weighted_percentile([], 0.95) == 0.0
        assert weighted_percentile([(5.0, 0.0)], 0.95) == 5.0


class TestTriangulationQuality:
    UNIT_QUAD_AREA = 1.0

    def test_a_clean_split_reports_nothing(self):
        children = [
            (7, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0))),
            (7, ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0))),
        ]
        report = triangulation_quality(
            children,
            {7: self.UNIT_QUAD_AREA},
            min_area_fraction=1e-6,
            min_angle_deg=1.0,
            max_list=5,
        )
        assert report["degenerate_triangles"] == 0
        assert report["sliver_triangles"] == 0
        assert report["min_area_fraction_seen"] == pytest.approx(0.5)
        assert report["min_angle_deg_seen"] == pytest.approx(45.0)

    def test_a_zero_area_child_is_named_with_its_source_face(self):
        children = [
            (3, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0))),
            (3, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),  # collinear
        ]
        report = triangulation_quality(
            children,
            {3: self.UNIT_QUAD_AREA},
            min_area_fraction=1e-6,
            min_angle_deg=1.0,
            max_list=5,
        )
        assert report["degenerate_triangles"] == 1
        assert report["degenerate_faces"][0]["prim"] == 3
        assert report["degenerate_faces"][0]["area_fraction"] == 0.0

    def test_a_needle_with_real_area_is_a_sliver_not_a_degenerate(self):
        """0.5 degrees at the tip, and a thousandth of the face's area."""
        children = [(9, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.004)))]
        report = triangulation_quality(
            children,
            {9: self.UNIT_QUAD_AREA},
            min_area_fraction=1e-6,
            min_angle_deg=1.0,
            max_list=5,
        )
        assert report["degenerate_triangles"] == 0
        assert report["sliver_triangles"] == 1
        assert report["sliver_faces"][0]["prim"] == 9
        assert report["sliver_faces"][0]["min_angle_deg"] < 1.0

    def test_the_thresholds_are_scale_free(self):
        """The same shape a thousand times bigger reports the same numbers."""
        small = [(0, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.004)))]
        large = [(0, tuple(tuple(value * 1000.0 for value in corner) for corner in small[0][1]))]
        arguments = {"min_area_fraction": 1e-6, "min_angle_deg": 1.0, "max_list": 5}
        report_small = triangulation_quality(small, {0: 1.0}, **arguments)
        report_large = triangulation_quality(large, {0: 1.0e6}, **arguments)
        assert report_small["sliver_triangles"] == report_large["sliver_triangles"] == 1
        assert report_small["min_area_fraction_seen"] == pytest.approx(
            report_large["min_area_fraction_seen"], rel=1e-6
        )
        assert report_small["min_angle_deg_seen"] == pytest.approx(
            report_large["min_angle_deg_seen"], abs=1e-6
        )

    def test_the_thresholds_are_reported_with_the_answer(self):
        report = triangulation_quality(
            [], {}, min_area_fraction=1e-5, min_angle_deg=2.0, max_list=5
        )
        assert report["thresholds"] == {"min_area_fraction": 1e-5, "min_angle_deg": 2.0}


class TestUVDistortion:
    def test_an_area_preserving_stretch_is_reported_as_anisotropy(self):
        """The regression the review named: density alone calls this perfect.

        A unit right triangle mapped to (0,0), (100,0), (0,0.01) keeps its area
        exactly -- every density ratio is 1.0 -- and is stretched ten thousand
        to one along U.
        """
        stats = uv_distortion_stats(
            [uv_tri(((0.0, 0.0), (100.0, 0.0), (0.0, 0.01)), world=UNIT_TRIANGLE)]
        )
        assert stats["density"]["ratio_min"] == pytest.approx(1.0)
        assert stats["density"]["ratio_max"] == pytest.approx(1.0)
        assert stats["anisotropy"]["max"] == pytest.approx(10000.0)
        assert stats["anisotropy"]["area_weighted_mean"] == pytest.approx(10000.0)
        # And it is emphatically not collapsed: sigma2 is 0.01, not the noise
        # left over from subtracting two eigenvalues that agree to eleven
        # digits, which is what the stable form avoids.
        assert "collapsed_uv" not in stats
        assert stats["measured_triangles"] == 1

    def test_a_conformal_mapping_has_anisotropy_one_whatever_its_density(self):
        stats = uv_distortion_stats(
            [
                uv_tri(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), world=UNIT_TRIANGLE),
                uv_tri(((0.0, 0.0), (4.0, 0.0), (0.0, 4.0)), prim=1, world=UNIT_TRIANGLE),
            ]
        )
        assert stats["anisotropy"]["max"] == pytest.approx(1.0)
        assert stats["density"]["ratio_max"] / stats["density"]["ratio_min"] == pytest.approx(4.0)

    def test_density_is_relative_to_this_mesh_own_mean(self):
        """Two faces of equal 3D area, one given twice the UV scale."""
        stats = uv_distortion_stats(
            [
                uv_tri(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), world=UNIT_TRIANGLE),
                uv_tri(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)), prim=1, world=UNIT_TRIANGLE),
            ]
        )
        assert stats["density"]["ratio_max"] / stats["density"]["ratio_min"] == pytest.approx(2.0)
        assert stats["world_area"] == pytest.approx(1.0)
        assert stats["measured_triangles"] == 2

    def test_a_collapsed_uv_is_reported_rather_than_crashing_the_report(self):
        """One collapsed triangle beside a valid one used to raise on log2(0)."""
        stats = uv_distortion_stats(
            [
                uv_tri(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), world=UNIT_TRIANGLE),
                uv_tri(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), prim=9, world=UNIT_TRIANGLE),
            ]
        )
        assert stats["collapsed_uv"] == [9]
        assert stats["measured_triangles"] == 1
        assert stats["anisotropy"]["undefined"] == 1
        assert stats["density"]["ratio_max"] == pytest.approx(1.0)

    def test_every_triangle_collapsed_leaves_no_distortion_numbers_at_all(self):
        stats = uv_distortion_stats(
            [uv_tri(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), world=UNIT_TRIANGLE)]
        )
        assert stats["collapsed_uv"] == [0]
        assert "density" not in stats
        assert "anisotropy" not in stats
        assert stats["uv_area"] == 0.0

    def test_a_degenerate_world_triangle_is_counted_apart(self):
        flat = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        stats = uv_distortion_stats([uv_tri(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), world=flat)])
        assert stats["degenerate_world_triangles"] == 1
        assert "density" not in stats

    def test_winding_is_counted_from_the_uv_triangles_as_given(self):
        stats = uv_distortion_stats(
            [
                uv_tri(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), world=UNIT_TRIANGLE),
                uv_tri(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)), prim=1, world=UNIT_TRIANGLE),
            ]
        )
        assert stats["mirrored_triangles"] == 1
        assert stats["uv_area"] == pytest.approx(1.0)

    def test_a_threshold_counts_both_faults_separately(self):
        stats = uv_distortion_stats(
            [
                uv_tri(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), world=UNIT_TRIANGLE),
                uv_tri(((0.0, 0.0), (100.0, 0.0), (0.0, 0.01)), prim=1, world=UNIT_TRIANGLE),
            ],
            threshold=2.0,
        )
        assert stats["anisotropy"]["over_threshold"] == 1
        assert stats["density"]["outside_threshold"] == 0

    def test_udim_tiles_are_named_by_where_the_triangles_sit(self):
        triangles = [
            uv_tri(((0.1, 0.1), (0.9, 0.1), (0.1, 0.9))),
            uv_tri(((1.1, 0.1), (1.9, 0.1), (1.1, 0.9)), prim=1),
            # A layout that exactly fills tile 1001 must not report four tiles.
            uv_tri(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), prim=2),
        ]
        assert udim_tiles(triangles) == [1001, 1002]
