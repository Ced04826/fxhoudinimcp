"""Measurement maths for the modeling acceptance tools.

Plain Python -- no ``hou``, no ``numpy``, no global state -- so the parts that
decide whether a mesh passes (what a distance means, when two UV triangles
overlap, what counts as an island, which corner is too sharp, how much a
mapping stretches) can be tested without a running Houdini.

What is deliberately *not* here: triangulation and closest-point search.
Houdini does both, and doing them again in Python was worse than redundant --
a triangulation of the UV polygon can pick a different diagonal than the 3D
one, which hides a folded mapping, and a mesh sampled on one triangulation but
queried against another compares two different surfaces. The handlers run the
native Divide verb once per side and everything below consumes *those*
triangles: the same ones the sampler walks, the same ones the native distance
pass answers from, and the same corners the UVs come off. Distances arrive
here through a callable that takes the whole batch of sample points, so the
handler can hand them to one native call and nothing loops per sample.

Conventions: a point is an ``(x, y, z)`` tuple, a UV a ``(u, v)`` tuple, a
triangle a tuple of three of either.
"""

from __future__ import annotations

# Built-in
import heapq
import math
import random
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

###### Small vector helpers
#
# Tuples rather than a Vector class: these run in the innermost loop of every
# tool here, and attribute lookup costs more than the arithmetic does.

Point3 = tuple[float, float, float]
Point2 = tuple[float, float]
Triangle3 = tuple[Point3, Point3, Point3]
Triangle2 = tuple[Point2, Point2, Point2]


def cross3(a: Point3, b: Point3) -> Point3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot3(a: Point3, b: Point3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sub3(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def length3(a: Point3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def is_finite_point(p: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in p)


def point_of(positions: Sequence[Sequence[float]], index: int) -> Point3:
    """Point *index* as a tuple; the origin when the index is out of range.

    Out of range means a face list and a point list disagree, which a report
    should survive rather than raise on: the face is still worth naming.
    """
    if 0 <= index < len(positions):
        p = positions[index]
        return (float(p[0]), float(p[1]), float(p[2]))
    return (0.0, 0.0, 0.0)


def face_points(point_ids: Sequence[int], positions: Sequence[Sequence[float]]) -> list[Point3]:
    return [point_of(positions, index) for index in point_ids]


def newell_normal(points: Sequence[Point3]) -> Point3:
    """Area-weighted normal of a polygon, robust to non-planarity."""
    nx = ny = nz = 0.0
    p = points[-1]
    for q in points:
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
        p = q
    return (0.5 * nx, 0.5 * ny, 0.5 * nz)


def triangle_area_3d(a: Point3, b: Point3, c: Point3) -> float:
    return 0.5 * length3(cross3(sub3(b, a), sub3(c, a)))


def polygon_area_3d(points: Sequence[Point3]) -> float:
    """Area of a planar-ish polygon, from the length of its Newell normal."""
    return length3(newell_normal(points))


def signed_area_2d(points: Sequence[Point2]) -> float:
    """Signed area of a 2D polygon; positive is counter-clockwise."""
    total = 0.0
    p = points[-1]
    for q in points:
        total += p[0] * q[1] - q[0] * p[1]
        p = q
    return 0.5 * total


def signed_triangle_area_2d(a: Point2, b: Point2, c: Point2) -> float:
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def significant(value: float, digits: int = 6) -> float:
    """Round to *digits* significant figures, not to a fixed decimal place.

    A distance is in whatever units the SOP works in. ``round(x, 6)`` turns a
    real 4e-8 error into 0.0 on a metre-scale model and keeps six meaningless
    digits on a millimetre-scale one; significant figures say the same thing at
    every scale, which is what lets a caller compare a receipt against a
    tolerance it chose itself.
    """
    if not math.isfinite(value) or value == 0.0:
        return value
    return round(value, -int(math.floor(math.log10(abs(value)))) + (digits - 1))


###### Per-face quality checks for a mixed-topology cage
#
# Neither check treats a face as bad for being a triangle or an n-gon: a planar
# hexagon with even corners is a legitimate face, and calling it a defect would
# train the reader to ignore the report.


def corner_angles_deg(points: Sequence[Point3]) -> list[float | None]:
    """Interior angle at every corner, in degrees; ``None`` where undefined.

    ``None`` means a zero-length edge met the corner, so there is no angle to
    report -- distinct from an angle of zero, which is a real, very sharp one.
    """
    count = len(points)
    angles: list[float | None] = []
    for index in range(count):
        previous = points[index - 1]
        current = points[index]
        following = points[(index + 1) % count]
        u = sub3(previous, current)
        v = sub3(following, current)
        lu, lv = length3(u), length3(v)
        if lu == 0.0 or lv == 0.0:
            angles.append(None)
            continue
        cosine = dot3(u, v) / (lu * lv)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return angles


def corner_angle_check(
    faces: Sequence[tuple[int, Sequence[int]]],
    positions: Sequence[Sequence[float]],
    *,
    min_deg: float,
    max_deg: float,
    max_list: int,
) -> dict:
    """Corners sharper than *min_deg* or flatter than *max_deg*.

    The gap this fills: a face with a corner at 179.6 degrees has real area and
    no repeated point, so a degenerate-face count of zero says nothing about
    it, yet it is what turns into a crease or a shading artefact after a
    subdivision.

    Angles are unsigned, so they run 0 to 180 and a reflex corner reports its
    explement: 181 degrees reads as 179, and 350 as 10. Both ends of the band
    still catch what they are for, because a corner nearly folded back on
    itself is a spike either way round.
    """
    below: list[dict] = []
    above: list[dict] = []
    undefined = 0
    sharpest = 180.0
    flattest = 0.0
    for prim_id, point_ids in faces:
        if len(point_ids) < 3:
            continue
        for corner, angle in enumerate(corner_angles_deg(face_points(point_ids, positions))):
            if angle is None:
                undefined += 1
                continue
            sharpest = min(sharpest, angle)
            flattest = max(flattest, angle)
            entry = {
                "prim": prim_id,
                "corner": corner,
                "point": point_ids[corner],
                "angle_deg": angle,
            }
            if angle < min_deg:
                below.append(entry)
            elif angle > max_deg:
                above.append(entry)

    report: dict = {
        "thresholds": {"min_deg": min_deg, "max_deg": max_deg},
        "below_min": len(below),
        "above_max": len(above),
    }
    if sharpest <= flattest:
        report["min_angle_deg"] = round(sharpest, 3)
        report["max_angle_deg"] = round(flattest, 3)
    if undefined:
        report["undefined_corners"] = undefined
    if max_list:
        if below:
            report["sharp"] = [
                {**entry, "angle_deg": round(entry["angle_deg"], 3)}
                for entry in sorted(below, key=lambda e: e["angle_deg"])[:max_list]
            ]
        if above:
            report["flat"] = [
                {**entry, "angle_deg": round(entry["angle_deg"], 3)}
                for entry in sorted(above, key=lambda e: -e["angle_deg"])[:max_list]
            ]
    return report


def plane_deviation(points: Sequence[Point3]) -> float:
    """Largest corner distance from the polygon's Newell plane.

    The plane runs through the corner average with the polygon's area-weighted
    (Newell) normal. That is *not* the least-squares best-fit plane -- fitting
    one would mean an eigen decomposition of the corner covariance -- but it is
    the plane the face's own winding defines, it is exact for a planar polygon,
    it degrades smoothly, and it costs one pass. Zero for any triangle.
    """
    count = len(points)
    if count < 4:
        return 0.0
    normal = newell_normal(points)
    scale = length3(normal)
    if scale == 0.0:
        return 0.0
    unit = (normal[0] / scale, normal[1] / scale, normal[2] / scale)
    centre = (
        sum(p[0] for p in points) / count,
        sum(p[1] for p in points) / count,
        sum(p[2] for p in points) / count,
    )
    return max(abs(dot3(sub3(p, centre), unit)) for p in points)


def planarity_check(
    faces: Sequence[tuple[int, Sequence[int]]],
    positions: Sequence[Sequence[float]],
    *,
    planarity_ratio: float,
    max_list: int,
) -> dict:
    """How far each face is from flat, as a scale-free ratio.

    The largest corner distance from the face's Newell plane (see
    ``plane_deviation``; not a least-squares fit), divided by the face's
    bounding-box diagonal: 0.01 is a corner a hundredth of the face's own width
    off the plane. Triangles are planar by definition and are never flagged.
    """
    flagged: list[dict] = []
    worst_ratio = 0.0
    checked = 0

    for prim_id, point_ids in faces:
        sides = len(point_ids)
        if sides < 4:
            continue
        checked += 1
        corners = face_points(point_ids, positions)
        # Bounding-box diagonal rather than the true diameter: the same scale
        # to within a constant, and linear in the corner count instead of
        # quadratic, which matters for a 200-sided fill.
        diameter = length3(
            (
                max(p[0] for p in corners) - min(p[0] for p in corners),
                max(p[1] for p in corners) - min(p[1] for p in corners),
                max(p[2] for p in corners) - min(p[2] for p in corners),
            )
        )
        if diameter <= 0.0:
            continue
        ratio = plane_deviation(corners) / diameter
        worst_ratio = max(worst_ratio, ratio)
        if ratio > planarity_ratio:
            flagged.append({"prim": prim_id, "sides": sides, "planarity": ratio})

    report: dict = {
        "threshold": planarity_ratio,
        "faces_checked": checked,
        "nonplanar": len(flagged),
        "max_planarity": significant(worst_ratio),
    }
    if flagged and max_list:
        report["nonplanar_faces"] = [
            {**entry, "planarity": significant(entry["planarity"])}
            for entry in sorted(flagged, key=lambda e: -e["planarity"])[:max_list]
        ]
    return report


def triangulation_quality(
    children: Sequence[tuple[int, Triangle3]],
    source_area: dict[int, float],
    *,
    min_area_fraction: float,
    min_angle_deg: float,
    max_list: int,
) -> dict:
    """What the triangulator actually produced, face by face.

    Counting triangles only says a face was split into as many pieces as
    expected; it says nothing about whether the pieces are usable. Two faults
    are measured here, both scale-free so a threshold means the same thing on a
    millimetre part and a building:

    ``area_fraction``  a child triangle's area over its source face's area. A
                       quad split into two triangles gives about 0.5 each; a
                       value near zero is a sliver the triangulator produced
                       from a bad corner choice or a near-degenerate face.
    ``min_angle_deg``  the smallest corner angle in the child triangle. A
                       sliver with a real area can still be a one-degree
                       needle, which is what shades badly and what a solver
                       trips over.

    Every entry names the source prim, which is the face the caller knows.
    """
    zero_area: list[dict] = []
    slivers: list[dict] = []
    worst_fraction = math.inf
    worst_angle = 180.0

    for prim_id, triangle in children:
        area = triangle_area_3d(*triangle)
        parent = source_area.get(prim_id, 0.0)
        fraction = area / parent if parent > 0.0 else 0.0
        angles = [angle for angle in corner_angles_deg(list(triangle)) if angle is not None]
        smallest = min(angles) if angles else 0.0

        if area <= 0.0 or (parent > 0.0 and fraction < min_area_fraction):
            zero_area.append(
                {"prim": prim_id, "area_fraction": fraction, "min_angle_deg": smallest}
            )
        elif smallest < min_angle_deg:
            slivers.append({"prim": prim_id, "area_fraction": fraction, "min_angle_deg": smallest})
        worst_fraction = min(worst_fraction, fraction)
        worst_angle = min(worst_angle, smallest)

    report: dict = {
        "thresholds": {
            "min_area_fraction": min_area_fraction,
            "min_angle_deg": min_angle_deg,
        },
        "degenerate_triangles": len(zero_area),
        "sliver_triangles": len(slivers),
    }
    if children:
        report["min_area_fraction_seen"] = significant(
            0.0 if worst_fraction is math.inf else worst_fraction
        )
        report["min_angle_deg_seen"] = round(worst_angle, 3)
    if max_list:
        for key, entries in (("degenerate_faces", zero_area), ("sliver_faces", slivers)):
            if entries:
                report[key] = [
                    {
                        "prim": entry["prim"],
                        "area_fraction": significant(entry["area_fraction"]),
                        "min_angle_deg": round(entry["min_angle_deg"], 3),
                    }
                    for entry in sorted(entries, key=lambda e: e["area_fraction"])[:max_list]
                ]
    return report


###### Deterministic area-weighted sampling


def triangle_areas(triangles: Sequence[Triangle3]) -> list[float]:
    return [triangle_area_3d(*tri) for tri in triangles]


def sample_triangles(
    triangles: Sequence[Triangle3], count: int, seed: int
) -> list[tuple[Point3, int]]:
    """*count* points spread over the triangles in proportion to their area.

    Stratified: the i-th sample draws its triangle from slice
    ``[i / count, (i + 1) / count)`` of the cumulative area, so a small region
    is hit at a rate set by its area rather than by luck, and two runs with the
    same seed produce the same points. Returns ``(position, triangle index)``.
    """
    if count <= 0 or not triangles:
        return []
    areas = triangle_areas(triangles)
    cumulative: list[float] = []
    running = 0.0
    for area in areas:
        running += area
        cumulative.append(running)
    if running <= 0.0:
        return []

    rng = random.Random(seed)
    samples: list[tuple[Point3, int]] = []
    last = len(triangles) - 1
    for index in range(count):
        target = (index + rng.random()) / count * running
        which = min(bisect_left(cumulative, target), last)
        a, b, c = triangles[which]
        # Turk's square-root mapping: uniform over the triangle, not clustered
        # at one corner the way raw barycentric coordinates would be.
        r1 = math.sqrt(rng.random())
        r2 = rng.random()
        w0 = 1.0 - r1
        w1 = r1 * (1.0 - r2)
        w2 = r1 * r2
        samples.append(
            (
                (
                    a[0] * w0 + b[0] * w1 + c[0] * w2,
                    a[1] * w0 + b[1] * w1 + c[1] * w2,
                    a[2] * w0 + b[2] * w1 + c[2] * w2,
                ),
                which,
            )
        )
    return samples


###### Distance statistics and the two-way surface comparison


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of an already sorted list.

    Nearest-rank rather than interpolated: every value reported is one that was
    actually measured, so it can be found in the dump.
    """
    if not sorted_values:
        return 0.0
    rank = math.ceil(fraction * len(sorted_values))
    return sorted_values[min(max(rank, 1), len(sorted_values)) - 1]


def weighted_percentile(pairs: Sequence[tuple[float, float]], fraction: float) -> float:
    """Nearest-rank percentile over ``(value, weight)`` pairs.

    Counting triangles instead of weighting them makes a per-triangle figure
    depend on how the mesh happens to be tessellated: split one bad face into
    a hundred and it dominates a count-based percentile without any change to
    the surface. Weighting by area asks "over 95% of the *surface*, how bad
    does it get", which is the question, and it does not move when the same
    shape is retessellated.
    """
    if not pairs:
        return 0.0
    ordered = sorted(pairs)
    total = sum(weight for _value, weight in ordered)
    if total <= 0.0:
        return ordered[-1][0]
    target = fraction * total
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= target:
            return value
    return ordered[-1][0]


@dataclass
class SurfaceSet:
    """One side of a comparison: triangles, and the prim each came from.

    Both lists come from the same natively triangulated geometry the distance
    oracle queries, which is what makes ``a`` compared with ``a`` measure zero
    even when its faces are not planar.
    """

    triangles: list[Triangle3] = field(default_factory=list)
    prim_ids: list[int] = field(default_factory=list)

    def area(self) -> float:
        return sum(triangle_areas(self.triangles))


@dataclass
class DirectionSamples:
    """One direction's measurements, as parallel arrays rather than records.

    At the sample ceiling this is the difference between twenty megabytes and
    a hundred: a dict per sample costs several hundred bytes, and the receipt
    only ever names a handful of them. The full records are built on demand,
    for the dump.
    """

    positions: list[Point3] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    from_prims: list[int | None] = field(default_factory=list)
    to_prims: list[int | None] = field(default_factory=list)

    def record(self, index: int) -> dict:
        return {
            "P": list(self.positions[index]),
            "distance": self.distances[index],
            "from_prim": self.from_prims[index],
            "to_prim": self.to_prims[index],
        }

    def records(self) -> list[dict]:
        return [self.record(index) for index in range(len(self.distances))]


def measure_direction(
    source: SurfaceSet,
    measure: Callable[[list[Point3]], tuple[Sequence[float], Sequence[int | None]]],
    *,
    samples: int,
    seed: int,
) -> DirectionSamples:
    """Sample *source* and measure every sample against the other surface.

    *measure* takes **all** the sample points at once and returns their
    distances and the prim each landed on. Taking the whole batch is what lets
    the handler hand the work to one native call instead of running a Python
    loop over tens of thousands of samples on Houdini's main thread; an oracle
    stated in closed form satisfies the same signature in a test.
    """
    points = sample_triangles(source.triangles, samples, seed)
    positions = [position for position, _index in points]
    distances, to_prims = measure(positions)
    if len(distances) != len(positions) or len(to_prims) != len(positions):
        raise ValueError(
            f"the distance measurement returned {len(distances)} distances and "
            f"{len(to_prims)} prim ids for {len(positions)} samples"
        )
    return DirectionSamples(
        positions=positions,
        distances=list(distances),
        from_prims=[
            source.prim_ids[index] if index < len(source.prim_ids) else None
            for _position, index in points
        ],
        to_prims=list(to_prims),
    )


def direction_stats(samples: DirectionSamples, max_list: int) -> dict:
    """Mean, P95 and max of one direction, with the worst sample points.

    The samples are already spread in proportion to area, so an unweighted
    percentile over them is an area-weighted percentile over the surface --
    the number does not shift when the same shape is retessellated.
    """
    distances = samples.distances
    if not distances:
        return {"samples": 0}
    ordered = sorted(distances)
    finite = [value for value in ordered if math.isfinite(value)]
    report: dict = {
        "samples": len(distances),
        "mean": (sum(finite) / len(finite)) if finite else math.inf,
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }
    if max_list:
        worst = heapq.nlargest(max_list, range(len(distances)), key=lambda i: distances[i])
        report["worst"] = [
            {
                "P": [significant(value) for value in samples.positions[index]],
                "distance": significant(distances[index]),
                "from_prim": samples.from_prims[index],
                "to_prim": samples.to_prims[index],
            }
            for index in worst
        ]
    return report


def coverage_table(
    a_distances: Sequence[float], b_distances: Sequence[float], tolerances: Sequence[float]
) -> list[dict]:
    """Share of samples within each tolerance, per direction.

    A list of ``{tolerance, a_to_b, b_to_a}`` rather than a dict keyed by the
    tolerance: two tolerances that differ below the formatting precision would
    collide into one key and quietly lose a row.

    Coverage is one-way containment, not similarity: a small surface entirely
    inside a much larger one covers at 1.0 in that direction.
    """
    table: list[dict] = []
    for tolerance in tolerances:
        row = {"tolerance": tolerance}
        if a_distances:
            row["a_to_b"] = sum(1 for value in a_distances if value <= tolerance) / len(a_distances)
        if b_distances:
            row["b_to_a"] = sum(1 for value in b_distances if value <= tolerance) / len(b_distances)
        table.append(row)
    return table


def compare_surface_sets(
    a: SurfaceSet,
    b: SurfaceSet,
    *,
    samples: int,
    seed: int,
    tolerances: Sequence[float],
    max_list: int,
    a_measure: Callable[[list[Point3]], tuple[Sequence[float], Sequence[int | None]]],
    b_measure: Callable[[list[Point3]], tuple[Sequence[float], Sequence[int | None]]],
) -> tuple[dict, dict[str, DirectionSamples]]:
    """Two-way sampled distance between two triangle sets.

    ``a_to_b`` samples A and measures to B, ``b_to_a`` the reverse; both are
    needed because one direction alone misses a surface that is missing a
    region -- every sample of the smaller side can sit on the larger one.

    Returns ``(report, samples)``: the receipt block, and both directions'
    measurements, which the caller can turn into a dump.
    """
    # Different seeds per direction: the same seed would draw the same
    # stratification on both sides, correlating the two answers on meshes that
    # happen to share a triangle order.
    a_samples = measure_direction(a, b_measure, samples=samples, seed=seed)
    b_samples = measure_direction(b, a_measure, samples=samples, seed=seed + 1)

    a_stats = direction_stats(a_samples, max_list)
    b_stats = direction_stats(b_samples, max_list)
    report = {
        "a_to_b": a_stats,
        "b_to_a": b_stats,
        "max_both": max(
            a_stats.get("max", 0.0) if a_samples.distances else 0.0,
            b_stats.get("max", 0.0) if b_samples.distances else 0.0,
        ),
        "coverage": coverage_table(a_samples.distances, b_samples.distances, tolerances),
    }
    return report, {"a_to_b": a_samples, "b_to_a": b_samples}


###### 2D overlap: convex clipping, and a grid to avoid testing every pair


def convex_clip_area(subject: Sequence[Point2], clipper: Sequence[Point2]) -> float:
    """Area of the intersection of two convex polygons, both counter-clockwise.

    Sutherland-Hodgman: clip the subject by each edge of the clipper in turn.
    Exact for the convex-convex case, which is every triangle pair, so an
    overlap is reported with the area it really has rather than a yes or no.
    """
    output = list(subject)
    for index in range(len(clipper)):
        if not output:
            return 0.0
        a = clipper[index]
        b = clipper[(index + 1) % len(clipper)]
        ex, ey = b[0] - a[0], b[1] - a[1]
        clipped: list[Point2] = []
        previous = output[-1]
        previous_side = ex * (previous[1] - a[1]) - ey * (previous[0] - a[0])
        for current in output:
            current_side = ex * (current[1] - a[1]) - ey * (current[0] - a[0])
            if current_side >= 0.0:
                if previous_side < 0.0:
                    clipped.append(_edge_crossing(previous, current, previous_side, current_side))
                clipped.append(current)
            elif previous_side >= 0.0:
                clipped.append(_edge_crossing(previous, current, previous_side, current_side))
            previous, previous_side = current, current_side
        output = clipped
    if len(output) < 3:
        return 0.0
    return abs(signed_area_2d(output))


def _edge_crossing(p: Point2, q: Point2, dp: float, dq: float) -> Point2:
    denom = dp - dq
    t = dp / denom if denom != 0.0 else 0.0
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def triangle_overlap_area(t1: Triangle2, t2: Triangle2) -> float:
    """Overlapping area of two 2D triangles, winding of either irrelevant.

    Two triangles that merely share an edge or a corner intersect in exactly
    zero area, so one area tolerance covers both "touching" and "overlapping"
    without a special case for adjacency -- and a fold that happens to share an
    edge is still reported.
    """
    a = t1 if signed_triangle_area_2d(*t1) >= 0.0 else (t1[0], t1[2], t1[1])
    b = t2 if signed_triangle_area_2d(*t2) >= 0.0 else (t2[0], t2[2], t2[1])
    return convex_clip_area(a, b)


GRID2D_MAX_CELLS_PER_AXIS = 512
GRID2D_MAX_INSERTS = 4_000_000


def overlap_candidate_pairs(
    boxes: Sequence[tuple[float, float, float, float]], max_pairs: int
) -> tuple[list[tuple[int, int]], bool]:
    """Index pairs whose bounding boxes share a grid cell, and a budget flag.

    Broadphase only: a pair here still has to pass the exact area test. The
    flag is True when the budget stopped the search, which the caller must
    report as an incomplete check rather than as a clean sheet.
    """
    if len(boxes) < 2:
        return [], False

    lo_x = min(box[0] for box in boxes)
    lo_y = min(box[1] for box in boxes)
    hi_x = max(box[2] for box in boxes)
    hi_y = max(box[3] for box in boxes)
    extent = max(hi_x - lo_x, hi_y - lo_y)
    if not math.isfinite(extent) or extent <= 0.0:
        extent = 1.0
    mean_extent = sum((box[2] - box[0]) + (box[3] - box[1]) for box in boxes) / (2.0 * len(boxes))
    cell = max(mean_extent, extent / GRID2D_MAX_CELLS_PER_AXIS)

    for _attempt in range(10):
        buckets: dict[tuple[int, int], list[int]] = {}
        inserts = 0
        overflowed = False
        for index, box in enumerate(boxes):
            i0 = int((box[0] - lo_x) / cell)
            i1 = int((box[2] - lo_x) / cell)
            j0 = int((box[1] - lo_y) / cell)
            j1 = int((box[3] - lo_y) / cell)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    buckets.setdefault((i, j), []).append(index)
                    inserts += 1
            if inserts > GRID2D_MAX_INSERTS:
                overflowed = True
                break
        if overflowed:
            cell *= 2.0
            continue

        pairs: set[tuple[int, int]] = set()
        for members in buckets.values():
            if len(members) < 2:
                continue
            for position, first in enumerate(members):
                box_a = boxes[first]
                for second in members[position + 1 :]:
                    box_b = boxes[second]
                    # The cell only says the boxes are near; this says they touch.
                    if (
                        box_a[0] > box_b[2]
                        or box_b[0] > box_a[2]
                        or box_a[1] > box_b[3]
                        or box_b[1] > box_a[3]
                    ):
                        continue
                    pairs.add((first, second) if first < second else (second, first))
                    if len(pairs) > max_pairs:
                        return sorted(pairs), True
        return sorted(pairs), False

    # Every attempt overflowed: the boxes are too unevenly sized to grid.
    return [], True


###### UV analysis
#
# Every triangle below is a triangle of the *3D* triangulation, carrying the
# UV its own corner had. That is the only way a fold shows up: choosing the UV
# diagonal independently can lay a folded quad out flat and report it clean.


@dataclass
class UVTriangle:
    """One triangulated corner triple, in UV and in the geometry it came from."""

    prim: int
    face_index: int
    uv: Triangle2
    world: Triangle3


def uv_island_labels(
    faces: Sequence[tuple[int, Sequence[int]]],
    corner_uvs: Sequence[Sequence[Point2]],
    *,
    seam_tolerance: float = 1e-6,
) -> tuple[list[int], int]:
    """Group faces into UV islands; returns a label per face and the count.

    Two faces are in the same island when they share a topological edge *and*
    both carry the same UV at each end of it. That second half is what makes
    this a UV island rather than a mesh shell: a cut runs along edges that are
    still shared in 3D.
    """
    count = len(faces)
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Edge -> the (face, uv at the low point, uv at the high point) using it.
    edges: dict[tuple[int, int], list[tuple[int, Point2, Point2]]] = {}
    for face_index, (_prim_id, point_ids) in enumerate(faces):
        uvs = corner_uvs[face_index]
        side_count = len(point_ids)
        if side_count < 2 or len(uvs) < side_count:
            continue
        for corner in range(side_count):
            a = point_ids[corner - 1]
            b = point_ids[corner]
            uv_a = uvs[corner - 1]
            uv_b = uvs[corner]
            key = (a, b) if a < b else (b, a)
            record = (face_index, uv_a, uv_b) if a < b else (face_index, uv_b, uv_a)
            edges.setdefault(key, []).append(record)

    for users in edges.values():
        if len(users) < 2:
            continue
        for i in range(len(users)):
            face_i, low_i, high_i = users[i]
            for j in range(i + 1, len(users)):
                face_j, low_j, high_j = users[j]
                if _uv_close(low_i, low_j, seam_tolerance) and _uv_close(
                    high_i, high_j, seam_tolerance
                ):
                    union(face_i, face_j)

    roots: dict[int, int] = {}
    labels: list[int] = []
    for index in range(count):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return labels, len(roots)


def _uv_close(a: Point2, b: Point2, tolerance: float) -> bool:
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def triangles_are_stacked(a: Triangle2, b: Triangle2, tolerance: float) -> bool:
    """Do these two UV triangles occupy the same three corners?

    The only relation ``allow_stacking`` may forgive. Deliberately strict: two
    triangles that overlap in part -- the accidental case -- are never stacked,
    so turning stacking on cannot hide one.
    """
    remaining = list(b)
    for corner in a:
        for index, other in enumerate(remaining):
            if _uv_close(corner, other, tolerance):
                del remaining[index]
                break
        else:
            return False
    return True


def uv_overlaps(
    triangles: Sequence[UVTriangle],
    *,
    area_tolerance: float,
    stack_tolerance: float,
    max_pairs: int,
    max_records: int = 20_000,
) -> dict:
    """Positive-area overlaps between UV triangles, with a budget.

    Every pair with an intersection area above *area_tolerance* is counted,
    split into ``stacked`` (the same three corners within *stack_tolerance*)
    and ``overlapping`` (everything else, which is the accidental kind).

    Counts and areas are exact for every pair the broadphase offered. The
    returned lists hold the *largest* *max_records* of each kind: a layout
    mapped by accident can produce millions of overlapping pairs, and one dict
    each is how a report turns into an out-of-memory error.
    """
    boxes = [
        (
            min(tri.uv[0][0], tri.uv[1][0], tri.uv[2][0]),
            min(tri.uv[0][1], tri.uv[1][1], tri.uv[2][1]),
            max(tri.uv[0][0], tri.uv[1][0], tri.uv[2][0]),
            max(tri.uv[0][1], tri.uv[1][1], tri.uv[2][1]),
        )
        for tri in triangles
    ]
    candidates, truncated = overlap_candidate_pairs(boxes, max_pairs)

    overlapping = _LargestFirst(max_records)
    stacked = _LargestFirst(max_records)
    overlap_area = 0.0
    stacked_area = 0.0
    overlap_count = 0
    stacked_count = 0
    for first, second in candidates:
        a, b = triangles[first], triangles[second]
        area = triangle_overlap_area(a.uv, b.uv)
        if area <= area_tolerance:
            continue
        if triangles_are_stacked(a.uv, b.uv, stack_tolerance):
            stacked.offer(area, a.prim, b.prim)
            stacked_area += area
            stacked_count += 1
        else:
            overlapping.offer(area, a.prim, b.prim)
            overlap_area += area
            overlap_count += 1

    return {
        "overlapping": overlapping.records(),
        "overlapping_count": overlap_count,
        "overlap_area": overlap_area,
        "stacked": stacked.records(),
        "stacked_count": stacked_count,
        "stacked_area": stacked_area,
        "candidate_pairs": len(candidates),
        "truncated": truncated,
        "records_truncated": overlapping.dropped or stacked.dropped,
    }


class _LargestFirst:
    """The *limit* largest entries offered, kept without holding all of them."""

    def __init__(self, limit: int):
        self.limit = max(0, limit)
        self.heap: list[tuple[float, int, int, int]] = []
        self.dropped = False
        self._tie = 0

    def offer(self, area: float, prim_a: int, prim_b: int) -> None:
        if self.limit == 0:
            self.dropped = True
            return
        # The counter keeps equal areas from comparing prim ids, and makes ties
        # resolve by insertion order, so a run is reproducible.
        self._tie += 1
        entry = (area, -self._tie, prim_a, prim_b)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, entry)
        elif entry > self.heap[0]:
            heapq.heapreplace(self.heap, entry)
            self.dropped = True
        else:
            self.dropped = True

    def records(self) -> list[dict]:
        return [
            {"prims": [prim_a, prim_b], "area": area}
            for area, _tie, prim_a, prim_b in sorted(self.heap, reverse=True)
        ]


###### UV distortion
#
# Two different faults, deliberately reported apart:
#
#   density variation  the same shape given more or less texture than its
#                      neighbours -- texels per unit, an area ratio;
#   anisotropy         the shape itself squashed in one direction, which an
#                      area ratio cannot see at all. A unit right triangle
#                      mapped to UV (0,0), (100,0), (0,0.01) keeps its area
#                      exactly and is stretched ten thousand to one.
#
# Both fall out of the singular values of the triangle's own affine map.


def uv_jacobian_singular_values(world: Triangle3, uv: Triangle2) -> tuple[float, float] | None:
    """Singular values (larger, smaller) of the map from surface to UV.

    ``None`` when the world triangle has no area, so there is no map to take
    apart. Otherwise ``sigma1 * sigma2`` is the UV area over the world area
    (the density, squared) and ``sigma1 / sigma2`` is the anisotropy, with
    1.0 meaning the mapping is locally a rotation and a uniform scale.
    """
    e1 = sub3(world[1], world[0])
    e2 = sub3(world[2], world[0])
    normal = cross3(e1, e2)
    scale = length3(normal)
    if scale == 0.0:
        return None

    # An orthonormal frame in the triangle's own plane, so the world triangle
    # becomes two 2D vectors and the map between the two is a 2x2 matrix.
    length_e1 = length3(e1)
    bx = (e1[0] / length_e1, e1[1] / length_e1, e1[2] / length_e1)
    bz = (normal[0] / scale, normal[1] / scale, normal[2] / scale)
    by = cross3(bz, bx)
    a00, a10 = length_e1, 0.0
    a01, a11 = dot3(e2, bx), dot3(e2, by)
    det = a00 * a11 - a01 * a10
    if det == 0.0:
        return None

    b00, b10 = uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]
    b01, b11 = uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]

    # J = B * inverse(A)
    i00, i01 = a11 / det, -a01 / det
    i10, i11 = -a10 / det, a00 / det
    j00 = b00 * i00 + b01 * i10
    j01 = b00 * i01 + b01 * i11
    j10 = b10 * i00 + b11 * i10
    j11 = b10 * i01 + b11 * i11

    # The larger singular value from the larger eigenvalue of J^T J, which is
    # symmetric 2x2 -- that subtraction is between well separated numbers.
    m00 = j00 * j00 + j10 * j10
    m01 = j00 * j01 + j10 * j11
    m11 = j01 * j01 + j11 * j11
    half_trace = 0.5 * (m00 + m11)
    gap = math.sqrt(max(0.0, half_trace * half_trace - (m00 * m11 - m01 * m01)))
    sigma1 = math.sqrt(max(0.0, half_trace + gap))
    if sigma1 <= 0.0:
        return (0.0, 0.0)
    # The smaller one from the determinant, never from half_trace - gap: at ten
    # thousand to one those two agree to eleven digits and their difference is
    # noise, which would report a perfectly good stretched triangle as
    # collapsed. sigma1 * sigma2 is |det J| exactly, so this division is stable.
    return (sigma1, abs(j00 * j11 - j01 * j10) / sigma1)


def uv_distortion_stats(triangles: Sequence[UVTriangle], *, threshold: float | None = None) -> dict:
    """UV area, winding, density variation and anisotropy.

    ``uv_area`` is the **sum** of the triangles' UV areas, not the union: where
    two triangles are stacked, the area under them is counted twice. That is
    the honest number for how much UV space was authored; a union would need a
    full boolean and is not computed.

    ``density`` is ``sqrt(uv area / world area)`` per triangle -- UV units per
    world unit -- reported as a ratio to the area-weighted mean, so 1.0 is the
    density of this mesh as a whole. ``anisotropy`` is the ratio of the two
    singular values, 1.0 for a mapping that is locally a rotation and a scale;
    it is the one that catches a shape squashed along one axis without any
    change in area.

    Means and percentiles are weighted by world area and carry that in their
    names; ``ratio_min``, ``ratio_max`` and ``max`` are plain extremes over the
    triangles, since an extreme has no weighting to do.

    A triangle whose UV corners collapse to a line or a point has no density
    and no anisotropy -- both are reported as collapsed rather than as a
    number, and never taken the logarithm of.
    """
    uv_area = 0.0
    world_area = 0.0
    mirrored = 0
    collapsed: list[int] = []
    degenerate_world = 0
    densities: list[tuple[float, float]] = []  # (density, world area weight)
    anisotropies: list[tuple[float, float]] = []

    for tri in triangles:
        signed = signed_triangle_area_2d(*tri.uv)
        uv_area += abs(signed)
        if signed < 0.0:
            mirrored += 1
        surface = triangle_area_3d(*tri.world)
        world_area += surface
        singular = uv_jacobian_singular_values(tri.world, tri.uv) if surface > 0.0 else None
        if singular is None:
            degenerate_world += 1
            continue
        sigma1, sigma2 = singular
        if sigma2 <= 0.0:
            # Flattened onto a line, or onto a point: zero UV area either way.
            collapsed.append(tri.prim)
            continue
        densities.append((math.sqrt(sigma1 * sigma2), surface))
        anisotropies.append((sigma1 / sigma2, surface))

    stats: dict = {
        "uv_area": uv_area,
        "world_area": world_area,
        "mirrored_triangles": mirrored,
        "measured_triangles": len(densities),
    }
    if collapsed:
        # The prims to name, and how many triangles collapsed: a face can lose
        # one triangle and keep another, and both numbers matter.
        stats["collapsed_uv"] = sorted(set(collapsed))
        stats["collapsed_uv_count"] = len(collapsed)
    if degenerate_world:
        stats["degenerate_world_triangles"] = degenerate_world
    if not densities:
        return stats

    # Every figure below is weighted by world area, and says so in its name:
    # a count-based percentile would move when the same surface is
    # retessellated, which is not a property an acceptance number may have.
    weight = sum(area for _value, area in densities)
    mean_density = sum(value * area for value, area in densities) / weight
    if mean_density > 0.0:
        ratios = [(value / mean_density, area) for value, area in densities]
        values = sorted(ratio for ratio, _area in ratios)
        density: dict = {
            "area_weighted_mean": mean_density,
            "ratio_min": values[0],
            "ratio_p95_area_weighted": weighted_percentile(ratios, 0.95),
            "ratio_max": values[-1],
            "abs_log2_area_weighted_mean": sum(
                abs(math.log2(ratio)) * area for ratio, area in ratios
            )
            / weight,
        }
        if threshold is not None:
            low, high = 1.0 / threshold, threshold
            outside = [(ratio, area) for ratio, area in ratios if not low <= ratio <= high]
            density["outside_threshold"] = len(outside)
            density["outside_threshold_area"] = sum(area for _ratio, area in outside)
        stats["density"] = density

    values = sorted(value for value, _area in anisotropies)
    anisotropy: dict = {
        "area_weighted_mean": sum(value * area for value, area in anisotropies) / weight,
        "p95_area_weighted": weighted_percentile(anisotropies, 0.95),
        "max": values[-1],
    }
    if collapsed:
        anisotropy["undefined"] = len(collapsed)
    if threshold is not None:
        over = [(value, area) for value, area in anisotropies if value > threshold]
        anisotropy["over_threshold"] = len(over)
        anisotropy["over_threshold_area"] = sum(area for _value, area in over)
    stats["anisotropy"] = anisotropy
    return stats


def udim_tiles(triangles: Sequence[UVTriangle]) -> list[int]:
    """UDIM tile numbers the triangles sit in, in order.

    The usual 1001 + u + 10 * v numbering, taken at each triangle's centre
    rather than at its corners: a layout that fills tile 1001 exactly has
    corners at u = 1 and v = 1, which by corner would report four tiles for a
    mesh that occupies one. Context for the area numbers, not a claim about
    which tiles are full.
    """
    tiles: set[int] = set()
    for tri in triangles:
        u = (tri.uv[0][0] + tri.uv[1][0] + tri.uv[2][0]) / 3.0
        v = (tri.uv[0][1] + tri.uv[1][1] + tri.uv[2][1]) / 3.0
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        tiles.add(1001 + int(math.floor(u)) + 10 * int(math.floor(v)))
    return sorted(tiles)
