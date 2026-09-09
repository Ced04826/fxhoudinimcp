"""Modeling (SOP) handlers for FXHoudini-MCP.

``modeling.get_mesh_report``
    Cage acceptance: face mix, pieces, boundary, non-manifold, degenerates,
    poles, folded quads. Topology from a ``.geo`` JSON round trip.

``modeling.edit_points``
    Batch point shaping through a native Edit SOP, with read-back evidence.

``modeling.compare_geometry``
    Topology, positions, and optional ``sourcept`` / ``sourceprim`` provenance.

The maths and validation below are deliberately free of ``hou`` and ``numpy``
so they can be tested without Houdini; only the handlers touch either.
"""

from __future__ import annotations

# Built-in
import array
import contextlib
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from typing import Any

# Third-party
import hou

# Internal
from fxhoudinimcp_server.dispatcher import register_handler
from fxhoudinimcp_server.errors import as_int, as_text

###### Limits

# Above this the .geo round trip and the Python passes stop being interactive,
# and a report that takes a minute is a report nobody runs.
MAX_PRIMS = 500_000

# A face whose area falls below this fraction of the bounding box area is a
# sliver a subdivision will turn into a crease or a hole.
AREA_EPSILON_FACTOR = 1e-12

# An Edit SOP stores its delta as float32, so a requested position and the one
# that comes back never match exactly. Scaled by the bounding box so the same
# number works for a 1-unit prop and a 1000-unit building; anything outside it
# means the edit did not land where it was asked to, which is worth saying.
POSITION_TOLERANCE_FACTOR = 1e-5

# Below this a point did not move; float32 noise alone reaches roughly 1e-7.
MOVED_EPSILON = 1e-7

# Where the input fingerprint is parked on the Edit node, and how the matching
# comment line starts so a later run can replace it instead of stacking up.
USER_DATA_KEY = "fxmcp_edit_fingerprint"
COMMENT_MARKER = "[fxmcp edit]"
###### .geo JSON parsing (no hou)
#
# Layouts observed on Houdini 22.0.368, all verified live:
#
#   document   flat key/value list:
#              ['fileversion', ..., 'pointcount', 10201, 'vertexcount', 40000,
#               'primitivecount', 10000, 'topology', [...], 'attributes', [...],
#               'primitives', [...], 'primitivegroups', [...]]
#   topology   flat too: ['pointref', ['indices', [<point number per vertex>]]]
#   primitives a LIST OF ENTRIES, each [type_list, data_list], both flat
#              key/value lists. Entries appear in primitive-number order and
#              their nprimitives sum to primitivecount, which is what lets the
#              prim id be a running counter.
#
# Entry shapes seen in the wild:
#
#   closed polygons, uniform side count
#       [['type', 'Polygon_run'],
#        ['startvertex', 0, 'nprimitives', 10000, 'nvertices_rle', [4, 10000]]]
#   closed polygons, mixed side count (poly sphere: 24 tris, 240 quads, 24 tris)
#       ['startvertex', 0, 'nprimitives', 288, 'nvertices_rle', [3, 24, 4, 240, 3, 24]]
#   a single closed polygon (circle SOP, 12-gon) -- plain list, no RLE
#       ['startvertex', 0, 'nprimitives', 1, 'nvertices', [12]]
#   OPEN polygons are a separate type, not a 'closed' flag:
#       [['type', 'PolygonCurve_run'], ['startvertex', 16, 'nprimitives', 1, 'nvertices', [2]]]
#   everything else carries a single 'vertex' integer and its own payload:
#       [['type', 'Sphere'], ['vertex', 18, 'transform', [...]]]
#       [['type', 'Volume'], ['vertex', 0, 'transform', [...], 'res', [10, 10, 10], ...]]
#       [['type', 'PackedGeometry'], ['parameters', {...}, 'pivot', [...], 'vertex', 0, ...]]
#
# The single non-run 'Polygon' entry (a 'vertex' list plus a 'closed' flag) never
# appeared on this build -- polygons always came out as runs -- but it is part of
# the format, so it is handled rather than trusted not to exist.

_CLOSED_POLY_TYPES = ("Polygon", "Poly")
_OPEN_POLY_TYPES = ("PolygonCurve", "PolyCurve")


def _flat_pairs(items: Any) -> dict[str, Any]:
    """Read one of the format's flat ``[key, value, key, value]`` lists."""
    if not isinstance(items, (list, tuple)):
        return {}
    # strict=False on purpose: a trailing key with no value means a document we
    # do not understand, and dropping it beats refusing to read the rest.
    return dict(zip(items[0::2], items[1::2], strict=False))


def _run_vertex_counts(data: dict[str, Any], nprims: int) -> list[int]:
    """Vertices per primitive for a ``*_run`` entry, RLE or plain."""
    plain = data.get("nvertices")
    if isinstance(plain, (list, tuple)):
        return [int(n) for n in plain]

    rle = data.get("nvertices_rle")
    counts: list[int] = []
    if isinstance(rle, (list, tuple)):
        for i in range(0, len(rle) - 1, 2):
            counts.extend([int(rle[i])] * int(rle[i + 1]))
    if len(counts) < nprims:
        # A run we cannot size is a run we must not guess at.
        raise ValueError(
            f"primitive run declares {nprims} primitives but lists sizes for {len(counts)}"
        )
    return counts[:nprims]


def extract_faces(document: list) -> dict[str, Any]:
    """Split a parsed ``.geo`` document into faces and everything else.

    Returns ``faces`` as ``(prim_id, [point ids])`` for closed polygons only,
    plus the prim ids of open polylines and of primitives that are not polygons
    at all, so the caller can report them without pretending they are faces.
    """
    doc = _flat_pairs(document)
    topology = _flat_pairs(doc.get("topology"))
    pointref = _flat_pairs(topology.get("pointref"))
    indices = pointref.get("indices") or []

    faces: list[tuple[int, list[int]]] = []
    open_polylines: list[int] = []
    other_prims: list[int] = []

    prim_id = 0
    for entry in doc.get("primitives") or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        type_name = str(_flat_pairs(entry[0]).get("type", ""))
        data = _flat_pairs(entry[1])

        if type_name.endswith("_run"):
            base = type_name[: -len("_run")]
            nprims = int(data.get("nprimitives", 0))
            if base in _CLOSED_POLY_TYPES or base in _OPEN_POLY_TYPES:
                offset = int(data.get("startvertex", 0))
                sizes = _run_vertex_counts(data, nprims)
                closed = base in _CLOSED_POLY_TYPES
                for size in sizes:
                    if closed:
                        faces.append((prim_id, list(indices[offset : offset + size])))
                    else:
                        open_polylines.append(prim_id)
                    offset += size
                    prim_id += 1
            else:
                other_prims.extend(range(prim_id, prim_id + nprims))
                prim_id += nprims
            continue

        # Single-primitive entry.
        vertex = data.get("vertex")
        if type_name in _CLOSED_POLY_TYPES or type_name in _OPEN_POLY_TYPES:
            closed = bool(data.get("closed", type_name in _CLOSED_POLY_TYPES))
            points = [indices[int(v)] for v in vertex] if isinstance(vertex, (list, tuple)) else []
            if closed and points:
                faces.append((prim_id, points))
            else:
                open_polylines.append(prim_id)
        else:
            other_prims.append(prim_id)
        prim_id += 1

    return {
        "faces": faces,
        "open_polylines": open_polylines,
        "other_prims": other_prims,
        "prim_count": prim_id,
        "point_count": int(doc.get("pointcount", 0)),
        "declared_prim_count": int(doc.get("primitivecount", prim_id)),
    }


###### Topology maths (no hou, no numpy -- unit-tested without Houdini)


def build_edge_faces(faces: list[tuple[int, list[int]]]) -> dict[tuple[int, int], list[int]]:
    """Map each undirected edge to the indices (into *faces*) that use it.

    The hot loop of the whole report: four iterations per quad, so 40 000 on a
    100x100 grid. Walking from the last point rather than indexing modulo the
    side count is what keeps that under a tenth of a second.
    """
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for i, (_prim_id, points) in enumerate(faces):
        if not points:
            continue
        a = points[-1]
        for b in points:
            key = (a, b) if a < b else (b, a)
            bucket = edge_faces.get(key)
            if bucket is None:
                edge_faces[key] = [i]
            else:
                bucket.append(i)
            a = b
    return edge_faces


def connected_pieces(
    face_count: int, edge_faces: dict[tuple[int, int], list[int]]
) -> tuple[int, list[int]]:
    """Number of edge-connected shells and their face counts, largest first."""
    if face_count <= 0:
        return 0, []

    parent = list(range(face_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for bucket in edge_faces.values():
        if len(bucket) < 2:
            continue
        root = find(bucket[0])
        for other in bucket[1:]:
            other_root = find(other)
            if other_root != root:
                parent[other_root] = root

    sizes: dict[int, int] = {}
    for i in range(face_count):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    return len(sizes), sorted(sizes.values(), reverse=True)


def boundary_report(
    edge_faces: dict[tuple[int, int], list[int]],
) -> tuple[list[tuple[int, int]], list[int], int]:
    """Boundary edges, closed loop sizes (largest first) and open chain count.

    A boundary edge carries exactly one face. Walking the graph they form gives
    one component per hole; a component where every point has exactly two
    boundary edges is a closed loop, anything else is a chain and is reported
    separately rather than counted as a loop it is not.
    """
    boundary = [edge for edge, faces in edge_faces.items() if len(faces) == 1]

    adjacency: dict[int, list[int]] = {}
    for a, b in boundary:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    seen: set[int] = set()
    loops: list[int] = []
    chains = 0
    for start in adjacency:
        if start in seen:
            continue
        component: set[int] = set()
        stack = [start]
        while stack:
            point = stack.pop()
            if point in component:
                continue
            component.add(point)
            for neighbour in adjacency[point]:
                if neighbour not in component:
                    stack.append(neighbour)
        seen |= component
        edge_count = sum(len(adjacency[p]) for p in component) // 2
        if all(len(adjacency[p]) == 2 for p in component):
            loops.append(edge_count)
        else:
            chains += 1
    return boundary, sorted(loops, reverse=True), chains


def valence_map(edge_faces: dict[tuple[int, int], list[int]]) -> dict[int, int]:
    """Distinct edges touching each point."""
    valence: dict[int, int] = {}
    for a, b in edge_faces:
        valence[a] = valence.get(a, 0) + 1
        valence[b] = valence.get(b, 0) + 1
    return valence


def find_poles(
    valence: dict[int, int], boundary_edges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Interior points whose valence is not 4, as ``(point, valence)``.

    Boundary points are excluded: a valence-3 point on the edge of an open
    sheet is what an open sheet looks like, not a defect.
    """
    on_boundary: set[int] = set()
    for a, b in boundary_edges:
        on_boundary.add(a)
        on_boundary.add(b)
    return sorted(
        (point, count)
        for point, count in valence.items()
        if count != 4 and point not in on_boundary
    )


def polygon_area(points: list[int], positions: list) -> float:
    """Area of a planar-ish polygon by Newell's method."""
    nx = ny = nz = 0.0
    p = positions[points[-1]]
    for index in points:
        q = positions[index]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
        p = q
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def degenerate_faces(
    faces: list[tuple[int, list[int]]], positions: list, area_epsilon: float
) -> list[int]:
    """Prim ids of faces that repeat a point or enclose no meaningful area."""
    bad: list[int] = []
    for prim_id, points in faces:
        if len(points) < 3 or len(set(points)) != len(points):
            bad.append(prim_id)
            continue
        if polygon_area(points, positions) < area_epsilon:
            bad.append(prim_id)
    return bad


def quad_fold_flags(p0, p1, p2, p3) -> tuple[bool, bool]:
    """Is this quad folded across diagonal 0-2, and across diagonal 1-3?

    Each diagonal splits the quad into two triangles; if their normals oppose,
    the quad is a bow tie or a crumpled face when triangulated that way. A quad
    can be sound on one diagonal and folded on the other, so both are reported.
    """
    a0, a1, a2 = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    b0, b1, b2 = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    c0, c1, c2 = p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]
    n1 = (a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0)
    n2 = (b1 * c2 - b2 * c1, b2 * c0 - b0 * c2, b0 * c1 - b1 * c0)
    folded02 = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2] < 0.0

    d0, d1, d2 = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    e0, e1, e2 = p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]
    f0, f1, f2 = p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]
    m1 = (d1 * e2 - d2 * e1, d2 * e0 - d0 * e2, d0 * e1 - d1 * e0)
    m2 = (e1 * f2 - e2 * f1, e2 * f0 - e0 * f2, e0 * f1 - e1 * f0)
    folded13 = m1[0] * m2[0] + m1[1] * m2[1] + m1[2] * m2[2] < 0.0

    return folded02, folded13


def folded_quads(
    faces: list[tuple[int, list[int]]], positions: list
) -> tuple[list[int], list[int]]:
    """Prim ids of quads folded across diagonal 0-2 and across diagonal 1-3."""
    diag02: list[int] = []
    diag13: list[int] = []
    for prim_id, points in faces:
        if len(points) != 4:
            continue
        f02, f13 = quad_fold_flags(
            positions[points[0]],
            positions[points[1]],
            positions[points[2]],
            positions[points[3]],
        )
        if f02:
            diag02.append(prim_id)
        if f13:
            diag13.append(prim_id)
    return diag02, diag13


###### modeling.get_mesh_report


def _get_sop_node(node_path: str, argument: str) -> hou.Node:
    """A node that can hand out geometry, or a sentence naming what it is instead.

    The ``geometry`` attribute is checked rather than called blind: on anything
    that is not a SOP -- ``/obj`` itself, an object node, a LOP -- calling it
    raised ``'OpNode' object has no attribute 'geometry'``, which names a Python
    internal instead of telling the caller they pointed at the wrong node.
    """
    node = hou.node(node_path)
    if node is None:
        raise hou.OperationFailed(f"Node not found: {node_path}")
    if getattr(node, "geometry", None) is None:
        raise hou.OperationFailed(
            f"{node_path} is a '{node.type().name()}' node, which carries no geometry. "
            f"{argument} must name a SOP, for example /obj/geo1/subdivide1."
        )
    # A geo object also has geometry() -- it returns the display SOP -- so the
    # attribute check above is not enough to refuse /obj/geo1 itself.
    if node.type().category() != hou.sopNodeTypeCategory():
        raise hou.OperationFailed(
            f"{node_path} is a '{node.type().name()}' node in the "
            f"{node.type().category().name()} context, not a SOP. "
            f"{argument} must name a SOP, for example /obj/geo1/subdivide1."
        )
    return node


def _get_sop_geo(node_path: str) -> hou.Geometry:
    """The cooked read-only geometry for a SOP node."""
    geo = _get_sop_node(node_path, "node_path").geometry()
    if geo is None:
        raise hou.OperationFailed(
            f"Node has no geometry: {node_path}. Only SOP nodes carry a mesh."
        )
    return geo


def _load_geo_document(geo: hou.Geometry) -> list:
    """Save the geometry to a temporary .geo and read it back as JSON."""
    handle, path = tempfile.mkstemp(prefix="fxmcp-mesh-", suffix=".geo", dir=tempfile.gettempdir())
    os.close(handle)
    try:
        geo.saveToFile(path)
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def _resolve_scope(
    geo: hou.Geometry,
    node_path: str,
    group: str,
    faces: list[tuple[int, list[int]]],
) -> tuple[set[int], str]:
    """Prim ids covered by *group*, resolved as a prim group then a point group.

    A prim group scopes every primitive it names. A point group scopes the faces
    that use any of its points; polylines and non-polygon primitives fall out of
    scope there, because the .geo round trip does not give them point ids.
    """
    prim_group = geo.findPrimGroup(group)
    if prim_group is not None:
        return {prim.number() for prim in prim_group.prims()}, "prim group"

    point_group = geo.findPointGroup(group)
    if point_group is not None:
        members = {point.number() for point in point_group.points()}
        return {
            prim_id for prim_id, points in faces if any(p in members for p in points)
        }, "point group"

    prim_names = sorted(g.name() for g in geo.primGroups())
    point_names = sorted(g.name() for g in geo.pointGroups())
    if not prim_names and not point_names:
        raise hou.OperationFailed(
            f"No group named '{group}' on {node_path}: this geometry has no groups at all."
        )
    raise hou.OperationFailed(
        f"No group named '{group}' on {node_path}. "
        f"Prim groups: {prim_names or 'none'}. Point groups: {point_names or 'none'}."
    )


def _write_dump(path: str, payload: dict[str, Any]) -> None:
    """Write the uncapped lists next to the capped receipt."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def _build_mesh_report(
    *,
    node_path: str,
    group: str,
    max_list: int,
    dump_target: str,
) -> dict[str, Any]:
    """Everything in the report except how long it took."""
    node_path = as_text(node_path, "node_path").strip()
    if not node_path:
        raise ValueError("node_path must name a SOP node, for example /obj/geo1/subdivide1")
    group_name = group.strip()

    geo = _get_sop_geo(node_path)

    total_prims = geo.intrinsicValue("primitivecount")
    if total_prims > MAX_PRIMS:
        raise hou.OperationFailed(
            f"{node_path} has {total_prims} primitives, more than the {MAX_PRIMS} this "
            "report can check at once. Pass a group to check one region at a time."
        )

    parsed = extract_faces(_load_geo_document(geo))
    faces = parsed["faces"]
    open_polylines = parsed["open_polylines"]
    other_prims = parsed["other_prims"]

    scope_label = "all"
    if group_name:
        scope, kind = _resolve_scope(geo, node_path, group_name, faces)
        scope_label = f"{group_name} ({kind})"
        faces = [face for face in faces if face[0] in scope]
        open_polylines = [prim_id for prim_id in open_polylines if prim_id in scope]
        other_prims = [prim_id for prim_id in other_prims if prim_id in scope]

    # Positions: one bulk read, reshaped by numpy. numpy ships with Houdini's
    # Python; the pure-Python helpers above take the .tolist() form, which is
    # also faster to index in the per-face loops than numpy rows are.
    import numpy

    flat = geo.pointFloatAttribValues("P")
    coordinates = numpy.asarray(flat, dtype=numpy.float64).reshape(-1, 3)
    positions = coordinates.tolist()

    used_points: set[int] = set()
    for _prim_id, points in faces:
        used_points.update(points)

    if len(coordinates) and used_points:
        subset = coordinates[sorted(used_points)]
        bbox_min = subset.min(axis=0).tolist()
        bbox_max = subset.max(axis=0).tolist()
    elif len(coordinates):
        bbox_min = coordinates.min(axis=0).tolist()
        bbox_max = coordinates.max(axis=0).tolist()
    else:
        bbox_min = bbox_max = [0.0, 0.0, 0.0]
    size = [bbox_max[i] - bbox_min[i] for i in range(3)]
    diagonal = math.sqrt(sum(component * component for component in size))

    edge_faces = build_edge_faces(faces)
    piece_count, piece_sizes = connected_pieces(len(faces), edge_faces)
    boundary_edges, loop_sizes, open_chains = boundary_report(edge_faces)
    nonmanifold = sorted(edge for edge, users in edge_faces.items() if len(users) > 2)
    valence = valence_map(edge_faces)
    poles = find_poles(valence, boundary_edges)
    bad_faces = degenerate_faces(faces, positions, AREA_EPSILON_FACTOR * diagonal * diagonal)
    diag02, diag13 = folded_quads(faces, positions)
    folded = sorted(set(diag02) | set(diag13))

    sides: dict[int, int] = {}
    for _prim_id, points in faces:
        count = len(points)
        sides[count] = sides.get(count, 0) + 1

    histogram: dict[str, int] = {}
    for count in valence.values():
        key = str(count)
        histogram[key] = histogram.get(key, 0) + 1

    counts: dict[str, Any] = {
        "points": len(used_points),
        "prims": len(faces) + len(open_polylines) + len(other_prims),
        "polygons": len(faces),
        "quads": sides.get(4, 0),
        "tris": sides.get(3, 0),
        "ngons": sum(n for side, n in sides.items() if side > 4),
        "other_prims": len(other_prims),
        "open_polylines": len(open_polylines),
    }
    loose = parsed["point_count"] - len(used_points)
    if not group_name and loose > 0:
        counts["loose_points"] = loose

    # Every list below is omitted when it would be empty, and when max_list is 0:
    # a key whose value is [] costs the reader a decision and tells them nothing.
    boundary: dict[str, Any] = {"edges": len(boundary_edges), "loops": len(loop_sizes)}
    if loop_sizes and max_list:
        boundary["loop_sizes"] = loop_sizes[:max_list]
    if open_chains:
        boundary["open_chains"] = open_chains

    nonmanifold_report: dict[str, Any] = {"count": len(nonmanifold)}
    if nonmanifold and max_list:
        nonmanifold_report["edges"] = [list(edge) for edge in nonmanifold[:max_list]]

    degenerate_report: dict[str, Any] = {"count": len(bad_faces)}
    if bad_faces and max_list:
        degenerate_report["prims"] = bad_faces[:max_list]

    poles_report: dict[str, Any] = {"count": len(poles)}
    if poles:
        by_valence: dict[str, int] = {}
        for _point, count in poles:
            key = str(count)
            by_valence[key] = by_valence.get(key, 0) + 1
        poles_report["by_valence"] = by_valence
        if max_list:
            poles_report["points"] = [
                {"point": point, "valence": count, "P": [round(v, 6) for v in positions[point]]}
                for point, count in poles[:max_list]
            ]

    folded_report: dict[str, Any] = {
        "count": len(folded),
        "diag02": len(diag02),
        "diag13": len(diag13),
    }
    if folded and max_list:
        folded_report["prims"] = folded[:max_list]

    report: dict[str, Any] = {
        "node_path": node_path,
        "scope": scope_label,
        "counts": counts,
        "pieces": {"count": piece_count, "largest": piece_sizes[:5]},
        "boundary": boundary,
        "nonmanifold_edges": nonmanifold_report,
        "degenerate": degenerate_report,
        "valence": histogram,
        "poles": poles_report,
        "folded_quads": folded_report,
        "bbox": {
            "min": [round(v, 6) for v in bbox_min],
            "max": [round(v, 6) for v in bbox_max],
            "size": [round(v, 6) for v in size],
            "diag": round(diagonal, 6),
        },
    }

    if dump_target:
        payload = {
            "node_path": node_path,
            "scope": scope_label,
            "piece_sizes": piece_sizes,
            "boundary_loop_sizes": loop_sizes,
            "nonmanifold_edges": [list(edge) for edge in nonmanifold],
            "degenerate_prims": bad_faces,
            "folded_quads": {"diag02": diag02, "diag13": diag13},
            "poles": [
                {"point": point, "valence": count, "P": positions[point]} for point, count in poles
            ],
            "valence": histogram,
        }
        try:
            _write_dump(dump_target, payload)
        except OSError as exc:
            # Say so rather than return a receipt that implies a file exists.
            report["dump_error"] = f"could not write {dump_target}: {exc}"
        else:
            report["dump"] = dump_target

    return report


def _get_mesh_report(
    *,
    node_path: str,
    group: str | None = None,
    max_list: int = 20,
    dump_path: str | None = None,
) -> dict[str, Any]:
    """Topology health of a polygon mesh: the check before accepting a cage.

    Counts are always reported; id lists appear only when something is wrong and
    are capped at *max_list*, so a clean mesh answers in a few hundred
    characters. Pass *dump_path* to receive every id in a JSON file.
    """
    started = time.perf_counter()

    # The analysis allocates a few hundred thousand short-lived tuples and lists,
    # which is exactly the pattern that makes the cyclic collector fire mid-pass.
    # Measured on the 10 000 quad grid, twelve runs each: 93-97 ms with spikes to
    # 188-212 ms when a collection lands in the edge pass, against a flat 88-94 ms
    # with the collector paused. Reference counting still frees everything as it
    # goes; only cycle detection waits, and it waits for under a tenth of a second.
    collecting = gc.isenabled()
    if collecting:
        gc.disable()
    try:
        report = _build_mesh_report(
            node_path=node_path,
            group=as_text(group, "group"),
            max_list=max(0, min(as_int(max_list, "max_list"), 1000)),
            dump_target=as_text(dump_path, "dump_path").strip(),
        )
    finally:
        if collecting:
            gc.enable()

    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return report


register_handler("modeling.get_mesh_report", _get_mesh_report)


###### Point editing: validation and arithmetic (no hou)


def _vector3(value: Any, label: str, key: str) -> tuple[float, float, float]:
    """Three finite numbers, or a sentence naming which entry and which axis."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} '{key}' must be three numbers [x, y, z], not {value!r}")
    components: list[float] = []
    for axis, component in zip("xyz", value, strict=True):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError(
                f"{label} '{key}' {axis} must be a number, not "
                f"{type(component).__name__}: {component!r}"
            )
        number = float(component)
        # A NaN or an infinity reaches the geometry intact and quietly ruins it.
        if not math.isfinite(number):
            raise ValueError(f"{label} '{key}' {axis} must be a finite number, not {component!r}")
        components.append(number)
    return (components[0], components[1], components[2])


def validate_moves(moves: Any, point_count: int, group_names: Any) -> list[dict[str, Any]]:
    """Check every entry before anything is written, and normalise it.

    Nothing here touches the scene, which is the point: a call that names one
    bad point number must not leave half its moves applied.
    """
    if not isinstance(moves, (list, tuple)):
        raise ValueError(
            "moves must be a list of entries like [{'point': 0, 'to': [0, 1, 0]}], "
            f"not {type(moves).__name__}"
        )
    if not moves:
        raise ValueError("moves is empty, so there is nothing to move")

    known_groups = set(group_names)
    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(moves):
        label = f"moves[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(
                f"{label} must be a dict like {{'point': 0, 'to': [0, 1, 0]}}, "
                f"not {type(raw).__name__}"
            )

        has_point = raw.get("point") is not None
        has_group = raw.get("group") is not None
        if has_point and has_group:
            raise ValueError(f"{label} names both 'point' and 'group'; give exactly one")
        if not has_point and not has_group:
            raise ValueError(f"{label} names neither 'point' nor 'group'; give exactly one")

        has_to = raw.get("to") is not None
        has_delta = raw.get("delta") is not None
        if has_to and has_delta:
            raise ValueError(f"{label} gives both 'to' and 'delta'; give exactly one")
        if not has_to and not has_delta:
            raise ValueError(f"{label} gives neither 'to' nor 'delta'; give exactly one")
        if has_group and has_to:
            raise ValueError(
                f"{label} uses 'to' with 'group': one absolute position for a whole group "
                "would stack every point on the same spot. Use 'delta'."
            )

        entry: dict[str, Any] = {
            "point": None,
            "group": None,
            "mode": "to" if has_to else "delta",
            "vector": _vector3(
                raw["to"] if has_to else raw["delta"], label, "to" if has_to else "delta"
            ),
        }

        if has_point:
            number = raw["point"]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(
                    f"{label} 'point' must be a whole number, not "
                    f"{type(number).__name__}: {number!r}"
                )
            if isinstance(number, float) and not number.is_integer():
                raise ValueError(f"{label} 'point' must be a whole number, not {number!r}")
            number = int(number)
            if not 0 <= number < point_count:
                raise ValueError(
                    f"{label} point {number} is out of range: the input geometry has "
                    f"{point_count} points, numbered 0 to {point_count - 1}"
                )
            entry["point"] = number
        else:
            group = raw["group"]
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"{label} 'group' must be a point group name, not {group!r}")
            group = group.strip()
            if group not in known_groups:
                available = sorted(known_groups)
                raise ValueError(
                    f"{label} names point group '{group}', which the input geometry does not "
                    f"have. Point groups: {available if available else 'none'}"
                )
            entry["group"] = group

        entries.append(entry)
    return entries


def apply_moves(
    positions: list[float],
    entries: list[dict[str, Any]],
    group_members: dict[str, list[int]],
) -> tuple[list[float], list[int], dict[int, tuple[float, float, float]]]:
    """Apply normalised entries to a flat xyz list.

    Returns the new positions, the touched points in the order they were first
    named, and the absolute position each ``to`` entry asked for -- carried
    forward through any later delta on the same point, so it stays comparable
    with what the Edit node reads back.
    """
    updated = list(positions)
    touched: dict[int, None] = {}
    requested: dict[int, tuple[float, float, float]] = {}

    for entry in entries:
        if entry["point"] is not None:
            points = [entry["point"]]
        else:
            points = group_members.get(entry["group"], [])
        vector = entry["vector"]

        for point in points:
            base = point * 3
            if entry["mode"] == "to":
                updated[base] = vector[0]
                updated[base + 1] = vector[1]
                updated[base + 2] = vector[2]
                requested[point] = vector
            else:
                updated[base] += vector[0]
                updated[base + 1] += vector[1]
                updated[base + 2] += vector[2]
                if point in requested:
                    was = requested[point]
                    requested[point] = (
                        was[0] + vector[0],
                        was[1] + vector[1],
                        was[2] + vector[2],
                    )
            touched[point] = None

    return updated, list(touched), requested


def geometry_fingerprint(point_count: int, position_bytes: bytes) -> str:
    """A short stamp of the geometry an edit was written against.

    Point count plus a digest of every position: the two ways upstream geometry
    can change under a stored delta that indexes points by number.
    """
    digest = hashlib.blake2b(position_bytes, digest_size=4).hexdigest()
    return f"{point_count}@{digest}"


def edit_comment_line(moved: int, max_displacement: float, fingerprint: str) -> str:
    """The one line the node shows in the network editor."""
    count, _, digest = fingerprint.partition("@")
    return (
        f"{COMMENT_MARKER} {moved} points moved, max {max_displacement:.4f}, "
        f"input {count} pts @{digest}"
    )


def replace_comment_line(comment: str, line: str, marker: str = COMMENT_MARKER) -> str:
    """Swap the previous marked line for a new one, keeping anything a human wrote."""
    kept = [row for row in (comment or "").splitlines() if not row.lstrip().startswith(marker)]
    kept.append(line)
    return "\n".join(kept).strip("\n")


###### modeling.edit_points


def _get_edit_node(node_path: str) -> hou.Node:
    """An existing Edit SOP, or a sentence saying what was found instead."""
    node = _get_sop_node(node_path, "edit_node")
    type_name = node.type().name()
    if type_name != "edit":
        raise hou.OperationFailed(
            f"{node_path} is a '{type_name}' SOP, not an 'edit' SOP. edit_node updates an "
            "existing Edit; pass 'after' instead to create one."
        )
    return node


def _edit_input(node: hou.Node) -> hou.Node:
    """Whatever feeds the Edit node's first input."""
    inputs = node.inputs()
    source = inputs[0] if inputs else None
    if source is None:
        raise hou.OperationFailed(
            f"{node.path()} has nothing wired into its first input, so it has no points to move."
        )
    return source


def _edit_points(
    *,
    after: str | None = None,
    edit_node: str | None = None,
    name: str | None = None,
    moves: list,
    expect_points: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Move control points through a native Edit SOP and report where they landed."""
    started = time.perf_counter()
    after_path = as_text(after, "after").strip()
    edit_path = as_text(edit_node, "edit_node").strip()
    node_name = as_text(name, "name").strip()

    if after_path and edit_path:
        raise ValueError(
            "Give either 'after' (create a new Edit below that node) or 'edit_node' "
            "(update an existing one), not both."
        )
    if not after_path and not edit_path:
        raise ValueError(
            "Give either 'after' (create a new Edit below that node) or 'edit_node' "
            "(update an existing one)."
        )

    ###### Phase 1: look at everything, change nothing.

    if after_path:
        source = _get_sop_node(after_path, "after")
        target = None
        if not node_name:
            raise ValueError(
                "name is required when creating an Edit with 'after'. Name it for what the "
                "edit is for, for example 'panel_inner_depth', so the graph stays readable."
            )
    else:
        target = _get_edit_node(edit_path)
        source = _edit_input(target)

    input_geo = source.geometry()
    if input_geo is None:
        raise hou.OperationFailed(f"{source.path()} produced no geometry to edit.")

    point_count = input_geo.intrinsicValue("pointcount")
    if expect_points is not None:
        expected = as_int(expect_points, "expect_points")
        if expected != point_count:
            raise hou.OperationFailed(
                f"expect_points is {expected} but {source.path()} has {point_count} points, "
                "so the point numbers in 'moves' would not mean the points you think. "
                "Nothing was changed."
            )

    entries = validate_moves(moves, point_count, [g.name() for g in input_geo.pointGroups()])
    fingerprint = geometry_fingerprint(point_count, input_geo.pointFloatAttribValuesAsString("P"))

    rebased = False
    if target is not None:
        stored = target.userData(USER_DATA_KEY)
        if stored and stored != fingerprint:
            if not force:
                raise hou.OperationFailed(
                    f"The geometry feeding {target.path()} has changed since this Edit was "
                    f"written: it was {stored}, it is now {fingerprint}. The stored offsets are "
                    "keyed by point number, so they no longer describe the same points and "
                    "nothing was changed. Pass force=True to discard them and start from the "
                    "current input."
                )
            rebased = True

    original = list(input_geo.pointFloatAttribValues("P"))
    if target is None or rebased:
        base = original
    else:
        # Accumulating onto an existing Edit means starting from what it already
        # outputs: setPointPositionsFromString replaces the whole delta, so a
        # patch built on the input positions would silently undo the earlier move.
        base = list(target.geometry().pointFloatAttribValues("P"))
        if len(base) != point_count * 3:
            raise hou.OperationFailed(
                f"{target.path()} outputs {len(base) // 3} points while its input has "
                f"{point_count}; an Edit SOP cannot bridge that, so nothing was changed."
            )

    group_members: dict[str, list[int]] = {}
    for entry in entries:
        group = entry["group"]
        if group is None or group in group_members:
            continue
        found = input_geo.findPointGroup(group)
        group_members[group] = [point.number() for point in found.points()] if found else []

    updated, touched, requested = apply_moves(base, entries, group_members)

    ###### Phase 2: everything is known to be sound, so write.

    created = target is None
    renamed_from = ""
    flags_moved = False
    if created:
        parent = source.parent()
        target = parent.createNode("edit", node_name)
        if target.name() != node_name:
            renamed_from = node_name
        target.setInput(0, source, 0)
        target.moveToGoodPosition()
        if source.isDisplayFlagSet():
            # Leaving the display flag upstream hides the very edit just made.
            target.setDisplayFlag(True)
            target.setRenderFlag(True)
            flags_moved = True

    target.geometryDelta().setPointPositionsFromString(array.array("f", updated).tobytes())
    target.cook(force=True)
    output = target.geometry().pointFloatAttribValues("P")

    ###### Phase 3: read back what actually happened.

    changed = 0
    max_displacement = 0.0
    for point in touched:
        base_index = point * 3
        dx = output[base_index] - original[base_index]
        dy = output[base_index + 1] - original[base_index + 1]
        dz = output[base_index + 2] - original[base_index + 2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance > MOVED_EPSILON:
            changed += 1
        max_displacement = max(max_displacement, distance)

    diagonal = input_geo.boundingBox().sizevec().length()
    tolerance = POSITION_TOLERANCE_FACTOR * max(1.0, diagonal)
    mismatches: list[dict[str, Any]] = []
    for point, wanted in requested.items():
        base_index = point * 3
        actual = (output[base_index], output[base_index + 1], output[base_index + 2])
        gap = math.sqrt(sum((actual[axis] - wanted[axis]) ** 2 for axis in range(3)))
        if gap > tolerance:
            mismatches.append(
                {
                    "point": point,
                    "requested": [round(value, 6) for value in wanted],
                    "actual": [round(value, 6) for value in actual],
                }
            )

    line = edit_comment_line(len(touched), max_displacement, fingerprint)
    target.setUserData(USER_DATA_KEY, fingerprint)
    target.setComment(replace_comment_line(target.comment(), line))
    target.setGenericFlag(hou.nodeFlag.DisplayComment, True)

    report: dict[str, Any] = {
        "node_path": target.path(),
        "created": created,
        "moved": len(touched),
        "changed": changed,
        "max_displacement": round(max_displacement, 6),
        "samples": [
            {
                "point": point,
                "from": [round(original[point * 3 + axis], 6) for axis in range(3)],
                "to": [round(output[point * 3 + axis], 6) for axis in range(3)],
            }
            for point in touched[:3]
        ],
        "fingerprint": fingerprint,
    }
    if renamed_from:
        report["renamed_from"] = renamed_from
    if flags_moved:
        report["flags_moved"] = True
    if rebased:
        report["rebased"] = True
    if mismatches:
        report["mismatch_count"] = len(mismatches)
        report["mismatches"] = mismatches[:10]

    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return report


register_handler("modeling.edit_points", _edit_points)


###### Comparison maths (no hou)


def _as_float(value: object, name: str) -> float:
    """A finite number, or a sentence naming the argument."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a number, not {type(value).__name__}: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a number, not {type(value).__name__}: {value!r}"
        ) from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, not {value!r}")
    return number


def topology_report(
    a_faces: list[tuple[int, list[int]]],
    b_faces: list[tuple[int, list[int]]],
    a_point_count: int,
    b_point_count: int,
    a_prim_count: int,
    b_prim_count: int,
) -> dict[str, Any]:
    """Identical only when counts match and face point lists agree in order."""
    first_mismatch: dict[str, Any] | None = None
    shared = min(len(a_faces), len(b_faces))
    for index in range(shared):
        a_prim, a_pts = a_faces[index]
        _b_prim, b_pts = b_faces[index]
        if list(a_pts) != list(b_pts):
            first_mismatch = {"prim": a_prim, "a": list(a_pts), "b": list(b_pts)}
            break
    if first_mismatch is None and len(a_faces) != len(b_faces):
        if len(a_faces) > len(b_faces):
            prim, pts = a_faces[shared]
            first_mismatch = {"prim": prim, "a": list(pts), "b": []}
        else:
            prim, pts = b_faces[shared]
            first_mismatch = {"prim": prim, "a": [], "b": list(pts)}

    identical = (
        a_point_count == b_point_count
        and a_prim_count == b_prim_count
        and len(a_faces) == len(b_faces)
        and first_mismatch is None
    )
    if identical:
        return {"identical": True, "points": a_point_count, "prims": a_prim_count}
    report: dict[str, Any] = {
        "identical": False,
        "points": {"a": a_point_count, "b": b_point_count},
        "prims": {"a": a_prim_count, "b": b_prim_count},
    }
    if first_mismatch is not None:
        report["first_mismatch"] = first_mismatch
    return report


def position_report(
    a_positions: list[float],
    b_positions: list[float],
    tolerance: float,
    max_list: int,
) -> dict[str, Any]:
    """Point-number deltas. Caller guarantees equal point counts."""
    count = min(len(a_positions), len(b_positions)) // 3
    if count <= 0:
        return {"max_delta": 0.0, "mean_delta": 0.0, "over_tolerance": 0}

    max_delta = 0.0
    total = 0.0
    over = 0
    worst: list[tuple[float, int]] = []
    for point in range(count):
        base = point * 3
        dx = a_positions[base] - b_positions[base]
        dy = a_positions[base + 1] - b_positions[base + 1]
        dz = a_positions[base + 2] - b_positions[base + 2]
        delta = math.sqrt(dx * dx + dy * dy + dz * dz)
        total += delta
        if delta > max_delta:
            max_delta = delta
        if delta > tolerance:
            over += 1
            worst.append((delta, point))

    report: dict[str, Any] = {
        "max_delta": round(max_delta, 6),
        "mean_delta": round(total / count, 6),
        "over_tolerance": over,
    }
    if worst and max_list:
        worst.sort(reverse=True)
        report["worst"] = [
            {"point": index, "delta": round(delta, 6)} for delta, index in worst[:max_list]
        ]
    return report


def classify_provenance(
    source_ids: list[int], source_count: int
) -> tuple[int, int, int, list[int], list[tuple[int, int]]]:
    """Walk b-element source ids against a's count.

    -1 or out of range is new. A source already claimed by an earlier b
    element is also new: PolyExtrude copies ``sourcept`` / ``sourceprim`` from
    the source vertex or face onto the elements it creates, so the extra copy
    is the new point or side face, not a second original.
    """
    claimed: set[int] = set()
    kept_pairs: list[tuple[int, int]] = []
    new_ids: list[int] = []
    for b_index, raw in enumerate(source_ids):
        src = int(raw)
        if src < 0 or src >= source_count or src in claimed:
            new_ids.append(b_index)
            continue
        claimed.add(src)
        kept_pairs.append((b_index, src))
    removed = source_count - len(claimed)
    return len(kept_pairs), len(new_ids), removed, new_ids, kept_pairs


def provenance_report(
    *,
    sourcept: list[int],
    a_point_count: int,
    sourceprim: list[int] | None,
    a_prim_count: int,
    a_positions: list[float],
    b_positions: list[float],
    max_list: int,
) -> dict[str, Any]:
    """Map each b element to its a source; extras and sentinels are new."""
    kept, new, removed, new_points, pairs = classify_provenance(sourcept, a_point_count)
    report: dict[str, Any] = {"points": {"kept": kept, "new": new, "removed": removed}}
    if new_points and max_list:
        report["new_points"] = new_points[:max_list]

    max_delta = 0.0
    a_limit = len(a_positions) // 3
    b_limit = len(b_positions) // 3
    for b_index, a_index in pairs:
        if b_index >= b_limit or a_index >= a_limit:
            continue
        ab, bb = a_index * 3, b_index * 3
        dx = a_positions[ab] - b_positions[bb]
        dy = a_positions[ab + 1] - b_positions[bb + 1]
        dz = a_positions[ab + 2] - b_positions[bb + 2]
        max_delta = max(max_delta, math.sqrt(dx * dx + dy * dy + dz * dz))
    if pairs:
        report["kept_max_delta"] = round(max_delta, 6)

    if sourceprim is not None:
        p_kept, p_new, p_removed, new_prims, _pairs = classify_provenance(sourceprim, a_prim_count)
        report["prims"] = {"kept": p_kept, "new": p_new, "removed": p_removed}
        if new_prims and max_list:
            report["new_prims"] = new_prims[:max_list]
    return report


def compare_meshes(
    *,
    a_faces: list[tuple[int, list[int]]],
    b_faces: list[tuple[int, list[int]]],
    a_point_count: int,
    b_point_count: int,
    a_prim_count: int,
    b_prim_count: int,
    a_positions: list[float],
    b_positions: list[float],
    sourcept: list[int] | None = None,
    sourceprim: list[int] | None = None,
    tolerance: float = 1e-5,
    max_list: int = 20,
) -> dict[str, Any]:
    """The comparison block: topology, positions and optional provenance."""
    topology = topology_report(
        a_faces, b_faces, a_point_count, b_point_count, a_prim_count, b_prim_count
    )
    result: dict[str, Any] = {"topology": topology}
    if a_point_count == b_point_count:
        positions = position_report(a_positions, b_positions, tolerance, max_list)
        result["positions"] = positions
        result["same"] = bool(topology["identical"] and positions["over_tolerance"] == 0)
    else:
        result["same"] = False
    if sourcept is not None:
        result["provenance"] = provenance_report(
            sourcept=sourcept,
            a_point_count=a_point_count,
            sourceprim=sourceprim,
            a_prim_count=a_prim_count,
            a_positions=a_positions,
            b_positions=b_positions,
            max_list=max_list,
        )
    return result


###### modeling.compare_geometry


def _int_attrib_values(geo: hou.Geometry, name: str, kind: str) -> list[int] | None:
    """An int point or prim attribute, or None if it is missing or not int."""
    if kind == "point":
        attrib = geo.findPointAttrib(name)
        reader = geo.pointIntAttribValues
    else:
        attrib = geo.findPrimAttrib(name)
        reader = geo.primIntAttribValues
    if attrib is None or attrib.dataType() != hou.attribData.Int:
        return None
    return list(reader(name))


def _geo_snapshot(geo: hou.Geometry) -> dict[str, Any]:
    """Faces, counts, positions, and provenance attributes from one SOP."""
    parsed = extract_faces(_load_geo_document(geo))
    return {
        "faces": parsed["faces"],
        "point_count": int(geo.intrinsicValue("pointcount")),
        "prim_count": int(geo.intrinsicValue("primitivecount")),
        "positions": list(geo.pointFloatAttribValues("P")),
        "sourcept": _int_attrib_values(geo, "sourcept", "point"),
        "sourceprim": _int_attrib_values(geo, "sourceprim", "prim"),
    }


def _compare_snapshots(
    snap_a: dict[str, Any],
    snap_b: dict[str, Any],
    *,
    tolerance: float,
    max_list: int,
) -> dict[str, Any]:
    return compare_meshes(
        a_faces=snap_a["faces"],
        b_faces=snap_b["faces"],
        a_point_count=snap_a["point_count"],
        b_point_count=snap_b["point_count"],
        a_prim_count=snap_a["prim_count"],
        b_prim_count=snap_b["prim_count"],
        a_positions=snap_a["positions"],
        b_positions=snap_b["positions"],
        sourcept=snap_b["sourcept"],
        sourceprim=snap_b["sourceprim"],
        tolerance=tolerance,
        max_list=max_list,
    )


def _dump_comparison(
    path: str,
    snap_a: dict[str, Any],
    snap_b: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    payload = {
        "a_faces": [[prim, pts] for prim, pts in snap_a["faces"]],
        "b_faces": [[prim, pts] for prim, pts in snap_b["faces"]],
        "a_point_count": snap_a["point_count"],
        "b_point_count": snap_b["point_count"],
        "a_prim_count": snap_a["prim_count"],
        "b_prim_count": snap_b["prim_count"],
        "same": comparison.get("same"),
        "topology": comparison.get("topology"),
        "positions": comparison.get("positions"),
        "provenance": comparison.get("provenance"),
    }
    _write_dump(path, payload)


def _compare_geometry(
    *,
    a: str,
    b: str,
    tolerance: float = 1e-5,
    max_list: int = 20,
    dump_path: str | None = None,
    **_,
) -> dict[str, Any]:
    """Compare two SOP meshes: topology, positions, optional sourcept provenance.

    Tag *before* the op that creates geometry. Attrib Create (point, int,
    sourcept, default -1) then ``i@sourcept = @ptnum;``; the same for
    sourceprim on primitives. A wrangle alone defaults new elements to 0,
    which is a valid source (point 0). PolyExtrude copies the attribute from
    the source vertex or face onto new elements; those extra copies are
    reported as new.
    """
    started = time.perf_counter()
    a_path = as_text(a, "a").strip()
    b_path = as_text(b, "b").strip()
    if not a_path:
        raise ValueError("a must name a SOP node, for example /obj/geo1/box1")
    if not b_path:
        raise ValueError("b must name a SOP node, for example /obj/geo1/box1")
    tol = _as_float(tolerance, "tolerance")
    if tol < 0:
        raise ValueError(f"tolerance must be >= 0, not {tol}")
    cap = max(0, min(as_int(max_list, "max_list"), 1000))
    dump_target = as_text(dump_path, "dump_path").strip()

    geo_a = _get_sop_geo(a_path)
    geo_b = _get_sop_geo(b_path)
    snap_a = _geo_snapshot(geo_a)
    snap_b = _geo_snapshot(geo_b)
    report = _compare_snapshots(snap_a, snap_b, tolerance=tol, max_list=cap)
    report["a"] = a_path
    report["b"] = b_path

    if dump_target:
        try:
            _dump_comparison(dump_target, snap_a, snap_b, report)
        except OSError as exc:
            report["dump_error"] = f"could not write {dump_target}: {exc}"
        else:
            report["dump"] = dump_target

    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return report


register_handler("modeling.compare_geometry", _compare_geometry)
