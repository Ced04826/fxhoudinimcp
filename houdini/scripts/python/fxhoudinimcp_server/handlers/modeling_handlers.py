"""Modeling (SOP) handlers for FXHoudini-MCP.

One command so far: ``modeling.get_mesh_report``, the acceptance check a modeler
runs before trusting a control cage -- face mix, pieces, boundary loops,
non-manifold edges, degenerate faces, poles and folded quads, in a single call.

Topology comes from a ``.geo`` JSON round trip rather than per-primitive HOM
calls. Measured on 22.0.368 with a 10 000 quad grid: 32 ms for
``saveToFile`` + ``json.load`` against 878 ms for walking ``prim.vertices()``.
The maths below is deliberately free of ``hou`` and ``numpy`` so it can be
tested without Houdini; only the handler itself touches either.
"""

from __future__ import annotations

# Built-in
import contextlib
import gc
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


def _get_sop_geo(node_path: str) -> hou.Geometry:
    """The cooked read-only geometry for a SOP node.

    The ``geometry`` attribute is checked rather than called blind: on anything
    that is not a SOP -- ``/obj`` itself, an object node, a LOP -- calling it
    raised ``'OpNode' object has no attribute 'geometry'``, which names a Python
    internal instead of telling the caller they pointed at the wrong node.
    """
    node = hou.node(node_path)
    if node is None:
        raise hou.OperationFailed(f"Node not found: {node_path}")
    reader = getattr(node, "geometry", None)
    if reader is None:
        raise hou.OperationFailed(
            f"{node_path} is a '{node.type().name()}' node, which carries no geometry. "
            "get_mesh_report reads a SOP, for example /obj/geo1/subdivide1."
        )
    geo = reader()
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
