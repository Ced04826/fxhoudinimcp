"""Modeling (SOP) handlers for FXHoudini-MCP.

``modeling.get_mesh_report``
    Cage acceptance: face mix, pieces, boundary, non-manifold, degenerates,
    poles, folded quads. Topology from a ``.geo`` JSON round trip.

``modeling.edit_points``
    Batch point shaping through a native Edit SOP, with read-back evidence.

``modeling.compare_geometry``
    Topology, positions, and optional ``sourcept`` / ``sourceprim`` provenance.
    Answers "is this the same mesh", and needs the two to be related.

``modeling.compare_surfaces``
    Two-way sampled distance between two surfaces with nothing in common.
    Answers "is this the same shape", for a rebuild against its reference.

``modeling.get_uv_report``
    Islands, winding, overlapping UV triangles, occupied area and stretch.

The maths and validation below are deliberately free of ``hou`` and ``numpy``
so they can be tested without Houdini; only the handlers touch either. The
heavier geometry -- triangulation, sampling, closest points, UV overlap -- lives
in ``fxhoudinimcp_server.geometry_math`` for the same reason.
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
from fxhoudinimcp_server import geometry_math as gm
from fxhoudinimcp_server.config import place_new_node
from fxhoudinimcp_server.dispatcher import register_handler
from fxhoudinimcp_server.errors import as_int, as_text

###### Limits

# Every report carries this. A caller that parses a receipt can assert the
# shape it was written against instead of discovering a rename at runtime.
SCHEMA_VERSION = 1

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


def extract_faces(document: list, with_vertices: bool = False) -> dict[str, Any]:
    """Split a parsed ``.geo`` document into faces and everything else.

    Returns ``faces`` as ``(prim_id, [point ids])`` for closed polygons only,
    plus the prim ids of open polylines and of primitives that are not polygons
    at all, so the caller can report them without pretending they are faces.

    With *with_vertices*, ``face_vertices`` comes back parallel to ``faces``,
    holding each face's vertex numbers as this file numbers them. Only the UV
    report needs them -- a vertex attribute like ``uv`` is per corner, and
    reading it per point instead is what loses a seam -- and on a large mesh
    they double the memory of the parse, so they are off by default.

    These numbers index the vertex attribute arrays of the *same document*
    (see ``read_numeric_attribute``), never HOM's ``vertexFloatAttribValues``:
    the file lists vertices face by face, while HOM returns them in the order
    they sit in memory, and after a Reverse, a Mirror or a deletion the two
    differ.
    """
    doc = _flat_pairs(document)
    topology = _flat_pairs(doc.get("topology"))
    pointref = _flat_pairs(topology.get("pointref"))
    indices = pointref.get("indices") or []

    faces: list[tuple[int, list[int]]] = []
    face_vertices: list[list[int]] = []
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
                        if with_vertices:
                            face_vertices.append(list(range(offset, offset + size)))
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
                if with_vertices:
                    face_vertices.append([int(v) for v in vertex])
            else:
                open_polylines.append(prim_id)
        else:
            other_prims.append(prim_id)
        prim_id += 1

    return {
        "faces": faces,
        "face_vertices": face_vertices,
        "open_polylines": open_polylines,
        "other_prims": other_prims,
        "prim_count": prim_id,
        "point_count": int(doc.get("pointcount", 0)),
        "vertex_count": int(doc.get("vertexcount", len(indices))),
        "declared_prim_count": int(doc.get("primitivecount", prim_id)),
    }


###### .geo JSON attributes (no hou)
#
# The layout, as Houdini's own reader ($HFS/houdini/public/hgeo/hgeo.py) takes it:
#
#   document   [..., 'attributes', ['vertexattributes', [<entry>, ...],
#                                   'pointattributes', [...], ...], ...]
#   entry      [definition, data], both flat key/value lists
#              definition  ['scope', 'public', 'type', 'numeric', 'name', 'uv', ...]
#              data        ['size', 3, 'storage', 'fpreal32', 'defaults', [...],
#                           'values', <values>]
#   values     ['size', 3, 'storage', 'fpreal32', <one of the three below>]
#              'tuples'       [[u, v, w], ...]            one tuple per element
#              'arrays'       [[u, u, ...], [v, ...], ...] one array per component
#              'rawpagedata'  flat numbers in pages, with 'packing', 'pagesize'
#                             and optionally 'constantpageflags'
#
# Element i of a vertex attribute belongs to the file's vertex number i, the
# same number extract_faces hands back, which is why reading both from one
# document keeps corners and values lined up.

_ATTRIBUTE_OWNERS = {
    "vertex": "vertexattributes",
    "point": "pointattributes",
    "primitive": "primitiveattributes",
    "detail": "globalattributes",
}


def _decode_raw_pages(
    raw: list,
    *,
    size: int,
    count: int,
    packing: list[int],
    pagesize: int,
    constant_flags: list | None,
) -> list[float]:
    """Paged storage to element-major values.

    Pages run in element order. Within a page each subvector of *packing* is
    stored whole before the next one: every element's components for that
    subvector, one element after another -- or, when the subvector's flag for
    that page is set, a single tuple shared by the whole page.
    """
    if sum(packing) != size or any(width <= 0 for width in packing):
        raise ValueError(f"packing {packing} does not add up to the tuple size {size}")
    if pagesize <= 0:
        raise ValueError(f"pagesize must be positive, not {pagesize}")

    starts: list[int] = []
    running = 0
    for width in packing:
        starts.append(running)
        running += width

    flags = constant_flags if isinstance(constant_flags, (list, tuple)) else []
    flat = [0.0] * (count * size)
    cursor = 0
    for page in range((count + pagesize - 1) // pagesize):
        first = page * pagesize
        on_page = min(pagesize, count - first)
        for sub, width in enumerate(packing):
            sub_flags = flags[sub] if sub < len(flags) else []
            constant = bool(sub_flags[page]) if page < len(sub_flags or []) else False
            if constant:
                shared = raw[cursor : cursor + width]
                cursor += width
            for element in range(first, first + on_page):
                if constant:
                    chunk = shared
                else:
                    chunk = raw[cursor : cursor + width]
                    cursor += width
                if len(chunk) != width:
                    raise ValueError("raw page data ends before its last page")
                base = element * size + starts[sub]
                flat[base : base + width] = chunk
    if cursor != len(raw):
        raise ValueError(
            f"raw page data holds {len(raw)} numbers but its pages account for {cursor}"
        )
    return flat


def _flatten_attribute_values(values: dict[str, Any], size: int, count: int) -> list[float]:
    """One storage block to element-major values: ``flat[i * size + c]``."""
    tuples = values.get("tuples")
    if isinstance(tuples, (list, tuple)):
        flat: list[float] = []
        for item in tuples:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    raw = values.get("rawpagedata")
    if isinstance(raw, (list, tuple)):
        packing = values.get("packing")
        return _decode_raw_pages(
            list(raw),
            size=size,
            count=count,
            packing=[int(width) for width in packing] if packing else [size],
            pagesize=int(values.get("pagesize", 0)),
            constant_flags=values.get("constantpageflags"),
        )

    arrays = values.get("arrays")
    if isinstance(arrays, (list, tuple)):
        if len(arrays) == size and all(len(component) == count for component in arrays):
            # One array per component; interleave them.
            flat = [0.0] * (count * size)
            for component, column in enumerate(arrays):
                flat[component::size] = list(column)
            return flat
        if len(arrays) == 1 and len(arrays[0]) == count * size:
            return list(arrays[0])
        raise ValueError(
            f"'arrays' storage holds {len(arrays)} arrays of lengths "
            f"{[len(component) for component in arrays][:4]} for {count} elements of size {size}"
        )

    raise ValueError(
        f"no 'tuples', 'arrays' or 'rawpagedata' in the values block (keys: {sorted(values)})"
    )


def read_numeric_attribute(document: list, owner: str, name: str) -> tuple[int, list[float]] | None:
    """A numeric attribute from a parsed ``.geo`` document, or None if absent.

    Returns ``(size, flat)`` with ``flat[i * size + c]`` the component *c* of
    element *i*, elements numbered as the file numbers them. A block this
    parser cannot read, or one whose length does not match the element count,
    raises rather than returning values that would be lined up wrongly.
    """
    doc = _flat_pairs(document)
    attributes = _flat_pairs(doc.get("attributes"))
    count_key = {"vertex": "vertexcount", "point": "pointcount", "primitive": "primitivecount"}
    count = int(doc.get(count_key[owner], 0)) if owner in count_key else 1

    for entry in attributes.get(_ATTRIBUTE_OWNERS[owner]) or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        definition = _flat_pairs(entry[0])
        if definition.get("name") != name:
            continue
        if definition.get("type") != "numeric":
            raise ValueError(
                f"{owner} attribute '{name}' is stored as '{definition.get('type')}', not numeric"
            )
        data = _flat_pairs(entry[1])
        values = _flat_pairs(data.get("values"))
        size = int(values.get("size", data.get("size", 1)))
        if size <= 0:
            raise ValueError(f"{owner} attribute '{name}' declares size {size}")
        flat = _flatten_attribute_values(values, size, count)
        if len(flat) != count * size:
            raise ValueError(
                f"{owner} attribute '{name}' holds {len(flat)} numbers for {count} elements "
                f"of size {size} ({count * size} expected)"
            )
        return size, flat
    return None


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


def nonfinite_points(flat: Any) -> list[int]:
    """Point numbers whose flat xyz position holds a NaN or an infinity.

    The sum is the screen: it is finite whenever every term is, so a clean mesh
    costs one pass in C. Only a bad value -- or a sum that overflowed on
    absurdly large but finite coordinates -- sends it through the per-point walk.
    """
    if math.isfinite(sum(flat)):
        return []
    isfinite = math.isfinite
    return [
        index // 3
        for index in range(0, len(flat) - 2, 3)
        if not (isfinite(flat[index]) and isfinite(flat[index + 1]) and isfinite(flat[index + 2]))
    ]


def _require_finite_positions(flat: Any, where: str) -> None:
    """Refuse geometry with non-finite positions before the ``.geo`` round trip.

    Houdini writes a NaN or an infinity into the JSON as a token no JSON reader
    accepts, so without this the caller gets a parse error that names neither
    the node nor the points -- the usual source being a PolyBevel that failed.
    """
    bad = nonfinite_points(flat)
    if not bad:
        return
    shown = ", ".join(str(point) for point in bad[:10])
    more = ", ..." if len(bad) > 10 else ""
    subject = f"{len(bad)} points have" if len(bad) != 1 else "1 point has"
    raise hou.OperationFailed(
        f"{where}: {subject} non-finite positions (NaN/inf): [{shown}{more}]. Nothing was measured."
    )


def _load_geo_document(geo: hou.Geometry) -> list:
    """Save the geometry to a temporary .geo and read it back as JSON.

    Callers check P for non-finite values first (``_require_finite_positions``);
    the error here covers what that cannot see, such as a NaN in another
    attribute.
    """
    handle, path = tempfile.mkstemp(prefix="fxmcp-mesh-", suffix=".geo", dir=tempfile.gettempdir())
    os.close(handle)
    try:
        geo.saveToFile(path)
        with open(path, encoding="utf-8") as stream:
            try:
                return json.load(stream)
            except ValueError as exc:
                raise hou.OperationFailed(
                    f"The geometry's .geo copy could not be read back as JSON ({exc}). A NaN "
                    "or infinite attribute value is the usual cause. Nothing was measured."
                ) from None
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


###### Optional per-face quality checks
#
# Off by default: they cost another pass over every corner, and the counts a
# cage report already gives answer most questions. They exist because those
# counts have a blind spot -- a face with a 179.6 degree corner has area, no
# repeated point and four sides, so nothing in the default report mentions it.

QUALITY_CHECKS = ("corner_angle", "triangulation")

_DEFAULT_THRESHOLDS = {
    # A corner outside this band is a crease or a sliver after subdivision.
    # Wide enough that an ordinary hexagon (120 degrees) and a right-angled
    # triangle (45/45/90) pass without comment.
    "min_corner_angle_deg": 10.0,
    "max_corner_angle_deg": 170.0,
    # Corner distance from the face's own plane, over the face's diameter.
    "planarity_ratio": 0.01,
    # A child triangle's area over its source face's area. Two triangles from a
    # quad give about 0.5 each; a millionth of the face is a sliver the
    # triangulator produced, not a piece of the surface.
    "min_triangle_area_fraction": 1e-6,
    # The smallest corner in a child triangle. A needle with real area still
    # shades badly and still breaks solvers.
    "min_triangle_angle_deg": 1.0,
}


def _quality_selection(quality_checks: Any) -> list[str]:
    """The requested checks, or a sentence naming what is on offer."""
    if quality_checks is None:
        return []
    if isinstance(quality_checks, str):
        quality_checks = [quality_checks]
    if not isinstance(quality_checks, (list, tuple)):
        raise ValueError(
            f"quality_checks must be a list of check names {list(QUALITY_CHECKS)}, "
            f"not {type(quality_checks).__name__}"
        )
    selected: list[str] = []
    for entry in quality_checks:
        name = as_text(entry, "quality_checks").strip()
        if name not in QUALITY_CHECKS:
            raise ValueError(
                f"quality_checks names '{name}', which is not a check this report knows. "
                f"Available: {list(QUALITY_CHECKS)}"
            )
        if name not in selected:
            selected.append(name)
    return selected


def _resolved_thresholds(thresholds: Any) -> dict[str, float]:
    """Defaults with the caller's overrides applied, each one checked."""
    values = dict(_DEFAULT_THRESHOLDS)
    if thresholds is None:
        return values
    if not isinstance(thresholds, dict):
        raise ValueError(
            f"thresholds must be a dict like {_DEFAULT_THRESHOLDS}, not {type(thresholds).__name__}"
        )
    for key, value in thresholds.items():
        if key not in values:
            raise ValueError(
                f"thresholds names '{key}', which no check reads. Available: {sorted(values)}"
            )
        number = _as_float(value, f"thresholds['{key}']")
        if number < 0:
            raise ValueError(f"thresholds['{key}'] must be >= 0, not {number}")
        values[key] = number
    if values["min_corner_angle_deg"] > values["max_corner_angle_deg"]:
        raise ValueError(
            "thresholds min_corner_angle_deg "
            f"({values['min_corner_angle_deg']}) is above max_corner_angle_deg "
            f"({values['max_corner_angle_deg']}), so every corner would be both too "
            "sharp and too flat."
        )
    return values


def _build_mesh_report(
    *,
    node_path: str,
    group: str,
    max_list: int,
    dump_target: str,
    quality_checks: list[str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Everything in the report except how long it took."""
    node_path = as_text(node_path, "node_path").strip()
    if not node_path:
        raise ValueError("node_path must name a SOP node, for example /obj/geo1/subdivide1")
    group_name = group.strip()

    geo = _get_sop_geo(node_path)

    # The limit is on what gets checked, not on what was opened: naming a
    # primitive group is the documented way to check a region of a mesh too
    # large to check whole, so guarding on the whole input would refuse the
    # very call the message asks for. A point group cannot be sized before the
    # faces are known, so that case still guards on the input.
    total_prims = int(geo.intrinsicValue("primitivecount"))
    named_prim_group = geo.findPrimGroup(group_name) if group_name else None
    scoped_prims = len(named_prim_group.prims()) if named_prim_group is not None else total_prims
    _check_scope_size(
        node_path, "node_path", scoped_prims, group_name if named_prim_group is not None else ""
    )

    # Positions: one bulk read, checked before the round trip that a NaN would
    # break, then reshaped by numpy. numpy ships with Houdini's Python; the
    # pure-Python helpers above take the .tolist() form, which is also faster
    # to index in the per-face loops than numpy rows are.
    flat = geo.pointFloatAttribValues("P")
    _require_finite_positions(flat, node_path)

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

    import numpy

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

    by_valence: dict[str, int] = {}
    for _point, count in poles:
        key = str(count)
        by_valence[key] = by_valence.get(key, 0) + 1

    def body(cap: int | None) -> dict[str, Any]:
        """The report with every id list cut at *cap*; None keeps them whole.

        The receipt and the dump both come from here, so they share every key
        and its meaning: the dump is the receipt with its lists uncut, plus the
        per-point and per-diagonal lists only a file has room for. A list is
        omitted when it would be empty, and in the receipt when max_list is 0:
        a key whose value is [] costs the reader a decision and tells them
        nothing.
        """

        def listed(values: list) -> list | None:
            if not values or cap == 0:
                return None
            return list(values) if cap is None else list(values[:cap])

        boundary: dict[str, Any] = {"edges": len(boundary_edges), "loops": len(loop_sizes)}
        if (sizes := listed(loop_sizes)) is not None:
            boundary["loop_sizes"] = sizes
        if open_chains:
            boundary["open_chains"] = open_chains

        nonmanifold_report: dict[str, Any] = {"count": len(nonmanifold)}
        if (edges := listed(nonmanifold)) is not None:
            nonmanifold_report["edges"] = [list(edge) for edge in edges]

        degenerate_report: dict[str, Any] = {"count": len(bad_faces)}
        if (prims := listed(bad_faces)) is not None:
            degenerate_report["prims"] = prims

        # Counts only in the receipt: a pole is worth a look in the viewport,
        # not twenty coordinates in the reply. Which points they are goes to
        # the dump.
        poles_report: dict[str, Any] = {"count": len(poles)}
        if poles:
            poles_report["by_valence"] = dict(by_valence)
            if cap is None:
                poles_report["points"] = [
                    {"point": point, "valence": count, "P": positions[point]}
                    for point, count in poles
                ]

        folded_report: dict[str, Any] = {
            "count": len(folded),
            "diag02": len(diag02),
            "diag13": len(diag13),
        }
        if (prims := listed(folded)) is not None:
            folded_report["prims"] = prims
        if cap is None:
            if diag02:
                folded_report["diag02_prims"] = list(diag02)
            if diag13:
                folded_report["diag13_prims"] = list(diag13)

        return {
            "schema_version": SCHEMA_VERSION,
            "node_path": node_path,
            "scope": scope_label,
            "counts": dict(counts),
            "pieces": {
                "count": piece_count,
                "largest": list(piece_sizes) if cap is None else piece_sizes[:5],
            },
            "boundary": boundary,
            "nonmanifold_edges": nonmanifold_report,
            "degenerate": degenerate_report,
            "valence": dict(histogram),
            "poles": poles_report,
            "folded_quads": folded_report,
            "bbox": {
                "min": [round(v, 6) for v in bbox_min],
                "max": [round(v, 6) for v in bbox_max],
                "size": [round(v, 6) for v in size],
                "diag": round(diagonal, 6),
            },
        }

    report = body(max_list)

    # The checks walk every corner, so they run once at whichever cap is
    # larger and the receipt takes a truncated view of the same answer.
    quality_cap = max(max_list, 1000) if dump_target else max_list
    quality: dict[str, Any] = {}
    if "corner_angle" in quality_checks:
        quality["corner_angle"] = gm.corner_angle_check(
            faces,
            positions,
            min_deg=thresholds["min_corner_angle_deg"],
            max_deg=thresholds["max_corner_angle_deg"],
            max_list=quality_cap,
        )
    if "triangulation" in quality_checks:
        quality["triangulation"] = _triangulation_check(
            geo,
            node_path=node_path,
            faces=faces,
            positions=positions,
            total_prims=total_prims,
            thresholds=thresholds,
            max_list=quality_cap,
        )
    if quality:
        report["quality"] = {
            name: _capped_lists(block, max_list) for name, block in quality.items()
        }

    if dump_target:
        payload = body(None)
        if quality:
            payload["quality"] = quality
        try:
            _write_dump(dump_target, payload)
        except OSError as exc:
            # Say so rather than return a receipt that implies a file exists.
            report["dump_error"] = f"could not write {dump_target}: {exc}"
        else:
            report["dump"] = dump_target

    return report


def _triangulation_check(
    geo: hou.Geometry,
    *,
    node_path: str,
    faces: list[tuple[int, list[int]]],
    positions: list,
    total_prims: int,
    thresholds: dict[str, float],
    max_list: int,
) -> dict[str, Any]:
    """What the native triangulator actually made of each face.

    The point is the quality of the triangles, not their number. Divide runs on
    a detached copy of exactly the faces in scope, and every child triangle is
    measured against its own source face: its share of that face's area, and
    its smallest corner. Both are scale-free, so the thresholds mean the same
    thing on a bolt and on a building, and every entry names the source prim.

    The counts are still reported -- a face that yields no triangles at all, or
    more than ``sides - 2``, is worth knowing about -- but they are not the
    check. Planarity is measured on the original faces, since a triangulation
    is planar by construction and would hide the question.

    Only closed polygons are passed to Divide, so a mesh that also carries
    curves or volumes is still checked rather than refused.
    """
    scoped = {prim_id for prim_id, _points in faces}
    triangulated = _triangulate(
        geo,
        node_path=node_path,
        label="quality_checks",
        keep=scoped,
        total_prims=total_prims,
        matrix=None,
    )

    produced: dict[int, int] = {}
    for prim_id in triangulated["prim_ids"]:
        produced[prim_id] = produced.get(prim_id, 0) + 1

    unexpected: list[dict[str, Any]] = []
    source_area: dict[int, float] = {}
    for prim_id, points in faces:
        source_area[prim_id] = gm.polygon_area_3d(gm.face_points(points, positions))
        count = produced.get(prim_id, 0)
        if count != len(points) - 2:
            unexpected.append({"prim": prim_id, "sides": len(points), "triangles": count})

    report: dict[str, Any] = {
        "triangles": len(triangulated["prim_ids"]),
        "faces_checked": len(faces),
        "untriangulated": len(triangulated["untriangulated"]),
        "unexpected": len(unexpected),
    }
    if unexpected and max_list:
        report["unexpected_faces"] = unexpected[:max_list]
    report.update(
        gm.triangulation_quality(
            list(zip(triangulated["prim_ids"], triangulated["triangles"], strict=True)),
            source_area,
            min_area_fraction=thresholds["min_triangle_area_fraction"],
            min_angle_deg=thresholds["min_triangle_angle_deg"],
            max_list=max_list,
        )
    )
    report.update(
        gm.planarity_check(
            faces, positions, planarity_ratio=thresholds["planarity_ratio"], max_list=max_list
        )
    )
    return report


def _capped_lists(block: dict[str, Any], limit: int) -> dict[str, Any]:
    """The same block with every list value truncated to *limit* entries."""
    return {
        key: (value[:limit] if isinstance(value, list) and limit >= 0 else value)
        for key, value in block.items()
    }


def _check_schema_version(requested: Any) -> None:
    """Refuse a receipt shape this build does not produce.

    A caller that pins a version wants to be told when the shape moved, not to
    parse a report with keys it has never seen.
    """
    if requested is None:
        return
    wanted = as_int(requested, "schema_version")
    if wanted != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {wanted} was asked for, but this plugin writes "
            f"version {SCHEMA_VERSION} receipts. Update the plugin, or drop the argument "
            "to take whatever it produces."
        )


def _get_mesh_report(
    *,
    node_path: str,
    group: str | None = None,
    max_list: int = 20,
    dump_path: str | None = None,
    quality_checks: list | str | None = None,
    thresholds: dict | None = None,
    schema_version: int | None = None,
) -> dict[str, Any]:
    """Topology health of a polygon mesh: the check before accepting a cage.

    Counts are always reported; id lists appear only when something is wrong and
    are capped at *max_list*, so a clean mesh answers in a few hundred
    characters. Poles are counted per valence only; which points they are is in
    the dump. Pass *dump_path* to receive a JSON file with the receipt's keys
    and every list whole, plus the pole points and the per-diagonal fold lists.

    Geometry with a NaN or infinite position is refused, naming the points.

    *quality_checks* adds per-face measurements the counts cannot express:
    ``corner_angle`` for corners outside the threshold band, ``triangulation``
    for faces that do not ear-clip cleanly and for how far from planar they
    are. Neither treats a triangle or an n-gon as a defect for its side count.
    """
    started = time.perf_counter()
    _check_schema_version(schema_version)
    selected = _quality_selection(quality_checks)
    limits = _resolved_thresholds(thresholds)

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
            quality_checks=selected,
            thresholds=limits,
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
        place_new_node(target)
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


def _geo_snapshot(geo: hou.Geometry, node_path: str) -> dict[str, Any]:
    """Faces, counts, positions, and provenance attributes from one SOP."""
    positions = list(geo.pointFloatAttribValues("P"))
    _require_finite_positions(positions, node_path)
    parsed = extract_faces(_load_geo_document(geo))
    return {
        "faces": parsed["faces"],
        "point_count": int(geo.intrinsicValue("pointcount")),
        "prim_count": int(geo.intrinsicValue("primitivecount")),
        "positions": positions,
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
    snap_a = _geo_snapshot(geo_a, a_path)
    snap_b = _geo_snapshot(geo_b, b_path)
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


###### Native triangulation on a detached copy
#
# Both new tools measure triangles, and every one of them has to come from the
# same place: Houdini's own Divide verb, run once per side on a copy that is
# not in the scene. That single triangulation is what the sampler walks, what
# nearestPrim answers from, and where the UVs are read -- so a non-planar quad
# compared with itself measures zero, and a folded quad keeps its fold in UV
# instead of being laid out flat by a diagonal chosen independently of the 3D
# one.
#
# hou.Geometry() is not in the scene graph: no node appears under /obj, nothing
# is left behind, and the user's geometry is only ever read.

# Travels from the input prims onto every triangle Divide makes from them, so a
# receipt can name the face the caller knows rather than a triangle number.
SOURCE_PRIM_ATTRIB = "fxmcp_srcprim"

_DIVIDE_PARMS = {"convex": 1, "numsides": 3}


def _divide_verb() -> Any:
    """The Divide SOP's verb, set to convex triangles, or a clear refusal.

    The parameter names are checked against the verb's own defaults rather than
    set blind: if a future build renames one, this says so instead of quietly
    running Divide with its defaults and handing back quads.
    """
    verb = hou.sopNodeTypeCategory().nodeVerb("divide")
    if verb is None:
        raise hou.OperationFailed(
            "This Houdini build has no 'divide' SOP verb, which is how these checks "
            "triangulate. Nothing was measured."
        )
    defaults = verb.parms()
    missing = sorted(name for name in _DIVIDE_PARMS if name not in defaults)
    if missing:
        raise hou.OperationFailed(
            f"The 'divide' SOP verb on this build has no parameter(s) {missing}, so it "
            "cannot be told to produce triangles. Nothing was measured."
        )
    verb.setParms(dict(_DIVIDE_PARMS))
    return verb


def _tagged_copy(
    geo: hou.Geometry, keep: set[int] | None, total_prims: int, matrix: Any | None
) -> hou.Geometry:
    """A detached copy of *geo*: scoped, tagged with its source prim numbers.

    Trimming happens before anything else reads the geometry, so scoping a
    small region of a large input costs one copy and then works at the size of
    the region.
    """
    work = hou.Geometry()
    work.merge(geo)
    work.addAttrib(hou.attribType.Prim, SOURCE_PRIM_ATTRIB, -1)
    work.setPrimIntAttribValues(SOURCE_PRIM_ATTRIB, list(range(total_prims)))
    if keep is not None and len(keep) < total_prims:
        work.deletePrims([work.prim(i) for i in range(total_prims) if i not in keep])
    if matrix is not None:
        work.transform(matrix)
    # Geometry edited outside a Python SOP carries stale data ids, and a verb
    # is entitled to trust them and reuse a cached result -- so a trimmed or
    # transformed copy could be triangulated as if it were still the original.
    # https://www.sidefx.com/docs/houdini/hom/hou/Geometry.html
    work.incrementAllDataIds()
    return work


def _triangulate(
    geo: hou.Geometry,
    *,
    node_path: str,
    label: str,
    keep: set[int] | None,
    total_prims: int,
    matrix: Any | None,
    uv_attribute: str = "",
) -> dict[str, Any]:
    """Triangulate a scope of *geo* natively and read the result back.

    Returns the triangulated geometry itself -- the caller keeps it alive for
    ``nearestPrim`` -- along with its triangles, the source prim each came
    from, and, when *uv_attribute* is named, each triangle corner's own UV.

    A primitive the Divide verb does not turn into a triangle is named and
    refused rather than dropped: a surface check that quietly skips the faces
    it could not handle is a check that passes for the wrong reason.
    """
    # The whole input, not just the scope: the copy keeps every point, and one
    # NaN anywhere in it is enough to break the round trip below.
    _require_finite_positions(geo.pointFloatAttribValues("P"), f"{node_path} ({label})")

    work = _tagged_copy(geo, keep, total_prims, matrix)
    triangulated = hou.Geometry()
    _divide_verb().execute(triangulated, [work])

    if int(triangulated.intrinsicValue("primitivecount")) <= 0:
        raise hou.OperationFailed(
            f"The native Divide produced no primitives from {node_path} ({label}), so "
            "there is nothing to measure."
        )

    document = _load_geo_document(triangulated)
    parsed = extract_faces(document, with_vertices=bool(uv_attribute))
    if not uv_attribute:
        # Only the UV read needs the parse past this point; on a large mesh it
        # is most of the memory in use.
        document = []
    faces = parsed["faces"]
    stragglers = len(parsed["open_polylines"]) + len(parsed["other_prims"])
    not_triangles = [prim_id for prim_id, points in faces if len(points) != 3]
    if stragglers or not_triangles:
        raise hou.OperationFailed(
            f"{node_path} ({label}) has {stragglers + len(not_triangles)} primitive(s) the "
            "native Divide could not turn into triangles. Open polylines, curves and "
            "non-polygon primitives cannot be triangulated; pass a region group naming "
            "only closed polygons."
        )

    source_ids = list(triangulated.primIntAttribValues(SOURCE_PRIM_ATTRIB))
    if len(source_ids) != len(faces):
        raise hou.OperationFailed(
            f"The native Divide returned {len(faces)} triangles but {len(source_ids)} "
            f"source ids for {node_path} ({label}); the two cannot be lined up, so "
            "nothing was measured."
        )
    orphans = sum(1 for source in source_ids if source < 0)
    if orphans:
        # The attribute defaults to -1, so this means Divide made triangles it
        # did not carry the tag onto. Every prim id in the receipt would be a
        # guess, which is worse than no receipt.
        raise hou.OperationFailed(
            f"{orphans} of the {len(source_ids)} triangles from {node_path} ({label}) carry "
            "no source face, so their prim ids cannot be reported. Nothing was measured."
        )

    flat = list(triangulated.pointFloatAttribValues("P"))
    positions = [[flat[i], flat[i + 1], flat[i + 2]] for i in range(0, len(flat), 3)]
    triangles = [
        (
            gm.point_of(positions, points[0]),
            gm.point_of(positions, points[1]),
            gm.point_of(positions, points[2]),
        )
        for _prim_id, points in faces
    ]

    covered = set(source_ids)
    scoped = keep if keep is not None else set(range(total_prims))
    result: dict[str, Any] = {
        "geometry": triangulated,
        "triangles": triangles,
        "prim_ids": source_ids,
        "faces": [(source_ids[i], points) for i, (_p, points) in enumerate(faces)],
        "positions": positions,
        "untriangulated": sorted(scoped - covered),
        "source_prims": covered,
    }
    if uv_attribute:
        result["corner_uvs"], result["uv_owner"], result["uv_size"] = _corner_uvs(
            triangulated,
            node_path,
            uv_attribute,
            faces,
            parsed["face_vertices"],
            document,
        )
    return result


def _corner_uvs(
    geo: hou.Geometry,
    node_path: str,
    name: str,
    faces: list,
    face_vertices: list[list[int]],
    document: list,
) -> tuple[list[list[tuple[float, float]]], str, int]:
    """Each triangle corner's own UV, from a vertex or a point attribute.

    Vertex first: that is where a seam lives. On a point attribute the same
    point carries one UV for every face using it, so there are no seams to
    preserve -- true, and worth saying in the receipt rather than leaving the
    reader to assume the mesh was cut.

    A vertex UV is read from the same ``.geo`` document the corners came from,
    so both use the file's vertex numbers. HOM's ``vertexFloatAttribValues``
    returns vertices in memory order, which after a Reverse, a Mirror or a
    deletion is not the order the file lists them in: indexing one with the
    other swapped corners and invented islands and stretch. HOM is still what
    finds the attribute and checks its type, because it names the candidates.
    """
    attrib = geo.findVertexAttrib(name)
    owner = "vertex"
    if attrib is None:
        attrib = geo.findPointAttrib(name)
        owner = "point"
    if attrib is None:
        candidates = sorted(
            {a.name() for a in geo.vertexAttribs() if a.size() >= 2}
            | {a.name() for a in geo.pointAttribs() if a.size() >= 2}
        )
        raise hou.OperationFailed(
            f"{node_path} has no vertex or point attribute named '{name}'. "
            f"Attributes with two or more components: {candidates or 'none'}."
        )
    if attrib.dataType() != hou.attribData.Float or attrib.size() < 2:
        raise hou.OperationFailed(
            f"'{name}' on {node_path} is a {attrib.dataType().name().lower()} attribute with "
            f"{attrib.size()} component(s). A UV attribute must be float with at least two."
        )

    size = attrib.size()
    if owner == "vertex":
        try:
            found = read_numeric_attribute(document, "vertex", name)
        except ValueError as exc:
            raise hou.OperationFailed(
                f"'{name}' on {node_path} could not be read from the .geo copy ({exc}), so "
                "corners and UVs cannot be lined up. Nothing was reported."
            ) from None
        if found is None:
            raise hou.OperationFailed(
                f"'{name}' is a vertex attribute on {node_path}, but the .geo copy of its "
                "triangulation carries no vertex attribute of that name, so corners and UVs "
                "cannot be lined up. Nothing was reported."
            )
        size, values = found
        if size < 2:
            raise hou.OperationFailed(
                f"'{name}' on {node_path} was saved with {size} component(s); a UV needs two."
            )
        return (
            [
                [(values[v * size], values[v * size + 1]) for v in vertices]
                for vertices in face_vertices
            ],
            owner,
            size,
        )

    values = list(geo.pointFloatAttribValues(name))
    limit = len(values) // size
    return (
        [
            [
                (values[p * size], values[p * size + 1]) if p < limit else (math.nan, math.nan)
                for p in point_ids
            ]
            for _prim_id, point_ids in faces
        ],
        owner,
        size,
    )


###### modeling.compare_surfaces
#
# compare_geometry answers "is this the same mesh": it walks point numbers and
# face point lists, so it says nothing about two meshes built differently. This
# answers the other question -- "is this the same shape" -- by sampling one
# surface and measuring to the nearest point on the other, in both directions,
# which needs no correspondence between the two topologies.

# Each sample is one accelerated closest-point query. The default is a check;
# the ceiling is where a check turns into a job worth waiting on deliberately.
DEFAULT_SURFACE_SAMPLES = 4_000
MAX_SURFACE_SAMPLES = 200_000

# When the caller names no tolerance: a thousandth of the shared bounding box
# diagonal, which is the scale at which a silhouette difference is visible.
AUTO_TOLERANCE_FACTOR = 1e-3

SURFACE_SPACES = ("sop", "world")


def _object_world_transform(node: hou.Node) -> Any | None:
    """The world transform of the object this SOP lives in, if any.

    Walks up rather than assuming the parent is the object: a SOP inside a
    subnet inside a geo object is still placed by that object's transform.
    """
    parent = node.parent()
    while parent is not None:
        if isinstance(parent, hou.ObjNode):
            return parent.worldTransform()
        parent = parent.parent()
    return None


def _region_scope(geo: hou.Geometry, node_path: str, group: str, argument: str) -> set[int] | None:
    """Prim ids in a named primitive group, or None when no group was given.

    Primitive groups only: a region of a surface is a set of faces, and a point
    group would silently pull in every face that merely touches the region.
    """
    if not group:
        return None
    prim_group = geo.findPrimGroup(group)
    if prim_group is None:
        names = sorted(g.name() for g in geo.primGroups())
        raise hou.OperationFailed(
            f"No primitive group named '{group}' on {node_path} ({argument}). "
            f"Primitive groups: {names or 'none'}."
        )
    members = {prim.number() for prim in prim_group.prims()}
    if not members:
        raise hou.OperationFailed(
            f"Primitive group '{group}' on {node_path} ({argument}) is empty, "
            "so it scopes no surface to measure."
        )
    return members


def _check_scope_size(node_path: str, label: str, scoped: int, group: str) -> None:
    """The size limit applies to what is measured, not to what was opened.

    Checking the whole input would refuse a fifty-face region of a million-face
    scan -- exactly the case the message tells callers to use a group for.
    """
    if scoped <= MAX_PRIMS:
        return
    where = f"group '{group}' on {node_path}" if group else f"{node_path} ({label})"
    raise hou.OperationFailed(
        f"{where} covers {scoped} primitives, more than the {MAX_PRIMS} these checks "
        "measure at once. Pass a primitive group naming a smaller region."
    )


###### Measuring a batch of samples against a surface
#
# Both modes below are Houdini's own closest-*surface* search -- the distance
# to the nearest point on a primitive, with no ray and no direction. They run
# against the triangulated copy the samples came from, so the two directions of
# a comparison each measure one surface rather than two versions of one, and
# both map the triangle they hit back to the prim id the caller knows.
#
#   batch       the Ray SOP verb in Minimum Distance mode, over every sample
#               in one native call. This is the production path.
#   per_sample  one hou.Geometry.nearestPrim call per sample. A verification
#               oracle and nothing else: measured live on 2026-09-11 at about
#               32 ms per call against a 44 402 triangle target, so 32 000
#               samples took seventeen minutes. It is capped far below that
#               and is not a fallback.
#
# There is no Attribute Wrangle verb on H22.0.368 -- nodeVerb('attribwrangle')
# is None -- so VEX xyzdist is not reachable this way at all.

QUERY_MODES = ("batch", "per_sample")

# Small enough that a mistaken call is a wait rather than an outage.
MAX_PER_SAMPLE_QUERIES = 2_000

# What the Ray SOP writes: "Point Intersection Distance: Ray point dist
# attribute gets value of distance to collision source" (SOP/ray help), and the
# primitive number under primnumattrib, whose own default name is used as-is.
_DISTANCE_ATTRIB = "dist"
_HIT_PRIM_ATTRIB = "hitprim"


def _ray_menu_index(parm_name: str, wanted: str, required: bool = True) -> int | None:
    """The index of a named entry in a Ray SOP menu, read from the node type.

    The order is looked up rather than hard-coded. Minimum Distance is not the
    Ray SOP's default -- Project Rays is -- so an index taken on faith and
    silently wrong would turn every distance into a ray hit along a normal,
    which is a different quantity that still looks like a number.
    """
    node_type = hou.sopNodeTypeCategory().nodeTypes().get("ray")
    if node_type is None:
        raise hou.OperationFailed(
            "This Houdini build has no 'ray' SOP, which is how these checks measure "
            "distances. Nothing was measured."
        )
    template = node_type.parmTemplateGroup().find(parm_name)
    items = [str(item) for item in (getattr(template, "menuItems", lambda: ())() or ())]
    labels = [str(label) for label in (getattr(template, "menuLabels", lambda: ())() or ())]
    for index, token in enumerate(items):
        label = labels[index].lower().replace(" ", "") if index < len(labels) else ""
        if token == wanted or wanted in label:
            return index
    if not required:
        return None
    raise hou.OperationFailed(
        f"The Ray SOP's '{parm_name}' menu on this build has no '{wanted}' entry: "
        f"tokens {items}, labels {labels}. Nothing was measured."
    )


def _ray_verb() -> Any:
    """The Ray SOP verb, set to find the closest point on the target surface.

    Minimum Distance ignores point normals, which is what lets a bare cloud of
    sample positions be measured, and it is a surface distance rather than a
    ray intersection. ``dotrans`` is off because the points do not need to
    move: only the distance and the primitive behind it are read back.

    Every parameter is checked against the verb's own defaults before it is
    set, so a rename in a future build is an error here rather than a silent
    fall back to Project Rays.
    """
    verb = hou.sopNodeTypeCategory().nodeVerb("ray")
    if verb is None:
        raise hou.OperationFailed(
            "This Houdini build has no 'ray' SOP verb, which is how these checks measure "
            "distances in one batch. Nothing was measured."
        )
    wanted: dict[str, Any] = {
        "method": _ray_menu_index("method", "minimum"),
        "dotrans": 0,
        "putdist": 1,
        "useprimnumattrib": 1,
        "primnumattrib": _HIT_PRIM_ATTRIB,
    }
    defaults = verb.parms()
    missing = sorted(name for name in wanted if name not in defaults)
    if missing:
        raise hou.OperationFailed(
            f"The 'ray' SOP verb on this build has no parameter(s) {missing}, so it cannot "
            "be set to measure minimum distance. Nothing was measured."
        )

    # Present on builds that can project whole primitives; the sample cloud has
    # none, so it is told to work on points. Best effort on purpose: a build
    # without the parameter should not be refused over an option it lacks.
    if "entity" in defaults:
        entity = _ray_menu_index("entity", "point", required=False)
        if entity is not None:
            wanted["entity"] = entity

    verb.setParms(wanted)
    return verb


def _require_point_attrib(geo: hou.Geometry, name: str, what: str) -> None:
    """Refuse to read a distance that is not there rather than return zeros."""
    if geo.findPointAttrib(name) is not None:
        return
    present = sorted(attrib.name() for attrib in geo.pointAttribs())
    raise hou.OperationFailed(
        f"The Ray pass produced no '{name}' point attribute, so there is no {what} to read. "
        f"Attributes present: {present}. Nothing was measured."
    )


def _batch_measure(work: hou.Geometry, prim_map: list[int]):
    """Measure every sample against *work* in one native call.

    The Ray verb's first input is the cloud of sample points and its second is
    the surface, and it neither adds nor reorders points, so what comes back
    lines up with the sampler's own list.

    The help notes that Minimum Distance can be wrong for packed primitives and
    for spheres or tubes with a non-uniform scale, because it works in their
    untransformed space. Neither can reach here: the target is always the
    polygon output of the Divide verb, and anything else is refused upstream.
    """

    def measure(points):
        if not points:
            return ([], [])
        cloud = hou.Geometry()
        cloud.createPoints([tuple(point) for point in points])
        cloud.incrementAllDataIds()
        answered = hou.Geometry()
        _ray_verb().execute(answered, [cloud, work])

        _require_point_attrib(answered, _DISTANCE_ATTRIB, "distance")
        _require_point_attrib(answered, _HIT_PRIM_ATTRIB, "primitive number")
        distances = list(answered.pointFloatAttribValues(_DISTANCE_ATTRIB))
        hits = list(answered.pointIntAttribValues(_HIT_PRIM_ATTRIB))
        if len(distances) != len(points) or len(hits) != len(points):
            raise hou.OperationFailed(
                f"The batch distance pass returned {len(distances)} distances and {len(hits)} "
                f"primitive numbers for {len(points)} samples, so they cannot be lined up. "
                "Nothing was measured."
            )
        return (distances, [_mapped_prim(prim_map, hit) for hit in hits])

    return measure


def _per_sample_measure(work: hou.Geometry, prim_map: list[int]):
    """The same measurement one ``nearestPrim`` call at a time, for checking.

    Not a fallback: see the cost measured above the mode list.
    """

    def measure(points):
        distances: list[float] = []
        hits: list[int | None] = []
        for point in points:
            prim, _u, _v, distance = work.nearestPrim(tuple(point))
            if prim is None:
                distances.append(math.inf)
                hits.append(None)
                continue
            distances.append(float(distance))
            hits.append(_mapped_prim(prim_map, prim.number()))
        return (distances, hits)

    return measure


def _mapped_prim(prim_map: list[int], number: int) -> int | None:
    return prim_map[number] if 0 <= number < len(prim_map) else None


def _surface_side(*, label: str, node_path: str, group: str, space: str) -> dict[str, Any]:
    """One side of the comparison: a triangulated copy and what it covers."""
    node = _get_sop_node(node_path, label)
    geo = _get_sop_geo(node_path)
    total_prims = int(geo.intrinsicValue("primitivecount"))
    if total_prims <= 0:
        raise hou.OperationFailed(
            f"{node_path} ({label}) has no primitives, so there is no surface to sample."
        )
    scope = _region_scope(geo, node_path, group, label)
    _check_scope_size(node_path, label, len(scope) if scope is not None else total_prims, group)

    matrix = _object_world_transform(node) if space == "world" else None
    triangulated = _triangulate(
        geo,
        node_path=node_path,
        label=label,
        keep=scope,
        total_prims=total_prims,
        matrix=matrix,
    )

    bad = [
        index
        for index, point in enumerate(triangulated["positions"])
        if not gm.is_finite_point(point)
    ]
    if bad:
        raise hou.OperationFailed(
            f"{node_path} ({label}) has {len(bad)} points whose position is not a finite "
            f"number (first: point {bad[0]}). A distance to a NaN is not a measurement."
        )

    surface = gm.SurfaceSet(triangles=triangulated["triangles"], prim_ids=triangulated["prim_ids"])
    area = surface.area()
    if area <= 0.0:
        raise hou.OperationFailed(
            f"{node_path} ({label}) encloses no area: every face in scope is degenerate, "
            "so there is no surface to sample."
        )

    return {
        "label": label,
        "node_path": node_path,
        "group": group,
        "triangulated": triangulated,
        "surface": surface,
        "area": area,
        "matrix": matrix,
        "prims": len(triangulated["source_prims"]),
    }


def _surface_scope_block(side: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {
        "prims": side["prims"],
        "triangles": len(side["surface"].triangles),
        "area": gm.significant(side["area"]),
    }
    if side["group"]:
        block["group"] = side["group"]
    untriangulated = side["triangulated"]["untriangulated"]
    if untriangulated:
        # Named, not hidden: these faces contributed no surface, so the
        # comparison covers less than the caller asked for.
        block["untriangulated"] = {"count": len(untriangulated), "prims": untriangulated[:10]}
    return block


def _surface_bbox_diagonal(*sides: dict[str, Any]) -> float:
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for side in sides:
        for tri in side["surface"].triangles:
            for corner in tri:
                for axis in range(3):
                    lo[axis] = min(lo[axis], corner[axis])
                    hi[axis] = max(hi[axis], corner[axis])
    if not all(math.isfinite(value) for value in lo + hi):
        return 0.0
    return math.sqrt(sum((hi[axis] - lo[axis]) ** 2 for axis in range(3)))


def _tolerance_list(tolerances: Any, diagonal: float) -> tuple[list[float], str]:
    """The tolerances to report coverage at, and where they came from."""
    if tolerances is None:
        auto = AUTO_TOLERANCE_FACTOR * diagonal
        if auto <= 0.0:
            auto = AUTO_TOLERANCE_FACTOR
        return [auto], f"auto ({AUTO_TOLERANCE_FACTOR:g} of the bounding box diagonal)"
    if isinstance(tolerances, (int, float)) and not isinstance(tolerances, bool):
        tolerances = [tolerances]
    if not isinstance(tolerances, (list, tuple)) or not tolerances:
        raise ValueError(
            "tolerances must be a number or a non-empty list of numbers in scene units, "
            f"not {tolerances!r}"
        )
    values: list[float] = []
    for entry in tolerances:
        number = _as_float(entry, "tolerances")
        if number < 0:
            raise ValueError(f"tolerances must be >= 0, not {number}")
        if number not in values:
            values.append(number)
    if len(values) > 4:
        raise ValueError(
            f"tolerances lists {len(values)} values; four is the most a receipt reports "
            "without becoming a table nobody reads."
        )
    return sorted(values), "given"


def _round_direction(block: dict[str, Any]) -> dict[str, Any]:
    """Significant figures, not decimal places -- see geometry_math.significant."""
    rounded = dict(block)
    for key in ("mean", "p95", "max"):
        if key in rounded:
            rounded[key] = gm.significant(rounded[key])
    return rounded


def _compare_surfaces(
    *,
    a: str,
    b: str,
    samples: int = DEFAULT_SURFACE_SAMPLES,
    seed: int = 0,
    tolerances: Any = None,
    region_group_a: str | None = None,
    region_group_b: str | None = None,
    space: str = "sop",
    query_mode: str = "batch",
    max_list: int = 5,
    dump_path: str | None = None,
    schema_version: int | None = None,
    **_,
) -> dict[str, Any]:
    """How far apart two polygon surfaces are, with no shared topology needed.

    Both sides are triangulated once by the native Divide verb on a detached
    copy; the samples, the closest-point queries and the reported prim ids all
    come from that one triangulation, which is what makes a surface compared
    with itself measure zero however non-planar its faces are.

    Samples are drawn in proportion to triangle area from a seeded stratified
    sequence, and each is measured to the closest point on the other surface's
    triangles -- not to its nearest vertex, which would read low on a dense
    mesh and high on a coarse one for the same shapes.

    Both directions are reported because one alone cannot see a missing region:
    every sample of a surface that stops short still lands on the larger one.
    ``coverage`` is the share of samples within a tolerance -- a one-way
    containment rate, not a similarity score, and not a verdict.

    ``space='sop'`` compares raw SOP positions and so assumes both nodes sit
    under the same object transform; the receipt warns when they do not.
    ``space='world'`` puts both through their object transforms first.

    *query_mode* selects how the distances are taken, and must not change
    them: ``batch`` (the default) runs the Ray SOP verb in Minimum Distance
    mode over every sample in one native call; ``per_sample`` calls
    ``hou.Geometry.nearestPrim`` once per sample and exists only so the two can
    be checked against each other. It is capped at a couple of thousand
    samples and is not a fallback -- one call per sample measured about 32 ms
    against a dense target.

    When a face produces no triangles it is excluded from the measurement and
    ``scope_complete`` goes false: the coverage below is then over what was
    measured, not over the whole input.
    """
    started = time.perf_counter()
    _check_schema_version(schema_version)

    a_path = as_text(a, "a").strip()
    b_path = as_text(b, "b").strip()
    if not a_path or not b_path:
        raise ValueError("a and b must both name a SOP node, for example /obj/geo1/subdivide1")

    count = as_int(samples, "samples")
    if not 1 <= count <= MAX_SURFACE_SAMPLES:
        raise ValueError(
            f"samples must be between 1 and {MAX_SURFACE_SAMPLES}, not {count}. "
            "Each sample is one closest-point query per direction."
        )
    seed_value = as_int(seed, "seed")
    cap = max(0, min(as_int(max_list, "max_list"), 100))

    space_name = as_text(space, "space").strip().lower() or "sop"
    if space_name not in SURFACE_SPACES:
        raise ValueError(f"space must be one of {list(SURFACE_SPACES)}, not '{space_name}'")
    mode = as_text(query_mode, "query_mode").strip().lower() or "batch"
    if mode not in QUERY_MODES:
        raise ValueError(f"query_mode must be one of {list(QUERY_MODES)}, not '{mode}'")
    if mode == "per_sample" and count > MAX_PER_SAMPLE_QUERIES:
        raise ValueError(
            f"query_mode='per_sample' is a verification oracle, not a fallback: one HOM call "
            f"per sample measured about 32 ms against a dense target, so it is capped at "
            f"{MAX_PER_SAMPLE_QUERIES} samples, not {count}. The default batch mode has no "
            "such cost."
        )

    side_a = _surface_side(
        label="a",
        node_path=a_path,
        group=as_text(region_group_a, "region_group_a").strip(),
        space=space_name,
    )
    side_b = _surface_side(
        label="b",
        node_path=b_path,
        group=as_text(region_group_b, "region_group_b").strip(),
        space=space_name,
    )

    warnings: list[str] = []
    if space_name == "sop":
        matrix_a = _object_world_transform(_get_sop_node(a_path, "a"))
        matrix_b = _object_world_transform(_get_sop_node(b_path, "b"))
        if matrix_a is not None and matrix_b is not None and matrix_a != matrix_b:
            warnings.append(
                "a and b sit under different object transforms, and space='sop' ignores "
                "them. Pass space='world' to compare where the surfaces actually are."
            )

    diagonal = _surface_bbox_diagonal(side_a, side_b)
    tolerance_values, tolerance_source = _tolerance_list(tolerances, diagonal)

    build_measure = _batch_measure if mode == "batch" else _per_sample_measure
    report, samples = gm.compare_surface_sets(
        side_a["surface"],
        side_b["surface"],
        samples=count,
        seed=seed_value,
        tolerances=tolerance_values,
        max_list=cap,
        a_measure=build_measure(
            side_a["triangulated"]["geometry"], side_a["triangulated"]["prim_ids"]
        ),
        b_measure=build_measure(
            side_b["triangulated"]["geometry"], side_b["triangulated"]["prim_ids"]
        ),
    )

    excluded = {
        label: len(side["triangulated"]["untriangulated"])
        for label, side in (("a", side_a), ("b", side_b))
        if side["triangulated"]["untriangulated"]
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "a": a_path,
        "b": b_path,
        "space": space_name,
        "algorithm": (
            "native Divide triangulation, area-weighted stratified sampling, "
            + (
                "Ray SOP verb in minimum-distance mode over the whole batch"
                if mode == "batch"
                else "hou.nearestPrim per sample (verification oracle)"
            )
        ),
        "samples_per_side": count,
        "seed": seed_value,
        "scope": {"a": _surface_scope_block(side_a), "b": _surface_scope_block(side_b)},
        # False means faces were left out, so everything below describes the
        # surface that was measured and not the whole input.
        "scope_complete": not excluded,
        "a_to_b": _round_direction(report["a_to_b"]),
        "b_to_a": _round_direction(report["b_to_a"]),
        "max_both": gm.significant(report["max_both"]),
        "coverage": [
            {key: (value if key == "tolerance" else round(value, 4)) for key, value in row.items()}
            for row in report["coverage"]
        ],
        "tolerance_source": tolerance_source,
    }
    if excluded:
        receipt["excluded_faces"] = excluded
        warnings.append(
            "faces that produced no triangles were left out of the sampled surface, so "
            "coverage is over the measured scope and not over the whole input."
        )
    if warnings:
        receipt["warnings"] = warnings

    dump_target = as_text(dump_path, "dump_path").strip()
    if dump_target:
        try:
            _write_dump(
                dump_target,
                {
                    "receipt": receipt,
                    "samples": {
                        direction: measured.records() for direction, measured in samples.items()
                    },
                },
            )
        except OSError as exc:
            receipt["dump_error"] = f"could not write {dump_target}: {exc}"
        else:
            receipt["dump"] = dump_target

    receipt["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return receipt


register_handler("modeling.compare_surfaces", _compare_surfaces)


###### modeling.get_uv_report

# The overlap pass is the expensive half: a broadphase over triangle boxes and
# an exact area test per surviving pair. Both budgets below are reported when
# they bite, because a check that quietly stopped early and printed "0
# overlaps" is worse than one that says it did not finish.
UV_MAX_TRIANGLES = 400_000
UV_MAX_OVERLAP_PAIRS = 2_000_000

# In UV units squared. Below this an "overlap" is the floating-point residue of
# clipping two triangles that share an edge. At a 2048 map this is about 4e-6
# of a pixel, so a real overlap of a thousandth of a pixel still reports.
DEFAULT_OVERLAP_AREA_TOLERANCE = 1e-12

# Two triangles count as stacked only when all three corners coincide this
# closely in UV.
DEFAULT_STACK_TOLERANCE = 1e-6


def _uv_thresholds(
    overlap_area_tolerance: Any,
    stack_tolerance: Any,
    texture_resolution: Any,
    stretch_threshold: Any,
) -> tuple[float, float, int | None, float | None]:
    """Check the four numbers a UV report takes, naming whichever is wrong."""
    area_tolerance = (
        DEFAULT_OVERLAP_AREA_TOLERANCE
        if overlap_area_tolerance is None
        else _as_float(overlap_area_tolerance, "overlap_area_tolerance")
    )
    if area_tolerance < 0:
        raise ValueError(f"overlap_area_tolerance must be >= 0, not {area_tolerance}")

    stack_eps = (
        DEFAULT_STACK_TOLERANCE
        if stack_tolerance is None
        else _as_float(stack_tolerance, "stack_tolerance")
    )
    if stack_eps < 0:
        raise ValueError(f"stack_tolerance must be >= 0, not {stack_eps}")

    resolution = None
    if texture_resolution is not None:
        resolution = as_int(texture_resolution, "texture_resolution")
        if resolution <= 0:
            raise ValueError(f"texture_resolution must be a positive pixel count, not {resolution}")

    threshold = None
    if stretch_threshold is not None:
        threshold = _as_float(stretch_threshold, "stretch_threshold")
        if threshold <= 1.0:
            raise ValueError(
                "stretch_threshold is a ratio above 1.0: a triangle counts when its density "
                "falls outside [1/t, t] times the mean, or its anisotropy exceeds t. "
                f"Not {threshold}"
            )
    return area_tolerance, stack_eps, resolution, threshold


def _get_uv_report(
    *,
    node_path: str,
    uv_attribute: str = "uv",
    group: str | None = None,
    overlap_area_tolerance: float | None = None,
    allow_stacking: bool = False,
    stack_tolerance: float | None = None,
    texture_resolution: int | None = None,
    stretch_threshold: float | None = None,
    max_list: int = 20,
    dump_path: str | None = None,
    schema_version: int | None = None,
    check_overlaps: bool = True,
    **_,
) -> dict[str, Any]:
    """UV quality of a polygon mesh: islands, winding, overlaps, distortion.

    Faces are triangulated by the native Divide verb and each triangle corner
    keeps the UV of the corner it came from. That is the point of doing it this
    way: choosing UV diagonals independently of the 3D ones can lay a folded
    mapping out flat and report it clean.

    An island is a set of faces joined by edges whose UVs match at both ends.
    An overlap is a pair of UV triangles with intersection area above
    *overlap_area_tolerance*; triangles sharing only an edge intersect in
    exactly zero area and never appear. ``allow_stacking`` forgives only pairs
    whose three corners coincide -- deliberately stacked shells -- and cannot
    hide a partial overlap, which is the accidental kind. Stacked pairs are
    counted either way.

    ``uv_area`` is the sum of the triangles' UV areas, not a union. Distortion
    is two separate things: ``density``, the texel scale relative to this
    mesh's own area-weighted mean, and ``anisotropy``, the ratio of the two
    singular values of each triangle's mapping -- the one that catches a shape
    squashed along one axis at unchanged area. A triangle whose UVs collapse
    to a line or a point has neither, and is counted as collapsed.

    The overlap pass is most of the cost on a large mesh. ``check_overlaps=False``
    skips it, and the overlaps block then says ``checked: false`` instead of
    carrying numbers that would read as zero overlaps.

    ``fingerprint`` names the geometry measured: its point count and a digest
    of P in the form edit_points uses, the primitive count, and a digest of the
    UV values, so two receipts can be told apart when only a layout changed.
    """
    started = time.perf_counter()
    _check_schema_version(schema_version)
    if not isinstance(check_overlaps, bool):
        raise ValueError(f"check_overlaps must be true or false, not {check_overlaps!r}")

    path = as_text(node_path, "node_path").strip()
    if not path:
        raise ValueError("node_path must name a SOP node, for example /obj/geo1/uvlayout1")
    attribute = as_text(uv_attribute, "uv_attribute").strip() or "uv"
    group_name = as_text(group, "group").strip()
    cap = max(0, min(as_int(max_list, "max_list"), 1000))
    area_tolerance, stack_eps, resolution, threshold = _uv_thresholds(
        overlap_area_tolerance, stack_tolerance, texture_resolution, stretch_threshold
    )

    geo = _get_sop_geo(path)
    total_prims = int(geo.intrinsicValue("primitivecount"))
    if total_prims <= 0:
        raise hou.OperationFailed(f"{path} has no primitives, so there are no UVs to report on.")
    scope = _region_scope(geo, path, group_name, "group")
    _check_scope_size(
        path, "node_path", len(scope) if scope is not None else total_prims, group_name
    )

    triangulated = _triangulate(
        geo,
        node_path=path,
        label="node_path",
        keep=scope,
        total_prims=total_prims,
        matrix=None,
        uv_attribute=attribute,
    )
    faces = triangulated["faces"]
    corner_uvs = triangulated["corner_uvs"]

    # A face can have one corner that is not a number and others that are, so
    # the split is by triangle rather than by face: a triangle with a
    # non-finite UV is left out of every measurement -- islands included, or it
    # would join nothing and count as one of its own -- and its prim is named.
    nonfinite_prims: set[int] = set()
    finite_faces: list[tuple[int, list[int]]] = []
    finite_uvs: list[list[tuple[float, float]]] = []
    triangles: list[gm.UVTriangle] = []
    for index, uvs in enumerate(corner_uvs):
        if not all(math.isfinite(uv[0]) and math.isfinite(uv[1]) for uv in uvs):
            nonfinite_prims.add(faces[index][0])
            continue
        triangles.append(
            gm.UVTriangle(
                prim=faces[index][0],
                face_index=len(triangles),
                uv=(uvs[0], uvs[1], uvs[2]),
                world=triangulated["triangles"][index],
            )
        )
        finite_faces.append(faces[index])
        finite_uvs.append(uvs)
    nonfinite = sorted(nonfinite_prims)

    labels, island_count = gm.uv_island_labels(finite_faces, finite_uvs)
    stats = gm.uv_distortion_stats(triangles, threshold=threshold)

    island_area: dict[int, float] = {}
    for tri in triangles:
        label = labels[tri.face_index]
        island_area[label] = island_area.get(label, 0.0) + gm.signed_triangle_area_2d(*tri.uv)
    mirrored_islands = sum(1 for area in island_area.values() if area < 0.0)

    # Faces excluded here are faces nothing below has looked at, so the
    # overlap block carries the fact too: a consumer reading only
    # ``overlaps.status == "complete"`` must not conclude the UVs are clean.
    excluded: dict[str, int] = {}
    if triangulated["untriangulated"]:
        excluded["untriangulated_faces"] = len(triangulated["untriangulated"])
    if nonfinite:
        excluded["nonfinite_uv_faces"] = len(nonfinite)

    if check_overlaps:
        overlaps, found, unintended = _uv_overlap_block(
            triangles,
            area_tolerance=area_tolerance,
            stack_eps=stack_eps,
            allow_stacking=allow_stacking,
            resolution=resolution,
            cap=cap,
            scope_complete=not excluded,
        )
    else:
        overlaps = {"checked": False, "status": "not_checked", "reason": "check_overlaps=False"}
        found, unintended = None, []

    area_block: dict[str, Any] = {
        "uv_area": gm.significant(stats["uv_area"]),
        "measure": "sum (not union)",
        "world_area": gm.significant(stats["world_area"]),
        "tiles": gm.udim_tiles(triangles)[:16],
    }
    if resolution:
        area_block["texture_resolution"] = resolution
        area_block["uv_area_px2"] = gm.significant(stats["uv_area"] * resolution * resolution)

    # Of the node's geometry as a whole, like the P check: the scope is named
    # beside it, and a digest of a subset would match nothing a caller holds.
    read_uvs = (
        geo.vertexFloatAttribValuesAsString
        if triangulated["uv_owner"] == "vertex"
        else geo.pointFloatAttribValuesAsString
    )
    fingerprint = {
        "geometry": geometry_fingerprint(
            int(geo.intrinsicValue("pointcount")), geo.pointFloatAttribValuesAsString("P")
        ),
        "prims": total_prims,
        "uv": hashlib.blake2b(read_uvs(attribute), digest_size=4).hexdigest(),
    }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "node_path": path,
        "scope": group_name or "all",
        "fingerprint": fingerprint,
        "uv": {
            "attribute": attribute,
            "owner": triangulated["uv_owner"],
            "components": triangulated["uv_size"],
        },
        "counts": {
            "faces": len(triangulated["source_prims"]),
            "uv_triangles": len(triangles),
            "islands": island_count,
        },
        # False means faces were left out of every number below.
        "scope_complete": not excluded,
        "winding": {
            "mirrored_triangles": stats["mirrored_triangles"],
            "mirrored_islands": mirrored_islands,
        },
        "overlaps": overlaps,
        "area": area_block,
    }
    if excluded:
        report["excluded_faces"] = excluded

    if triangulated["untriangulated"]:
        report["untriangulated"] = {
            "faces": len(triangulated["untriangulated"]),
            "prims": triangulated["untriangulated"][:cap],
        }
    if nonfinite:
        report["nonfinite_uv"] = {"faces": len(nonfinite), "prims": nonfinite[:cap]}
    collapsed = stats.get("collapsed_uv") or []
    if collapsed:
        # Zero UV area over real surface area: no density and no anisotropy to
        # report for these, which is why they are a count and not a number.
        report["collapsed_uv"] = {
            "triangles": stats["collapsed_uv_count"],
            "prims": collapsed[:cap],
        }
    if stats.get("degenerate_world_triangles"):
        report["degenerate_world_triangles"] = stats["degenerate_world_triangles"]
    _add_distortion(report, stats, threshold)

    dump_target = as_text(dump_path, "dump_path").strip()
    if dump_target:
        payload = {
            "receipt": report,
            # One label per triangle, in the order the triangulation produced
            # them, because that is the unit everything else here counts.
            "island_per_triangle": labels,
            "island_signed_area": {str(k): v for k, v in sorted(island_area.items())},
            "collapsed_uv_prims": collapsed,
            "nonfinite_uv_prims": nonfinite,
            "untriangulated_prims": triangulated["untriangulated"],
        }
        if found is not None:
            # Absent rather than empty when the pass did not run: an empty
            # list here would read as a layout with no overlaps.
            payload["overlapping_pairs"] = sorted(unintended, key=lambda e: -e["area"])
            payload["stacked_pairs"] = found["stacked"]
        try:
            _write_dump(dump_target, payload)
        except OSError as exc:
            report["dump_error"] = f"could not write {dump_target}: {exc}"
        else:
            report["dump"] = dump_target

    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return report


def _uv_overlap_block(
    triangles: list,
    *,
    area_tolerance: float,
    stack_eps: float,
    allow_stacking: bool,
    resolution: int | None,
    cap: int,
    scope_complete: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    """The overlaps section, plus the raw finding for the dump.

    Returns ``(block, finding, unintended pairs)``; *finding* is None when the
    check did not run, and the block says so rather than reporting zero.

    ``status`` is about the pair budget; ``scope`` is about which faces were in
    the check at all. They are separate because they fail separately, and
    because "complete" on its own would otherwise read as "all of the UVs".
    """
    scope = "all_faces" if scope_complete else "measured_faces_only"
    if len(triangles) > UV_MAX_TRIANGLES:
        return (
            {
                "checked": False,
                "status": "skipped",
                "scope": scope,
                "reason": (
                    f"{len(triangles)} UV triangles is above the {UV_MAX_TRIANGLES} this check "
                    "runs. Pass a primitive group to check one region at a time."
                ),
            },
            None,
            [],
        )

    found = gm.uv_overlaps(
        triangles,
        area_tolerance=area_tolerance,
        stack_tolerance=stack_eps,
        max_pairs=UV_MAX_OVERLAP_PAIRS,
    )
    unintended = list(found["overlapping"])
    unintended_area = found["overlap_area"]
    unintended_pairs = found["overlapping_count"]
    if not allow_stacking:
        # Stacking was not asked for, so a stacked pair is still an overlap.
        unintended += found["stacked"]
        unintended_area += found["stacked_area"]
        unintended_pairs += found["stacked_count"]

    block: dict[str, Any] = {
        "checked": True,
        "status": "incomplete" if found["truncated"] else "complete",
        "scope": scope,
        "allow_stacking": bool(allow_stacking),
        "area_tolerance": area_tolerance,
        "pairs": unintended_pairs,
        "area": gm.significant(unintended_area),
        "stacked_pairs": found["stacked_count"],
        "stacked_area": gm.significant(found["stacked_area"]),
        "candidate_pairs": found["candidate_pairs"],
    }
    if found["records_truncated"]:
        # The counts and areas above are exact; the lists are the largest ones.
        block["listed"] = len(unintended)
    if found["truncated"]:
        block["reason"] = (
            f"the broadphase hit its {UV_MAX_OVERLAP_PAIRS} candidate pair budget, so the "
            "pairs below are some of the overlaps, not all of them."
        )
    if unintended and cap:
        block["worst"] = [
            {"prims": entry["prims"], "area": gm.significant(entry["area"])}
            for entry in sorted(unintended, key=lambda e: -e["area"])[:cap]
        ]
    if resolution:
        block["area_px2"] = gm.significant(unintended_area * resolution * resolution)
    return block, found, unintended


def _add_distortion(report: dict[str, Any], stats: dict[str, Any], threshold: float | None) -> None:
    """Density variation and anisotropy, reported apart because they differ.

    Every averaged figure carries ``area_weighted`` in its name and every
    extreme does not, so a reader never has to guess which one they have: a
    count-based average over triangles would move when the same surface is
    retessellated.
    """
    if "density" in stats:
        density = stats["density"]
        block = {
            "area_weighted_mean": gm.significant(density["area_weighted_mean"]),
            "ratio_min": round(density["ratio_min"], 4),
            "ratio_p95_area_weighted": round(density["ratio_p95_area_weighted"], 4),
            "ratio_max": round(density["ratio_max"], 4),
            "abs_log2_area_weighted_mean": round(density["abs_log2_area_weighted_mean"], 4),
        }
        if "outside_threshold" in density:
            block["threshold"] = threshold
            block["outside_threshold"] = density["outside_threshold"]
            block["outside_threshold_area"] = gm.significant(density["outside_threshold_area"])
        report["density"] = block

    if "anisotropy" in stats:
        anisotropy = stats["anisotropy"]
        block = {
            "area_weighted_mean": round(anisotropy["area_weighted_mean"], 4),
            "p95_area_weighted": round(anisotropy["p95_area_weighted"], 4),
            "max": gm.significant(anisotropy["max"]),
        }
        if "undefined" in anisotropy:
            block["undefined"] = anisotropy["undefined"]
        if "over_threshold" in anisotropy:
            block["threshold"] = threshold
            block["over_threshold"] = anisotropy["over_threshold"]
            block["over_threshold_area"] = gm.significant(anisotropy["over_threshold_area"])
        report["anisotropy"] = block


register_handler("modeling.get_uv_report", _get_uv_report)
