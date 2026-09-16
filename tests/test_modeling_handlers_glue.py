"""End-to-end tests for the two new handlers against a stand-in for ``hou``.

The maths has its own tests in test_geometry_math.py; what is left is the glue,
and the glue is where the quiet mistakes live: a vertex attribute read with
point numbers, a group filter applied to faces but not to their vertices, a
prim id that no longer maps back to the face the caller knows, a receipt key
that stopped matching what the tool promises.

The stub answers only the HOM calls these handlers make, and its "Divide verb"
fans each face into triangles carrying the source prim attribute and the
corner UVs -- which is what the real verb does for the convex fixtures used
here. It stands in for the plumbing, not for Houdini's triangulator: whether
the real verb preserves attributes and handles concave and non-planar faces is
what the live probe is for.
"""

from __future__ import annotations

# Built-in
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

from fxhoudinimcp_server.handlers import modeling_handlers as mh  # noqa: E402

###### numpy, when it is not installed
#
# The mesh report reads positions through numpy, which ships with Houdini's
# Python but is not a dependency of this package, so the development
# environment has none. These four methods are all it uses; when the real numpy
# is present it is used instead. It is installed per test by the fixture below,
# never at import time: pytest reads sys.modules["numpy"] on every comparison,
# and a stand-in left lying about would follow this file into the whole suite.

try:  # pragma: no cover - depends on the environment, not on the code
    import numpy  # noqa: F401

    _numpy = None
except ImportError:  # pragma: no cover

    class _Row(list):
        def tolist(self):
            return list(self)

    class _Rows:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, key):
            if isinstance(key, (list, tuple)):
                return _Rows([self.rows[index] for index in key])
            return self.rows[key]

        def tolist(self):
            return [list(row) for row in self.rows]

        def min(self, axis=0):
            return _Row([min(row[i] for row in self.rows) for i in range(3)])

        def max(self, axis=0):
            return _Row([max(row[i] for row in self.rows) for i in range(3)])

    class _Flat:
        def __init__(self, values):
            self.values = list(values)

        def reshape(self, rows, columns):
            assert (rows, columns) == (-1, 3), "only the xyz reshape is emulated"
            return _Rows([self.values[i : i + 3] for i in range(0, len(self.values), 3)])

    _numpy = type(sys)("numpy")
    _numpy.float64 = float
    _numpy.asarray = lambda values, dtype=None: _Flat(values)


def approx(expected, tolerance=1e-9):
    """``pytest.approx`` in one line: with a stand-in numpy installed pytest's
    own comparison takes a path through numpy this stub does not have."""

    class _Close:
        def __eq__(self, other):
            return abs(float(other) - float(expected)) <= tolerance

        def __repr__(self):
            return f"approx({expected!r} +- {tolerance!r})"

    return _Close()


###### A stand-in for the corner of HOM these handlers use


class OperationFailed(Exception):
    """What the handlers raise; a real hou.OperationFailed is also an Exception."""


class Enum:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


# One instance per type: hou.attribData.Float is a singleton and the handlers
# compare against it by identity.
FLOAT = Enum("Float")
INT = Enum("Int")
PRIM_ATTRIB = Enum("Prim")


class StubAttrib:
    def __init__(self, name, size, data_type):
        self._name, self._size, self._type = name, size, data_type

    def name(self):
        return self._name

    def size(self):
        return self._size

    def dataType(self):
        return self._type


class StubPrim:
    def __init__(self, number):
        self._number = number

    def number(self):
        return self._number


class StubGroup:
    def __init__(self, name, prims):
        self._name, self._prims = name, prims

    def name(self):
        return self._name

    def prims(self):
        return [StubPrim(number) for number in self._prims]


class StubMatrix:
    """Enough of hou.Matrix4 for a translation and an equality test."""

    def __init__(self, translate=(0.0, 0.0, 0.0)):
        self.translate = tuple(translate)

    def __eq__(self, other):
        return isinstance(other, StubMatrix) and self.translate == other.translate

    def __hash__(self):
        return hash(self.translate)


class StubGeometry:
    """Points, polygons, one optional UV attribute, one prim int attribute.

    ``saveToFile`` writes the ``.geo`` JSON layout Houdini 22 writes, so the
    handlers' own parser reads this back: the round trip is part of what is
    being tested. Anything the handlers do not call is simply absent, so a
    wrong call raises instead of returning a mock that looks like an answer.
    """

    def __init__(
        self, points=(), faces=(), vertex_uv=None, point_uv=None, prim_groups=None, curves=()
    ):
        self.points_list = [list(map(float, p)) for p in points]
        self.faces = [list(face) for face in faces]
        # Open polylines: Divide passes them through untouched, which is the
        # case the handlers have to refuse rather than silently drop.
        self.curves = [list(curve) for curve in curves]
        self.vertex_uv = list(vertex_uv) if vertex_uv else None
        self.point_uv = list(point_uv) if point_uv else None
        self.groups = dict(prim_groups or {})
        self.prim_attribs: dict[str, list[int]] = {}
        self.point_float_attribs: dict[str, list[float]] = {}
        self.point_int_attribs: dict[str, list[int]] = {}
        self.drop_prims: set[int] = set()
        # SideFX: geometry edited outside a Python SOP must have its data ids
        # incremented before a verb sees it, or the verb may reuse a cached
        # result. The stub verbs below refuse dirty input, which is how the
        # handler's invalidation is tested rather than assumed.
        self.dirty = False
        self.data_id_bumps = 0

    ###### Reading

    def saveToFile(self, path):
        indices = [point for face in self.faces for point in face]
        curve_start = len(indices)
        indices += [point for curve in self.curves for point in curve]
        primitives = [
            [
                ["type", "Polygon_run"],
                [
                    "startvertex",
                    0,
                    "nprimitives",
                    len(self.faces),
                    "nvertices",
                    [len(face) for face in self.faces],
                ],
            ]
        ]
        if self.curves:
            primitives.append(
                [
                    ["type", "PolygonCurve_run"],
                    [
                        "startvertex",
                        curve_start,
                        "nprimitives",
                        len(self.curves),
                        "nvertices",
                        [len(curve) for curve in self.curves],
                    ],
                ]
            )
        document = [
            "fileversion",
            "22.0.368",
            "pointcount",
            len(self.points_list),
            "vertexcount",
            len(indices),
            "primitivecount",
            len(self.faces) + len(self.curves),
            "topology",
            ["pointref", ["indices", indices]],
            "primitives",
            primitives,
        ]
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(document, stream)

    def intrinsicValue(self, name):
        return {
            "pointcount": len(self.points_list),
            "primitivecount": len(self.faces) + len(self.curves),
            "vertexcount": sum(len(face) for face in self.faces)
            + sum(len(curve) for curve in self.curves),
        }[name]

    def pointFloatAttribValues(self, name):
        if name == "P":
            return [value for point in self.points_list for value in point]
        if name == "uv" and self.point_uv is not None:
            return [value for uv in self.point_uv for value in uv]
        if name in self.point_float_attribs:
            return list(self.point_float_attribs[name])
        raise AssertionError(f"unexpected point attribute read: {name}")

    def pointIntAttribValues(self, name):
        if name in self.point_int_attribs:
            return list(self.point_int_attribs[name])
        raise AssertionError(f"unexpected point attribute read: {name}")

    def vertexFloatAttribValues(self, name):
        assert name == "uv" and self.vertex_uv is not None
        return [value for uv in self.vertex_uv for value in uv]

    def primIntAttribValues(self, name):
        return list(self.prim_attribs.get(name, []))

    def findVertexAttrib(self, name):
        if name == "uv" and self.vertex_uv is not None:
            return StubAttrib("uv", len(self.vertex_uv[0]), FLOAT)
        return None

    def findPointAttrib(self, name):
        if name == "uv" and self.point_uv is not None:
            return StubAttrib("uv", len(self.point_uv[0]), FLOAT)
        if name in self.point_float_attribs:
            return StubAttrib(name, 1, FLOAT)
        if name in self.point_int_attribs:
            return StubAttrib(name, 1, INT)
        return None

    def vertexAttribs(self):
        return [StubAttrib("uv", 2, FLOAT)] if self.vertex_uv else []

    def pointAttribs(self):
        attribs = [StubAttrib("P", 3, FLOAT)]
        if self.point_uv:
            attribs.append(StubAttrib("uv", 2, FLOAT))
        attribs.extend(StubAttrib(name, 1, FLOAT) for name in self.point_float_attribs)
        attribs.extend(StubAttrib(name, 1, INT) for name in self.point_int_attribs)
        return attribs

    def findPrimGroup(self, name):
        prims = self.groups.get(name)
        return StubGroup(name, prims) if prims is not None else None

    def primGroups(self):
        return [StubGroup(name, prims) for name, prims in self.groups.items()]

    def prim(self, index):
        return StubPrim(index)

    ###### Writing, on a detached copy only

    def merge(self, other):
        self.dirty = True
        offset = len(self.points_list)
        self.points_list.extend([list(p) for p in other.points_list])
        self.faces.extend([[p + offset for p in face] for face in other.faces])
        if other.vertex_uv is not None:
            self.vertex_uv = (self.vertex_uv or []) + list(other.vertex_uv)
        if other.point_uv is not None:
            self.point_uv = (self.point_uv or []) + list(other.point_uv)
        self.curves.extend([[p + offset for p in curve] for curve in other.curves])
        self.drop_prims |= {
            index + len(self.faces) - len(other.faces) for index in other.drop_prims
        }
        self.groups.update(other.groups)

    def addAttrib(self, attrib_type, name, default):
        self.dirty = True
        assert attrib_type is PRIM_ATTRIB
        self.prim_attribs[name] = [default] * (len(self.faces) + len(self.curves))

    def setPrimIntAttribValues(self, name, values):
        self.dirty = True
        assert len(values) == len(self.faces) + len(self.curves), "one value per primitive"
        self.prim_attribs[name] = list(values)

    def deletePrims(self, prims):
        self.dirty = True
        doomed = {prim.number() for prim in prims}
        keep = [index for index in range(len(self.faces)) if index not in doomed]
        vertex_start = []
        running = 0
        for face in self.faces:
            vertex_start.append(running)
            running += len(face)
        if self.vertex_uv is not None:
            self.vertex_uv = [
                self.vertex_uv[vertex_start[index] + corner]
                for index in keep
                for corner in range(len(self.faces[index]))
            ]
        for name, values in self.prim_attribs.items():
            self.prim_attribs[name] = [values[index] for index in keep]
        self.faces = [self.faces[index] for index in keep]

    def transform(self, matrix):
        self.dirty = True
        dx, dy, dz = matrix.translate
        self.points_list = [[p[0] + dx, p[1] + dy, p[2] + dz] for p in self.points_list]

    def createPoints(self, positions):
        self.dirty = True
        start = len(self.points_list)
        self.points_list.extend([list(map(float, p)) for p in positions])
        return [StubPrim(index) for index in range(start, len(self.points_list))]

    def incrementAllDataIds(self):
        self.dirty = False
        self.data_id_bumps += 1

    ###### The one query the comparison makes
    #
    # Every fixture here is a flat sheet in a plane of constant Y, so the
    # distance to it is the distance to an axis-aligned rectangle, which can be
    # written down. The assertion below keeps a future fixture from silently
    # getting a wrong answer out of that shortcut. Both the batch wrangle and
    # nearestPrim answer from here, so the two modes agree in the stub for the
    # same reason they should agree in Houdini.

    def surface_distance(self, position):
        used = sorted({point for face in self.faces for point in face})
        ys = {round(self.points_list[p][1], 9) for p in used}
        assert len(ys) == 1, "the stub oracle only knows flat sheets"
        y = ys.pop()
        xs = [self.points_list[p][0] for p in used]
        zs = [self.points_list[p][2] for p in used]
        dx = max(min(xs) - position[0], 0.0, position[0] - max(xs))
        dz = max(min(zs) - position[2], 0.0, position[2] - max(zs))
        dy = position[1] - y

        def centroid_distance(index):
            face = self.faces[index]
            cx = sum(self.points_list[p][0] for p in face) / len(face)
            cz = sum(self.points_list[p][2] for p in face) / len(face)
            return (cx - position[0]) ** 2 + (cz - position[2]) ** 2

        nearest = min(range(len(self.faces)), key=centroid_distance)
        return (math.sqrt(dx * dx + dy * dy + dz * dz), nearest)

    def nearestPrim(self, position):
        if not self.faces:
            return (None, 0.0, 0.0, 0.0)
        distance, nearest = self.surface_distance(position)
        return (StubPrim(nearest), 0.0, 0.0, distance)


class StubVerb:
    """A stand-in for the Divide verb: fans each face into triangles.

    Carries the source prim attribute onto every triangle and keeps each
    corner's UV, which is the contract the handlers rely on. A face with fewer
    than three corners, or one marked degenerate by the fixture, produces no
    triangles -- the case the handlers have to report rather than hide.
    """

    def __init__(self):
        self.set = {}

    def parms(self):
        return {"convex": 1, "numsides": 3, "avoidsmallangles": 0}

    def setParms(self, parms):
        self.set = dict(parms)

    def execute(self, dest, inputs):
        assert self.set.get("numsides") == 3, "the verb must be told to make triangles"
        _refuse_stale(inputs)
        source = inputs[0]
        dest.points_list = [list(p) for p in source.points_list]
        dest.faces = []
        dest.curves = [list(curve) for curve in source.curves]
        dest.vertex_uv = [] if source.vertex_uv is not None else None
        dest.point_uv = list(source.point_uv) if source.point_uv is not None else None
        dest.prim_attribs = {name: [] for name in source.prim_attribs}

        start = 0
        for index, face in enumerate(source.faces):
            corners = list(range(start, start + len(face)))
            start += len(face)
            if len(face) < 3 or index in source.drop_prims:
                continue
            for corner in range(1, len(face) - 1):
                dest.faces.append([face[0], face[corner], face[corner + 1]])
                if dest.vertex_uv is not None:
                    dest.vertex_uv.extend(
                        [
                            source.vertex_uv[corners[0]],
                            source.vertex_uv[corners[corner]],
                            source.vertex_uv[corners[corner + 1]],
                        ]
                    )
                for name, values in source.prim_attribs.items():
                    dest.prim_attribs[name].append(values[index])


class StubMenuTemplate:
    """A menu parameter template, for looking an index up by name."""

    def __init__(self, items, labels):
        self._items, self._labels = tuple(items), tuple(labels)

    def menuItems(self):
        return self._items

    def menuLabels(self):
        return self._labels


class StubParmTemplateGroup:
    # The method menu is what the parent read off this build; the entity menu
    # is a plausible stand-in, here to prove the lookup picks an index by name
    # rather than by position.
    TEMPLATES = {
        "method": StubMenuTemplate(("minimum", "project"), ("Minimum Distance", "Project Rays")),
        "entity": StubMenuTemplate(("points", "prims"), ("Points", "Primitives")),
    }

    def find(self, name):
        return self.TEMPLATES.get(name)


class StubNodeType:
    def parmTemplateGroup(self):
        return StubParmTemplateGroup()


class StubRayVerb:
    """A stand-in for the Ray verb in minimum-distance mode.

    It answers what the real one answers -- the distance to the closest point
    on the second input's surface and the primitive it belongs to -- and
    refuses to run in any other configuration, so setting Project Rays,
    forgetting the distance attribute or moving the points would fail here
    rather than quietly returning a different quantity.
    """

    def __init__(self):
        self.set = {}
        # Flipped by a test to stand in for a build that does not write the
        # distance attribute: the handler has to say so, not read zeros.
        self.omit_distance = False

    def parms(self):
        # The defaults the parent read from this build, plus entity.
        return {
            "method": 1,
            "dotrans": True,
            "putdist": False,
            "useprimnumattrib": False,
            "primnumattrib": "hitprim",
            "useprimuvwattrib": False,
            "primuvwattrib": "hitprimuv",
            "gethitgroups": False,
            "hitgrp": "rayHitGroup",
            "maxraydistcheck": False,
            "maxraydist": 0.0,
            "scale": 1.0,
            "lift": 0.0,
            "entity": 0,
        }

    def setParms(self, parms):
        self.set = dict(parms)

    def execute(self, dest, inputs):
        assert self.set.get("method") == 0, "minimum distance, not project rays"
        assert self.set.get("entity") == 0, "the sample cloud has points, not primitives"
        assert self.set.get("putdist") == 1, "the distance attribute has to be asked for"
        assert self.set.get("useprimnumattrib") == 1, "so has the primitive number"
        assert self.set.get("dotrans") == 0, "the sample points are not meant to move"
        assert self.set.get("primnumattrib") == mh._HIT_PRIM_ATTRIB
        _refuse_stale(inputs)
        cloud, target = inputs[0], inputs[1]

        dest.points_list = [list(p) for p in cloud.points_list]
        distances = []
        hits = []
        for position in dest.points_list:
            if not target.faces:
                distances.append(1e18)
                hits.append(-1)
                continue
            distance, prim = target.surface_distance(position)
            distances.append(distance)
            hits.append(prim)
        if not self.omit_distance:
            dest.point_float_attribs[mh._DISTANCE_ATTRIB] = distances
        dest.point_int_attribs[mh._HIT_PRIM_ATTRIB] = hits


def _refuse_stale(inputs):
    """What a verb is entitled to do with geometry whose data ids are stale.

    Houdini would be free to reuse a cached cook; the stub is stricter and
    says so, which is how the handler's incrementAllDataIds gets tested
    instead of assumed.
    """
    for index, geometry in enumerate(inputs):
        if getattr(geometry, "dirty", False):
            raise OperationFailed(
                f"input {index} was modified without incrementAllDataIds; a verb may "
                "reuse a stale cook"
            )


class StubType:
    def name(self):
        return "grid"

    def category(self):
        return "Sop"


class StubObj:
    """An object node, for the world-space path."""

    def __init__(self, matrix):
        self.matrix = matrix

    def worldTransform(self):
        return self.matrix

    def parent(self):
        return None


class StubNode:
    def __init__(self, path, geometry, obj=None):
        self._path, self._geometry, self._obj = path, geometry, obj

    def path(self):
        return self._path

    def geometry(self):
        return self._geometry

    def type(self):
        return StubType()

    def parent(self):
        return self._obj


@pytest.fixture
def houdini(monkeypatch):
    """Point the handler module at the stub, and hand back a node registry."""
    registry: dict[str, StubNode] = {}

    stub = MagicMock()
    stub.OperationFailed = OperationFailed
    stub.attribData.Float = FLOAT
    stub.attribType.Prim = PRIM_ATTRIB
    stub.sopNodeTypeCategory.return_value = "Sop"
    stub.sopNodeTypeCategory.return_value = MagicMock()
    verbs = {"divide": StubVerb(), "ray": StubRayVerb()}
    stub.sopNodeTypeCategory.return_value.nodeVerb.side_effect = verbs.get
    stub.sopNodeTypeCategory.return_value.nodeTypes.return_value = {"ray": StubNodeType()}
    stub.node.side_effect = registry.get
    stub.Geometry = StubGeometry
    stub.ObjNode = StubObj

    # _get_sop_node compares the node's category against this one.
    category = stub.sopNodeTypeCategory.return_value
    monkeypatch.setattr(StubType, "category", lambda self: category)
    monkeypatch.setattr(mh, "hou", stub)
    if _numpy is not None:
        monkeypatch.setitem(sys.modules, "numpy", _numpy)

    def add(path, geometry, translate=None):
        obj = StubObj(StubMatrix(translate)) if translate is not None else None
        registry[path] = StubNode(path, geometry, obj)
        return path

    add.verbs = verbs
    return add


###### Fixtures in plain data
#
#   A unit quad and its neighbour, sharing edge 1-2:
#
#   3 - 2 - 5
#   |   |   |
#   0 - 1 - 4

TWO_QUADS_POINTS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [2.0, 0.0, 0.0],
    [2.0, 0.0, 1.0],
]
TWO_QUADS_FACES = [[0, 1, 2, 3], [1, 4, 5, 2]]

# Corner UVs in vertex order: face 0 is vertices 0-3, face 1 is vertices 4-7.
SIDE_BY_SIDE_UV = [
    (0.0, 0.0),
    (0.5, 0.0),
    (0.5, 1.0),
    (0.0, 1.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.5, 1.0),
]


def two_quads(**kwargs):
    return StubGeometry(TWO_QUADS_POINTS, TWO_QUADS_FACES, **kwargs)


###### get_uv_report


class TestUVReportGlue:
    def test_a_clean_two_quad_layout(self, houdini):
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=SIDE_BY_SIDE_UV))
        report = mh._get_uv_report(node_path=path)

        assert report["schema_version"] == mh.SCHEMA_VERSION
        assert report["uv"] == {"attribute": "uv", "owner": "vertex", "components": 2}
        assert report["counts"] == {"faces": 2, "uv_triangles": 4, "islands": 1}
        assert report["overlaps"]["status"] == "complete"
        assert report["overlaps"]["pairs"] == 0
        assert report["area"]["uv_area"] == approx(1.0)
        assert report["area"]["measure"] == "sum (not union)"
        assert report["area"]["tiles"] == [1001]
        assert report["winding"]["mirrored_triangles"] == 0
        # Each unit quad is mapped into a half-width UV rectangle, so the
        # mapping really is twice as dense across U as along V: anisotropy 2.
        assert report["anisotropy"]["max"] == approx(2.0, 1e-6)
        assert report["density"]["ratio_max"] == approx(1.0, 1e-6)
        assert "collapsed_uv" not in report

    def test_uvs_are_read_per_corner_so_a_seam_survives(self, houdini):
        """The two faces share points 1 and 2 but carry different UVs there.

        Read per point this would be one island with an impossible layout; read
        per corner -- which is what a vertex attribute means -- it is two.
        """
        seamed = list(SIDE_BY_SIDE_UV)
        seamed[4:8] = [(0.7, 0.0), (1.2, 0.0), (1.2, 1.0), (0.7, 1.0)]
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=seamed))
        report = mh._get_uv_report(node_path=path)
        assert report["counts"]["islands"] == 2
        assert report["overlaps"]["pairs"] == 0

    def test_a_point_attribute_is_used_and_named(self, houdini):
        point_uv = [(p[0] / 2.0, p[2]) for p in TWO_QUADS_POINTS]
        path = houdini("/obj/geo1/uv", two_quads(point_uv=point_uv))
        report = mh._get_uv_report(node_path=path)
        assert report["uv"]["owner"] == "point"
        assert report["counts"]["islands"] == 1

    def test_stacked_faces_are_overlaps_by_default_and_forgiven_on_request(self, houdini):
        """Both faces mapped onto the same square: a deliberate stack."""
        stacked = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] * 2
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=stacked))

        strict = mh._get_uv_report(node_path=path)
        # Two triangles per quad, each meeting its opposite number exactly.
        assert strict["overlaps"]["pairs"] == 2
        assert strict["overlaps"]["stacked_pairs"] == 2
        assert strict["overlaps"]["area"] == approx(1.0)

        relaxed = mh._get_uv_report(node_path=path, allow_stacking=True)
        assert relaxed["overlaps"]["pairs"] == 0
        assert relaxed["overlaps"]["stacked_pairs"] == 2
        assert relaxed["overlaps"]["allow_stacking"] is True

    def test_allow_stacking_still_reports_a_partial_overlap(self, houdini):
        """The requirement that matters: forgiving stacks must not forgive a fold."""
        folded = list(SIDE_BY_SIDE_UV)
        folded[4] = (0.2, 0.0)  # face 1 slid back over face 0
        folded[7] = (0.2, 1.0)
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=folded))
        report = mh._get_uv_report(node_path=path, allow_stacking=True)
        assert report["overlaps"]["pairs"] > 0
        assert report["overlaps"]["area"] > 0.0
        assert report["overlaps"]["worst"][0]["prims"] == [0, 1]

    def test_a_mirrored_face_is_reported_without_being_called_a_fault(self, houdini):
        mirrored = list(SIDE_BY_SIDE_UV)
        mirrored[4:8] = [(1.0, 0.0), (0.5, 0.0), (0.5, 1.0), (1.0, 1.0)]
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=mirrored))
        report = mh._get_uv_report(node_path=path)
        assert report["winding"]["mirrored_triangles"] == 2
        assert report["winding"]["mirrored_islands"] == 1

    def test_a_collapsed_uv_face_is_counted_not_crashed_on(self, houdini):
        """Zero UV area over real surface: no density, no anisotropy, a count."""
        collapsed = list(SIDE_BY_SIDE_UV)
        collapsed[4:8] = [(0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0)]
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=collapsed))
        report = mh._get_uv_report(node_path=path)
        assert report["collapsed_uv"] == {"triangles": 2, "prims": [1]}
        assert report["anisotropy"]["undefined"] == 2
        assert report["density"]["ratio_max"] == approx(1.0, 1e-6)

    def test_an_area_preserving_stretch_shows_up_as_anisotropy(self, houdini):
        """Density says this mapping is perfect; anisotropy says it is not."""
        points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        squashed = [(0.0, 0.0), (100.0, 0.0), (100.0, 0.01), (0.0, 0.01)]
        path = houdini("/obj/geo1/uv", StubGeometry(points, [[0, 1, 2, 3]], vertex_uv=squashed))
        report = mh._get_uv_report(node_path=path, stretch_threshold=2.0)
        assert report["density"]["ratio_max"] == approx(1.0, 1e-6)
        assert report["density"]["outside_threshold"] == 0
        assert report["anisotropy"]["max"] == approx(10000.0, 1.0)
        assert report["anisotropy"]["over_threshold"] == 2

    def test_a_nonfinite_uv_is_named_and_left_out(self, houdini):
        broken = list(SIDE_BY_SIDE_UV)
        broken[5] = (float("nan"), 0.0)
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=broken))
        report = mh._get_uv_report(node_path=path)
        assert report["nonfinite_uv"]["prims"] == [1]
        # Only the one triangle that uses the bad corner drops out; the other
        # half of that face is still measured, and its prim is still named.
        assert report["counts"]["uv_triangles"] == 3
        # And it drops out of the island count too: a triangle whose UVs are
        # not numbers matches no neighbour, so leaving it in would invent an
        # island out of a bad value.
        assert report["counts"]["islands"] == 1

    def test_a_face_the_triangulator_drops_is_reported(self, houdini):
        geometry = two_quads(vertex_uv=SIDE_BY_SIDE_UV)
        geometry.drop_prims = {1}
        path = houdini("/obj/geo1/uv", geometry)
        report = mh._get_uv_report(node_path=path)
        assert report["untriangulated"] == {"faces": 1, "prims": [1]}
        assert report["counts"]["uv_triangles"] == 2

    def test_excluded_faces_scope_the_overlap_verdict(self, houdini):
        """ "complete" must not read as "all of the UVs are clean"."""
        geometry = two_quads(vertex_uv=SIDE_BY_SIDE_UV)
        geometry.drop_prims = {1}
        path = houdini("/obj/geo1/uv", geometry)
        report = mh._get_uv_report(node_path=path)
        assert report["scope_complete"] is False
        assert report["excluded_faces"] == {"untriangulated_faces": 1}
        # The budget was fine, so the status is complete -- but over what.
        assert report["overlaps"]["status"] == "complete"
        assert report["overlaps"]["scope"] == "measured_faces_only"

    def test_a_whole_layout_says_so_in_the_overlap_block(self, houdini):
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=SIDE_BY_SIDE_UV))
        report = mh._get_uv_report(node_path=path)
        assert report["scope_complete"] is True
        assert report["overlaps"]["scope"] == "all_faces"
        assert "excluded_faces" not in report

    def test_a_nonfinite_uv_also_narrows_the_scope(self, houdini):
        broken = list(SIDE_BY_SIDE_UV)
        broken[5] = (float("nan"), 0.0)
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=broken))
        report = mh._get_uv_report(node_path=path)
        assert report["scope_complete"] is False
        assert report["excluded_faces"] == {"nonfinite_uv_faces": 1}
        assert report["overlaps"]["scope"] == "measured_faces_only"

    def test_a_missing_attribute_names_the_candidates(self, houdini):
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=SIDE_BY_SIDE_UV))
        with pytest.raises(OperationFailed, match="no vertex or point attribute named 'uv2'"):
            mh._get_uv_report(node_path=path, uv_attribute="uv2")

    def test_texture_resolution_turns_areas_into_pixels(self, houdini):
        stacked = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] * 2
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=stacked))
        report = mh._get_uv_report(node_path=path, texture_resolution=2048)
        assert report["area"]["texture_resolution"] == 2048
        # Six significant figures, which is what the receipt promises.
        assert report["area"]["uv_area_px2"] == approx(2.0 * 2048 * 2048, 10.0)
        assert report["overlaps"]["area_px2"] == approx(1.0 * 2048 * 2048, 10.0)

    def test_a_group_scopes_faces_and_their_vertices_together(self, houdini):
        """If the UVs were not filtered alongside the faces, this would read
        face 1's corners against face 0's point list."""
        geometry = two_quads(vertex_uv=SIDE_BY_SIDE_UV)
        geometry.groups["right"] = [1]
        path = houdini("/obj/geo1/uv", geometry)
        report = mh._get_uv_report(node_path=path, group="right")
        assert report["scope"] == "right"
        assert report["counts"]["faces"] == 1
        assert report["area"]["uv_area"] == approx(0.5)

    def test_a_small_group_of_an_input_over_the_limit_still_runs(self, houdini, monkeypatch):
        """The documented remedy for a large mesh has to actually work."""
        monkeypatch.setattr(mh, "MAX_PRIMS", 1)
        geometry = two_quads(vertex_uv=SIDE_BY_SIDE_UV)
        geometry.groups["right"] = [1]
        path = houdini("/obj/geo1/uv", geometry)

        report = mh._get_uv_report(node_path=path, group="right")
        assert report["counts"]["faces"] == 1
        with pytest.raises(OperationFailed, match="more than the 1"):
            mh._get_uv_report(node_path=path)

    def test_an_empty_or_missing_group_is_refused(self, houdini):
        geometry = two_quads(vertex_uv=SIDE_BY_SIDE_UV)
        geometry.groups["nothing"] = []
        path = houdini("/obj/geo1/uv", geometry)
        with pytest.raises(OperationFailed, match="is empty"):
            mh._get_uv_report(node_path=path, group="nothing")
        with pytest.raises(OperationFailed, match="No primitive group named 'nope'"):
            mh._get_uv_report(node_path=path, group="nope")

    def test_bad_arguments_are_refused_by_name(self, houdini):
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=SIDE_BY_SIDE_UV))
        with pytest.raises(ValueError, match="overlap_area_tolerance"):
            mh._get_uv_report(node_path=path, overlap_area_tolerance=-1)
        with pytest.raises(ValueError, match="texture_resolution"):
            mh._get_uv_report(node_path=path, texture_resolution=0)
        with pytest.raises(ValueError, match="stretch_threshold"):
            mh._get_uv_report(node_path=path, stretch_threshold=0.5)
        with pytest.raises(ValueError, match="schema_version"):
            mh._get_uv_report(node_path=path, schema_version=99)

    def test_the_dump_carries_what_the_receipt_caps(self, houdini, tmp_path):
        stacked = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] * 2
        path = houdini("/obj/geo1/uv", two_quads(vertex_uv=stacked))
        target = tmp_path / "uv.json"
        report = mh._get_uv_report(node_path=path, dump_path=str(target))
        assert report["dump"] == str(target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert len(payload["overlapping_pairs"]) == 2
        # One label per triangle, and two islands rather than one: the faces
        # share an edge in 3D but carry different UVs at both of its ends,
        # which is what a cut looks like.
        assert payload["island_per_triangle"] == [0, 0, 1, 1]


###### compare_surfaces


class TestCompareSurfacesGlue:
    def test_the_same_node_measures_zero_both_ways(self, houdini):
        path = houdini("/obj/geo1/a", two_quads())
        report = mh._compare_surfaces(a=path, b=path, samples=200)
        assert report["schema_version"] == mh.SCHEMA_VERSION
        assert report["space"] == "sop"
        assert "minimum-distance" in report["algorithm"]
        assert report["max_both"] == approx(0.0)
        assert report["a_to_b"]["samples"] == 200
        assert report["scope"]["a"] == {"prims": 2, "triangles": 4, "area": 2.0}
        assert report["scope_complete"] is True
        assert "excluded_faces" not in report

    def test_an_offset_surface_reports_the_offset(self, houdini):
        lifted = StubGeometry(
            [[p[0], p[1] + 0.25, p[2]] for p in TWO_QUADS_POINTS], TWO_QUADS_FACES
        )
        a = houdini("/obj/geo1/a", two_quads())
        b = houdini("/obj/geo1/b", lifted)
        report = mh._compare_surfaces(a=a, b=b, samples=200, tolerances=[0.1, 0.5])
        assert report["a_to_b"]["mean"] == approx(0.25)
        assert report["coverage"] == [
            {"tolerance": 0.1, "a_to_b": 0.0, "b_to_a": 0.0},
            {"tolerance": 0.5, "a_to_b": 1.0, "b_to_a": 1.0},
        ]
        assert report["tolerance_source"] == "given"

    def test_a_missing_region_shows_up_in_one_direction_only(self, houdini):
        half = StubGeometry(TWO_QUADS_POINTS, [TWO_QUADS_FACES[0]])
        a = houdini("/obj/geo1/a", two_quads())
        b = houdini("/obj/geo1/b", half)
        report = mh._compare_surfaces(a=a, b=b, samples=400)
        assert report["b_to_a"]["max"] == approx(0.0)
        assert report["a_to_b"]["max"] > 0.8
        assert report["a_to_b"]["worst"][0]["from_prim"] == 1

    def test_a_region_group_scopes_the_comparison_at_both_ends(self, houdini):
        geometry = two_quads()
        geometry.groups["left"] = [0]
        half = StubGeometry(TWO_QUADS_POINTS, [TWO_QUADS_FACES[0]])
        a = houdini("/obj/geo1/a", geometry)
        b = houdini("/obj/geo1/b", half)
        report = mh._compare_surfaces(a=a, b=b, samples=200, region_group_a="left")
        assert report["scope"]["a"]["group"] == "left"
        assert report["scope"]["a"]["prims"] == 1
        assert report["max_both"] == approx(0.0)

    def test_a_small_group_of_an_input_over_the_limit_still_runs(self, houdini, monkeypatch):
        monkeypatch.setattr(mh, "MAX_PRIMS", 1)
        geometry = two_quads()
        geometry.groups["left"] = [0]
        path = houdini("/obj/geo1/a", geometry)
        report = mh._compare_surfaces(
            a=path, b=path, samples=50, region_group_a="left", region_group_b="left"
        )
        assert report["scope"]["a"]["prims"] == 1
        with pytest.raises(OperationFailed, match="more than the 1"):
            mh._compare_surfaces(a=path, b=path, samples=50)

    def test_prim_ids_come_back_as_the_source_faces_not_triangle_numbers(self, houdini):
        """Divide makes four triangles here; the receipt must say 0 and 1."""
        path = houdini("/obj/geo1/a", two_quads())
        report = mh._compare_surfaces(a=path, b=path, samples=100, max_list=10)
        named = {entry["from_prim"] for entry in report["a_to_b"]["worst"]}
        assert named <= {0, 1}
        assert report["scope"]["a"]["triangles"] == 4

    def test_the_two_query_modes_give_the_same_numbers(self, houdini):
        """The batch pass is an optimisation, so it may not change an answer."""
        lifted = StubGeometry(
            [[p[0], p[1] + 0.25, p[2]] for p in TWO_QUADS_POINTS], TWO_QUADS_FACES
        )
        a = houdini("/obj/geo1/a", two_quads())
        b = houdini("/obj/geo1/b", lifted)
        batch = mh._compare_surfaces(a=a, b=b, samples=200, seed=2)
        per_sample = mh._compare_surfaces(a=a, b=b, samples=200, seed=2, query_mode="per_sample")
        for key in ("mean", "p95", "max"):
            assert batch["a_to_b"][key] == per_sample["a_to_b"][key]
            assert batch["b_to_a"][key] == per_sample["b_to_a"][key]
        assert batch["coverage"] == per_sample["coverage"]
        assert "Ray SOP verb in minimum-distance mode" in batch["algorithm"]
        assert "nearestPrim per sample (verification oracle)" in per_sample["algorithm"]

    def test_the_ray_verb_is_asked_for_minimum_distance_by_name(self, houdini):
        """Project Rays is the Ray SOP's default, and it answers a different
        question. The index is looked up in the menu rather than assumed, and
        the stub verb refuses to run in any other configuration -- so this
        passing is the whole parameter contract holding."""
        path = houdini("/obj/geo1/a", two_quads())
        report = mh._compare_surfaces(a=path, b=path, samples=20)
        assert report["max_both"] == approx(0.0)
        assert mh._ray_menu_index("method", "minimum") == 0
        assert mh._ray_menu_index("method", "nosuchmode", required=False) is None

    def test_a_menu_without_minimum_distance_is_refused(self, houdini, monkeypatch):
        monkeypatch.setitem(
            StubParmTemplateGroup.TEMPLATES,
            "method",
            StubMenuTemplate(("project",), ("Project Rays",)),
        )
        path = houdini("/obj/geo1/a", two_quads())
        with pytest.raises(OperationFailed, match="has no 'minimum' entry"):
            mh._compare_surfaces(a=path, b=path, samples=20)

    def test_a_missing_distance_attribute_is_refused_not_read_as_zero(self, houdini):
        houdini.verbs["ray"].omit_distance = True
        path = houdini("/obj/geo1/a", two_quads())
        with pytest.raises(OperationFailed, match="no 'dist' point attribute"):
            mh._compare_surfaces(a=path, b=path, samples=20)

    def test_per_sample_is_capped_because_it_loops_on_the_main_thread(self, houdini):
        path = houdini("/obj/geo1/a", two_quads())
        with pytest.raises(ValueError, match="capped at"):
            mh._compare_surfaces(
                a=path, b=path, samples=mh.MAX_PER_SAMPLE_QUERIES + 1, query_mode="per_sample"
            )
        with pytest.raises(ValueError, match="query_mode must be one of"):
            mh._compare_surfaces(a=path, b=path, query_mode="fast")

    def test_the_detached_copies_have_their_data_ids_bumped_before_a_verb(self, houdini):
        """A verb may reuse a cached cook otherwise; the stub verbs refuse."""
        path = houdini("/obj/geo1/a", two_quads())
        report = mh._compare_surfaces(a=path, b=path, samples=50)
        assert report["max_both"] == approx(0.0)

    def test_dropped_faces_make_the_measured_scope_incomplete(self, houdini):
        """Coverage over a surface missing faces may not read as whole-input."""
        geometry = two_quads()
        geometry.drop_prims = {1}
        path = houdini("/obj/geo1/a", geometry)
        report = mh._compare_surfaces(a=path, b=path, samples=100)
        assert report["scope_complete"] is False
        assert report["excluded_faces"] == {"a": 1, "b": 1}
        assert report["scope"]["a"]["untriangulated"] == {"count": 1, "prims": [1]}
        assert any("measured scope" in warning for warning in report["warnings"])

    def test_a_missing_or_empty_group_is_refused(self, houdini):
        geometry = two_quads()
        geometry.groups["left"] = [0]
        geometry.groups["none"] = []
        path = houdini("/obj/geo1/a", geometry)
        with pytest.raises(OperationFailed, match=r"Primitive groups: \['left', 'none'\]"):
            mh._compare_surfaces(a=path, b=path, region_group_a="lefft")
        with pytest.raises(OperationFailed, match="is empty"):
            mh._compare_surfaces(a=path, b=path, region_group_a="none")

    def test_a_primitive_the_triangulator_cannot_handle_is_refused(self, houdini):
        """An open polyline in the input is named, not quietly skipped."""
        geometry = StubGeometry(TWO_QUADS_POINTS, [TWO_QUADS_FACES[0]], curves=[[4, 5]])
        path = houdini("/obj/geo1/a", geometry)
        with pytest.raises(OperationFailed, match="could not turn into triangles"):
            mh._compare_surfaces(a=path, b=path, samples=50)

    def test_a_nonfinite_position_is_refused_rather_than_measured(self, houdini):
        broken = StubGeometry([[float("nan"), 0.0, 0.0], *TWO_QUADS_POINTS[1:]], TWO_QUADS_FACES)
        a = houdini("/obj/geo1/a", two_quads())
        b = houdini("/obj/geo1/b", broken)
        with pytest.raises(OperationFailed, match="not a finite number"):
            mh._compare_surfaces(a=a, b=b)

    def test_a_surface_with_no_area_is_refused(self, houdini):
        flat = StubGeometry([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0, 1, 2]])
        a = houdini("/obj/geo1/a", two_quads())
        b = houdini("/obj/geo1/b", flat)
        with pytest.raises(OperationFailed, match="encloses no area"):
            mh._compare_surfaces(a=a, b=b)

    def test_the_default_tolerance_is_derived_and_said_so(self, houdini):
        path = houdini("/obj/geo1/a", two_quads())
        report = mh._compare_surfaces(a=path, b=path, samples=50)
        assert report["tolerance_source"].startswith("auto")
        assert report["coverage"][0]["tolerance"] == approx(0.001 * (5.0**0.5))

    def test_sop_space_warns_when_the_two_objects_differ(self, houdini):
        a = houdini("/obj/geo1/a", two_quads(), translate=(0.0, 0.0, 0.0))
        b = houdini("/obj/geo2/b", two_quads(), translate=(0.0, 5.0, 0.0))
        report = mh._compare_surfaces(a=a, b=b, samples=50)
        assert report["max_both"] == approx(0.0)
        assert "different object transforms" in report["warnings"][0]

    def test_world_space_measures_the_gap_the_objects_put_there(self, houdini):
        a = houdini("/obj/geo1/a", two_quads(), translate=(0.0, 0.0, 0.0))
        b = houdini("/obj/geo2/b", two_quads(), translate=(0.0, 5.0, 0.0))
        report = mh._compare_surfaces(a=a, b=b, samples=50, space="world")
        assert report["space"] == "world"
        assert report["max_both"] == approx(5.0)
        assert "warnings" not in report

    def test_bad_arguments_are_refused_by_name(self, houdini):
        path = houdini("/obj/geo1/a", two_quads())
        with pytest.raises(ValueError, match="samples must be between"):
            mh._compare_surfaces(a=path, b=path, samples=0)
        with pytest.raises(ValueError, match="space must be one of"):
            mh._compare_surfaces(a=path, b=path, space="object")
        with pytest.raises(ValueError, match="tolerances"):
            mh._compare_surfaces(a=path, b=path, tolerances=[-1.0])
        with pytest.raises(ValueError, match="schema_version"):
            mh._compare_surfaces(a=path, b=path, schema_version=99)

    def test_the_same_seed_gives_the_same_numbers(self, houdini):
        lifted = StubGeometry([[p[0], p[1] + 0.3, p[2]] for p in TWO_QUADS_POINTS], TWO_QUADS_FACES)
        a = houdini("/obj/geo1/a", two_quads())
        b = houdini("/obj/geo1/b", lifted)
        first = mh._compare_surfaces(a=a, b=b, samples=100, seed=4)
        again = mh._compare_surfaces(a=a, b=b, samples=100, seed=4)
        assert first["a_to_b"]["worst"] == again["a_to_b"]["worst"]

    def test_the_dump_holds_every_sample(self, houdini, tmp_path):
        path = houdini("/obj/geo1/a", two_quads())
        target = tmp_path / "surface.json"
        report = mh._compare_surfaces(a=path, b=path, samples=64, dump_path=str(target))
        assert report["dump"] == str(target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert len(payload["samples"]["a_to_b"]) == 64
        assert len(payload["samples"]["b_to_a"]) == 64
        assert payload["receipt"]["a"] == path


###### get_mesh_report, with the new checks


class TestMeshReportQualityGlue:
    def test_the_default_receipt_gains_nothing_but_a_version(self, houdini):
        path = houdini("/obj/geo1/a", two_quads())
        report = mh._get_mesh_report(node_path=path)
        assert report["schema_version"] == mh.SCHEMA_VERSION
        assert "quality" not in report
        assert report["counts"]["quads"] == 2

    def test_a_nearly_flat_corner_is_found_where_the_degenerate_count_is_zero(self, houdini):
        """Point 1 pulled onto the line from point 0 to point 2."""
        points = list(TWO_QUADS_POINTS)
        points[1] = [0.5, 0.0, 0.499]
        path = houdini("/obj/geo1/a", StubGeometry(points, [TWO_QUADS_FACES[0]]))
        report = mh._get_mesh_report(node_path=path, quality_checks=["corner_angle"])
        assert report["degenerate"]["count"] == 0
        corner = report["quality"]["corner_angle"]
        assert corner["above_max"] == 1
        assert corner["flat"][0]["prim"] == 0
        assert corner["flat"][0]["point"] == 1

    def test_the_triangulation_check_reports_what_the_native_verb_did(self, houdini):
        path = houdini("/obj/geo1/a", two_quads())
        report = mh._get_mesh_report(node_path=path, quality_checks=["triangulation"])
        triangulation = report["quality"]["triangulation"]
        assert triangulation["triangles"] == 4  # two per quad
        assert triangulation["faces_checked"] == 2
        assert triangulation["untriangulated"] == 0
        assert triangulation["unexpected"] == 0
        assert triangulation["nonplanar"] == 0

    def test_a_face_the_native_verb_drops_is_reported_as_unexpected(self, houdini):
        geometry = two_quads()
        geometry.drop_prims = {1}
        path = houdini("/obj/geo1/a", geometry)
        triangulation = mh._get_mesh_report(node_path=path, quality_checks=["triangulation"])[
            "quality"
        ]["triangulation"]
        assert triangulation["untriangulated"] == 1
        assert triangulation["unexpected"] == 1
        assert triangulation["unexpected_faces"][0] == {"prim": 1, "sides": 4, "triangles": 0}

    def test_a_child_triangle_with_no_area_is_the_finding_not_the_count(self, houdini):
        """A quad with three collinear corners: two triangles, one of them nothing.

        The count check passes -- four sides gave two triangles -- so counting
        alone would call this clean. The area fraction is what catches it, and
        it names the source face.
        """
        points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
        path = houdini("/obj/geo1/a", StubGeometry(points, [[0, 1, 2, 3]]))
        triangulation = mh._get_mesh_report(node_path=path, quality_checks=["triangulation"])[
            "quality"
        ]["triangulation"]
        assert triangulation["unexpected"] == 0
        assert triangulation["untriangulated"] == 0
        assert triangulation["degenerate_triangles"] == 1
        assert triangulation["degenerate_faces"][0]["prim"] == 0
        assert triangulation["thresholds"]["min_area_fraction"] == 1e-6

    def test_a_sliver_child_triangle_is_named_with_its_threshold(self, houdini):
        """A very thin quad: both children are needles, neither is empty."""
        points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.002], [0.0, 0.0, 0.002]]
        path = houdini("/obj/geo1/a", StubGeometry(points, [[0, 1, 2, 3]]))
        triangulation = mh._get_mesh_report(
            node_path=path,
            quality_checks=["triangulation"],
            thresholds={"min_triangle_angle_deg": 5.0},
        )["quality"]["triangulation"]
        assert triangulation["degenerate_triangles"] == 0
        assert triangulation["sliver_triangles"] == 2
        assert triangulation["sliver_faces"][0]["prim"] == 0
        assert triangulation["min_angle_deg_seen"] < 5.0
        assert triangulation["thresholds"]["min_angle_deg"] == 5.0

    def test_a_clean_quad_mesh_reports_no_triangle_faults(self, houdini):
        path = houdini("/obj/geo1/a", two_quads())
        triangulation = mh._get_mesh_report(node_path=path, quality_checks=["triangulation"])[
            "quality"
        ]["triangulation"]
        assert triangulation["degenerate_triangles"] == 0
        assert triangulation["sliver_triangles"] == 0
        assert triangulation["min_area_fraction_seen"] == approx(0.5, 1e-6)
        assert triangulation["min_angle_deg_seen"] == approx(45.0, 1e-3)

    def test_a_bent_quad_is_reported_as_nonplanar(self, houdini):
        points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 1.0], [0.0, 0.0, 1.0]]
        path = houdini("/obj/geo1/a", StubGeometry(points, [[0, 1, 2, 3]]))
        triangulation = mh._get_mesh_report(node_path=path, quality_checks=["triangulation"])[
            "quality"
        ]["triangulation"]
        assert triangulation["nonplanar"] == 1
        assert triangulation["nonplanar_faces"][0]["prim"] == 0
        assert triangulation["untriangulated"] == 0

    def test_both_checks_can_run_together_with_custom_thresholds(self, houdini):
        path = houdini("/obj/geo1/a", two_quads())
        report = mh._get_mesh_report(
            node_path=path,
            quality_checks=["corner_angle", "triangulation"],
            thresholds={"max_corner_angle_deg": 80.0},
        )
        quality = report["quality"]
        # Every corner of a square is 90 degrees, so a band ending at 80 flags
        # all eight -- which is what proves the threshold is being read.
        assert quality["corner_angle"]["above_max"] == 8
        assert quality["triangulation"]["triangles"] == 4

    def test_the_receipt_list_is_capped_while_the_dump_is_not(self, houdini, tmp_path):
        path = houdini("/obj/geo1/a", two_quads())
        target = tmp_path / "mesh.json"
        report = mh._get_mesh_report(
            node_path=path,
            quality_checks=["corner_angle"],
            thresholds={"max_corner_angle_deg": 80.0},
            max_list=2,
            dump_path=str(target),
        )
        assert len(report["quality"]["corner_angle"]["flat"]) == 2
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert len(payload["quality"]["corner_angle"]["flat"]) == 8

    def test_a_small_prim_group_of_a_large_input_is_not_refused(self, houdini, monkeypatch):
        monkeypatch.setattr(mh, "MAX_PRIMS", 1)
        geometry = two_quads()
        geometry.groups["right"] = [1]
        path = houdini("/obj/geo1/a", geometry)
        report = mh._get_mesh_report(node_path=path, group="right")
        assert report["counts"]["polygons"] == 1
        with pytest.raises(OperationFailed, match="more than the 1"):
            mh._get_mesh_report(node_path=path)

    def test_an_unknown_check_or_threshold_is_refused(self, houdini):
        path = houdini("/obj/geo1/a", two_quads())
        with pytest.raises(ValueError, match="not a check"):
            mh._get_mesh_report(node_path=path, quality_checks=["corner_angles"])
        with pytest.raises(ValueError, match="no check reads"):
            mh._get_mesh_report(node_path=path, thresholds={"nope": 1})
        with pytest.raises(ValueError, match="schema_version"):
            mh._get_mesh_report(node_path=path, schema_version=99)
