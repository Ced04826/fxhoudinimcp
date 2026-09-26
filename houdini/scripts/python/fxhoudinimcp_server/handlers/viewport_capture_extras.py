"""Close-ups, overlays, contact sheets and shot lists for capture_viewport.

Split from viewport_capture_handlers so that file keeps the framing and
restore logic. The pure helpers at the top take no hou and are unit-tested;
the rest builds the proxy object's node chain, reads regions off geometry,
measures how many pixels an edge gets, and composes images with Qt (Houdini
ships PySide; no extra dependency).

Why each piece exists: an agent cannot drag the camera around a part the way
an artist does. It looks at an orbit sheet to find suspect places, then shoots
close-ups there -- a region (point + radius, group, face numbers), a view
straight at the surface, and a clip so the near side of the part does not hide
the far side. Overlays mark what a quick look misses (triangles, n-gons,
poles, stretched faces, zebra stripes). A source-vs-result pair with the same
camera shows a missing feature. The shot list lets a repair be checked from
exactly the same views.
"""

from __future__ import annotations

# Built-in
import contextlib
import json
import math
import os
from typing import Any

###### Constants

OVERLAYS = ("triangles", "ngons", "poles", "stretch", "zebra")

OVERLAY_LEGEND = {
    "triangles": "triangles blue",
    "ngons": "n-gons magenta",
    "poles": "poles: 3-edge cyan, 5+-edge orange",
    "stretch": "stretch: grey (aspect <= 1.5) to red (>= 4)",
    "zebra": "zebra: black/white bands of N . direction, per pixel",
}

# Headlight matcap: brightness from how squarely a surface faces the light,
# which sits just left of the camera. Straight on is `high`, edge-on `low`.
# The floor keeps every face well above the Wire Shaded line colour (0.2 in
# the stock schemes), so the wiring reads on the dark side too; the light is
# off-axis sideways so the two sides of a corner seen at 45 degrees differ.
HEADLIGHT = {"low": 0.42, "high": 0.88, "light": (-0.34, 0.0, 0.94)}
ZEBRA_BANDS = (0.06, 1.0)
MATCAP_SIZE = 256
ZEBRA_MATCAP_SIZE = 512

REGION_KEYS = {"bbox", "center", "radius", "group", "prims", "node"}

# A cell's label strip, in sheet pixels.
LABEL_H = 22
SHEET_GAP = 4

###### Pure helpers (no hou; unit-tested)


def parse_overlays(overlays: Any) -> list[str]:
    """A list of overlay names, in a fixed order; raises ValueError on unknown ones."""
    if overlays is None or overlays is False:
        return []
    if isinstance(overlays, str):
        overlays = [overlays]
    if not isinstance(overlays, (list, tuple)):
        raise ValueError(f"overlays must be a list of {list(OVERLAYS)}, not {overlays!r}")
    unknown = sorted({str(o) for o in overlays} - set(OVERLAYS))
    if unknown:
        raise ValueError(f"unknown overlays {unknown}; use any of {list(OVERLAYS)}")
    return [o for o in OVERLAYS if o in overlays]


def parse_region(spec: Any, where: str = "region") -> dict[str, Any] | None:
    """Validate a region spec. One of:

    {"bbox": [xmin, ymin, zmin, xmax, ymax, zmax]}
    {"center": [x, y, z], "radius": r}
    {"group": "name"}             primitive group, else point group
    {"prims": [3, 17, ...]}       primitive numbers
    group and prims read the first target unless "node" names another SOP.
    Coordinates are world space.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"{where} must be a dict, not {spec!r}")
    unknown = set(spec) - REGION_KEYS
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)}")
    kinds = [k for k in ("bbox", "center", "group", "prims") if k in spec]
    if len(kinds) != 1:
        raise ValueError(f"{where} needs exactly one of bbox, center (+radius), group, prims")
    kind = kinds[0]
    region: dict[str, Any] = {"kind": kind, "node": spec.get("node")}
    if kind == "bbox":
        box = spec["bbox"]
        if not isinstance(box, (list, tuple)) or len(box) != 6:
            raise ValueError(f"{where}.bbox must be [xmin, ymin, zmin, xmax, ymax, zmax]")
        lo, hi = [float(v) for v in box[:3]], [float(v) for v in box[3:]]
        if any(h < low for low, h in zip(lo, hi, strict=True)):
            raise ValueError(f"{where}.bbox max is below min")
        region["bbox"] = lo + hi
    elif kind == "center":
        center = spec["center"]
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            raise ValueError(f"{where}.center must be [x, y, z]")
        radius = spec.get("radius")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError(f"{where}.center needs a positive radius")
        region["center"] = [float(v) for v in center]
        region["radius"] = float(radius)
    elif kind == "group":
        if not isinstance(spec["group"], str) or not spec["group"]:
            raise ValueError(f"{where}.group must be a group name")
        region["group"] = spec["group"]
    else:
        prims = spec["prims"]
        if (
            not isinstance(prims, (list, tuple))
            or not prims
            or not all(isinstance(p, int) and not isinstance(p, bool) and p >= 0 for p in prims)
        ):
            raise ValueError(f"{where}.prims must be a non-empty list of primitive numbers")
        region["prims"] = sorted(set(prims))
    return region


def clip_shape(region: dict[str, Any], frame_box: list[float]) -> dict[str, Any]:
    """The clip volume of a resolved region: its sphere, else the framed box."""
    if region["kind"] == "center":
        return {"shape": "sphere", "center": region["center"], "radius": region["radius"]}
    return {"shape": "box", "bbox": list(frame_box)}


def orbit_views(orbit: Any) -> list[dict[str, Any]]:
    """View specs for an orbit: rings of elevations times azimuths.

    True gives two rings (30 and -20 degrees) of eight azimuths: 16 cells,
    every side seen from above and below. A dict may set "elevations" (list)
    and "azimuths" (a count, spread evenly from 0, or a list of degrees), and
    "projection".
    """
    if orbit is None or orbit is False:
        return []
    if orbit is True:
        orbit = {}
    if not isinstance(orbit, dict):
        raise ValueError(f"orbit must be true or a dict, not {orbit!r}")
    unknown = set(orbit) - {"elevations", "azimuths", "projection"}
    if unknown:
        raise ValueError(f"orbit has unknown keys {sorted(unknown)}")
    elevations = orbit.get("elevations", [30, -20])
    azimuths = orbit.get("azimuths", 8)
    if isinstance(azimuths, int) and not isinstance(azimuths, bool):
        if not 1 <= azimuths <= 36:
            raise ValueError("orbit.azimuths must be within 1..36")
        azimuths = [round(360.0 * i / azimuths, 3) for i in range(azimuths)]
    if not isinstance(elevations, (list, tuple)) or not elevations:
        raise ValueError("orbit.elevations must be a non-empty list of degrees")
    if not isinstance(azimuths, (list, tuple)) or not azimuths:
        raise ValueError("orbit.azimuths must be a count or a list of degrees")
    views = []
    for elevation in elevations:
        for azimuth in azimuths:
            views.append(
                {
                    "name": f"az{float(azimuth):g}_el{float(elevation):g}",
                    "azimuth": float(azimuth),
                    "elevation": float(elevation),
                    "projection": orbit.get("projection"),
                }
            )
    if len(views) > 36:
        raise ValueError(f"orbit makes {len(views)} views; keep it to 36")
    return views


def angles_of(axes) -> tuple[float, float]:
    """(azimuth, elevation) in degrees of a camera whose back axis is axes[2].

    The inverse of view_rotation: azimuth 0 looks from +Z, 90 from +X.
    """
    bx, by, bz = axes[2]
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, by))))
    azimuth = 0.0 if abs(bx) < 1e-9 and abs(bz) < 1e-9 else math.degrees(math.atan2(bx, bz))
    return round(azimuth, 2), round(elevation, 2)


def dominant_normal(areas, normals) -> list[float] | None:
    """The direction most of the region's surface faces.

    A plain area-weighted mean cancels on a thin part: a sphere around a hole
    in a plate holds the top and the bottom. So the direction is chosen among
    the face normals and the six axes as the one with the most area facing
    it (sum of area * max(0, n . d)), then refined as the area-weighted mean
    of the faces within 60 degrees of it.
    """
    import numpy as np

    a = np.asarray(areas, float)
    n = np.asarray(normals, float).reshape(-1, 3)
    if not len(a) or a.sum() <= 0:
        return None
    order = np.argsort(-a)[:64]
    axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    candidates = np.vstack([n[order], axes])
    scores = (np.clip(candidates @ n.T, 0, None) * a).sum(1)
    best = candidates[int(np.argmax(scores))]
    near = (n @ best) > 0.5
    mean = (n[near] * a[near, None]).sum(0)
    length = float(np.linalg.norm(mean))
    if length < 1e-12:
        return None
    return [round(float(v), 6) for v in mean / length]


def facing_angles(normal) -> tuple[float, float]:
    """(azimuth, elevation) of a camera looking straight at a surface with *normal*."""
    length = math.sqrt(sum(c * c for c in normal))
    if length < 1e-12:
        raise ValueError("the region's faces have no net normal to face (they cancel out)")
    return angles_of([None, None, [c / length for c in normal]])


def sheet_layout(
    sizes: list[tuple[int, int]], max_side: int, columns: int | None = None
) -> dict[str, Any]:
    """Grid for a contact sheet of images with *sizes*.

    Every cell has the same size, the largest image aspect fits in it, and
    the sheet's long side is at most *max_side*. Returns columns, rows,
    cell_w, cell_h (image area, the label strip is extra), sheet size and the
    scale each image gets.
    """
    n = len(sizes)
    if n == 0:
        raise ValueError("no images to lay out")
    if columns is None:
        columns = math.ceil(math.sqrt(n))
    columns = max(1, min(columns, n))
    rows = math.ceil(n / columns)
    # Cell aspect: the median image aspect, so most images fill their cell.
    aspects = sorted(w / h for w, h in sizes)
    aspect = aspects[len(aspects) // 2]
    # Largest cell_w with cols*cell_w and rows*(cell_h + label) inside max_side.
    usable_w = max_side - SHEET_GAP * (columns + 1)
    usable_h = max_side - SHEET_GAP * (rows + 1) - LABEL_H * rows
    cell_w = min(usable_w / columns, usable_h / rows * aspect)
    cell_w = max(16, int(cell_w))
    cell_h = max(16, int(cell_w / aspect))
    scales = [min(cell_w / w, cell_h / h) for w, h in sizes]
    width = columns * cell_w + SHEET_GAP * (columns + 1)
    height = rows * (cell_h + LABEL_H) + SHEET_GAP * (rows + 1)
    return {
        "columns": columns,
        "rows": rows,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "size": [width, height],
        "scales": scales,
    }


def crop_box(content, size, pad_fraction: float = 0.03) -> list[int] | None:
    """Pixel box [x0, y0, x1, y1] (x1/y1 exclusive) around *content*, padded, inside the image.

    *content* is the drawn target's pixel box; the crop always holds all of
    it. None when there is nothing to crop to.
    """
    if not content:
        return None
    w, h = int(size[0]), int(size[1])
    pad = max(4, round(max(w, h) * pad_fraction))
    x0 = max(0, int(math.floor(content[0])) - pad)
    y0 = max(0, int(math.floor(content[1])) - pad)
    x1 = min(w, int(math.ceil(content[2])) + pad)
    y1 = min(h, int(math.ceil(content[3])) + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [x0, y0, x1, y1]


def camera_direction(axes, direction) -> list[float]:
    """*direction* (world) in camera space: components along right, up, back."""
    d = [float(c) for c in direction]
    length = math.sqrt(sum(c * c for c in d)) or 1.0
    d = [c / length for c in d]
    return [sum(axes[i][j] * d[j] for j in range(3)) for i in range(3)]


def _matcap_normals(size: int):
    """Camera-space normal per matcap texel.

    Measured on 22.0.368: texel column follows the normal's x (left to
    right), texel row its y (top of the image is +y), values are used as
    stored. Texels outside the disc take the rim normal, so a silhouette
    never samples a seam.
    """
    import numpy as np

    centres = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    nx, ny = np.meshgrid(centres, -centres)
    r = np.hypot(nx, ny)
    over = r > 1.0
    nx = np.where(over, nx / np.maximum(r, 1e-9), nx)
    ny = np.where(over, ny / np.maximum(r, 1e-9), ny)
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
    return nx, ny, nz


def headlight_matcap(size: int = MATCAP_SIZE):
    """(size, size) grey levels 0..1 of the headlight matcap."""
    import numpy as np

    nx, ny, nz = _matcap_normals(size)
    lx, ly, lz = HEADLIGHT["light"]
    length = math.sqrt(lx * lx + ly * ly + lz * lz)
    facing = np.clip((nx * lx + ny * ly + nz * lz) / length, 0.0, 1.0)
    return HEADLIGHT["low"] + (HEADLIGHT["high"] - HEADLIGHT["low"]) * facing


def zebra_matcap(direction_camera, stripes: int, size: int = ZEBRA_MATCAP_SIZE):
    """(size, size) grey levels of bands of N . direction, direction in camera space.

    The viewport looks the matcap up per pixel with the mesh's interpolated
    normal, so the bands follow the surface being checked, not a denser copy.
    Bands are shifted a quarter period: a face square to the direction (N . d
    of -1, 0 or 1) sits in the middle of a band, not on a band edge where the
    smallest wobble of its normal would flip it.
    """
    import numpy as np

    nx, ny, nz = _matcap_normals(size)
    d = direction_camera
    value = nx * d[0] + ny * d[1] + nz * d[2]
    band = np.mod((value + 1.0) * 0.5 * int(stripes) + 0.25, 1.0) < 0.5
    return np.where(band, ZEBRA_BANDS[0], ZEBRA_BANDS[1])


def edge_pixels(segments, params: dict[str, Any], width, height, out_w) -> dict[str, Any] | None:
    """Projected length of edges, in output-image pixels.

    *segments* is an (m, 2, 3) numpy array of world-space edge end points.
    Edges with an end behind the camera are skipped. None when nothing is left.
    """
    import numpy as np

    pts = np.asarray(segments, float).reshape(-1, 3)
    rows = np.asarray(params["axes"], float)
    pivot = np.asarray(params["pivot"], float)
    t = np.asarray(params["t"], float)
    cam = (pts - pivot) @ rows.T - (t - pivot)
    if params["ortho"]:
        k = width / params["ortho_width"]
        sx, sy = cam[:, 0] * k, cam[:, 1] * k
        ok = np.ones(len(pts), bool)
    else:
        depth = -cam[:, 2]
        ok = depth > 1e-9
        k = params["focal"] / (params["aperture"] / 2.0) * (width / 2.0)
        safe = np.where(ok, depth, 1.0)
        sx, sy = cam[:, 0] / safe * k, cam[:, 1] / safe * k
    ok = ok.reshape(-1, 2).all(1)
    if not ok.any():
        return None
    scale = out_w / width
    d = np.hypot(sx[0::2] - sx[1::2], sy[0::2] - sy[1::2])[ok] * scale
    return {
        "mean": round(float(d.mean()), 2),
        "median": round(float(np.median(d)), 2),
        "p10": round(float(np.percentile(d, 10)), 2),
        "edges": int(ok.sum()),
    }


###### Overlay VEX


POINT_VEX = """
// Interior points whose edge count is not 4 are poles. A boundary point has an
// outgoing half-edge that no other face shares.
int boundary = 0;
int h = pointhedge(0, @ptnum);
while (h != -1) {
    if (hedge_equivcount(0, h) == 1) boundary = 1;
    h = pointhedgenext(0, h);
}
i@__valence = neighbourcount(0, @ptnum);
i@__pole = boundary ? 0 : (i@__valence != 4);
// Marker size: a quarter of the shortest edge here, so it stays inside the
// faces around the point. Colour by edge count.
float shortest = 1e30;
int nb[] = neighbours(0, @ptnum);
foreach (int q; nb) shortest = min(shortest, distance(@P, point(0, "P", q)));
f@pscale = len(nb) ? 0.25 * shortest : 0;
v@Cd = i@__valence < 4 ? {0.0, 0.95, 0.95} : {1.0, 0.55, 0.0};
"""

# Replaces the geometry with one small octahedron per pole, coloured by
# vertex colour (the mesh it is merged with may carry vertex Cd, which would
# override a point colour).
POLE_POINTS_VEX = """
addvertexattrib(0, "Cd", {0.72, 0.72, 0.72});
int n = npoints(0);
vector d[] = array({1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1});
int faces[] = array(0, 4, 2, 2, 4, 1, 1, 4, 3, 3, 4, 0, 2, 5, 0, 1, 5, 2, 3, 5, 1, 0, 5, 3);
for (int pr = 0; pr < nprimitives(0); pr++) removeprim(0, pr, 0);
for (int p = 0; p < n; p++) {
    if (point(0, "__pole", p)) {
        vector c = point(0, "P", p);
        float s = point(0, "pscale", p);
        vector col = point(0, "Cd", p);
        int ids[];
        foreach (vector v; d) append(ids, addpoint(0, c + v * s));
        for (int f = 0; f < 8; f++) {
            int prim = addprim(0, "poly", ids[faces[f * 3]], ids[faces[f * 3 + 1]], ids[faces[f * 3 + 2]]);
            for (int k = 0; k < 3; k++) setvertexattrib(0, "Cd", prim, k, col);
        }
    }
    removepoint(0, p);
}
"""


def vertex_vex(overlays: list[str]) -> str:
    """Vertex wrangle writing v@Cd for the face-class overlays (not zebra)."""
    lines = ["vector col = {0.72, 0.72, 0.72};", "int nv = primvertexcount(0, @primnum);"]
    if "stretch" in overlays:
        lines.append(
            """
int pts[] = primpoints(0, @primnum);
float el[];
for (int i = 0; i < nv; i++) {
    vector a = point(0, "P", pts[i]);
    vector b = point(0, "P", pts[(i + 1) % nv]);
    append(el, length(b - a));
}
float aspect = 1;
if (nv == 4) {
    float u = (el[0] + el[2]) * 0.5, v = (el[1] + el[3]) * 0.5;
    aspect = max(u, v) / max(min(u, v), 1e-12);
} else if (nv > 0) {
    aspect = max(el) / max(min(el), 1e-12);
}
col = lerp(col, {1.0, 0.1, 0.05}, clamp((aspect - 1.5) / 2.5, 0, 1));
f@__aspect = aspect;"""
        )
    if "triangles" in overlays:
        lines.append("if (nv == 3) col = {0.15, 0.4, 1.0};")
    if "ngons" in overlays:
        lines.append("if (nv > 4) col = {1.0, 0.15, 1.0};")
    lines.append("v@Cd = col;")
    return "\n".join(lines)


def clip_vex(clip: dict[str, Any]) -> str:
    """Primitive wrangle removing faces with no point and no centroid in the clip volume."""
    if clip["shape"] == "sphere":
        c, r = clip["center"], float(clip["radius"])
        test = f"distance(q, set({c[0]!r}, {c[1]!r}, {c[2]!r})) <= {r!r}"
    else:
        lo, hi = clip["bbox"][:3], clip["bbox"][3:]
        test = (
            f"q.x >= {lo[0]!r} && q.y >= {lo[1]!r} && q.z >= {lo[2]!r} && "
            f"q.x <= {hi[0]!r} && q.y <= {hi[1]!r} && q.z <= {hi[2]!r}"
        )
    return (
        "int keep = 0;\n"
        "vector q = v@P;\n"
        f"if ({test}) keep = 1;\n"
        "foreach (int pt; primpoints(0, @primnum)) {\n"
        '    q = point(0, "P", pt);\n'
        f"    if ({test}) keep = 1;\n"
        "}\n"
        "if (!keep) removeprim(0, @primnum, 1);\n"
    )


###### Shot list


def write_shot_list(path: str, record: dict[str, Any]) -> str:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1)
    return path


def read_shot_list(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, dict) or record.get("kind") != "fxhoudinimcp.capture_viewport.shots":
        raise ValueError(f"{path} is not a capture_viewport shot list")
    for i, shot in enumerate(record.get("shots") or []):
        camera = shot.get("camera") or {}
        missing = {"axes", "pivot", "t", "ortho", "ortho_width", "focal", "aperture"} - set(camera)
        if missing or not shot.get("pixels"):
            raise ValueError(f"{path}: shot {i} lacks camera fields {sorted(missing)} or pixels")
    if not record.get("shots"):
        raise ValueError(f"{path} has no shots")
    return record


###### Images (Qt, inside Houdini)


def _qt():
    try:
        from PySide6 import QtCore, QtGui
    except ImportError:  # pragma: no cover - older Houdini
        from PySide2 import QtCore, QtGui
    return QtCore, QtGui


def _font(QtGui, size: int):
    font = QtGui.QFont("Segoe UI")
    font.setPixelSize(size)
    return font


def write_grey(values, path: str) -> str:
    """Save a (h, w) array of grey levels 0..1 as an opaque PNG."""
    import numpy as np

    _, QtGui = _qt()
    grey = np.clip(np.round(np.asarray(values, float) * 255.0), 0, 255).astype(np.uint8)
    h, w = grey.shape
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = grey
    rgba[..., 3] = 255
    image = QtGui.QImage(rgba.data, w, h, w * 4, QtGui.QImage.Format_ARGB32).copy()
    if not image.save(path):
        raise RuntimeError(f"could not write {path}")
    return path


def flatten(path: str, top, bottom) -> bool:
    """Put a flipbook image over the viewport's background, top to bottom gradient.

    The flipbook writes the background as transparent: an image viewer shows
    it white or checkered while a sheet cell shows the sheet's own fill. Over
    the scheme's own colours every image looks like the viewport. Returns
    whether anything was transparent.
    """
    QtCore, QtGui = _qt()
    image = QtGui.QImage(path)
    if image.isNull() or not image.hasAlphaChannel():
        return False
    flat = QtGui.QImage(image.width(), image.height(), QtGui.QImage.Format_RGB32)
    gradient = QtGui.QLinearGradient(0, 0, 0, image.height())
    gradient.setColorAt(0.0, QtGui.QColor.fromRgbF(*top))
    gradient.setColorAt(1.0, QtGui.QColor.fromRgbF(*bottom))
    painter = QtGui.QPainter(flat)
    try:
        painter.fillRect(flat.rect(), QtGui.QBrush(gradient))
        painter.drawImage(0, 0, image)
    finally:
        painter.end()
    if not flat.save(path):
        raise RuntimeError(f"could not write {path}")
    return True


def compose_pair(left: str, right: str, labels: tuple[str, str], path: str) -> dict[str, Any]:
    """Two same-camera images side by side, each with a label strip above."""
    QtCore, QtGui = _qt()
    a, b = QtGui.QImage(left), QtGui.QImage(right)
    if a.isNull() or b.isNull():
        raise RuntimeError(f"cannot read {left if a.isNull() else right}")
    w, h = max(a.width(), b.width()), max(a.height(), b.height())
    strip = 26
    sheet = QtGui.QImage(
        2 * w + 3 * SHEET_GAP, h + strip + 2 * SHEET_GAP, QtGui.QImage.Format_RGB32
    )
    sheet.fill(QtGui.QColor(40, 40, 40))
    painter = QtGui.QPainter(sheet)
    try:
        painter.setFont(_font(QtGui, 16))
        for i, (image, label) in enumerate(((a, labels[0]), (b, labels[1]))):
            x = SHEET_GAP + i * (w + SHEET_GAP)
            painter.fillRect(x, SHEET_GAP, w, strip, QtGui.QColor(20, 20, 20))
            painter.setPen(QtGui.QColor(255, 210, 90) if i == 0 else QtGui.QColor(120, 220, 255))
            painter.drawText(
                QtCore.QRect(x + 6, SHEET_GAP, w - 12, strip),
                int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft),
                label,
            )
            painter.drawImage(x, SHEET_GAP + strip, image)
    finally:
        painter.end()
    if not sheet.save(path):
        raise RuntimeError(f"could not write {path}")
    return {"path": path, "pixels": [sheet.width(), sheet.height()]}


def compose_sheet(
    cells: list[dict[str, Any]], path: str, max_side: int, columns: int | None, header: str | None
) -> dict[str, Any]:
    """Contact sheet: each cell {path, label, crop?}; returns layout and per-cell scale.

    crop ([x0, y0, x1, y1], see crop_box) trims the empty border around the
    target so it fills more of its cell.
    """
    QtCore, QtGui = _qt()
    images = []
    for cell in cells:
        image = QtGui.QImage(cell["path"])
        if image.isNull():
            raise RuntimeError(f"cannot read {cell['path']}")
        crop = cell.get("crop")
        if crop:
            image = image.copy(QtCore.QRect(crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1]))
        images.append(image)
    head = LABEL_H if header else 0
    layout = sheet_layout([(im.width(), im.height()) for im in images], max_side - head, columns)
    width, height = layout["size"]
    sheet = QtGui.QImage(width, height + head, QtGui.QImage.Format_RGB32)
    sheet.fill(QtGui.QColor(40, 40, 40))
    painter = QtGui.QPainter(sheet)
    try:
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        painter.setFont(_font(QtGui, 14))
        if header:
            painter.setPen(QtGui.QColor(230, 230, 230))
            painter.drawText(
                QtCore.QRect(SHEET_GAP, 0, width - 2 * SHEET_GAP, head),
                int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft),
                header,
            )
        cw, ch = layout["cell_w"], layout["cell_h"]
        for i, (cell, image) in enumerate(zip(cells, images, strict=True)):
            col, row = i % layout["columns"], i // layout["columns"]
            x = SHEET_GAP + col * (cw + SHEET_GAP)
            y = head + SHEET_GAP + row * (ch + LABEL_H + SHEET_GAP)
            painter.fillRect(x, y, cw, LABEL_H, QtGui.QColor(20, 20, 20))
            painter.setPen(
                QtGui.QColor(255, 120, 90) if cell.get("warn") else QtGui.QColor(235, 235, 235)
            )
            painter.drawText(
                QtCore.QRect(x + 4, y, cw - 8, LABEL_H),
                int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft),
                cell["label"],
            )
            scale = layout["scales"][i]
            tw, th = max(1, round(image.width() * scale)), max(1, round(image.height() * scale))
            scaled = image.scaled(
                tw, th, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation
            )
            painter.drawImage(x + (cw - tw) // 2, y + LABEL_H + (ch - th) // 2, scaled)
    finally:
        painter.end()
    if not sheet.save(path):
        raise RuntimeError(f"could not write {path}")
    return {
        "path": path,
        "pixels": [sheet.width(), sheet.height()],
        "columns": layout["columns"],
        "rows": layout["rows"],
        "scales": [round(s, 4) for s in layout["scales"]],
    }


###### Geometry (hou)


def region_facts(region: dict[str, Any], default_node) -> dict[str, Any]:
    """Resolve a region on geometry: framing corners, prims inside, mean normal.

    Returns {"corners", "frame_box", "normal" (area-weighted, world, may be
    None), "prims" (count used for the normal), "node"}. bbox and sphere
    regions frame their own volume and take the normal from the faces whose
    centroid lies inside; group and prims regions frame those faces.
    """
    import hou

    node = hou.node(region["node"]) if region.get("node") else default_node
    if region.get("node") and node is None:
        raise ValueError(f"region node {region['node']!r} does not exist")
    geometry = transform = None
    if node is not None:
        category = node.type().category().name()
        if category == "Sop":
            geometry = node.geometry()
            creator = node.creator()
            with contextlib.suppress(Exception):
                if creator is not None and creator.type().category().name() == "Object":
                    transform = creator.worldTransform()
        elif category == "Object" and node.displayNode() is not None:
            geometry, transform = node.displayNode().geometry(), node.worldTransform()
    kind = region["kind"]
    if kind in ("group", "prims") and geometry is None:
        raise ValueError(f"a {kind} region needs a SOP target or region.node")

    def world(v):
        p = hou.Vector3(v)
        return p * transform if transform is not None else p

    normal_xform = transform.inverted().transposed() if transform is not None else None

    def world_normal(n):
        if normal_xform is None:
            return hou.Vector3(n)
        v = hou.Vector4(n[0], n[1], n[2], 0.0) * normal_xform
        return hou.Vector3(v[0], v[1], v[2])

    prims = []
    if kind == "group":
        group = geometry.findPrimGroup(region["group"])
        if group is not None:
            prims = list(group.prims())
        else:
            pgroup = geometry.findPointGroup(region["group"])
            if pgroup is None:
                raise ValueError(
                    f"{node.path()} has no primitive or point group {region['group']!r}"
                )
            wanted = {p.number() for p in pgroup.points()}
            prims = [
                pr for pr in geometry.prims() if any(p.number() in wanted for p in pr.points())
            ]
    elif kind == "prims":
        count = geometry.intrinsicValue("primitivecount")
        bad = [i for i in region["prims"] if i >= count]
        if bad:
            raise ValueError(f"{node.path()} has {count} primitives; no {bad[:5]}")
        prims = [geometry.prim(i) for i in region["prims"]]

    if kind in ("group", "prims"):
        if not prims:
            raise ValueError(f"the {kind} region on {node.path()} holds no primitives")
        pts = [world(p.position()) for pr in prims for p in pr.points()]
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
    elif kind == "bbox":
        lo, hi = region["bbox"][:3], region["bbox"][3:]
    else:
        c, r = region["center"], region["radius"]
        lo, hi = [c[i] - r for i in range(3)], [c[i] + r for i in range(3)]

    if kind in ("bbox", "center") and geometry is not None:
        # Faces whose centroid lies in the region give the normal.
        for pr in geometry.prims():
            with contextlib.suppress(Exception):
                c = world(pr.boundingBox().center())
                if kind == "bbox":
                    inside = all(lo[i] <= c[i] <= hi[i] for i in range(3))
                else:
                    inside = (c - hou.Vector3(region["center"])).length() <= region["radius"]
                if inside:
                    prims.append(pr)
    normal = None
    if prims:
        areas, normals = [], []
        for pr in prims:
            with contextlib.suppress(Exception):
                areas.append(pr.intrinsicValue("measuredarea"))
                normals.append(list(world_normal(pr.normal()).normalized()))
        normal = dominant_normal(areas, normals)
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return {
        "corners": corners,
        "frame_box": [round(float(v), 6) for v in list(lo) + list(hi)],
        "normal": normal,
        "prims": len(prims),
        "node": node.path() if node is not None else None,
    }


def build_chain(proxy, merge, clip: dict[str, Any] | None, overlays: list[str]):
    """Clip and overlay nodes after *merge* inside *proxy*; returns (measure, out).

    measure is the clipped geometry (what the edge metric reads), out the
    node that gets the display flag. Zebra is not in the chain: it is a
    matcap the viewport applies per pixel (see zebra_matcap), and the
    geometry is left as it is.
    """
    node = merge
    if clip is not None:
        # A face stays when one of its points or its centroid is inside: the
        # wiring around the region stays whole, while a large face that only
        # passes through the region (a cover plate far away) goes.
        cut = proxy.createNode("attribwrangle", "clip_region")
        cut.setInput(0, node)
        cut.parm("class").set("primitive")
        cut.parm("snippet").set(clip_vex(clip))
        node = cut
    measure = node
    faces = [o for o in overlays if o not in ("zebra", "poles")]
    if faces:
        wv = proxy.createNode("attribwrangle", "face_colours")
        wv.setInput(0, node)
        wv.parm("class").set("vertex")
        wv.parm("snippet").set(vertex_vex(faces))
        node = wv
    if "poles" in overlays:
        # Poles as small spheres on the points: a vertex colour would bleed
        # across every face around the point and hide the face colours.
        wp = proxy.createNode("attribwrangle", "pole_points")
        wp.setInput(0, measure)
        wp.parm("class").set("point")
        wp.parm("snippet").set(POINT_VEX)
        only = proxy.createNode("attribwrangle", "pole_only")
        only.setInput(0, wp)
        only.parm("class").set("detail")
        only.parm("snippet").set(POLE_POINTS_VEX)
        both = proxy.createNode("merge", "with_pole_markers")
        both.setInput(0, node)
        both.setInput(1, only)
        node = both
    return measure, node


def segments_of(geometry, transform=None, limit: int = 20000):
    """(m, 2, 3) world-space edge end points of up to *limit* faces, evenly sampled."""
    import numpy as np

    count = geometry.intrinsicValue("primitivecount")
    if count == 0:
        return np.zeros((0, 2, 3))
    positions = np.asarray(geometry.pointFloatAttribValues("P"), float).reshape(-1, 3)
    if transform is not None:
        m = np.asarray(transform.asTuple(), float).reshape(4, 4)
        positions = positions @ m[:3, :3] + m[3, :3]
    stride = max(1, count // limit)
    pairs = []
    prims = geometry.prims()
    for index in range(0, count, stride):
        pts = [p.number() for p in prims[index].points()]
        if len(pts) < 2:
            continue
        closed = True
        with contextlib.suppress(Exception):
            closed = prims[index].isClosed()
        ends = pts[1:] + (pts[:1] if closed else [])
        pairs.extend(zip(pts, ends, strict=False))
    if not pairs:
        return np.zeros((0, 2, 3))
    index = np.asarray(pairs, int)
    return positions[index]


def file_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0
