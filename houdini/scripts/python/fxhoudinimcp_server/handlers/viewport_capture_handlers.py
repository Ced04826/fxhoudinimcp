"""Framed captures of what the Scene Viewer shows.

A capture is a flipbook of the viewport itself: the same shading, wireframe,
display flags and guides the artist is looking at. It is not a render (an
OpenGL ROP shows no wiring and crashed a GUI session twice, see
shelved/README.md), and it never looks through a camera node, whose resolution
and aspect would decide what gets cut off.

Framing is computed, then checked. The viewport camera is placed from its own
parameters -- pivot on the target's centre, distance or ortho width solved so
every corner of the target's bounding box lands inside the frame with the
requested margin -- and the output image takes the target's aspect, so a tall
part gets a tall image instead of a sliver in a wide one. mapToScreen is not
used: on 22.0.368 it does not follow camera edits made in the same call, even
after draw(). The receipt reports where the target landed and how much of that
region is drawn on, read from the written pixels.

The viewer is put back as it was: view type, the camera of every view type
visited, the camera node it was looking through, and the shading this call
changed. GeometryViewport.setDefaultCamera(stash) alone does not restore the
ortho width, so every camera field is written back explicitly.
"""

from __future__ import annotations

# Built-in
import contextlib
import hashlib
import math
import os
import tempfile
import time
from typing import Any

# Third-party
import hou

# Internal
from fxhoudinimcp_server.dispatcher import register_handler
from fxhoudinimcp_server.handlers import viewport_capture_extras as extras
from fxhoudinimcp_server.handlers.rendering_handlers import _find_flipbook_output
from fxhoudinimcp_server.handlers.viewport_handlers import _find_scene_viewer, _no_mplay

###### Constants

_AXIS_VIEWS = {
    "front": "Front",
    "back": "Back",
    "left": "Left",
    "right": "Right",
    "top": "Top",
    "bottom": "Bottom",
}

_SHADING_NAMES = {
    "wireframe": "Wire",
    "wire": "Wire",
    "smooth": "Smooth",
    "shaded": "Smooth",
    "smooth_wire": "SmoothWire",
    "flat": "Flat",
    "flat_wire": "FlatWire",
    "hidden_line": "HiddenLineGhost",
    "matcap": "MatCap",
    "matcap_wire": "MatCapWire",
}

# The display sets a requested shading is applied to. DisplayModel is the
# geometry of the object being edited -- what a SOP-level artist looks at --
# and setting only SceneObject, as set_viewport_display does, left it wired
# while the tool reported smooth.
_SHADED_SETS = ("DisplayModel", "SceneObject", "SelectedObject")

# An image wider than 8:1 is a strip nobody can read; past that the target
# gets extra margin on the short side instead.
_MAX_ASPECT = 8.0
_MIN_SIDE = 16
_FIT_ITERATIONS = 12
_FIT_TOLERANCE = 0.002

# A pixel differing from the background by more than this (sum over RGB) is
# counted as drawn.
_DRAWN_THRESHOLD = 30
_BLANK_FRACTION = 0.0005

_LIGHTING = ("headlight", "viewport")

_VIEW_KEYS = {
    "name",
    "direction",
    "azimuth",
    "elevation",
    "projection",
    "targets",
    "bbox",
    "region",
    "clip",
}


###### Pure helpers (no hou; unit-tested)


def output_size(half_x: float, half_y: float, max_size: int) -> tuple[int, int]:
    """Image size whose aspect is the target's projected aspect, long side max_size.

    Clamped to 8:1 either way; a thinner target keeps its full length and gets
    extra room across.
    """
    aspect = half_x / half_y if half_y > 0 else _MAX_ASPECT
    aspect = min(max(aspect, 1.0 / _MAX_ASPECT), _MAX_ASPECT)
    if aspect >= 1.0:
        return int(max_size), max(_MIN_SIDE, round(max_size / aspect))
    return max(_MIN_SIDE, round(max_size * aspect)), int(max_size)


def zoom_factor(
    half_x: float,
    half_y: float,
    viewport_w: float,
    out_w: int,
    out_h: int,
    margin: float,
) -> float:
    """How much to widen the view so the target plus margin fits the output.

    A flipbook at a resolution other than the viewport's keeps the viewport's
    horizontal extent and derives the vertical one from the output aspect
    (measured on 22.0.368: 1600x400 from a 1219x1915 viewport showed the same
    width and cut the top and bottom). So the frame, in viewport pixels, is
    viewport_w wide and viewport_w * out_h / out_w tall, centred.

    half_x/half_y are the target's largest distances from the viewport centre,
    in viewport pixels. Above 1 the view must widen by that factor; below 1
    it can close in.
    """
    usable = max(1e-6, 1.0 - 2.0 * margin)
    frame_half_x = viewport_w / 2.0
    frame_half_y = frame_half_x * out_h / out_w
    return max(half_x / (frame_half_x * usable), half_y / (frame_half_y * usable))


def to_image(
    px: float,
    py: float,
    viewport_w: float,
    viewport_h: float,
    out_w: int,
    out_h: int,
) -> tuple[float, float]:
    """Viewport pixel (origin bottom-left) to output pixel (origin top-left).

    Same centre, uniform scale out_w / viewport_w, for the reason zoom_factor
    gives.
    """
    k = out_w / viewport_w
    return (px - viewport_w / 2.0) * k + out_w / 2.0, out_h / 2.0 - (py - viewport_h / 2.0) * k


def view_rotation(azimuth: float, elevation: float) -> list[list[float]]:
    """Camera axes (right, up, back) for a view from azimuth/elevation.

    Degrees. Azimuth 0 looks from +Z (the Front view), 90 from +X (Right);
    elevation 90 looks straight down (Top). The identity is the Front view.
    """
    a = math.radians(azimuth)
    e = math.radians(elevation)
    back = (math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e))
    if abs(elevation) > 89.0:
        # Looking along Y: "up" is -Z from above and +Z from below, the way
        # Houdini's Top and Bottom views are oriented.
        world_up = (0.0, 0.0, -1.0) if elevation > 0 else (0.0, 0.0, 1.0)
    else:
        world_up = (0.0, 1.0, 0.0)
    right = _normalize(_cross(world_up, back))
    up = _cross(back, right)
    return [list(right), list(up), list(back)]


def to_camera(point, params: dict[str, Any]) -> tuple[float, float, float]:
    """Camera-space coordinates of a world point; the camera looks down -z.

    Houdini's viewport camera maps p to R(p - pivot) - (t - pivot), where the
    rows of R are the camera axes (right, up, back) in world space. Worked out
    from read-backs on 22.0.368: after framing a box, pivot is the box centre
    and t is pivot + (0, 0, distance), and a panned view keeps t - pivot as
    the pan in camera space.
    """
    rows, pivot, t = params["axes"], params["pivot"], params["t"]
    d = (point[0] - pivot[0], point[1] - pivot[1], point[2] - pivot[2])
    return tuple(
        rows[i][0] * d[0] + rows[i][1] * d[1] + rows[i][2] * d[2] - (t[i] - pivot[i])
        for i in range(3)
    )


def project(point, params: dict[str, Any], width: float, height: float):
    """Viewport pixel (origin bottom-left) of a world point, or None behind the camera.

    Ortho: ortho_width is the horizontal extent. Perspective: aperture is the
    horizontal film width in the focal length's units.
    """
    x, y, z = to_camera(point, params)
    if params["ortho"]:
        k = width / params["ortho_width"]
        return width / 2.0 + x * k, height / 2.0 + y * k
    depth = -z
    if depth <= 1e-9:
        return None
    k = params["focal"] / (params["aperture"] / 2.0) * (width / 2.0)
    return width / 2.0 + x / depth * k, height / 2.0 + y / depth * k


def half_extents(points, params, width, height) -> tuple[float, float] | None:
    """Largest distances of the projected points from the viewport centre."""
    half_x = half_y = 0.0
    for point in points:
        screen = project(point, params, width, height)
        if screen is None:
            return None
        half_x = max(half_x, abs(screen[0] - width / 2.0))
        half_y = max(half_y, abs(screen[1] - height / 2.0))
    return half_x, half_y


def centred_on(params: dict[str, Any], center, radius: float) -> dict[str, Any]:
    """*params* with the pivot on *center*, no pan, and a start distance or width."""
    placed = dict(params)
    placed["pivot"] = list(center)
    if placed["ortho"]:
        placed["t"] = [center[0], center[1], center[2] + max(radius * 4.0, 1e-3)]
        placed["ortho_width"] = max(radius * 2.0, 1e-6)
    else:
        half_fov = math.atan(placed["aperture"] / 2.0 / placed["focal"])
        distance = max(radius / math.sin(half_fov) * 1.1, 1e-3)
        placed["t"] = [center[0], center[1], center[2] + distance]
    return placed


def zoomed(params: dict[str, Any], factor: float, min_distance: float = 0.0) -> dict[str, Any]:
    """*params* widened by *factor*: ortho width, or distance to the pivot.

    A perspective camera never comes closer than *min_distance*.
    """
    result = dict(params)
    if result["ortho"]:
        result["ortho_width"] = result["ortho_width"] * factor
    else:
        pivot, t = result["pivot"], result["t"]
        distance = max((t[2] - pivot[2]) * factor, min_distance)
        result["t"] = [t[0], t[1], pivot[2] + distance]
    return result


def fit(points, params, width, height, max_size, margin, min_distance=0.0):
    """Solve the framing. Returns (params, out_w, out_h, half_extents).

    A perspective camera stays at least *min_distance* from the pivot (the
    target's bounding radius): inside it the near corners explode and no
    distance satisfies the margin.
    """
    extents = half_extents(points, params, width, height)
    while extents is None:
        # A corner behind the camera: back off until everything is in front.
        params = zoomed(params, 2.0)
        extents = half_extents(points, params, width, height)
    out_w, out_h = output_size(extents[0], extents[1], max_size)
    if not params["ortho"] and out_h > out_w:
        # The flipbook keeps the horizontal field of view, so a tall frame
        # with the viewport's lens sees 100 degrees and more vertically, and
        # filling it needs the camera inside the target. Narrow the lens so
        # the long side keeps the field of view the short side had. Changing
        # the aperture scales the projection about the centre, so the aspect
        # just measured still holds.
        params = dict(params)
        params["aperture"] = params["aperture"] * out_w / out_h
        extents = half_extents(points, params, width, height)
    for _ in range(_FIT_ITERATIONS):
        factor = zoom_factor(extents[0], extents[1], width, out_w, out_h, margin)
        if abs(factor - 1.0) <= _FIT_TOLERANCE:
            break
        if not params["ortho"]:
            # Distance is not linear in the projected size once the near
            # corners dominate; small steps converge where one big jump can
            # put the camera inside the box.
            factor = min(max(factor, 0.5), 2.0)
        candidate = zoomed(params, factor, min_distance)
        candidate_extents = half_extents(points, candidate, width, height)
        if candidate_extents is None:
            candidate = zoomed(params, 1.0 + (factor - 1.0) / 2.0, min_distance)
            candidate_extents = half_extents(points, candidate, width, height)
            if candidate_extents is None:
                break
        params, extents = candidate, candidate_extents
    return params, out_w, out_h, extents


def _cross(a, b) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalize(v) -> tuple[float, float, float]:
    length = math.sqrt(sum(c * c for c in v)) or 1.0
    return (v[0] / length, v[1] / length, v[2] / length)


def _axes_of(rotation) -> list[list[float]]:
    """Camera axes (right, up, back) of a viewport camera rotation.

    They are the matrix's columns: a user's orbit view, which has no roll,
    has a horizontal first column and a tilted first row.
    """
    rows = rotation.asTupleOfTuples()
    return [[rows[j][i] for j in range(3)] for i in range(3)]


def _rotation_of(axes):
    """The viewport rotation whose columns are *axes*."""
    return hou.Matrix3(tuple(tuple(axes[i][j] for i in range(3)) for j in range(3)))


def parse_view(spec: Any, index: int) -> dict[str, Any]:
    """One entry of `views` as {name, direction, azimuth, elevation, projection, ...}.

    A string is a direction. Raises ValueError naming what is wrong.
    """
    if isinstance(spec, str):
        spec = {"direction": spec}
    if not isinstance(spec, dict):
        raise ValueError(f"views[{index}] must be a direction string or a dict, not {spec!r}")
    unknown = set(spec) - _VIEW_KEYS
    if unknown:
        raise ValueError(f"views[{index}] has unknown keys {sorted(unknown)}")
    direction = spec.get("direction")
    has_angles = spec.get("azimuth") is not None or spec.get("elevation") is not None
    projection = spec.get("projection")
    if projection is not None and projection not in ("ortho", "persp"):
        raise ValueError(
            f"views[{index}]: projection must be 'ortho' or 'persp', not {projection!r}"
        )
    common = {
        "targets": spec.get("targets"),
        "bbox": spec.get("bbox"),
        "region": extras.parse_region(spec.get("region"), f"views[{index}].region"),
        "clip": spec.get("clip"),
    }
    if has_angles:
        if direction is not None:
            raise ValueError(f"views[{index}]: give a direction or azimuth/elevation, not both")
        azimuth = float(spec.get("azimuth") or 0.0)
        elevation = float(spec.get("elevation") or 0.0)
        if not -90.0 <= elevation <= 90.0:
            raise ValueError(f"views[{index}]: elevation must be within -90..90, not {elevation}")
        return {
            "name": spec.get("name") or f"az{azimuth:g}_el{elevation:g}",
            "direction": None,
            "azimuth": azimuth,
            "elevation": elevation,
            "projection": projection or "persp",
            **common,
        }
    direction = str(direction or "current").lower()
    if direction == "perspective":
        direction = "persp"
    if direction not in _AXIS_VIEWS and direction not in ("current", "persp", "facing"):
        raise ValueError(
            f"views[{index}]: unknown direction {direction!r}; use one of "
            f"{sorted(_AXIS_VIEWS) + ['persp', 'current', 'facing']}, or azimuth/elevation"
        )
    if projection is not None and direction in _AXIS_VIEWS and projection != "ortho":
        raise ValueError(f"views[{index}]: the {direction} view is orthographic")
    return {
        "name": spec.get("name") or direction,
        "direction": direction,
        "azimuth": None,
        "elevation": None,
        "projection": projection,
        **common,
    }


def safe_name(name: Any) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name)) or "view"


###### Targets


def _target_corners(targets: Any, bbox: Any) -> tuple[list, list[str], Any]:
    """Framing corners, the target paths to draw, and what the framing came from.

    targets decide what is drawn; bbox, when given, decides the framing on its
    own, so a close-up of part of a target is one call. Without a bbox the
    union of the targets' bounding boxes is framed. The third value is
    "bbox", the list of target paths, or None when there is nothing to frame.
    """
    target_corners: list = []
    described: list[str] = []
    if isinstance(targets, str):
        targets = [targets]
    for path in targets or []:
        node = hou.node(str(path))
        if node is None:
            raise ValueError(f"target {path!r} does not exist")
        geometry, transform = _geometry_of(node)
        if geometry is None or geometry.intrinsicValue("pointcount") == 0:
            raise ValueError(f"target {node.path()} has no points to frame")
        box = geometry.boundingBox()
        target_corners += _box_corners(list(box.minvec()), list(box.maxvec()), transform)
        described.append(node.path())
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 6:
            raise ValueError(f"bbox must be [xmin, ymin, zmin, xmax, ymax, zmax], not {bbox!r}")
        lo, hi = [float(v) for v in bbox[:3]], [float(v) for v in bbox[3:]]
        if any(h < low for low, h in zip(lo, hi, strict=True)):
            raise ValueError(f"bbox max is below min: {bbox!r}")
        return _box_corners(lo, hi, None), described, "bbox"
    return target_corners, described, (described or None)


def _frame_box(corners: list) -> list[float] | None:
    """[xmin, ymin, zmin, xmax, ymax, zmax] of the framing corners, for the receipt."""
    if not corners:
        return None
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    return [round(v, 6) for v in lo + hi]


def _geometry_of(node: hou.Node) -> tuple[Any, Any]:
    """(cooked geometry, object world transform) for a SOP or object node."""
    category = node.type().category().name()
    if category == "Sop":
        creator = node.creator()
        transform = None
        with contextlib.suppress(Exception):
            if creator is not None and creator.type().category().name() == "Object":
                transform = creator.worldTransform()
        return node.geometry(), transform
    if category == "Object":
        sop = node.displayNode() if hasattr(node, "displayNode") else None
        if sop is None:
            raise ValueError(f"target {node.path()} has no display SOP to frame")
        return sop.geometry(), node.worldTransform()
    raise ValueError(
        f"target {node.path()} is a {category} node; frame SOP or object nodes, "
        f"or pass bbox in world coordinates"
    )


def _box_corners(lo: list, hi: list, transform: Any) -> list:
    corners = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                point = hou.Vector3(x, y, z)
                if transform is not None:
                    point = point * transform
                corners.append((point[0], point[1], point[2]))
    return corners


def _center_radius(corners: list) -> tuple[list[float], float]:
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    center = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
    radius = math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3))) / 2.0
    return center, max(radius, 1e-6)


###### Viewport camera and viewer state


def _read_camera(viewport) -> dict[str, Any]:
    camera = viewport.defaultCamera()
    return {
        "axes": _axes_of(camera.rotation()),
        "pivot": list(camera.pivot()),
        "t": list(camera.translation()),
        "ortho": bool(camera.isOrthographic()),
        "ortho_width": float(camera.orthoWidth()),
        "focal": float(camera.focalLength()),
        "aperture": float(camera.aperture()),
    }


def _write_camera(viewport, params: dict[str, Any]) -> None:
    """Write the fields of *params* that differ from what the camera holds.

    Only those: the fixed Front/Top/... views refuse setRotation outright
    ("Cannot change the rotation of fixed orthographic views"), even to the
    rotation they already have.
    """
    current = _read_camera(viewport)
    camera = viewport.defaultCamera()
    if not _close(current["axes"], params["axes"]):
        camera.setRotation(_rotation_of(params["axes"]))
    if current["ortho"] != params["ortho"]:
        camera.setPerspective(not params["ortho"])
    camera.setPivot(hou.Vector3(params["pivot"]))
    camera.setTranslation(hou.Vector3(params["t"]))
    # After setPerspective: the projection switch must not get the last word
    # on the width.
    camera.setOrthoWidth(params["ortho_width"])
    if not _close(current["focal"], params["focal"]):
        camera.setFocalLength(params["focal"])
    if not _close(current["aperture"], params["aperture"]):
        camera.setAperture(params["aperture"])
    viewport.setDefaultCamera(camera)


def _camera_differences(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    return [key for key in expected if not _close(expected[key], actual[key])]


def _close(a: Any, b: Any, tolerance: float = 1e-5) -> bool:
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tolerance) for x, y in zip(a, b, strict=True))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return abs(float(a) - float(b)) <= tolerance * max(1.0, abs(float(a)))


def _save_state(viewport) -> dict[str, Any]:
    state: dict[str, Any] = {
        "type": viewport.type(),
        "camera": _read_camera(viewport),
        "camera_node": None,
        "camera_path": None,
        "locked": False,
        "shading": {},
        # The camera of every view type this call switches to, as found.
        "visited": {},
    }
    with contextlib.suppress(Exception):
        state["camera_node"] = viewport.camera()
    with contextlib.suppress(Exception):
        state["camera_path"] = viewport.cameraPath() or None
    with contextlib.suppress(Exception):
        state["locked"] = bool(viewport.isCameraLockedToView())
    settings = viewport.settings()
    for name in _SHADED_SETS:
        with contextlib.suppress(Exception):
            display_set = settings.displaySet(getattr(hou.displaySetType, name))
            state["shading"][name] = display_set.shadedMode()
    return state


def _enter_type(viewport, state: dict[str, Any], view_type) -> dict[str, Any]:
    """Switch the viewport type; returns that type's camera as the caller had it.

    As the caller had it, not as the previous view left it: a tall frame
    narrows the lens, and the next view reading the lens back would narrow it
    again.
    """
    if viewport.type() != view_type:
        viewport.changeType(view_type)
    if view_type == state["type"]:
        return dict(state["camera"])
    key = str(view_type)
    if key not in state["visited"]:
        state["visited"][key] = (view_type, _read_camera(viewport))
    return dict(state["visited"][key][1])


def _restore_state(viewport, state: dict[str, Any], base_camera: dict[str, Any]) -> list[str]:
    """Put the viewer back; returns what could not be put back."""
    problems: list[str] = []
    for key, (view_type, params) in state["visited"].items():
        try:
            viewport.changeType(view_type)
            _write_camera(viewport, params)
        except Exception as exc:  # noqa: BLE001 - reported
            problems.append(f"{key} camera: {exc}")
    try:
        viewport.changeType(state["type"])
        _write_camera(viewport, state["camera"])
    except Exception as exc:  # noqa: BLE001 - reported
        problems.append(f"viewport camera: {exc}")
    if state["camera_node"] is not None or state["camera_path"]:
        try:
            viewport.setCamera(state["camera_node"] or state["camera_path"])
            if state["locked"]:
                viewport.lockCameraToView(True)
        except Exception as exc:  # noqa: BLE001 - reported
            problems.append(f"camera {state['camera_path']}: {exc}")
    settings = viewport.settings()
    for name, mode in state["shading"].items():
        try:
            settings.displaySet(getattr(hou.displaySetType, name)).setShadedMode(mode)
        except Exception as exc:  # noqa: BLE001 - reported
            problems.append(f"{name} shading: {exc}")
    return problems


def _restore_mismatches(viewport, state: dict[str, Any]) -> list[str]:
    """Read the viewer back and name every way it differs from *state*."""
    mismatches: list[str] = []
    if viewport.type() != state["type"]:
        mismatches.append(f"view type is {viewport.type()}, was {state['type']}")
    differences = _camera_differences(state["camera"], _read_camera(viewport))
    if differences:
        mismatches.append(f"viewport camera differs in {differences}")
    now = None
    with contextlib.suppress(Exception):
        now = viewport.cameraPath() or None
    if now != (state["camera_path"] or None):
        mismatches.append(f"looking through {now}, was {state['camera_path']}")
    settings = viewport.settings()
    for name, mode in state["shading"].items():
        with contextlib.suppress(Exception):
            current = settings.displaySet(getattr(hou.displaySetType, name)).shadedMode()
            if current != mode:
                mismatches.append(f"{name} shading is {current.name()}, was {mode.name()}")
    return mismatches


def _apply_shading(viewport, shading: str) -> None:
    """Set *shading* on the display sets that draw the model; read it back."""
    token = _SHADING_NAMES[str(shading).lower()]
    mode = getattr(hou.glShadingType, token)
    settings = viewport.settings()
    wrong: dict[str, str] = {}
    for name in _SHADED_SETS:
        display_set = settings.displaySet(getattr(hou.displaySetType, name))
        display_set.setShadedMode(mode)
        if display_set.shadedMode() != mode:
            wrong[name] = display_set.shadedMode().name()
    if wrong:
        raise RuntimeError(f"asked for {token} shading, the viewport reports {wrong}")


def _current_shading(viewport) -> dict[str, str]:
    settings = viewport.settings()
    shading: dict[str, str] = {}
    for name in _SHADED_SETS:
        with contextlib.suppress(Exception):
            shading[name] = (
                settings.displaySet(getattr(hou.displaySetType, name)).shadedMode().name()
            )
    return shading


def _adopt_camera_view(viewport, camera_node) -> dict[str, Any] | None:
    """The viewport camera given the viewpoint and lens of *camera_node*.

    Everything but the camera's resolution and aspect, which is the part that
    decides what gets cut off. None when the node is not a camera object.
    """
    try:
        transform = camera_node.worldTransform()
        rotation = transform.extractRotationMatrix3()
        position = transform.extractTranslates()
    except Exception:  # noqa: BLE001 - not a camera object; the caller says so
        return None
    params = _read_camera(viewport)
    # An object transform's rows are its local axes in world space; a camera
    # looks down its local -Z.
    axes = [list(row) for row in rotation.asTupleOfTuples()]
    distance = 5.0
    pivot = [position[i] - axes[2][i] * distance for i in range(3)]
    params.update(axes=axes, pivot=pivot, t=[pivot[0], pivot[1], pivot[2] + distance])
    projection = None
    with contextlib.suppress(Exception):
        projection = camera_node.parm("projection").evalAsString()
    params["ortho"] = projection == "ortho"
    with contextlib.suppress(Exception):
        if params["ortho"]:
            params["ortho_width"] = float(camera_node.parm("orthowidth").eval())
        else:
            params["focal"] = float(camera_node.parm("focal").eval())
            params["aperture"] = float(camera_node.parm("aperture").eval())
    _write_camera(viewport, params)
    return params


###### One view


def _orient(viewport, state: dict[str, Any], view: dict[str, Any], base: dict[str, Any]):
    """Point the viewport for *view*, starting from the view the caller had.

    Returns the camera parameters to frame from.
    """
    direction = view["direction"]
    if direction in _AXIS_VIEWS:
        return _enter_type(
            viewport, state, getattr(hou.geometryViewportType, _AXIS_VIEWS[direction])
        )
    if direction == "current":
        _enter_type(viewport, state, state["type"])
        params = dict(base)
    else:
        params = _enter_type(viewport, state, hou.geometryViewportType.Perspective)
        if direction is None:
            params["axes"] = view_rotation(view["azimuth"], view["elevation"])
    if view["projection"] is not None:
        params["ortho"] = view["projection"] == "ortho"
    return params


def _flipbook(
    scene_viewer, viewport, path: str, out_w: int, out_h: int, visible: list[str] | None
) -> str:
    settings = scene_viewer.flipbookSettings().stash()
    frame = hou.frame()
    settings.frameRange((frame, frame))
    settings.output(path)
    _no_mplay(settings)
    settings.useResolution(True)
    settings.resolution((int(out_w), int(out_h)))
    if visible:
        # A mask on this copy of the settings only: the viewer, the scene and
        # the display flags are untouched. Measured on 22.0.368: it hides the
        # other displayed objects at /obj and their ghosts inside a SOP
        # network, and full paths match objects nested in subnets.
        settings.visibleObjects(" ".join(visible))
    scene_viewer.flipbook(viewport, settings)
    return _find_flipbook_output(path, frame)


def _owning_object(node):
    """The object *node* is part of: itself for an object node, else None."""
    while node is not None:
        with contextlib.suppress(Exception):
            if node.type().category().name() == "Object":
                return node
        node = node.parent()
    return None


def _parse_isolate(isolate: Any) -> bool | list[str]:
    """True/False, or the object paths to show; node paths map to their object."""
    if isinstance(isolate, bool):
        return isolate
    if isinstance(isolate, str):
        isolate = [isolate]
    if not isinstance(isolate, (list, tuple)) or not isolate:
        raise ValueError(f"isolate must be true, false or a list of node paths, not {isolate!r}")
    objects: list[str] = []
    for path in isolate:
        node = hou.node(path) if isinstance(path, str) else None
        if node is None:
            raise ValueError(f"isolate: no node at {path!r}")
        owner = _owning_object(node)
        if owner is None:
            raise ValueError(f"isolate: {node.path()} is not an object or inside one")
        objects.append(owner.path())
    return sorted(set(objects))


def _isolated_objects(isolate: bool | list[str], described: list[str]) -> list[str] | None:
    """The objects this view draws alone, or None to draw what the viewer shows."""
    if isolate is False:
        return None
    if isolate is not True:
        return isolate
    owners = [_owning_object(hou.node(path)) for path in described]
    return sorted({owner.path() for owner in owners if owner is not None}) or None


def _pixel_facts(path: str, rect: list[float] | None) -> dict[str, Any]:
    """Size of the written image, how much of *rect* is drawn on, and where.

    The flipbook writes the background transparent (measured on 22.0.368),
    so a pixel with any alpha is drawn. An image without transparency falls
    back to colour: the background is the median of the side edges (the HUD
    sits in the corners, and a framed target leaves the margin clear) and a
    pixel counts as drawn when it differs from it by more than
    _DRAWN_THRESHOLD. Without a rect the whole image is checked.
    content_rect is the box of the drawn pixels inside the rect (the sheet
    crops to it).
    """
    try:
        import numpy as np

        try:
            from PySide6 import QtGui
        except ImportError:
            from PySide2 import QtGui
    except ImportError as exc:
        return {"pixel_check": f"unavailable: {exc}"}
    image = QtGui.QImage(path)
    if image.isNull():
        return {"readable": False}
    image = image.convertToFormat(QtGui.QImage.Format_ARGB32)
    w, h = image.width(), image.height()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    # ARGB32 is B, G, R, A in memory.
    channels = raw.reshape(h, image.bytesPerLine())[:, : w * 4].reshape(h, w, 4)
    if rect is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0 = min(max(int(math.floor(rect[0])), 0), w - 1)
        y0 = min(max(int(math.floor(rect[1])), 0), h - 1)
        x1 = min(max(int(math.ceil(rect[2])), x0 + 1), w)
        y1 = min(max(int(math.ceil(rect[3])), y0 + 1), h)
    alpha = channels[:, :, 3]
    if (alpha < 255).any():
        drawn = alpha[y0:y1, x0:x1] > 0
    else:
        pixels = channels[:, :, :3].astype(np.int16)
        lo, hi = int(h * 0.2), max(int(h * 0.8), int(h * 0.2) + 1)
        edges = np.concatenate([pixels[lo:hi, 0], pixels[lo:hi, w - 1]])
        background = np.median(edges, axis=0)
        drawn = np.abs(pixels[y0:y1, x0:x1] - background).sum(axis=2) > _DRAWN_THRESHOLD
    facts: dict[str, Any] = {
        "readable": True,
        "pixels": [w, h],
        "drawn_fraction": round(float(drawn.mean()), 4),
    }
    ys, xs = np.nonzero(drawn)
    if len(xs):
        facts["content_rect"] = [
            int(xs.min()) + x0,
            int(ys.min()) + y0,
            int(xs.max()) + x0 + 1,
            int(ys.max()) + y0 + 1,
        ]
    return facts


def _shoot(
    scene_viewer,
    viewport,
    state,
    base,
    view,
    corners,
    path,
    max_size,
    margin,
    visible,
    fixed=None,
):
    """One flipbook. *fixed* ({"camera", "pixels"}) replays a stored camera as is."""
    size = viewport.size()
    width, height = float(size[2]), float(size[3])
    rect = None
    if fixed is not None:
        # A stored camera: the perspective viewport takes any rotation and
        # either projection, so every stored view replays there unchanged.
        _enter_type(viewport, state, hou.geometryViewportType.Perspective)
        params = {k: fixed["camera"][k] for k in _CAMERA_KEYS}
        out_w, out_h = (int(v) for v in fixed["pixels"])
        _write_camera(viewport, params)
    else:
        params = _orient(viewport, state, view, base)
        if corners:
            center, radius = _center_radius(corners)
            params, out_w, out_h, _ = fit(
                corners,
                centred_on(params, center, radius),
                width,
                height,
                max_size,
                margin,
                min_distance=radius * 1.05,
            )
            _write_camera(viewport, params)
        else:
            _write_camera(viewport, params)
            if view["direction"] != "current":
                viewport.frameAll()
            if width >= height:
                out_w, out_h = int(max_size), max(_MIN_SIDE, round(max_size * height / width))
            else:
                out_w, out_h = max(_MIN_SIDE, round(max_size * width / height)), int(max_size)

    # What Houdini holds now, not what was asked for.
    placed = _read_camera(viewport)
    azimuth, elevation = extras.angles_of(placed["axes"])
    shot: dict[str, Any] = {
        "name": view["name"],
        "direction": view["direction"]
        or {"azimuth": view["azimuth"], "elevation": view["elevation"]},
        "viewport_type": viewport.type().name(),
        "projection": "ortho" if placed["ortho"] else "persp",
        # The camera's own angles, whatever the view was asked as: shoot a
        # close-up from this cell with these.
        "azimuth": azimuth,
        "elevation": elevation,
        "pivot": [round(v, 6) for v in placed["pivot"]],
        # The camera looks along -back.
        "looking_along": [round(-v, 6) for v in placed["axes"][2]],
    }
    if placed["ortho"]:
        shot["ortho_width"] = round(placed["ortho_width"], 6)
    else:
        shot["distance"] = round(placed["t"][2] - placed["pivot"][2], 6)
        # Horizontal field of view of the written image, which a tall frame
        # narrows (see fit).
        shot["fov_x_deg"] = round(
            math.degrees(2.0 * math.atan(placed["aperture"] / 2.0 / placed["focal"])), 2
        )
    if _camera_differences(params, placed):
        shot["camera_not_as_set"] = _camera_differences(params, placed)
    if corners:
        points = [project(c, placed, width, height) for c in corners]
        if any(p is None for p in points):
            shot["target_in_frame"] = False
        else:
            image_points = [to_image(p[0], p[1], width, height, out_w, out_h) for p in points]
            xs = [p[0] for p in image_points]
            ys = [p[1] for p in image_points]
            rect = [min(xs), min(ys), max(xs), max(ys)]
            shot["target_rect"] = [round(v, 1) for v in rect]
            shot["target_in_frame"] = (
                rect[0] >= -0.5
                and rect[1] >= -0.5
                and rect[2] <= out_w + 0.5
                and rect[3] <= out_h + 0.5
            )
            shot["target_fill"] = [
                round((rect[2] - rect[0]) / out_w, 3),
                round((rect[3] - rect[1]) / out_h, 3),
            ]

    shot["isolated"] = visible
    written, facts, retried = _flipbook_checked(
        scene_viewer, viewport, path, out_w, out_h, visible, rect
    )
    if retried:
        shot["retried"] = True
    shot["path"] = written
    shot["file_exists"] = os.path.isfile(written)
    if shot["file_exists"]:
        shot["bytes"] = os.path.getsize(written)
    shot.update(facts)
    shot.setdefault("pixels", [out_w, out_h])
    # For the shot list and the edge metric; popped before the receipt.
    shot["_camera"] = placed
    shot["_frame"] = (width, height, out_w, out_h, rect)
    return shot


_CAMERA_KEYS = ("axes", "pivot", "t", "ortho", "ortho_width", "focal", "aperture")


def _flipbook_checked(scene_viewer, viewport, path, out_w, out_h, visible, rect):
    """Flipbook and read the pixels back; one redraw and retry on a blank frame.

    Then the transparent background is filled with the viewport's own (see
    extras.flatten), so every image, alone or on a sheet, has the same one.
    """
    written = _flipbook(scene_viewer, viewport, path, out_w, out_h, visible)
    facts = _pixel_facts(written, rect) if os.path.isfile(written) else {"readable": False}
    retried = False
    if facts.get("drawn_fraction") is not None and facts["drawn_fraction"] < _BLANK_FRACTION:
        # A viewport that has just changed type has been seen to hand the
        # flipbook an empty frame.
        with contextlib.suppress(Exception):
            viewport.draw()
        time.sleep(0.2)
        written = _flipbook(scene_viewer, viewport, path, out_w, out_h, visible)
        facts = _pixel_facts(written, rect) if os.path.isfile(written) else {"readable": False}
        retried = True
    if facts.get("readable"):
        top, bottom = _background_colours(viewport)
        try:
            extras.flatten(written, top, bottom)
        except Exception as exc:  # noqa: BLE001 - reported with the shot
            facts["background_problem"] = f"{type(exc).__name__}: {exc}"
    return written, facts, retried


def _background_colours(viewport) -> tuple[tuple, tuple]:
    """Top and bottom background colours of the viewport's colour scheme."""
    settings = viewport.settings()
    colours = []
    for name, fallback in (("BackgroundColor", 0.3), ("BackgroundBottomColor", 0.25)):
        try:
            colours.append(tuple(settings.colorFromName(name).rgb()))
        except Exception:  # noqa: BLE001 - a scheme without the entry
            colours.append((fallback,) * 3)
    return colours[0], colours[1]


###### Lighting: headlight matcap, material display for the zebra

_MATCAP_DIR = os.path.join(tempfile.gettempdir(), "fxhoudinimcp_matcaps")


def _read_look(viewport) -> dict[str, Any]:
    """The viewport settings a matcap capture changes."""
    settings = viewport.settings()
    return {
        "material_type": settings.getDefaultMaterialType(),
        "matcap": settings.getDefaultMaterialMatCapFile(),
        "matcap_intensity": float(settings.getDefaultMaterialMatCapIntensity()),
        "materials": bool(settings.showingMaterials()),
    }


def _use_matcap(viewport, path: str) -> None:
    """Draw every surface with the default material as the matcap at *path*; read it back.

    Material display goes off for the shot, so a part with a material gets
    the same light as one without.
    """
    settings = viewport.settings()
    settings.setDefaultMaterialType(hou.viewportDefaultMaterial.MatCap)
    settings.setDefaultMaterialMatCapFile(path)
    settings.setDefaultMaterialMatCapIntensity(1.0)
    settings.showMaterials(False)
    look = _read_look(viewport)
    if (
        look["material_type"] != hou.viewportDefaultMaterial.MatCap
        or os.path.normcase(look["matcap"]) != os.path.normcase(path)
        or look["materials"]
    ):
        raise RuntimeError(f"the viewport did not take the matcap {path}: {look}")


def _restore_look(viewport, look: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    settings = viewport.settings()
    try:
        settings.setDefaultMaterialMatCapFile(look["matcap"])
        settings.setDefaultMaterialMatCapIntensity(look["matcap_intensity"])
        settings.setDefaultMaterialType(look["material_type"])
        settings.showMaterials(look["materials"])
    except Exception as exc:  # noqa: BLE001 - reported
        problems.append(f"default material: {exc}")
    return problems


def _look_mismatches(viewport, look: dict[str, Any]) -> list[str]:
    now = _read_look(viewport)
    return [
        f"{key} is {now[key]}, was {look[key]}"
        for key in look
        if (now[key] != look[key] if key != "matcap_intensity" else not _close(now[key], look[key]))
    ]


def _headlight_file() -> str:
    """The headlight matcap, written once per set of constants."""
    tag = hashlib.sha1(repr((extras.HEADLIGHT, extras.MATCAP_SIZE)).encode("utf-8")).hexdigest()[
        :10
    ]
    path = os.path.join(_MATCAP_DIR, f"headlight_{tag}.png").replace("\\", "/")
    if not os.path.isfile(path):
        os.makedirs(_MATCAP_DIR, exist_ok=True)
        extras.write_grey(extras.headlight_matcap(), path)
    return path


def _show_materials(viewport) -> None:
    """Material display on, read back: the zebra is a material on the proxy.

    With a headlight capture the default material stays the headlight matcap,
    so pole markers and anything else without a material keep that light.
    """
    settings = viewport.settings()
    settings.showMaterials(True)
    if not settings.showingMaterials():
        raise RuntimeError("the viewport did not turn material display on for the zebra")


def _drawn(scene_viewer) -> dict[str, Any]:
    """The viewer's network and its display node, for the receipt."""
    facts: dict[str, Any] = {"viewer_network": None, "display_node": None}
    with contextlib.suppress(Exception):
        network = scene_viewer.pwd()
        facts["viewer_network"] = network.path()
        display = network.displayNode() if hasattr(network, "displayNode") else None
        facts["display_node"] = display.path() if display is not None else None
    return facts


_PROXY_NAME = "__fxmcp_capture_proxy"


def _make_proxy(sops: list, chain: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """A temporary object at /obj that draws *sops* in world space.

    An Object Merge "into this object" on an object with no transform: the
    geometry the targets cook to, where it sits in the world, drawn by an
    object the flipbook mask can show alone. *chain* adds the clip and the
    overlays after it (extras.build_chain). No flag, network or parameter of
    the user's nodes is touched, and nothing enters the undo history. Returns
    (proxy object, facts); the caller destroys it.
    """
    chain = chain or {}
    with hou.undos.disabler():
        obj_root = hou.node("/obj")
        name, index = _PROXY_NAME, 1
        while obj_root.node(name) is not None:
            index += 1
            name = f"{_PROXY_NAME}{index}"
        proxy = obj_root.createNode("geo", name, run_init_scripts=False)
        try:
            merge = proxy.createNode("object_merge", "targets")
            merge.parm("numobj").set(len(sops))
            for i, node in enumerate(sops, start=1):
                merge.parm(f"objpath{i}").set(node.path())
            merge.parm("xformtype").set("local")
            measure, out = extras.build_chain(
                proxy, merge, chain.get("clip"), chain.get("overlays") or [], chain.get("zebra")
            )
            out.setDisplayFlag(True)
            out.setRenderFlag(True)
            proxy.setDisplayFlag(True)
            out.cook(force=True)
            errors = [e for n in proxy.children() for e in n.errors()]
            if errors:
                raise RuntimeError(f"the capture proxy chain failed: {errors[:3]}")
            merged = merge.geometry()
            points = merged.intrinsicValue("pointcount") if merged is not None else 0
            kept = measure.geometry().intrinsicValue("primitivecount")
        except Exception:
            proxy.destroy()
            raise
    expected = 0
    for node in sops:
        with contextlib.suppress(Exception):
            expected += node.geometry().intrinsicValue("pointcount")
    facts = {"proxy": proxy.path(), "proxy_points": points, "target_points": expected}
    if chain.get("clip") is not None:
        facts["clipped_prims"] = kept
    facts["_measure"] = measure.path()
    return proxy, facts


def _show_targets(
    scene_viewer,
    described,
    follow,
    show_target,
    moved_flags,
    proxies=None,
    force_proxy=False,
    chain=None,
) -> dict[str, Any]:
    """Point the viewer at the targets and report whether it draws them.

    The viewer draws the display node of the network it is in. A SOP target
    that is its network's display node is drawn there as the artist sees it.
    Any other SOP target -- upstream of the display flag, or in a network
    other than the one the viewer can follow -- is drawn through a temporary
    proxy object (see _make_proxy), with the viewer at /obj for the shot,
    unless show_target moves the display flag onto it instead. force_proxy
    (a clip, overlays or a comparison, which live in the proxy's chain)
    always takes the proxy. Without a proxy list (proxies=None) such a target
    is reported as framed but not drawn. *moved_flags* collects {network:
    previous display node} for every display flag moved here; *proxies*
    collects the proxy objects made.
    """
    nodes = [hou.node(path) for path in described]
    sops = [n for n in nodes if n is not None and n.type().category().name() == "Sop"]
    objects = [n for n in nodes if n is not None and n.type().category().name() == "Object"]
    facts: dict[str, Any] = {"via": "viewer"}
    if follow and sops and not force_proxy:
        networks = sorted({n.parent().path() for n in sops})
        if len(networks) == 1:
            if scene_viewer.pwd().path() != networks[0]:
                scene_viewer.setPwd(hou.node(networks[0]))
        else:
            facts["note"] = f"the targets are in {len(networks)} networks; the viewer shows one"
    network = scene_viewer.pwd()
    drawn: dict[str, bool] = {}
    for node in sops:
        parent = node.parent()
        display = parent.displayNode() if hasattr(parent, "displayNode") else None
        showing = display is not None and display.path() == node.path()
        if show_target and not showing and parent.path() == network.path() and not force_proxy:
            moved_flags.setdefault(parent.path(), display.path() if display is not None else None)
            node.setDisplayFlag(True)
            display = parent.displayNode()
            showing = display is not None and display.path() == node.path()
        drawn[node.path()] = showing and parent.path() == network.path() and not force_proxy
    if proxies is not None and sops and any(not drawn[n.path()] for n in sops):
        # One proxy for all SOP targets: the viewer can be in one network only,
        # and at /obj the natively drawn ones would disappear too.
        proxy, proxy_facts = _make_proxy(sops, chain)
        proxies.append(proxy.path())
        scene_viewer.setPwd(hou.node("/obj"))
        network = scene_viewer.pwd()
        whole = proxy_facts["proxy_points"] == proxy_facts["target_points"]
        for node in sops:
            drawn[node.path()] = whole
        facts.update(via="proxy", **proxy_facts)
        facts.pop("note", None)
    for node in objects:
        drawn[node.path()] = bool(node.isDisplayFlagSet())
    display = network.displayNode() if hasattr(network, "displayNode") else None
    facts.update(
        network=network.path(),
        display_node=display.path() if display is not None else None,
        targets_drawn=drawn,
    )
    return facts


def _restore_targets(
    scene_viewer, viewer_network: str, moved_flags, restore_view: bool, proxies=()
) -> list[str]:
    """Remove the proxies, put back the display flags moved and the viewer's network.

    The viewer's network goes back before its camera does (see
    capture_viewport): on 22.0.368 setPwd rewrites the viewport camera.
    """
    problems: list[str] = []
    for path in proxies:
        with hou.undos.disabler(), contextlib.suppress(Exception):
            node = hou.node(path)
            if node is not None:
                node.destroy()
        if hou.node(path) is not None:
            problems.append(f"the capture proxy {path} could not be removed")
    for network, previous in moved_flags.items():
        if previous is None:
            continue
        with contextlib.suppress(Exception):
            hou.node(previous).setDisplayFlag(True)
        now = None
        with contextlib.suppress(Exception):
            now = hou.node(network).displayNode()
        if now is None or now.path() != previous:
            problems.append(
                f"the display flag in {network} is on {now.path() if now else None}, was on {previous}"
            )
    if restore_view and scene_viewer.pwd().path() != viewer_network:
        with contextlib.suppress(Exception):
            scene_viewer.setPwd(hou.node(viewer_network))
        if scene_viewer.pwd().path() != viewer_network:
            problems.append(
                f"the viewer is in {scene_viewer.pwd().path()}, was in {viewer_network}"
            )
    return problems


def _edge_metric(shot, drawn, described) -> dict[str, Any] | None:
    """Mean projected edge length of what was drawn, in the image's pixels."""
    width, height, out_w, _, _ = shot["_frame"]
    segments = []
    measure = drawn.get("_measure")
    if measure and hou.node(measure) is not None:
        segments.append(extras.segments_of(hou.node(measure).geometry()))
    else:
        for path in described:
            node = hou.node(path)
            if node is None:
                continue
            with contextlib.suppress(Exception):
                geometry, transform = _geometry_of(node)
                if geometry is not None:
                    segments.append(extras.segments_of(geometry, transform))
    segments = [s for s in segments if len(s)]
    if not segments:
        return None
    import numpy as np

    return extras.edge_pixels(np.concatenate(segments), shot["_camera"], width, height, out_w)


def _resolve_view(spec, targets, bbox, region, clip):
    """Framing, drawing and clip of one view.

    Returns (corners, described, framed_by, region_facts or None, clip shape
    or None). A view's own targets/bbox/region/clip replace the shared ones.
    """
    view_targets = targets if spec["targets"] is None else spec["targets"]
    view_region = region if spec.get("region") is None else spec["region"]
    view_bbox = bbox if spec["bbox"] is None else spec["bbox"]
    if spec.get("region") is not None and spec["bbox"] is None:
        view_bbox = None
    if spec["bbox"] is not None and spec.get("region") is None:
        view_region = None
    if view_region is not None and view_bbox is not None:
        raise ValueError(f"view {spec['name']}: give bbox or region, not both")
    corners, described, framed_by = _target_corners(view_targets, view_bbox)
    facts = None
    if view_region is not None:
        first = next(
            (
                hou.node(p)
                for p in described
                if hou.node(p) is not None and hou.node(p).type().category().name() == "Sop"
            ),
            None,
        )
        facts = extras.region_facts(view_region, first)
        corners = facts["corners"]
        framed_by = f"region:{view_region['kind']}"
    view_clip = clip if spec.get("clip") is None else spec["clip"]
    shape = None
    if view_clip:
        if facts is not None:
            shape = extras.clip_shape(view_region, facts["frame_box"])
        elif view_bbox is not None:
            shape = {"shape": "box", "bbox": [float(v) for v in view_bbox]}
        else:
            raise ValueError(f"view {spec['name']}: clip needs a region or a bbox to clip to")
    if spec["direction"] == "facing" and (facts is None or facts["normal"] is None):
        raise ValueError(
            f"view {spec['name']}: 'facing' needs a region with faces in it to take the normal from"
        )
    return corners, described, framed_by, facts, shape


###### Handler: viewport.capture_viewport


def capture_viewport(
    output_dir: str,
    views: list | None = None,
    targets: Any = None,
    bbox: list | None = None,
    shading: str | None = None,
    max_size: int = 1600,
    margin: float = 0.05,
    prefix: str = "view",
    pane_name: str | None = None,
    restore_view: bool = True,
    follow_targets: bool = True,
    show_target: bool = False,
    isolate: Any = True,
    region: dict | None = None,
    clip: bool = False,
    compare: str | None = None,
    overlays: Any = None,
    zebra_direction: list | None = None,
    zebra_stripes: int = 16,
    orbit: Any = None,
    sheet: bool | None = None,
    sheet_columns: int | None = None,
    sheet_max: int = 2000,
    min_edge_px: float = 6.0,
    replay: str | None = None,
    lighting: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Flipbook the Scene Viewer from one or more views, framed on a target.

    See the module docstring for what is captured and how framing is checked.
    targets decide what is drawn; bbox or region, when given, decide the
    framing. follow_targets points the viewer at the SOP targets' network for
    the capture; a SOP target that is not its network's display node -- and
    every SOP target when clip, overlays or compare is used -- is drawn
    through a temporary proxy object at /obj (see _show_targets), or, with
    show_target, by moving the display flag onto it. Network, flags and
    proxies are put back or removed afterwards. isolate draws only the
    targets' objects (or the objects listed), so other displayed objects and
    their ghosts do not overlap the target.

    Every call writes <prefix>_shots.json (the solved cameras); replay reads
    one back and shoots the same views again. Several views are also laid out
    on <prefix>_sheet.png, each cell cropped to the drawn target.

    lighting "headlight" (default) draws every surface with a matcap lit from
    the camera, so a face turned to the camera is bright from any side and
    the wiring reads on the underside too. The viewport's own lights are not
    used for it: on 22.0.368 the work light's headAltitude/headAzimuth and
    headlightDirection change nothing in the drawn image, the light stays
    over the shoulder. "viewport" keeps the viewer's lighting and materials.
    The zebra overlay is a MaterialX material on the proxy, evaluated per
    pixel on the mesh's own normals; material display is on for it.
    Default material, matcap and material display are put back and read back.
    """
    started = time.perf_counter()
    if not isinstance(max_size, int) or isinstance(max_size, bool) or not 256 <= max_size <= 4096:
        raise ValueError(f"max_size must be an integer within 256..4096, not {max_size!r}")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0.0 <= margin <= 0.4:
        raise ValueError(f"margin must be within 0..0.4, not {margin!r}")
    if (
        isinstance(sheet_max, bool)
        or not isinstance(sheet_max, int)
        or not 256 <= sheet_max <= 8000
    ):
        raise ValueError(f"sheet_max must be an integer within 256..8000, not {sheet_max!r}")
    if isinstance(zebra_stripes, bool) or not isinstance(zebra_stripes, int) or zebra_stripes < 1:
        raise ValueError(f"zebra_stripes must be a positive integer, not {zebra_stripes!r}")
    zebra = {"direction": list(zebra_direction or [0.0, 1.0, 0.0]), "stripes": zebra_stripes}
    if len(zebra["direction"]) != 3 or not any(float(c) for c in zebra["direction"]):
        raise ValueError(f"zebra_direction must be a non-zero [x, y, z], not {zebra_direction!r}")

    record = None
    if replay is not None:
        if views is not None or orbit or bbox is not None or region is not None:
            raise ValueError(
                "replay takes the views from the shot list; drop views/orbit/bbox/region"
            )
        record = extras.read_shot_list(replay)
        targets = record.get("targets")
        compare = record.get("compare") if compare is None else compare
        overlays = record.get("overlays") if overlays is None else overlays
        zebra = record.get("zebra") or zebra
        shading = record.get("shading") if shading is None else shading
        isolate = record.get("isolate", isolate)
        lighting = record.get("lighting") if lighting is None else lighting
    lighting = "headlight" if lighting is None else str(lighting).lower()
    if lighting not in _LIGHTING:
        raise ValueError(f"lighting must be one of {list(_LIGHTING)}, not {lighting!r}")
    overlays = extras.parse_overlays(overlays)
    if shading is not None and str(shading).lower() not in _SHADING_NAMES:
        raise ValueError(f"unknown shading {shading!r}; use one of {sorted(_SHADING_NAMES)}")
    region = extras.parse_region(region)
    if compare is not None:
        source = hou.node(str(compare))
        if source is None or source.type().category().name() != "Sop":
            raise ValueError(f"compare must be a SOP node path, not {compare!r}")
        compare = source.path()

    fixed: list[dict[str, Any] | None]
    if record is not None:
        specs, fixed = [], []
        for shot in record["shots"]:
            specs.append(
                {
                    "name": shot["name"],
                    "direction": None,
                    "azimuth": shot.get("azimuth"),
                    "elevation": shot.get("elevation"),
                    "projection": None,
                    "targets": shot.get("targets"),
                    "bbox": None,
                    "region": None,
                    "clip": None,
                    "_clip_shape": shot.get("clip"),
                }
            )
            fixed.append(shot)
    else:
        raw = list(views or [])
        specs = [parse_view(spec, i) for i, spec in enumerate(raw)]
        specs += [
            parse_view(spec, len(specs) + i) for i, spec in enumerate(extras.orbit_views(orbit))
        ]
        if not specs:
            specs = [parse_view("current", 0)]
        fixed = [None] * len(specs)
    names = [safe_name(spec["name"]) for spec in specs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"view names must be unique, repeated: {duplicates}")

    # Targets and regions are resolved before the viewer is touched, so a bad
    # path costs nothing but the error. A view's own targets, bbox, region and
    # clip replace the shared ones one by one: a per-view bbox keeps the
    # shared targets as what is drawn (targets: [] drops them).
    isolate = _parse_isolate(isolate)
    per_view = []
    for spec in specs:
        if record is not None:
            # The stored framing box, so the in-frame check means what it did.
            box = next(
                s.get("frame_bbox") for s in record["shots"] if safe_name(s["name"]) == spec["name"]
            )
            _, described, _ = _target_corners(spec["targets"], None)
            corners = _box_corners(box[:3], box[3:], None) if box else []
            per_view.append((corners, described, "replay", None, spec["_clip_shape"]))
        else:
            per_view.append(_resolve_view(spec, targets, bbox, region, clip))

    os.makedirs(output_dir, exist_ok=True)
    scene_viewer = _find_scene_viewer(pane_name)
    viewport = scene_viewer.curViewport()
    size = viewport.size()
    state = _save_state(viewport)
    base = state["camera"]
    viewer_network = scene_viewer.pwd().path()
    moved_flags: dict[str, str | None] = {}
    proxies: list[list[str]] = []
    restore_problems: list[str] = []
    stem = safe_name(prefix)

    result: dict[str, Any] = {
        "viewport": viewport.name(),
        "viewport_px": [int(size[2]), int(size[3])],
        # What the images show: the viewer's network and, inside a SOP
        # network, the node whose geometry is drawn as the edited model.
        **_drawn(scene_viewer),
        "camera_detached": None,
        "views": [],
        "problems": [],
        "warnings": [],
    }
    if overlays:
        result["overlays"] = {o: extras.OVERLAY_LEGEND[o] for o in overlays}
    shot_list: list[dict[str, Any]] = []
    # Default material, matcap and material display as found; put back after
    # a headlight or zebra capture whatever restore_view says.
    look = _read_look(viewport)
    look_changed = False
    use_zebra = "zebra" in overlays
    result["lighting"] = {"mode": lighting}
    if use_zebra:
        result["lighting"]["zebra"] = (
            "the targets are drawn with a per-pixel zebra material on the proxy instead"
        )
    try:
        if lighting == "headlight" or use_zebra:
            look_changed = True
            if lighting == "headlight":
                result["lighting"]["matcap"] = _headlight_file()
                _use_matcap(viewport, result["lighting"]["matcap"])
            if use_zebra:
                _show_materials(viewport)

        if state["camera_node"] is not None or state["camera_path"]:
            # A camera node's resolution and aspect would decide the frame, and
            # moving the view would move the camera. The viewport's own camera
            # is used instead and the camera is put back afterwards; "current"
            # still means the camera's viewpoint and lens.
            viewport.useDefaultCamera()
            detached = state["camera_path"] or state["camera_node"].path()
            result["camera_detached"] = detached
            adopted = None
            if state["camera_node"] is not None:
                adopted = _adopt_camera_view(viewport, state["camera_node"])
            if adopted is not None:
                base = adopted
            else:
                result["camera_note"] = (
                    f"{detached} is not a camera object whose view could be copied; "
                    f"'current' views use the viewport's own camera"
                )
        if shading is not None:
            _apply_shading(viewport, shading)
        result["shading"] = _current_shading(viewport)
        for spec, name, (corners, described, framed_by, rfacts, shape), stored in zip(
            specs, names, per_view, fixed, strict=True
        ):
            path = os.path.join(output_dir, f"{stem}_{name}.png").replace("\\", "/")
            view = dict(spec)
            if spec["direction"] == "facing":
                azimuth, elevation = extras.facing_angles(rfacts["normal"])
                view.update(
                    direction=None,
                    azimuth=azimuth,
                    elevation=elevation,
                    projection=spec["projection"] or "persp",
                )
            chain = {"clip": shape, "overlays": overlays, "zebra": zebra}
            force_proxy = bool(shape or overlays or compare)
            shot_proxies: list[str] = []
            proxies += [shot_proxies]
            # Each view starts from the caller's network: an earlier view may have
            # followed its targets elsewhere or gone to /obj for a proxy. The
            # camera is written for every shot, so setPwd's camera change is moot.
            if scene_viewer.pwd().path() != viewer_network:
                scene_viewer.setPwd(hou.node(viewer_network))
            drawn = _show_targets(
                scene_viewer,
                described,
                follow_targets,
                show_target,
                moved_flags,
                shot_proxies,
                force_proxy,
                chain,
            )
            proxied = set()
            if drawn.get("via") == "proxy":
                proxied = {
                    p
                    for p in described
                    if hou.node(p) is not None and hou.node(p).type().category().name() == "Sop"
                }
                if isolate is True:
                    others = [p for p in described if p not in proxied]
                    visible = sorted(set(shot_proxies) | set(_isolated_objects(True, others) or []))
                elif isolate is False:
                    visible = None
                else:
                    visible = sorted(set(isolate) | set(shot_proxies))
            else:
                visible = _isolated_objects(isolate, described)
            if visible:
                hidden = []
                for target in drawn["targets_drawn"]:
                    if target in proxied:
                        continue
                    owner = _owning_object(hou.node(target))
                    if owner is None or owner.path() not in visible:
                        hidden.append(target)
                        drawn["targets_drawn"][target] = False
                if hidden:
                    drawn["hidden_by_isolate"] = hidden
            shot = _shoot(
                scene_viewer,
                viewport,
                state,
                base,
                view,
                corners,
                path,
                max_size,
                float(margin),
                visible,
                stored,
            )
            if spec["direction"] == "facing":
                shot["direction"] = {
                    "facing": rfacts["normal"],
                    "azimuth": view["azimuth"],
                    "elevation": view["elevation"],
                }
            # framed: what decided the framing ("bbox", "region:<kind>" or the
            # target paths); frame_bbox: the world box fitted into the image.
            shot["framed"] = framed_by
            shot["frame_bbox"] = _frame_box(corners)
            if rfacts is not None:
                shot["region"] = {
                    "node": rfacts["node"],
                    "faces": rfacts["prims"],
                    "normal": rfacts["normal"],
                }
            if shape is not None:
                shot["clip"] = shape
            metric = _edge_metric(shot, drawn, described)
            if metric is not None:
                shot["edge_px"] = metric
            if compare is not None and shot.get("file_exists"):
                _shoot_compare(
                    scene_viewer, viewport, shot, compare, chain, isolate, stem, name, described
                )
            drawn.pop("_measure", None)
            shot["drawn"] = drawn
            camera = shot.pop("_camera")
            shot.pop("_frame")
            shot_list.append(
                {
                    "name": name,
                    "targets": described,
                    "azimuth": shot["azimuth"],
                    "elevation": shot["elevation"],
                    "framed": framed_by,
                    "frame_bbox": shot["frame_bbox"],
                    "clip": shape,
                    "camera": {k: camera[k] for k in _CAMERA_KEYS},
                    "pixels": list(shot["pixels"]),
                }
            )
            result["views"].append(shot)
            # A proxy lives for its own shot only: with isolate=False the next
            # view would draw it too.
            restore_problems += _restore_targets(
                scene_viewer, viewer_network, {}, False, shot_proxies
            )
            shot_proxies.clear()
    finally:
        # Network and flags first: setPwd rewrites the viewport camera, so the
        # camera is written back after it.
        restore_problems += _restore_targets(
            scene_viewer,
            viewer_network,
            moved_flags,
            restore_view,
            [p for group in proxies for p in group],
        )
        if restore_view:
            restore_problems = _restore_state(viewport, state, base) + restore_problems
        if look_changed:
            look_problems = _restore_look(viewport, look) + _look_mismatches(viewport, look)
            result["lighting"]["restored"] = not look_problems
            if look_problems:
                result["lighting"]["restore_mismatches"] = look_problems
            restore_problems += look_problems

    if shot_list:
        shots_path = os.path.join(output_dir, f"{stem}_shots.json").replace("\\", "/")
        extras.write_shot_list(
            shots_path,
            {
                "kind": "fxhoudinimcp.capture_viewport.shots",
                "version": 1,
                "targets": targets if record is None else record.get("targets"),
                "compare": compare,
                "overlays": overlays,
                "zebra": zebra,
                "shading": shading,
                "lighting": lighting,
                "isolate": isolate,
                "margin": margin,
                "shots": shot_list,
            },
        )
        result["shots_file"] = shots_path
    if record is not None:
        result["replayed"] = replay

    _make_sheet(
        result, output_dir, stem, sheet, sheet_columns, sheet_max, compare, overlays, min_edge_px
    )
    _readability(result, min_edge_px)

    for shot in result["views"]:
        if not shot.get("file_exists"):
            result["problems"].append(f"{shot['name']}: no image was written")
        elif shot.get("readable") is False:
            result["problems"].append(f"{shot['name']}: the image cannot be read")
        elif shot.get("drawn_fraction") is not None and shot["drawn_fraction"] < _BLANK_FRACTION:
            result["problems"].append(f"{shot['name']}: nothing is drawn where the target is")
        if shot.get("target_in_frame") is False:
            result["problems"].append(f"{shot['name']}: the target is not fully in frame")
        if shot.get("background_problem"):
            result["problems"].append(
                f"{shot['name']}: background not filled: {shot['background_problem']}"
            )
        if (shot.get("drawn") or {}).get("clipped_prims") == 0:
            result["problems"].append(f"{shot['name']}: the clip region holds no faces")
        pair = shot.get("compare") or {}
        if pair.get("problem"):
            result["problems"].append(f"{shot['name']}: {pair['problem']}")
        drawn = shot.get("drawn") or {}
        for target in drawn.get("hidden_by_isolate") or []:
            result["problems"].append(
                f"{shot['name']}: {target} is framed but its object is not in "
                f"isolate {shot.get('isolated')}"
            )
        for target, showing in (drawn.get("targets_drawn") or {}).items():
            if target in (drawn.get("hidden_by_isolate") or []):
                continue
            if not showing and drawn.get("via") == "proxy":
                result["problems"].append(
                    f"{shot['name']}: {target} is framed but the capture proxy drew "
                    f"{drawn.get('proxy_points')} points of the targets' "
                    f"{drawn.get('target_points')}"
                )
            elif not showing:
                result["problems"].append(
                    f"{shot['name']}: {target} is framed but not drawn -- the viewer draws "
                    f"{drawn.get('display_node')} in {drawn.get('network')}; set its display "
                    f"flag or pass show_target=True"
                )
        if shot.get("camera_not_as_set"):
            result["problems"].append(
                f"{shot['name']}: the viewport camera did not take {shot['camera_not_as_set']}"
            )
    if restore_view:
        mismatches = restore_problems + _restore_mismatches(viewport, state)
        result["restored"] = not mismatches
        if mismatches:
            result["restore_mismatches"] = mismatches
    else:
        result["restored"] = None
        # Display flags this call moved are put back whatever restore_view says.
        result["problems"] += restore_problems
    if not result["warnings"]:
        result.pop("warnings")
    result["success"] = not result["problems"] and result["restored"] is not False
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return result


def _shoot_compare(scene_viewer, viewport, shot, compare, chain, isolate, stem, name, described):
    """The source through its own proxy, same camera and size; then the pair image."""
    out_w, out_h = shot["pixels"]
    rect = shot["_frame"][4]
    source_path = os.path.join(os.path.dirname(shot["path"]), f"{stem}_{name}_source.png")
    source_path = source_path.replace("\\", "/")
    pair: dict[str, Any] = {"source": compare}
    proxy = None
    try:
        proxy, facts = _make_proxy([hou.node(compare)], chain)
        if scene_viewer.pwd().path() != "/obj":
            scene_viewer.setPwd(hou.node("/obj"))
            # setPwd rewrites the camera: put this shot's camera back.
            _write_camera(viewport, shot["_camera"])
        visible = [proxy.path()] if isolate is not False else None
        written, pixel, _ = _flipbook_checked(
            scene_viewer, viewport, source_path, out_w, out_h, visible, rect
        )
        pair.update(source_image=written, source_drawn_fraction=pixel.get("drawn_fraction"))
        if facts.get("clipped_prims") is not None:
            pair["source_clipped_prims"] = facts["clipped_prims"]
        if _camera_differences(shot["_camera"], _read_camera(viewport)):
            pair["problem"] = "the camera moved between the result and the source shot"
        elif not extras.file_ok(written):
            pair["problem"] = "no source image was written"
        else:
            combined = os.path.join(
                os.path.dirname(shot["path"]), f"{stem}_{name}_compare.png"
            ).replace("\\", "/")
            composed = extras.compose_pair(
                written,
                shot["path"],
                (f"SOURCE  {compare}", f"RESULT  {', '.join(described)}"),
                combined,
            )
            pair.update(image=composed["path"], pixels=composed["pixels"])
    except Exception as exc:  # noqa: BLE001 - reported as a problem
        pair["problem"] = f"compare failed: {type(exc).__name__}: {exc}"
    finally:
        if proxy is not None:
            with hou.undos.disabler(), contextlib.suppress(Exception):
                proxy.destroy()
    shot["compare"] = pair


def _make_sheet(
    result, output_dir, stem, sheet, columns, sheet_max, compare, overlays, min_edge_px
):
    shots = [s for s in result["views"] if s.get("file_exists")]
    if sheet is None:
        sheet = len(result["views"]) > 1
    if not sheet or not shots:
        return
    cells = []
    for shot in shots:
        pair = (shot.get("compare") or {}).get("image")
        label = f"{shot['name']}   az {shot['azimuth']:g}  el {shot['elevation']:g}"
        cell = {"path": pair or shot["path"], "label": label, "shot": shot}
        if not pair:
            # The cell shows the drawn target and a small border, not the
            # framing margin and the slack of the box corners around it.
            cell["crop"] = extras.crop_box(shot.get("content_rect"), shot["pixels"])
        cells.append(cell)
    header_bits = []
    if compare:
        header_bits.append("each cell: SOURCE left, RESULT right")
    if overlays:
        header_bits.append("; ".join(extras.OVERLAY_LEGEND[o] for o in overlays))
    path = os.path.join(output_dir, f"{stem}_sheet.png").replace("\\", "/")
    try:
        # Labels are drawn before the edge metric is known per cell scale, so
        # scale first, then label.
        sizes = []
        _, QtGui = extras._qt()
        for cell in cells:
            crop = cell.get("crop")
            if crop:
                sizes.append((crop[2] - crop[0], crop[3] - crop[1]))
            else:
                image = QtGui.QImage(cell["path"])
                sizes.append((image.width(), image.height()))
        head = extras.LABEL_H if header_bits else 0
        layout = extras.sheet_layout(sizes, sheet_max - head, columns)
        for cell, scale in zip(cells, layout["scales"], strict=True):
            shot = cell["shot"]
            edge = (shot.get("edge_px") or {}).get("median")
            if edge is not None:
                # A pair image holds the result at full size: same scale.
                shot["sheet_edge_px"] = round(edge * scale, 2)
                cell["label"] += f"   edge {shot['sheet_edge_px']:g}px"
                cell["warn"] = shot["sheet_edge_px"] < min_edge_px
            shot["sheet_scale"] = round(scale, 4)
            if cell.get("crop"):
                shot["sheet_crop"] = cell["crop"]
        composed = extras.compose_sheet(
            cells, path, sheet_max, columns, "   |   ".join(header_bits) or None
        )
    except Exception as exc:  # noqa: BLE001 - reported
        result["problems"].append(f"contact sheet failed: {type(exc).__name__}: {exc}")
        return
    result["sheet"] = {k: composed[k] for k in ("path", "pixels", "columns", "rows")}


def _readability(result, min_edge_px):
    """Warn where edges are too short on screen to read the wiring.

    The median, not the mean: a few long edges (a cover plate, a big flat
    face) lift the mean while the dense wiring next to them is unreadable.
    """
    for shot in result["views"]:
        where = "sheet" if "sheet_edge_px" in shot else "image"
        value = shot.get("sheet_edge_px")
        if value is None:
            value = (shot.get("edge_px") or {}).get("median")
        if value is None:
            continue
        if value < min_edge_px:
            result["warnings"].append(
                f"{shot['name']}: median edge {value:g}px in the {where} (< {min_edge_px:g}); "
                f"the wiring cannot be read -- split the views or shoot a close-up "
                f"(az {shot['azimuth']:g}, el {shot['elevation']:g})"
            )


register_handler("viewport.capture_viewport", capture_viewport)
