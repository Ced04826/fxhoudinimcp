"""Viewport and UI handlers for FXHoudini-MCP.

Provides tools for inspecting and controlling Houdini's viewport panes,
network editor navigation, display modes, camera assignment, and
screenshot capture of various pane types.
"""

from __future__ import annotations

# Built-in
import contextlib
import logging
import os
import time

# Third-party
import hou

# Internal
from fxhoudinimcp_server.dispatcher import register_handler
from fxhoudinimcp_server.ui import require_ui

logger = logging.getLogger(__name__)


def _no_mplay(settings) -> None:
    """Keep a flipbook from launching MPlay.

    MPlay is a child process of Houdini and inherits its file descriptors, the
    hwebserver listening socket among them. After Houdini exits, that MPlay
    keeps the port open and the next Houdini's auto-start fails with "an
    existing server is running on port N". The frame is written to disk either
    way; a viewer nobody asked for is not worth losing the next session's
    server.
    """
    with contextlib.suppress(Exception):
        settings.outputToMPlay(False)


def _capture_pane_tab_qt(pane_tab, output_path: str) -> None:
    """Capture a pane tab screenshot via Qt.

    Houdini 20.x exposed PaneTab.qtParentWidget(); Houdini 21 removed it
    in favor of qtParentWindow()/qtScreenGeometry(), so fall back to
    grabbing the pane's screen region.
    """
    pixmap = None

    if hasattr(pane_tab, "qtParentWidget"):
        try:
            widget = pane_tab.qtParentWidget()
        except Exception:
            widget = None
        if widget is not None:
            pixmap = widget.grab()

    if pixmap is None and hasattr(pane_tab, "qtScreenGeometry"):
        rect = pane_tab.qtScreenGeometry()
        try:
            from PySide6 import QtCore, QtGui
        except ImportError:
            from PySide2 import QtCore, QtGui
        # Render the pane's region from Houdini's own window. A screen grab
        # returns whatever is on top of Houdini at that moment: a terminal, a
        # browser, a white dialog (the blank capture in #38).
        window = None
        with contextlib.suppress(Exception):
            panel = pane_tab.pane().floatingPanel()
            window = hou.qt.floatingPanelWindow(panel) if panel else hou.qt.mainWindow()
        if window is not None:
            local = QtCore.QRect(window.mapFromGlobal(rect.topLeft()), rect.size())
            pixmap = window.grab(local)

    if pixmap is None and hasattr(pane_tab, "qtScreenGeometry"):
        screens = QtGui.QGuiApplication.screens()
        screen = QtGui.QGuiApplication.primaryScreen()
        for candidate in screens:
            if candidate.geometry().contains(rect.center()):
                screen = candidate
                break
        geometry = screen.geometry()
        pixmap = screen.grabWindow(
            0,
            rect.x() - geometry.x(),
            rect.y() - geometry.y(),
            rect.width(),
            rect.height(),
        )

    if pixmap is None:
        raise RuntimeError(
            f"Cannot capture pane '{pane_tab.name()}': no Qt capture API "
            f"available in this Houdini build."
        )

    if not pixmap.save(output_path):
        raise RuntimeError(
            f"Failed to save screenshot to '{output_path}'. "
            f"Ensure the path is writable and the format is supported (e.g. .png, .jpg)."
        )


def _viewer_stage(scene_viewer):
    """The USD stage a Solaris viewer is displaying, or None.

    Tried in order because which of these is a LOP depends on what the user last
    clicked: the current node, the network the viewer is in, and that network's
    display node.
    """
    candidates = []
    with contextlib.suppress(Exception):
        candidates.append(scene_viewer.currentNode())
    with contextlib.suppress(Exception):
        candidates.append(scene_viewer.pwd())
    with contextlib.suppress(Exception):
        candidates.append(scene_viewer.pwd().displayNode())
    for node in candidates:
        if node is None or not hasattr(node, "stage"):
            continue
        with contextlib.suppress(Exception):
            stage = node.stage()
            if stage is not None:
                return stage
    return None


def _camera_prims(stage, limit: int = 12) -> list[str]:
    """Camera prim paths on a stage, for error messages worth reading."""
    found: list[str] = []
    with contextlib.suppress(Exception):
        for prim in stage.Traverse():
            if str(prim.GetTypeName()) == "Camera":
                found.append(str(prim.GetPath()))
                if len(found) >= limit:
                    break
    return found


def _is_scene_graph_view(scene_viewer) -> bool:
    """Whether Hydra delegates apply at all.

    hydraRenderers() raises "Specified view is not a scene graph view" in an
    object-level viewport, which is a different situation from a build that has
    no such API, and reporting them the same way sent a caller looking for a
    Houdini upgrade instead of a Solaris viewport.
    """
    try:
        scene_viewer.hydraRenderers()
    except Exception:
        return False
    return True


###### viewport.list_panes


def list_panes() -> dict:
    """List all visible pane tabs, their types, and associated information."""
    require_ui("list Houdini's panes")
    pane_tabs = hou.ui.paneTabs()
    panes = []
    for pt in pane_tabs:
        info = {
            "name": pt.name(),
            "type": pt.type().name(),
            "is_current_tab": pt.isCurrentTab(),
        }
        # For scene viewers, add viewport info
        if pt.type() == hou.paneTabType.SceneViewer:
            try:
                cur_vp = pt.curViewport()
                info["current_viewport"] = cur_vp.name()
                info["viewport_count"] = len(pt.viewports())
            except (hou.OperationFailed, hou.ObjectWasDeleted, AttributeError) as e:
                logger.debug("Could not read viewport info for pane '%s': %s", pt.name(), e)
        # For network editors, add current path
        if pt.type() == hou.paneTabType.NetworkEditor:
            try:
                info["current_path"] = pt.pwd().path()
            except (hou.OperationFailed, hou.ObjectWasDeleted, AttributeError) as e:
                logger.debug("Could not read network editor path for pane '%s': %s", pt.name(), e)
        panes.append(info)

    return {
        "panes": panes,
        "count": len(panes),
    }


###### viewport.get_viewport_info


def get_viewport_info(pane_name: str = None) -> dict:
    """Get current viewport settings including camera, display mode, and view transform.

    Args:
        pane_name: Optional pane tab name. If None, uses the first Scene Viewer found.
    """
    scene_viewer = _find_scene_viewer(pane_name)
    viewport = scene_viewer.curViewport()

    info = {
        "pane_name": scene_viewer.name(),
        "viewport_name": viewport.name(),
    }

    # Camera
    try:
        cam = viewport.camera()
        info["camera"] = cam.path() if cam is not None else None
    except (hou.OperationFailed, hou.ObjectWasDeleted, AttributeError) as e:
        logger.debug("Could not read viewport camera: %s", e)
        info["camera"] = None

    # camera() returns a hou.Node, so it is None for a USD camera prim even when
    # the viewport is looking through one. cameraPath() answers for both, and
    # without it there was no way to ask "what am I actually looking through".
    info["camera_path"] = None
    with contextlib.suppress(Exception):
        info["camera_path"] = viewport.cameraPath() or None

    # cameraPath() is an echo of whatever was last set, and Houdini will happily
    # echo a path that does not exist. Repeating it unchecked would make this tool
    # a second source of the same false confidence, so resolve it: as a node, or
    # as a prim on the viewer's stage.
    if info["camera_path"]:
        resolved = hou.node(info["camera_path"]) is not None
        if not resolved:
            stage = _viewer_stage(scene_viewer)
            if stage is not None:
                with contextlib.suppress(Exception):
                    prim = stage.GetPrimAtPath(info["camera_path"])
                    resolved = bool(prim and prim.IsValid())
        info["camera_path_resolves"] = resolved

    # The active Hydra delegate. Nothing reported this before, so a viewport that
    # had silently reverted to GL looked identical to one rendering in Karma
    # until someone eyeballed a screenshot.
    info["renderer"] = None
    with contextlib.suppress(Exception):
        info["renderer"] = scene_viewer.currentHydraRenderer()
    with contextlib.suppress(Exception):
        info["available_renderers"] = list(scene_viewer.hydraRenderers())

    # Display mode / shading
    try:
        settings = viewport.settings()
        display_set = settings.displaySet(hou.displaySetType.SceneObject)
        info["shading_mode"] = str(display_set.shadedMode())
    except (hou.OperationFailed, AttributeError) as e:
        logger.debug("Could not read shading mode: %s", e)
        info["shading_mode"] = None

    # View transform (model-view matrix)
    try:
        xform = viewport.viewTransform()
        info["view_transform"] = [list(row) for row in xform.asTupleOfTuples()]
    except (hou.OperationFailed, AttributeError) as e:
        logger.debug("Could not read view transform: %s", e)
        info["view_transform"] = None

    # Viewport type (perspective, top, front, right, UV, etc.)
    try:
        info["viewport_type"] = str(viewport.type())
    except (hou.OperationFailed, AttributeError) as e:
        logger.debug("Could not read viewport type: %s", e)
        info["viewport_type"] = None

    return info


###### viewport.set_viewport_camera


def set_viewport_camera(
    camera_path: str,
    pane_name: str = None,
) -> dict:
    """Set the viewport to look through a specific camera.

    Args:
        camera_path: Path to the camera node (e.g. '/obj/cam1').
        pane_name: Optional pane tab name.
    """
    scene_viewer = _find_scene_viewer(pane_name)
    viewport = scene_viewer.curViewport()

    # A USD camera prim is not a hou.node, so resolving through hou.node() alone
    # made every Solaris shot unframeable: the tool raised "Camera node not
    # found" for a path that is perfectly valid on the stage. A recorded session
    # lost nine execute_python calls to this and still shipped the wrong framing.
    cam_node = hou.node(camera_path)
    if cam_node is not None:
        viewport.setCamera(cam_node)
        # camera() returns the node, so comparing it is real verification.
        bound = None
        with contextlib.suppress(Exception):
            bound = viewport.camera()
        if bound is None or bound.path() != cam_node.path():
            raise RuntimeError(
                f"Asked to look through '{camera_path}' but the viewport reports "
                f"'{bound.path() if bound else None}'."
            )
        return {
            "success": True,
            "verified": True,
            "camera_path": cam_node.path(),
            "is_usd_prim": False,
            "pane_name": scene_viewer.name(),
            "viewport_name": viewport.name(),
        }

    # A USD camera prim is not a hou.node, so resolving through hou.node() alone
    # made every Solaris shot unframeable. But cameraPath() cannot verify the
    # result either: it is a pure echo, and setCamera("/cameras/does_not_exist")
    # returns without error and leaves cameraPath() reporting that exact
    # nonexistent path. Verification has to be a stage lookup.
    stage = _viewer_stage(scene_viewer)
    if stage is None:
        raise ValueError(
            f"'{camera_path}' is not a node, and this viewport has no USD stage "
            f"to look it up on. A Solaris camera prim requires the viewer to be "
            f"in a LOP network."
        )
    prim = stage.GetPrimAtPath(camera_path)
    if prim is None or not prim.IsValid():
        cameras = _camera_prims(stage)
        raise ValueError(
            f"No prim at '{camera_path}' on this stage."
            + (f" Camera prims present: {cameras}" if cameras else " The stage has no cameras.")
        )
    type_name = str(prim.GetTypeName())
    if type_name != "Camera":
        raise ValueError(
            f"'{camera_path}' is a {type_name or 'typeless'} prim, not a Camera. "
            f"Camera prims present: {_camera_prims(stage)}"
        )

    viewport.setCamera(camera_path)
    echoed = None
    with contextlib.suppress(Exception):
        echoed = viewport.cameraPath() or None
    if echoed is not None and echoed.rstrip("/") != camera_path.rstrip("/"):
        raise RuntimeError(
            f"Asked to look through '{camera_path}' but the viewport is on '{echoed}'."
        )

    return {
        "success": True,
        "verified": True,
        "camera_path": camera_path,
        "is_usd_prim": True,
        "prim_type": type_name,
        "pane_name": scene_viewer.name(),
        "viewport_name": viewport.name(),
    }


###### viewport.set_viewport_display


def set_viewport_display(
    display_mode: str,
    pane_name: str = None,
) -> dict:
    """Set the viewport display/shading mode.

    Args:
        display_mode: One of 'wireframe', 'shaded', 'smooth', 'smooth_wire',
            'hidden_line', 'flat', 'flat_wire', 'point'.
        pane_name: Optional pane tab name.
    """
    mode_map = {
        "wireframe": hou.glShadingType.Wire,
        "wire": hou.glShadingType.Wire,
        "shaded": hou.glShadingType.Smooth,
        "smooth": hou.glShadingType.Smooth,
        "smooth_wire": hou.glShadingType.SmoothWire,
        "hidden_line": hou.glShadingType.HiddenLineGhost,
        "flat": hou.glShadingType.Flat,
        "flat_wire": hou.glShadingType.FlatWire,
        "matcap": hou.glShadingType.MatCap,
        "matcap_wire": hou.glShadingType.MatCapWire,
    }

    gl_mode = mode_map.get(display_mode.lower())
    if gl_mode is None:
        raise ValueError(
            f"Unknown display mode: '{display_mode}'. Supported modes: {list(mode_map.keys())}"
        )

    scene_viewer = _find_scene_viewer(pane_name)
    viewport = scene_viewer.curViewport()

    settings = viewport.settings()
    display_set = settings.displaySet(hou.displaySetType.SceneObject)
    display_set.setShadedMode(gl_mode)

    return {
        "success": True,
        "display_mode": display_mode,
        "pane_name": scene_viewer.name(),
    }


###### viewport.set_viewport_direction


def set_viewport_direction(
    direction: str,
    pane_name: str = None,
) -> dict:
    """Set the viewport to a standard viewing direction.

    Args:
        direction: One of 'front', 'back', 'top', 'bottom', 'left', 'right',
            'perspective'.
        pane_name: Optional pane tab name.
    """
    direction_map = {
        "front": hou.geometryViewportType.Front,
        "back": hou.geometryViewportType.Back,
        "top": hou.geometryViewportType.Top,
        "bottom": hou.geometryViewportType.Bottom,
        "left": hou.geometryViewportType.Left,
        "right": hou.geometryViewportType.Right,
        "perspective": hou.geometryViewportType.Perspective,
    }

    view_type = direction_map.get(direction.lower())
    if view_type is None:
        raise ValueError(
            f"Unknown direction '{direction}'. Supported: {list(direction_map.keys())}"
        )

    scene_viewer = _find_scene_viewer(pane_name)
    viewport = scene_viewer.curViewport()
    viewport.changeType(view_type)
    viewport.frameAll()

    return {
        "success": True,
        "direction": direction,
        "pane_name": scene_viewer.name(),
        "viewport_name": viewport.name(),
    }


###### viewport.set_viewport_renderer


def set_viewport_renderer(
    renderer: str,
    pane_name: str = None,
) -> dict:
    """Set the viewport's Hydra rendering delegate.

    In LOPs/Solaris the viewport can render through different Hydra delegates
    (GL, Storm, Karma CPU, Karma XPU, etc.) without writing to disk.

    Args:
        renderer: Renderer name — e.g. "GL", "Storm", "Karma CPU",
            "Karma XPU", "Houdini GL". Case-insensitive partial match.
        pane_name: Optional pane tab name.
    """
    scene_viewer = _find_scene_viewer(pane_name)
    viewport = scene_viewer.curViewport()

    # hou.SceneViewer.hydraRenderers/setHydraRenderer/currentHydraRenderer is the
    # documented pair for this. The previous implementation went through
    # viewport.settings().setRenderer() with an hscript fallback and, crucially,
    # never read the renderer back: it returned success whenever a setter did not
    # raise. A recorded session spent nine execute_python calls discovering that
    # the viewport was still on GL while this tool reported Karma, and every
    # screenshot it verified against was therefore the wrong image.
    available: list[str] = []
    with contextlib.suppress(Exception):
        available = list(scene_viewer.hydraRenderers())
    if not available:
        with contextlib.suppress(Exception):
            settings = viewport.settings()
            if hasattr(settings, "rendererNames"):
                available = list(settings.rendererNames())

    target = renderer.strip().lower()
    matched_name = next((n for n in available if n.lower() == target), None)
    if matched_name is None:
        matched_name = next((n for n in available if target in n.lower()), None)
    if matched_name is None and not available:
        matched_name = renderer
    if matched_name is None:
        raise ValueError(f"Renderer '{renderer}' not found. Available renderers: {available}")

    def _current() -> str | None:
        with contextlib.suppress(Exception):
            return scene_viewer.currentHydraRenderer()
        return None

    before = _current()
    attempts: list[str] = []
    for label, apply in (
        ("setHydraRenderer", lambda: scene_viewer.setHydraRenderer(matched_name)),
        (
            "settings.setRenderer",
            lambda: viewport.settings().setRenderer(matched_name),
        ),
        (
            "hscript viewdisplay",
            lambda: hou.hscript(f'viewdisplay -R "{matched_name}" {viewport.name()}'),
        ),
    ):
        try:
            apply()
        except Exception as exc:  # noqa: BLE001 - each route is best-effort
            attempts.append(f"{label}: {type(exc).__name__}")
            continue
        attempts.append(f"{label}: no error")
        if (_current() or "").lower() == matched_name.lower():
            break

    active = _current()
    # The renderer Houdini reports is the answer. Anything else is a guess about
    # whether a setter worked.
    applied = active is not None and active.lower() == matched_name.lower()
    if not applied and active is None:
        if not _is_scene_graph_view(scene_viewer):
            # Not a missing API: Hydra delegates only exist for a scene graph
            # view, so this request is meaningless in an object-level viewport
            # and saying "unverifiable" would hide a real mistake.
            raise ValueError(
                "This viewport is not a scene graph view, so it has no Hydra "
                "delegate to set. Hydra renderers (Karma, Storm) apply to a "
                "Solaris viewport: put the viewer in a LOP network first."
            )
        # A scene graph view with no readback: state genuinely unknown, so say so
        # rather than claiming the requested renderer is live.
        return {
            "success": True,
            "verified": False,
            "requested": matched_name,
            "renderer": None,
            "note": (
                "This Houdini exposes no currentHydraRenderer(), so the active "
                "renderer could not be confirmed. Capture a screenshot before "
                "trusting the viewport."
            ),
            "available_renderers": available,
            "attempts": attempts,
            "pane_name": scene_viewer.name(),
            "viewport_name": viewport.name(),
        }
    if not applied:
        raise RuntimeError(
            f"Asked for renderer '{matched_name}' but the viewport is still on "
            f"'{active}' (was '{before}'). Tried: {'; '.join(attempts)}. "
            f"Available renderers: {available}"
        )

    return {
        "success": True,
        "verified": True,
        "renderer": active,
        "previous_renderer": before,
        "available_renderers": available,
        "pane_name": scene_viewer.name(),
        "viewport_name": viewport.name(),
    }


###### viewport.frame_selection


def frame_selection(pane_name: str = None) -> dict:
    """Frame the current selection in the viewport.

    Args:
        pane_name: Optional pane tab name.
    """
    scene_viewer = _find_scene_viewer(pane_name)
    viewport = scene_viewer.curViewport()

    viewport.frameSelected()

    return {
        "success": True,
        "pane_name": scene_viewer.name(),
        "viewport_name": viewport.name(),
    }


###### viewport.frame_all


def frame_all(pane_name: str = None) -> dict:
    """Frame all geometry in the viewport (home all).

    Args:
        pane_name: Optional pane tab name.
    """
    scene_viewer = _find_scene_viewer(pane_name)
    viewport = scene_viewer.curViewport()

    viewport.homeAll()

    return {
        "success": True,
        "pane_name": scene_viewer.name(),
        "viewport_name": viewport.name(),
    }


###### viewport.capture_screenshot


def capture_screenshot(
    output_path: str,
    pane_name: str = None,
) -> dict:
    """Capture a screenshot of a specific pane tab, or the active viewport.

    Args:
        output_path: Destination image path.
        pane_name: Name of the pane tab to capture. If not provided,
            captures the first Scene Viewer found.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Named pane if asked for, otherwise the first Scene Viewer.
    pane_tab = _find_pane_by_name(pane_name) if pane_name is not None else _find_scene_viewer()

    cur_frame = hou.frame()

    # For scene viewers, use flipbook for capture
    if pane_tab.type() == hou.paneTabType.SceneViewer:
        viewport = pane_tab.curViewport()
        settings = pane_tab.flipbookSettings().stash()
        settings.frameRange((cur_frame, cur_frame))
        settings.output(output_path)
        _no_mplay(settings)
        pane_tab.flipbook(viewport, settings)

        # Handle frame number that flipbook may insert
        from fxhoudinimcp_server.handlers.rendering_handlers import _find_flipbook_output

        actual_path = _find_flipbook_output(output_path, cur_frame)
    else:
        actual_path = output_path
        # For other pane types, use Qt widget grab
        _capture_pane_tab_qt(pane_tab, output_path)

    return {
        "success": True,
        "pane_name": pane_tab.name(),
        "output_path": actual_path,
        "file_exists": os.path.isfile(actual_path),
    }


###### viewport.capture_network_editor


def capture_network_editor(
    output_path: str,
    node_path: str = None,
    network_path: str = None,
    bounds: list = None,
    margin: float = 0.05,
) -> dict:
    """Capture the network editor, framed on a whole network, a node, or bounds.

    The pane is put back where it was (network and visible area) and the
    selection and current node are never touched: framing a node used to
    select it, which changed what the user had picked.

    Args:
        output_path: Destination image path.
        node_path: Frame this node and its direct inputs and outputs.
        network_path: Frame this whole network (every node, box, sticky
            note and dot). Default: the pane's current network, whole.
        bounds: [x0, y0, x1, y1] in network units, inside network_path (or
            the pane's current network).
        margin: Border around the framed items, as a fraction of their size.
    """
    require_ui(
        "capture the network editor",
        alternative="Wiring can be read without a UI through list_children or get_node_info.",
    )
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 <= margin <= 1:
        raise ValueError(f"margin must be within 0..1, not {margin!r}")
    if node_path is not None and bounds is not None:
        raise ValueError("give node_path or bounds, not both")

    editors = [p for p in hou.ui.paneTabs() if p.type() == hou.paneTabType.NetworkEditor]
    if not editors:
        raise RuntimeError("No Network Editor pane on the current desktop.")
    # The largest one: a small editor draws a few nodes at an unreadable size,
    # and too small a pane has produced solid black images reported as success.
    pane = max(editors, key=lambda p: p.qtScreenGeometry().width() * p.qtScreenGeometry().height())
    geometry = pane.qtScreenGeometry()
    if geometry.width() < 300 or geometry.height() < 200:
        raise RuntimeError(
            f"The largest Network Editor pane ('{pane.name()}') is "
            f"{geometry.width()}x{geometry.height()} pixels, too small to capture "
            f"legibly. Enlarge it or switch to a desktop with a larger one."
        )

    focus = None
    if node_path is not None:
        focus = hou.node(node_path)
        if focus is None:
            raise ValueError(f"Node not found: {node_path}")
        network = focus.parent()
    elif network_path is not None:
        network = hou.node(network_path)
        if network is None:
            raise ValueError(f"Network not found: {network_path}")
    else:
        network = pane.pwd()

    if bounds is not None:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            raise ValueError(f"bounds must be [x0, y0, x1, y1], not {bounds!r}")
        x0, y0, x1, y1 = (float(v) for v in bounds)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"bounds are empty: {bounds!r}")
        framed, item_count = "bounds", None
    else:
        if focus is not None:
            items = [focus, *focus.inputs(), *focus.outputs()]
            items = [
                item
                for item in items
                if item is not None and item.parent().path() == network.path()
            ]
            framed = "node"
        else:
            items = _network_items(network)
            framed = "network"
        if not items:
            raise ValueError(f"{network.path()} has nothing to frame")
        x0, y0, x1, y1 = _items_rect(items)
        item_count = len(items)
    pad_x = max((x1 - x0) * margin, 0.5)
    pad_y = max((y1 - y0) * margin, 0.5)
    x0, x1, y0, y1 = x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y
    # Grow the short side to the pane's aspect, so the pane shows exactly this.
    aspect = float(geometry.width()) / geometry.height()
    if (x1 - x0) / (y1 - y0) < aspect:
        cx, half = (x0 + x1) / 2.0, (y1 - y0) * aspect / 2.0
        x0, x1 = cx - half, cx + half
    else:
        cy, half = (y0 + y1) / 2.0, (x1 - x0) / aspect / 2.0
        y0, y1 = cy - half, cy + half

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    old_network = pane.pwd()
    old_bounds = pane.visibleBounds()
    try:
        if pane.pwd().path() != network.path():
            pane.cd(network.path())
        pane.setVisibleBounds(hou.BoundingRect(x0, y0, x1, y1), 0.0)
        _let_qt_repaint()
        visible = pane.visibleBounds()
        _capture_pane_tab_qt(pane, output_path)
    finally:
        with contextlib.suppress(Exception):
            if pane.pwd().path() != old_network.path():
                pane.cd(old_network.path())
            pane.setVisibleBounds(old_bounds, 0.0)

    restored = pane.pwd().path() == old_network.path()
    result = {
        "success": True,
        "output_path": output_path,
        "file_exists": os.path.isfile(output_path),
        "pane_name": pane.name(),
        "network": network.path(),
        "framed": framed,
        "node_path": node_path,
        "visible_bounds": [round(v, 3) for v in (*visible.min(), *visible.max())],
        "pane_restored": restored,
    }
    if item_count is not None:
        result["items_framed"] = item_count
    if result["file_exists"]:
        result["bytes"] = os.path.getsize(output_path)
        facts = _image_spread(output_path)
        result.update(facts)
        if facts.get("uniform"):
            # A solid image is what a pane that did not draw produces, and it
            # used to come back as success.
            result["success"] = False
            result["error"] = (
                "The captured image is a single colour: the pane did not draw. "
                "Make sure the Network Editor is visible and not collapsed."
            )
    else:
        result["success"] = False
        result["error"] = "No image was written."
    if not restored:
        result["success"] = False
        result["error"] = f"The pane was left in {pane.pwd().path()}, not {old_network.path()}."
    return result


def _network_items(network) -> list:
    """Everything in *network* that has a place: nodes, boxes, notes, dots."""
    items: list = list(network.children())
    for reader in ("networkBoxes", "stickyNotes", "networkDots", "indirectInputs"):
        with contextlib.suppress(Exception):
            items.extend(getattr(network, reader)())
    return items


def _items_rect(items) -> tuple:
    """Bounding rectangle, in network units, of items with a position.

    A node's size() is its tile; its name is drawn to the right of it, and a
    frame on the tiles alone cut the last label in half.
    """
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for item in items:
        with contextlib.suppress(Exception):
            position = item.position()
            size = None
            with contextlib.suppress(Exception):
                size = item.size()
            w, h = (size[0], size[1]) if size is not None else (0.2, 0.2)
            label = 0.0
            if isinstance(item, (hou.Node, hou.SubnetIndirectInput)):
                text = item.name() if isinstance(item, hou.Node) else "Sub-Network Input #00"
                label = 0.3 + 0.14 * len(text)
            x0, y0 = min(x0, position[0]), min(y0, position[1])
            x1, y1 = max(x1, position[0] + w + label), max(y1, position[1] + h)
    if x0 == float("inf"):
        raise ValueError("none of the items reports a position")
    return x0, y0, x1, y1


def _let_qt_repaint() -> None:
    """Let the editor redraw the new area: Qt repaints dirty regions only, and
    a grab taken straight away shows the previous view in places."""
    try:
        from PySide6 import QtWidgets
    except ImportError:
        from PySide2 import QtWidgets
    for _ in range(8):
        QtWidgets.QApplication.processEvents()
        time.sleep(0.06)


def _image_spread(path: str) -> dict:
    """Pixel size of the image and whether it is a single colour."""
    try:
        from PySide6 import QtGui
    except ImportError:
        from PySide2 import QtGui
    image = QtGui.QImage(path)
    if image.isNull():
        return {"readable": False}
    w, h = image.width(), image.height()
    colours = set()
    step_x, step_y = max(1, w // 64), max(1, h // 64)
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            colours.add(image.pixel(x, y))
            if len(colours) > 4:
                return {"pixels": [w, h], "uniform": False}
    return {"pixels": [w, h], "uniform": len(colours) <= 1}


###### viewport.set_current_network


def set_current_network(network_path: str) -> dict:
    """Navigate the network editor to a specific network path.

    Args:
        network_path: Path to the network to navigate to (e.g. '/obj/geo1').
    """
    node = hou.node(network_path)
    if node is None:
        raise ValueError(f"Network path not found: {network_path}")

    require_ui(
        "navigate the network editor",
        alternative="Without a UI the network can still be read with list_children.",
    )
    network_editor = None
    for pane_tab in hou.ui.paneTabs():
        if pane_tab.type() == hou.paneTabType.NetworkEditor:
            network_editor = pane_tab
            break

    if network_editor is None:
        raise RuntimeError("No Network Editor pane found.")

    network_editor.cd(network_path)

    return {
        "success": True,
        "network_path": network_path,
        "pane_name": network_editor.name(),
    }


###### viewport.find_error_nodes


def find_error_nodes(root_path: str = "/") -> dict:
    """Find all nodes with errors or warnings, recursively from a root path.

    Args:
        root_path: Root node path to start searching from. Defaults to '/'.
    """
    root = hou.node(root_path)
    if root is None:
        raise ValueError(f"Root path not found: {root_path}")

    error_nodes = []
    warning_nodes = []
    # A node whose errors() raises (hou.PermissionError on a locked asset's
    # insides, for one) is neither clean nor erroring: it was not read. It is
    # listed, and the scan carries on, instead of the whole call failing or the
    # node passing as error-free.
    unreadable_nodes = []

    def _unreadable(node, what, exc):
        path = "<unknown>"
        with contextlib.suppress(Exception):
            path = node.path()
        logger.debug("Could not read %s of node '%s': %s", what, path, exc)
        unreadable_nodes.append({"path": path, "error": f"{what}: {type(exc).__name__}: {exc}"})

    def _check_node(node):
        """Recursively check nodes for errors and warnings."""
        try:
            errors = node.errors()
            if errors:
                error_nodes.append(
                    {
                        "path": node.path(),
                        "name": node.name(),
                        "type": node.type().name(),
                        "errors": list(errors),
                    }
                )
        except (hou.Error, AttributeError) as e:
            _unreadable(node, "errors()", e)

        try:
            warnings = node.warnings()
            if warnings:
                warning_nodes.append(
                    {
                        "path": node.path(),
                        "name": node.name(),
                        "type": node.type().name(),
                        "warnings": list(warnings),
                    }
                )
        except (hou.Error, AttributeError) as e:
            _unreadable(node, "warnings()", e)

        # Recurse into children
        try:
            for child in node.children():
                _check_node(child)
        except hou.Error as e:
            _unreadable(node, "children()", e)

    _check_node(root)

    return {
        "error_nodes": error_nodes,
        "warning_nodes": warning_nodes,
        "error_count": len(error_nodes),
        "warning_count": len(warning_nodes),
        "unreadable_nodes": unreadable_nodes,
        "unreadable_count": len(unreadable_nodes),
        "root_path": root_path,
    }


###### Helpers


def _find_scene_viewer(pane_name: str = None):
    """Find a Scene Viewer pane tab by name, or the first one available.

    Args:
        pane_name: Optional specific pane tab name.

    Returns:
        A hou.SceneViewer pane tab.

    Raises:
        RuntimeError: If no Scene Viewer is found.
        ValueError: If the named pane is not a Scene Viewer.
    """
    require_ui(
        "find a Scene Viewer",
        alternative="Geometry can still be inspected with get_geometry_info and sample_geometry.",
    )
    if pane_name is not None:
        pane_tab = _find_pane_by_name(pane_name)
        if pane_tab.type() != hou.paneTabType.SceneViewer:
            raise ValueError(
                f"Pane '{pane_name}' is a {pane_tab.type().name()}, not a Scene Viewer."
            )
        return pane_tab

    for pane_tab in hou.ui.paneTabs():
        if pane_tab.type() == hou.paneTabType.SceneViewer:
            return pane_tab

    raise RuntimeError("No Scene Viewer pane found.")


def _find_pane_by_name(pane_name: str):
    """Find a pane tab by its name.

    Args:
        pane_name: The pane tab name.

    Returns:
        The matching hou.PaneTab.

    Raises:
        ValueError: If no pane with the given name exists.
    """
    require_ui(f"find the pane {pane_name!r}")
    for pane_tab in hou.ui.paneTabs():
        if pane_tab.name() == pane_name:
            return pane_tab

    available = [pt.name() for pt in hou.ui.paneTabs()]
    raise ValueError(f"Pane tab not found: '{pane_name}'. Available panes: {available}")


###### viewport.log_status


def log_status(message: str, severity: str = "message") -> dict:
    """Display a status message in Houdini's status bar.

    Args:
        message: The status message to display.
        severity: Severity level — "message" (default), "important",
            "warning", or "error".
    """
    severity_map = {
        "message": hou.severityType.Message,
        "important": hou.severityType.ImportantMessage,
        "warning": hou.severityType.Warning,
        "error": hou.severityType.Error,
    }
    sev = severity_map.get(severity.lower(), hou.severityType.Message)
    if hou.isUIAvailable():
        hou.ui.setStatusMessage(message, severity=sev)
    else:
        # Headless sessions (hython/hbatch) have no status bar; stay a
        # harmless no-op so instruction-following clients never error.
        print(f"[status] {message}")
    return {"message": message, "severity": severity}


###### Registration

register_handler("viewport.list_panes", list_panes)
register_handler("viewport.get_viewport_info", get_viewport_info)
register_handler("viewport.set_viewport_camera", set_viewport_camera)
register_handler("viewport.set_viewport_display", set_viewport_display)
register_handler("viewport.set_viewport_direction", set_viewport_direction)
register_handler("viewport.set_viewport_renderer", set_viewport_renderer)
register_handler("viewport.frame_selection", frame_selection)
register_handler("viewport.frame_all", frame_all)
register_handler("viewport.capture_screenshot", capture_screenshot)
register_handler("viewport.capture_network_editor", capture_network_editor)
register_handler("viewport.set_current_network", set_current_network)
register_handler("viewport.find_error_nodes", find_error_nodes)
register_handler("viewport.log_status", log_status)


###### viewport.set_viewer_context


def set_viewer_context(
    network_path: str,
    current_node: str = None,
    pane_name: str = None,
) -> dict:
    """Point the Scene Viewer at a network, and optionally a node within it.

    set_current_network moves the network EDITOR. This moves the VIEWER, which is
    a different pane and the one that decides whether a scene graph view exists at
    all. Without it there was no way to enter Solaris, so no way to preview a USD
    stage or set a Hydra delegate: a recorded session burned several
    execute_python calls on exactly this, and even this project's own GUI checks
    had to reach for Python.

    Args:
        network_path: Network for the viewer to display, e.g. "/stage".
        current_node: Optional node inside it to make current, which is what
            selects the stage a Solaris viewport shows.
        pane_name: Optional pane tab name.
    """
    network = hou.node(network_path)
    if network is None:
        raise ValueError(f"Network not found: {network_path}")

    scene_viewer = _find_scene_viewer(pane_name)
    scene_viewer.setPwd(network)

    if current_node is not None:
        node = hou.node(current_node)
        if node is None:
            raise ValueError(f"Node not found: {current_node}")
        scene_viewer.setCurrentNode(node)

    result = {
        "success": True,
        "network_path": scene_viewer.pwd().path(),
        "pane_name": scene_viewer.name(),
        # Whether Hydra delegates apply here, which is the question the caller is
        # usually really asking.
        "is_scene_graph_view": _is_scene_graph_view(scene_viewer),
    }
    with contextlib.suppress(Exception):
        current = scene_viewer.currentNode()
        result["current_node"] = current.path() if current else None
    return result


register_handler("viewport.set_viewer_context", set_viewer_context)
