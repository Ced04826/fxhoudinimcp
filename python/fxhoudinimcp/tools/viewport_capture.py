"""MCP tool for framed captures of what the Scene Viewer shows."""

from __future__ import annotations

# Built-in
from typing import Any

# Third-party
from fxhoudinimcp._sdk import Context

# Internal
from fxhoudinimcp.server import _get_bridge, mcp


@mcp.tool()
async def capture_viewport(
    ctx: Context,
    output_dir: str,
    views: list[str | dict[str, Any]] | None = None,
    targets: list[str] | None = None,
    bbox: list[float] | None = None,
    shading: str | None = None,
    max_size: int = 1600,
    margin: float = 0.05,
    prefix: str = "view",
    pane_name: str | None = None,
    restore_view: bool = True,
) -> dict:
    """Capture what the Scene Viewer shows, from several views in one call,
    each framed so the target is whole and fills the image.

    This is the viewport itself (a flipbook: shading, wireframe, display
    flags, templates as the viewer draws them), not a render, and it never
    looks through a camera node: a camera's resolution and aspect would
    decide what gets cut off. If the viewer is looking through one, the
    capture uses the viewport's own camera with the camera's viewpoint and
    lens, then puts the camera back.

    The image takes the target's aspect: a tall part gets a tall image, a
    long one a wide image (clamped to 8:1), long side max_size pixels.
    Framing is solved from the camera parameters, then checked against the
    written pixels. The viewer's view type, cameras and shading are restored
    afterwards and read back (`restored`).

    Look at the images with your file reader; nothing is inlined.

    Args:
        output_dir: Folder for the PNGs, named <prefix>_<view name>.png.
        views: Each a direction string, or a dict with "direction" or
            "azimuth"/"elevation" (degrees; azimuth 0 looks from +Z, 90 from
            +X; elevation 90 from above), optional "projection" ("ortho" or
            "persp"; custom angles default to persp), "name", and per-view
            "targets"/"bbox". Directions: "front", "back", "left", "right",
            "top", "bottom" (orthographic), "persp" (the perspective view's
            current angle), "current" (the view as it is). Default
            ["current"].
        targets: SOP or object node paths to frame; the union of their
            bounding boxes. Without targets or bbox, "current" is captured
            as is and other directions frame everything.
        bbox: World-space box to frame, [xmin, ymin, zmin, xmax, ymax, zmax].
        shading: Shading for the capture, e.g. "smooth_wire" to see the
            wiring, "wire", "smooth", "flat_wire", "hidden_line". Applied to
            the edited object, other objects and selected objects, and
            restored afterwards. None keeps the viewer's shading.
        max_size: Long side of each image in pixels, 256-4096.
        margin: Empty border on each side, as a fraction of the image.
        prefix: File name prefix.
        pane_name: Scene Viewer pane tab; default the first one.
        restore_view: Put the viewer back as it was (default True).

    Returns per view: path, pixels, target_rect (image pixels, origin top
    left), target_in_frame, target_fill, drawn_fraction (share of the target
    region actually drawn on), projection, looking_along, pivot, and
    ortho_width or distance with fov_x_deg. success is false when an image
    is missing or blank where the target is, a target is not fully in frame,
    or the viewer could not be restored; problems says which.
    """
    bridge = _get_bridge(ctx)
    params: dict[str, Any] = {
        "output_dir": output_dir,
        "max_size": max_size,
        "margin": margin,
        "prefix": prefix,
        "restore_view": restore_view,
    }
    for key, value in (
        ("views", views),
        ("targets", targets),
        ("bbox", bbox),
        ("shading", shading),
        ("pane_name", pane_name),
    ):
        if value is not None:
            params[key] = value
    return await bridge.execute("viewport.capture_viewport", params)
