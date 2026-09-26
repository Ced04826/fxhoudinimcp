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
    follow_targets: bool = True,
    show_target: bool = False,
    isolate: bool | list[str] = True,
    region: dict[str, Any] | None = None,
    clip: bool = False,
    compare: str | None = None,
    overlays: list[str] | None = None,
    zebra_direction: list[float] | None = None,
    zebra_stripes: int = 16,
    orbit: bool | dict[str, Any] | None = None,
    sheet: bool | None = None,
    sheet_columns: int | None = None,
    sheet_max: int = 2000,
    min_edge_px: float = 6.0,
    replay: str | None = None,
    lighting: str | None = None,
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

    targets decide what is drawn; bbox, when given, decides the framing on
    its own, so a close-up of part of a target inside a subnet is one call
    (targets + bbox). The viewer draws the display node of the network it is
    in: the capture points it at the SOP targets' network (follow_targets),
    and a target that is not that network's display node (or targets in
    several networks) is drawn through a temporary proxy object at /obj --
    an Object Merge of the targets, removed after the shot -- so no display
    flag anywhere moves. show_target moves the flag onto the target instead.
    `drawn` says how (`via`: viewer or proxy) and `targets_drawn` whether
    each target is in the image. Network, flags and camera are put back
    afterwards.

    At /obj every displayed object is drawn, and inside a SOP network the
    other objects are ghosted over the one being edited. isolate draws only
    the targets' objects, through the flipbook's own object mask: nothing in
    the scene or the viewer changes.

    Reading wiring: look at an orbit sheet (orbit=True, 16 cells) to find
    suspect places, then shoot close-ups there: region + clip, the "facing"
    direction, overlays; compare against the source for missing features.
    Every call writes <prefix>_shots.json; replay it after a repair to shoot
    the same views. Several views are also laid out on <prefix>_sheet.png
    (long side sheet_max), each cell labelled with its view name, azimuth
    and elevation, and edge_px says how long an edge is on screen: below
    min_edge_px a warning says the wiring cannot be read there.

    Surfaces are lit from the camera by default (lighting="headlight"): a
    face turned to the camera is bright from any side, a face seen edge-on
    darker but never so dark that the wire lines vanish, so the wiring of an
    underside reads as well as the top. Every image has the viewport's
    background, alone or on a sheet.

    Look at the images with your file reader; nothing is inlined.

    Args:
        output_dir: Folder for the PNGs, named <prefix>_<view name>.png.
        views: Each a direction string, or a dict with "direction" or
            "azimuth"/"elevation" (degrees; azimuth 0 looks from +Z, 90 from
            +X; elevation 90 from above), optional "projection" ("ortho" or
            "persp"; custom angles default to persp), "name", and per-view
            "targets"/"bbox", each replacing the shared one on its own (a
            per-view bbox keeps the shared targets; "targets": [] drops
            them; the same for "region" and "clip"). Directions: "front",
            "back", "left", "right", "top", "bottom" (orthographic), "persp"
            (the perspective view's current angle), "current" (the view as
            it is), "facing" (straight at the side of the region most of its
            area faces -- both sides of a thin part do not cancel; needs a
            region; the receipt gives the normal). Default ["current"].
            Every view reports the azimuth/elevation it was shot from.
        targets: SOP or object node paths to draw; framed by the union of
            their bounding boxes unless bbox is given. Without targets or
            bbox, "current" is captured as is and other directions frame
            everything.
        bbox: World-space box to frame, [xmin, ymin, zmin, xmax, ymax, zmax].
            With targets: frames this box only, the targets are what is
            drawn.
        shading: Shading for the capture, e.g. "smooth_wire" to see the
            wiring, "wire", "smooth", "flat_wire", "hidden_line". Applied to
            the edited object, other objects and selected objects, and
            restored afterwards. None keeps the viewer's shading.
        max_size: Long side of each image in pixels, 256-4096.
        margin: Empty border on each side, as a fraction of the image.
        prefix: File name prefix.
        pane_name: Scene Viewer pane tab; default the first one.
        restore_view: Put the viewer back as it was (default True).
        follow_targets: Point the viewer at the SOP targets' network for the
            capture (default True).
        show_target: Move the display flag onto a SOP target that is not its
            network's display node, for the capture only (default False).
        isolate: True (default) draws only the objects that hold the node
            targets; without node targets nothing is hidden. False draws
            everything the viewer shows, for context. A list of node paths
            draws those objects (a SOP path means its object), e.g. the body
            and a wheel together.
        region: Where to shoot, instead of bbox: {"center": [x,y,z],
            "radius": r}, {"bbox": [...]}, {"group": name} (primitive group,
            else point group) or {"prims": [numbers]}; group and prims read
            the first SOP target, or "node". World space. Frames the region.
        clip: Draw only the faces in the region (its sphere, else its box; a
            face stays when one of its points or its centroid is inside), so
            the near side of the part does not hide the far side. Done in the proxy object; your nodes are
            untouched.
        compare: A source SOP shot with exactly the same camera; each view
            also gets <name>_compare.png, SOURCE left and RESULT right.
        overlays: Any of "triangles" (blue), "ngons" (magenta), "poles"
            (interior points with other than 4 edges: 3 cyan, 5+ orange),
            "stretch" (face aspect, grey to red at 4:1), "zebra" (bands of
            the angle between the normal and zebra_direction, per pixel, from
            the mesh's own interpolated normals; the geometry is not
            subdivided or changed). A face square to the direction is white,
            so flat faces are one colour and a bump shows as rings; the
            bands stay on the surface from every view. Pass
            shading="smooth_wire" to see the edges over the bands. Face
            colours and pole markers are on the proxy object only; the zebra
            is a temporary MaterialX material on it, with material display
            on for the call. Default none.
        zebra_direction: World direction the zebra angle is measured from,
            default [0,1,0].
        zebra_stripes: Bands over 0-180 degrees, default 16 (each band
            11.25 degrees, so a tilt of about 3 degrees off a flat face
            turns it dark).
        orbit: True for rings at elevation 30 and -20 times 8 azimuths (16
            views), or {"elevations": [...], "azimuths": count or [...],
            "projection": ...}; added after views.
        sheet: Lay all views out on one PNG; default when there are several.
            Each cell is cropped to the drawn target plus a small border.
        sheet_columns: Columns of the sheet; default about square.
        sheet_max: Long side of the sheet in pixels, default 2000 (what Read
            shows without shrinking).
        min_edge_px: Mean on-screen edge length below which a view warns
            that the wiring cannot be read, default 6.
        replay: Path of a <prefix>_shots.json: shoots its views again with
            the same cameras, targets, clip, overlays and compare; views,
            orbit, bbox and region are not taken.
        lighting: "headlight" (default; a replay keeps the recorded one)
            draws every surface with a matcap lit from the camera and
            material display off, for the capture only. "viewport" keeps
            the viewer's own lights, materials and default material.

    Returns per view: path, pixels, isolated (the objects drawn alone, or
    null), target_rect (image pixels, origin top
    left), target_in_frame, target_fill, drawn_fraction (share of the target
    region actually drawn on), framed ("bbox" or the target paths),
    frame_bbox (the world box fitted), drawn (via, network, display_node,
    targets_drawn, and proxy with proxy_points/target_points),
    projection, looking_along, pivot, and ortho_width or distance with
    fov_x_deg, azimuth/elevation, edge_px (mean/median/p10 on-screen edge
    length) and sheet_edge_px, region, clip (with drawn.clipped_prims),
    compare (source image, pair image), content_rect (image pixels of the
    drawn target) and sheet_crop. Also shots_file, sheet (path, pixels,
    columns, rows), overlays (legend), lighting (mode, matcap, restored)
    and warnings. success is false when an image is missing or blank where the
    target is, a target is not fully in frame or not drawn, or the viewer
    could not be restored; problems says which.
    """
    bridge = _get_bridge(ctx)
    params: dict[str, Any] = {
        "output_dir": output_dir,
        "max_size": max_size,
        "margin": margin,
        "prefix": prefix,
        "restore_view": restore_view,
        "follow_targets": follow_targets,
        "show_target": show_target,
        "isolate": isolate,
        "clip": clip,
        "zebra_stripes": zebra_stripes,
        "sheet_max": sheet_max,
        "min_edge_px": min_edge_px,
    }
    for key, value in (
        ("views", views),
        ("targets", targets),
        ("bbox", bbox),
        ("shading", shading),
        ("pane_name", pane_name),
        ("region", region),
        ("compare", compare),
        ("overlays", overlays),
        ("zebra_direction", zebra_direction),
        ("orbit", orbit),
        ("sheet", sheet),
        ("sheet_columns", sheet_columns),
        ("replay", replay),
        ("lighting", lighting),
    ):
        if value is not None:
            params[key] = value
    return await bridge.execute("viewport.capture_viewport", params)
