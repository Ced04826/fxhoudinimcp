"""MCP tools for polygon modeling: cage checks, point edits, comparison, UVs."""

from __future__ import annotations

# Built-in
from typing import Any

# Third-party
from fxhoudinimcp._sdk import Context

# Internal
from fxhoudinimcp.server import _get_bridge, mcp


@mcp.tool()
async def get_mesh_report(
    ctx: Context,
    node_path: str,
    group: str | None = None,
    max_list: int = 20,
    dump_path: str | None = None,
    quality_checks: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
    schema_version: int | None = None,
) -> dict:
    """One-call cage acceptance check for a polygon mesh.

    Reports quads/tris/ngons, pieces, boundary edges and closed loops,
    non-manifold edges, degenerate faces, valence with interior poles, and quads
    folded across either diagonal. Counts always; id lists only when non-empty,
    capped at max_list. Poles are counted per valence; their point ids and
    positions go only to dump_path, which gets the receipt's keys with every
    list whole. NaN/inf positions are refused with the point ids.

    quality_checks adds per-face measurements the counts cannot express:
    "corner_angle" for corners outside the threshold band (a 179-degree corner
    has area, so degenerate=0 says nothing about it), "triangulation" for faces
    that do not ear-clip cleanly and how far from planar they are. A triangle
    or an n-gon is never a defect for its side count alone.

    Args:
        node_path: SOP node path.
        group: Prim group name, else point group name, limiting scope.
        max_list: Cap on ids per list.
        dump_path: JSON file for the full lists.
        quality_checks: Any of ["corner_angle", "triangulation"].
        thresholds: Overrides for min_corner_angle_deg (10), max_corner_angle_deg
            (170), planarity_ratio (0.01, corner offset over face diameter).
        schema_version: Refuse unless the plugin writes this receipt version.
    """
    bridge = _get_bridge(ctx)
    params: dict[str, Any] = {
        "node_path": node_path,
        "max_list": max_list,
    }
    if group is not None:
        params["group"] = group
    if dump_path is not None:
        params["dump_path"] = dump_path
    if quality_checks is not None:
        params["quality_checks"] = quality_checks
    if thresholds is not None:
        params["thresholds"] = thresholds
    if schema_version is not None:
        params["schema_version"] = schema_version
    return await bridge.execute("modeling.get_mesh_report", params)


@mcp.tool()
async def edit_points(
    ctx: Context,
    moves: list[dict[str, Any]],
    after: str | None = None,
    edit_node: str | None = None,
    name: str | None = None,
    expect_points: int | None = None,
    force: bool = False,
) -> dict:
    """Move points of a SOP through an Edit node in one call.

    Entries: {"point": n, "to"|"delta": [x,y,z]} or {"group": g, "delta": ...}.
    Coordinates are SOP-local. The receipt gives read-back positions, not
    requested ones. It refuses, changing nothing, if the input changed.

    Args:
        moves: Point or group moves (above).
        after: SOP to create the Edit below.
        edit_node: Existing Edit SOP to update.
        name: Name for the new Edit (with after).
        expect_points: Refuse unless input has this count.
        force: Rebase onto a changed input.
    """
    bridge = _get_bridge(ctx)
    params: dict[str, Any] = {
        "moves": moves,
        "force": force,
    }
    if after is not None:
        params["after"] = after
    if edit_node is not None:
        params["edit_node"] = edit_node
    if name is not None:
        params["name"] = name
    if expect_points is not None:
        params["expect_points"] = expect_points
    return await bridge.execute("modeling.edit_points", params)


@mcp.tool()
async def compare_geometry(
    ctx: Context,
    a: str,
    b: str,
    tolerance: float = 1e-5,
    max_list: int = 20,
    dump_path: str | None = None,
) -> dict:
    """Compare two SOP meshes: topology, positions, sourcept provenance.

    Topology matches when counts and face point lists agree in order.
    Positions only when point counts match. sourcept on b maps each element
    to a; -1, out of range, or a duplicate source is new. dump_path gets JSON.

    Args:
        a: First SOP path.
        b: Second SOP path.
        tolerance: Position delta in scene units.
        max_list: Cap on mismatch and new-id lists.
        dump_path: JSON file for the full lists.
    """
    bridge = _get_bridge(ctx)
    params: dict[str, Any] = {
        "a": a,
        "b": b,
        "tolerance": tolerance,
        "max_list": max_list,
    }
    if dump_path is not None:
        params["dump_path"] = dump_path
    return await bridge.execute("modeling.compare_geometry", params)


@mcp.tool()
async def compare_surfaces(
    ctx: Context,
    a: str,
    b: str,
    samples: int = 4000,
    seed: int = 0,
    tolerances: list[float] | None = None,
    region_group_a: str | None = None,
    region_group_b: str | None = None,
    space: str = "sop",
    max_list: int = 5,
    dump_path: str | None = None,
    schema_version: int | None = None,
) -> dict:
    """Distance between two polygon surfaces that share no topology.

    Use when compare_geometry cannot: a rebuild against its reference, a retopo
    against the scan, a boolean result against the cage.

    Both sides are triangulated once by Houdini's Divide verb on a detached
    copy, then sampled in proportion to triangle area from a seeded stratified
    sequence; every sample is measured in one native batch (the Ray verb in
    minimum-distance mode) to the closest point on the other surface's
    triangles, never to its nearest vertex and never along a ray. Returns per direction mean/p95/max with the worst sample points,
    plus a coverage row per tolerance. Coverage is one-way containment, not
    similarity and not a verdict; both directions are reported because one
    alone cannot see a missing region. scope_complete goes false when a face
    produced no triangles, and then every number is over the measured surface
    rather than the whole input. Limits: 200000 samples and 500000 primitives
    per side, both counted after any region group.

    Args:
        a: First SOP path.
        b: Second SOP path.
        samples: Samples per side (1..200000); one closest-point query each.
        seed: Sampling seed; the same seed gives the same points.
        tolerances: Up to four distances in scene units. Default: 0.001 of the
            shared bounding box diagonal, reported in the receipt.
        region_group_a: Primitive group scoping the region compared on a.
        region_group_b: Primitive group scoping the region compared on b.
        space: "sop" compares raw SOP positions and assumes one object
            transform (warns when they differ); "world" applies each.
        max_list: Cap on worst-sample entries per direction.
        dump_path: JSON file for every sample position and distance.
        schema_version: Refuse unless the plugin writes this receipt version.
    """
    bridge = _get_bridge(ctx)
    params: dict[str, Any] = {
        "a": a,
        "b": b,
        "samples": samples,
        "seed": seed,
        "space": space,
        "max_list": max_list,
    }
    if tolerances is not None:
        params["tolerances"] = tolerances
    if region_group_a is not None:
        params["region_group_a"] = region_group_a
    if region_group_b is not None:
        params["region_group_b"] = region_group_b
    if dump_path is not None:
        params["dump_path"] = dump_path
    if schema_version is not None:
        params["schema_version"] = schema_version
    return await bridge.execute("modeling.compare_surfaces", params)


@mcp.tool()
async def get_uv_report(
    ctx: Context,
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
) -> dict:
    """UV acceptance check: islands, winding, overlaps, area, distortion.

    Faces are triangulated by Houdini's Divide verb and each corner keeps its
    own UV, so seams survive and a fold stays folded -- picking UV diagonals
    independently of the 3D ones can lay a folded mapping out flat.

    An island is a set of faces joined by edges whose UVs match at both ends.
    An overlap is a pair of UV triangles with intersection area above
    overlap_area_tolerance; triangles sharing only an edge intersect in exactly
    zero area and never appear. allow_stacking forgives only pairs whose three
    corners coincide (deliberately stacked shells) and cannot hide a partial
    overlap; stacked pairs are counted either way. Mirrored winding is
    reported, not judged. uv_area is the sum of triangle areas, not a union.
    Distortion comes as two numbers: density (texel scale against this mesh's
    own mean) and anisotropy (the ratio of the mapping's singular values, which
    catches a shape squashed along one axis at unchanged area); means and p95
    are weighted by world area. Faces with no triangles or non-finite UVs are
    excluded, which sets scope_complete false and overlaps.scope to
    measured_faces_only. Limits: 500000 primitives after the group, 400000 UV
    triangles for the overlap pass; over budget overlaps.status says incomplete
    or skipped, never a clean sheet. fingerprint names the geometry measured
    (points@P digest as edit_points writes it, prims, UV digest).

    Args:
        node_path: SOP node path.
        uv_attribute: Attribute name; vertex is preferred over point.
        group: Primitive group name limiting scope.
        overlap_area_tolerance: UV units squared; default 1e-12.
        allow_stacking: Do not count exactly stacked pairs as overlaps.
        stack_tolerance: UV distance for "the same corner"; default 1e-6.
        texture_resolution: Pixels per tile side; adds pixel-squared areas.
        stretch_threshold: Ratio above 1.0; counts triangles whose density is
            outside [1/t, t] of the mean or whose anisotropy exceeds t.
        max_list: Cap on ids and worst-overlap entries.
        dump_path: JSON file for every island, overlap and flagged prim.
        schema_version: Refuse unless the plugin writes this receipt version.
        check_overlaps: False skips the costly overlap pass; overlaps then says checked: false.
    """
    bridge = _get_bridge(ctx)
    params: dict[str, Any] = {
        "node_path": node_path,
        "uv_attribute": uv_attribute,
        "allow_stacking": allow_stacking,
        "check_overlaps": check_overlaps,
        "max_list": max_list,
    }
    if group is not None:
        params["group"] = group
    if overlap_area_tolerance is not None:
        params["overlap_area_tolerance"] = overlap_area_tolerance
    if stack_tolerance is not None:
        params["stack_tolerance"] = stack_tolerance
    if texture_resolution is not None:
        params["texture_resolution"] = texture_resolution
    if stretch_threshold is not None:
        params["stretch_threshold"] = stretch_threshold
    if dump_path is not None:
        params["dump_path"] = dump_path
    if schema_version is not None:
        params["schema_version"] = schema_version
    return await bridge.execute("modeling.get_uv_report", params)
