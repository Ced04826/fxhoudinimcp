"""MCP tools for polygon modeling checks, comparison, reload proof, and views."""

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
) -> dict:
    """One-call cage acceptance check for a polygon mesh.

    Reports quads/tris/ngons, pieces, boundary edges and closed loops,
    non-manifold edges, degenerate faces, valence with interior poles, and quads
    folded across either diagonal. Counts always; id lists only when non-empty,
    capped at max_list. dump_path receives the full lists as JSON.

    Args:
        node_path: SOP node path.
        group: Prim group name, else point group name, limiting scope.
        max_list: Cap on ids per list.
        dump_path: JSON file for the full lists.
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
