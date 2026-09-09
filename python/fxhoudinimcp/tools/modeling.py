"""MCP tools for polygon modeling checks."""

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
