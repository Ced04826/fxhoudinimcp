"""MCP tool for cooking nodes on demand."""

from __future__ import annotations

# Built-in
from typing import Any

# Third-party
from fxhoudinimcp._sdk import Context

# Internal
from fxhoudinimcp.server import _get_bridge, mcp


@mcp.tool()
async def cook_node(ctx: Context, node_paths: list[str], force: bool = True) -> dict:
    """Cook nodes now (forced by default) and report what each cook produced.

    For a node that must be recomputed before it is read: a Python SOP whose
    code file changed on disk (no parameter changed, so Houdini keeps the
    cached result), or anything whose inputs changed outside Houdini.

    Per node: cooked, cook_ms, the first errors and warnings with their
    counts, and points/prims for SOPs. success is false when a node is
    missing, fails to cook or reports errors, and when Houdini's update mode
    keeps nodes from cooking (warning says so).

    Args:
        node_paths: Node paths to cook, in order.
        force: Cook even if Houdini considers the node up to date (default).
    """
    bridge = _get_bridge(ctx)
    params: dict[str, Any] = {"node_paths": node_paths, "force": force}
    return await bridge.execute("cook.cook_node", params)
