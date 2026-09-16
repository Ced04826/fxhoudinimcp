"""MCP tools for Houdini parameter operations.

Exposes 12 tools covering parameter get/set, expressions, channel
references, locking, schema inspection, and spare parameter creation.
"""

from __future__ import annotations

# Built-in
from typing import Any

# Third-party
from fxhoudinimcp._sdk import Context

# Internal
from fxhoudinimcp._types import Value
from fxhoudinimcp.server import _get_bridge, mcp

###### parameters.get_parameter


@mcp.tool()
async def get_parameter(ctx: Context, node_path: str, parm_name: str) -> dict:
    """Get the value and metadata of a parameter.

    Args:
        node_path: Node path.
        parm_name: Parameter name.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "parameters.get_parameter",
        {"node_path": node_path, "parm_name": parm_name},
    )


###### parameters.set_parameter


@mcp.tool()
async def set_parameter(ctx: Context, node_path: str, parm_name: str, value: Value) -> dict:
    """Set a parameter value.

    Args:
        node_path: Node path.
        parm_name: Parameter name.
        value: New value (int, float, string, bool, or list).
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "parameters.set_parameter",
        {"node_path": node_path, "parm_name": parm_name, "value": value},
    )


###### parameters.set_parameters


@mcp.tool()
async def set_parameters(ctx: Context, node_path: str, params: dict[str, Any]) -> dict:
    """Batch-set multiple parameters on a node.

    Args:
        node_path: Node path.
        params: Mapping of parameter names to values.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "parameters.set_parameters",
        {"node_path": node_path, "params": params},
    )


###### parameters.get_parameter_schema


@mcp.tool()
async def get_parameter_schema(
    ctx: Context,
    node_path: str,
    parm_name: str | None = None,
    filter: str | None = None,
) -> dict:
    """Get the template schema for parameter(s) on a node.

    Most nodes have dozens of parameters; many have 100+. Always use
    `parm_name` or `filter` unless you genuinely need the full list.

    Args:
        node_path: Node path.
        parm_name: Exact parameter name for a single-parameter lookup.
        filter: Substring to match against parameter name or label
                (case-insensitive). Use instead of dumping all params.
    """
    bridge = _get_bridge(ctx)
    payload: dict[str, Any] = {"node_path": node_path}
    if parm_name is not None:
        payload["parm_name"] = parm_name
    if filter is not None:
        payload["filter"] = filter
    return await bridge.execute("parameters.get_parameter_schema", payload)


###### parameters.get_parm_references


@mcp.tool()
async def get_parm_references(
    ctx: Context,
    node_path: str,
    parm_name: str | None = None,
    direction: str = "both",
    limit: int = 200,
) -> dict:
    """Who references a parameter, and what it references — in one call.

    `incoming`: for each parameter of the node (or just parm_name), the
    parameters elsewhere whose expressions read it — what breaks if this
    control is renamed. `outgoing`: what this node's expressions read,
    resolved to parameter paths (pure ch() links and richer expressions
    alike). `node_dependents` / `node_references` give the node-level view.

    Args:
        node_path: Node to inspect.
        parm_name: One parameter instead of all of them.
        direction: "both", "incoming" or "outgoing".
        limit: Cap on reported entries.
    """
    bridge = _get_bridge(ctx)
    payload: dict[str, Any] = {"node_path": node_path, "direction": direction, "limit": limit}
    if parm_name is not None:
        payload["parm_name"] = parm_name
    return await bridge.execute("parameters.get_parm_references", payload)


###### parameters.get_parm_template_tree


@mcp.tool()
async def get_parm_template_tree(
    ctx: Context,
    node_path: str | None = None,
    type_name: str | None = None,
    context: str = "Sop",
    folder: str | list[str] | None = None,
    max_entries: int = 400,
) -> dict:
    """The whole parameter interface as a tree, the way Type Properties shows
    it: folders (with folder_type — tabs, collapsible, multiparm), every
    parameter in order with defaults, default expressions, ranges, menu
    items, Hide/Disable When conditionals, callbacks, naming scheme.

    get_hda_info shows only the top folders and get_parameter_schema
    flattens the structure away; read this before editing an interface.
    Give node_path for a node (its instance interface, spares included) or
    type_name + context for a type.

    Args:
        node_path: Node whose interface to read.
        type_name: Node type instead (with context).
        context: Category of type_name — "Sop", "Object", "Lop", ...
        folder: Narrow to one folder by label, or a list of nested labels.
        max_entries: Cap on entries (depth-first); the reply says when it cut.
    """
    bridge = _get_bridge(ctx)
    payload: dict[str, Any] = {"context": context, "max_entries": max_entries}
    if node_path is not None:
        payload["node_path"] = node_path
    if type_name is not None:
        payload["type_name"] = type_name
    if folder is not None:
        payload["folder"] = folder
    return await bridge.execute("parameters.get_parm_template_tree", payload)


###### parameters.set_expression


@mcp.tool()
async def set_expression(
    ctx: Context,
    node_path: str,
    parm_name: str,
    expression: str,
    language: str = "hscript",
) -> dict:
    """Set an expression on a parameter.

    Args:
        node_path: Node path.
        parm_name: Parameter name.
        expression: Expression string.
        language: "hscript" (default) or "python".
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "parameters.set_expression",
        {
            "node_path": node_path,
            "parm_name": parm_name,
            "expression": expression,
            "language": language,
        },
    )


###### parameters.get_expression


@mcp.tool()
async def get_expression(ctx: Context, node_path: str, parm_name: str) -> dict:
    """Get the expression on a parameter.

    Args:
        node_path: Node path.
        parm_name: Parameter name.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "parameters.get_expression",
        {"node_path": node_path, "parm_name": parm_name},
    )


###### parameters.revert_parameter


@mcp.tool()
async def revert_parameter(ctx: Context, node_path: str, parm_name: str) -> dict:
    """Revert a parameter to its default value.

    Args:
        node_path: Node path.
        parm_name: Parameter name.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "parameters.revert_parameter",
        {"node_path": node_path, "parm_name": parm_name},
    )


###### parameters.link_parameters


@mcp.tool()
async def link_parameters(
    ctx: Context,
    source_path: str,
    source_parm: str,
    dest_path: str,
    dest_parm: str,
) -> dict:
    """Create a channel reference from one parameter to another.

    The destination gets an HScript expression that reads the source as its
    own type: chs() for a String parameter, ch() for numbers, toggles and
    menus. The path is relative to the destination node (chs("../CTRL/mat")),
    so the link survives moving the pair, collapsing into a subnet or
    instancing an HDA. The reply carries the expression, the function used
    and the destination's evaluated value.

    Args:
        source_path: Source node path.
        source_parm: Source parameter name.
        dest_path: Destination node path.
        dest_parm: Destination parameter name.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "parameters.link_parameters",
        {
            "source_path": source_path,
            "source_parm": source_parm,
            "dest_path": dest_path,
            "dest_parm": dest_parm,
        },
    )


###### parameters.lock_parameter


@mcp.tool()
async def lock_parameter(ctx: Context, node_path: str, parm_name: str, locked: bool) -> dict:
    """Lock or unlock a parameter.

    Args:
        node_path: Node path.
        parm_name: Parameter name.
        locked: True to lock, False to unlock.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "parameters.lock_parameter",
        {"node_path": node_path, "parm_name": parm_name, "locked": locked},
    )


###### parameters.create_spare_parameter


@mcp.tool()
async def create_spare_parameter(
    ctx: Context,
    node_path: str,
    parm_name: str,
    parm_type: str,
    label: str,
    default_value: Value | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> dict:
    """Add a spare parameter to a node.

    Args:
        node_path: Node path.
        parm_name: Internal parameter name.
        parm_type: "float", "int", "string", "toggle", or "menu".
        label: UI label.
        default_value: Default value.
        min_val: Minimum value (float/int only).
        max_val: Maximum value (float/int only).
    """
    bridge = _get_bridge(ctx)
    payload: dict[str, Any] = {
        "node_path": node_path,
        "parm_name": parm_name,
        "parm_type": parm_type,
        "label": label,
    }
    if default_value is not None:
        payload["default_value"] = default_value
    if min_val is not None:
        payload["min_val"] = min_val
    if max_val is not None:
        payload["max_val"] = max_val
    return await bridge.execute("parameters.create_spare_parameter", payload)


@mcp.tool()
async def create_spare_parameters(
    ctx: Context,
    node_path: str,
    parameters: list[dict[str, Any]],
    folder_name: str | None = None,
    folder_type: str = "Tabs",
) -> dict:
    """Batch-create multiple spare parameters in one call, optionally in a folder tab.

    Args:
        node_path: Node path.
        parameters: List of parameter specs. Each dict has keys:
            parm_name (str), parm_type (str: "float"/"int"/"string"/"toggle"/"menu"),
            label (str), default_value (optional), min_val (optional), max_val (optional).
        folder_name: If provided, wraps all parameters in a named folder tab.
        folder_type: Folder style: "Tabs", "Collapsible", or "Simple".
    """
    bridge = _get_bridge(ctx)
    payload: dict[str, Any] = {
        "node_path": node_path,
        "parameters": parameters,
    }
    if folder_name is not None:
        payload["folder_name"] = folder_name
        payload["folder_type"] = folder_type
    return await bridge.execute("parameters.create_spare_parameters", payload)


@mcp.tool()
async def get_parameters(
    ctx: Context,
    node_path: str,
    patterns: list[str] | None = None,
    include_defaults: bool = False,
) -> dict:
    """Read many parameter values at once, matched by name or label substring.

    The batch counterpart of set_parameters. Several unrelated groups of
    settings ("flame", "wind", "buoy") come back in one call instead of one call
    each, and unlike get_node_card these are the live values on this node rather
    than the defaults for its type.

    Args:
        node_path: Node to read.
        patterns: Substrings matched against parameter name and label. Omit for
            everything, up to the cap.
        include_defaults: Also report whether each value is still the default.
    """
    bridge = _get_bridge(ctx)
    params: dict[str, Any] = {
        "node_path": node_path,
        "include_defaults": include_defaults,
    }
    if patterns is not None:
        params["patterns"] = patterns
    return await bridge.execute("parameters.get_parameters", params)
