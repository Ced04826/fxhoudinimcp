"""Which tools the server registers: FXHOUDINIMCP_TOOLSET=full|modeling.

Every registered tool's name is listed to every session and every subagent,
over 200 of them in about 7k characters, while a modeling job uses a few dozen:
two batches of ThreeWheels auto tasks (47 sessions, 1,839 calls) used 28
tools, six of them 94% of the calls, and no DOP, USD, render, TOP, CHOP or
HDA tool at all. "modeling" registers the tools that history actually called
plus the few that answer the same questions (attributes, groups, sections,
docs); "full" registers everything. The script client (fxhoudinimcp.client)
always sees every tool: the switch only decides what an MCP client is shown.

The rest are not gone: under a toolset three gateway tools are registered as
well (see gateway.py), list_toolsets, describe_tools and call_tool, the way
the Unreal MCP exposes its toolsets, so a DOP or USD tool is two calls away
with no restart and no config change. The instruction header says so.
"""

from __future__ import annotations

# Built-in
import logging
import os

logger = logging.getLogger(__name__)

ENV = "FXHOUDINIMCP_TOOLSET"

# ThreeWheels (131 session files, 5,339 calls) and the other Houdini sessions:
# every tool called at least 4 times there or 3 times elsewhere, less the
# viewport-framing tools capture_viewport replaces (frame_all, render_viewport,
# set_viewport_display, ...), plus the modeling family's unused members.
MODELING = frozenset(
    {
        # scripting and scene
        "execute_python",
        "get_houdini_connection_status",
        "get_scene_info",
        "save_scene",
        "export_file",
        # looking
        "capture_viewport",
        "capture_screenshot",
        "capture_network_editor",
        "set_viewport_direction",
        # building and editing networks
        "build_network",
        "verify_network",
        "create_node",
        "delete_node",
        "rename_node",
        "copy_node",
        "connect_nodes",
        "connect_nodes_batch",
        "disconnect_node",
        "set_node_flags",
        "set_node_position",
        "set_node_color",
        "create_network_box",
        "set_parameter",
        "set_parameters",
        "set_expression",
        "validate_vex",
        # reading nodes and networks
        "get_node_card",
        "get_node_info",
        "get_parameters",
        "list_children",
        "find_nodes",
        "list_node_types",
        "get_network_overview",
        "find_error_nodes",
        "get_node_errors_detailed",
        # geometry and modeling
        "get_mesh_report",
        "get_uv_report",
        "compare_surfaces",
        "compare_geometry",
        "section_geometry",
        "edit_points",
        "get_geometry_info",
        "get_bounding_box",
        "get_attrib_values",
        "get_attrib_stats",
        "get_attribute_info",
        "get_groups",
        # docs
        "search_help",
        "get_help_page",
        "get_workflow_guide",
    }
)

TOOLSETS: dict[str, frozenset[str] | None] = {"full": None, "modeling": MODELING}

# Registered under any toolset that leaves tools out; see gateway.py.
GATEWAY = frozenset({"list_toolsets", "describe_tools", "call_tool"})

# Filled by fxhoudinimcp.server as the tool modules register: tool name ->
# the tool module's short name (the category), and tool name -> function for
# every tool, registered or not, which is what call_tool runs.
REGISTERED: dict[str, str] = {}
SKIPPED: dict[str, str] = {}
FUNCTIONS: dict[str, object] = {}

_warned: set[str] = set()


def active() -> str:
    """The toolset named by FXHOUDINIMCP_TOOLSET; "full" when unset or unknown."""
    raw = os.getenv(ENV, "").strip().lower()
    if not raw:
        return "full"
    if raw not in TOOLSETS:
        if raw not in _warned:
            _warned.add(raw)
            logger.warning(
                "%s=%r is not one of %s; registering every tool", ENV, raw, sorted(TOOLSETS)
            )
        return "full"
    return raw


def allows(tool_name: str, toolset: str) -> bool:
    members = TOOLSETS.get(toolset)
    return members is None or tool_name in members or tool_name in GATEWAY


def summary(registered: dict[str, str], skipped: dict[str, str]) -> str:
    """The instruction header's count: "215 tools across 26 categories", or
    "53 tools, 165 more via list_toolsets" when a toolset left some out."""
    if skipped:
        return f"{len(registered)} tools, {len(skipped)} more via list_toolsets"
    categories = len(set(registered.values()))
    return f"{len(registered)} tools across {categories} categories"


def current_summary() -> str:
    return summary(REGISTERED, SKIPPED)
