"""The gateway to the tools a toolset leaves unregistered.

Under FXHOUDINIMCP_TOOLSET=modeling the MCP client is shown 50 tools; the
other 165 would otherwise need a config change and a restart. The Unreal MCP
solves the same problem with three gateway tools over its toolsets, and this
is the same shape, so an agent learns one pattern for both:

    list_toolsets            what else exists, by category, names only
    describe_tools           the parameters and docstring of the ones wanted
    call_tool(name, args)    run it

call_tool runs the very function the tool would have been registered with,
so the answer is the tool's own receipt. The arguments are checked against
the function's signature and annotations first, the way the client's schema
would have, and a wrong one is named. Registered only when tools were left
out: under "full" there is nothing behind the gateway.
"""

from __future__ import annotations

# Built-in
import inspect
from difflib import get_close_matches
from typing import Any

# Third-party
from pydantic import TypeAdapter, ValidationError

# Internal
from fxhoudinimcp import toolsets
from fxhoudinimcp._sdk import Context

# One line per category: the module docstrings say "MCP tool wrappers for
# ...", which tells an agent nothing about when to look there.
ABOUT = {
    "animation": "keyframes, expressions on channels, playbar and frame range",
    "cache": "file caches: list, write, clear, status",
    "chops": "CHOP networks, channels, exporting channels to parameters",
    "code": "Python, HScript, expressions, environment, plugin reload",
    "context": "scene summaries, cook chains, selection, node explanations, snapshots",
    "cook": "cooking nodes on demand",
    "cops": "Copernicus (COP) networks, layers, flags",
    "dops": "simulations: DOP objects, fields, relationships, stepping (setup_* is in workflows)",
    "geometry": "points, prims, groups, attributes, volumes, sampling",
    "graph": "build and verify networks, node cards, cook profiling, frame ranges",
    "hda": "digital assets: create, install, interface, sections, versions",
    "help": "SideFX documentation and workflow guides",
    "lops": "Solaris / USD: stages, prims, layers, variants, materials, lights",
    "materials": "materials and shader networks, assignment",
    "modeling": "mesh reports, point edits, surface and topology comparison, UVs",
    "nodes": "create, wire, move, colour, flag, box and annotate nodes",
    "parameters": "read and write parameters, expressions, spare parameters, links",
    "rendering": "render settings, ROPs, Karma and husk renders, viewport renders",
    "scene": "hip files: new, load, save, import, export, undo",
    "shelf": "shelf tools",
    "takes": "takes: list, create, switch",
    "tops": "PDG / TOP networks: cook, work items, schedulers, logs",
    "vex": "wrangles and VEX: create, edit, validate",
    "viewport": "panes, viewport camera and display, screenshots, error nodes",
    "viewport_capture": "framed captures of the Scene Viewer",
    "workflows": "composite setups: sims, renders, SOP chains",
}

# Docstring characters a summary keeps.
_SUMMARY_CHARS = 300


def _hidden() -> dict[str, str]:
    """Tool name -> category of the tools the toolset left out."""
    return dict(toolsets.SKIPPED)


def _function(name: str):
    function = toolsets.FUNCTIONS.get(name)
    if function is None:
        close = get_close_matches(name, sorted(toolsets.FUNCTIONS), n=3, cutoff=0.5)
        hint = f" Did you mean: {close}?" if close else ""
        raise ValueError(f"no tool named {name!r}.{hint} list_toolsets names them all.")
    return function


def _parameters(function) -> dict[str, inspect.Parameter]:
    signature = inspect.signature(function, eval_str=True)
    return {n: p for n, p in signature.parameters.items() if n != "ctx"}


def _describe(name: str, full: bool) -> dict[str, Any]:
    function = _function(name)
    doc = inspect.cleandoc(function.__doc__ or "")
    if not full:
        doc = doc.split("\n\n", 1)[0]
        if len(doc) > _SUMMARY_CHARS:
            doc = doc[:_SUMMARY_CHARS].rstrip() + " ..."
    parameters = {}
    for pname, parameter in _parameters(function).items():
        entry: dict[str, Any] = {"type": inspect.formatannotation(parameter.annotation)}
        if parameter.default is inspect.Parameter.empty:
            entry["required"] = True
        else:
            entry["default"] = parameter.default
        parameters[pname] = entry
    category = toolsets.SKIPPED.get(name) or toolsets.REGISTERED.get(name)
    return {"name": name, "toolset": category, "description": doc, "parameters": parameters}


async def list_toolsets(ctx: Context) -> dict:
    """List the Houdini tools that are not registered directly, by category.

    This server shows the everyday modeling tools directly; the rest (DOPs,
    USD/LOPs, rendering, TOPs, CHOPs, HDAs, COPs, materials, animation, ...)
    are reached through describe_tools and call_tool, with the same receipts.
    A tool missing from your tool list is here, not absent.
    """
    groups: dict[str, dict[str, Any]] = {}
    for name, category in sorted(_hidden().items()):
        group = groups.setdefault(category, {"about": ABOUT.get(category, ""), "tools": []})
        group["tools"].append(name)
    return {
        "registered_directly": len(toolsets.REGISTERED),
        "via_call_tool": len(toolsets.SKIPPED),
        "toolsets": groups,
        "next": "describe_tools(names=[...]) for parameters, then call_tool(name, arguments)",
    }


async def describe_tools(
    ctx: Context,
    names: list[str] | None = None,
    toolset: str | None = None,
) -> dict:
    """Parameters and description of tools, before calling them via call_tool.

    Args:
        names: Tool names; each comes with its full description.
        toolset: A category from list_toolsets; every tool in it comes with a
            one-paragraph summary. Ask again by names for the full text.
    """
    if not names and not toolset:
        raise ValueError("give names=[...] or toolset=<category from list_toolsets>")
    tools = [_describe(name, full=True) for name in names or []]
    if toolset:
        members = sorted(
            name
            for name, category in {**toolsets.SKIPPED, **toolsets.REGISTERED}.items()
            if category == toolset and name not in toolsets.GATEWAY
        )
        if not members:
            known = sorted(set(toolsets.SKIPPED.values()) | set(toolsets.REGISTERED.values()))
            raise ValueError(f"no toolset {toolset!r}; toolsets: {known}")
        tools += [_describe(name, full=False) for name in members if name not in (names or [])]
    return {"tools": tools}


async def call_tool(ctx: Context, name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Run a Houdini tool by name, registered or not, and return its own receipt.

    The arguments are checked against the tool's parameters first: an unknown,
    missing or mistyped one is named before anything reaches Houdini.

    Args:
        name: Tool name, e.g. "setup_pyro_sim" (see list_toolsets).
        arguments: The tool's arguments as a JSON object; see describe_tools.
    """
    if name in toolsets.GATEWAY:
        raise ValueError(f"{name} is a gateway tool; call it directly")
    function = _function(name)
    parameters = _parameters(function)
    given = dict(arguments or {})
    unknown = sorted(set(given) - set(parameters))
    if unknown:
        close = sorted({c for k in unknown for c in get_close_matches(k, list(parameters), 2, 0.5)})
        hint = f" Did you mean: {close}?" if close else ""
        raise ValueError(
            f"{name}: unknown argument(s) {unknown}.{hint} Parameters: {sorted(parameters)}"
        )
    missing = sorted(
        pname
        for pname, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty and pname not in given
    )
    if missing:
        raise ValueError(f"{name}: missing required argument(s) {missing}")
    checked = {}
    for pname, value in given.items():
        annotation = parameters[pname].annotation
        if annotation is inspect.Parameter.empty:
            checked[pname] = value
            continue
        try:
            checked[pname] = TypeAdapter(annotation).validate_python(value)
        except ValidationError as exc:
            problem = exc.errors()[0].get("msg", str(exc))
            raise ValueError(
                f"{name}: argument {pname!r} must be {inspect.formatannotation(annotation)}, "
                f"got {value!r} ({problem})"
            ) from exc
    return await function(ctx, **checked)


def register() -> None:
    """Register the three gateway tools when the toolset left any tool out."""
    if not toolsets.SKIPPED:
        return
    from fxhoudinimcp.server import mcp

    for function in (list_toolsets, describe_tools, call_tool):
        mcp.tool()(function)
