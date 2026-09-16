"""MCP tools for code execution inside Houdini.

Exposes 4 tools for running Python code, HScript commands, evaluating
expressions, and reading Houdini environment variables.
"""

from __future__ import annotations

# Built-in
from typing import Any

# Third-party
from fxhoudinimcp._sdk import Context

# Internal
from fxhoudinimcp.server import _get_bridge, mcp

###### code.execute_python


@mcp.tool()
async def execute_python(
    ctx: Context,
    code: str,
    justification: str,
    return_expression: str | None = None,
    dump_path: str | None = None,
    max_stdout_chars: int = 102400,
    max_return_chars: int = 8192,
    return_format: str = "auto",
) -> dict:
    """Execute arbitrary Python code inside Houdini. LAST RESORT only.

    DO NOT use this to:
    - Create nodes or networks → use build_network or create_node
    - Set parameters → use set_parameter or set_parameters
    - Create wrangles or write Python SOPs → use create_wrangle
    - Connect nodes → use connect_nodes or connect_nodes_batch
    - Read geometry → use get_geometry_info, get_points, sample_geometry

    ONLY use this when no dedicated tool exists for the operation — i.e.
    hou.* API calls or Python-level state that no other tool exposes.
    The justification parameter is mandatory: name the dedicated tools
    you considered and why none covers this operation.

    The receipt is bounded and says so. `execution_success` is whether the
    code ran; `output_complete` is whether what you are looking at is all
    of it. When output_complete is false, the elided text is not recoverable
    by reading harder — re-run with dump_path (or a bigger cap) if you need
    it, and note that re-running runs the side effects again.

    Printing a large result and then reading it back is the expensive way to
    move data: pass dump_path and read the file.

    Args:
        code: Python source code to execute.
        justification: Which dedicated tools you considered and why none
            covers this operation.
        return_expression: Python expression to evaluate after execution.
            Its printed output is captured too.
        dump_path: Write the record — full stdout, full stderr, the
            serialisable return value and both full tracebacks — to this JSON
            file. `dump_streams_complete` says the printed output is all there;
            `dump_return_lossless` says whether the return value in it is the
            value or an altered copy (the dump holds the same JSON-safe copy
            the receipt does). `full_output_in_dump` is true only when both are.
        max_stdout_chars: Cap on stdout/stderr in the receipt (default 102400).
        max_return_chars: Cap on the returned value in the receipt (default
            8192); past it you get length, hash and samples instead.
        return_format: "auto" (default) stringifies what JSON cannot hold and
            lists what it stringified; "json" refuses instead; "repr" returns
            repr() of the value; "none" reports only its type.
    """
    bridge = _get_bridge(ctx)
    payload: dict[str, Any] = {
        "code": code,
        "max_stdout_chars": max_stdout_chars,
        "max_return_chars": max_return_chars,
        "return_format": return_format,
    }
    if return_expression is not None:
        payload["return_expression"] = return_expression
    if dump_path is not None:
        payload["dump_path"] = dump_path
    result = await bridge.execute("code.execute_python", payload)
    if isinstance(result, dict):
        result["justification"] = justification
    return result


###### code.execute_hscript


@mcp.tool()
async def execute_hscript(ctx: Context, command: str) -> dict:
    """Execute an HScript command in Houdini.

    Args:
        command: HScript command string to execute.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute("code.execute_hscript", {"command": command})


###### code.evaluate_expression


@mcp.tool()
async def evaluate_expression(ctx: Context, expression: str, language: str = "hscript") -> dict:
    """Evaluate an expression in Houdini and return its result.

    Args:
        expression: Expression string to evaluate.
        language: Expression language, "hscript" or "python".
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "code.evaluate_expression",
        {"expression": expression, "language": language},
    )


###### code.get_env_variable


@mcp.tool()
async def get_env_variable(ctx: Context, var_name: str) -> dict:
    """Get a Houdini environment variable value.

    Args:
        var_name: Name of the environment variable.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute("code.get_env_variable", {"var_name": var_name})
