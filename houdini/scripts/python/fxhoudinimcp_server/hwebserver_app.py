"""hwebserver endpoint registration for the FXHoudini-MCP plugin.

Registers a custom URL handler on Houdini's built-in HTTP server that the
external MCP server communicates with over HTTP.

Calling convention (JSON-encoded RPC):
    GET /fxapi?json=["mcp.execute", [], {"command": "...", ...}]

Large requests use ``?file=...`` and are read once from the guarded
``%TEMP%/fxhoudinimcp`` directory.  This avoids two Houdini 22 UI bugs: the
built-in web API shadows ``/api``, and request bodies are not readable.

Responses are JSON-encoded here rather than left to hwebserver, so that a
value HOM cannot serialise degrades instead of collapsing into an opaque 500.
"""

from __future__ import annotations

# Built-in
import json
import os
import tempfile
import traceback
from pathlib import Path

# Third-party
import hwebserver

# Internal
from fxhoudinimcp_server import dispatcher
from fxhoudinimcp_server.serialize import json_default

###### Registration

TUNNEL_DIRECTORY = "fxhoudinimcp"
MAX_TUNNEL_BYTES = 64 * 1024 * 1024


def _api_function(namespace: str):
    """Register an API function with hwebserver and keep the module attribute.

    hwebserver's module-level ``apiFunction`` decorator returns ``None``,
    because ``Server._apiFunction`` has no return statement. Registration
    itself works, but the decorated name would otherwise be bound to ``None``,
    which breaks anything that later imports the function -- including tests.

    Registration is thread-local: hwebserver keeps its ``Server`` in a
    ``threading.local()``, so the thread that imports this module must also be
    the thread that calls ``hwebserver.run()``. See startup.py.
    """

    def decorator(function):
        hwebserver.apiFunction(namespace=namespace)(function)
        return function

    return decorator


def _url_handler(path: str):
    """Register a URL handler without losing the decorated function."""

    def decorator(function):
        hwebserver.urlHandler(path)(function)
        return function

    return decorator


def _json_response(payload: dict) -> hwebserver.Response:
    """Encode *payload* as an HTTP JSON response.

    hwebserver would otherwise call ``json.dumps`` itself, from a place
    outside its own exception handling, so an unserialisable value there
    escapes as a bare HTTP 500 with no diagnostic. Encoding here lets a
    ``default=`` hook coerce stray HOM objects, and lets a genuine encoding
    failure come back as a readable error instead of a blank 500.
    """
    try:
        body = json.dumps(payload, default=json_default)
    except Exception as exc:
        body = json.dumps(
            {
                "status": "error",
                "error": {
                    "code": "SERIALIZATION_ERROR",
                    "message": (f"Result could not be JSON-encoded: {type(exc).__name__}: {exc}"),
                    "traceback": traceback.format_exc(),
                },
            }
        )
    return hwebserver.Response(body.encode("utf-8"), 200, "application/json")


def _query_value(request, name: str) -> str | None:
    value = request.GET().get(name)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value if isinstance(value, str) else None


def _read_tunnel_file(filename: str) -> str:
    """Read and delete a payload file, refusing paths outside our temp root."""
    root = (Path(tempfile.gettempdir()) / TUNNEL_DIRECTORY).resolve()
    path = Path(filename).resolve(strict=True)
    if os.path.commonpath((str(root), str(path))) != str(root):
        raise ValueError("Payload file is outside the fxhoudinimcp temp directory")
    if path.suffix != ".json" or not path.name.startswith("rpc-"):
        raise ValueError("Payload file name is not an fxhoudinimcp RPC file")
    try:
        size = path.stat().st_size
        if size > MAX_TUNNEL_BYTES:
            raise ValueError(f"Payload file exceeds {MAX_TUNNEL_BYTES} bytes")
        return path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)


def _rpc_error(code: str, message: str) -> hwebserver.Response:
    return _json_response({"status": "error", "error": {"code": code, "message": message}})


###### Endpoints


@_api_function("mcp")
def execute(request, command="", params=None, request_id=""):
    """Single entry point for all MCP tool calls.

    Args:
        request: hwebserver.Request (always first arg).
        command: Dotted command name (e.g. "scene.get_scene_info").
        params: Tool-specific parameters dict.
        request_id: Correlation ID echoed back in the response.
    """
    if params is None:
        params = {}

    result = dispatcher.dispatch(command, params)
    result["request_id"] = request_id
    return _json_response(result)


@_api_function("mcp")
def health(request):
    """Liveness check. Deliberately free of any hou.* access.

    This is what startup polls to decide the server is ready, and hwebserver
    serves it from a worker thread. Touching HOM here deadlocks a GUI session:
    the main thread is inside startup's readiness loop and so is not running
    Houdini's event loop, while HOM access from the worker needs precisely
    that main thread to make progress. Neither side can advance.

    Version comes from the environment for the same reason -- Houdini exports
    HOUDINI_VERSION, so reporting it costs no HOM call. Anything needing the
    scene itself belongs in session_info.
    """
    return {
        "status": "ok",
        "pid": os.getpid(),
        "houdini_version": os.environ.get("HOUDINI_VERSION", "unknown"),
    }


@_api_function("mcp")
def session_info(request):
    """Scene-level session details, marshalled to the main thread.

    Separate from health because this does touch HOM: it goes through the
    normal dispatch path, so it is only safe once the session is idle.
    """
    return _json_response(dispatcher.dispatch("scene.get_scene_info", {}))


@_api_function("mcp")
def list_commands(request):
    """List all registered command names for introspection."""
    return {"commands": dispatcher.list_commands()}


@_url_handler("/fxapi")
def fxapi(request):
    """Body-free RPC endpoint compatible with graphical Houdini 22 sessions."""
    inline = _query_value(request, "json")
    filename = _query_value(request, "file")
    if bool(inline) == bool(filename):
        return _rpc_error("INVALID_REQUEST", "Provide exactly one of 'json' or 'file'")

    try:
        raw = inline if inline is not None else _read_tunnel_file(filename)
        call = json.loads(raw)
        if not isinstance(call, list) or len(call) != 3:
            raise ValueError("RPC payload must be [function, args, kwargs]")
        function_name, args, kwargs = call
        if not isinstance(function_name, str):
            raise ValueError("RPC function name must be a string")
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise ValueError("RPC args and kwargs must be a list and object")

        functions = {
            "mcp.execute": execute,
            "mcp.health": health,
            "mcp.session_info": session_info,
            "mcp.list_commands": list_commands,
        }
        function = functions.get(function_name)
        if function is None:
            return _rpc_error("UNKNOWN_RPC_FUNCTION", f"Unknown RPC function: {function_name}")
        result = function(request, *args, **kwargs)
        return result if isinstance(result, hwebserver.Response) else _json_response(result)
    except Exception as exc:
        return _rpc_error("INVALID_REQUEST", f"{type(exc).__name__}: {exc}")
