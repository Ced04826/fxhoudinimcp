"""Call fxhoudini's MCP tools from a script, without starting an MCP server.

A script that reached Houdini through an MCP stdio client started a whole
server process for every run: about 2.5 of the 3.5 seconds a network tidy
took. This runs the same tool functions the server registers, in the script's
own process, against one bridge to Houdini -- so the answer is the one the
MCP tool gives, argument handling included -- for the cost of importing the
package (about 1.3 s) and then about 0.1 s per call.

    from fxhoudinimcp.client import connect

    with connect() as houdini:
        info = houdini.call("get_scene_info")
        out = houdini.call("execute_python", code="...", justification="...")

    # or, inside asyncio code
    async with HoudiniClient() as houdini:
        info = await houdini.call("get_scene_info")

From a shell, one call, the result printed as JSON:

    python -m fxhoudinimcp.client get_scene_info
    python -m fxhoudinimcp.client capture_viewport '{"output_dir": "C:/tmp/shots"}'
    python -m fxhoudinimcp.client execute_python @args.json

The connection follows the server's own rules: HOUDINI_HOST, HOUDINI_PORT
(pinned) or a scan for the first serving Houdini from 8100, and the same
timeouts.
"""

from __future__ import annotations

# Built-in
import asyncio
import inspect
import json
import os
import sys
from typing import Any

# Internal
from fxhoudinimcp.bridge import HoudiniBridge, find_servers


class _Context:
    """The part of an MCP request context the tool functions use."""

    def __init__(self, bridge: HoudiniBridge) -> None:
        self.request_context = type("RequestContext", (), {})()
        self.request_context.lifespan_context = {"bridge": bridge}

    async def report_progress(self, *_: Any, **__: Any) -> None:
        return None


_TOOLS: dict[str, Any] | None = None


def tools() -> dict[str, Any]:
    """Every registered tool function by tool name.

    Importing the tools package is what registers them on the server; the
    functions stay plain coroutine functions taking ctx first.
    """
    global _TOOLS
    if _TOOLS is None:
        import importlib
        import pkgutil

        import fxhoudinimcp.tools as package

        found: dict[str, Any] = {}
        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{info.name}")
            for name, value in vars(module).items():
                if (
                    inspect.iscoroutinefunction(value)
                    and value.__module__ == module.__name__
                    and not name.startswith("_")
                    and next(iter(inspect.signature(value).parameters), None) == "ctx"
                ):
                    found[name] = value
        _TOOLS = found
    return _TOOLS


async def _open_bridge() -> HoudiniBridge:
    """A bridge found the way the MCP server finds one (see server.lifespan).

    The default host is 127.0.0.1, the address the plugin binds, rather than
    "localhost", which on Windows first spends a quarter of a second on ::1.
    """
    host = os.getenv("HOUDINI_HOST", "127.0.0.1")
    pinned = os.getenv("HOUDINI_PORT")
    port = int(pinned) if pinned else 8100
    if not pinned:
        servers = await find_servers(host, port)
        if servers:
            port = servers[0]["port"]
    plugin_timeout = float(os.getenv("FXHOUDINIMCP_TIMEOUT", "120"))
    timeout = float(os.getenv("HOUDINI_TIMEOUT", str(plugin_timeout + 15)))
    return HoudiniBridge(host=host, port=port, timeout=timeout)


class HoudiniClient:
    """Async client: `await client.call(tool_name, **arguments)`."""

    def __init__(self, bridge: HoudiniBridge | None = None) -> None:
        self._bridge = bridge
        self._owns_bridge = bridge is None
        self._context: _Context | None = None

    async def __aenter__(self) -> HoudiniClient:
        if self._bridge is None:
            self._bridge = await _open_bridge()
        self._context = _Context(self._bridge)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._owns_bridge and self._bridge is not None:
            await self._bridge.close()

    async def call(self, tool: str, **arguments: Any) -> Any:
        """Run MCP tool *tool* with *arguments*, exactly as the server would."""
        if self._context is None:
            raise RuntimeError("use the client as a context manager: async with HoudiniClient()")
        function = tools().get(tool)
        if function is None:
            from difflib import get_close_matches

            close = get_close_matches(tool, list(tools()), n=3, cutoff=0.5)
            hint = f" Did you mean: {close}?" if close else ""
            raise ValueError(f"no fxhoudini tool named {tool!r}.{hint}")
        return await function(self._context, **arguments)


class SyncClient:
    """Blocking client for plain scripts: `client.call(tool_name, **arguments)`."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._client = HoudiniClient()

    def __enter__(self) -> SyncClient:
        self._loop.run_until_complete(self._client.__aenter__())
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self._loop.run_until_complete(self._client.__aexit__(*exc))
        finally:
            self._loop.close()

    def call(self, tool: str, **arguments: Any) -> Any:
        return self._loop.run_until_complete(self._client.call(tool, **arguments))


def connect() -> SyncClient:
    """A blocking client, to be used as a context manager."""
    return SyncClient()


def main(argv: list[str] | None = None) -> int:
    """`python -m fxhoudinimcp.client TOOL [JSON | @file.json]`; prints the result."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--list":
        print("\n".join(sorted(tools())))
        return 0
    arguments: dict[str, Any] = {}
    if len(argv) > 1:
        text = argv[1]
        if text.startswith("@"):
            with open(text[1:], encoding="utf-8") as handle:
                text = handle.read()
        arguments = json.loads(text)
        if not isinstance(arguments, dict):
            raise SystemExit("arguments must be a JSON object")
    with connect() as houdini:
        result = houdini.call(argv[0], **arguments)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
