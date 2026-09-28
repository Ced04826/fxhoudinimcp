"""One import site for the MCP SDK, so a renamed module is a one-line fix.

mcp 2.0.0 removed ``mcp.server.fastmcp``. The server class moved to
``mcp.server.mcpserver`` and was renamed ``FastMCP`` -> ``MCPServer``. Every
tool, resource and prompt in this package imported ``Context`` from the old
path, so a fresh ``pip install`` resolving mcp 2.x produced a server that could
not import at all, with 26 import sites to chase.

The public API this package uses is unchanged between the two: ``MCPServer(name,
instructions)``, the ``tool()``/``prompt()``/``resource(uri)`` decorators, and
``Context``. So supporting both is an import shim rather than a port, and the
dependency does not need pinning to a major version.
"""

from __future__ import annotations

try:
    # mcp >= 2.0
    from mcp.server.mcpserver import Context as Context
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # pragma: no cover - depends on the installed mcp
    # mcp 1.x
    from mcp.server.fastmcp import Context as Context
    from mcp.server.fastmcp import FastMCP as Server


def build_server(*, name: str, instructions: str, lifespan, version: str):
    """Construct the SDK server and give it a version, on either major.

    The version is why this is a function rather than a bare class re-export.
    mcp 1.x has no public way to set it, so the only option was poking
    ``server._mcp_server.version``. mcp 2.0 accepts ``version`` on the
    constructor and has no ``_mcp_server`` at all, so the poke raised
    AttributeError at import time -- which is a broken server, not a missing
    version string.
    """
    try:
        return Server(
            name=name,
            instructions=instructions,
            lifespan=lifespan,
            version=version,
        )
    except TypeError:
        # mcp 1.x: no version kwarg, so set it on the wrapped low-level server.
        server = Server(name=name, instructions=instructions, lifespan=lifespan)
        inner = getattr(server, "_mcp_server", None)
        if inner is not None:
            inner.version = version
        return server


def set_instructions(server, text: str) -> None:
    """Replace the instructions a server hands clients at initialize.

    Neither major has a public setter; the text lives on the low-level
    server, named ``_lowlevel_server`` in 2.x and ``_mcp_server`` in 1.x.
    """
    inner = getattr(server, "_lowlevel_server", None) or getattr(server, "_mcp_server", None)
    if inner is None:
        raise AttributeError("cannot find the low-level server to set instructions on")
    inner.instructions = text


__all__ = ["Context", "Server", "build_server", "set_instructions"]
