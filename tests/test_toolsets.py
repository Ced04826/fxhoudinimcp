"""FXHOUDINIMCP_TOOLSET decides which tools the server registers.

The modeling toolset exists because every registered tool's name is sent to
every session and subagent; a modeling job used 28 of 214. What has to hold:
every name in a toolset is a real tool (a typo would silently drop it), an
MCP client under "modeling" sees exactly that set, the header says there are
more, and the script client still reaches every tool.
"""

from __future__ import annotations

# Built-in
import json
import os
import subprocess
import sys

# Internal
from fxhoudinimcp import client, toolsets


def test_every_modeling_name_is_a_real_tool():
    assert toolsets.MODELING - set(client.tools()) == set()


def test_unset_or_unknown_means_full(monkeypatch):
    monkeypatch.delenv(toolsets.ENV, raising=False)
    assert toolsets.active() == "full"
    monkeypatch.setenv(toolsets.ENV, "modelling")
    assert toolsets.active() == "full"
    monkeypatch.setenv(toolsets.ENV, " Modeling ")
    assert toolsets.active() == "modeling"


def test_the_header_says_when_tools_were_left_out():
    registered = {"a": "x", "b": "y"}
    assert toolsets.summary(registered, {}, "full") == "2 tools across 2 categories"
    assert toolsets.summary(registered, {"c": "z"}, "modeling") == "2 of 3 tools (toolset modeling)"


_PROBE = """
import asyncio, json
import fxhoudinimcp.tools
from fxhoudinimcp.server import mcp
from fxhoudinimcp import client
names = sorted(tool.name for tool in asyncio.run(mcp.list_tools()))
print(json.dumps({"names": names, "header": mcp.instructions.splitlines()[0],
                  "client": len(client.tools())}))
"""


def _server_under(toolset: str) -> dict:
    env = dict(os.environ, **{toolsets.ENV: toolset})
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_modeling_server_registers_exactly_the_modeling_tools():
    seen = _server_under("modeling")
    total = len(client.tools())
    assert set(seen["names"]) == set(toolsets.MODELING)
    assert f"{len(toolsets.MODELING)} of {total} tools (toolset modeling)" in seen["header"]
    assert seen["client"] == total


def test_a_full_server_registers_everything():
    seen = _server_under("full")
    assert len(seen["names"]) == len(client.tools())
    assert "tools across" in seen["header"]
