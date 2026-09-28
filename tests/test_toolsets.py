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

# Third-party
import pytest

# Internal
from fxhoudinimcp import client, gateway, toolsets


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
    assert toolsets.summary(registered, {}) == "2 tools across 2 categories"
    assert toolsets.summary(registered, {"c": "z"}) == "2 tools, 1 more via list_toolsets"


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


def test_a_modeling_server_registers_the_modeling_tools_and_the_gateway():
    seen = _server_under("modeling")
    total = len(client.tools())
    assert set(seen["names"]) == set(toolsets.MODELING) | set(toolsets.GATEWAY)
    hidden = total - len(toolsets.MODELING)
    assert f"{len(seen['names'])} tools, {hidden} more via list_toolsets" in seen["header"]
    assert seen["client"] == total


def test_a_full_server_registers_everything_and_no_gateway():
    seen = _server_under("full")
    assert len(seen["names"]) == len(client.tools())
    assert not set(seen["names"]) & toolsets.GATEWAY
    assert "tools across" in seen["header"]


class TestGateway:
    """call_tool runs the tool's own function; bad arguments are named first."""

    @pytest.mark.asyncio
    async def test_call_tool_runs_the_tool_with_its_receipt(self, mock_ctx, mock_bridge):
        mock_bridge.execute.return_value = {"success": True, "note": "from the tool"}
        result = await gateway.call_tool(
            mock_ctx, "step_simulation", {"node_path": "/obj/dopnet1", "steps": 2}
        )
        assert result == {"success": True, "note": "from the tool"}
        command, params = mock_bridge.execute.call_args.args
        assert command.startswith("dops.") and params["node_path"] == "/obj/dopnet1"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name, arguments, message",
        [
            ("setup_pyro_sm", {}, "Did you mean: .*setup_pyro_sim"),
            ("get_parameter", {"node_path": "/obj/a", "nam": "tx"}, "unknown argument.*parm_name"),
            ("get_parameter", {"node_path": "/obj/a"}, "missing required.*parm_name"),
            ("step_simulation", {"node_path": "/obj/d", "steps": "many"}, "argument 'steps'"),
            ("call_tool", {}, "gateway tool"),
        ],
    )
    async def test_bad_calls_are_named_before_houdini(
        self, mock_ctx, mock_bridge, name, arguments, message
    ):
        with pytest.raises(ValueError, match=message):
            await gateway.call_tool(mock_ctx, name, arguments)
        mock_bridge.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_describe_gives_parameters_and_a_toolset_gives_summaries(self, mock_ctx):
        full = await gateway.describe_tools(mock_ctx, names=["step_simulation"])
        tool = full["tools"][0]
        assert tool["toolset"] == "dops" and tool["parameters"]["node_path"]["required"]
        summary = await gateway.describe_tools(mock_ctx, toolset="dops")
        assert {"step_simulation", "get_dop_field"} <= {t["name"] for t in summary["tools"]}
        with pytest.raises(ValueError, match="no toolset"):
            await gateway.describe_tools(mock_ctx, toolset="dop")

    def test_every_category_has_a_line(self):
        assert set(toolsets.REGISTERED.values()) | set(toolsets.SKIPPED.values()) <= set(
            gateway.ABOUT
        )


_E2E = """
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "fxhoudinimcp"], env=dict(os.environ))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.call_tool("list_toolsets", {})
            described = await session.call_tool("describe_tools", {"names": ["setup_pyro_sim"]})
            called = await session.call_tool(
                "call_tool", {"name": "get_houdini_connection_status", "arguments": {}}
            )
            bad = await session.call_tool("call_tool", {"name": "setup_pyro_sm"})
            text = lambda r: r.content[0].text
            print(json.dumps({
                "header": init.instructions.splitlines()[0],
                "listed": json.loads(text(listed)),
                "described": json.loads(text(described))["tools"][0]["name"],
                "called": json.loads(text(called)),
                "bad": [bad.is_error, text(bad)],
            }))

asyncio.run(main())
"""


def test_the_gateway_over_a_real_stdio_session():
    # A port nothing listens on: the connection-status tool answers either
    # way, so the whole path runs without touching a Houdini.
    env = dict(os.environ, **{toolsets.ENV: "modeling", "HOUDINI_PORT": "8199"})
    out = subprocess.run(
        [sys.executable, "-c", _E2E], env=env, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert "more via list_toolsets" in seen["header"]
    assert "setup_pyro_sim" in seen["listed"]["toolsets"]["workflows"]["tools"]
    assert seen["described"] == "setup_pyro_sim"
    assert seen["called"]["connected"] is False
    assert seen["bad"][0] is True and "setup_pyro_sim" in seen["bad"][1]
