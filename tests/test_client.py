"""Tests for fxhoudinimcp.client: MCP tools called in-process from a script."""

from __future__ import annotations

# Built-in
import json

# Third-party
import pytest

# Internal
from fxhoudinimcp import client


class TestToolDiscovery:
    def test_every_registered_tool_is_callable_by_name(self):
        found = client.tools()
        assert "execute_python" in found
        assert "capture_viewport" in found
        assert all(callable(fn) for fn in found.values())

    def test_the_discovered_set_is_the_server_s(self):
        # The header count in server_instructions.md is checked against the
        # server's registry elsewhere; the client must see the same tools.
        import re

        from fxhoudinimcp._loader import load_markdown

        header = load_markdown("instructions/server_instructions.md").splitlines()[0]
        claimed = int(re.search(r"(\d+) tools", header).group(1))
        assert len(client.tools()) == claimed


class TestCall:
    @pytest.mark.asyncio
    async def test_a_call_runs_the_tool_against_the_bridge(self, mock_bridge):
        mock_bridge.execute.return_value = {"executed": True, "return_value": 2}
        async with client.HoudiniClient(bridge=mock_bridge) as houdini:
            result = await houdini.call(
                "execute_python", code="x = 1 + 1", justification="test", return_expression="x"
            )
        assert result["return_value"] == 2
        command, params = mock_bridge.execute.call_args.args[:2]
        assert command == "code.execute_python"
        assert params["code"] == "x = 1 + 1"
        # A bridge the caller passed in is the caller's to close.
        mock_bridge.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unknown_tool_is_named_with_suggestions(self, mock_bridge):
        async with client.HoudiniClient(bridge=mock_bridge) as houdini:
            with pytest.raises(ValueError, match="get_scene_info"):
                await houdini.call("get_scene_infoo")

    @pytest.mark.asyncio
    async def test_calling_outside_the_context_manager_is_refused(self):
        with pytest.raises(RuntimeError, match="context manager"):
            await client.HoudiniClient().call("get_scene_info")


class TestCommandLine:
    def test_list_prints_tool_names(self, capsys):
        assert client.main(["--list"]) == 0
        assert "capture_viewport" in capsys.readouterr().out.split()

    def test_arguments_come_from_json_or_a_file(self, tmp_path, monkeypatch, capsys):
        seen = {}

        class FakeSync:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def call(self, tool, **arguments):
                seen["tool"], seen["arguments"] = tool, arguments
                return {"ok": True}

        monkeypatch.setattr(client, "connect", FakeSync)
        args_file = tmp_path / "args.json"
        args_file.write_text(json.dumps({"node_path": "/obj/geo1"}), encoding="utf-8")
        assert client.main(["get_node_info", f"@{args_file}"]) == 0
        assert seen == {"tool": "get_node_info", "arguments": {"node_path": "/obj/geo1"}}
        assert json.loads(capsys.readouterr().out) == {"ok": True}
