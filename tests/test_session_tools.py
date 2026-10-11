"""Choosing, starting and stopping the Houdini this server talks to."""

from __future__ import annotations

# Built-in
from unittest.mock import AsyncMock, MagicMock

# Third-party
import pytest

# Internal
from fxhoudinimcp.tools import session


@pytest.fixture
def bridge(monkeypatch):
    bridge = MagicMock(host="localhost", port=8100)
    bridge.retarget = AsyncMock()
    ctx = MagicMock()
    monkeypatch.setattr(session, "_get_bridge", lambda _ctx: bridge)
    monkeypatch.setattr(session, "_started", {})
    return bridge, ctx


def _serving(*ports):
    async def find(host, base, max_tries=16, timeout=1.0):
        return [{"port": p, "pid": p * 10} for p in ports if base <= p < base + max_tries]

    return find


@pytest.mark.asyncio
async def test_sessions_mark_the_current_one(bridge, monkeypatch):
    b, _ = bridge
    monkeypatch.setattr(session, "find_servers", _serving(8100, 8101))
    found = await session.list_sessions(b)
    assert [(s["port"], s["current"]) for s in found] == [(8100, True), (8101, False)]


@pytest.mark.asyncio
async def test_connect_refuses_a_port_nothing_serves_and_names_what_does(bridge, monkeypatch):
    b, ctx = bridge
    monkeypatch.setattr(session, "find_servers", _serving(8100))
    with pytest.raises(ValueError, match=r"No Houdini answers on port 8105\. Serving: 8100"):
        await session.connect_houdini(ctx, 8105)
    b.retarget.assert_not_called()


@pytest.mark.asyncio
async def test_stop_refuses_a_session_it_did_not_start(bridge):
    _, ctx = bridge
    with pytest.raises(ValueError, match="not started by this server"):
        await session.stop_houdini(ctx, pid=1234)


@pytest.mark.asyncio
async def test_stop_goes_back_to_the_previous_session(bridge, monkeypatch):
    b, ctx = bridge
    b.port = 8101
    process = MagicMock(pid=81010)
    session._started[81010] = (process, 8101, 8100)
    monkeypatch.setattr(session, "find_servers", _serving(8100))
    result = await session.stop_houdini(ctx)
    process.terminate.assert_called_once()
    b.retarget.assert_awaited_once_with(8100)
    assert result == {"stopped": 81010, "port": 8101, "reconnected_to": 8100}


def test_headless_script_loads_the_plugin_and_the_scene():
    script = session._hython_script("C:/shots/a.hip")
    compile(script, "<hython>", "exec")
    assert "hou.hipFile.load('C:/shots/a.hip'" in script
    assert "startup.start(background=False)" in script
