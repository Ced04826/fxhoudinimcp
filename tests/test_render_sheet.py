"""render_sheet: argument checks and the child's result line."""

from __future__ import annotations

# Built-in
import json
from unittest.mock import AsyncMock, MagicMock

# Third-party
import pytest

# Internal
from fxhoudinimcp.tools import rendering


@pytest.fixture
def bridge(monkeypatch, tmp_path):
    snapshot = tmp_path / "snap.hip"
    snapshot.write_text("hip")
    bridge = MagicMock()
    bridge.execute = AsyncMock(
        return_value={
            "snapshot": str(snapshot),
            "hip_file": "C:/shots/a.hip",
            "houdini_version": "22.0.368",
        }
    )
    monkeypatch.setattr(rendering, "_get_bridge", lambda _ctx: bridge)
    monkeypatch.setattr(rendering, "_hython_for", lambda version: "hython")
    return bridge, snapshot


def _child_prints(monkeypatch, text: str, code: int = 0):
    process = MagicMock(returncode=code)
    process.communicate = AsyncMock(return_value=(text.encode(), b""))
    monkeypatch.setattr(
        rendering.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("start,end,step", [(1, 200, 1), (5, 1, 1), (1, 10, 0)])
async def test_frame_counts_outside_1_to_64_are_refused(bridge, start, end, step):
    with pytest.raises(ValueError, match="1 to 64"):
        await rendering.render_sheet(MagicMock(), start=start, end=end, step=step)


@pytest.mark.asyncio
async def test_the_result_line_is_returned_and_the_snapshot_removed(bridge, monkeypatch):
    _, snapshot = bridge
    line = json.dumps({"output_path": "sheet.png", "frames": [1, 6]})
    _child_prints(monkeypatch, "noise\n__MCP_RESULT__ " + line + "\n")
    result = await rendering.render_sheet(MagicMock(), start=1, end=6, step=5)
    assert result["frames"] == [1, 6] and result["hip_file"] == "C:/shots/a.hip"
    assert not snapshot.exists()


@pytest.mark.asyncio
async def test_a_child_error_is_raised(bridge, monkeypatch):
    _child_prints(monkeypatch, "__MCP_RESULT__ " + json.dumps({"error": "Pass camera=: none"}), 2)
    with pytest.raises(ValueError, match="Pass camera"):
        await rendering.render_sheet(MagicMock(), start=1, end=2)


@pytest.mark.asyncio
async def test_a_crash_shows_the_tail_of_its_output(bridge, monkeypatch):
    _child_prints(monkeypatch, "loading\nFatal error: Segmentation fault\n", 139)
    with pytest.raises(RuntimeError, match="code 139:\nloading\nFatal error"):
        await rendering.render_sheet(MagicMock(), start=1, end=2)
