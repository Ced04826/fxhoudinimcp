"""MCP tools that choose, start and stop the Houdini session this server talks to.

The server connects at startup to the first Houdini it finds. With several open,
or none, the agent had no way to pick one or to get one: these list what is
serving, switch the bridge to another port, and start a Houdini of its own.
"""

from __future__ import annotations

# Built-in
import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Third-party
from fxhoudinimcp._sdk import Context

# Internal
from fxhoudinimcp.bridge import find_servers
from fxhoudinimcp.houdini_discovery import find_hython
from fxhoudinimcp.houdini_package import plugin_path
from fxhoudinimcp.server import _get_bridge, mcp

# The plugin's own port search starts here, so the sessions live from here up.
BASE_PORT = 8100

# Sessions this server started: pid -> (process, port it serves, port to go back to).
_started: dict[int, tuple[subprocess.Popen, int, int]] = {}


async def list_sessions(bridge) -> list[dict[str, Any]]:
    """Every Houdini serving the plugin, the one this server talks to marked current."""
    sessions = await find_servers(bridge.host, BASE_PORT)
    for session in sessions:
        session["current"] = session["port"] == bridge.port
        session["started_here"] = session.get("pid") in _started
    return sessions


@mcp.tool()
async def connect_houdini(ctx: Context, port: int) -> dict:
    """Talk to the Houdini serving on ``port`` from now on.

    get_houdini_connection_status lists the sessions and their ports.

    Args:
        port: Port of a Houdini serving the plugin.
    """
    bridge = _get_bridge(ctx)
    found = await find_servers(bridge.host, port, max_tries=1)
    if not found:
        others = await list_sessions(bridge)
        raise ValueError(
            f"No Houdini answers on port {port}. Serving: "
            + (", ".join(f"{s['port']} (pid {s.get('pid')})" for s in others) or "none")
            + ". start_houdini starts one."
        )
    previous = bridge.port
    await bridge.retarget(port)
    return {"connected": True, "port": port, "previous_port": previous, "health": found[0]}


def _hython_script(hip_file: str | None) -> str:
    """What the headless hython runs: load the plugin, open the hip, serve until stopped."""
    scripts = (plugin_path() / "scripts" / "python").as_posix()
    load = f"hou.hipFile.load({hip_file!r}, suppress_save_prompt=True)\n" if hip_file else ""
    return (
        "import sys\n"
        f"sys.path.insert(0, {scripts!r})\n"
        "import hou\n"
        f"{load}"
        "from fxhoudinimcp_server import startup\n"
        "startup.start(background=False)\n"
    )


def _log_tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


@mcp.tool()
async def start_houdini(
    ctx: Context,
    gui: bool = False,
    hip_file: str | None = None,
    wait_seconds: float = 180,
) -> dict:
    """Start a Houdini of this server's own and connect to it.

    Headless (hython) by default: no window, no viewport, so the capture tools
    refuse there. gui=True starts the full application, which serves only if
    the plugin's package is installed (fxhoudinimcp install). Either takes a
    license seat until stop_houdini. Set HYTHON to choose the build.

    Args:
        gui: Start the Houdini application instead of headless hython.
        hip_file: Scene to open.
        wait_seconds: How long to wait for it to serve.
    """
    bridge = _get_bridge(ctx)
    hython = find_hython()
    if hython is None:
        raise RuntimeError("No Houdini found. Install one, or set HYTHON to its hython executable.")
    if gui:
        name = "houdini.exe" if sys.platform == "win32" else "houdini"
        command = [str(hython.with_name(name))] + ([hip_file] if hip_file else [])
    else:
        command = [str(hython), "-c", _hython_script(hip_file)]

    before = {s.get("pid") for s in await find_servers(bridge.host, BASE_PORT)}
    log = (
        Path(tempfile.gettempdir()) / "fxhoudinimcp" / f"houdini_{time.time_ns() // 1_000_000}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    with open(log, "wb") as out:
        process = subprocess.Popen(
            command,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=os.environ.copy(),
            creationflags=flags,
            start_new_session=sys.platform != "win32",
        )

    started = time.monotonic()
    while time.monotonic() - started < wait_seconds:
        if process.poll() is not None:
            raise RuntimeError(
                f"Houdini exited with code {process.returncode} before serving. "
                f"Last lines of {log}:\n{_log_tail(log)}"
            )
        # A GUI launcher can hand off to another pid, so any new session counts.
        new = [s for s in await find_servers(bridge.host, BASE_PORT) if s.get("pid") not in before]
        if new:
            session = next((s for s in new if s.get("pid") == process.pid), new[0])
            previous = bridge.port
            _started[session["pid"]] = (process, session["port"], previous)
            await bridge.retarget(session["port"])
            return {
                "started": True,
                "mode": "gui" if gui else "headless",
                "pid": session["pid"],
                "port": session["port"],
                "previous_port": previous,
                "seconds": round(time.monotonic() - started, 1),
                "log": str(log),
            }
        await asyncio.sleep(1.0)

    raise TimeoutError(
        f"Houdini (pid {process.pid}) did not serve within {wait_seconds:g}s and is still "
        f"running; stop_houdini(pid={process.pid}) ends it. "
        + ("With gui=True the plugin serves only if its package is installed. " if gui else "")
        + f"Last lines of {log}:\n{_log_tail(log)}"
    )


@mcp.tool()
async def stop_houdini(ctx: Context, pid: int | None = None) -> dict:
    """Stop a Houdini that start_houdini started, and go back to the previous session.

    Only sessions started by this server can be stopped: anything else may be
    someone's unsaved work.

    Args:
        pid: Session to stop. Default: the current one, if this server started it.
    """
    bridge = _get_bridge(ctx)
    if pid is None:
        pid = next((p for p, (_, port, _) in _started.items() if port == bridge.port), None)
    if pid not in _started:
        raise ValueError(
            "That session was not started by this server, so it is not stopped. "
            f"Started here: {sorted(_started) or 'none'}."
        )
    process, port, previous = _started.pop(pid)
    if process.pid == pid:
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 15)
        except subprocess.TimeoutExpired:
            process.kill()
    else:
        # A GUI launcher handed off to this pid; the launcher is not what serves.
        os.kill(pid, signal.SIGTERM)
    back = None
    if bridge.port == port and await find_servers(bridge.host, previous, max_tries=1):
        await bridge.retarget(previous)
        back = previous
    return {"stopped": pid, "port": port, "reconnected_to": back}
