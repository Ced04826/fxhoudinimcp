"""HTTP bridge connecting the MCP server to Houdini's hwebserver.

Houdini 22 UI sessions reserve ``/api`` and cannot reliably read request
bodies.  The plugin therefore exposes a body-free RPC endpoint at ``/fxapi``:
small JSON payloads travel in the query string and larger payloads travel via
a single-use file in the local temporary directory.
"""

from __future__ import annotations

# Built-in
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

# Third-party
import httpx

# Internal
from fxhoudinimcp.errors import ConnectionError, HoudiniCommandError

logger = logging.getLogger(__name__)


INLINE_RPC_LIMIT = 1500
TUNNEL_DIRECTORY = "fxhoudinimcp"

# Characters that do not survive the inline query path. hwebserver decodes the
# query string a second time after the transport already has, so a literal "+"
# arrives as a space and a "%" followed by two hex digits is re-expanded: on
# 22.0.368 the code ``x = 1 + 2`` reached execute_python as ``x = 1   2`` and a
# ``+=`` in a wrangle silently became ``=``. The file tunnel is read verbatim,
# so any payload containing these goes through it regardless of size.
_INLINE_UNSAFE = ("+", "%")


def _rpc_payload(func_name: str, **kwargs: Any) -> str:
    """Build the compact JSON payload understood by ``/fxapi``."""
    return json.dumps([func_name, [], kwargs], separators=(",", ":"))


def _rpc_query(payload: str) -> tuple[dict[str, str], Path | None]:
    """Return query parameters and an optional single-use payload file."""
    inline_safe = not any(ch in payload for ch in _INLINE_UNSAFE)
    if inline_safe and len(urlencode({"json": payload})) <= INLINE_RPC_LIMIT:
        return {"json": payload}, None

    directory = Path(tempfile.gettempdir()) / TUNNEL_DIRECTORY
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix="rpc-", suffix=".json", dir=directory)
    path = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"file": str(path)}, path


# Matches the plugin's own search range: a second Houdini moves itself to the
# next free port, so the client has to look there rather than assume 8100.
PORT_SEARCH_RANGE = 16


async def find_servers(
    host: str,
    base: int,
    max_tries: int = PORT_SEARCH_RANGE,
    timeout: float = 1.0,
) -> list[dict[str, Any]]:
    """Probe base..base+max_tries for live plugins, lowest port first.

    Each entry is the mcp.health payload plus the port it answered on. Returns
    every server found rather than just the first, so a caller can say how many
    Houdini sessions are running instead of silently picking one.

    Probing is cheap because mcp.health touches no HOM: a closed port refuses
    immediately, and a live one answers without waiting on Houdini's main thread.
    """
    found: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for port in range(base, base + max_tries):
            try:
                response = await client.get(
                    f"http://{host}:{port}/fxapi", params={"json": _rpc_payload("mcp.health")}
                )
                response.raise_for_status()
                payload = response.json()
            except Exception:
                continue  # nothing there, or not our endpoint
            if isinstance(payload, dict) and payload.get("status") == "ok":
                found.append({**payload, "port": port})
    return found


class HoudiniBridge:
    """Manages HTTP communication between the MCP server and Houdini's hwebserver.

    Calls use the plugin's ``/fxapi`` URL handler so they also work in H22 UI
    sessions, where Houdini shadows the built-in ``/api`` endpoint.
    """

    def __init__(self, host: str = "localhost", port: int = 8100, timeout: float = 60.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def _api_url(self) -> str:
        return f"{self.base_url}/fxapi"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def _reset_client(self) -> httpx.AsyncClient:
        """Discard the connection pool and return a fresh client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def _request(
        self,
        payload: str,
        timeout: float | None = None,
    ) -> httpx.Response:
        """GET from the bridge, retrying once past a dead pooled connection.

        Houdini closes its side of the keep-alive connections when it exits,
        so the first request after a Houdini restart reuses a socket that is
        already gone and httpx raises RemoteProtocolError. Retrying on a fresh
        pool reconnects to the new Houdini, instead of leaving this process
        permanently "disconnected" until the MCP client itself is restarted.
        """
        # httpx reads timeout=None as "wait forever", so fall back to the
        # configured timeout rather than passing None straight through.
        effective = self.timeout if timeout is None else timeout

        params, tunnel_file = _rpc_query(payload)
        client = await self._get_client()
        try:
            try:
                return await client.get(self._api_url, params=params, timeout=effective)
            except httpx.RemoteProtocolError:
                logger.info("Stale connection to Houdini; reconnecting.")
                # A file removed by the server may already have executed.  Do
                # not replay that mutation with a now-missing payload.  If the
                # file still exists, the stale connection died before Houdini
                # consumed it and one reconnect is safe.
                if tunnel_file is not None and not tunnel_file.exists():
                    raise
                client = await self._reset_client()
                return await client.get(self._api_url, params=params, timeout=effective)
        finally:
            if tunnel_file is not None:
                tunnel_file.unlink(missing_ok=True)

    async def execute(
        self,
        command: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Execute a command on Houdini and return the result data.

        Args:
            command: The command name (e.g. "scene.get_scene_info")
            params: Command parameters
            timeout: Override timeout for this request (seconds)

        Returns:
            The response data dict on success.

        Raises:
            ConnectionError: Cannot reach Houdini
            HoudiniCommandError: Houdini returned an error
        """
        request_id = str(uuid.uuid4())
        logger.info("→ Houdini: %s", command)

        try:
            response = await self._request(
                _rpc_payload(
                    "mcp.execute",
                    command=command,
                    params=params or {},
                    request_id=request_id,
                ),
                timeout=timeout or self.timeout,
            )
            response.raise_for_status()
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            raise ConnectionError(
                f"Cannot connect to Houdini at {self.base_url}. "
                "Is Houdini running with the fxhoudinimcp plugin loaded?",
                details={"url": self.base_url, "original_error": str(e)},
            ) from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(
                f"Houdini returned HTTP {e.response.status_code}",
                details={
                    "status_code": e.response.status_code,
                    "body": e.response.text,
                },
            ) from e
        except httpx.TimeoutException as e:
            raise ConnectionError(
                f"Request to Houdini timed out after {timeout or self.timeout}s",
                details={"timeout": timeout or self.timeout},
            ) from e
        except httpx.TransportError as e:
            # Everything the branches above do not name: ReadError/WriteError/
            # CloseError on a broken socket, and any transport error a future
            # httpx adds. Without this they reached the MCP client as raw httpx
            # exceptions, and several carry an empty message -- so the client
            # saw a failure with no indication of what went wrong or that
            # Houdini was the cause. ReadError is the one that actually escaped
            # in testing, which is why naming individual classes is a losing
            # game.
            raise ConnectionError(
                f"Lost the connection to Houdini at {self.base_url} "
                f"({type(e).__name__}). Has Houdini been closed or restarted?",
                details={"url": self.base_url, "original_error": str(e)},
            ) from e

        result = response.json()
        timing = result.get("timing_ms", "") if isinstance(result, dict) else ""
        logger.info("← Houdini: %s (%sms)", command, timing)

        if isinstance(result, dict) and result.get("status") == "error":
            err = result.get("error", {})
            raise HoudiniCommandError(
                message=err.get("message", "Unknown Houdini error"),
                code=err.get("code", "UNKNOWN"),
                details=err,
            )

        if isinstance(result, dict) and result.get("status") == "success":
            return result.get("data", {})

        # apiFunction may return the raw result directly
        return result

    async def health_check(self) -> dict[str, Any]:
        """Check if Houdini is responsive.

        Deliberately cheap: the plugin answers this without touching HOM, so it
        works while Houdini's main thread is busy. That is also why it reports
        no scene details -- use scene.get_scene_info for hip_file.

        Returns:
            Dict with status, pid and houdini_version.
        """
        try:
            response = await self._request(_rpc_payload("mcp.health"))
            response.raise_for_status()
            return response.json()
        except httpx.TransportError as e:
            # TransportError is the base for ConnectError, the timeout family,
            # RemoteProtocolError and the socket errors, so this covers every
            # way the transport can fail rather than the ones we happened to
            # name.
            raise ConnectionError(
                f"Health check failed: cannot reach Houdini at {self.base_url} "
                f"({type(e).__name__})",
                details={"original_error": str(e)},
            ) from e

    async def list_commands(self) -> list[str]:
        """Return the command names the connected plugin has registered.

        Used to detect a plugin older than this server. Calls mcp.list_commands
        rather than going through mcp.execute, so it works even when the
        dispatcher is missing commands.
        """
        try:
            response = await self._request(_rpc_payload("mcp.list_commands"))
            response.raise_for_status()
            payload = response.json()
        except httpx.TransportError as e:
            raise ConnectionError(
                f"Could not list plugin commands at {self.base_url} ({type(e).__name__})",
                details={"original_error": str(e)},
            ) from e

        commands = payload.get("commands") if isinstance(payload, dict) else None
        return commands if isinstance(commands, list) else []

    async def close(self) -> None:
        """Close the HTTP client connection."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
