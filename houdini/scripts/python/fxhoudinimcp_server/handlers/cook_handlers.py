"""Cook nodes on demand and report what each cook produced.

Changing a Python SOP's code file on disk changes no parameter, so Houdini
keeps the cached result; seven modeling rounds reached for execute_python to
call cook(force=True) and then read the node back by hand. This does both for
a list of nodes: whether each cooked, how long it took, its errors and
warnings, and the geometry it produced.
"""

from __future__ import annotations

# Built-in
import contextlib
import time
from typing import Any

# Third-party
import hou

# Internal
from fxhoudinimcp_server.config import update_mode_warning
from fxhoudinimcp_server.dispatcher import register_handler
from fxhoudinimcp_server.errors import readable_message
from fxhoudinimcp_server.handlers.state_receipt_helpers import clip

# Messages kept per node; the count says how many there were.
_MESSAGES = 5
_MESSAGE_CHARS = 400


def _messages(entry: dict[str, Any], key: str, messages: list) -> None:
    if messages:
        entry[key] = [clip(str(m), _MESSAGE_CHARS) for m in messages[:_MESSAGES]]
        entry[f"{key[:-1]}_count"] = len(messages)


def cook_node(node_paths: Any, force: bool = True, **_: Any) -> dict[str, Any]:
    """Cook each node in *node_paths* (forced by default) and read it back.

    success is false when a node is missing, fails to cook or reports errors,
    and when Houdini's update mode keeps nodes from cooking at all: measured
    upstream on 22.0, in Manual not even cook(force=True) cooks, so a cook
    that "worked" there proves nothing.
    """
    if isinstance(node_paths, str):
        node_paths = [node_paths]
    if not isinstance(node_paths, (list, tuple)) or not node_paths:
        raise ValueError("node_paths must be a non-empty list of node paths")
    if not isinstance(force, bool):
        raise ValueError(f"force must be true or false, not {force!r}")

    nodes: list[dict[str, Any]] = []
    failed = 0
    for path in node_paths:
        node = hou.node(str(path))
        if node is None:
            nodes.append({"path": str(path), "cooked": False, "error": "node not found"})
            failed += 1
            continue
        entry: dict[str, Any] = {"path": node.path(), "type": node.type().name()}
        started = time.perf_counter()
        try:
            node.cook(force=force)
            entry["cooked"] = True
        except Exception as exc:  # noqa: BLE001 - a failed cook is the answer, not a crash
            entry["cooked"] = False
            entry["cook_error"] = clip(readable_message(exc), _MESSAGE_CHARS)
        entry["cook_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        errors: list = []
        with contextlib.suppress(Exception):
            errors = list(node.errors())
        warnings: list = []
        with contextlib.suppress(Exception):
            warnings = list(node.warnings())
        _messages(entry, "errors", errors)
        _messages(entry, "warnings", warnings)
        geometry = None
        if hasattr(node, "geometry"):
            with contextlib.suppress(Exception):
                geometry = node.geometry()
        if geometry is not None:
            entry["points"] = geometry.intrinsicValue("pointcount")
            entry["prims"] = geometry.intrinsicValue("primitivecount")
        if not entry["cooked"] or errors:
            failed += 1
        nodes.append(entry)

    result: dict[str, Any] = {"success": failed == 0, "failed": failed, "force": force, "nodes": nodes}
    warning = update_mode_warning()
    if warning:
        result["success"] = False
        result["warning"] = warning
    return result


register_handler("cook.cook_node", cook_node)
