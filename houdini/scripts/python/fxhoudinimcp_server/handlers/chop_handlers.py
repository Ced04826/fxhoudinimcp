"""CHOP handlers for FXHoudini-MCP.

Provides tools for inspecting, creating, and exporting CHOP
(Channel Operator) nodes and their channel data within Houdini.
"""

from __future__ import annotations

# Built-in
from typing import Any

# Third-party
import hou

# Internal
from fxhoudinimcp_server.config import place_new_node
from fxhoudinimcp_server.dispatcher import register_handler
from fxhoudinimcp_server.errors import readable_message
from fxhoudinimcp_server.handlers.node_handlers import _refuse_taken_name
from fxhoudinimcp_server.ui import focus_network_editor as _focus_network_editor

###### Helpers


def _get_node(node_path: str) -> hou.Node:
    """Resolve a node path and raise a clear error if it does not exist."""
    node = hou.node(node_path)
    if node is None:
        raise ValueError(f"Node not found: {node_path}")
    return node


###### chops.get_chop_data


def _get_chop_data(
    *,
    node_path: str,
    channel_name: str | None = None,
    start: int | None = None,
    end: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Get CHOP node track data.

    If channel_name is given, returns that channel's samples within the
    optional frame range. Otherwise returns all channel names and metadata.

    Args:
        node_path: Path to the CHOP node.
        channel_name: Optional specific channel name to retrieve samples for.
        start: Optional start sample index.
        end: Optional end sample index.
    """
    node = _get_node(node_path)

    if node.type().category().name() != "Chop":
        raise ValueError(f"Node is not a CHOP node: {node_path}")

    # Collect channel metadata
    channels: list[dict[str, Any]] = []
    for track in node.tracks():
        track_name = track.name()
        num_samples = track.numSamples()
        clip = track.clip()
        sample_rate = clip.sampleRate()
        start_idx = int(clip.sampleRange()[0])
        end_idx = start_idx + num_samples - 1

        # Compute min/max values
        all_samples = track.allSamples()
        min_val = min(all_samples) if all_samples else 0.0
        max_val = max(all_samples) if all_samples else 0.0

        channels.append(
            {
                "name": track_name,
                "sample_count": num_samples,
                "start": start_idx,
                "end": end_idx,
                "rate": sample_rate,
                "min_val": min_val,
                "max_val": max_val,
            }
        )

    result: dict[str, Any] = {
        "node_path": node_path,
        "channels": channels,
    }

    # If a specific channel is requested, also return its samples
    if channel_name is not None:
        track = node.track(channel_name)
        if track is None:
            raise ValueError(f"Channel '{channel_name}' not found on CHOP node: {node_path}")

        all_samples = list(track.allSamples())

        # Apply start/end range if provided
        if start is not None or end is not None:
            clip = track.clip()
            clip_start = int(clip.sampleRange()[0])
            s = (start - clip_start) if start is not None else 0
            e = (end - clip_start + 1) if end is not None else len(all_samples)
            s = max(0, s)
            e = min(len(all_samples), e)
            all_samples = all_samples[s:e]

        # Cap output to avoid huge responses
        if len(all_samples) > 10000:
            all_samples = all_samples[:10000]

        result["samples"] = all_samples

    return result


register_handler("chops.get_chop_data", _get_chop_data)


###### chops.create_chop_node


def _create_chop_node(
    *,
    parent_path: str,
    chop_type: str,
    name: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Create a CHOP node inside a parent network.

    Args:
        parent_path: Path to the parent network.
        chop_type: Type of CHOP node to create (e.g. "wave", "noise", "math").
        name: Optional name for the new node.
    """
    parent = _get_node(parent_path)
    if name:
        _refuse_taken_name(parent, name)

    try:
        node = parent.createNode(chop_type, node_name=name)
    except hou.OperationFailed as e:
        raise ValueError(
            f"Failed to create CHOP node of type '{chop_type}' inside '{parent_path}': {readable_message(e)}"
        ) from e

    place_new_node(node)
    _focus_network_editor(node)

    return {
        "node_path": node.path(),
        "type": node.type().name(),
        "name": node.name(),
    }


register_handler("chops.create_chop_node", _create_chop_node)


###### chops.list_chop_channels


def _track_entry(track: Any) -> dict[str, Any]:
    """Name, sample count, rate and value range of one CHOP track."""
    samples = track.allSamples()
    return {
        "name": track.name(),
        "length": track.numSamples(),
        "rate": track.clip().sampleRate(),
        "min": min(samples) if samples else 0.0,
        "max": max(samples) if samples else 0.0,
    }


def _list_chop_channels(*, node_path: str, **_: Any) -> dict[str, Any]:
    """List all tracks/channels on a CHOP node.

    For each channel returns: name, length, sample_rate, and value range.

    Args:
        node_path: Path to the CHOP node.
    """
    node = _get_node(node_path)

    if node.type().category().name() != "Chop":
        raise ValueError(f"Node is not a CHOP node: {node_path}")

    channels = [_track_entry(track) for track in node.tracks()]

    return {
        "node_path": node_path,
        "count": len(channels),
        "channels": channels,
    }


register_handler("chops.list_chop_channels", _list_chop_channels)


###### chops.export_chop_to_parm


def _export_chop_to_parm(
    *,
    chop_path: str,
    channel_name: str,
    target_node_path: str,
    target_parm_name: str,
    **_: Any,
) -> dict[str, Any]:
    """Create a CHOP export reference on a parameter.

    Sets a chop() expression on the target parameter that references
    the specified CHOP channel.

    Args:
        chop_path: Path to the CHOP node.
        channel_name: Name of the channel to export.
        target_node_path: Path to the target node containing the parameter.
        target_parm_name: Name of the parameter to set the expression on.
    """
    # Validate the CHOP node and channel
    chop_node = _get_node(chop_path)
    track = chop_node.track(channel_name)
    if track is None:
        names = [t.name() for t in chop_node.tracks()]
        shown = names[:30] + (["..."] if len(names) > 30 else [])
        raise ValueError(
            f"Channel '{channel_name}' not found on CHOP node: {chop_path}. Channels: {shown}"
        )

    # Validate the target parameter
    target_node = _get_node(target_node_path)
    parm = target_node.parm(target_parm_name)
    if parm is None:
        from fxhoudinimcp_server.handlers.parameter_handlers import parm_labels, suggest_parms

        close = suggest_parms(target_parm_name, parm_labels(target_node))
        hint = f" Did you mean: {close}?" if close else ""
        raise ValueError(
            f"Parameter '{target_parm_name}' not found on node: {target_node_path}.{hint}"
        )

    from fxhoudinimcp_server.handlers.parameter_handlers import set_whole_expression

    expression = f'chop("{chop_path}/{channel_name}")'
    replaced_keys = set_whole_expression(parm, expression, hou.exprLanguage.Hscript)

    result: dict[str, Any] = {
        "expression": expression,
        "target_parm": f"{target_node_path}/{target_parm_name}",
        "value": parm.eval(),
    }
    if replaced_keys:
        result["replaced_keyframes"] = replaced_keys
    return result


register_handler("chops.export_chop_to_parm", _export_chop_to_parm)
