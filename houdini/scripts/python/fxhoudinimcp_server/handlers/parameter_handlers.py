"""Houdini-side handlers for parameter operations.

Provides 14 command handlers for reading, writing, and managing
node parameters, expressions, channel references, and spare parameters.
"""

from __future__ import annotations

import contextlib
import re

# Built-in
from difflib import get_close_matches
from typing import Any

# Third-party
import hou

# Internal
from fxhoudinimcp_server.dispatcher import register_handler
from fxhoudinimcp_server.handlers.state_receipt_helpers import (
    clear_channels,
    clip,
    component_type_error,
    parm_state,
    raw_differs,
    resolve_expression_policy,
    set_parm_value,
    tuple_length_error,
    value_receipt,
    write_refusal,
)
from fxhoudinimcp_server.serialize import geometry_summary

###### Helpers


def _resolve_node(node_path: str) -> hou.Node:
    """Return the hou.Node at *node_path* or raise."""
    node = hou.node(node_path)
    if node is None:
        raise ValueError(f"Node not found: {node_path}")
    return node


def _available_parm_names(node: hou.Node) -> list[str]:
    """Return sorted list of parameter names on a node."""
    return sorted(p.name() for p in node.parms())


def _resolve_parm(node_path: str, parm_name: str) -> hou.Parm:
    """Return the hou.Parm on *node_path* named *parm_name* or raise."""
    node = _resolve_node(node_path)
    parm = node.parm(parm_name)
    if parm is None:
        available = _available_parm_names(node)
        close = get_close_matches(parm_name, available, n=3, cutoff=0.4)
        hint = f" Did you mean: {close}?" if close else ""
        raise ValueError(
            f"Parameter '{parm_name}' not found on node '{node_path}'.{hint} "
            f"Available parameters: {available}"
        )
    return parm


def _parm_type_name(parm_template: hou.ParmTemplate) -> str:
    """Return a human-readable type string for a parameter template."""
    return parm_template.type().name()


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-safe Python type."""
    if isinstance(value, hou.Vector2):
        return list(value)
    if isinstance(value, hou.Vector3):
        return list(value)
    if isinstance(value, hou.Vector4):
        return list(value)
    if isinstance(value, hou.Matrix3):
        return [list(row) for row in value.asTupleOfTuples()]
    if isinstance(value, hou.Matrix4):
        return [list(row) for row in value.asTupleOfTuples()]
    if isinstance(value, hou.Ramp):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    return value


def _data_parm_summary(parm: hou.Parm, pt: hou.ParmTemplate) -> dict[str, Any]:
    """What a Data parameter holds, never the blob itself.

    Counts come from serialize.geometry_summary (intrinsics), not from
    parm.asData(): that serialises the whole geometry to a string on the main
    thread, which is ruinous on a heavy stash and measures characters, not
    bytes.
    """
    summary: dict[str, Any] = {"is_set": False}
    with contextlib.suppress(Exception):
        summary["data_parm_type"] = pt.dataParmType().name()
    value = None
    with contextlib.suppress(Exception):
        value = parm.eval()
    if isinstance(value, hou.Geometry):
        # Set, possibly to an empty geometry: the counts say which.
        summary["is_set"] = True
        summary["geometry"] = geometry_summary(value)
    elif isinstance(value, dict):
        # A KeyValueDictionary evaluates to {} when empty.
        summary["is_set"] = bool(value)
        summary["key_count"] = len(value)
    elif value is not None:
        summary["is_set"] = True
        summary["value_type"] = type(value).__name__
    return summary


def _data_parm_value(parm: hou.Parm, pt: hou.ParmTemplate) -> dict[str, Any] | None:
    """For a Data parameter, the fields a reader reports; None for any other.

    eval() on a Data parameter is a hou.Geometry or None and rawValue() is
    empty either way, so `value` alone cannot tell "unset" from "set to an
    empty geometry". `value` stays the geometry summary it always serialised
    to; `data` says whether the blob is set and what it holds.
    """
    if _parm_type_name(pt) != "Data":
        return None
    data = _data_parm_summary(parm, pt)
    value = data.get("geometry")
    if "key_count" in data:
        # A KeyValueDictionary always answered with the dictionary itself.
        with contextlib.suppress(Exception):
            value = _serialize_value(parm.eval())
    return {"value": value, "data": data}


def _template_to_dict(pt: hou.ParmTemplate) -> dict[str, Any]:
    """Convert a ParmTemplate to a JSON-serialisable dictionary."""
    info: dict[str, Any] = {
        "name": pt.name(),
        "label": pt.label(),
        "type": _parm_type_name(pt),
        "num_components": pt.numComponents(),
        "is_hidden": pt.isHidden(),
    }

    # Default value
    try:
        info["default_value"] = list(pt.defaultValue())
    except Exception:
        try:
            info["default_value"] = pt.defaultValue()
        except Exception:
            info["default_value"] = None

    # Range
    try:
        info["min"] = pt.minValue()
        info["max"] = pt.maxValue()
        info["min_is_strict"] = pt.minIsStrict()
        info["max_is_strict"] = pt.maxIsStrict()
    except Exception:
        pass

    # Menu items — capped at 50 to avoid enormous enum lists
    try:
        items = pt.menuItems()
        labels = pt.menuLabels()
        if items:
            info["menu_items"] = list(items)[:50]
            info["menu_labels"] = list(labels)[:50]
            if len(items) > 50:
                info["menu_items_truncated"] = True
    except Exception:
        pass

    # Naming scheme (for multi-component parms)
    with contextlib.suppress(Exception):
        info["naming_scheme"] = pt.namingScheme().name()

    # Conditionals and tags omitted — internal Houdini UI metadata,
    # not useful for LLM-driven parameter setting.

    return info


###### Handler: parameters.get_parameter


def _get_parameter(node_path: str, parm_name: str, **_: Any) -> dict[str, Any]:
    """Get the current value, expression, keyframe info, and metadata of a parameter."""
    parm = _resolve_parm(node_path, parm_name)
    pt = parm.parmTemplate()

    data_parm = _data_parm_value(parm, pt)
    result: dict[str, Any] = {
        "node_path": node_path,
        "parm_name": parm_name,
        "value": data_parm["value"] if data_parm else _serialize_value(parm.eval()),
        "raw_value": _serialize_value(parm.rawValue()),
        "parm_type": _parm_type_name(pt),
        "is_locked": parm.isLocked(),
        "is_at_default": parm.isAtDefault(),
    }
    if data_parm:
        result["data"] = data_parm["data"]

    # Expression
    try:
        result["expression"] = parm.expression()
        result["expression_language"] = parm.expressionLanguage().name()
    except hou.OperationFailed:
        result["expression"] = None
        result["expression_language"] = None

    # Keyframes
    keyframes = parm.keyframes()
    result["keyframe_count"] = len(keyframes)

    return result


register_handler("parameters.get_parameter", _get_parameter)


###### Writing values: the core both set_parameter and set_parameters run on
#
# The two used to be separate implementations, and behaved differently in ways
# nobody chose: the single write understood a list as a vector and the batch
# did not, so the same request succeeded or failed depending on which command
# carried it. Everything below is shared, so they cannot drift again.

# Above this many characters a value is reported as length, hash and samples
# rather than in full. Parameter values are normally three floats; the ones
# that are not are file lists and VEX snippets, and a receipt that quietly
# clips one of those is how a caller concludes a write did something it did not.
_VALUE_CHARS = 200

# Expressions and exception text go into the receipt as excerpts. Both can be
# arbitrarily long -- a snippet-driven parameter, an exception whose message
# quotes the value that caused it -- and neither is worth an unbounded receipt.
_EXPRESSION_CHARS = 400

_RETURN_MODES = ("summary", "full", "none")


def _resolve_return_values(mode: Any) -> str:
    """The return_values token, or a message naming the ones that exist."""
    if mode is None:
        return "summary"
    if isinstance(mode, str) and mode.lower() in _RETURN_MODES:
        return mode.lower()
    raise ValueError(f"return_values must be one of {list(_RETURN_MODES)}, not {mode!r}")


def _address(node: hou.Node, parm_name: str, value: Any) -> tuple:
    """(components, is_tuple, error) for the thing this write addresses.

    A list addresses a parm tuple and nothing else. A scalar addresses a
    parameter, or broadcasts across a tuple when the name is the tuple's —
    which is what build_network has always done and what a caller writing
    ``{"t": 0}`` plainly means.
    """
    available = _available_parm_names(node)
    if isinstance(value, (list, tuple)):
        parm_tuple = node.parmTuple(parm_name)
        if parm_tuple is None:
            parm = node.parm(parm_name)
            if parm is not None:
                if len(value) != 1:
                    return (), False, tuple_length_error(parm_name, 1, len(value))
                return (parm,), False, None
            close = get_close_matches(parm_name, available, n=3, cutoff=0.4)
            hint = f" Did you mean: {close}?" if close else ""
            return (), False, f"Parameter '{parm_name}' not found.{hint}"
        if len(value) != len(parm_tuple):
            return (), True, tuple_length_error(parm_name, len(parm_tuple), len(value))
        return tuple(parm_tuple), True, None

    parm = node.parm(parm_name)
    if parm is not None:
        return (parm,), False, None
    parm_tuple = node.parmTuple(parm_name)
    if parm_tuple is not None:
        return tuple(parm_tuple), True, None
    close = get_close_matches(parm_name, available, n=3, cutoff=0.4)
    hint = f" Did you mean: {close}?" if close else ""
    return (), False, f"Parameter '{parm_name}' not found.{hint}"


def _matches(requested: Any, actual: Any) -> bool | None:
    """Whether the node ended up with what was asked for, or None if unknowable.

    The motivating case: writing 0 to a pivot that carries ``$CEX`` reported
    success and the parameter went on evaluating to the expression's answer.
    Comparing the read-back to the request is what makes that visible.
    """
    if isinstance(requested, (list, tuple)) or isinstance(actual, (list, tuple)):
        if not isinstance(requested, (list, tuple)) or not isinstance(actual, (list, tuple)):
            return None
        if len(requested) != len(actual):
            return False
        verdicts = [_matches(a, b) for a, b in zip(requested, actual, strict=True)]
        return None if None in verdicts else all(verdicts)
    if isinstance(requested, bool) or isinstance(actual, bool):
        if isinstance(requested, (bool, int)) and isinstance(actual, (bool, int)):
            return bool(requested) == bool(actual)
        return None
    if isinstance(requested, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(requested) - float(actual)) <= 1e-6 * max(1.0, abs(float(requested)))
    if isinstance(requested, str) and isinstance(actual, str):
        return requested == actual
    # A number written to a string parameter, or a menu token read back as an
    # index: the write may be perfectly correct, and saying so would be a guess.
    return None


def _read_back(components: tuple, is_tuple: bool, mode: str, max_chars: int) -> dict[str, Any]:
    """What the node evaluates to now, plus the raw text behind it."""
    states = [parm_state(parm) for parm in components]
    values = [_serialize_value(state["value"]) for state in states]
    raws = [state["raw_value"] for state in states]
    expressions = [state["expression"] for state in states]

    entry: dict[str, Any] = {}
    actual: Any = values if is_tuple else values[0]
    receipt = value_receipt(actual, max_chars, full=mode == "full")
    if receipt["complete"]:
        entry["new_value"] = receipt["value"]
    else:
        # Deliberately no "new_value" key: a clipped value under the name a
        # caller compares against is worse than no value at all.
        entry["new_value_summary"] = {k: v for k, v in receipt.items() if k != "value"}

    # Raw text only when it says something the value does not: an expression,
    # or a literal like "$HIP/x.bgeo" that the value shows already expanded.
    differs = any(raw_differs(raw, value) for raw, value in zip(raws, values, strict=True))
    if differs:
        raw_receipt = value_receipt(raws if is_tuple else raws[0], max_chars, full=mode == "full")
        if raw_receipt["complete"]:
            entry["raw_value"] = raw_receipt["value"]
        else:
            entry["raw_value_summary"] = {k: v for k, v in raw_receipt.items() if k != "value"}
        if not any(expressions):
            # No expression, yet the raw text differs: Houdini expanded a
            # variable. Calling that an expression sent callers looking for one.
            entry["value_is_expanded"] = True
    if any(expressions):
        # Clipped for the same reason the value is: an expression can be a
        # whole VEX snippet, and three components of one triple the bill. The
        # clip marker carries the real length and a hash of the whole thing.
        clipped = [clip(item, _EXPRESSION_CHARS) if isinstance(item, str) else item
                   for item in expressions]
        entry["expression"] = clipped if is_tuple else clipped[0]
        if any(isinstance(item, str) and len(item) > _EXPRESSION_CHARS for item in expressions):
            entry["expression_truncated"] = True
        entry["expression_language"] = next(
            (state["expression_language"] for state in states if state["expression"]), None
        )
    keyframes = sum(state.get("keyframes") or 0 for state in states)
    if keyframes:
        entry["keyframes"] = keyframes
    eval_errors = [state["eval_error"] for state in states if state.get("eval_error")]
    if eval_errors:
        entry["eval_errors"] = eval_errors
    return entry


def _write_parm(
    node: hou.Node,
    parm_name: str,
    value: Any,
    policy: str,
    mode: str,
    max_chars: int,
) -> tuple:
    """Apply one value. Returns (entry, error): exactly one of them is None.

    Every check that can be made without touching the node is made first, so a
    refusal leaves the parameter exactly as it was.
    """
    components, is_tuple, error = _address(node, parm_name, value)
    if error:
        return None, error

    label = f"{node.path()}/{parm_name}"
    requested = list(value) if isinstance(value, (list, tuple)) else [value] * len(components)

    # Both pre-flight passes run over every component before any of them is
    # touched, because "replace" deletes an expression before it writes:
    # finding out at set() time that the value was a dictionary would leave
    # the parameter stripped of what drove it and holding nothing new.
    for index, parm in enumerate(components):
        state = parm_state(parm)
        component_label = label if len(components) == 1 else f"{label}[{index}]"
        refusal = write_refusal(state, policy, component_label)
        if refusal:
            return None, refusal
    for index, (parm, component) in enumerate(zip(components, requested, strict=True)):
        component_label = label if len(components) == 1 else f"{label}[{index}]"
        type_error = component_type_error(parm, component, component_label)
        if type_error:
            return None, type_error

    entry: dict[str, Any] = {"parm_name": parm_name}
    cleared: list[dict[str, Any]] = []
    if policy == "replace":
        for parm in components:
            outcome = clear_channels(parm)
            if outcome.get("clear_error"):
                # The channel is still there. Writing now would go through the
                # reference to whatever node it points at, so this write stops
                # here and says which component refused to be cleared.
                entry["cleared"] = cleared
                return entry, (
                    f"'{parm_name}' ({parm.name()}): could not clear the existing "
                    f"expression or keyframes, so nothing was written: "
                    f"{outcome['clear_error']}"
                )
            if outcome.get("cleared_expression") or outcome.get("cleared_keyframes"):
                cleared.append(outcome)

    # Component by component rather than ParmTuple.set(), so a failure names
    # the component it happened on and the ones already written are reported
    # as written rather than lost in one opaque exception.
    applied = 0
    write_error: str | None = None
    for parm, component in zip(components, requested, strict=True):
        try:
            set_parm_value(parm, component)
            applied += 1
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            # An exception's message can quote the value that caused it, so it
            # is as unbounded as the value was.
            detail = clip(str(exc), _EXPRESSION_CHARS)
            write_error = (
                f"'{parm_name}' component {applied} ({parm.name()}): {detail}"
                if len(components) > 1
                else f"'{parm_name}': {detail}"
            )
            break

    if mode != "none":
        entry.update(_read_back(components, is_tuple, mode, max_chars))
        if isinstance(value, (list, tuple)):
            asked: Any = list(value)
        elif is_tuple:
            asked = [value] * len(components)
        else:
            asked = value
        verdict = _matches(asked, entry.get("new_value"))
        if verdict is False:
            entry["matches_requested"] = False
            asked_receipt = value_receipt(asked, max_chars, full=mode == "full")
            entry["requested"] = (
                asked_receipt["value"]
                if asked_receipt["complete"]
                else {k: v for k, v in asked_receipt.items() if k != "value"}
            )
        elif verdict is True:
            entry["matches_requested"] = True
    if cleared:
        entry["expression_cleared"] = True
        entry["cleared"] = cleared
    if write_error:
        entry["applied_components"] = applied
        return entry, write_error
    return entry, None


###### Handler: parameters.set_parameter


def _expression_driven(parms: list[hou.Parm]) -> str | None:
    """A sentence naming the first of *parms* driven by an expression, or None.

    Houdini answers a set() on such a parameter with a generic permission
    error ("locked assets, takes, product permissions..."); resolutiony on a
    karmarendersettings is the common case, driven by the autoheight expression
    while res_mode says so (#41).
    """
    for parm in parms:
        try:
            expression = parm.expression()
        except hou.OperationFailed:
            continue
        return (
            f"'{parm.name()}' on {parm.node().path()} is driven by the expression "
            f"{expression!r}, so it cannot be set directly. Set the parameter that "
            f"controls it, or remove the expression with revert_parameter first."
        )
    return None


def _set_tuple(node: hou.Node, parm_name: str, value: list | tuple) -> Any | None:
    """Apply a list value to the parm tuple of that name; None if there is none.

    A list/tuple value addressed at a vector parameter name (e.g. "size" on a
    box, "t" on a transform, a light's colour) is applied to the whole parm
    tuple, so callers are not forced to know the per-component names.
    """
    parm_tuple = node.parmTuple(parm_name)
    if parm_tuple is None:
        return None
    if len(value) != len(parm_tuple):
        raise ValueError(
            f"Parameter '{parm_name}' on {node.path()} has "
            f"{len(parm_tuple)} components, got {len(value)} values."
        )
    try:
        parm_tuple.set(value)
    except hou.PermissionError:
        reason = _expression_driven(list(parm_tuple))
        if reason:
            raise ValueError(reason) from None
        raise
    return [_serialize_value(p.eval()) for p in parm_tuple]


def _set_parameter(
    node_path: str,
    parm_name: str,
    value: Any,
    expression_policy: str = "preserve",
    return_values: str = "summary",
    max_value_chars: int = _VALUE_CHARS,
    **_: Any,
) -> dict[str, Any]:
    """Set one parameter and report what the node actually evaluates afterwards.

    A list/tuple value addressed at a vector parameter name (e.g. "size" on a
    box, "t" on a transform) is applied to the whole parm tuple, so callers are
    not forced to know the per-component names (sizex, ...). Scalars broadcast
    across a tuple.

    Same rules as the batch write, with one deliberate difference in how the
    answer arrives: with a single parameter asked for, a failure is the whole
    result, so it is raised rather than buried in an errors list a caller has
    to remember to read.
    """
    result = _set_parameters(
        node_path,
        {parm_name: value},
        expression_policy=expression_policy,
        return_values=return_values,
        max_value_chars=max_value_chars,
        _single=parm_name,
    )
    if result["errors"]:
        message = result["errors"][0]["error"]
        applied = (result["set"][0] if result["set"] else {}).get("applied_components")
        if applied:
            message = f"{message} ({applied} component(s) were written before the failure)"
        raise ValueError(message)
    return result


register_handler("parameters.set_parameter", _set_parameter)


###### Handler: parameters.set_parameters


def _set_parameters(
    node_path: str,
    params: dict[str, Any],
    expression_policy: str = "preserve",
    return_values: str = "summary",
    max_value_chars: int = _VALUE_CHARS,
    _single: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Batch-set parameters on one node, with the same rules as the single write.

    expression_policy decides what happens when the parameter is already driven
    by something. "preserve" (the default) refuses rather than overwriting, and
    names the expression in the refusal. "replace" clears the keyframes and
    expression ON THE ADDRESSED COMPONENT first, then writes the literal; a
    channel reference is cleared where it is written and never followed to the
    parameter it points at, which belongs to a node the caller did not name.

    Arguments are validated before anything is written. Per-parameter failures
    do not stop the rest — they are reported, and success is false if any
    occurred.
    """
    policy = resolve_expression_policy(expression_policy)
    mode = _resolve_return_values(return_values)
    if not isinstance(max_value_chars, int) or isinstance(max_value_chars, bool):
        raise ValueError(f"max_value_chars must be an integer, got {max_value_chars!r}")
    if max_value_chars < 1:
        raise ValueError(f"max_value_chars must be at least 1, got {max_value_chars}")
    if not isinstance(params, dict) or not params:
        raise ValueError("'params' must be a non-empty mapping of parameter names to values")

    node = _resolve_node(node_path)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for name, value in params.items():
        # A list on a vector name sets the whole tuple, exactly as the single
        # setter does: _address resolves the name to its components, so the
        # batch path is not limited to per-component names either.
        entry, error = _write_parm(node, name, value, policy, mode, max_value_chars)
        if entry is not None:
            results.append(entry)
        if error:
            errors.append({"parm_name": name, "error": error})

    if mode == "none":
        # "none" asks for a count, not a list: echoing every name back cost a
        # 900-parameter Add write about fifteen thousand tokens. An entry
        # stays only when it says more than its name -- an expression this
        # write cleared, or components written before a failure.
        failed = {error["parm_name"] for error in errors}
        written = sum(1 for entry in results if entry["parm_name"] not in failed)
        results = [entry for entry in results if len(entry) > 1]
    result: dict[str, Any] = {
        "node_path": node_path,
        "success": not errors,
        "set": results,
        "errors": errors,
        "expression_policy": policy,
        # Evaluation is frame-dependent, so a value read back without saying
        # when it was read is not a fact about the parameter.
        "context": _eval_context(),
    }
    if mode == "none":
        result["set_count"] = written
    if _single is not None:
        # The single write predates the batch and callers read new_value off
        # the top level; keeping that promise costs one key.
        result["parm_name"] = _single
        first = results[0] if results else {}
        if "new_value" in first:
            result["new_value"] = first["new_value"]
        elif "new_value_summary" in first:
            result["new_value_summary"] = first["new_value_summary"]
        if errors:
            result["error"] = errors[0]["error"]
    return result


def _eval_context() -> dict[str, Any]:
    """When the read-back values were evaluated."""
    context: dict[str, Any] = {}
    for key, reader in (("frame", hou.frame), ("time", hou.time), ("fps", hou.fps)):
        with contextlib.suppress(Exception):
            context[key] = reader()
    return context


register_handler("parameters.set_parameters", _set_parameters)


###### Handler: parameters.get_parameter_schema


def _get_parameter_schema(
    node_path: str,
    parm_name: str | None = None,
    filter: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Get full parameter template info.

    If *parm_name* is given, return info for that one parameter.
    If *filter* is given, return only parameters whose name or label
    contains the filter string (case-insensitive).
    Otherwise return all non-hidden parameters.
    """
    node = _resolve_node(node_path)

    if parm_name is not None:
        parm = node.parm(parm_name)
        if parm is None:
            return {
                "node_path": node_path,
                "error": f"Parameter '{parm_name}' not found",
                "available_parameters": _available_parm_names(node),
            }
        return {
            "node_path": node_path,
            "parameter": _template_to_dict(parm.parmTemplate()),
        }

    # All parameters — hidden params skipped to keep the response compact.
    ptg = node.parmTemplateGroup()
    parm_infos: list[dict[str, Any]] = []

    def _walk(entries: tuple) -> None:
        for entry in entries:
            if isinstance(entry, hou.FolderParmTemplate):
                _walk(entry.parmTemplates())
            elif not entry.isHidden():
                parm_infos.append(_template_to_dict(entry))

    _walk(ptg.parmTemplates())

    if filter:
        f = filter.lower()
        parm_infos = [p for p in parm_infos if f in p["name"].lower() or f in p["label"].lower()]

    return {
        "node_path": node_path,
        "parameter_count": len(parm_infos),
        "parameters": parm_infos,
    }


register_handler("parameters.get_parameter_schema", _get_parameter_schema)


###### Handler: parameters.set_expression


def _set_expression(
    node_path: str,
    parm_name: str,
    expression: str,
    language: str = "hscript",
    **_: Any,
) -> dict[str, Any]:
    """Set an expression on a parameter."""
    parm = _resolve_parm(node_path, parm_name)

    lang = hou.exprLanguage.Python if language.lower() == "python" else hou.exprLanguage.Hscript

    parm.setExpression(expression, lang)

    return {
        "node_path": node_path,
        "parm_name": parm_name,
        "expression": expression,
        "language": language,
    }


register_handler("parameters.set_expression", _set_expression)


###### Handler: parameters.get_expression


def _get_expression(node_path: str, parm_name: str, **_: Any) -> dict[str, Any]:
    """Get the current expression on a parameter."""
    parm = _resolve_parm(node_path, parm_name)

    try:
        expr = parm.expression()
        lang = parm.expressionLanguage().name()
    except hou.OperationFailed:
        expr = None
        lang = None

    return {
        "node_path": node_path,
        "parm_name": parm_name,
        "expression": expr,
        "language": lang,
    }


register_handler("parameters.get_expression", _get_expression)


###### Handler: parameters.revert_parameter


def _revert_parameter(node_path: str, parm_name: str, **_: Any) -> dict[str, Any]:
    """Revert a parameter to its default value."""
    parm = _resolve_parm(node_path, parm_name)

    parm.revertToDefaults()

    return {
        "node_path": node_path,
        "parm_name": parm_name,
        "reverted": True,
        "value": _serialize_value(parm.eval()),
    }


register_handler("parameters.revert_parameter", _revert_parameter)


###### Handler: parameters.link_parameters


def _channel_function(parm: hou.Parm) -> str:
    """The HScript channel function that reads *parm* as its own type.

    ``ch()`` evaluates the referenced channel as a number, so a String parameter
    linked with it reads back as "0". Strings need ``chs()``; every
    numeric kind (float, int, toggle, int-valued menu) reads with ``ch()``.
    """
    try:
        kind = parm.parmTemplate().type()
    except Exception:
        return "ch"
    return "chs" if kind == hou.parmTemplateType.String else "ch"


def _relative_channel_path(dst: hou.Parm, src: hou.Parm) -> str:
    """Path to *src* as *dst* would write it: relative, the way Houdini's own
    Paste Relative References does.

    An absolute path (``ch("/obj/gaps/CONTROL/mat")``) breaks the moment the
    pair is moved, collapsed into a subnet or saved into an HDA and instanced
    elsewhere; a relative one survives all three.
    """
    rel = dst.node().relativePathTo(src.node())
    if rel in ("", "."):
        return src.name()
    return f"{rel}/{src.name()}"


_CH_REF = re.compile(r"""\bch[fis]?\(\s*["']([^"']+)["']\s*\)""")


def _reaches(parm: hou.Parm, goal: str, seen: set[str]) -> bool:
    """True when *parm* reads *goal* through static HScript ch() references.

    Only literal ``ch("path")`` calls are followed; Python or computed
    references cannot be validated and are treated as leaves.
    """
    if parm.path() == goal:
        return True
    if parm.path() in seen:
        return False
    seen.add(parm.path())
    for key in parm.keyframes():
        with contextlib.suppress(hou.OperationFailed):
            for ref in _CH_REF.findall(key.expression()):
                dep = parm.node().parm(ref)
                if dep is not None and _reaches(dep, goal, seen):
                    return True
    return False


def _link_parameters(
    source_path: str,
    source_parm: str,
    dest_path: str,
    dest_parm: str,
    replace_existing: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Create a channel reference from destination parameter to source parameter.

    The expression uses ``chs()`` for a String destination and ``ch()`` for
    everything else, and a path relative to the destination node. The reply
    reads the destination back so the caller sees the linked value, not just
    the expression text.

    A destination that already has keyframes or an expression is refused
    unless *replace_existing* is set, and a link whose source already reads
    the destination through static ch() references is refused as a cycle.
    """
    src = _resolve_parm(source_path, source_parm)
    dst = _resolve_parm(dest_path, dest_parm)

    if dst.keyframes() and not replace_existing:
        raise ValueError(
            f"{dst.path()} already has animation or an expression; "
            "pass replace_existing=True to overwrite it."
        )
    if _reaches(src, dst.path(), set()):
        raise ValueError(f"Linking {dst.path()} to {src.path()} would create a channel cycle.")

    function = _channel_function(dst)
    channel_path = _relative_channel_path(dst, src)
    ref_expr = f'{function}("{channel_path}")'
    dst.setExpression(ref_expr, hou.exprLanguage.Hscript)

    reply: dict[str, Any] = {
        "source": src.path(),
        "destination": dst.path(),
        "expression": ref_expr,
        "function": function,
        "relative": not channel_path.startswith("/"),
    }
    with contextlib.suppress(Exception):
        reply["value"] = _serialize_value(dst.eval())
    src_kind = _channel_function(src)
    if src_kind != function:
        reply["warning"] = (
            f"'{dst.name()}' is a {'String' if function == 'chs' else 'numeric'} parameter "
            f"linked to a {'String' if src_kind == 'chs' else 'numeric'} source; "
            f"{function}() converts the value on read."
        )
    return reply


register_handler("parameters.link_parameters", _link_parameters)


###### Handler: parameters.lock_parameter


def _lock_parameter(node_path: str, parm_name: str, locked: bool, **_: Any) -> dict[str, Any]:
    """Lock or unlock a parameter."""
    parm = _resolve_parm(node_path, parm_name)

    parm.lock(locked)

    return {
        "node_path": node_path,
        "parm_name": parm_name,
        "locked": parm.isLocked(),
    }


register_handler("parameters.lock_parameter", _lock_parameter)


###### Handler: parameters.create_spare_parameter


def _create_spare_parameter(
    node_path: str,
    parm_name: str,
    parm_type: str,
    label: str,
    default_value: Any = None,
    min_val: float | None = None,
    max_val: float | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Add a custom spare parameter to a node."""
    node = _resolve_node(node_path)

    # Map string type names to ParmTemplate constructors
    type_map: dict[str, type] = {
        "float": hou.FloatParmTemplate,
        "int": hou.IntParmTemplate,
        "string": hou.StringParmTemplate,
        "toggle": hou.ToggleParmTemplate,
        "menu": hou.MenuParmTemplate,
    }

    template_cls = type_map.get(parm_type.lower())
    if template_cls is None:
        raise ValueError(
            f"Unsupported parm_type '{parm_type}'. Supported types: {list(type_map.keys())}"
        )

    # Build keyword arguments for the template constructor
    kwargs: dict[str, Any] = {}

    if template_cls in (hou.FloatParmTemplate, hou.IntParmTemplate):
        # Cast to the correct numeric type for the template
        _cast = int if template_cls is hou.IntParmTemplate else float

        # These require num_components; default to 1
        if default_value is not None:
            if not isinstance(default_value, (list, tuple)):
                default_value = [default_value]
            kwargs["num_components"] = len(default_value)
            kwargs["default_value"] = tuple(_cast(v) for v in default_value)
        else:
            kwargs["num_components"] = 1

        if min_val is not None:
            kwargs["min"] = _cast(min_val)
            kwargs["min_is_strict"] = False
        if max_val is not None:
            kwargs["max"] = _cast(max_val)
            kwargs["max_is_strict"] = False

        pt = template_cls(parm_name, label, **kwargs)

    elif template_cls is hou.StringParmTemplate:
        kwargs["num_components"] = 1
        if default_value is not None:
            if not isinstance(default_value, (list, tuple)):
                default_value = [default_value]
            kwargs["default_value"] = tuple(str(v) for v in default_value)
        pt = template_cls(parm_name, label, **kwargs)

    elif template_cls is hou.ToggleParmTemplate:
        dv = bool(default_value) if default_value is not None else False
        pt = template_cls(parm_name, label, default_value=dv)

    elif template_cls is hou.MenuParmTemplate:
        # For menu type, default_value should be a list of menu items
        items = default_value if isinstance(default_value, (list, tuple)) else []
        pt = template_cls(
            parm_name,
            label,
            menu_items=tuple(str(i) for i in items),
            menu_labels=tuple(str(i) for i in items),
        )
    else:
        pt = template_cls(parm_name, label, **kwargs)

    # Add to node
    ptg = node.parmTemplateGroup()
    ptg.addParmTemplate(pt)
    node.setParmTemplateGroup(ptg)

    return {
        "node_path": node_path,
        "parm_name": parm_name,
        "parm_type": parm_type,
        "label": label,
        "created": True,
    }


register_handler("parameters.create_spare_parameter", _create_spare_parameter)


###### Handler: parameters.create_spare_parameters


def _build_parm_template(spec: dict) -> hou.ParmTemplate:
    """Build a single ParmTemplate from a specification dict."""
    type_map: dict[str, type] = {
        "float": hou.FloatParmTemplate,
        "int": hou.IntParmTemplate,
        "string": hou.StringParmTemplate,
        "toggle": hou.ToggleParmTemplate,
        "menu": hou.MenuParmTemplate,
    }

    parm_name = spec["parm_name"]
    parm_type = spec["parm_type"].lower()
    label = spec["label"]
    default_value = spec.get("default_value")
    min_val = spec.get("min_val")
    max_val = spec.get("max_val")

    template_cls = type_map.get(parm_type)
    if template_cls is None:
        raise ValueError(f"Unsupported parm_type '{parm_type}' for parameter '{parm_name}'.")

    kwargs: dict[str, Any] = {}

    if template_cls in (hou.FloatParmTemplate, hou.IntParmTemplate):
        _cast = int if template_cls is hou.IntParmTemplate else float
        if default_value is not None:
            if not isinstance(default_value, (list, tuple)):
                default_value = [default_value]
            kwargs["num_components"] = len(default_value)
            kwargs["default_value"] = tuple(_cast(v) for v in default_value)
        else:
            kwargs["num_components"] = 1
        if min_val is not None:
            kwargs["min"] = _cast(min_val)
            kwargs["min_is_strict"] = False
        if max_val is not None:
            kwargs["max"] = _cast(max_val)
            kwargs["max_is_strict"] = False
        return template_cls(parm_name, label, **kwargs)

    if template_cls is hou.StringParmTemplate:
        kwargs["num_components"] = 1
        if default_value is not None:
            if not isinstance(default_value, (list, tuple)):
                default_value = [default_value]
            kwargs["default_value"] = tuple(str(v) for v in default_value)
        return template_cls(parm_name, label, **kwargs)

    if template_cls is hou.ToggleParmTemplate:
        dv = bool(default_value) if default_value is not None else False
        return template_cls(parm_name, label, default_value=dv)

    if template_cls is hou.MenuParmTemplate:
        items = default_value if isinstance(default_value, (list, tuple)) else []
        return template_cls(
            parm_name,
            label,
            menu_items=tuple(str(i) for i in items),
            menu_labels=tuple(str(i) for i in items),
        )

    return template_cls(parm_name, label, **kwargs)


_FOLDER_TYPE_MAP = {
    "Tabs": hou.folderType.Tabs,
    "tabs": hou.folderType.Tabs,
    "Collapsible": hou.folderType.Collapsible,
    "collapsible": hou.folderType.Collapsible,
    "Simple": hou.folderType.Simple,
    "simple": hou.folderType.Simple,
}


def _create_spare_parameters(
    node_path: str,
    parameters: list,
    folder_name: str | None = None,
    folder_type: str = "Tabs",
    **_: Any,
) -> dict[str, Any]:
    """Batch-create spare parameters, optionally inside a folder/tab."""
    node = _resolve_node(node_path)
    ptg = node.parmTemplateGroup()

    templates = []
    created = []
    updated = []
    for spec in parameters:
        pt = _build_parm_template(spec)
        existing = ptg.find(spec["parm_name"])
        if existing is None:
            # A tuple component such as "tx" is not a template name, so the
            # group does not know it, but Houdini refuses it at commit time.
            component = node.parm(spec["parm_name"])
            if component is not None:
                raise ValueError(
                    f"'{spec['parm_name']}' is a component of the existing tuple "
                    f"'{component.tuple().name()}'; pick another name."
                )
            templates.append(pt)
            created.append(spec["parm_name"])
            continue
        # Same name: edit in place. Houdini keeps the current value and
        # keyframes when the name and type are unchanged, so a type change
        # is refused rather than silently dropping data.
        current = node.parm(spec["parm_name"]) or node.parmTuple(spec["parm_name"])
        if current is not None and not (
            current.isSpare()
            if isinstance(current, hou.Parm)
            else all(p.isSpare() for p in current)
        ):
            raise ValueError(
                f"'{spec['parm_name']}' is a built-in parameter; only spare parameters can be edited."
            )
        if existing.type() != pt.type():
            raise ValueError(
                f"'{spec['parm_name']}' exists as {existing.type().name()}; "
                f"cannot change it to {pt.type().name()} without losing its value."
            )
        ptg.replace(spec["parm_name"], pt)
        updated.append(spec["parm_name"])

    if not templates:
        pass
    elif folder_name is not None:
        ft = _FOLDER_TYPE_MAP.get(folder_type, hou.folderType.Tabs)
        folder = hou.FolderParmTemplate(
            folder_name.lower().replace(" ", "_"),
            folder_name,
            parm_templates=templates,
            folder_type=ft,
        )
        ptg.addParmTemplate(folder)
    else:
        for pt in templates:
            ptg.addParmTemplate(pt)

    node.setParmTemplateGroup(ptg)

    return {
        "node_path": node_path,
        "created": created,
        "updated": updated,
        "count": len(created) + len(updated),
        "folder_name": folder_name,
    }


register_handler("parameters.create_spare_parameters", _create_spare_parameters)


###### parameters.get_parameters

_GET_PARMS_CAP = 60


def _get_parameters(
    node_path: str,
    patterns: list[str] | str | None = None,
    include_defaults: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Current values for every parameter matching any of several patterns.

    set_parameters has been batch from the start while reading stayed one parm
    per call, so checking five unrelated groups of settings cost five round
    trips. get_node_card reports names and defaults for a node *type*; this
    reports the live values on a specific node.

    Args:
        node_path: Node to read.
        patterns: Substrings matched against parameter name and label. Omit for
            every non-hidden parameter, up to the cap.
        include_defaults: Also report whether each value is still the default.
    """
    node = hou.node(node_path)
    if node is None:
        raise hou.OperationFailed(f"Node not found: {node_path}")

    if isinstance(patterns, str):
        patterns = [patterns]
    lowered = [p.lower() for p in patterns] if patterns else None

    values: dict[str, Any] = {}
    matched = 0
    for parm in node.parms():
        name = parm.name()
        if lowered is not None:
            label = parm.parmTemplate().label().lower()
            if not any(p in name.lower() or p in label for p in lowered):
                continue
        matched += 1
        if len(values) >= _GET_PARMS_CAP:
            continue
        data_parm = _data_parm_value(parm, parm.parmTemplate())
        entry: dict[str, Any] = data_parm or {"value": _serialize_value(parm.eval())}
        raw = parm.rawValue()
        # Only worth reporting when it differs: an expression is the thing a
        # caller most often needs to see and a literal is just noise.
        if not data_parm and isinstance(raw, str) and raw != str(entry["value"]):
            entry["raw_value"] = raw
        if include_defaults:
            entry["is_at_default"] = parm.isAtDefault()
        values[name] = entry

    return {
        "node_path": node_path,
        "node_type": node.type().name(),
        "patterns": patterns,
        "matched": matched,
        "returned": len(values),
        "truncated": matched > len(values),
        "parameters": values,
    }


register_handler("parameters.get_parameters", _get_parameters)


###### Handler: parameters.get_parm_references

# Every HScript function that reads a channel by path
# ($HFS/houdini/help/expressions.zip). Longest first, so chsop is not read
# as chs.
_CHANNEL_FUNCTIONS = (
    "ch", "chexist", "chexpr", "chexprf", "chexprt", "chf", "chramp", "chrampf",
    "chrampraw", "chrampt", "chs", "chsop", "chsoplist", "chsraw", "cht",
)  # fmt: skip
_CHANNEL_NAMES = "|".join(sorted(_CHANNEL_FUNCTIONS, key=len, reverse=True))
_CHANNEL_REF_RE = re.compile(r"\b(?:" + _CHANNEL_NAMES + r""")\s*\(\s*['"]([^'"]+)['"]""")
_BACKTICKS_RE = re.compile(r"`([^`]*)`")


def _outgoing_reference(parm: hou.Parm) -> dict[str, Any] | None:
    """What *parm* reads from, or None when it reads no other channel.

    A pure `ch("../src/tx")` resolves through getReferencedParm(). Anything
    richer -- `ch("../src/scale") * 2` -- answers with the parm itself there
    (measured on 22.0.429), so the channel references are read out of the
    expression text and resolved relative to the node. A string parameter's
    backtick expressions (`$HIP/`chs("../CTRL/version")`/geo.bgeo.sc`) are not
    an expression() at all; they are read from unexpandedString(). An
    expression that reads no channel (`$F * 2`) is not a reference.
    """
    expression = None
    with contextlib.suppress(Exception):
        expression = parm.expression()
    texts: list[str] = []
    in_backticks = False
    if expression:
        texts = [expression]
    else:
        with contextlib.suppress(Exception):
            raw = parm.unexpandedString()
            texts = _BACKTICKS_RE.findall(raw)
            if texts:
                expression, in_backticks = raw, True
    if not texts:
        return None
    entry: dict[str, Any] = {"parm": parm.name(), "expression": expression}
    if in_backticks:
        entry["in_backticks"] = True
    else:
        with contextlib.suppress(Exception):
            direct = parm.getReferencedParm()
            if direct is not None and direct.path() != parm.path():
                entry["references"] = [direct.path()]
                entry["pure_reference"] = True
                return entry
    resolved: list[str] = []
    unresolved: list[str] = []
    node = parm.node()
    for text in texts:
        for token in _CHANNEL_REF_RE.findall(text):
            target = None
            with contextlib.suppress(Exception):
                target = node.parm(token)
            if target is not None:
                resolved.append(target.path())
            else:
                unresolved.append(token)
    if not resolved and not unresolved:
        return None
    entry["references"] = resolved
    if unresolved:
        # Written in the expression, but no such parameter now: a renamed or
        # deleted target, which is exactly what a rename audit is after.
        entry["unresolved"] = unresolved
    entry["pure_reference"] = False
    return entry


def _capped_paths(nodes: Any, exclude: str, limit: int) -> tuple[list[str], int]:
    paths = sorted({n.path() for n in nodes} - {exclude})
    return paths[:limit], len(paths)


def _get_parm_references(
    node_path: str,
    parm_name: str | None = None,
    direction: str = "both",
    limit: int = 200,
    **_: Any,
) -> dict[str, Any]:
    """Who references a parameter, and what it references -- both directions.

    `incoming` lists, per parameter of *node_path* (or the one *parm_name*),
    the parameters elsewhere whose expressions read it (parmsReferencingThis).
    `outgoing` lists what this node's expressions and backtick strings read.
    Node-level `dependents` / `references` round it off, so "what breaks if I
    rename this control" is one call instead of a HOM script.
    """
    if direction not in ("both", "incoming", "outgoing"):
        raise ValueError("direction must be 'both', 'incoming' or 'outgoing'.")
    limit = int(limit)
    node = _resolve_node(node_path)
    parms = [_resolve_parm(node_path, parm_name)] if parm_name is not None else list(node.parms())

    dependents: list = []
    with contextlib.suppress(Exception):
        dependents = list(node.dependents(include_children=False))
    # parmsReferencingThis() walks the whole scene, once per parameter. A
    # reader through ch() or backticks makes its node a dependent (a
    # self-reference makes this node its own), so with no dependents there is
    # nothing to find and the scan is skipped.
    scan_incoming = direction in ("both", "incoming") and bool(dependents)

    incoming: list[dict[str, Any]] = []
    outgoing: list[dict[str, Any]] = []
    truncated = False
    for parm in parms:
        found: list[tuple[list, dict[str, Any]]] = []
        if scan_incoming:
            with contextlib.suppress(Exception):
                refs = [p.path() for p in parm.parmsReferencingThis()]
                if refs:
                    found.append((incoming, {"parm": parm.name(), "referenced_by": refs}))
        if direction in ("both", "outgoing"):
            entry = _outgoing_reference(parm)
            if entry is not None:
                found.append((outgoing, entry))
        if not found:
            continue
        # Truncated only when there is an entry that does not fit.
        if len(incoming) + len(outgoing) + len(found) > limit:
            truncated = True
            break
        for target, entry in found:
            target.append(entry)

    result: dict[str, Any] = {
        "node_path": node.path(),
        "parm_name": parm_name,
        "direction": direction,
        "incoming": incoming,
        "outgoing": outgoing,
        "truncated": truncated,
    }
    # This node only (include_children=False: a subnet's descendants are not
    # its own references), capped like the parameter lists.
    result["node_dependents"], count = _capped_paths(dependents, node.path(), limit)
    if count > limit:
        result["node_dependents_count"] = count
    with contextlib.suppress(Exception):
        references = node.references(include_children=False)
        result["node_references"], count = _capped_paths(references, node.path(), limit)
        if count > limit:
            result["node_references_count"] = count
    with contextlib.suppress(Exception):
        if node.needsToCook():
            result["note"] = (
                "node_dependents / node_references, and the dependents check that "
                "decides whether `incoming` is scanned, are as of this node's last "
                "cook (HOM: they can differ until it cooks); `outgoing` is parsed "
                "from expressions and is not."
            )
    return result


register_handler("parameters.get_parm_references", _get_parm_references)


###### Handler: parameters.get_parm_template_tree


def _template_tree_entry(pt: hou.ParmTemplate) -> dict[str, Any]:
    """One template as the Type Properties dialog shows it: folders, ranges,
    menus, conditionals, callbacks, defaults -- nothing evaluated.

    Built on _template_to_dict, so the tree and get_parameter_schema report a
    template with the same keys (default_value, is_hidden, min_is_strict,
    menu_items...) and the same fixes; the tree adds what only the tree needs.
    """
    entry = _template_to_dict(pt)
    with contextlib.suppress(Exception):
        conditionals = pt.conditionals()
        if conditionals:
            entry["conditionals"] = {
                key.name() if hasattr(key, "name") else str(key): value
                for key, value in conditionals.items()
            }
    with contextlib.suppress(Exception):
        help_text = pt.help()
        if help_text:
            entry["help"] = help_text
    with contextlib.suppress(Exception):
        if pt.joinsWithNext():
            entry["join_with_next"] = True
    with contextlib.suppress(Exception):
        tags = dict(pt.tags())
        if tags:
            entry["tags"] = {
                k: (v if len(str(v)) <= 120 else str(v)[:120] + "...") for k, v in tags.items()
            }
    if entry["type"] == "Folder":
        with contextlib.suppress(Exception):
            entry["folder_type"] = pt.folderType().name()
            if "Multiparm" in entry["folder_type"]:
                # A multiparm folder's default is the instance count a fresh
                # node gets.
                entry["default_instances"] = entry.get("default_value")
        with contextlib.suppress(Exception):
            if pt.endsTabGroup():
                entry["ends_tab_group"] = True
        children: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            for child in pt.parmTemplates():
                # One unreadable child costs that child, not the folder.
                with contextlib.suppress(Exception):
                    children.append(_template_tree_entry(child))
        entry["children"] = children
        return entry
    with contextlib.suppress(Exception):
        expressions = [e for e in pt.defaultExpression() if e]
        if expressions:
            entry["default_expression"] = expressions
    with contextlib.suppress(Exception):
        callback = pt.scriptCallback()
        if callback:
            entry["callback"] = callback
            entry["callback_language"] = pt.scriptCallbackLanguage().name()
    with contextlib.suppress(Exception):
        string_type = pt.stringType().name()
        if string_type != "Regular":
            entry["string_type"] = string_type
    with contextlib.suppress(Exception):
        entry["data_parm_type"] = pt.dataParmType().name()
    with contextlib.suppress(Exception):
        look = pt.look().name()
        if look != "Regular":
            entry["look"] = look
    return entry


def _count_tree(entries: list[dict[str, Any]]) -> int:
    return sum(1 + _count_tree(entry.get("children", [])) for entry in entries)


def _prune_tree(entries: list[dict[str, Any]], budget: list[int]) -> list[dict[str, Any]]:
    """Keep the first *budget* entries depth-first; the rest are cut."""
    kept: list[dict[str, Any]] = []
    for entry in entries:
        if budget[0] <= 0:
            break
        budget[0] -= 1
        if "children" in entry:
            entry["children"] = _prune_tree(entry["children"], budget)
        kept.append(entry)
    return kept


def _node_type_for_tree(context: str, type_name: str):
    """Resolve *type_name* in *context* the way createNode would, through the
    same resolver build_network and get_node_card use."""
    from fxhoudinimcp_server.handlers.graph_handlers import _resolve_node_type

    categories = hou.nodeTypeCategories()
    category = categories.get(context)
    if category is None:
        raise ValueError(f"Unknown context '{context}'. Available: {sorted(categories)}")
    resolved = _resolve_node_type(category, type_name)
    if resolved is None:
        close = get_close_matches(type_name, list(category.nodeTypes()), n=5, cutoff=0.4)
        raise ValueError(f"Node type '{type_name}' not found in {context}. Close: {close}")
    return resolved


def _get_parm_template_tree(
    node_path: str | None = None,
    type_name: str | None = None,
    context: str = "Sop",
    folder: Any = None,
    max_entries: int = 400,
    **_: Any,
) -> dict[str, Any]:
    """The whole parameter interface as a tree -- folders, conditionals, menu
    items, multiparm blocks, callbacks -- for a node or a node type.

    get_hda_info shows the top folders and get_parameter_schema flattens the
    rest away; neither can answer "what is in the Controls tab, in order,
    with its Hide When rules". This does. `folder` narrows to one folder by
    label (or a list of nested labels).
    """
    if node_path is not None:
        node = _resolve_node(node_path)
        group = node.parmTemplateGroup()
        subject: dict[str, Any] = {"node_path": node.path(), "type": node.type().name()}
    elif type_name is not None:
        node_type = _node_type_for_tree(context, type_name)
        group = node_type.parmTemplateGroup()
        subject = {"type": node_type.name(), "context": context}
    else:
        raise ValueError("Give node_path or type_name.")

    if folder is not None:
        labels = tuple(folder) if isinstance(folder, (list, tuple)) else (str(folder),)
        found = group.findFolder(labels)
        if found is None:
            available = [e.label() for e in group.entries() if _parm_type_name(e) == "Folder"]
            raise ValueError(f"No folder labelled {labels!r}. Top-level folders: {available}")
        entries = [_template_tree_entry(found)]
    else:
        entries = [_template_tree_entry(entry) for entry in group.entries()]

    total = _count_tree(entries)
    truncated = total > int(max_entries)
    if truncated:
        entries = _prune_tree(entries, [int(max_entries)])

    result = dict(subject)
    result.update(
        {
            "folder": folder,
            "entry_count": total,
            "truncated": truncated,
            "entries": entries,
        }
    )
    if truncated:
        result["note"] = (
            f"{total} entries, showing the first {int(max_entries)}. Narrow with "
            f"folder=<label> or raise max_entries."
        )
    return result


register_handler("parameters.get_parm_template_tree", _get_parm_template_tree)
