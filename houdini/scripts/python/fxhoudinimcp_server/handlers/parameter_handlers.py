"""Houdini-side handlers for parameter operations.

Provides 10 command handlers for reading, writing, and managing
node parameters, expressions, channel references, and spare parameters.
"""

from __future__ import annotations

import contextlib

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

    result: dict[str, Any] = {
        "node_path": node_path,
        "parm_name": parm_name,
        "value": _serialize_value(parm.eval()),
        "raw_value": _serialize_value(parm.rawValue()),
        "parm_type": _parm_type_name(pt),
        "is_locked": parm.isLocked(),
        "is_at_default": parm.isAtDefault(),
    }

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
        entry, error = _write_parm(node, name, value, policy, mode, max_value_chars)
        if entry is not None:
            results.append(entry)
        if error:
            errors.append({"parm_name": name, "error": error})

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


def _link_parameters(
    source_path: str,
    source_parm: str,
    dest_path: str,
    dest_parm: str,
    **_: Any,
) -> dict[str, Any]:
    """Create a channel reference from destination parameter to source parameter."""
    src = _resolve_parm(source_path, source_parm)
    dst = _resolve_parm(dest_path, dest_parm)

    # Build the channel reference expression
    ref_expr = f'ch("{src.path()}")'
    dst.setExpression(ref_expr, hou.exprLanguage.Hscript)

    return {
        "source": src.path(),
        "destination": dst.path(),
        "expression": ref_expr,
    }


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

    templates = []
    created = []
    for spec in parameters:
        pt = _build_parm_template(spec)
        templates.append(pt)
        created.append(spec["parm_name"])

    ptg = node.parmTemplateGroup()

    if folder_name is not None:
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
        "count": len(created),
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
        entry: dict[str, Any] = {"value": _serialize_value(parm.eval())}
        raw = parm.rawValue()
        # Only worth reporting when it differs: an expression is the thing a
        # caller most often needs to see and a literal is just noise.
        if isinstance(raw, str) and raw != str(entry["value"]):
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
