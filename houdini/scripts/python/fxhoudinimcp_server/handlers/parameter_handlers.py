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


def _set_parameter(node_path: str, parm_name: str, value: Any, **_: Any) -> dict[str, Any]:
    """Set a parameter value, auto-detecting the appropriate type."""
    if isinstance(value, (list, tuple)):
        new_value = _set_tuple(_resolve_node(node_path), parm_name, value)
        if new_value is not None:
            return {"node_path": node_path, "parm_name": parm_name, "new_value": new_value}

    parm = _resolve_parm(node_path, parm_name)

    try:
        parm.set(value)
    except hou.PermissionError:
        reason = _expression_driven([parm])
        if reason:
            raise ValueError(reason) from None
        raise

    return {
        "node_path": node_path,
        "parm_name": parm_name,
        "new_value": _serialize_value(parm.eval()),
    }


register_handler("parameters.set_parameter", _set_parameter)


###### Handler: parameters.set_parameters


def _set_parameters(node_path: str, params: dict[str, Any], **_: Any) -> dict[str, Any]:
    """Batch-set multiple parameters on a single node."""
    node = _resolve_node(node_path)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    available = _available_parm_names(node)
    for name, value in params.items():
        # A list on a vector name sets the whole tuple, exactly as the single
        # setter does. The batch path used to know only per-component names,
        # which made "prefer set_parameters" and "set a light colour" collide.
        if isinstance(value, (list, tuple)):
            try:
                new_value = _set_tuple(node, name, value)
            except Exception as exc:
                errors.append({"parm_name": name, "error": str(exc)})
                continue
            if new_value is not None:
                results.append({"parm_name": name, "new_value": new_value})
                continue
        parm = node.parm(name)
        if parm is None:
            close = get_close_matches(name, available, n=3, cutoff=0.4)
            hint = f" Did you mean: {close}?" if close else ""
            errors.append({"parm_name": name, "error": f"Parameter '{name}' not found.{hint}"})
            continue
        try:
            parm.set(value)
            results.append(
                {
                    "parm_name": name,
                    "new_value": _serialize_value(parm.eval()),
                }
            )
        except Exception as exc:
            errors.append({"parm_name": name, "error": str(exc)})

    return {
        "node_path": node_path,
        "set": results,
        "errors": errors,
    }


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


def _link_parameters(
    source_path: str,
    source_parm: str,
    dest_path: str,
    dest_parm: str,
    **_: Any,
) -> dict[str, Any]:
    """Create a channel reference from destination parameter to source parameter.

    The expression uses ``chs()`` for a String destination and ``ch()`` for
    everything else, and a path relative to the destination node. The reply
    reads the destination back so the caller sees the linked value, not just
    the expression text.
    """
    src = _resolve_parm(source_path, source_parm)
    dst = _resolve_parm(dest_path, dest_parm)

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
                "node_dependents / node_references are as of this node's last cook "
                "(HOM: they can differ until it cooks); parameter references are not."
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
