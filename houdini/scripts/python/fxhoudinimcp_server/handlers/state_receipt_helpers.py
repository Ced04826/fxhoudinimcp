"""Houdini-free helpers behind honest receipts and honest parameter writes.

Three commands -- ``graph.build_network``, ``parameters.set_parameter(s)`` and
``code.execute_python`` -- were all failing the same way in different clothes:
the receipt said a thing had happened without carrying the evidence, and a
caller could not tell a complete answer from a clipped one. A network reported
the geometry of whatever node happened to hold the display flag; a parameter
write reported the value it asked for rather than the value the node now
evaluates; a 300 KB print came back cut at 100 KB with a four-word note.

The deciding logic for all three is bookkeeping and arithmetic rather than HOM,
so it lives here, where a test exercises it without a running Houdini. Nothing
in this module imports ``hou``: the few functions that need a live parameter
take the object and duck-type it, so a fake in a test walks the same branch a
``hou.Parm`` does.
"""

from __future__ import annotations

# Built-in
import contextlib
import hashlib
import json
import math
from typing import Any

###### Text and value budgets

# How much of a clipped string is worth keeping at each end. The head is where
# a JSON document or a traceback announces what it is; the tail is where a
# truncated write usually shows it was truncated.
_HEAD_CHARS = 160
_TAIL_CHARS = 60

# Enough hash to tell two 2 MB dumps apart, short enough to sit in a receipt.
_HASH_CHARS = 12

# A sample is meant to show the shape of what was elided, so it is bounded in
# its own right: a summary of five 100 000-character strings that quotes the
# strings is not a summary. Depth and width caps keep a nested sample from
# growing the same way one level down.
_SAMPLE_ITEMS = 3
_SAMPLE_DEPTH = 2
_KEY_CHARS = 60


def _shrink(value: Any, budget: int, depth: int = 0) -> Any:
    """*value* cut down to something that fits in a receipt, shape intact.

    Every leaf is capped, and the caps apply again inside a nested list or
    dict, so no single element can carry the whole of what was summarised.
    """
    if isinstance(value, str):
        if len(value) <= budget:
            return value
        return f"{value[:budget]}...[{len(value)} chars]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= _SAMPLE_DEPTH:
        return f"<{type(value).__name__}>"
    inner = max(20, budget // _SAMPLE_ITEMS)
    if isinstance(value, (list, tuple)):
        shown = [_shrink(entry, inner, depth + 1) for entry in list(value)[:_SAMPLE_ITEMS]]
        if len(value) > _SAMPLE_ITEMS:
            shown.append(f"...+{len(value) - _SAMPLE_ITEMS} more")
        return shown
    if isinstance(value, dict):
        shown_dict: dict[str, Any] = {}
        for key, entry in list(value.items())[:_SAMPLE_ITEMS]:
            shown_dict[_shrink(str(key), _KEY_CHARS, _SAMPLE_DEPTH)] = _shrink(
                entry, inner, depth + 1
            )
        if len(value) > _SAMPLE_ITEMS:
            shown_dict["..."] = f"+{len(value) - _SAMPLE_ITEMS} more keys"
        return shown_dict
    return _shrink(_safe_str(value), budget, depth)


def _safe_str(value: Any) -> str:
    """``str(value)`` that cannot take the receipt down with it.

    An object whose ``__str__`` raises is rare and entirely possible, and it
    would otherwise turn "here is a summary of your result" into a traceback
    that loses the result and the run that produced it.
    """
    try:
        return str(value)
    except Exception as exc:  # noqa: BLE001 - the failure is the answer here
        return f"<unrepresentable {type(value).__name__}: {type(exc).__name__}>"


def _fit(samples: list, budget: int) -> tuple:
    """(samples, dropped) — drop from the end until the whole list fits.

    The per-element caps bound each sample; this bounds their sum, so a
    summary's size follows the budget rather than the data it summarises.
    """
    dropped = 0
    while samples and len(_encode(samples) or "") > budget and len(samples) > 1:
        samples = samples[:-1]
        dropped += 1
    if samples and len(_encode(samples) or "") > budget:
        samples = [_shrink(samples[0], max(20, budget // 2), _SAMPLE_DEPTH)]
    return samples, dropped


def utf8_bytes(text: str) -> int:
    """Byte length of *text* once encoded, which is what a transport pays for.

    Character counts and byte counts diverge by a factor of three on Chinese
    text and four on emoji, so a receipt that reports only one of them is
    reporting the wrong one half the time.
    """
    return len(text.encode("utf-8", "surrogatepass"))


def short_hash(text: str) -> str:
    """A short, stable fingerprint of *text* for comparing elided values."""
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()[:_HASH_CHARS]


# Text that goes inside a sentence -- an expression being refused, the message
# of an exception, a traceback -- rather than into a field of its own. All of
# those can be arbitrarily long: a VEX snippet driving a parameter, an
# exception carrying the value that caused it, a recursion traceback. A
# receipt's size must not follow them.
_EXCERPT_CHARS = 400


def clip(text: Any, limit: int = _EXCERPT_CHARS) -> str:
    """*text* short enough to sit in a message, saying what it left out."""
    if not isinstance(text, str):
        text = _safe_str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[{len(text)} chars, sha1 {short_hash(text)}]"


def excerpt_fields(name: str, text: str, limit: int = _EXCERPT_CHARS) -> dict[str, Any]:
    """A receipt field for *text*, with its real size when it did not all fit."""
    fields: dict[str, Any] = {name: text[:limit] if isinstance(text, str) else text}
    if isinstance(text, str) and len(text) > limit:
        fields[f"{name}_chars"] = len(text)
        fields[f"{name}_truncated"] = True
        fields[f"{name}_sha1"] = short_hash(text)
    return fields


def bounded_text_fields(name: str, text: str, limit: int) -> dict[str, Any]:
    """A long text as its beginning and its end, plus the size of the whole.

    For a traceback both ends matter and the middle rarely does: the first
    frames say where the run started and the last line says what went wrong.
    Keeping only the head loses the exception; keeping only the tail loses the
    entry point.
    """
    if not isinstance(text, str) or len(text) <= limit:
        return {name: text}
    head_len = max(1, limit * 2 // 5)
    head, tail = text[:head_len], text[-(limit - head_len) :]
    return {
        name: f"{head}\n...[{len(text) - limit} chars elided]...\n{tail}",
        f"{name}_chars": len(text),
        f"{name}_truncated": True,
        f"{name}_sha1": short_hash(text),
    }


def text_receipt(text: str, max_chars: int) -> dict[str, Any]:
    """A capped copy of *text* plus the measurements of the whole of it.

    ``chars`` and ``bytes`` always describe the complete text, never the copy,
    which is the whole point: the caller learns how much it is not seeing.
    """
    text = text or ""
    total = len(text)
    receipt: dict[str, Any] = {
        "truncated": total > max_chars,
        "chars": total,
        "bytes": utf8_bytes(text),
    }
    if total > max_chars:
        receipt["text"] = text[:max_chars]
        receipt["omitted_chars"] = total - max_chars
        receipt["sha1"] = short_hash(text)
    else:
        receipt["text"] = text
    return receipt


def _encode(value: Any) -> str | None:
    """JSON for *value*, or None when it is not JSON at all."""
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=False)
    except (TypeError, ValueError):
        return None


def value_receipt(value: Any, max_chars: int, full: bool = False) -> dict[str, Any]:
    """A small value as itself; a large one as length, hash and samples.

    The alternative -- clipping the value and reporting it under the same key
    -- is how a caller ends up comparing a truncated string to a complete one
    and concluding the write worked.
    """
    encoded = _encode(value)
    text = encoded if encoded is not None else _safe_str(value)
    size = len(text)
    if full or size <= max_chars:
        return {"complete": True, "value": value, "chars": size}

    receipt: dict[str, Any] = {
        "complete": False,
        "chars": size,
        "bytes": utf8_bytes(text),
        "sha1": short_hash(text),
        "hint": "pass return_values='full' (or raise the cap) for the whole value",
    }
    # Everything below is bounded by *max_chars*, never by the size of what is
    # being summarised: five hundred-thousand-character strings summarise to
    # the same few hundred characters as five short ones.
    head = min(_HEAD_CHARS, max(40, max_chars // 2))
    if isinstance(value, str):
        receipt["kind"] = "string"
        receipt["head"] = value[:head]
        receipt["tail"] = value[-min(_TAIL_CHARS, head) :]
    elif isinstance(value, (list, tuple)):
        receipt["kind"] = "list"
        receipt["count"] = len(value)
        # First three and last one: enough to see the shape of the elements and
        # that the list really does run to the end it claims.
        indices = list(range(min(3, len(value))))
        if len(value) > 4:
            indices.append(len(value) - 1)
        per_sample = max(40, max_chars // max(1, len(indices)))
        samples = [_shrink(value[i], per_sample) for i in indices]
        samples, dropped = _fit(samples, max_chars)
        receipt["sample"] = samples
        receipt["sample_indices"] = indices[: len(samples)]
        if dropped:
            receipt["samples_dropped"] = dropped
    elif isinstance(value, dict):
        receipt["kind"] = "object"
        receipt["count"] = len(value)
        keys = [_shrink(_safe_str(k), _KEY_CHARS, _SAMPLE_DEPTH) for k in list(value)[:8]]
        keys, dropped = _fit(keys, max_chars)
        receipt["sample_keys"] = keys
        if dropped:
            receipt["sample_keys_dropped"] = dropped
    else:
        receipt["kind"] = type(value).__name__
        receipt["head"] = _safe_str(value)[:head]
    return receipt


###### JSON safety

_MAX_DEPTH = 24

# A structure with fifty thousand unserialisable leaves produces fifty thousand
# note paths, and nobody reads the fifty-thousandth. The count is what matters
# past the first few; the paths are there to find the offender.
_NOTE_CAP = 50


def json_safe(value: Any, _path: str = "$", _seen: tuple = (), _depth: int = 0) -> tuple:
    """(JSON-safe copy, notes) -- with every lossy step written down.

    ``str()`` on an arbitrary object produces something that serialises and
    means nothing, and ``json.dumps`` emits bare ``NaN``, which is not JSON and
    which a strict parser on the other end rejects. Both are recorded in
    ``notes`` so a receipt can say the value came back changed rather than
    claiming it came back whole.

    ``notes`` carries at most ``_NOTE_CAP`` paths per category alongside a
    ``<category>_count`` of how many there really were, and each path is
    bounded, because a dictionary key can itself be a hundred thousand
    characters long.
    """
    notes: dict[str, Any] = {"nonfinite": [], "coerced": [], "circular": []}
    counts: dict[str, int] = {"nonfinite": 0, "coerced": 0, "circular": 0}
    # JSON keys are strings, so an int or finite float key written as its
    # decimal text is what json.dumps itself does: counted, not a loss.
    stringified = [0]

    def _note(category: str, path: str) -> None:
        counts[category] += 1
        if len(notes[category]) < _NOTE_CAP:
            notes[category].append(path)

    def _walk(item: Any, path: str, seen: tuple, depth: int) -> Any:
        if item is None or isinstance(item, (bool, int, str)):
            return item
        if isinstance(item, float):
            if math.isfinite(item):
                return item
            _note("nonfinite", path)
            # A string, because JSON has no nan/inf and a null would read as
            # "there was no value" rather than "the value was not finite".
            return "NaN" if math.isnan(item) else ("Infinity" if item > 0 else "-Infinity")
        if isinstance(item, bytes):
            _note("coerced", path)
            return item.decode("utf-8", "replace")
        if depth >= _MAX_DEPTH:
            _note("coerced", path)
            return f"<depth limit: {type(item).__name__}>"
        if id(item) in seen:
            _note("circular", path)
            return "<circular reference>"
        if isinstance(item, (list, tuple, set, frozenset)):
            if isinstance(item, (set, frozenset)):
                _note("coerced", path)
                try:
                    items = sorted(item, key=repr)
                except Exception:  # noqa: BLE001 - an unsortable set is still a set
                    items = list(item)
            else:
                items = list(item)
            return [
                _walk(entry, f"{path}[{index}]", seen + (id(item),), depth + 1)
                for index, entry in enumerate(items)
            ]
        if isinstance(item, dict):
            out: dict[str, Any] = {}
            for key, entry in item.items():
                # The key goes into the path, so it is bounded there: a
                # hundred-thousand-character key must not produce a
                # hundred-thousand-character note.
                label = _shrink(_safe_str(key), _KEY_CHARS, _SAMPLE_DEPTH)
                text = _safe_str(key)
                if text in out:
                    # Two keys that print the same (1 and "1") leave one value
                    # behind whatever their types.
                    _note("coerced", f"{path}.<key {label}>")
                elif type(key) is int or (type(key) is float and math.isfinite(key)):
                    stringified[0] += 1
                elif not isinstance(key, str):
                    _note("coerced", f"{path}.<key {label}>")
                out[text] = _walk(entry, f"{path}.{label}", seen + (id(item),), depth + 1)
            return out
        _note("coerced", path)
        return _safe_str(item)

    safe = _walk(value, _path, _seen, _depth)
    summary: dict[str, Any] = {}
    for category, paths in notes.items():
        if not paths:
            continue
        summary[category] = paths
        summary[f"{category}_count"] = counts[category]
    if stringified[0]:
        summary["keys_stringified"] = stringified[0]
    return safe, summary


###### build_network: input specs

# "exact" is the only policy that disconnects anything, and it disconnects only
# after every creation callback has run, so an auto-wiring OnCreated script is
# allowed to do its thing and then be overruled -- never suppressed.
INPUT_POLICIES = ("preserve", "exact")


def resolve_input_policy(policy: Any) -> str:
    """The policy token, or a message naming the ones that exist."""
    if policy is None:
        return "preserve"
    if isinstance(policy, str) and policy.lower() in INPUT_POLICIES:
        return policy.lower()
    raise ValueError(f"input_policy must be one of {list(INPUT_POLICIES)}, not {policy!r}")


def _as_index(value: Any) -> int | None:
    """An input index, or None when the value is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not float(value).is_integer():
        return None
    return int(value)


def parse_input_entries(
    entries: Any,
    max_inputs: int,
    label: str,
    resolve_name: Any = None,
) -> tuple[list, list]:
    """Normalise a spec's ``inputs`` into index/source/output entries.

    Returns ``(parsed, errors)``. A ``None`` entry holds a position without
    requesting a connection, which is how a caller wires input 1 and leaves
    input 0 alone in a list that has to be positional.

    A dict may name its connector with ``input_name`` instead of ``index``
    (the name wins); *resolve_name* turns the name into an index, raising
    ValueError with the reason, or returning None when there is nothing to
    look it up in (an error already reported elsewhere). A dict with
    ``indirect_input`` wires from that connector of the parent subnet rather
    than from a node, and then carries no ``source``.

    Each parsed entry has index, source, source_output, input_name and
    indirect.
    """
    errors: list[str] = []
    if entries is None:
        return [], errors
    if not isinstance(entries, (list, tuple)):
        return [], [f"node {label}: 'inputs' must be a list, not {type(entries).__name__}"]

    parsed: list[dict[str, Any]] = []
    claimed: dict[int, int] = {}
    for position, entry in enumerate(entries):
        if entry is None:
            continue
        input_name = None
        indirect = None
        if isinstance(entry, dict):
            source = entry.get("source")
            input_name = entry.get("input_name") or None
            if input_name is not None:
                if resolve_name is None:
                    errors.append(
                        f"node {label}: input entry {position} names input "
                        f"{input_name!r}, but its connectors could not be read"
                    )
                    continue
                try:
                    index = resolve_name(str(input_name))
                except ValueError as exc:
                    errors.append(f"node {label}: {exc}")
                    continue
                if index is None:
                    continue
            else:
                index = _as_index(entry.get("index", position))
            output = _as_index(entry.get("source_output", 0))
            if index is None:
                errors.append(
                    f"node {label}: input entry {position} has a non-integer "
                    f"'index': {entry.get('index')!r}"
                )
                continue
            if output is None:
                errors.append(
                    f"node {label}: input {index} has a non-integer "
                    f"'source_output': {entry.get('source_output')!r}"
                )
                continue
            if output < 0:
                errors.append(f"node {label}: input {index} has a negative source_output {output}")
                continue
            if entry.get("indirect_input") is not None:
                # The parent subnet's own connector: not a node, so it has no
                # path a source string could name, and it has one output.
                indirect = entry["indirect_input"]
                if source is not None:
                    errors.append(
                        f"node {label}: an input takes either 'source' or "
                        f"'indirect_input', not both (got {source!r} and {indirect!r})"
                    )
                    continue
                if output != 0:
                    errors.append(
                        f"node {label}: a subnet input connector has one output; "
                        f"source_output must be 0, got {output}"
                    )
                    continue
                if isinstance(indirect, float) and indirect.is_integer():
                    indirect = int(indirect)
                if isinstance(indirect, bool) or not isinstance(indirect, int):
                    errors.append(
                        f"node {label}: indirect_input must be an integer, got {indirect!r}"
                    )
                    continue
            elif source is None:
                # An explicit null source is a request to leave that index
                # empty, which "exact" then enforces and "preserve" ignores.
                continue
        elif isinstance(entry, str):
            source, index, output = entry, position, 0
        else:
            errors.append(
                f"node {label}: input entry {position} must be a node name or a "
                f"dict, not {type(entry).__name__}"
            )
            continue

        if indirect is None and (not isinstance(source, str) or not source.strip()):
            errors.append(f"node {label}: input {index} has an empty source")
            continue
        if index < 0:
            errors.append(f"node {label}: input index {index} is negative")
            continue
        if max_inputs > 0 and index >= max_inputs:
            errors.append(
                f"node {label}: input {index} exceeds max inputs ({max_inputs}) of its type"
            )
            continue
        if index in claimed:
            errors.append(
                f"node {label}: input {index} is specified twice (entries "
                f"{claimed[index]} and {position})"
            )
            continue
        claimed[index] = position
        parsed.append(
            {
                "index": index,
                "source": source,
                "source_output": output,
                "input_name": input_name,
                "indirect": indirect,
            }
        )

    parsed.sort(key=lambda entry: entry["index"])
    return parsed, errors


# Parameter template types whose value is a number. A string on one of these is
# left alone on purpose: Houdini accepts a good deal of coercion here, and a
# dry run that refuses something the real set would have accepted is worse than
# one that stays quiet.
_NUMERIC_PARM_TYPES = ("Int", "Float", "Toggle", "Menu")


def check_parm_value(label: str, type_name: str, parm_name: str, value: Any, info: dict) -> list:
    """Everything about a spec'd parameter value that is knowable before the set.

    This is deliberately not a dry cook: it checks shape and obviously wrong
    Python types, the two things that made a validated build fail during the
    build. Anything that depends on what the node does with the value is left
    to the node.
    """
    errors: list[str] = []
    if info is None:
        return errors
    components = int(info.get("components") or 1)
    template = str(info.get("type") or "")

    if isinstance(value, dict):
        errors.append(f"node {label}: parm '{parm_name}' takes a value, not an object")
        return errors
    if value is None:
        errors.append(f"node {label}: parm '{parm_name}' was given null")
        return errors

    if isinstance(value, (list, tuple)):
        if not info.get("is_tuple") and components <= 1 and len(value) != 1:
            errors.append(
                f"node {label}: parm '{parm_name}' on {type_name} is a single "
                f"value, got a list of {len(value)}"
            )
            return errors
        if components and len(value) != components:
            errors.append(
                f"node {label}: parm '{parm_name}' on {type_name} has "
                f"{components} components, got {len(value)}"
            )
            return errors
        for position, component in enumerate(value):
            if isinstance(component, (list, tuple, dict)):
                errors.append(
                    f"node {label}: parm '{parm_name}' component {position} is "
                    f"a {type(component).__name__}; components are scalars"
                )
            elif component is None:
                errors.append(f"node {label}: parm '{parm_name}' component {position} is null")
            elif template in _NUMERIC_PARM_TYPES and not isinstance(component, (int, float, str)):
                errors.append(
                    f"node {label}: parm '{parm_name}' is {template}; component "
                    f"{position} is a {type(component).__name__}"
                )
    return errors


def terminal_names(order: list, edges: list) -> list:
    """Created nodes nothing else created consumes -- the outputs of the build.

    Cooking these and reporting their geometry is the honest answer to "what
    did this build produce". The display node is not: it can belong to
    something the caller never mentioned.
    """
    consumed = {source for _target, source in edges if source in set(order)}
    terminals = [name for name in order if name not in consumed]
    return terminals or (order[-1:] if order else [])


###### Parameter writes

# "preserve" refuses rather than overwrites, because the failure it exists to
# prevent -- a write that reports success while an expression keeps driving the
# parameter -- is silent, and a refusal is not.
EXPRESSION_POLICIES = ("preserve", "replace")


def resolve_expression_policy(policy: Any) -> str:
    """The policy token, or a message naming the ones that exist."""
    if policy is None:
        return "preserve"
    if isinstance(policy, str) and policy.lower() in EXPRESSION_POLICIES:
        return policy.lower()
    raise ValueError(
        f"expression_policy must be one of {list(EXPRESSION_POLICIES)}, not {policy!r}"
    )


def parm_state(parm: Any) -> dict[str, Any]:
    """What is on a parameter right now: value, raw text, expression, keys, lock.

    Every read is guarded. ``expression()`` raises on a parameter that has none
    -- which is most of them -- and a diagnostic that raises while diagnosing
    is worthless.
    """
    state: dict[str, Any] = {
        "name": None,
        "value": None,
        "raw_value": None,
        "expression": None,
        "expression_language": None,
        "keyframes": 0,
        "locked": False,
    }
    with contextlib.suppress(Exception):
        state["name"] = parm.name()
    try:
        state["value"] = parm.eval()
    except Exception as exc:  # noqa: BLE001
        state["eval_error"] = str(exc).splitlines()[0][:200]
    with contextlib.suppress(Exception):
        state["raw_value"] = parm.rawValue()
    try:
        expression = parm.expression()
        state["expression"] = expression
        try:
            state["expression_language"] = parm.expressionLanguage().name()
        except Exception:  # noqa: BLE001
            state["expression_language"] = None
    except Exception:  # noqa: BLE001 - no expression is the common case
        state["expression"] = None
    with contextlib.suppress(Exception):
        state["keyframes"] = len(parm.keyframes())
    with contextlib.suppress(Exception):
        state["locked"] = bool(parm.isLocked())
    return state


def raw_differs(raw: Any, value: Any) -> bool:
    """Whether the raw text says anything the evaluated value does not.

    A float parameter holding 1.0 has a raw value of "1", and reporting that as
    a difference would announce an expression on every second parameter in the
    scene. Only a raw string that is not simply the same number spelled
    differently is worth a caller's attention.
    """
    if not isinstance(raw, str):
        return False
    if raw == str(value):
        return False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    try:
        return float(raw) != float(value)
    except (TypeError, ValueError):
        # Not a number at all: a variable, a channel reference, an expression.
        return True


def write_refusal(state: dict, policy: str, parm_label: str) -> str | None:
    """Why this write must not happen, or None when it may.

    A locked parameter is refused under both policies: unlocking one to satisfy
    a write is a decision about the user's scene, not a detail of the write.
    """
    if state.get("locked"):
        return (
            f"'{parm_label}' is locked; unlock it deliberately "
            f"(parameters.lock_parameter) before writing"
        )
    if policy != "preserve":
        return None
    expression = state.get("expression")
    if expression:
        # Clipped: a parameter can be driven by a page of VEX, and a refusal
        # that quotes the whole of it costs more than the write would have.
        return (
            f"'{parm_label}' is driven by the expression {clip(expression)!r}; "
            f"expression_policy='preserve' will not overwrite it. Pass "
            f"expression_policy='replace' to clear the expression on this "
            f"component and set a literal."
        )
    if state.get("keyframes"):
        return (
            f"'{parm_label}' has {state['keyframes']} keyframe(s); "
            f"expression_policy='preserve' will not overwrite them. Pass "
            f"expression_policy='replace' to clear them on this component."
        )
    return None


def clear_channels(parm: Any) -> dict[str, Any]:
    """Remove keyframes and expression from *this* parameter, and say what went.

    Only the addressed component is touched. A channel reference is cleared
    where it is written, never followed to the parameter it points at: that
    parameter belongs to some other node the caller did not name.
    """
    outcome: dict[str, Any] = {"cleared_expression": False, "cleared_keyframes": 0}
    before = parm_state(parm)
    if before.get("expression"):
        # What was removed is worth reporting and is not worth reporting in
        # full: a snippet-driven parameter would otherwise put kilobytes of
        # VEX into the receipt for every component it cleared.
        outcome.update(excerpt_fields("previous_expression", before["expression"]))
    outcome["cleared_keyframes"] = before.get("keyframes") or 0
    if before.get("expression") or before.get("keyframes"):
        try:
            parm.deleteAllKeyframes()
            outcome["cleared_expression"] = bool(before.get("expression"))
        except Exception as exc:  # noqa: BLE001
            outcome["clear_error"] = clip(str(exc).splitlines()[0], 200)
    return outcome


def tuple_length_error(parm_label: str, want: int, got: int) -> str:
    """The one message both the single and the batch write use for a bad tuple."""
    return f"'{parm_label}' has {want} components, got {got} values"


# HOM's Parm.set() defaults to follow_parm_reference=True: writing to a
# parameter that carries a channel reference writes to the parameter it points
# at, on a node the caller never named. Every write this server makes is
# addressed at one parameter, so every write says so explicitly.
def set_parm_value(parm: Any, value: Any) -> None:
    """Write *value* to this parameter and to no other.

    Raises rather than silently falling back to a following write: a build of
    Houdini that does not take the keyword would otherwise edit somebody
    else's node, which is the exact thing the keyword is here to prevent.
    """
    try:
        parm.set(value, follow_parm_reference=False)
    except TypeError as exc:
        if "follow_parm_reference" in str(exc):
            raise TypeError(
                "this Houdini build's Parm.set() does not accept "
                "follow_parm_reference; refusing to write, because the default "
                "would follow the channel reference and edit its source"
            ) from exc
        raise


def _template_facts(parm: Any) -> dict[str, Any]:
    """(type name, has a menu) for a parameter, or nothing knowable."""
    facts: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        template = parm.parmTemplate()
        facts["type"] = template.type().name()
        with contextlib.suppress(Exception):
            facts["has_menu"] = bool(template.menuItems())
    return facts


def component_type_error(parm: Any, value: Any, label: str) -> str | None:
    """Why this value obviously cannot be written, or None if it might be.

    Checked BEFORE anything is cleared, because "replace" deletes an
    expression first: discovering at set() time that the value was a
    dictionary would leave the parameter stripped and unwritten.

    Deliberately conservative. A string on a numeric parameter is left alone
    when the parameter has a menu, because a menu token is a string and
    Houdini accepts it.
    """
    if value is None:
        return f"'{label}' was given null"
    if isinstance(value, (list, tuple, dict, set)):
        return f"'{label}' takes a scalar, got a {type(value).__name__}"
    facts = _template_facts(parm)
    template = facts.get("type")
    if template in ("Int", "Float", "Toggle") and isinstance(value, str):
        if facts.get("has_menu"):
            return None
        try:
            float(value)
        except ValueError:
            return (
                f"'{label}' is a {template} parameter with no menu; "
                f"{value!r} is not a number"
            )
    if template == "String" and isinstance(value, (bytes, bytearray)):
        return f"'{label}' is a String parameter; pass text, not bytes"
    return None


###### Shared limit checking


def positive_int(value: Any, name: str, maximum: int | None = None) -> int:
    """A positive integer argument, checked before anything irreversible runs."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive integer, not {type(value).__name__}: {value!r}")
    if isinstance(value, float) and not float(value).is_integer():
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be at least 1, got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must not exceed {maximum}, got {number}")
    return number
