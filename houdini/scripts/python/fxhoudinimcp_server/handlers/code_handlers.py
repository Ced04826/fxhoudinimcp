"""Houdini-side handlers for code execution operations.

Provides 4 command handlers for executing Python code, HScript commands,
evaluating expressions, and reading environment variables within Houdini.
"""

from __future__ import annotations

# Built-in
import contextlib
import io
import json
import os
import sys
import time
import traceback
from typing import Any

# Third-party
import hou

# Internal
from fxhoudinimcp_server.dispatcher import register_handler
from fxhoudinimcp_server.errors import as_text
from fxhoudinimcp_server.handlers.state_receipt_helpers import (
    bounded_text_fields,
    clip,
    json_safe,
    positive_int,
    short_hash,
    text_receipt,
    utf8_bytes,
    value_receipt,
)

###### Constants

_MAX_CAPTURE_BYTES = 100 * 1024  # 100 KB

# A returned value goes into the receipt whole up to this size. Past it the
# receipt carries length, hash and samples, and the whole thing goes to
# dump_path if the caller asked for one. The old behaviour -- no limit at all --
# is how a single call put a megabyte of JSON into a conversation.
_MAX_RETURN_CHARS = 8 * 1024

# Hard ceiling on what a caller may raise the caps to. Past this the answer is
# a file, not a receipt.
_LIMIT_CEILING = 4 * 1024 * 1024

_RETURN_FORMATS = ("auto", "json", "repr", "none")

# A traceback is output too, and it was the one kind this command did not
# bound: an exception message that quotes the data that caused it, or a
# recursion traceback, is as big as anything printed. The receipt carries both
# ends of it; dump_path carries all of it.
_MAX_ERROR_CHARS = 4000
_MAX_MESSAGE_CHARS = 400


###### Helpers


def _truncate_output(text: str) -> str:
    """Truncate captured output to _MAX_CAPTURE_BYTES if it exceeds the limit."""
    if len(text) > _MAX_CAPTURE_BYTES:
        return text[:_MAX_CAPTURE_BYTES] + "\n[truncated]"
    return text


def _serialize_result(value: Any) -> Any:
    """Convert arbitrary Python objects to JSON-safe types."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_serialize_result(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_result(v) for k, v in value.items()}
    # Fallback: stringify
    return str(value)


###### Handler: code.execute_python


def _exception_line(exc: BaseException) -> str:
    """"ValueError: what went wrong", bounded, whatever the message contains."""
    try:
        message = clip(str(exc), _MAX_MESSAGE_CHARS)
    except Exception:  # noqa: BLE001 - a __str__ that raises is still an exception
        message = "<message unavailable>"
    return f"{_type_name(exc)}: {message}"


def _type_name(value: Any) -> str:
    """The type's name, or something printable when even that misbehaves."""
    try:
        return type(value).__name__
    except Exception:  # noqa: BLE001 - naming a type must not raise
        return "<unknown>"


def _resolve_return_format(value: Any) -> str:
    """The return_format token, or a message naming the ones that exist."""
    if value is None:
        return "auto"
    if isinstance(value, str) and value.lower() in _RETURN_FORMATS:
        return value.lower()
    raise ValueError(f"return_format must be one of {list(_RETURN_FORMATS)}, not {value!r}")


def _write_dump(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Write the complete record beside the capped receipt, and say how it went.

    A dump that failed to be written is not an execution failure and must not
    be reported as one — the code ran either way — but it is absolutely not the
    silent success it used to look like when nothing checked.
    """
    status: dict[str, Any] = {"requested_path": path, "written": False}
    try:
        absolute = os.path.abspath(path)
        directory = os.path.dirname(absolute)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(absolute, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        status["written"] = True
        status["path"] = absolute
        with contextlib.suppress(OSError):
            status["bytes"] = os.path.getsize(absolute)
    except Exception as exc:  # noqa: BLE001 - reported as data
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def _execute_python(
    code: str,
    return_expression: str | None = None,
    dump_path: str | None = None,
    max_stdout_chars: int = _MAX_CAPTURE_BYTES,
    max_return_chars: int = _MAX_RETURN_CHARS,
    return_format: str = "auto",
    **_: Any,
) -> dict[str, Any]:
    """Execute arbitrary Python code inside Houdini's interpreter.

    The code is executed via `exec()` in a namespace that has `hou`
    pre-imported. If *return_expression* is given it is evaluated with `eval()`
    in the same namespace after execution, inside the same output capture, so
    anything it prints is captured too.

    The receipt is bounded and says so. ``execution_success`` answers whether
    the code ran; ``output_complete`` answers whether what came back is all of
    it, unchanged. They are separate keys because they are separate facts, and
    a 300 KB print used to come back cut down to 100 KB with nothing but the
    word "truncated" to say how much was missing.

    ``output_complete`` is false when anything was clipped (``*_truncated``)
    AND when anything was altered on the way into JSON (``return_lossless``
    false, detailed by ``return_coerced``/``return_nonfinite``/
    ``return_circular``). A stringified hou.Node serialises perfectly and is
    not the node, so a receipt carrying one is not complete.

    The dump makes two separate promises, and they are reported separately.
    ``dump_streams_complete`` means every character printed is in the file.
    ``dump_return_lossless`` means the file's copy of the return value is the
    value rather than an altered rendering of it -- the dump holds the same
    JSON-safe copy the receipt does, so a stringified object is stringified
    there too. ``full_output_in_dump`` is only true when both hold.

    Limits are checked BEFORE the code runs: a bad limit must not be discovered
    after a side effect has already happened, and nothing here ever re-runs
    code that has already executed.

    Args:
        code: Python source to execute.
        return_expression: Expression evaluated after the code, in its namespace.
        dump_path: Write the complete record — full stdout, full stderr, the
            serialisable return value, and both error tracebacks — to this JSON
            file. The receipt stays small either way.
        max_stdout_chars: Cap on stdout and stderr in the receipt. The full
            text still goes to dump_path.
        max_return_chars: Cap on the returned value in the receipt.
        return_format: "auto" stringifies what JSON cannot hold and says which
            parts it stringified; "json" refuses instead of stringifying;
            "repr" returns repr() of the value; "none" drops the value and
            reports only its type.
    """
    # Everything checkable is checked first, while refusing still costs nothing.
    max_stdout_chars = positive_int(max_stdout_chars, "max_stdout_chars", _LIMIT_CEILING)
    max_return_chars = positive_int(max_return_chars, "max_return_chars", _LIMIT_CEILING)
    return_format = _resolve_return_format(return_format)
    if dump_path is not None:
        dump_path = as_text(dump_path, "dump_path").strip()
        if not dump_path:
            raise ValueError("dump_path must be a path, not an empty string")

    namespace: dict[str, Any] = {"hou": hou}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr

    exec_error: str | None = None
    eval_error: str | None = None
    exec_exception: str | None = None
    eval_exception: str | None = None
    result: Any = None
    began = time.time()
    sys.stdout, sys.stderr = stdout_buf, stderr_buf
    try:
        try:
            exec(code, namespace)  # noqa: S102
        except Exception as exc:
            exec_error = traceback.format_exc()
            exec_exception = _exception_line(exc)
        if exec_error is None and return_expression is not None:
            try:
                result = eval(return_expression, namespace)  # noqa: S307
            except Exception as exc:
                eval_error = traceback.format_exc()
                eval_exception = _exception_line(exc)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    elapsed_ms = round((time.time() - began) * 1000, 1)
    stdout_full = stdout_buf.getvalue()
    stderr_full = stderr_buf.getvalue()
    stdout_receipt = text_receipt(stdout_full, max_stdout_chars)
    stderr_receipt = text_receipt(stderr_full, max_stdout_chars)

    response: dict[str, Any] = {
        # "executed" predates the server-wide "success" convention. Code that
        # raised is a failure, and a caller checking the usual key got None --
        # falsy, but not False, and easy to read as "no opinion".
        "executed": exec_error is None,
        "execution_success": exec_error is None,
        "elapsed_ms": elapsed_ms,
    }
    # Tracebacks are output too, and nothing was bounding them: an exception
    # whose message quotes the data that caused it, or a recursion traceback,
    # put megabytes into a receipt that caps every other field. The whole text
    # goes to dump_path; the receipt gets the end of it, which is where the
    # exception and its message are.
    for source, key, headline in (
        (exec_error, "error", exec_exception),
        (eval_error, "eval_error", eval_exception),
    ):
        if not source:
            continue
        response.update(bounded_text_fields(key, source, _MAX_ERROR_CHARS))
        # Built from the exception itself rather than cut out of the
        # traceback, so it names the exception even when the message that
        # follows it is the thing that was enormous.
        response[f"{key}_summary"] = headline

    for stream, receipt in (("stdout", stdout_receipt), ("stderr", stderr_receipt)):
        if not receipt["chars"]:
            continue
        response[stream] = receipt["text"]
        response[f"{stream}_chars"] = receipt["chars"]
        response[f"{stream}_bytes"] = receipt["bytes"]
        response[f"{stream}_truncated"] = receipt["truncated"]
        if receipt["truncated"]:
            response[f"{stream}_omitted_chars"] = receipt["omitted_chars"]
            response[f"{stream}_sha1"] = receipt["sha1"]

    ###### The returned value: serialised honestly or not claimed at all

    return_truncated = False
    # True until something is stringified, renamed or dropped on the way into
    # the receipt. Separate from truncation: a value can be complete and lossy,
    # or lossless and clipped, and a caller needs to know which.
    return_lossless = True
    dump_return: Any = None
    serialization_notes: dict[str, Any] = {}
    serialization_error: str | None = None
    if exec_error is None and eval_error is None and return_expression is not None:
        if return_format == "none":
            response["return_value"] = None
            response["return_type"] = _type_name(result)
            response["return_omitted"] = True
            # Asked for and deliberately not carried: still not the whole of
            # what the run produced, and output_complete says so.
            return_lossless = False
        elif return_format == "repr":
            try:
                text = repr(result)
            except Exception as exc:  # noqa: BLE001 - the code already ran
                # A __repr__ that raises must not cost the caller the evidence
                # that the code itself succeeded.
                serialization_error = (
                    f"repr() of the {_type_name(result)} result raised "
                    f"{type(exc).__name__}: {exc}"
                )
                text = None
            if text is not None:
                receipt = text_receipt(text, max_return_chars)
                response["return_value"] = receipt["text"]
                response["return_serialization"] = "repr"
                response["return_chars"] = receipt["chars"]
                return_truncated = receipt["truncated"]
                dump_return = text
            # repr() is a description of a value, not the value: JSON that is
            # round-tripped from it is not the object that was returned.
            return_lossless = False
        else:
            safe, notes = json_safe(result)
            dump_return = safe
            serialization_notes = notes
            for category, key in (
                ("coerced", "return_coerced"),
                ("nonfinite", "return_nonfinite"),
                ("circular", "return_circular"),
            ):
                if notes.get(category):
                    # Stringified objects, NaN, infinity and cycles all leave
                    # the receipt holding something other than what the code
                    # returned. Each is located and counted.
                    response[key] = notes[category][:10]
                    response[f"{key}_count"] = notes[f"{category}_count"]
            # Lossless means the receipt holds the value, not a rendering of
            # it: no stringified objects, no nan turned into a word, no cycle
            # cut. Truncation is a separate question, answered separately.
            lossy = [
                category
                for category in ("coerced", "nonfinite", "circular")
                if notes.get(category)
            ]
            return_lossless = not lossy
            if lossy and return_format == "json":
                # Strict means strict: a NaN that came back as the word "NaN"
                # is as much not-the-value as a hou.Node that came back as its
                # path, and refusing only one of them was an accident.
                located = {
                    category: notes[category][:5] for category in lossy
                }
                counts = {category: notes[f"{category}_count"] for category in lossy}
                serialization_error = (
                    f"return value is not JSON: {counts}, first places {located}; "
                    f"use return_format='auto' to accept the altered copy, or "
                    f"return a JSON-safe value"
                )
                dump_return = None
            if serialization_error is None:
                receipt = value_receipt(safe, max_return_chars)
                if receipt["complete"]:
                    response["return_value"] = safe
                else:
                    # No "return_value" key on purpose: a clipped value under
                    # the name the caller reads is how half an answer gets used
                    # as a whole one.
                    response["return_value_summary"] = {
                        k: v for k, v in receipt.items() if k != "value"
                    }
                    return_truncated = True
    else:
        response["return_value"] = None

    if serialization_error:
        response["serialization_error"] = serialization_error
        response["return_value"] = None
        return_lossless = False

    if return_expression is not None and exec_error is None and eval_error is None:
        response["return_lossless"] = return_lossless
        response["return_serializable"] = return_lossless and not serialization_error
    response["return_truncated"] = return_truncated
    # Complete means: everything this run produced is in front of you, exactly
    # as it was. Anything clipped, stringified, renamed or dropped makes it
    # false, and the keys above say which of those happened.
    response["output_complete"] = not (
        stdout_receipt["truncated"]
        or stderr_receipt["truncated"]
        or return_truncated
        or not return_lossless
    )
    response["success"] = exec_error is None and eval_error is None and not serialization_error

    ###### The complete record, if one was asked for

    if dump_path is not None:
        payload: dict[str, Any] = {
            "command": "code.execute_python",
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # The code itself is not echoed back anywhere: the caller wrote it
            # and a fingerprint is enough to prove which run this was.
            "code_sha1": short_hash(code),
            "code_chars": len(code),
            "execution_success": exec_error is None,
            "error": exec_error,
            "eval_error": eval_error,
            "return_expression": return_expression,
            "return_format": return_format,
            "return_lossless": return_lossless,
            "serialization_error": serialization_error,
            # Where the serialisable copy differs from what the code returned.
            "serialization_notes": serialization_notes,
            "stdout": stdout_full,
            "stderr": stderr_full,
            "stdout_chars": len(stdout_full),
            "stdout_bytes": utf8_bytes(stdout_full),
            "stderr_chars": len(stderr_full),
            "stderr_bytes": utf8_bytes(stderr_full),
            "elapsed_ms": elapsed_ms,
        }
        if return_format != "none":
            payload["return_value"] = dump_return
        status = _write_dump(dump_path, payload)
        response["dump"] = status
        if not status["written"]:
            # The execution verdict is untouched: the code ran. What failed is
            # the record of it, and the caller needs to know which is which.
            response["dump_failed"] = True
        else:
            # Two different promises, and conflating them claimed the dump
            # held an original value that nothing anywhere still has.
            # Capture: every character the run printed is in the file.
            # Lossless: the file's copy of the return value is the value,
            # not a stringified, NaN-renamed or cycle-cut rendering of it.
            response["dump_streams_complete"] = True
            if return_expression is not None:
                response["dump_return_lossless"] = return_lossless and not serialization_error
            response["full_output_in_dump"] = bool(
                return_lossless and not serialization_error
            )

    return response


register_handler("code.execute_python", _execute_python)


###### Handler: code.execute_hscript


def _execute_hscript(command: str, **_: Any) -> dict[str, Any]:
    """Execute an HScript command and return its output."""
    # A non-string reached the SWIG binding and came back as "in method
    # 'hscript', argument 1 of type 'char const *'".
    output, errors = hou.hscript(as_text(command, "command"))

    return {
        "output": _truncate_output(output) if output else output,
        "errors": _truncate_output(errors) if errors else None,
    }


register_handler("code.execute_hscript", _execute_hscript)


###### Handler: code.evaluate_expression


def _evaluate_expression(expression: str, language: str = "hscript", **_: Any) -> dict[str, Any]:
    """Evaluate an expression and return its result.

    Supports both HScript expressions (via `hou.hscriptExpression()`)
    and Python expressions (via `eval()`).
    """
    if language.lower() == "python":
        namespace: dict[str, Any] = {"hou": hou}
        result = eval(expression, namespace)  # noqa: S307
    else:
        result = hou.hscriptExpression(expression)

    return {
        "expression": expression,
        "language": language,
        "result": _serialize_result(result),
    }


register_handler("code.evaluate_expression", _evaluate_expression)


###### Handler: code.get_env_variable


def _get_env_variable(var_name: str, **_: Any) -> dict[str, Any]:
    """Get a Houdini environment variable."""
    value = hou.getenv(as_text(var_name, "var_name"))

    return {
        "var_name": var_name,
        "value": value,
        "exists": value is not None,
    }


register_handler("code.get_env_variable", _get_env_variable)
