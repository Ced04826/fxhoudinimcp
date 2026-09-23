"""Tests for get_parameters(inside=...), one pattern read across a network.

"Every file parameter of this material library, unexpanded" had no verb: it
was find_nodes plus one get_parameters per node, 68 calls in one session, or
an execute_python. get_parameters now takes `inside` (a network) instead of
`node_path` and answers one table of `rows` {node, parm, value, raw_value},
capped at 2000 rows with `truncated`. `patterns` is required there, and
`node_path` with `inside` is refused.

hou is mocked here and only through monkeypatch; the live check ran on
Houdini 22.0.429 (six file SOPs, one call, six rows, `$JOB` visible raw).
"""

from __future__ import annotations

# Built-in
import os
import sys
from unittest.mock import MagicMock

# Third-party
import pytest

sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

# Internal
import fxhoudinimcp_server.handlers.parameter_handlers as parameters  # noqa: E402


class _OperationFailed(Exception):
    pass


@pytest.fixture(autouse=True)
def _hou(monkeypatch):
    monkeypatch.setattr(parameters, "_parm_type_name", lambda pt: pt.kind)
    monkeypatch.setattr(parameters.hou, "OperationFailed", _OperationFailed)
    # The shared hou mock has no real Vector/Matrix types to test against.
    monkeypatch.setattr(parameters, "_serialize_value", lambda value: value)


def _parm(name, value, raw=None, kind="String", label=None):
    parm = MagicMock()
    parm.name.return_value = name
    parm.eval.return_value = value
    parm.rawValue.return_value = raw if raw is not None else str(value)
    parm.isAtDefault.return_value = False
    template = MagicMock()
    template.kind = kind
    template.label.return_value = label or name.title()
    template.type.return_value.name.return_value = kind
    parm.parmTemplate.return_value = template
    return parm


def _node(path, type_name, parms):
    node = MagicMock()
    node.path.return_value = path
    node.type.return_value.name.return_value = type_name
    node.parms.return_value = list(parms)
    return node


def _image(path, raw):
    return _node(
        path,
        "mtlximage",
        [
            _parm("file", raw.replace("$JOB", "/work/proj"), raw=raw),
            _parm("filtertype", 1, kind="Int"),
        ],
    )


def _library(monkeypatch, children, descendants=None):
    parent = MagicMock()
    parent.path.return_value = "/mat/lib"
    parent.children.return_value = list(children)
    parent.allSubChildren.return_value = list(descendants or children)
    monkeypatch.setattr(parameters.hou, "node", lambda path: parent if path == "/mat/lib" else None)
    return parent


class TestGetParametersInsideANetwork:
    def test_one_call_one_row_per_parm_with_the_raw_text(self, monkeypatch):
        _library(
            monkeypatch,
            [
                _image("/mat/lib/base_color", "$JOB/tex/wood_col.exr"),
                _image("/mat/lib/rough", "/abs/tex/wood_rough.exr"),
            ],
        )
        result = parameters._get_parameters(inside="/mat/lib", patterns=["file"])
        assert [(r["node"], r["parm"], r.get("raw_value")) for r in result["rows"]] == [
            ("/mat/lib/base_color", "file", "$JOB/tex/wood_col.exr"),
            ("/mat/lib/rough", "file", None),
        ]
        assert result["rows"][0]["value"] == "/work/proj/tex/wood_col.exr"
        assert result["nodes_scanned"] == 2
        assert result["nodes_matched"] == 2
        assert result["truncated"] is False

    def test_patterns_are_required(self, monkeypatch):
        _library(monkeypatch, [_image("/mat/lib/a", "$JOB/a.exr")])
        with pytest.raises(ValueError, match="patterns is required"):
            parameters._get_parameters(inside="/mat/lib")

    def test_node_path_and_inside_together_are_refused(self):
        with pytest.raises(ValueError, match="not both"):
            parameters._get_parameters(
                node_path="/mat/lib/rough", inside="/mat/lib", patterns=["file"]
            )

    def test_neither_node_path_nor_inside_is_refused(self):
        with pytest.raises(ValueError, match="node_path is required"):
            parameters._get_parameters(patterns=["file"])

    def test_a_missing_network_is_named(self, monkeypatch):
        _library(monkeypatch, [])
        with pytest.raises(_OperationFailed, match="/mat/nope"):
            parameters._get_parameters(inside="/mat/nope", patterns=["file"])

    def test_node_type_narrows_and_recursive_reaches_descendants(self, monkeypatch):
        image = _image("/mat/lib/sub/img", "$JOB/a.exr")
        other = _node("/mat/lib/sub/null", "null", [_parm("file", "x")])
        parent = _library(monkeypatch, [], descendants=[image, other])
        result = parameters._get_parameters(
            inside="/mat/lib", patterns=["file"], recursive=True, node_type="mtlximage"
        )
        assert [r["node"] for r in result["rows"]] == ["/mat/lib/sub/img"]
        assert result["nodes_scanned"] == 1
        parent.children.assert_not_called()

    def test_a_button_matched_by_label_is_not_a_row(self, monkeypatch):
        node = _node(
            "/mat/lib/cache",
            "filecache",
            [_parm("file", "a.bgeo"), _parm("reload", 0, kind="Button", label="Reload File")],
        )
        _library(monkeypatch, [node])
        result = parameters._get_parameters(inside="/mat/lib", patterns=["file"])
        assert [r["parm"] for r in result["rows"]] == ["file"]

    def test_rows_past_the_cap_are_counted_and_flagged(self, monkeypatch):
        monkeypatch.setattr(parameters, "_SWEEP_ROW_CAP", 3)
        _library(monkeypatch, [_image(f"/mat/lib/img{i}", "$JOB/a.exr") for i in range(5)])
        result = parameters._get_parameters(inside="/mat/lib", patterns="file")
        assert result["returned"] == 3
        assert result["matched"] == 5
        assert result["truncated"] is True
        assert result["patterns"] == ["file"]

    def test_include_defaults_reaches_the_rows(self, monkeypatch):
        _library(monkeypatch, [_image("/mat/lib/a", "$JOB/a.exr")])
        result = parameters._get_parameters(
            inside="/mat/lib", patterns=["file"], include_defaults=True
        )
        assert result["rows"][0]["is_at_default"] is False

    def test_a_single_node_still_answers_parameters(self, monkeypatch):
        node = _image("/mat/lib/a", "$JOB/a.exr")
        monkeypatch.setattr(parameters.hou, "node", lambda path: node)
        result = parameters._get_parameters("/mat/lib/a", patterns=["file"])
        assert result["parameters"]["file"]["raw_value"] == "$JOB/a.exr"
        assert "rows" not in result
