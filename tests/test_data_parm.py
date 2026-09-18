"""Tests for get_parameter / get_parameters on a Data parameter.

A Data parameter (a Curve SOP's stash, a Stash SOP's geometry) holds a blob:
eval() is a hou.Geometry or None and rawValue() is empty either way, so
get_parameter answered `null` for an empty parameter and a geometry summary
for a set one, with nothing saying which case a caller was looking at. Both
readers now add `data`: whether the blob is set and what it holds.

hou is mocked here and only through monkeypatch; the live path was checked on
Houdini 22.0.429 (a Stash SOP before and after Stash Input).
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


class _Geometry:
    def __init__(self, points=8, prims=6, vertices=24):
        self._counts = {"pointcount": points, "primitivecount": prims, "vertexcount": vertices}

    def intrinsicValue(self, name):
        return self._counts[name]


@pytest.fixture(autouse=True)
def _hou(monkeypatch):
    monkeypatch.setattr(parameters.hou, "Geometry", _Geometry)
    monkeypatch.setattr(parameters, "_parm_type_name", lambda pt: pt.kind)
    monkeypatch.setattr(parameters.hou, "OperationFailed", _OperationFailed)
    # The shared hou mock has no real Vector/Matrix types to test against.
    monkeypatch.setattr(parameters, "_serialize_value", lambda value: value)


def _data_parm(value):
    parm = MagicMock(
        spec=[
            "eval",
            "rawValue",
            "name",
            "parmTemplate",
            "isLocked",
            "isAtDefault",
            "expression",
            "expressionLanguage",
            "keyframes",
        ]
    )
    parm.eval.return_value = value
    parm.rawValue.return_value = ""
    parm.name.return_value = "stash"
    template = MagicMock()
    template.kind = "Data"
    template.dataParmType.return_value.name.return_value = "Geometry"
    parm.parmTemplate.return_value = template
    parm.isLocked.return_value = False
    parm.isAtDefault.return_value = value is None
    parm.expression.side_effect = _OperationFailed
    parm.keyframes.return_value = ()
    return parm, template


class TestDataParmSummary:
    def test_a_set_geometry_blob_reports_the_standard_geometry_summary(self):
        parm, template = _data_parm(_Geometry())
        summary = parameters._data_parm_summary(parm, template)
        assert summary["is_set"] is True
        assert summary["geometry"] == {
            "type": "Geometry",
            "point_count": 8,
            "prim_count": 6,
            "vertex_count": 24,
        }

    def test_an_unset_blob_is_unset(self):
        parm, template = _data_parm(None)
        summary = parameters._data_parm_summary(parm, template)
        assert summary == {"is_set": False, "data_parm_type": "Geometry"}

    def test_an_empty_dictionary_is_unset(self):
        parm, template = _data_parm({})
        assert parameters._data_parm_summary(parm, template)["is_set"] is False

    def test_a_failing_intrinsic_drops_only_that_count(self):
        geometry = _Geometry()
        geometry._counts.pop("primitivecount")
        parm, template = _data_parm(geometry)
        summary = parameters._data_parm_summary(parm, template)
        assert summary["geometry"]["point_count"] == 8
        assert "prim_count" not in summary["geometry"]


class TestReaders:
    def test_get_parameter_keeps_value_and_raw_value_and_adds_data(self, monkeypatch):
        parm, _ = _data_parm(_Geometry())
        monkeypatch.setattr(parameters, "_resolve_parm", lambda node_path, parm_name: parm)
        result = parameters._get_parameter("/obj/geo1/stash1", "stash")
        assert result["value"]["point_count"] == 8
        assert result["raw_value"] == ""
        assert result["data"]["is_set"] is True
        assert result["parm_type"] == "Data"
        assert result["keyframe_count"] == 0

    def test_get_parameters_reports_data_the_same_way(self, monkeypatch):
        parm, _ = _data_parm(None)
        node = MagicMock()
        node.parms.return_value = [parm]
        node.type.return_value.name.return_value = "stash"
        monkeypatch.setattr(parameters.hou, "node", lambda path: node)
        result = parameters._get_parameters("/obj/geo1/stash1")
        entry = result["parameters"]["stash"]
        assert entry["value"] is None
        assert entry["data"]["is_set"] is False
