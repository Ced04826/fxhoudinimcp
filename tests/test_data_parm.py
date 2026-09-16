"""Tests for get_parameter on a Data parameter.

A Data parameter (a Curve SOP's stash, a Stash SOP's geometry) holds a blob:
eval() is a hou.Geometry or None and rawValue() is empty either way, so
get_parameter answered `null` whether the parameter was empty or held a whole
mesh. It now reports whether the blob is set, what it holds and how big it is.

hou is mocked here; the handler is exercised directly.
"""

from __future__ import annotations

# Built-in
import os
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

# Internal
import fxhoudinimcp_server.handlers.parameter_handlers as parameters  # noqa: E402


def _geometry_type():
    geometry = MagicMock(spec=["intrinsicValue"])
    parameters.hou.Geometry = type(geometry)
    return geometry


class TestDataParmSummary:
    def test_a_set_geometry_blob_reports_its_counts(self):
        geometry = _geometry_type()
        geometry.intrinsicValue.side_effect = lambda name: {
            "pointcount": 8,
            "primitivecount": 6,
        }[name]
        parm = MagicMock()
        parm.eval.return_value = geometry
        parm.asData.return_value = {"geometry": "x" * 120}
        template = MagicMock()
        template.dataParmType.return_value.name.return_value = "Geometry"
        summary = parameters._data_parm_summary(parm, template)
        assert summary["is_set"] is True
        assert summary["geometry"] == {"points": 8, "prims": 6}
        assert summary["size_bytes"] == 120

    def test_an_unset_blob_is_not_null_but_unset(self):
        _geometry_type()
        parm = MagicMock()
        parm.eval.return_value = None
        parm.asData.return_value = {"geometry": ""}
        template = MagicMock()
        template.dataParmType.return_value.name.return_value = "Geometry"
        summary = parameters._data_parm_summary(parm, template)
        assert summary["is_set"] is False
        assert summary["size_bytes"] == 0


class TestGetParameterOnData:
    def test_a_data_parameter_answers_with_the_summary_not_a_value(self, monkeypatch):
        _geometry_type()
        parm = MagicMock()
        parm.name.return_value = "stash"
        parm.eval.return_value = None
        parm.asData.return_value = {"geometry": ""}
        parm.isLocked.return_value = False
        parm.isAtDefault.return_value = True
        template = parm.parmTemplate.return_value
        template.dataParmType.return_value.name.return_value = "Geometry"
        monkeypatch.setattr(parameters, "_resolve_parm", lambda path, name: parm)
        monkeypatch.setattr(parameters, "_parm_type_name", lambda pt: "Data")
        result = parameters._get_parameter("/obj/geo1/stash1", "stash")
        assert result["parm_type"] == "Data"
        assert result["data"]["is_set"] is False
        assert "value" not in result
        parm.rawValue.assert_not_called()

    def test_an_ordinary_parameter_is_untouched(self, monkeypatch):
        parm = MagicMock()
        parm.eval.return_value = 2.5
        parm.rawValue.return_value = "2.5"
        parm.keyframes.return_value = []
        monkeypatch.setattr(parameters, "_resolve_parm", lambda path, name: parm)
        monkeypatch.setattr(parameters, "_parm_type_name", lambda pt: "Float")
        monkeypatch.setattr(parameters, "_serialize_value", lambda v: v)
        result = parameters._get_parameter("/obj/geo1/xform1", "tx")
        assert result["value"] == 2.5
        assert "data" not in result
