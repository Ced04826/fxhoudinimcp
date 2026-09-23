"""get_attrib_stats refused vertex attributes.

uv and N usually live on vertices, and the UV range is the number that decides
a texture's tiling multiplier, yet attrib_class="vertex" answered "has no
per-element statistics". Vertex is now accepted; values come from one
{point,prim,vertex}{Float,Int}AttribValues call, element_count from the
geometry's intrinsics, and the per-element loop is only the fallback.

hou is mocked here; the live check ran on Houdini 22.0.429.
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
import fxhoudinimcp_server.handlers.geometry_handlers as geometry  # noqa: E402

hou = geometry.hou


def _attrib(name, size, data_type):
    attrib = MagicMock()
    attrib.name.return_value = name
    attrib.isArrayType.return_value = False
    attrib.dataType.return_value = data_type
    attrib.size.return_value = size
    return attrib


def _geometry(monkeypatch, counts):
    geo = MagicMock()
    geo.intrinsicValue.side_effect = lambda name: counts[name]
    monkeypatch.setattr(geometry, "_get_sop_geo", lambda path: geo)
    # hou is a MagicMock here, and an except clause needs a real class.
    monkeypatch.setattr(hou, "OperationFailed", RuntimeError)
    return geo


class TestVertexStatistics:
    def test_uv_on_vertices_has_a_range(self, monkeypatch):
        geo = _geometry(monkeypatch, {"vertexcount": 4})
        geo.vertexAttribs.return_value = [_attrib("uv", 3, hou.attribData.Float)]
        geo.vertexFloatAttribValues.return_value = [0, 0, 0, 1, 0, 0, 1, 4, 0, 0, 4, 0]
        result = geometry._get_attrib_stats(
            node_path="/obj/geo1/uvproject1", attribs=["uv"], attrib_class="vertex"
        )
        entry = result["stats"]["uv"]
        assert result["element_count"] == 4
        assert entry["count"] == 4
        assert entry["per_component"][0]["max"] == 1
        assert entry["per_component"][1]["max"] == 4
        geo.vertexFloatAttribValues.assert_called_once_with("uv")
        geo.prims.assert_not_called()  # the fast path, no per-vertex loop

    def test_vertices_fall_back_to_the_element_loop(self, monkeypatch):
        geo = _geometry(monkeypatch, {"vertexcount": 3})
        uv = _attrib("uv", 2, hou.attribData.Float)
        geo.vertexAttribs.return_value = [uv]
        geo.vertexFloatAttribValues.side_effect = AttributeError
        vertices = []
        for u in (-4.5, 0.5, 5.5):
            vertex = MagicMock()
            vertex.attribValue.return_value = (u, 0.5)
            vertices.append(vertex)
        prim = MagicMock()
        prim.vertices.return_value = vertices
        geo.prims.return_value = [prim]
        result = geometry._get_attrib_stats(
            node_path="/obj/geo1/uvproject1", attribs=["uv"], attrib_class="vertex"
        )
        entry = result["stats"]["uv"]
        assert entry["count"] == 3
        assert entry["per_component"][0] == {"min": -4.5, "max": 5.5, "mean": 0.5}
        assert entry["per_component"][1]["min"] == entry["per_component"][1]["max"] == 0.5


class TestFastPathsForEveryClass:
    def test_an_int_prim_attribute_reads_in_one_call(self, monkeypatch):
        geo = _geometry(monkeypatch, {"primitivecount": 3})
        geo.primAttribs.return_value = [_attrib("piece", 1, hou.attribData.Int)]
        geo.primIntAttribValues.return_value = [0, 2, 7]
        result = geometry._get_attrib_stats(
            node_path="/obj/geo1/out", attribs=["piece"], attrib_class="prim"
        )
        entry = result["stats"]["piece"]
        assert (entry["min"], entry["max"], entry["count"]) == (0, 7, 3)
        assert result["element_count"] == 3
        geo.primIntAttribValues.assert_called_once_with("piece")
        geo.prims.assert_not_called()

    def test_points_still_read_through_the_float_call(self, monkeypatch):
        geo = _geometry(monkeypatch, {"pointcount": 2})
        geo.pointAttribs.return_value = [_attrib("fuel", 1, hou.attribData.Float)]
        geo.pointFloatAttribValues.return_value = [0.25, 0.75]
        result = geometry._get_attrib_stats(node_path="/obj/geo1/out", attribs=["fuel"])
        assert result["stats"]["fuel"]["mean"] == pytest.approx(0.5)
        assert result["element_count"] == 2
        geo.points.assert_not_called()

    def test_element_count_falls_back_when_the_intrinsic_does_not_answer(self, monkeypatch):
        geo = _geometry(monkeypatch, {})  # every intrinsic lookup raises
        geo.pointAttribs.return_value = [_attrib("fuel", 1, hou.attribData.Float)]
        geo.pointFloatAttribValues.return_value = [1.0, 2.0, 3.0]
        geo.points.return_value = [object(), object(), object()]
        result = geometry._get_attrib_stats(node_path="/obj/geo1/out", attribs=["fuel"])
        assert result["element_count"] == 3


class TestClientTool:
    @pytest.mark.asyncio
    async def test_vertex_class_reaches_the_handler(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.geometry import get_attrib_stats

        await get_attrib_stats(
            mock_ctx, node_path="/obj/geo1/uvproject1", attribs=["uv"], attrib_class="vertex"
        )
        mock_bridge.execute.assert_called_once_with(
            "geometry.get_attrib_stats",
            {"node_path": "/obj/geo1/uvproject1", "attrib_class": "vertex", "attribs": ["uv"]},
        )
