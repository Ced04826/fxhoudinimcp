"""get_prim_intrinsics read one prim per call.

Finding the packed prims whose `bounds` ran away among 281 of them took 281
calls, about 14 s at 50 ms each, or a sweep in execute_python. prim_indices,
prim_range and intrinsics now read many prims in one call and answer with a
table plus min/max/avg and the prim each extreme belongs to.

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


class _FakePrim:
    def __init__(self, bounds, type_name="PackedGeometry"):
        self._bounds = bounds
        self._type = type_name

    def type(self):
        return MagicMock(**{"name.return_value": self._type})

    def intrinsicNames(self):  # noqa: N802 - HOM spelling
        return ("bounds", "primitivetype")

    def intrinsicValue(self, name):  # noqa: N802
        if name == "bounds":
            return self._bounds
        if name == "primitivetype":
            return self._type
        raise ValueError(name)


def _geometry(monkeypatch, prims):
    geo = MagicMock()
    geo.intrinsicValue.return_value = len(prims)
    geo.prim.side_effect = lambda i: prims[i]
    monkeypatch.setattr(geometry, "_get_sop_geo", lambda path: geo)
    monkeypatch.setattr(geometry, "_vec_to_list", lambda v: v)
    monkeypatch.setattr(geometry.hou, "OperationFailed", RuntimeError)
    return geo


@pytest.fixture
def five_prims(monkeypatch):
    return _geometry(monkeypatch, [_FakePrim(float(i * 10)) for i in range(5)])


class TestPrimIntrinsicsInOneCall:
    def test_one_intrinsic_sweeps_every_prim_with_min_and_max(self, five_prims):
        result = geometry._get_prim_intrinsics(node_path="/obj/geo1/pack1", intrinsics=["bounds"])
        assert result["prim_count"] == 5
        assert result["total_prims"] == 5
        assert result["stats"]["bounds"]["min"] == 0.0
        assert result["stats"]["bounds"]["max"] == 40.0
        assert result["stats"]["bounds"]["max_prim"] == 4
        assert result["prims"][2]["bounds"] == 20.0
        assert "truncated" not in result

    def test_a_vector_intrinsic_gets_per_component_extremes(self, monkeypatch):
        # `bounds` is six floats: one min/max over whole lists answers
        # nothing, and "whose bounds ran away" is what the table is for.
        _geometry(monkeypatch, [_FakePrim([0.0, float(i), -float(i)]) for i in range(4)])
        result = geometry._get_prim_intrinsics(node_path="/obj/geo1/pack1", intrinsics=["bounds"])
        components = result["stats"]["bounds"]["components"]
        assert [c["index"] for c in components] == [0, 1, 2]
        assert components[1]["max"] == 3.0
        assert components[1]["max_prim"] == 3
        assert components[2]["min"] == -3.0
        assert components[2]["min_prim"] == 3

    def test_a_range_reads_just_those_prims(self, five_prims):
        result = geometry._get_prim_intrinsics(
            node_path="/obj/geo1/pack1", prim_range=[1, 3], intrinsics=["bounds"]
        )
        assert [row["prim_index"] for row in result["prims"]] == [1, 2, 3]
        assert result["stats"]["bounds"]["min_prim"] == 1

    def test_a_list_of_indices_works_too(self, five_prims):
        result = geometry._get_prim_intrinsics(
            node_path="/obj/geo1/pack1", prim_indices=[0, 4], intrinsics=["bounds"]
        )
        assert [row["prim_index"] for row in result["prims"]] == [0, 4]

    def test_indices_without_intrinsics_read_every_intrinsic(self, five_prims):
        result = geometry._get_prim_intrinsics(node_path="/obj/geo1/pack1", prim_indices=[1])
        assert result["prims"][0]["primitivetype"] == "PackedGeometry"
        assert result["prims"][0]["bounds"] == 10.0

    def test_an_out_of_range_index_is_refused(self, five_prims):
        with pytest.raises(RuntimeError, match="out of range"):
            geometry._get_prim_intrinsics(node_path="/obj/geo1/pack1", prim_indices=[9])

    def test_both_index_arguments_at_once_is_refused(self, five_prims):
        with pytest.raises(RuntimeError, match="not both"):
            geometry._get_prim_intrinsics(
                node_path="/obj/geo1/pack1", prim_indices=[0], prim_range=[0, 1]
            )

    @pytest.mark.parametrize("bad", [[0, 10**9], [3, 1], [-1, 2]])
    def test_a_bad_range_is_refused_without_building_it(self, five_prims, bad):
        with pytest.raises(RuntimeError, match="out of range"):
            geometry._get_prim_intrinsics(node_path="/obj/geo1/pack1", prim_range=bad)

    def test_a_malformed_range_is_refused(self, five_prims):
        with pytest.raises(RuntimeError, match=r"\[start, end\]"):
            geometry._get_prim_intrinsics(node_path="/obj/geo1/pack1", prim_range=[0])

    def test_a_long_request_is_windowed_and_says_so(self, monkeypatch):
        monkeypatch.setattr(geometry, "_INTRINSIC_ROW_CAP", 3)
        _geometry(monkeypatch, [_FakePrim(float(i)) for i in range(5)])
        result = geometry._get_prim_intrinsics(node_path="/obj/geo1/pack1", intrinsics=["bounds"])
        assert result["prim_count"] == 3
        assert result["truncated"] is True
        assert result["requested_count"] == 5

    def test_a_single_prim_still_answers_the_old_shape(self, five_prims):
        result = geometry._get_prim_intrinsics(node_path="/obj/geo1/pack1", prim_index=2)
        assert result["prim_index"] == 2
        assert result["prim_type"] == "PackedGeometry"
        assert result["intrinsics"] == {"bounds": 20.0, "primitivetype": "PackedGeometry"}

    def test_no_arguments_still_answers_the_summary(self, five_prims):
        result = geometry._get_prim_intrinsics(node_path="/obj/geo1/pack1")
        assert result["total_prims"] == 5
        assert result["summary"]["bounds"]["max"] == 40.0
        assert "prims" not in result


class TestClientTool:
    @pytest.mark.asyncio
    async def test_batch_arguments_reach_the_handler(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.geometry import get_prim_intrinsics

        await get_prim_intrinsics(
            mock_ctx, node_path="/obj/geo1/pack1", prim_range=[0, 9], intrinsics=["bounds"]
        )
        mock_bridge.execute.assert_called_once_with(
            "geometry.get_prim_intrinsics",
            {"node_path": "/obj/geo1/pack1", "prim_range": [0, 9], "intrinsics": ["bounds"]},
        )

    @pytest.mark.asyncio
    async def test_the_old_call_sends_what_it_always_sent(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.geometry import get_prim_intrinsics

        await get_prim_intrinsics(mock_ctx, node_path="/obj/geo1/pack1", prim_index=3)
        mock_bridge.execute.assert_called_once_with(
            "geometry.get_prim_intrinsics",
            {"node_path": "/obj/geo1/pack1", "prim_index": 3},
        )
