"""get_usd_attribute on a time-sampled attribute answered `value: null`.

A PointInstancer's `positions` / `protoIndices`, authored per frame, have
nothing in the default slot, so Get() with the default time code returns
None: `is_authored: true` next to `value: null` read as "the attribute is
empty". With no time given and samples present, the current frame is read
(the first sample when the frame is outside the sampled range), and the
reply names the time used and the samples found.

hou and pxr are mocked here; the live check ran on Houdini 22.0.429.
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
import fxhoudinimcp_server.handlers.lops_handlers as lops  # noqa: E402


class _Attr:
    """A Usd.Attribute whose Get() answers per time code, None on default."""

    def __init__(self, samples=(), values=None, type_name="point3f[]"):
        self._samples = list(samples)
        self._values = values or {}
        self._type = type_name

    def IsValid(self):
        return True

    def IsAuthored(self):
        return True

    def GetTypeName(self):
        return self._type

    def GetTimeSamples(self):
        return self._samples

    def Get(self, time_code):
        return self._values.get(getattr(time_code, "value", None))


def _stage(monkeypatch, attr, frame=12.0):
    prim = MagicMock()
    prim.IsValid.return_value = True
    prim.GetAttribute.return_value = attr
    stage = MagicMock()
    stage.GetPrimAtPath.return_value = prim
    monkeypatch.setattr(lops, "_get_lop_stage", lambda node_path: stage)
    usd = MagicMock()
    usd.TimeCode.side_effect = lambda t: MagicMock(value=t)
    usd.TimeCode.Default.return_value = MagicMock(value=None)
    monkeypatch.setattr(lops, "Usd", usd, raising=False)
    monkeypatch.setattr(lops.hou, "frame", lambda: frame)


@pytest.fixture
def sampled():
    return _Attr(samples=[1.0, 24.0], values={1.0: [1, 2, 3], 12.0: [4, 5, 6]})


class TestTimeSampledAttributes:
    def test_no_time_reads_the_current_frame_and_says_so(self, monkeypatch, sampled):
        _stage(monkeypatch, sampled)
        reply = lops._get_usd_attribute(
            node_path="/stage/pi", prim_path="/pi", attr_name="positions"
        )
        assert reply["value"] == [4, 5, 6]
        assert reply["time"] == 12.0
        assert reply["requested_time"] is None
        assert reply["time_source"] == "stage_frame"
        assert reply["time_samples"] == 2
        assert reply["time_range"] == [1.0, 24.0]
        assert "time-sampled" in reply["note"]

    def test_a_frame_outside_the_samples_falls_back_to_the_first(self, monkeypatch, sampled):
        _stage(monkeypatch, sampled, frame=900.0)
        reply = lops._get_usd_attribute(
            node_path="/stage/pi", prim_path="/pi", attr_name="positions"
        )
        assert reply["value"] == [1, 2, 3]
        assert reply["time"] == 1.0
        assert reply["time_source"] == "first_sample"

    def test_an_explicit_time_is_honoured_without_a_note(self, monkeypatch, sampled):
        _stage(monkeypatch, sampled)
        reply = lops._get_usd_attribute(
            node_path="/stage/pi", prim_path="/pi", attr_name="positions", time=1.0
        )
        assert reply["value"] == [1, 2, 3]
        assert reply["time"] == 1.0
        assert reply["time_source"] == "requested"
        assert reply["time_samples"] == 2
        assert "note" not in reply

    def test_a_static_attribute_reads_the_default_slot_as_before(self, monkeypatch):
        _stage(monkeypatch, _Attr(values={None: "rightHanded"}, type_name="token"))
        reply = lops._get_usd_attribute(
            node_path="/stage/x", prim_path="/p", attr_name="orientation"
        )
        assert reply["value"] == "rightHanded"
        assert reply["time"] is None
        assert reply["time_source"] == "default"
        assert "time_samples" not in reply
        assert "note" not in reply
