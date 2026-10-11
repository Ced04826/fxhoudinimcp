"""USD reads at another frame, and world transforms over frames.

A LOP stage holds what its nodes authored at the frame they cooked on, so an
Xform LOP with ty = $F carries one sample at the current frame. Read at
time=10 from frame 1, get_usd_attribute answered ty = 1 under "time": 10.
"""

from __future__ import annotations

# Third-party
import hou
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def animated(call):
    """/parent (tx=2, ry=90) over /parent/cube (ty=$F)."""
    stage = hou.node("/stage")
    cube = stage.createNode("cube")
    cube.parm("primpath").set("/parent/cube")
    child = stage.createNode("xform")
    child.setFirstInput(cube)
    child.parm("primpattern").set("/parent/cube")
    child.parm("ty").setExpression("$F")
    parent = stage.createNode("xform")
    parent.setFirstInput(child)
    parent.parm("primpattern").set("/parent")
    parent.parm("tx").set(2)
    parent.parm("ry").set(90)
    hou.setFrame(1)
    return parent.path()


@pytest.mark.parametrize(
    "command,extra",
    [
        ("lops.get_usd_attribute", {"attr_name": "xformOp:translate"}),
        ("lops.get_usd_prim", {"attr_patterns": ["xformOp:translate"]}),
    ],
)
def test_a_requested_time_is_cooked_not_just_labelled(call, animated, command, extra):
    result = call(command, node_path=animated, prim_path="/parent/cube", time=10, **extra)
    value = result["value"] if "value" in result else result["prim"]["attributes"][0]["value"]
    assert value[1] == pytest.approx(10.0)
    assert hou.frame() == 1


def test_world_transform_composes_parents_at_each_frame(call, animated):
    result = call(
        "lops.get_usd_world_transform",
        node_path=animated,
        prim_paths=["/parent/cube", "/parent"],
        frames=[1, 24],
    )
    cube = result["prims"]["/parent/cube"]
    assert [entry["translate"] for entry in cube] == [
        pytest.approx([2, 1, 0], abs=1e-6),
        pytest.approx([2, 24, 0], abs=1e-6),
    ]
    assert cube[0]["rotate"][1] == pytest.approx(90)
    assert result["prims"]["/parent"][1]["translate"] == pytest.approx([2, 0, 0], abs=1e-6)
    assert hou.frame() == 1


def test_world_transform_names_a_missing_prim(call, animated):
    error = call(
        "lops.get_usd_world_transform",
        node_path=animated,
        prim_paths=["/nope"],
        expect_error=True,
    )
    assert "/nope" in error["message"]
