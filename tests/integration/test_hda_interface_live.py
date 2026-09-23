"""Live HDA-interface tests: written through the definition, read back off it."""

from __future__ import annotations

# Third-party
import hou
import pytest

pytestmark = pytest.mark.integration

CONTROLS = {
    "name": "controls",
    "label": "Controls",
    "type": "folder",
    "children": [
        {"name": "stud_count", "type": "int", "default": 4, "min": 1, "max": 8},
        {"name": "clr", "type": "color"},
    ],
}


@pytest.fixture
def asset(tmp_path):
    subnet = hou.node("/obj").createNode("subnet", "brick")
    library = str(tmp_path / "brick.hda").replace("\\", "/")
    node = subnet.createDigitalAsset(
        name="fxh::brick_live::1.0", hda_file_name=library, description="Brick"
    )
    yield node
    definition = node.type().definition()
    node.destroy()
    definition.destroy()


class TestSetHdaInterface:
    def test_a_tab_folder_renamed_by_houdini_is_reported_and_its_parms_land(self, call, asset):
        data = call("hda.set_hda_interface", node_path=asset.path(), parameters=[CONTROLS])
        renamed = data["renamed_by_houdini"]
        assert [r["requested"] for r in renamed] == ["controls"]
        assert renamed[0]["label"] == "Controls"
        assert data["not_found_after_write"] == []
        # The Transform/Subnet tabs were renumbered with the series, and said so.
        assert {r["label"] for r in data["renumbered_by_houdini"]} >= {"Transform"}
        # clr lands as clrr/clrg/clrb and is found through its tuple.
        assert data["instance_parms_missing"] == []
        assert asset.parm("stud_count").eval() == 4

    def test_a_retry_is_refused_by_name_and_writes_nothing(self, call, asset):
        call("hda.set_hda_interface", node_path=asset.path(), parameters=[CONTROLS])
        before = asset.type().definition().parmTemplateGroup().asDialogScript()
        error = call(
            "hda.set_hda_interface",
            node_path=asset.path(),
            parameters=[CONTROLS],
            expect_error=True,
        )
        assert "already exist" in error["message"]
        assert asset.type().definition().parmTemplateGroup().asDialogScript() == before


class TestEditHdaInterface:
    def test_a_parameter_of_a_removed_folder_can_be_inserted_again(self, call, asset):
        call("hda.set_hda_interface", node_path=asset.path(), parameters=[CONTROLS])
        data = call(
            "hda.edit_hda_interface",
            node_path=asset.path(),
            ops=[
                {"op": "remove", "name": "Controls"},
                {"op": "insert", "spec": {"name": "stud_count", "type": "int", "default": 2}},
            ],
        )
        assert data["instance_parms_missing"] == []
        assert asset.parm("stud_count").eval() == 2
