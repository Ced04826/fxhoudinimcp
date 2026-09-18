"""Live material-binding tests: resolution by purpose, inheritance, collections."""

from __future__ import annotations

# Third-party
import hou
import pytest

pytestmark = pytest.mark.integration

_STAGE = """
from pxr import Sdf, Usd, UsdGeom, UsdShade
stage = hou.pwd().editableStage()
UsdGeom.Xform.Define(stage, "/grp")
UsdGeom.Sphere.Define(stage, "/grp/sph1")
UsdGeom.Xform.Define(stage, "/set")
UsdGeom.Cube.Define(stage, "/set/chair")
srf = UsdShade.Material.Define(stage, "/materials/srf")
full = UsdShade.Material.Define(stage, "/materials/full_only")
furn = UsdShade.Material.Define(stage, "/materials/furn")
grp = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/grp"))
grp.Bind(srf)
grp.Bind(full, materialPurpose=UsdShade.Tokens.full)
set_prim = stage.GetPrimAtPath("/set")
furniture = Usd.CollectionAPI.Apply(set_prim, "furn")
furniture.CreateIncludesRel().AddTarget("/set/chair")
UsdShade.MaterialBindingAPI.Apply(set_prim).Bind(
    furniture, furn, bindingName="furn", materialPurpose=UsdShade.Tokens.preview
)
UsdGeom.Xform.Define(stage, "/broken")
UsdGeom.Sphere.Define(stage, "/broken/s")
broken = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/broken"))
broken.GetDirectBindingRel().SetTargets([Sdf.Path("/materials/missing")])
"""


@pytest.fixture
def lop():
    node = hou.node("/stage").createNode("pythonscript")
    node.parm("python").set(_STAGE)
    node.cook(force=True)
    return node.path()


class TestBoundMaterial:
    def test_the_default_is_the_full_purpose_karma_renders(self, call, lop):
        data = call("lops.get_usd_bound_material", node_path=lop, prim_paths=["/grp/sph1"])
        row = data["bindings"][0]
        assert row["material"] == "/materials/full_only"
        assert row["source"]["kind"] == "inherited"
        assert row["source"]["binding_prim"] == "/grp"
        all_purpose = call(
            "lops.get_usd_bound_material", node_path=lop, prim_paths=["/grp/sph1"], purpose="all"
        )
        assert all_purpose["bindings"][0]["material"] == "/materials/srf"

    def test_a_purpose_specific_collection_is_named_by_its_path(self, call, lop):
        data = call(
            "lops.get_usd_bound_material",
            node_path=lop,
            prim_paths=["/set/chair"],
            purpose="preview",
        )
        source = data["bindings"][0]["source"]
        assert source["kind"] == "collection"
        assert source["collection"] == "/set.collection:furn"

    def test_a_binding_to_a_missing_material_is_reported(self, call, lop):
        data = call("lops.get_usd_bound_material", node_path=lop, prim_paths=["/broken/s"])
        row = data["bindings"][0]
        assert row["material"] is None
        assert row["missing_material"] == ["/materials/missing"]


class TestMaterialsListWhereTheyRender:
    def test_inherited_geometry_is_listed_as_rendered_on(self, call, lop):
        data = call("lops.get_usd_materials", node_path=lop)
        by_path = {m["path"]: m for m in data["materials"]}
        assert by_path["/materials/srf"]["bound_to"] == ["/grp"]
        assert by_path["/materials/full_only"]["rendered_on"] == ["/grp/sph1"]
        assert by_path["/materials/srf"]["rendered_on_count"] == 0
