"""get_usd_bound_material: the material a prim renders with, and why.

get_usd_materials reports where bindings are authored, so a prim bound
through its parent or a collection looked unbound. This resolves bindings the
way the renderer does (UsdShade ComputeBoundMaterials, one batch) and names
the source. Behaviour measured on Houdini 22.0.429: the allPurpose token is
the empty string; ComputeBoundMaterial("full") returns a full-purpose binding
and falls back to an allPurpose one; a collection binding's relationship is
material:binding:collection:<purpose>:<name>; a binding to a missing material
prim returns an invalid material with a valid relationship.

pxr is faked here.
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
from _usd_fakes import Path, prim  # noqa: E402


def _rel(binding_prim, name="material:binding", collection=None, targets=()):
    rel = MagicMock()
    rel.GetPath.return_value = Path(f"{binding_prim}.{name}")
    rel.GetPrim.return_value.GetPath.return_value = Path(binding_prim)
    rel.GetName.return_value = name
    rel.GetTargets.return_value = [Path(t) for t in targets]
    rel.collection = collection
    rel.__bool__ = lambda self: True
    return rel


def _material(path, exists=True):
    material = MagicMock()
    material.GetPath.return_value = Path(path)
    material.GetPrim.return_value.IsValid.return_value = exists
    material.__bool__ = lambda self: exists
    return material


def _stage(monkeypatch, bindings, direct=None):
    """bindings: prim path -> (material | None, rel | None)."""
    monkeypatch.setattr(lops, "HAS_PXR", True)
    stage = MagicMock()
    prims = {path: prim(path) for path in bindings}
    stage.GetPrimAtPath.side_effect = lambda path: prims.get(path, prim(path, valid=False))
    monkeypatch.setattr(lops, "_get_lop_stage", lambda node_path: stage)

    calls = []

    def compute(prims_in, token):
        calls.append(([str(p.GetPath()) for p in prims_in], token))
        rows = [bindings[str(p.GetPath())] for p in prims_in]
        return [m for m, _ in rows], [r for _, r in rows]

    def api(target):
        api_obj = MagicMock()
        path = str(target.GetPath())
        api_obj.GetDirectBinding.side_effect = lambda token: MagicMock(
            GetMaterialPath=MagicMock(return_value=Path((direct or {}).get((path, token), "")))
        )
        return api_obj

    usd_shade = MagicMock()
    usd_shade.MaterialBindingAPI.side_effect = api
    usd_shade.MaterialBindingAPI.ComputeBoundMaterials.side_effect = compute
    usd_shade.MaterialBindingAPI.GetMaterialBindingStrength.return_value = "weakerThanDescendants"
    usd_shade.MaterialBindingAPI.CollectionBinding.IsCollectionBindingRel.side_effect = lambda rel: (
        rel.collection is not None
    )
    usd_shade.MaterialBindingAPI.CollectionBinding.side_effect = lambda rel: MagicMock(
        GetCollectionPath=MagicMock(return_value=Path(rel.collection))
    )
    # The real tokens: allPurpose is the empty string.
    usd_shade.Tokens.allPurpose = ""
    usd_shade.Tokens.full = "full"
    usd_shade.Tokens.preview = "preview"
    monkeypatch.setattr(lops, "UsdShade", usd_shade, raising=False)
    return calls


class TestBoundMaterial:
    def test_the_default_purpose_is_what_karma_renders_and_one_batch(self, monkeypatch):
        calls = _stage(
            monkeypatch,
            {
                "/grp/sph1": (
                    _material("/materials/full_only"),
                    _rel("/grp", "material:binding:full"),
                ),
                "/grp/sph2": (
                    _material("/materials/full_only"),
                    _rel("/grp", "material:binding:full"),
                ),
            },
        )
        lops._get_usd_bound_material(node_path="/stage/x", prim_paths=["/grp/sph1", "/grp/sph2"])
        assert calls == [(["/grp/sph1", "/grp/sph2"], "full")]

    def test_all_is_the_empty_all_purpose_token(self, monkeypatch):
        calls = _stage(monkeypatch, {"/p": (_material("/materials/m"), _rel("/p"))})
        lops._get_usd_bound_material(node_path="/stage/x", prim_paths=["/p"], purpose="all")
        assert calls[0][1] == ""

    def test_an_inherited_binding_names_the_ancestor(self, monkeypatch):
        _stage(
            monkeypatch,
            {
                "/grp/sph1": (_material("/materials/srf"), _rel("/grp")),
                "/grp": (_material("/materials/srf"), _rel("/grp")),
            },
            direct={("/grp", "full"): "/materials/srf"},
        )
        reply = lops._get_usd_bound_material(
            node_path="/stage/assign", prim_paths=["/grp/sph1", "/grp"]
        )
        child, parent = reply["bindings"]
        assert child["material"] == "/materials/srf"
        assert child["source"]["kind"] == "inherited"
        assert child["source"]["binding_prim"] == "/grp"
        assert child["source"]["strength"] == "weakerThanDescendants"
        assert child["direct_binding"] is None
        assert parent["source"]["kind"] == "direct"
        # The direct binding is read for the same purpose as the resolution.
        assert parent["direct_binding"] == "/materials/srf"
        assert reply["bound"] == 2

    def test_a_purpose_specific_collection_binding_names_the_collection(self, monkeypatch):
        rel = _rel(
            "/set", "material:binding:collection:preview:furn", collection="/set.collection:furn"
        )
        _stage(monkeypatch, {"/set/chair": (_material("/materials/furn"), rel)})
        reply = lops._get_usd_bound_material(
            node_path="/stage/x", prim_paths="/set/chair", purpose="preview"
        )
        source = reply["bindings"][0]["source"]
        assert source["kind"] == "collection"
        assert source["collection"] == "/set.collection:furn"

    def test_a_binding_to_a_missing_material_is_not_unbound(self, monkeypatch):
        rel = _rel("/broken", targets=["/materials/missing"])
        _stage(monkeypatch, {"/broken/s": (_material("", exists=False), rel)})
        reply = lops._get_usd_bound_material(node_path="/stage/x", prim_paths=["/broken/s"])
        row = reply["bindings"][0]
        assert row["material"] is None
        assert row["missing_material"] == ["/materials/missing"]
        assert row["source"]["binding_prim"] == "/broken"
        assert reply["missing_material"] == 1

    def test_every_row_has_a_material_key_and_the_counts_tell_rows_apart(self, monkeypatch):
        _stage(monkeypatch, {"/materials": (None, None)})
        reply = lops._get_usd_bound_material(
            node_path="/stage/x", prim_paths=["/materials", "/nope"]
        )
        unbound, missing = reply["bindings"]
        assert unbound["material"] is None and "source" not in unbound
        assert missing["material"] is None
        assert missing["error"] == "prim not found on this stage"
        assert (reply["count"], reply["bound"], reply["not_found"]) == (2, 0, 1)

    def test_an_unknown_purpose_is_refused(self, monkeypatch):
        _stage(monkeypatch, {})
        with pytest.raises(ValueError, match="purpose must be one of"):
            lops._get_usd_bound_material(node_path="/stage/x", prim_paths=["/p"], purpose="render")

    def test_an_empty_prim_list_is_refused(self, monkeypatch):
        _stage(monkeypatch, {})
        with pytest.raises(ValueError, match="non-empty list"):
            lops._get_usd_bound_material(node_path="/stage/x", prim_paths=[])
