"""Tests for get_usd_bound_material.

get_usd_materials reads each prim's direct binding only, so a prim that
inherits its material from an ancestor, or receives it through a collection,
is listed as unbound -- while the renderer, resolving the binding with
ComputeBoundMaterial, shades it. get_usd_bound_material answers the way the
renderer does and says where the binding comes from: the prim itself, which
ancestor, or which collection. Measured live on Houdini 22.0.429: a sphere
under /grp with the material assigned to /grp resolves to the material with
`kind: inherited`, `binding_prim: /grp`, `strength: weakerThanDescendants`.

hou and pxr are mocked here.
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


class _Path:
    def __init__(self, path):
        self._path = path

    def __str__(self):
        return self._path


def _prim(path, valid=True):
    prim = MagicMock()
    prim.IsValid.return_value = valid
    prim.__bool__ = lambda self: valid
    prim.GetPath.return_value = _Path(path)
    return prim


def _rel(binding_prim, name="material:binding"):
    rel = MagicMock()
    rel.GetPath.return_value = _Path(f"{binding_prim}.{name}")
    rel.GetPrim.return_value.GetPath.return_value = _Path(binding_prim)
    rel.GetName.return_value = name
    return rel


def _material(path):
    material = MagicMock()
    material.GetPath.return_value = _Path(path)
    material.GetPrim.return_value.IsValid.return_value = True
    material.__bool__ = lambda self: True
    return material


def _stage(monkeypatch, bindings, direct=None):
    """bindings: prim path -> (material path | None, binding prim, rel name)."""
    monkeypatch.setattr(lops, "HAS_PXR", True)
    stage = MagicMock()
    prims = {path: _prim(path) for path in bindings}

    def prim_at(path):
        return prims.get(path, _prim(path, valid=False))

    stage.GetPrimAtPath.side_effect = prim_at
    monkeypatch.setattr(lops, "_get_lop_stage", lambda node_path: stage)

    api_for = {}

    def api(prim):
        path = str(prim.GetPath())
        entry = bindings.get(path)
        api_obj = MagicMock()
        if entry and entry[0]:
            api_obj.ComputeBoundMaterial.return_value = (
                _material(entry[0]),
                _rel(entry[1], entry[2]),
            )
        else:
            api_obj.ComputeBoundMaterial.return_value = (None, None)
        api_obj.GetDirectBinding.return_value.GetMaterialPath.return_value = _Path(
            (direct or {}).get(path, "")
        )
        api_for[path] = api_obj
        return api_obj

    usd_shade = MagicMock()
    usd_shade.MaterialBindingAPI.side_effect = api
    usd_shade.MaterialBindingAPI.GetMaterialBindingStrength.return_value = "weakerThanDescendants"
    usd_shade.Tokens.allPurpose = "allPurpose"
    usd_shade.Tokens.full = "full"
    usd_shade.Tokens.preview = "preview"
    monkeypatch.setattr(lops, "UsdShade", usd_shade, raising=False)
    return api_for


class TestBoundMaterial:
    def test_an_inherited_binding_names_the_ancestor(self, monkeypatch):
        _stage(
            monkeypatch,
            {
                "/grp/sph1": ("/materials/srf", "/grp", "material:binding"),
                "/grp": ("/materials/srf", "/grp", "material:binding"),
            },
            direct={"/grp": "/materials/srf"},
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
        assert parent["direct_binding"] == "/materials/srf"
        assert reply["bound"] == 2

    def test_a_collection_binding_names_the_collection(self, monkeypatch):
        _stage(
            monkeypatch,
            {"/set/chair": ("/materials/wood", "/set", "material:binding:collection:furniture")},
        )
        reply = lops._get_usd_bound_material(node_path="/stage/x", prim_paths="/set/chair")
        source = reply["bindings"][0]["source"]
        assert source["kind"] == "collection"
        assert source["collection"] == "furniture"

    def test_unbound_and_missing_prims_answer_in_their_own_rows(self, monkeypatch):
        _stage(monkeypatch, {"/materials": (None, None, None)})
        reply = lops._get_usd_bound_material(
            node_path="/stage/x", prim_paths=["/materials", "/nope"]
        )
        unbound, missing = reply["bindings"]
        assert unbound["material"] is None
        assert "source" not in unbound
        assert missing["error"] == "prim not found on this stage"
        assert reply["bound"] == 0

    def test_purpose_is_passed_as_the_usd_token(self, monkeypatch):
        api_for = _stage(monkeypatch, {"/p": ("/materials/m", "/p", "material:binding")})
        lops._get_usd_bound_material(node_path="/stage/x", prim_paths=["/p"], purpose="preview")
        api_for["/p"].ComputeBoundMaterial.assert_called_once_with("preview")

    def test_an_unknown_purpose_is_refused(self, monkeypatch):
        _stage(monkeypatch, {})
        with pytest.raises(ValueError, match="purpose must be one of"):
            lops._get_usd_bound_material(node_path="/stage/x", prim_paths=["/p"], purpose="render")

    def test_an_empty_prim_list_is_refused(self, monkeypatch):
        _stage(monkeypatch, {})
        with pytest.raises(ValueError, match="non-empty list"):
            lops._get_usd_bound_material(node_path="/stage/x", prim_paths=[])
