"""The enumerating USD tools did not see prims under instanceable prototypes.

The default USD walk stops at an instanceable prim: its children live on the
prototype and GetChildren() answers (). list_usd_prims showed the prototype
with `children: []` and get_usd_prim_stats counted one Mesh where five were
under it. `traverse_instance_proxies` on list_usd_prims, get_usd_prim,
get_usd_prim_stats and find_usd_prims walks through instances; without it an
instanceable prim is flagged with how many prims it hides.

hou and pxr are mocked here; the live check ran on Houdini 22.0.429.
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
import fxhoudinimcp_server.handlers.lops_handlers as lops  # noqa: E402

# Shared with the other USD handler tests.
from _usd_fakes import prim as _prim  # noqa: E402


def _usd(monkeypatch):
    usd = MagicMock()
    usd.TraverseInstanceProxies.return_value = "PROXIES"
    usd.PrimRange.side_effect = lambda *args: list(args)
    usd.ModelAPI.return_value = None
    monkeypatch.setattr(lops, "Usd", usd, raising=False)
    return usd


def _instanceable(path, instanceable, type_name="Xform"):
    fake = _prim(path, type_name=type_name)
    fake.IsInstanceable.return_value = instanceable
    fake.IsInstanceProxy.return_value = False
    return fake


class TestTraversal:
    def test_the_predicate_is_used_only_when_asked_for(self, monkeypatch):
        _usd(monkeypatch)
        stage = MagicMock()
        lops._traverse(stage, None, False)
        stage.Traverse.assert_called_once_with()
        stage.Traverse.reset_mock()
        lops._traverse(stage, None, True)
        stage.Traverse.assert_called_once_with("PROXIES")

    def test_a_subtree_walk_takes_the_predicate_too(self, monkeypatch):
        _usd(monkeypatch)
        root = MagicMock()
        assert lops._traverse(MagicMock(), root, False) == [root]
        assert lops._traverse(MagicMock(), root, True) == [root, "PROXIES"]

    def test_list_usd_prims_passes_the_flag_through(self, monkeypatch):
        _usd(monkeypatch)
        stage = MagicMock()
        stage.GetPrimAtPath.return_value = _prim("/")
        seen = []
        monkeypatch.setattr(lops, "_get_lop_stage", lambda node_path: stage)
        monkeypatch.setattr(
            lops, "_traverse", lambda s, root, proxies: seen.append((root, proxies)) or []
        )
        reply = lops._list_usd_prims(node_path="/stage/out", traverse_instance_proxies=True)
        assert seen == [(None, True)]
        assert reply["instance_proxies_included"] is True

    def test_find_usd_prims_reaches_a_mesh_under_a_prototype(self, monkeypatch):
        _usd(monkeypatch)
        mesh = _instanceable("/pi/Prototypes/obj_0/mesh_0", False, "Mesh")
        stage = MagicMock()
        monkeypatch.setattr(lops, "_get_lop_stage", lambda node_path: stage)
        monkeypatch.setattr(lops, "_traverse", lambda s, root, proxies: [mesh] if proxies else [])
        assert lops._find_usd_prims(node_path="/stage/out", pattern="mesh_0")["count"] == 0
        reply = lops._find_usd_prims(
            node_path="/stage/out", pattern="mesh_0", traverse_instance_proxies=True
        )
        assert reply["count"] == 1
        assert reply["prims"][0]["path"] == "/pi/Prototypes/obj_0/mesh_0"


class TestAnEmptyChildrenListIsLegible:
    def test_an_instanceable_prim_says_what_it_hides(self, monkeypatch):
        _usd(monkeypatch)
        monkeypatch.setattr(lops, "_hidden_descendants", lambda prim: 5)
        fake = _instanceable("/pi/Prototypes/obj_0", True)
        fake.GetAttributes.return_value = []
        fake.GetChildren.return_value = ()
        info = lops._prim_to_dict(fake, include_attrs=True)
        assert info["is_instanceable"] is True
        assert info["hidden_descendants"] == 5

    def test_a_list_entry_does_not_walk_the_prototype(self, monkeypatch):
        _usd(monkeypatch)
        walked = []
        monkeypatch.setattr(lops, "_hidden_descendants", lambda prim: walked.append(prim) or 5)
        info = lops._prim_to_dict(_instanceable("/pi/Prototypes/obj_0", True))
        assert info["is_instanceable"] is True
        assert walked == []

    def test_a_plain_prim_carries_no_instancing_keys(self, monkeypatch):
        _usd(monkeypatch)
        info = lops._prim_to_dict(_instanceable("/geo", False))
        assert "is_instanceable" not in info
        assert "hidden_descendants" not in info

    def test_get_usd_prim_lists_prototype_children_with_the_flag(self, monkeypatch):
        usd = _usd(monkeypatch)
        proto = _instanceable("/pi/Prototypes/obj_0", True)
        proto.GetAttributes.return_value = []
        proto.GetChildren.return_value = ()
        proto.GetFilteredChildren.return_value = [_prim("/pi/Prototypes/obj_0/mesh_0")]
        monkeypatch.setattr(lops, "_hidden_descendants", lambda prim: 1)
        stage = MagicMock()
        stage.GetPrimAtPath.return_value = proto
        monkeypatch.setattr(lops, "_get_lop_stage", lambda node_path: stage)

        plain = lops._get_usd_prim(node_path="/stage/out", prim_path="/pi/Prototypes/obj_0")
        assert plain["prim"]["children"] == []
        assert plain["prim"]["hidden_descendants"] == 1

        walked = lops._get_usd_prim(
            node_path="/stage/out",
            prim_path="/pi/Prototypes/obj_0",
            traverse_instance_proxies=True,
        )
        assert walked["prim"]["children"] == ["/pi/Prototypes/obj_0/mesh_0"]
        assert walked["prim"]["instance_proxies_included"] is True
        usd.TraverseInstanceProxies.assert_called_with(usd.PrimDefaultPredicate)


class TestPrimStats:
    def _run(self, monkeypatch, prims, **kwargs):
        _usd(monkeypatch)
        stage = MagicMock()
        stage.GetPrimAtPath.return_value = _prim("/")
        monkeypatch.setattr(lops, "_get_lop_stage", lambda node_path: stage)
        monkeypatch.setattr(lops, "_traverse", lambda s, root, proxies: prims)
        return lops._get_usd_prim_stats(node_path="/stage/out", **kwargs)

    def test_the_note_points_at_the_flag(self, monkeypatch):
        prims = [
            _instanceable("/city", False),
            _instanceable("/city/tree_0", True),
        ]
        reply = self._run(monkeypatch, prims)
        assert reply["instanceable_prims"] == 1
        assert reply["instance_proxies_included"] is False
        assert "traverse_instance_proxies=true" in reply["note"]

    def test_with_the_flag_there_is_nothing_to_warn_about(self, monkeypatch):
        prims = [_instanceable("/city/tree_0", True), _instanceable("/m", False, "Mesh")]
        reply = self._run(monkeypatch, prims, traverse_instance_proxies=True)
        assert reply["instance_proxies_included"] is True
        assert reply["type_counts"] == {"Xform": 1, "Mesh": 1}
        assert "note" not in reply
