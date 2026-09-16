"""Tests for merge_hip.

`load_scene(merge=True)` pulls a whole file in and says nothing about what
arrived. Merging a few named nodes meant `hou.hipFile.merge` from
execute_python, and its pattern has two surprises measured live on Houdini
22.0.429: a relative name (`null1`) matches nothing, and a container named
alone (`/obj/geo1`) arrives without its children while `/obj/geo1/*` on its
own raises "Missing the parent of a node". merge_hip sends every path as
`<path> <path>/*`, snapshots the root contexts before and after, and reports
what was merged, what collided and what the file did not have.

hou is mocked here.
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
import fxhoudinimcp_server.handlers.scene_handlers as scene  # noqa: E402


class TestMergeHip:
    def _scene(self, monkeypatch, before, after, existing=()):
        hou = scene.hou
        snapshots = iter([before, after])
        monkeypatch.setattr(scene, "_top_level_nodes", lambda: next(snapshots))
        monkeypatch.setattr(scene, "require_inside_project_root", lambda p, w: p)
        monkeypatch.setattr(scene.os.path, "isfile", lambda p: True)
        monkeypatch.setattr(hou, "node", lambda path: MagicMock() if path in existing else None)
        hou.hipFile.merge.reset_mock()
        hou.hipFile.merge.side_effect = None
        hou.hipFile.path.return_value = "/proj/cur.hip"
        hou.hipFile.hasUnsavedChanges.return_value = True
        return hou

    def test_each_path_is_sent_with_its_children_and_new_nodes_are_listed(self, monkeypatch):
        hou = self._scene(
            monkeypatch,
            {"/obj": {"/obj/a"}},
            {"/obj": {"/obj/a", "/obj/building"}},
        )
        result = scene.merge_hip("/proj/other.hip", ["/obj/building"])
        hou.hipFile.merge.assert_called_once_with(
            "/proj/other.hip",
            node_pattern="/obj/building /obj/building/*",
            overwrite_on_conflict=False,
            ignore_load_warnings=False,
        )
        assert result["merged_nodes"] == ["/obj/building"]
        assert result["conflicts"] == []
        assert result["not_found_in_file"] == []

    def test_no_paths_means_everything(self, monkeypatch):
        hou = self._scene(monkeypatch, {"/obj": set()}, {"/obj": {"/obj/a", "/obj/b"}})
        result = scene.merge_hip("/proj/other.hip")
        assert hou.hipFile.merge.call_args.kwargs["node_pattern"] == "*"
        assert result["merged_nodes"] == ["/obj/a", "/obj/b"]

    def test_a_relative_path_is_refused(self, monkeypatch):
        self._scene(monkeypatch, {}, {})
        with pytest.raises(ValueError, match="absolute"):
            scene.merge_hip("/proj/other.hip", ["building"])

    def test_a_missing_file_is_refused(self, monkeypatch):
        self._scene(monkeypatch, {}, {})
        monkeypatch.setattr(scene.os.path, "isfile", lambda p: False)
        with pytest.raises(FileNotFoundError):
            scene.merge_hip("/proj/other.hip")

    def test_an_existing_node_is_reported_as_renamed(self, monkeypatch):
        self._scene(
            monkeypatch,
            {"/obj": {"/obj/null1"}},
            {"/obj": {"/obj/null1", "/obj/null2"}},
            existing=("/obj/null1",),
        )
        result = scene.merge_hip("/proj/other.hip", ["/obj/null1"])
        assert result["conflicts"] == [
            {
                "requested": "/obj/null1",
                "existed": True,
                "outcome": "merged under a new name",
                "merged_as": ["/obj/null2"],
            }
        ]

    def test_overwrite_is_reported_in_place(self, monkeypatch):
        hou = self._scene(
            monkeypatch,
            {"/obj": {"/obj/null1"}},
            {"/obj": {"/obj/null1"}},
            existing=("/obj/null1",),
        )
        result = scene.merge_hip("/proj/other.hip", ["/obj/null1"], overwrite_on_conflict=True)
        assert hou.hipFile.merge.call_args.kwargs["overwrite_on_conflict"] is True
        assert result["conflicts"][0]["outcome"] == "overwritten in place"

    def test_load_warnings_come_back_as_data(self, monkeypatch):
        hou = self._scene(monkeypatch, {"/obj": set()}, {"/obj": set()})

        class LoadWarning(Exception):
            pass

        hou.LoadWarning = LoadWarning
        hou.hipFile.merge.side_effect = LoadWarning("Warnings\n Missing the parent of a node")
        result = scene.merge_hip("/proj/other.hip", ["/obj/gone"])
        assert result["warnings"] == ["Warnings", "Missing the parent of a node"]
        assert result["not_found_in_file"] == ["/obj/gone"]
        assert "note" in result
