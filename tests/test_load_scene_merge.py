"""Tests for load_scene: merging named nodes and load warnings as data.

load_scene(merge=True) used to merge a whole file and report nothing; a
session that wanted one asset from a lookdev hip reached for execute_python.
Houdini's node_pattern must be absolute and a container named alone comes
without its children, so each path is sent as `<path> <path>/*`, and the
reply is attributed per requested path: arrived, renumbered next to an
existing node, overwritten in place, or not found.

hou is faked here; the behaviour was measured on Houdini 22.0.429.
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


class _LoadWarning(Exception):
    pass


class _Node:
    def __init__(self, tree, path):
        self._tree, self._path = tree, path

    def path(self):
        return self._path

    def children(self):
        prefix = self._path.rstrip("/") + "/"
        return [
            _Node(self._tree, p)
            for p in sorted(self._tree)
            if p.startswith(prefix) and "/" not in p[len(prefix) :]
        ]

    def allSubChildren(self):
        prefix = self._path + "/"
        return [_Node(self._tree, p) for p in sorted(self._tree) if p.startswith(prefix)]


class _Scene:
    """A node tree plus a merge that behaves the way 22.0.429 was measured to."""

    def __init__(self, scene_nodes, file_nodes, warning=None):
        self.tree = {"/", "/obj", "/out"} | set(scene_nodes)
        self.file = set(file_nodes)
        self.warning = warning
        self.merge_calls = []

    def node(self, path):
        return _Node(self.tree, path) if path in self.tree else None

    def _selected(self, pattern):
        if pattern == "*":
            return set(self.file)
        wanted = pattern.split()
        return {
            p
            for p in self.file
            if p in wanted or any(p.startswith(w[:-1]) for w in wanted if w.endswith("/*"))
        }

    def collisions(self, file_path, node_pattern="*"):
        return [_Node(self.tree, p) for p in sorted(self._selected(node_pattern) & self.tree)]

    def merge(
        self, file_path, node_pattern="*", overwrite_on_conflict=False, ignore_load_warnings=False
    ):
        self.merge_calls.append(node_pattern)
        selected = self._selected(node_pattern)
        tops = sorted(p for p in selected if p.rpartition("/")[0] not in selected)
        for top in tops:
            parent, _, name = top.rpartition("/")
            if parent not in self.tree:
                continue
            target = top
            if top in self.tree and not overwrite_on_conflict:
                stem, n = name.rstrip("0123456789"), 2
                while f"{parent}/{stem}{n}" in self.tree:
                    n += 1
                target = f"{parent}/{stem}{n}"
            for p in selected:
                if p == top or p.startswith(top + "/"):
                    self.tree.add(target + p[len(top) :])
        if self.warning:
            raise _LoadWarning(self.warning)


@pytest.fixture
def hip(tmp_path):
    path = tmp_path / "other.hip"
    path.write_bytes(b"hip")
    return str(path)


def _install(monkeypatch, fake):
    monkeypatch.setattr(scene, "require_inside_project_root", lambda path, what="": path)
    monkeypatch.setattr(scene.hou, "node", fake.node)
    monkeypatch.setattr(scene.hou, "LoadWarning", _LoadWarning)
    monkeypatch.setattr(scene.hou.text, "expandString", lambda s: s)
    hip_file = MagicMock()
    hip_file.merge.side_effect = fake.merge
    hip_file.collisionNodesIfMerged.side_effect = fake.collisions
    hip_file.path.return_value = "/proj/shot.hip"
    hip_file.hasUnsavedChanges.return_value = True
    monkeypatch.setattr(scene.hou, "hipFile", hip_file)
    return hip_file


FILE = ["/obj/null1", "/obj/building", "/obj/building/walls", "/obj/geo1", "/obj/geo1/sphere1"]


class TestMergeNamedNodes:
    def test_a_container_arrives_with_its_contents(self, monkeypatch, hip):
        fake = _Scene([], FILE)
        _install(monkeypatch, fake)
        result = scene.load_scene(hip, merge=True, node_paths=["/obj/building"])
        assert fake.merge_calls == ["/obj/building /obj/building/*"]
        assert result["merged_nodes"] == ["/obj/building"]
        assert result["merged_descendant_count"] == 1
        assert result["success"] is True

    def test_a_node_merged_into_an_existing_container_is_reported(self, monkeypatch, hip):
        fake = _Scene(["/obj/geo1"], FILE)
        _install(monkeypatch, fake)
        result = scene.load_scene(hip, merge=True, node_paths=["/obj/geo1/sphere1"])
        assert result["merged_nodes"] == ["/obj/geo1/sphere1"]
        assert result["not_found_in_file"] == []

    def test_a_collision_is_renumbered_and_only_the_renumbered_node_is_claimed(
        self, monkeypatch, hip
    ):
        fake = _Scene(["/obj/null1"], FILE)
        _install(monkeypatch, fake)
        result = scene.load_scene(hip, merge=True, node_paths=["/obj/null1", "/obj/building"])
        assert result["merged_nodes"] == ["/obj/building", "/obj/null2"]
        assert result["conflicts"] == [
            {
                "requested": "/obj/null1",
                "outcome": "merged under a new name",
                "merged_as": "/obj/null2",
            }
        ]

    def test_an_overwrite_is_listed_as_merged(self, monkeypatch, hip):
        fake = _Scene(["/obj/null1"], FILE)
        _install(monkeypatch, fake)
        result = scene.load_scene(
            hip, merge=True, node_paths=["/obj/null1"], overwrite_on_conflict=True
        )
        assert result["merged_nodes"] == ["/obj/null1"]
        assert result["conflicts"] == [
            {"requested": "/obj/null1", "outcome": "overwritten in place"}
        ]

    def test_an_existing_node_absent_from_the_file_is_not_found_not_a_conflict(
        self, monkeypatch, hip
    ):
        fake = _Scene(["/obj/local_only"], FILE)
        _install(monkeypatch, fake)
        result = scene.load_scene(hip, merge=True, node_paths=["/obj/local_only"])
        assert result["conflicts"] == []
        assert result["not_found_in_file"] == ["/obj/local_only"]
        assert result["success"] is False

    def test_an_empty_list_merges_nothing(self, monkeypatch, hip):
        fake = _Scene([], FILE)
        _install(monkeypatch, fake)
        result = scene.load_scene(hip, merge=True, node_paths=[])
        assert fake.merge_calls == []
        assert result["merged_nodes"] == []

    def test_patterns_and_relative_paths_are_refused(self, monkeypatch, hip):
        _install(monkeypatch, _Scene([], FILE))
        with pytest.raises(ValueError, match="not patterns"):
            scene.load_scene(hip, merge=True, node_paths=["/obj/geo*"])
        with pytest.raises(ValueError, match="must be absolute"):
            scene.load_scene(hip, merge=True, node_paths=["building"])


class TestMergeEverything:
    def test_everything_reports_arrivals_and_renumbered_collisions(self, monkeypatch, hip):
        fake = _Scene(["/obj/null1"], FILE)
        _install(monkeypatch, fake)
        result = scene.load_scene(hip, merge=True)
        assert fake.merge_calls == ["*"]
        assert result["merged_nodes"] == ["/obj/building", "/obj/geo1", "/obj/null2"]
        assert result["conflicts"][0]["merged_as"] == "/obj/null2"
        assert result["not_found_in_file"] == []

    def test_a_warning_with_nothing_merged_is_not_success(self, monkeypatch, hip):
        fake = _Scene([], [], warning="Missing asset\nfoo::1.0")
        _install(monkeypatch, fake)
        result = scene.load_scene(hip, merge=True)
        assert result["warnings"] == ["Missing asset", "foo::1.0"]
        assert result["success"] is False


class TestLoad:
    def test_load_warnings_come_back_as_data(self, monkeypatch, hip):
        hip_file = _install(monkeypatch, _Scene([], []))
        hip_file.load.side_effect = _LoadWarning("Missing asset foo::1.0")
        result = scene.load_scene(hip)
        assert result["success"] is True
        assert result["warnings"] == ["Missing asset foo::1.0"]

    def test_a_houdini_variable_path_is_expanded_before_the_existence_check(self, monkeypatch, hip):
        hip_file = _install(monkeypatch, _Scene([], []))
        monkeypatch.setattr(
            scene.hou.text, "expandString", lambda s: s.replace("$HIP", os.path.dirname(hip))
        )
        scene.load_scene("$HIP/other.hip")
        assert hip_file.load.call_args.args[0] == "$HIP/other.hip"

    def test_merge_options_without_merge_are_refused(self, monkeypatch, hip):
        _install(monkeypatch, _Scene([], []))
        with pytest.raises(ValueError, match="only with merge=True"):
            scene.load_scene(hip, node_paths=["/obj/a"])
