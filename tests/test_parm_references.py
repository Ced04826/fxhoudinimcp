"""Tests for get_parm_references and get_parm_template_tree.

Two questions a session kept answering with execute_python: "who reads this
parameter, and what does it read" (parmsReferencingThis and getReferencedParm
through HOM), and "what is in this folder of the interface, in order, with its
Hide When rules" -- get_parameter_schema flattens folders, conditionals, menu
items and multiparm blocks away, and get_hda_info shows only the top folders.

getReferencedParm() resolves only a pure `ch("../src/tx")`; for
`ch("../src/scale") * 2` it answers with the parm itself (measured on 22.0.429),
so the richer expressions are parsed for channel references instead.

hou is mocked here; the handlers are exercised directly.
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
import fxhoudinimcp_server.handlers.parameter_handlers as parameters  # noqa: E402


def _node(path, type_name="null"):
    node = MagicMock()
    node.path.return_value = path
    node.name.return_value = path.rsplit("/", 1)[-1]
    node.type.return_value.name.return_value = type_name
    return node


def _template(name, kind="Float", components=1, scheme="XYZW", label=None):
    template = MagicMock()
    template.name.return_value = name
    template.label.return_value = label or name.title()
    template.type.return_value.name.return_value = kind
    template.numComponents.return_value = components
    template.namingScheme.return_value.name.return_value = scheme
    template.parmTemplates.return_value = ()
    return template


###### get_parm_references


class TestParmReferences:
    def _parm(self, path, expression=None, referenced=None, referencing=()):
        parm = MagicMock()
        parm.path.return_value = path
        parm.name.return_value = path.rsplit("/", 1)[-1]
        parm.expression.return_value = expression
        if expression is None:
            parm.expression.side_effect = RuntimeError("no expression")
        parm.getReferencedParm.return_value = referenced if referenced is not None else parm
        parm.parmsReferencingThis.return_value = list(referencing)
        return parm

    def test_a_pure_channel_reference_resolves_through_hom(self):
        target = self._parm("/obj/src/tx")
        parm = self._parm("/obj/dst/tx", 'ch("../src/tx")', referenced=target)
        entry = parameters._outgoing_reference(parm)
        assert entry["references"] == ["/obj/src/tx"]
        assert entry["pure_reference"] is True

    def test_a_richer_expression_is_parsed_and_resolved_relative_to_the_node(self):
        parm = self._parm("/obj/dst/scale", 'ch("../src/scale") * 2 + chs("../src/name")')
        resolved = MagicMock()
        resolved.path.return_value = "/obj/src/scale"
        parm.node.return_value.parm.side_effect = lambda token: (
            resolved if token == "../src/scale" else None
        )
        entry = parameters._outgoing_reference(parm)
        assert entry["references"] == ["/obj/src/scale", "../src/name"]
        assert entry["pure_reference"] is False

    def test_a_plain_value_has_no_outgoing_entry(self):
        assert parameters._outgoing_reference(self._parm("/obj/dst/ty")) is None

    def test_both_directions_are_listed(self, monkeypatch):
        by = self._parm("/obj/dst/tx")
        src_tx = self._parm("/obj/src/tx", referencing=[by])
        src_ty = self._parm("/obj/src/ty")
        node = _node("/obj/src")
        node.parms.return_value = [src_tx, src_ty]
        node.dependents.return_value = [_node("/obj/dst"), node]
        node.references.return_value = []
        monkeypatch.setattr(parameters, "_resolve_node", lambda path: node)
        result = parameters._get_parm_references("/obj/src")
        assert result["incoming"] == [{"parm": "tx", "referenced_by": ["/obj/dst/tx"]}]
        assert result["outgoing"] == []
        assert result["node_dependents"] == ["/obj/dst"]

    def test_one_parameter_can_be_asked_for(self, monkeypatch):
        by = self._parm("/obj/dst/tx")
        src_tx = self._parm("/obj/src/tx", referencing=[by])
        node = _node("/obj/src")
        node.parms.return_value = [src_tx, self._parm("/obj/src/ty")]
        monkeypatch.setattr(parameters, "_resolve_node", lambda path: node)
        monkeypatch.setattr(parameters, "_resolve_parm", lambda path, name: src_tx)
        result = parameters._get_parm_references("/obj/src", parm_name="tx", direction="incoming")
        assert result["incoming"] == [{"parm": "tx", "referenced_by": ["/obj/dst/tx"]}]
        assert result["parm_name"] == "tx"

    def test_direction_is_checked(self, monkeypatch):
        monkeypatch.setattr(parameters, "_resolve_node", lambda path: _node("/obj/src"))
        with pytest.raises(ValueError, match="direction"):
            parameters._get_parm_references("/obj/src", direction="sideways")


###### get_parm_template_tree


class TestParmTemplateTree:
    def _float(self, name, hidden=False, cond=None):
        template = _template(name, "Float")
        template.isHidden.return_value = hidden
        template.conditionals.return_value = cond or {}
        template.help.return_value = ""
        template.joinsWithNext.return_value = False
        template.tags.return_value = {}
        template.defaultValue.return_value = (0.5,)
        template.defaultExpression.return_value = ("",)
        template.minValue.return_value = 0.0
        template.maxValue.return_value = 1.0
        template.minIsStrict.return_value = False
        template.maxIsStrict.return_value = True
        template.menuItems.return_value = ()
        template.scriptCallback.return_value = ""
        template.stringType.side_effect = AttributeError("not a string")
        template.dataParmType.side_effect = AttributeError("not data")
        template.look.return_value.name.return_value = "Regular"
        return template

    def _folder(self, name, label, children, folder_type="Tabs"):
        folder = _template(name, "Folder", label=label)
        folder.isHidden.return_value = False
        folder.conditionals.return_value = {}
        folder.help.return_value = ""
        folder.joinsWithNext.return_value = False
        folder.tags.return_value = {}
        folder.folderType.return_value.name.return_value = folder_type
        folder.endsTabGroup.return_value = False
        folder.parmTemplates.return_value = children
        return folder

    def test_folders_nest_and_parameters_carry_their_rules(self):
        cond_key = MagicMock()
        cond_key.name.return_value = "HideWhen"
        bevel = self._float("bevel", cond={cond_key: "{ stud_count == 1 }"})
        folder = self._folder("stdswitcher3_2", "Controls", [bevel])
        entry = parameters._template_tree_entry(folder)
        assert entry["type"] == "Folder"
        assert entry["folder_type"] == "Tabs"
        child = entry["children"][0]
        assert child["conditionals"] == {"HideWhen": "{ stud_count == 1 }"}
        assert child["default"] == [0.5]
        assert child["max_strict"] is True
        assert "hidden" not in child

    def test_menu_items_come_with_their_labels(self):
        menu = self._float("splittype")
        menu.type.return_value.name.return_value = "Menu"
        menu.menuItems.return_value = ("edge", "point")
        menu.menuLabels.return_value = ("Edge", "Point")
        entry = parameters._template_tree_entry(menu)
        assert entry["menu"] == [
            {"value": "edge", "label": "Edge"},
            {"value": "point", "label": "Point"},
        ]

    def test_the_tree_is_cut_at_max_entries_and_says_so(self, monkeypatch):
        parms = [self._float(f"p{i}") for i in range(6)]
        folder = self._folder("f", "Folder", parms)
        node = _node("/obj/asset", "asset")
        node.parmTemplateGroup.return_value.entries.return_value = [folder]
        monkeypatch.setattr(parameters, "_resolve_node", lambda path: node)
        result = parameters._get_parm_template_tree("/obj/asset", max_entries=3)
        assert result["entry_count"] == 7
        assert result["truncated"] is True
        assert len(result["entries"][0]["children"]) == 2
        assert "folder=" in result["note"]

    def test_a_folder_filter_that_matches_nothing_lists_the_folders(self, monkeypatch):
        node = _node("/obj/asset", "asset")
        group = node.parmTemplateGroup.return_value
        group.findFolder.return_value = None
        group.entries.return_value = [self._folder("f", "Transform", [])]
        monkeypatch.setattr(parameters, "_resolve_node", lambda path: node)
        with pytest.raises(ValueError, match="Transform"):
            parameters._get_parm_template_tree("/obj/asset", folder="Controls")

    def test_a_type_is_read_through_its_preferred_version(self, monkeypatch):
        hou = parameters.hou
        category = MagicMock()
        category.name.return_value = "Sop"
        monkeypatch.setattr(hou, "nodeTypeCategories", lambda: {"Sop": category})
        preferred = MagicMock()
        preferred.name.return_value = "polyextrude::2.0"
        preferred.parmTemplateGroup.return_value.entries.return_value = [self._float("dist")]
        monkeypatch.setattr(hou, "preferredNodeType", lambda name: preferred)
        result = parameters._get_parm_template_tree(type_name="polyextrude", context="Sop")
        assert result["type"] == "polyextrude::2.0"
        assert [e["name"] for e in result["entries"]] == ["dist"]

    def test_node_path_or_type_name_is_required(self):
        with pytest.raises(ValueError, match="node_path or type_name"):
            parameters._get_parm_template_tree()
