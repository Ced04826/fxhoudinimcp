"""Tests for set_hda_interface and edit_hda_interface — an asset's Type
Properties interface, authored and edited through the definition.

create_spare_parameter adds parameters to a node INSTANCE and never reaches
the type; nothing wrote the definition's own interface, and once written
nothing could edit it: insert at a position, remove, hide, rename, a default
expression, a button with a callback, a multiparm block. Two things measured
live on Houdini 22.0.429 shape the edit verb: Houdini refuses a whole group
over a component-name collision (`fh_range` under Base1 makes `fh_range2`, and a
template called `fh_range2` next to it fails with a bare OperationFailed), and
built-in parameters of the node type cannot be removed from an asset's
interface — Houdini puts them back at the top level without a word.

The in-Houdini handlers import `hou`; it is mocked here.
"""

from __future__ import annotations

# Built-in
import os
import sys
from unittest.mock import MagicMock

# Third-party
import pytest

# Mock Houdini modules before importing the in-Houdini server package
sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

# Internal
import fxhoudinimcp_server.handlers.hda_handlers as hda  # noqa: E402


class _Parm:
    def __init__(self, name, spare):
        self._name = name
        self._spare = spare

    def name(self):
        return self._name

    def isSpare(self):
        return self._spare


class _PlainNode:
    """A node type that simply does not carry createDigitalAsset, like merge."""

    def path(self):
        return "/obj/unit"

    def type(self):
        node_type = MagicMock()
        node_type.name.return_value = "merge"
        return node_type


class _Node:
    """A node that is NOT a subnet — the case the old guard refused."""

    def __init__(self, type_name="geo", definition=None, parms=()):
        self._type_name = type_name
        self._definition = definition
        self._parms = list(parms)
        self.editable = False
        self.propagated = None
        self.changed_to = None
        self.created = None

    def path(self):
        return "/obj/unit"

    def type(self):
        node_type = MagicMock()
        node_type.name.return_value = self._type_name
        node_type.definition.return_value = self._definition
        return node_type

    def isSubNetwork(self):
        return False

    def createDigitalAsset(self, **kwargs):
        self.created = kwargs
        return _Node(type_name=kwargs.get("name", "asset"))

    def allowEditingOfContents(self, propagate=False):
        self.propagated = propagate
        self.editable = True

    def isEditable(self):
        return self.editable

    def removeSpareParms(self):
        self._parms = [p for p in self._parms if not p.isSpare()]

    def parms(self):
        return tuple(self._parms)

    def changeNodeType(self, new_type_name, **kwargs):
        self.changed_to = new_type_name
        return _Node(type_name=new_type_name)


def _node(path, type_name="xform", **attrs):
    node = MagicMock()
    node.path.return_value = path
    node.name.return_value = path.rsplit("/", 1)[-1]
    node.type.return_value.name.return_value = type_name
    for key, value in attrs.items():
        getattr(node, key).return_value = value
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


class _Template:
    """Stands in for hou.ParmTemplate — records what was asked for."""

    def __init__(self, name, label, kind, **kwargs):
        self._name = name
        self._label = label
        self._kind = kind
        self.kwargs = kwargs
        self.conditionals_set = {}
        self.help = None

    def name(self):
        return self._name

    def label(self):
        return self._label

    def type(self):
        return MagicMock(**{"name.return_value": self._kind})

    def setConditional(self, cond_type, expression):
        self.conditionals_set[str(cond_type)] = expression

    def conditionals(self):
        return dict(self.conditionals_set)

    def setHelp(self, text):
        self.help = text

    def parmTemplates(self):
        return tuple(self.kwargs.get("parm_templates", ()))


class _Group:
    def __init__(self):
        self.templates = []
        self.cleared = False

    def clear(self):
        self.cleared = True
        self.templates = []

    def append(self, template):
        self.templates.append(template)

    def parmTemplates(self):
        return tuple(self.templates)


@pytest.fixture
def interface_env(monkeypatch):
    """Give the mocked hou just enough ParmTemplate machinery."""
    hou = hda.hou
    monkeypatch.setattr(
        hou,
        "IntParmTemplate",
        lambda n, label, c, **kw: _Template(n, label, "Int", components=c, **kw),
        raising=False,
    )
    monkeypatch.setattr(
        hou,
        "FloatParmTemplate",
        lambda n, label, c, **kw: _Template(n, label, "Float", components=c, **kw),
        raising=False,
    )
    monkeypatch.setattr(
        hou,
        "StringParmTemplate",
        lambda n, label, c, **kw: _Template(n, label, "String", components=c, **kw),
        raising=False,
    )
    monkeypatch.setattr(
        hou,
        "ToggleParmTemplate",
        lambda n, label, **kw: _Template(n, label, "Toggle", **kw),
        raising=False,
    )
    monkeypatch.setattr(
        hou,
        "MenuParmTemplate",
        lambda n, label, **kw: _Template(n, label, "Menu", **kw),
        raising=False,
    )
    monkeypatch.setattr(
        hou,
        "FolderParmTemplate",
        lambda n, label, **kw: _Template(n, label, "Folder", **kw),
        raising=False,
    )
    monkeypatch.setattr(hou, "FolderParmTemplate", hou.FolderParmTemplate, raising=False)
    monkeypatch.setattr(hda.hou, "folderType", MagicMock(), raising=False)
    monkeypatch.setattr(hda.hou, "menuType", MagicMock(), raising=False)
    monkeypatch.setattr(
        hda.hou,
        "parmCondType",
        MagicMock(HideWhen="parmCondType.HideWhen", DisableWhen="parmCondType.DisableWhen"),
        raising=False,
    )
    return hou


def _hda_node(monkeypatch, group):
    definition = MagicMock()
    definition.parmTemplateGroup.return_value = group
    definition.nodeTypeName.return_value = "brick"
    definition.libraryFilePath.return_value = "/tmp/brick.hda"
    node = _Node(type_name="brick", definition=definition)
    monkeypatch.setattr(hda, "_get_node", lambda path: node)
    monkeypatch.setattr(hda.hou, "node", lambda path: node, raising=False)
    return node, definition


class TestBuildParmTemplate:
    def test_strict_range_reaches_houdini(self, interface_env):
        t = hda._build_parm_template(
            {
                "name": "stud_count",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 8,
                "min_strict": True,
                "max_strict": True,
            },
            set(),
        )
        assert t.kwargs["min"] == 1
        assert t.kwargs["max"] == 8
        assert t.kwargs["min_is_strict"] is True
        assert t.kwargs["max_is_strict"] is True
        assert t.kwargs["default_value"] == (4,)

    def test_label_defaults_to_a_readable_name(self, interface_env):
        t = hda._build_parm_template({"name": "stud_count", "type": "int"}, set())
        assert t.label() == "Stud Count"

    def test_hide_when_is_attached(self, interface_env):
        t = hda._build_parm_template(
            {"name": "bevel", "type": "float", "hide_when": "{ stud_count == 1 }"}, set()
        )
        assert t.conditionals_set["parmCondType.HideWhen"] == "{ stud_count == 1 }"

    def test_menu_pairs_and_string_default(self, interface_env):
        t = hda._build_parm_template(
            {
                "name": "material",
                "type": "menu",
                "default": "metal",
                "menu_items": [["plastic", "Plastic"], ["metal", "Metal"]],
            },
            set(),
        )
        assert t.kwargs["menu_items"] == ("plastic", "metal")
        assert t.kwargs["menu_labels"] == ("Plastic", "Metal")
        assert t.kwargs["default_value"] == 1

    def test_plain_string_menu_items_are_allowed(self, interface_env):
        t = hda._build_parm_template(
            {"name": "material", "type": "menu", "menu_items": ["a", "b"]}, set()
        )
        assert t.kwargs["menu_items"] == ("a", "b")
        assert t.kwargs["menu_labels"] == ("a", "b")

    def test_menu_default_outside_items_is_rejected(self, interface_env):
        with pytest.raises(ValueError, match="not one of its items"):
            hda._build_parm_template(
                {"name": "material", "type": "menu", "default": "gold", "menu_items": ["plastic"]},
                set(),
            )

    def test_folder_nests_children(self, interface_env):
        t = hda._build_parm_template(
            {
                "name": "controls",
                "type": "folder",
                "children": [{"name": "a", "type": "int"}, {"name": "b", "type": "float"}],
            },
            set(),
        )
        assert [c.name() for c in t.parmTemplates()] == ["a", "b"]

    def test_duplicate_names_are_rejected(self, interface_env):
        seen = set()
        hda._build_parm_template({"name": "size", "type": "int"}, seen)
        with pytest.raises(ValueError, match="Duplicate parameter name"):
            hda._build_parm_template({"name": "size", "type": "float"}, seen)

    def test_unknown_type_lists_the_supported_ones(self, interface_env):
        with pytest.raises(
            ValueError, match="int, float, vector, color, string, file, oppath, toggle, menu"
        ):
            hda._build_parm_template({"name": "x", "type": "ramp"}, set())

    def test_unknown_folder_type_is_rejected(self, interface_env):
        with pytest.raises(ValueError, match="Unknown folder_type"):
            hda._build_parm_template(
                {"name": "f", "type": "folder", "folder_type": "accordion"}, set()
            )

    def test_missing_name_is_rejected(self, interface_env):
        with pytest.raises(ValueError, match="missing 'name'"):
            hda._build_parm_template({"type": "int"}, set())

    def test_menu_without_items_is_rejected(self, interface_env):
        with pytest.raises(ValueError, match="needs menu_items"):
            hda._build_parm_template({"name": "m", "type": "menu"}, set())


class TestSetHdaInterface:
    def test_appends_by_default_and_writes_the_definition(self, interface_env, monkeypatch):
        group = _Group()
        group.append(_Template("stdswitcher", "Transform", "Folder"))
        node, definition = _hda_node(monkeypatch, group)

        result = hda.set_hda_interface(
            "/obj/unit",
            [
                {
                    "name": "controls",
                    "type": "folder",
                    "children": [
                        {
                            "name": "stud_count",
                            "type": "int",
                            "min": 1,
                            "max": 8,
                            "min_strict": True,
                            "max_strict": True,
                        }
                    ],
                }
            ],
        )

        assert group.cleared is False
        assert [t.name() for t in group.parmTemplates()] == ["stdswitcher", "controls"]
        definition.setParmTemplateGroup.assert_called_once_with(group)
        assert result["requested"] == ["controls"]
        assert result["applied"][0]["children"][0]["name"] == "stud_count"
        assert result["renamed_by_houdini"] == []

    def test_replace_clears_first(self, interface_env, monkeypatch):
        group = _Group()
        group.append(_Template("stdswitcher", "Transform", "Folder"))
        _hda_node(monkeypatch, group)

        hda.set_hda_interface("/obj/unit", [{"name": "only", "type": "int"}], replace=True)
        assert group.cleared is True
        assert [t.name() for t in group.parmTemplates()] == ["only"]

    def test_plain_node_is_rejected_with_the_alternative(self, interface_env, monkeypatch):
        node = _Node(type_name="geo", definition=None)
        monkeypatch.setattr(hda, "_get_node", lambda path: node)
        with pytest.raises(ValueError, match="create_spare_parameter"):
            hda.set_hda_interface("/obj/unit", [{"name": "x", "type": "int"}])

    def test_empty_spec_is_rejected(self, interface_env, monkeypatch):
        _hda_node(monkeypatch, _Group())
        with pytest.raises(ValueError, match="non-empty list"):
            hda.set_hda_interface("/obj/unit", [])

    def test_nothing_is_written_when_a_spec_is_bad(self, interface_env, monkeypatch):
        group = _Group()
        _, definition = _hda_node(monkeypatch, group)
        with pytest.raises(ValueError):
            hda.set_hda_interface(
                "/obj/unit", [{"name": "good", "type": "int"}, {"name": "bad", "type": "ramp"}]
            )
        definition.setParmTemplateGroup.assert_not_called()


class TestHoudiniRenamesTabFolders:
    """Houdini renames a tab folder appended next to an existing tab group.

    Measured live: asking for "controls" on a Geo-derived asset stored it as
    "stdswitcher4_3" — label and child parameters intact. Matching the result
    by the requested name reported "nothing applied" while the interface was
    sitting right there, so the verb diffs the group instead and says so.
    """

    def test_rename_is_reported_not_swallowed(self, interface_env, monkeypatch):
        group = _Group()
        group.append(_Template("stdswitcher4", "Transform", "Folder"))
        node, definition = _hda_node(monkeypatch, group)

        def rename_on_write(written):
            # what Houdini does: the appended folder joins the tab series
            written.templates[-1]._name = "stdswitcher4_1"

        definition.setParmTemplateGroup.side_effect = rename_on_write

        result = hda.set_hda_interface(
            "/obj/unit",
            [
                {
                    "name": "controls",
                    "label": "Controls",
                    "type": "folder",
                    "children": [{"name": "stud_count", "type": "int"}],
                }
            ],
        )

        assert result["renamed_by_houdini"] == [
            {"requested": "controls", "stored_as": "stdswitcher4_1", "label": "Controls"}
        ]
        assert "tab group" in result["note"]
        # And the interface is still reported, not lost:
        assert result["applied"][0]["children"][0]["name"] == "stud_count"

    def test_child_parms_are_checked_on_the_instance(self, interface_env, monkeypatch):
        group = _Group()
        node, _ = _hda_node(monkeypatch, group)
        node._parms = [_Parm("stud_count", False)]

        result = hda.set_hda_interface(
            "/obj/unit",
            [
                {
                    "name": "controls",
                    "type": "folder",
                    "children": [
                        {"name": "stud_count", "type": "int"},
                        {"name": "bevel", "type": "float"},
                    ],
                }
            ],
        )

        assert result["instance_parms_present"] == ["stud_count"]
        assert result["instance_parms_missing"] == ["bevel"]


class TestComponentNames:
    def test_schemes_predict_what_houdini_creates(self):
        assert hda._component_names(_template("fh_range", components=2, scheme="Base1")) == [
            "fh_range1",
            "fh_range2",
        ]
        assert hda._component_names(_template("fh_xy", components=2, scheme="XYZW")) == [
            "fh_xyx",
            "fh_xyy",
        ]
        assert hda._component_names(_template("clr", components=3, scheme="RGBA")) == [
            "clrr",
            "clrg",
            "clrb",
        ]
        assert hda._component_names(_template("single")) == ["single"]
        assert hda._component_names(_template("item#", components=2)) == ["item#"]
        assert hda._component_names(_template("f", "Folder")) == ["f"]

    def test_a_collision_between_two_templates_is_named(self):
        group = MagicMock()
        group.entries.return_value = [
            _template("fh_range", components=2, scheme="Base1"),
            _template("fh_range2", components=2, scheme="XYZW"),
        ]
        collisions = hda._component_collisions(group)
        assert collisions == [{"component": "fh_range2", "templates": ["fh_range", "fh_range2"]}]

    def test_no_collision_between_distinct_names(self):
        group = MagicMock()
        group.entries.return_value = [
            _template("fh_range", components=2, scheme="Base1"),
            _template("fh_xy", components=2, scheme="XYZW"),
        ]
        assert hda._component_collisions(group) == []


class TestEditHdaInterface:
    def _asset(self, monkeypatch, names=("stdswitcher", "t", "r", "s")):
        node = _node("/obj/asset1", "asset")
        definition = node.type.return_value.definition.return_value
        definition.nodeTypeName.return_value = "asset"
        definition.libraryFilePath.return_value = "/tmp/asset.hda"
        templates = {name: _template(name) for name in names}
        group = MagicMock()
        # The group under edit behaves like a list: remove() drops the entry.
        group.entries.side_effect = lambda: list(group.live)
        group.live = list(templates.values())
        group.remove.side_effect = lambda t: group.live.remove(t)
        group.find.side_effect = lambda name: templates.get(name)
        group.findFolder.return_value = None
        # What Houdini stores after the write: here, everything it started with
        # (built-ins come back), plus whatever the test appends to `stored`.
        stored = MagicMock()
        stored.entries.side_effect = lambda: list(templates.values()) + list(stored.extra)
        stored.extra = []
        stored.isHidden.return_value = True
        stored.isFolderHidden.return_value = True
        definition.parmTemplateGroup.side_effect = [group, stored]
        group.stored = stored
        node.parms.return_value = []
        node.parmTuples.return_value = []
        hou = hda.hou
        monkeypatch.setattr(hou, "node", lambda path: node)
        monkeypatch.setattr(hda, "_get_node", lambda path: node)
        monkeypatch.setattr(hda, "_component_collisions", lambda g: [])
        monkeypatch.setattr(hda, "_describe_template", lambda t: {"name": t.name()})
        return node, definition, group, templates

    def test_ops_are_applied_in_order_and_written_once(self, monkeypatch):
        node, definition, group, templates = self._asset(monkeypatch)
        built = _template("stud_count", "Int")
        monkeypatch.setattr(
            hda, "_build_parm_template", lambda spec, seen: (seen.add(spec["name"]), built)[1]
        )
        group.insertAfter.side_effect = lambda anchor, t: group.live.append(t)
        group.stored.extra.append(built)
        result = hda.edit_hda_interface(
            "/obj/asset1",
            [
                {"op": "insert", "spec": {"name": "stud_count", "type": "int"}, "after": "t"},
                {"op": "hide", "name": "r"},
                {"op": "remove", "name": "s"},
            ],
        )
        group.insertAfter.assert_called_once_with(templates["t"], built)
        group.hide.assert_called_once_with(templates["r"], True)
        group.remove.assert_called_once_with(templates["s"])
        definition.setParmTemplateGroup.assert_called_once_with(group)
        assert result["added"] == ["stud_count"]
        assert result["ops"][0]["placed"] == "after t"
        assert result["ops"][1]["stored_hidden"] is True
        assert result["removed"] == []
        assert result["reinstated_by_houdini"] == ["s"]

    def test_a_removed_builtin_that_comes_back_is_reported(self, monkeypatch):
        node, definition, group, templates = self._asset(monkeypatch)
        # Houdini re-adds `s` after the write: the stored group still has it.
        result = hda.edit_hda_interface("/obj/asset1", [{"op": "remove", "name": "s"}])
        assert result["removed"] == []
        assert result["reinstated_by_houdini"] == ["s"]
        assert "cannot be removed" in result["note"]

    def test_a_failing_op_writes_nothing(self, monkeypatch):
        node, definition, group, templates = self._asset(monkeypatch)
        with pytest.raises(ValueError, match="No parameter named or folder labelled 'tx'"):
            hda.edit_hda_interface("/obj/asset1", [{"op": "remove", "name": "tx"}])
        definition.setParmTemplateGroup.assert_not_called()

    def test_a_collision_is_refused_before_the_write(self, monkeypatch):
        node, definition, group, templates = self._asset(monkeypatch)
        monkeypatch.setattr(
            hda,
            "_component_collisions",
            lambda g: [{"component": "fh_range2", "templates": ["fh_range", "fh_range2"]}],
        )
        with pytest.raises(ValueError, match="'fh_range2' is produced by both"):
            hda.edit_hda_interface("/obj/asset1", [{"op": "hide", "name": "r"}])
        definition.setParmTemplateGroup.assert_not_called()

    def test_dry_run_reports_the_plan_without_writing(self, monkeypatch):
        node, definition, group, templates = self._asset(monkeypatch)
        result = hda.edit_hda_interface("/obj/asset1", [{"op": "hide", "name": "r"}], dry_run=True)
        assert result["dry_run"] is True
        definition.setParmTemplateGroup.assert_not_called()

    def test_modify_clones_and_replaces(self, monkeypatch):
        node, definition, group, templates = self._asset(monkeypatch)
        clone = templates["r"].clone.return_value
        clone.name.return_value = "r"
        result = hda.edit_hda_interface(
            "/obj/asset1",
            [{"op": "set_conditional", "name": "r", "disable_when": "{ t == 0 }"}],
        )
        hou = hda.hou
        clone.setConditional.assert_called_once_with(hou.parmCondType.DisableWhen, "{ t == 0 }")
        group.replace.assert_called_once_with(templates["r"], clone)
        assert result["ops"][0]["changed"] == ["disable_when"]

    def test_an_unknown_op_is_named(self, monkeypatch):
        self._asset(monkeypatch)
        with pytest.raises(ValueError, match="unknown op 'explode'"):
            hda.edit_hda_interface("/obj/asset1", [{"op": "explode", "name": "r"}])

    def test_a_non_asset_is_refused(self, monkeypatch):
        node = _node("/obj/geo1", "geo")
        node.type.return_value.definition.return_value = None
        monkeypatch.setattr(hda, "_get_node", lambda path: node)
        with pytest.raises(ValueError, match="not an HDA instance"):
            hda.edit_hda_interface("/obj/geo1", [{"op": "hide", "name": "r"}])


class TestExtendedSpecs:
    def test_naming_scheme_is_looked_up_case_insensitively(self):
        hou = hda.hou
        assert hda._naming_scheme({"naming_scheme": "base1"}, None) is hou.parmNamingScheme.Base1
        assert hda._naming_scheme({}, "default") == "default"
        with pytest.raises(ValueError, match="Unknown naming_scheme"):
            hda._naming_scheme({"naming_scheme": "abc", "name": "x"}, None)

    def test_component_defaults_take_a_list_or_repeat_a_scalar(self):
        assert hda._component_defaults({"default": [0, 1]}, 2, float, 0.0) == (0.0, 1.0)
        assert hda._component_defaults({"default": 3}, 2, int, 0) == (3, 3)
        with pytest.raises(ValueError, match="2 value\\(s\\) for 3"):
            hda._component_defaults({"default": [0, 1], "name": "v"}, 3, float, 0.0)

    def test_a_callback_defaults_to_python(self):
        hou = hda.hou
        kwargs = hda._common_template_kwargs({"callback": "hou.pwd().cook()", "hidden": True})
        assert kwargs["script_callback"] == "hou.pwd().cook()"
        assert kwargs["script_callback_language"] is hou.scriptLanguage.Python
        assert kwargs["is_hidden"] is True

    def test_default_expression_is_spread_over_components(self):
        template = MagicMock()
        template.numComponents.return_value = 2
        hda._apply_default_expression(template, {"name": "v", "default_expression": "ch('tx')"})
        template.setDefaultExpression.assert_called_once_with(("ch('tx')", "ch('tx')"))
        with pytest.raises(ValueError, match="1 default expression"):
            hda._apply_default_expression(
                template, {"name": "v", "default_expression": ["ch('tx')"]}
            )
