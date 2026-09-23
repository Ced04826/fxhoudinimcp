"""The 2026-09-23 batch of small defects, one class per item.

a. find_error_nodes aborted on a node whose errors() raised hou.PermissionError,
   and get_node_errors_detailed counted such a node as error-free.
b. disconnect_node(disconnect_all=True) walked a merge's inputs upwards and ran
   past the end as the merge compacted, then reported nothing about what was
   left.
c. Every create/connect/flag call moved the network editor (and every pane
   linked to it) to the node it touched.
d. `raise SystemExit` in execute_python escaped the handler.
e. A dict with number keys made execute_python call its return value lossy.
f. build_network repeated the queried node's geometry inside `inspected`.
g. build_network set multiparm instances before their count, never checked an
   instance number against the count, and get_node_card did not say where
   instance numbers start.

hou is a stub throughout; the multiparm start default of 1 (no
``multistartoffset`` tag) is an assumption to be checked live on an Add SOP.
"""

from __future__ import annotations

# Built-in
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third-party
import pytest

sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

# Internal
import fxhoudinimcp_server.config as config  # noqa: E402
import fxhoudinimcp_server.handlers.code_handlers as code  # noqa: E402
import fxhoudinimcp_server.handlers.context_handlers as context  # noqa: E402
import fxhoudinimcp_server.handlers.graph_handlers as graph  # noqa: E402
import fxhoudinimcp_server.handlers.node_handlers as nodes  # noqa: E402
import fxhoudinimcp_server.handlers.viewport_handlers as viewport  # noqa: E402
from fxhoudinimcp_server.handlers.state_receipt_helpers import json_safe  # noqa: E402


class _HouError(Exception):
    """Stands in for hou.Error, which the stub does not define."""


class _PermissionError(_HouError):
    """hou.PermissionError is a hou.Error, not the builtin PermissionError."""


###### a. Unreadable nodes are listed, never passed as clean


class _ErrNode:
    def __init__(self, path, errors=(), warnings=(), raises=None, children=(), children_raise=None):
        self._path = path
        self._errors, self._warnings = errors, warnings
        self._raises = raises
        self._children = list(children)
        self._children_raise = children_raise

    def path(self):
        return self._path

    def name(self):
        return self._path.rsplit("/", 1)[-1]

    def type(self):
        return SimpleNamespace(name=lambda: "null")

    def errors(self):
        if self._raises:
            raise self._raises
        return self._errors

    def warnings(self):
        if self._raises:
            raise self._raises
        return self._warnings

    def children(self):
        if self._children_raise:
            raise self._children_raise
        return self._children

    def allSubChildren(self):  # noqa: N802 - HOM spelling
        found = []
        for child in self._children:
            found.append(child)
            found += child.allSubChildren()
        return found

    def parms(self):
        return []


def _tree():
    locked = _ErrNode("/obj/asset/inside", raises=_PermissionError("locked asset contents"))
    broken = _ErrNode("/obj/geo1/file1", errors=("Unable to read file",))
    shut = _ErrNode("/obj/shut", children_raise=_HouError("no children for you"))
    asset = _ErrNode("/obj/asset", children=[locked])
    geo = _ErrNode("/obj/geo1", children=[broken])
    root = _ErrNode("/obj", children=[asset, geo, shut])
    return root, locked, broken


class TestFindErrorNodes:
    @pytest.fixture(autouse=True)
    def _stub(self, monkeypatch):
        monkeypatch.setattr(viewport.hou, "Error", _HouError, raising=False)

    def test_a_permission_error_is_listed_and_the_scan_goes_on(self, monkeypatch):
        root, locked, broken = _tree()
        monkeypatch.setattr(viewport.hou, "node", lambda path: root, raising=False)

        result = viewport.find_error_nodes("/obj")

        # The node after the unreadable one was still read.
        assert [n["path"] for n in result["error_nodes"]] == [broken.path()]
        paths = [n["path"] for n in result["unreadable_nodes"]]
        assert paths.count(locked.path()) == 2  # errors() and warnings()
        assert "/obj/shut" in paths
        first = next(n for n in result["unreadable_nodes"] if n["path"] == locked.path())
        assert "_PermissionError" in first["error"] and "locked asset" in first["error"]
        assert result["unreadable_count"] == len(result["unreadable_nodes"])

    def test_a_clean_scan_says_nothing_was_unreadable(self, monkeypatch):
        root = _ErrNode("/obj", children=[_ErrNode("/obj/a")])
        monkeypatch.setattr(viewport.hou, "node", lambda path: root, raising=False)
        result = viewport.find_error_nodes("/obj")
        assert result["unreadable_nodes"] == []
        assert result["unreadable_count"] == 0


class TestGetNodeErrorsDetailed:
    def test_an_unreadable_node_is_listed_not_counted_clean(self, monkeypatch):
        root, locked, broken = _tree()
        monkeypatch.setattr(context, "_get_node", lambda path: root)

        result = context._get_node_errors_detailed(root_path="/obj")

        assert result["error_node_count"] == 1
        assert [d["node_path"] for d in result["details"]] == [broken.path()]
        assert result["unreadable_nodes"] == [
            {
                "path": locked.path(),
                "error": "errors(): _PermissionError: locked asset contents; "
                "warnings(): _PermissionError: locked asset contents",
            }
        ]
        assert result["unreadable_count"] == 1

    def test_readable_warnings_survive_an_unreadable_error_list(self, monkeypatch):
        class Half(_ErrNode):
            def errors(self):
                raise _PermissionError("no")

        node = Half("/obj/half", warnings=("careful",))
        monkeypatch.setattr(context, "_get_node", lambda path: node)
        result = context._get_node_errors_detailed(node_path="/obj/half")
        assert result["details"][0]["warnings"] == ["careful"]
        assert result["unreadable_nodes"][0]["path"] == "/obj/half"


###### b. disconnect_all walks downwards and reads back


class _Conn:
    def __init__(self, index, item):
        self._index, self._item = index, item

    def inputIndex(self):  # noqa: N802 - HOM spelling
        return self._index

    def inputItem(self):  # noqa: N802 - HOM spelling
        return self._item


class _Merge:
    """A merge that compacts: removing input i shifts the ones above it down."""

    def __init__(self, count):
        self.wires = [SimpleNamespace(path=lambda i=i: f"/obj/geo1/box{i}") for i in range(count)]
        self.refuse: set = set()
        self.ignore: set = set()
        self.calls: list = []

    def path(self):
        return "/obj/geo1/merge1"

    def inputConnections(self):  # noqa: N802 - HOM spelling
        return [_Conn(i, item) for i, item in enumerate(self.wires)]

    def setInput(self, index, item, output=0):  # noqa: N802 - HOM spelling
        self.calls.append(index)
        if index in self.refuse:
            raise RuntimeError(f"input {index} is locked")
        if index >= len(self.wires):
            raise IndexError("tuple index out of range")
        if index not in self.ignore:
            del self.wires[index]


class TestDisconnectAll:
    def _install(self, monkeypatch, node):
        monkeypatch.setattr(nodes, "_get_node", lambda path: node)

    def test_a_29_input_merge_is_emptied(self, monkeypatch):
        merge = _Merge(29)
        self._install(monkeypatch, merge)

        result = nodes.disconnect_node(merge.path(), disconnect_all=True)

        assert result["success"] is True, result
        assert merge.calls == list(range(28, -1, -1))
        assert result["disconnected_inputs"] == list(range(28, -1, -1))
        assert result["remaining"] == []
        assert "error" not in result

    def test_a_failure_part_way_reports_done_and_left(self, monkeypatch):
        merge = _Merge(8)
        merge.refuse = {5}
        self._install(monkeypatch, merge)

        result = nodes.disconnect_node(merge.path(), disconnect_all=True)

        assert result["success"] is False
        assert result["disconnected_inputs"] == [7, 6]
        assert [r["index"] for r in result["remaining"]] == [0, 1, 2, 3, 4, 5]
        assert result["remaining"][0]["source"] == "/obj/geo1/box0"
        assert "input 5" in result["error"] and "locked" in result["error"]

    def test_a_disconnect_that_does_not_stick_is_not_success(self, monkeypatch):
        merge = _Merge(3)
        merge.ignore = {1}
        self._install(monkeypatch, merge)

        result = nodes.disconnect_node(merge.path(), disconnect_all=True)

        assert result["success"] is False
        assert [r["index"] for r in result["remaining"]] == [0]
        assert "still connected" in result["error"]

    def test_the_single_index_path_is_unchanged(self, monkeypatch):
        merge = _Merge(3)
        self._install(monkeypatch, merge)
        result = nodes.disconnect_node(merge.path(), input_index=1)
        assert result == {
            "success": True,
            "node_path": merge.path(),
            "disconnected_inputs": [1],
        }


###### c. The editor stays where the user left it unless asked


class TestFocusEditorSwitch:
    @pytest.mark.parametrize(
        ("hou_value", "env_value", "expected"),
        [
            (None, None, False),  # default off in this fork
            (None, "1", True),
            ("1", None, True),
            ("0", "1", False),  # hou.getenv wins, as for AUTO_LAYOUT
            (" OFF ", None, False),
        ],
    )
    def test_the_flag_reads_like_auto_layout(self, monkeypatch, hou_value, env_value, expected):
        monkeypatch.setattr(config.hou, "getenv", lambda name: hou_value, raising=False)
        if env_value is None:
            monkeypatch.delenv("FXHOUDINIMCP_FOCUS_EDITOR", raising=False)
        else:
            monkeypatch.setenv("FXHOUDINIMCP_FOCUS_EDITOR", env_value)
        assert config.focus_editor_enabled() is expected

    def _editor(self, monkeypatch, enabled):
        pane = MagicMock()
        pane.type.return_value = nodes.hou.paneTabType.NetworkEditor
        monkeypatch.setattr(nodes.hou.ui, "paneTabs", lambda: [pane], raising=False)
        monkeypatch.setattr(nodes, "focus_editor_enabled", lambda: enabled)
        layouts = []
        monkeypatch.setattr(
            nodes, "layout_if_enabled", lambda parent, place=True: layouts.append((parent, place))
        )
        parent = MagicMock()
        parent.path.return_value = "/obj/geo1"
        node = MagicMock()
        node.parent.return_value = parent
        return pane, node, parent, layouts

    def test_off_places_but_does_not_navigate(self, monkeypatch):
        pane, node, parent, layouts = self._editor(monkeypatch, enabled=False)
        nodes._focus_network_editor(node)
        assert layouts == [(parent, True)]
        pane.cd.assert_not_called()
        pane.setCurrentNode.assert_not_called()
        pane.homeToSelection.assert_not_called()

    def test_on_navigates_as_before(self, monkeypatch):
        pane, node, parent, layouts = self._editor(monkeypatch, enabled=True)
        nodes._focus_network_editor(node, place_unpositioned=False)
        assert layouts == [(parent, False)]
        pane.cd.assert_called_once_with("/obj/geo1")
        pane.setCurrentNode.assert_called_once_with(node)
        pane.homeToSelection.assert_called_once_with()


###### d. SystemExit in user code is a failed run, not a lost call


class TestExecutePythonExits:
    def test_system_exit_with_a_code(self):
        result = code._execute_python("print('before')\nraise SystemExit(3)\nprint('after')")
        assert result["execution_success"] is False
        assert result["success"] is False
        assert result["stdout"] == "before\n"
        assert result["error_summary"] == "SystemExit: exit code 3"
        assert "SystemExit" in result["error"]

    def test_bare_sys_exit(self):
        result = code._execute_python("import sys\nsys.exit()")
        assert result["execution_success"] is False
        assert result["error_summary"] == "SystemExit: exit code None"

    def test_keyboard_interrupt(self):
        result = code._execute_python("print('x')\nraise KeyboardInterrupt")
        assert result["execution_success"] is False
        assert result["error_summary"].startswith("KeyboardInterrupt")
        assert result["stdout"] == "x\n"

    def test_exit_in_the_return_expression(self):
        result = code._execute_python(
            "x = 1", return_expression="__import__('sys').exit(2) if x else 0"
        )
        assert result["execution_success"] is True
        assert result["success"] is False
        assert result["eval_error_summary"] == "SystemExit: exit code 2"

    def test_streams_are_restored(self):
        before = sys.stdout, sys.stderr
        code._execute_python("raise SystemExit(1)")
        assert (sys.stdout, sys.stderr) == before


###### e. Number keys are JSON's own convention, not a loss


class TestNumberKeys:
    def test_int_and_float_keys_are_counted_not_coerced(self):
        safe, notes = json_safe({344.0: "a", 7: {1: "b"}})
        assert safe == {"344.0": "a", "7": {"1": "b"}}
        assert notes == {"keys_stringified": 3}

    @pytest.mark.parametrize("key", [None, True, float("nan"), ("t", 1)])
    def test_other_keys_are_still_coerced(self, key):
        _safe, notes = json_safe({key: 1})
        assert notes["coerced_count"] == 1
        assert "keys_stringified" not in notes

    def test_a_key_collision_is_a_loss(self):
        safe, notes = json_safe({1: "int", "1": "str"})
        assert safe == {"1": "str"}
        assert notes["coerced_count"] == 1

    def test_execute_python_calls_it_lossless(self):
        result = code._execute_python(
            "d = {344.0: 1.5, 12: [1, 2], 'z': {0: 'a'}}", return_expression="d"
        )
        assert result["return_lossless"] is True
        assert result["output_complete"] is True
        assert result["return_keys_stringified"] == 3
        assert "return_coerced" not in result
        assert result["return_value"] == {"344.0": 1.5, "12": [1, 2], "z": {"0": "a"}}

    def test_strict_json_accepts_number_keys(self):
        result = code._execute_python("d = {1: 2}", return_expression="d", return_format="json")
        assert result["success"] is True
        assert result["return_value"] == {"1": 2}

    def test_a_real_loss_is_still_flagged(self):
        result = code._execute_python("d = {1: object()}", return_expression="d")
        assert result["return_lossless"] is False
        assert result["return_coerced_count"] == 1


###### f and g. build_network over an Add-SOP-like fake


class _Kind:
    """A parmTemplateType / folderType member: compared by identity, has name()."""

    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


_FOLDER = _Kind("Folder")
_INT = _Kind("Int")
_TABS = _Kind("Tabs")
_MULTI = _Kind("MultiparmBlock")


class _Template:
    def __init__(self, name, kind=_INT):
        self._name, self._kind = name, kind

    def name(self):
        return self._name

    def label(self):
        return self._name

    def type(self):
        return self._kind

    def numComponents(self):  # noqa: N802 - HOM spelling
        return 1

    def isHidden(self):  # noqa: N802 - HOM spelling
        return False

    def defaultValue(self):  # noqa: N802 - HOM spelling
        return (0,)

    def menuItems(self):  # noqa: N802 - HOM spelling
        return ()


class _Folder(_Template):
    def __init__(self, name, children, folder_type=_TABS, tags=None):
        super().__init__(name, _FOLDER)
        self._children, self._folder_type, self._tags = children, folder_type, tags or {}

    def parmTemplates(self):  # noqa: N802 - HOM spelling
        return tuple(self._children)

    def folderType(self):  # noqa: N802 - HOM spelling
        return self._folder_type

    def tags(self):
        return dict(self._tags)


class _Group:
    def __init__(self, entries):
        self._entries = entries

    def entries(self):
        return tuple(self._entries)

    def entriesWithoutFolders(self):  # noqa: N802 - HOM spelling
        flat = []

        def walk(templates):
            for template in templates:
                if isinstance(template, _Folder):
                    walk(template.parmTemplates())
                else:
                    flat.append(template)

        walk(self._entries)
        return tuple(flat)


class _AddType:
    def __init__(self, start_tag):
        tags = {} if start_tag is None else {"multistartoffset": start_tag}
        self.start = 1 if start_tag is None else int(start_tag)
        block = _Folder(
            "points",
            [_Template("usept#"), _Template("weight#")],
            folder_type=_MULTI,
            tags=tags,
        )
        self._group = _Group([_Folder("Points", [block])])

    def name(self):
        return "add"

    def description(self):
        return "Add"

    def maxNumInputs(self):  # noqa: N802 - HOM spelling
        return 1

    def minNumInputs(self):  # noqa: N802 - HOM spelling
        return 0

    def maxNumOutputs(self):  # noqa: N802 - HOM spelling
        return 1

    def definition(self):
        return None

    def parmTemplateGroup(self):  # noqa: N802 - HOM spelling
        return self._group


class _AddParm:
    def __init__(self, node, name):
        self._node, self._name = node, name

    def name(self):
        return self._name

    def parmTemplate(self):  # noqa: N802 - HOM spelling
        return _Template(self._name)

    def menuItems(self):  # noqa: N802 - HOM spelling
        return ()

    def set(self, value, follow_parm_reference=True):  # noqa: FBT002 - HOM's own default
        self._node.values[self._name] = value
        self._node.set_order.append(self._name)


class _AddNode:
    """Instance parms exist only once the count says so, as on a real node."""

    def __init__(self, net, name):
        self._net, self._name = net, name
        self.values = {"points": 0}
        self.set_order: list = []

    def name(self):
        return self._name

    def path(self):
        return f"{self._net.path()}/{self._name}"

    def type(self):
        return self._net.add_type

    def _live(self):
        start = self._net.add_type.start
        names = ["points"]
        for number in range(start, start + int(self.values["points"])):
            names += [f"usept{number}", f"weight{number}"]
        return names

    def parm(self, name):
        return _AddParm(self, name) if name in self._live() else None

    def parms(self):
        return [_AddParm(self, name) for name in self._live()]

    def parmTuple(self, name):  # noqa: N802 - HOM spelling
        return None

    def parmTuples(self):  # noqa: N802 - HOM spelling
        return ()

    def errors(self):
        return ()

    def warnings(self):
        return ()

    def isBypassed(self):  # noqa: N802 - HOM spelling
        return False

    def inputConnections(self):  # noqa: N802 - HOM spelling
        return []

    def cook(self, force=False):  # noqa: FBT002 - HOM's own default
        pass

    def destroy(self):
        self._net.kids.remove(self)


class _Net:
    def __init__(self, start_tag=None):
        self.add_type = _AddType(start_tag)
        self.kids: list = []
        self.counter = 0
        category = SimpleNamespace(name=lambda: "Sop", nodeTypes=lambda: {"add": self.add_type})
        self.category = category

    def path(self):
        return "/obj/geo1"

    def childTypeCategory(self):  # noqa: N802 - HOM spelling
        return self.category

    def children(self):
        return list(self.kids)

    def node(self, name):
        return next((k for k in self.kids if name in (k.name(), k.path())), None)

    def createNode(self, type_name, name=None, **_):  # noqa: N802 - HOM spelling
        self.counter += 1
        node = _AddNode(self, name or f"{type_name}{self.counter}")
        self.kids.append(node)
        return node

    def displayNode(self):  # noqa: N802 - HOM spelling
        return None

    def renderNode(self):  # noqa: N802 - HOM spelling
        return None


def _raise(*_a, **_k):
    raise RuntimeError("no preferred type in the stub")


@pytest.fixture
def add_net(monkeypatch):
    """A /obj/geo1 whose only type is an Add-SOP-like node with one multiparm."""

    def make(start_tag=None):
        net = _Net(start_tag)
        hou = graph.hou
        monkeypatch.setattr(graph, "_TYPE_KNOWLEDGE", {})
        monkeypatch.setattr(graph, "_MULTIPARM_FOLDERS", (_MULTI,))
        monkeypatch.setattr(graph, "place_new_nodes", lambda nodes: None)
        monkeypatch.setattr(
            hou,
            "parmTemplateType",
            SimpleNamespace(Folder=_FOLDER, Int=_INT, Menu=object(), String=object()),
            raising=False,
        )
        monkeypatch.setattr(hou, "FolderParmTemplate", _Folder, raising=False)
        monkeypatch.setattr(hou, "preferredNodeType", _raise, raising=False)
        monkeypatch.setattr(hou, "selectedNodes", lambda: (), raising=False)
        monkeypatch.setattr(
            hou, "OperationFailed", type("OperationFailed", (Exception,), {}), raising=False
        )
        monkeypatch.setattr(
            hou, "node", lambda path: net if path == net.path() else net.node(path), raising=False
        )
        return net

    return make


def _build(net, specs, **kwargs):
    kwargs.setdefault("layout", False)
    return graph.build_network(net.path(), specs, **kwargs)


class TestQueriedGeometryIsNotRepeated:
    def test_the_queried_row_points_at_the_top_level(self, add_net, monkeypatch):
        net = add_net()
        monkeypatch.setattr(
            graph, "_geometry_summary", lambda node: {"points": len(node.name()), "of": node.name()}
        )
        result = _build(
            net,
            [{"type": "add", "name": "a"}, {"type": "add", "name": "bb"}],
            inspect_nodes=["a", "bb"],
        )
        assert result["success"] is True, result
        assert result["queried_node"] == "/obj/geo1/a"
        assert result["geometry"] == {"points": 1, "of": "a"}
        rows = {row["name"]: row for row in result["inspected"]}
        assert rows["a"]["geometry"] == "see top-level geometry"
        # Every other inspected node keeps its own figures.
        assert rows["bb"]["geometry"] == {"points": 2, "of": "bb"}

    def test_no_geometry_stays_none(self, add_net, monkeypatch):
        net = add_net()
        monkeypatch.setattr(graph, "_geometry_summary", lambda node: None)
        result = _build(net, [{"type": "add", "name": "a"}])
        assert result["geometry"] is None
        assert "geometry" not in result["inspected"][0]


class TestMultiparmCountsFirst:
    def test_an_instance_before_its_count_in_the_spec_still_builds(self, add_net):
        net = add_net("0")
        result = _build(
            net,
            [{"type": "add", "name": "a", "parms": {"usept1": 1, "weight0": 0.5, "points": 2}}],
        )
        assert result["success"] is True, result
        node = net.node("a")
        assert node.set_order == ["points", "usept1", "weight0"]
        assert node.values["usept1"] == 1

    def test_other_parms_keep_spec_order(self, add_net):
        net = add_net()
        assert graph._count_parms_first(
            {"a": 1, "usept1": 1, "points": 1, "b": 2},
            graph._multiparm_blocks(net.add_type),
        ) == [("points", 1), ("a", 1), ("usept1", 1), ("b", 2)]


class TestMultiparmRangeValidation:
    def test_an_instance_past_the_count_is_refused_naming_count_and_start(self, add_net):
        net = add_net("0")
        result = _build(
            net,
            [{"type": "add", "name": "a", "parms": {"points": 2, "usept2": 1}}],
            dry_run=True,
        )
        assert result["valid"] is False
        message = " ".join(result["errors"])
        assert "'usept2' is instance 2 of multiparm 'points'" in message
        assert "2 instance(s) starting at 0" in message
        assert "valid: 0..1" in message
        assert net.kids == [], "a dry run leaves nothing behind"

    def test_the_default_start_is_one(self, add_net):
        net = add_net()  # no multistartoffset tag
        bad = _build(
            net, [{"type": "add", "name": "a", "parms": {"points": 2, "usept0": 1}}], dry_run=True
        )
        assert bad["valid"] is False
        assert "starting at 1" in " ".join(bad["errors"])
        good = _build(
            net,
            [{"type": "add", "name": "b", "parms": {"points": 2, "usept1": 1, "weight2": 0.1}}],
            dry_run=True,
        )
        assert good["valid"] is True, good

    def test_a_zero_count_admits_no_instance(self, add_net):
        net = add_net("0")
        result = _build(
            net, [{"type": "add", "name": "a", "parms": {"points": 0, "usept0": 1}}], dry_run=True
        )
        assert result["valid"] is False
        assert "valid: none" in " ".join(result["errors"])

    def test_without_a_count_in_the_spec_nothing_is_range_checked(self, add_net):
        net = add_net("0")
        result = _build(net, [{"type": "add", "name": "a", "parms": {"usept9": 1}}], dry_run=True)
        assert result["valid"] is True, result


class TestNodeCardStartOffset:
    @pytest.mark.parametrize(
        ("tags", "expected"),
        [
            ({}, 1),
            ({"multistartoffset": "0"}, 0),
            ({"multistartoffset": " 2 "}, 2),
            ({"multistartoffset": "x"}, None),
        ],
    )
    def test_the_start_comes_from_the_tag(self, tags, expected):
        folder = _Folder("points", [], folder_type=_MULTI, tags=tags)
        assert graph._multiparm_start(folder) == expected

    def test_a_folder_whose_tags_cannot_be_read_defaults_to_one(self):
        folder = MagicMock()
        folder.tags.side_effect = RuntimeError("no tags")
        assert graph._multiparm_start(folder) == 1

    def test_the_card_carries_start_offset(self, add_net, monkeypatch):
        net = add_net("0")
        monkeypatch.setattr(
            graph.hou, "nodeTypeCategories", lambda: {"Sop": net.category}, raising=False
        )
        monkeypatch.setattr(
            graph,
            "_connectors_for_type",
            lambda context, node_type: ({"inputs": [], "outputs": []}, None),
        )
        card = graph.get_node_card("add", "Sop", include_help=False)
        assert card["multiparms"] == [
            {
                "count_parm": "points",
                "label": "points",
                "folder_type": "MultiparmBlock",
                "start_offset": 0,
                "instance_parms": ["usept#", "weight#"],
            }
        ]
