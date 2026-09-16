"""get_node_card had no connector list, and build_network wired by index only.

The card reported min/max input counts and nothing else about the connectors,
so the index of `texcoord` on mtlximage (3) could not be read anywhere; and a
build_network spec could name an input only by index, while connect_nodes had
taken an input_name for months. The card now lists `inputs` / `outputs` with
index, name and label, probed on a throwaway instance (hou.NodeType has no
inputNames), and build_network resolves `input_name` at dry-run time with a
did-you-mean over names and labels.

hou is mocked here; the live check ran on Houdini 22.0.429.
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
import fxhoudinimcp_server.handlers.graph_handlers as graph  # noqa: E402

hou = graph.hou


def _probe(inputs, labels, outputs=("output1",), data_types=()):
    probe = MagicMock()
    probe.inputNames.return_value = list(inputs)
    probe.inputLabels.return_value = list(labels)
    probe.inputDataTypes.return_value = list(data_types)
    probe.outputNames.return_value = list(outputs)
    probe.outputLabels.return_value = [o.title() for o in outputs]
    probe.parms.return_value = []
    probe.parmTuples.return_value = []
    return probe


class TestConnectorsOfANode:
    def test_index_name_label_and_data_type_are_reported(self):
        probe = _probe(
            ["file", "layer", "default", "texcoord"],
            ["Filename", "Layer", "Default Color", "Texture Coordinates"],
            outputs=["out"],
            data_types=["ustring", "ustring", "color", "vector2"],
        )
        connectors = graph._connectors_of(probe)
        assert connectors["inputs"][3] == {
            "index": 3,
            "name": "texcoord",
            "label": "Texture Coordinates",
            "data_type": "vector2",
        }
        assert connectors["outputs"] == [{"index": 0, "name": "out", "label": "Out"}]

    def test_a_node_without_data_types_still_lists_its_inputs(self):
        probe = _probe(["input1", "input2"], ["Geometry for operation", "Spine curve"])
        probe.inputDataTypes.side_effect = AttributeError("SOPs have none")
        inputs = graph._connectors_of(probe)["inputs"]
        assert [i["label"] for i in inputs] == ["Geometry for operation", "Spine curve"]
        assert "data_type" not in inputs[0]

    def test_a_name_or_a_label_resolves_to_the_index(self):
        connectors = graph._connectors_of(
            _probe(["input1", "input2"], ["Geometry for operation", "Spine curve"])
        )
        assert graph._connector_index(connectors, "input2") == 1
        assert graph._connector_index(connectors, "Spine curve") == 1
        assert graph._connector_index(connectors, "spine") is None
        assert graph._connector_index(None, "input1") is None

    def test_the_hint_names_close_connectors_or_all_of_them(self):
        connectors = graph._connectors_of(
            _probe(["base_color", "sheen_color", "coat_color"], ["Base Color", "Sheen", "Coat"])
        )
        assert "base_color" in graph._connector_hint(connectors, "base_colour")
        assert "Inputs:" in graph._connector_hint(connectors, "zzz")
        assert graph._connector_hint(None, "x") == ""


class TestProbingAType:
    def test_no_container_for_the_category_means_no_probe(self, monkeypatch):
        node = MagicMock()
        monkeypatch.setattr(hou, "node", node)
        assert graph._connectors_for_type("Shop", MagicMock()) is None
        node.assert_not_called()

    def test_the_probe_is_made_in_a_throwaway_container_and_destroyed(self, monkeypatch):
        root = MagicMock()
        scratch = MagicMock()
        root.createNode.return_value = scratch
        scratch.createNode.return_value = _probe(["input1"], ["Input 1"])
        monkeypatch.setattr(hou, "node", lambda path: root)
        node_type = MagicMock()
        node_type.name.return_value = "polyextrude::2.0"

        connectors = graph._connectors_for_type("Sop", node_type)

        root.createNode.assert_called_once_with("geo", "fxhoudinimcp_card_probe")
        scratch.createNode.assert_called_once_with("polyextrude::2.0")
        assert connectors["inputs"] == [{"index": 0, "name": "input1", "label": "Input 1"}]
        scratch.destroy.assert_called_once()

    def test_the_container_is_destroyed_even_when_the_type_fails(self, monkeypatch):
        root = MagicMock()
        scratch = MagicMock()
        root.createNode.return_value = scratch
        scratch.createNode.side_effect = RuntimeError("no license for this type")
        monkeypatch.setattr(hou, "node", lambda path: root)
        assert graph._connectors_for_type("Sop", MagicMock()) is None
        scratch.destroy.assert_called_once()

    def test_the_parm_probe_fills_connectors_when_asked(self, monkeypatch):
        scratch = MagicMock()
        scratch.createNode.return_value = _probe(["input1"], ["Input 1"])
        monkeypatch.setattr(graph, "_instance_patterns", lambda t: [])
        connectors: dict = {}
        result = graph._parm_names_for_type(scratch, MagicMock(), connectors)
        assert len(result) == 4
        assert connectors["inputs"] == [{"index": 0, "name": "input1", "label": "Input 1"}]

    def test_the_parm_probe_is_unchanged_for_callers_without_the_argument(self):
        scratch = MagicMock()
        scratch.createNode.return_value = _probe(["input1"], ["Input 1"])
        parm_names, tuple_names, menus, patterns = graph._parm_names_for_type(scratch, MagicMock())
        assert (parm_names, tuple_names, menus) == (set(), set(), {})


class TestBuildNetworkWiresByName:
    def _network(self, monkeypatch, input_names, input_labels):
        parent = MagicMock()
        parent.path.return_value = "/obj/mat1"
        parent.children.return_value = []
        parent.displayNode.return_value = None
        parent.renderNode.return_value = None
        category = MagicMock()
        category.name.return_value = "Vop"
        parent.childTypeCategory.return_value = category
        srf_type = MagicMock()
        srf_type.name.return_value = "mtlxstandard_surface"
        srf_type.maxNumInputs.return_value = len(input_names)
        img_type = MagicMock()
        img_type.name.return_value = "mtlximage"
        img_type.maxNumInputs.return_value = 0
        types = {"mtlxstandard_surface": srf_type, "mtlximage": img_type}
        probe = _probe(input_names, input_labels)
        parent.createNode.return_value = probe
        monkeypatch.setattr(hou, "node", lambda path: parent if path == "/obj/mat1" else None)
        monkeypatch.setattr(graph, "_resolve_node_type", lambda cat, name: types.get(name))
        monkeypatch.setattr(graph, "_instance_patterns", lambda t: [])
        return parent

    def test_a_wrong_name_fails_the_dry_run_with_a_hint(self, monkeypatch):
        self._network(monkeypatch, ["base_color", "sheen_color"], ["Base Color", "Sheen Color"])
        result = graph.build_network(
            "/obj/mat1",
            [
                {"type": "mtlximage", "name": "img"},
                {
                    "type": "mtlxstandard_surface",
                    "name": "srf",
                    "inputs": [{"source": "img", "input_name": "base_colour"}],
                },
            ],
            dry_run=True,
        )
        assert result["valid"] is False
        assert "no input named 'base_colour'" in result["errors"][0]
        assert "base_color" in result["errors"][0]

    def test_a_known_name_validates(self, monkeypatch):
        self._network(monkeypatch, ["base_color", "sheen_color"], ["Base Color", "Sheen Color"])
        result = graph.build_network(
            "/obj/mat1",
            [
                {"type": "mtlximage", "name": "img"},
                {
                    "type": "mtlxstandard_surface",
                    "name": "srf",
                    "inputs": [{"source": "img", "input_name": "Sheen Color"}],
                },
            ],
            dry_run=True,
        )
        assert result["valid"] is True, result

    def test_an_index_past_the_connectors_is_still_refused(self, monkeypatch):
        self._network(monkeypatch, ["base_color"], ["Base Color"])
        result = graph.build_network(
            "/obj/mat1",
            [
                {"type": "mtlximage", "name": "img"},
                {
                    "type": "mtlxstandard_surface",
                    "name": "srf",
                    "inputs": [{"source": "img", "index": 5}],
                },
            ],
            dry_run=True,
        )
        assert result["valid"] is False
        assert "exceeds max inputs" in result["errors"][0]
