"""A subnet's input connectors could not be wired from by any verb.

They are `SubnetIndirectInput` items, not nodes: they have no path, so
connect_nodes, connect_nodes_batch and build_network — all of which name a
source by path — had no way to feed the first node of a chain built inside a
subnet from the subnet's own input. That took `node.setInput(0,
subnet.indirectInputs()[0])` in execute_python. The three verbs now take
`indirect_input=n` (build_network: `{"indirect_input": n}` in place of a
source), and an index past the subnet's connectors, or a plain node given as
the subnet, is refused by name.

hou is mocked here; the live check ran on Houdini 22.0.429, where the wire is
visible through inputConnections() — hou.Node.inputs() lists nodes only and
shows nothing for an indirect input.
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
import fxhoudinimcp_server.handlers.graph_handlers as graph  # noqa: E402
import fxhoudinimcp_server.handlers.node_handlers as nodes  # noqa: E402

hou = graph.hou


def _node(path, type_name="xform"):
    node = MagicMock()
    node.path.return_value = path
    node.name.return_value = path.rsplit("/", 1)[-1]
    node.type.return_value.name.return_value = type_name
    return node


def _subnet(path="/obj/geo1/subnet1", connectors=1):
    subnet = _node(path, "subnet")
    items = [MagicMock(name=f"indirect{i}") for i in range(connectors)]
    subnet.indirectInputs.return_value = items
    return subnet, items


class TestConnectNodes:
    def _install(self, monkeypatch, lookup):
        monkeypatch.setattr(nodes, "_get_node", lambda path: lookup[path])
        monkeypatch.setattr(nodes, "_focus_network_editor", lambda *a, **k: None)

    def test_wires_from_the_subnets_connector(self, monkeypatch):
        subnet, (connector,) = _subnet()
        dest = _node("/obj/geo1/subnet1/matchsize1", "matchsize")
        self._install(monkeypatch, {subnet.path(): subnet, dest.path(): dest})

        result = nodes.connect_nodes(subnet.path(), dest.path(), indirect_input=0)

        dest.setInput.assert_called_once_with(0, connector, 0)
        assert result["indirect_input"] == 0
        assert result["source_path"] == "/obj/geo1/subnet1 (indirect input 0)"

    def test_an_index_out_of_range_is_named(self, monkeypatch):
        subnet, _ = _subnet()
        dest = _node("/obj/geo1/subnet1/matchsize1")
        self._install(monkeypatch, {subnet.path(): subnet, dest.path(): dest})
        with pytest.raises(ValueError, match="1 indirect input"):
            nodes.connect_nodes(subnet.path(), dest.path(), indirect_input=3)

    def test_a_plain_node_as_the_subnet_is_refused_clearly(self, monkeypatch):
        box = _node("/obj/geo1/box1", "box")
        box.indirectInputs.side_effect = AttributeError("no indirectInputs")
        dest = _node("/obj/geo1/xform1")
        self._install(monkeypatch, {box.path(): box, dest.path(): dest})
        with pytest.raises(ValueError, match="no indirect inputs"):
            nodes.connect_nodes(box.path(), dest.path(), indirect_input=0)

    def test_without_the_argument_nothing_changes(self, monkeypatch):
        src = _node("/obj/geo1/box1", "box")
        dest = _node("/obj/geo1/xform1")
        self._install(monkeypatch, {src.path(): src, dest.path(): dest})
        result = nodes.connect_nodes(src.path(), dest.path())
        dest.setInput.assert_called_once_with(0, src, 0)
        assert result["source_path"] == "/obj/geo1/box1"
        assert "indirect_input" not in result

    def test_the_batch_takes_the_same_key(self, monkeypatch):
        subnet, (connector,) = _subnet()
        dest = _node("/obj/geo1/subnet1/matchsize1")
        self._install(monkeypatch, {subnet.path(): subnet, dest.path(): dest})
        result = nodes.connect_nodes_batch(
            [{"source_path": subnet.path(), "dest_path": dest.path(), "indirect_input": 0}]
        )
        assert result["success"] is True
        dest.setInput.assert_called_once_with(0, connector, 0)
        assert result["connected"][0]["indirect_input"] == 0

    def test_a_bad_index_in_the_batch_lands_in_errors_not_an_exception(self, monkeypatch):
        subnet, _ = _subnet()
        dest = _node("/obj/geo1/subnet1/matchsize1")
        self._install(monkeypatch, {subnet.path(): subnet, dest.path(): dest})
        result = nodes.connect_nodes_batch(
            [{"source_path": subnet.path(), "dest_path": dest.path(), "indirect_input": 4}]
        )
        assert result["success"] is False
        assert "1 indirect input" in result["errors"][0]["error"]


class TestBuildNetwork:
    def test_validation_names_the_problem(self):
        subnet, _ = _subnet(connectors=4)
        assert graph._indirect_input_error(subnet, 0) is None
        assert "4 indirect input(s); asked for #9" in graph._indirect_input_error(subnet, 9)
        assert "integer" in graph._indirect_input_error(subnet, "x")
        plain = _node("/obj/geo1/box1", "box")
        plain.indirectInputs.side_effect = AttributeError("no")
        assert "not a subnet" in graph._indirect_input_error(plain, 0)

    def _network(self, monkeypatch, connectors=4):
        parent, items = _subnet("/obj/geo1/subnet1", connectors)
        parent.children.return_value = []
        parent.displayNode.return_value = None
        parent.renderNode.return_value = None
        category = MagicMock()
        category.name.return_value = "Sop"
        parent.childTypeCategory.return_value = category
        node_type = MagicMock()
        node_type.name.return_value = "matchsize"
        node_type.maxNumInputs.return_value = 2
        probe = MagicMock()
        probe.parms.return_value = []
        probe.parmTuples.return_value = []
        parent.createNode.return_value = probe
        monkeypatch.setattr(hou, "node", lambda path: parent if path == parent.path() else None)
        monkeypatch.setattr(graph, "_resolve_node_type", lambda cat, name: node_type)
        monkeypatch.setattr(graph, "_instance_patterns", lambda t: [])
        return parent, items, probe

    def test_a_connector_index_validates_without_a_source(self, monkeypatch):
        self._network(monkeypatch)
        result = graph.build_network(
            "/obj/geo1/subnet1",
            [{"type": "matchsize", "name": "ms", "inputs": [{"indirect_input": 0}]}],
            dry_run=True,
        )
        assert result["valid"] is True, result

    def test_an_index_past_the_connectors_fails_the_dry_run(self, monkeypatch):
        self._network(monkeypatch)
        result = graph.build_network(
            "/obj/geo1/subnet1",
            [{"type": "matchsize", "name": "ms", "inputs": [{"indirect_input": 9}]}],
            dry_run=True,
        )
        assert result["valid"] is False
        assert "4 indirect input(s); asked for #9" in result["errors"][0]
