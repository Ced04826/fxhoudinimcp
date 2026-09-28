"""build_network probes each node type once per session, not once per call.

Probing was 90% of a dry run (750 of 850 ms for eight types). A type whose
strict menu is filled by a script (filecache's take list) is probed afresh.

hou is mocked here; the helper is exercised directly.
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


def _type(name, generator=""):
    parm = MagicMock()
    parm.name.return_value = "mode"
    parm.menuItems.return_value = ("a", "b")
    parm.parmTemplate.return_value.itemGeneratorScript.return_value = generator
    probe = MagicMock()
    probe.parms.return_value = [parm]
    probe.parmTuples.return_value = []
    probe.inputNames.return_value = ()
    probe.outputNames.return_value = ()
    node_type = MagicMock()
    node_type.name.return_value = name
    node_type.category.return_value.name.return_value = "Sop"
    node_type.definition.return_value = None
    scratch = MagicMock()
    scratch.createNode.return_value = probe
    return scratch, node_type


def _probe(scratch, node_type, monkeypatch):
    monkeypatch.setattr(graph, "_is_strict_menu", lambda template: True)
    monkeypatch.setattr(graph, "_connectors_of", lambda node: {"inputs": [], "outputs": []})
    monkeypatch.setattr(graph, "_instance_patterns", lambda node_type: [])
    return graph._type_knowledge(scratch, node_type, node_type.category())


def test_second_probe_comes_from_the_cache(monkeypatch):
    monkeypatch.setattr(graph, "_TYPE_KNOWLEDGE", {})
    scratch, node_type = _type("static_menu_sop")
    first = _probe(scratch, node_type, monkeypatch)
    second = _probe(scratch, node_type, monkeypatch)
    assert scratch.createNode.call_count == 1
    assert second[2]["mode"] == ["a", "b"]
    assert second.shapes == first.shapes


def test_script_filled_menu_is_probed_every_time(monkeypatch):
    monkeypatch.setattr(graph, "_TYPE_KNOWLEDGE", {})
    scratch, node_type = _type("take_menu_sop", generator="echo takes")
    _probe(scratch, node_type, monkeypatch)
    _probe(scratch, node_type, monkeypatch)
    assert scratch.createNode.call_count == 2


def test_long_frame_ranges_keep_spaced_rows_and_every_problem():
    rows = [{"frame": float(f), "cook_ms": 1.0} for f in range(1, 241)]
    rows[101]["errors"] = ["boom"]  # frame 102, off the spaced rows
    shown = graph._thin_frame_rows(rows)
    frames = [row["frame"] for row in shown]
    assert frames[0] == 1.0 and frames[-1] == 240.0
    assert 102.0 in frames
    assert len(shown) == graph._FRAME_ROWS_SHOWN + 1
    short = rows[:10]
    assert graph._thin_frame_rows(short) is short


def test_display_goes_to_the_nearest_upstream_node_that_has_the_flag():
    class Rop:  # a usdrender_rop: no setDisplayFlag
        def __init__(self, source):
            self.source = source

        def inputs(self):
            return (self.source,)

        def path(self):
            return "/obj/lop/render"

    settings = MagicMock()
    settings.path.return_value = "/obj/lop/karma_settings"
    assert graph._upstream_with(Rop(settings), "setDisplayFlag") is settings
    assert graph._upstream_with(settings, "setDisplayFlag", include_self=True) is settings
