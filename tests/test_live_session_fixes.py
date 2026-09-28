"""Fixes from driving the server against a live Houdini 22.0.368.

* A ch() to a node that does not exist cooks clean: Houdini evaluates it to 0
  and reports nothing, so a terrain went flat while verify_network said healthy.
* "minvalue" on attribrandomize got ['values', 'valueb', 'valuea'] as its
  did-you-mean; the parm is `min`, labelled "Min Value".
* Every result was indent=2 JSON with doubles printed to 17 digits.

hou is mocked here; the helpers are exercised directly.
"""

from __future__ import annotations

# Built-in
import json
import os
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

# Internal
import fxhoudinimcp_server.handlers.parameter_handlers as parameters  # noqa: E402

from fxhoudinimcp.server import compact_json  # noqa: E402


def _expression_parm(expression, targets):
    """A parm holding *expression*; *targets* maps a reference token to its parm or None."""
    parm = MagicMock()
    parm.name.return_value = "height"
    parm.path.return_value = "/obj/terrain/hills/height"
    parm.keyframes.return_value = [MagicMock()]
    parm.expression.return_value = expression
    parm.getReferencedParm.return_value = parm
    parm.node.return_value.parm.side_effect = targets.get
    return parm


class TestBrokenReferences:
    def test_a_reference_to_a_missing_node_is_reported(self):
        parm = _expression_parm('ch("../CTRL/hill_height")', {})
        (message,) = parameters.broken_references(parm)
        assert "../CTRL/hill_height" in message

    def test_a_resolving_reference_is_not(self):
        target = MagicMock()
        target.path.return_value = "/obj/terrain/ground/sizex"
        parm = _expression_parm('ch("../ground/sizex") * 0.1', {"../ground/sizex": target})
        assert parameters.broken_references(parm) == []

    def test_a_runtime_path_is_not_guessed_at(self):
        parm = _expression_parm('ch("../$OS/tx")', {})
        assert parameters.broken_references(parm) == []

    def test_a_plain_value_is_skipped_without_parsing(self):
        parm = MagicMock()
        parm.keyframes.return_value = []
        assert parameters.broken_references(parm) == []
        parm.expression.assert_not_called()


class TestSuggestParms:
    LABELS = {
        "min": "Min Value",
        "max": "Max Value",
        "mindiscrete": "Min Value",
        "valuea": "Value A",
        "valueb": "Value B",
        "values": "Number of Values",
        "dimensions": "Dimensions",
    }

    def test_a_name_guessed_from_the_label_finds_the_parm(self):
        assert parameters.suggest_parms("minvalue", self.LABELS)[0] in ("min", "mindiscrete")
        assert "max" in parameters.suggest_parms("maxvalue", self.LABELS)

    def test_a_misspelled_name_still_matches_by_name(self):
        assert parameters.suggest_parms("dimension", self.LABELS)[0] == "dimensions"

    def test_nothing_close_suggests_nothing(self):
        assert parameters.suggest_parms("zzzz", self.LABELS) == []


class TestCompactJson:
    def test_no_whitespace_and_float32_precision(self):
        text = compact_json({"bbox_min": [-10.388985633850098, 0.5], "n": 3, "ok": True})
        assert text == '{"bbox_min":[-10.38899,0.5],"n":3,"ok":true}'

    def test_nested_values_and_non_json_types_survive(self):
        data = json.loads(compact_json({"a": [{"b": (1.0, "x")}], "path": object}))
        assert data["a"] == [{"b": [1.0, "x"]}]
        assert isinstance(data["path"], str)


###### Second live session: simulations


import fxhoudinimcp_server.handlers.cache_handlers as cache  # noqa: E402
import fxhoudinimcp_server.handlers.graph_handlers as graph  # noqa: E402
import fxhoudinimcp_server.ui as ui  # noqa: E402


class TestSimulationEvidence:
    def test_a_range_whose_bounds_never_move_is_static(self):
        row = {"points": 306, "prims": 203, "bbox": [[0, 0, 0], [1, 1, 1]]}
        fallen = {**row, "bbox": [[0, -1, 0], [1, 0, 1]]}
        assert graph._shape(row) == graph._shape(dict(row))
        assert graph._shape(row) != graph._shape(fallen)

    def test_the_sim_io_nodes_count_as_caches(self):
        for type_name in ("rbdio::2.0", "vellumio::2.0", "filecache::2.0", "rop_geometry"):
            node = MagicMock()
            node.type.return_value.name.return_value = type_name
            assert cache._is_cache_node(node), type_name

    def test_an_unknown_viewer_mode_is_refused(self, monkeypatch):
        monkeypatch.setattr(ui, "ui_available", lambda: True)
        try:
            ui.set_other_objects("invisible")
        except ValueError as exc:
            assert "hide" in str(exc)
        else:
            raise AssertionError("expected ValueError")
        assert ui.set_other_objects(None) is None


class TestSolverMessages:
    def test_repeated_lines_are_dropped_and_the_rest_capped(self):
        noise = "\n".join(
            f"   merge_field: Error cooking SOP: FLIP_DATA {i % 3}" for i in range(40)
        )
        text = graph._condensed(noise + "\nRequired attribute pscale is missing.")
        assert text.count("FLIP_DATA 0") == 1
        assert "pscale is missing" in text

    def test_long_distinct_messages_say_how_much_was_cut(self):
        text = graph._condensed("\n".join(f"line {i}" for i in range(20)))
        assert text.endswith("... 8 more distinct lines")

    def test_vectors_report_each_axis(self):
        entry = {"min": -3.7, "max": 3.7, "mean": 0.2, "sum": 9.0, "count": 5}
        entry["per_component"] = [{"min": -3.7, "max": 3.7}, {"min": -0.1, "max": 2.8}]
        row = graph._frame_stats(entry)
        assert row["per_axis"][1] == [-0.1, 2.8]
        assert "count" not in row


class TestWorkflowGuides:
    def test_every_guide_is_served_by_the_tool(self):
        """solaris, tops, assets, model and troubleshooting raised KeyError."""
        from fxhoudinimcp.prompts.workflows import _MD_DIR, workflow_guide_text

        for path in sorted((_MD_DIR / "workflows").glob("*.md")):
            assert "Render the crate" in workflow_guide_text(path.stem, "Render the crate"), (
                path.stem
            )


class TestStartRenderErrors:
    async def test_a_silent_failure_reads_the_errors_husk_posts_later(
        self, mock_ctx, mock_bridge, monkeypatch
    ):
        import fxhoudinimcp.tools.rendering as rendering
        from fxhoudinimcp.tools.rendering import start_render

        monkeypatch.setattr(rendering, "_ERROR_POLL_SECONDS", 0)

        license = "Command Exit Code: 3\nNo licenses could be found to run this application."
        mock_bridge.execute.side_effect = [
            {"success": False, "wrote_files": False, "message": "Render reported no errors, ..."},
            {"errors": []},  # husk's error has not landed on the first look
            {"errors": [license], "license_error": license},
        ]
        result = await start_render(mock_ctx, "/stage/render", frame_range=[1, 1])
        assert result["errors"] == [license]
        assert "no license" in result["message"]

    async def test_a_successful_render_costs_one_call(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.rendering import start_render

        mock_bridge.execute.return_value = {"success": True, "wrote_files": True}
        await start_render(mock_ctx, "/stage/render")
        assert mock_bridge.execute.call_count == 1


###### Third live session: assets


import fxhoudinimcp_server.handlers.hda_handlers as hda  # noqa: E402


class TestCreateHdaKeepsTheInterface:
    def test_spares_move_into_the_definition_with_their_values(self, monkeypatch):
        failed = type("OperationFailed", (Exception,), {})
        monkeypatch.setattr(hda.hou, "OperationFailed", failed)
        parm = MagicMock()
        parm.name.return_value = "seed"
        parm.tuple.return_value.name.return_value = "seed"
        parm.expression.side_effect = failed  # a plain value, no expression
        parm.eval.return_value = 7
        node = MagicMock()
        node.spareParms.return_value = [parm]
        type_parm = node.parm.return_value

        assert hda._promote_spares_to_definition(node) == ["seed"]
        node.removeSpareParms.assert_called_once()
        definition = node.type.return_value.definition.return_value
        definition.setParmTemplateGroup.assert_called_once_with(node.parmTemplateGroup.return_value)
        type_parm.set.assert_called_once_with(7)

    def test_a_subnet_without_spares_is_left_alone(self):
        node = MagicMock()
        node.spareParms.return_value = []
        assert hda._promote_spares_to_definition(node) == []
        node.removeSpareParms.assert_not_called()


def test_expression_on_animated_parm_replaces_every_key():
    """setExpression alone rewrote only the segment at the current frame."""
    parm = MagicMock()
    parm.keyframes.return_value = [MagicMock(), MagicMock(), MagicMock()]
    assert parameters.set_whole_expression(parm, "$F") == 3
    parm.deleteAllKeyframes.assert_called_once()
    parm.setExpression.assert_called_once_with("$F")

    held = MagicMock()
    held.keyframes.return_value = [MagicMock()]  # an expression is one key
    assert parameters.set_whole_expression(held, "$F") == 0
    held.deleteAllKeyframes.assert_not_called()
