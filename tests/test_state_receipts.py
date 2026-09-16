"""Tests for honest receipts: build specs, parameter writes, bounded output.

The bugs these cover are all the same bug wearing different clothes -- a
command reporting that something happened without carrying the evidence, and a
caller unable to tell a complete answer from a clipped one. So the tests are
mostly about the unhappy shapes: the value that was refused, the output that
did not fit, the return that JSON cannot hold.

The parameter and output logic runs here for real. ``hou`` is a stub, as in
tests/test_modeling_tools.py, and the parameter fakes implement the handful of
hou.Parm methods the policy code duck-types, so a refusal tested here is the
same refusal a live parameter gets.
"""

from __future__ import annotations

# Built-in
import json
import os
import sys
from unittest.mock import MagicMock

# Third-party
import pytest

sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

# Internal
import hou  # noqa: E402  (the stub above)
from fxhoudinimcp_server.handlers import code_handlers, parameter_handlers  # noqa: E402
from fxhoudinimcp_server.handlers.state_receipt_helpers import (  # noqa: E402
    check_parm_value,
    clear_channels,
    json_safe,
    parm_state,
    parse_input_entries,
    positive_int,
    raw_differs,
    resolve_expression_policy,
    resolve_input_policy,
    set_parm_value,
    terminal_names,
    text_receipt,
    utf8_bytes,
    value_receipt,
    write_refusal,
)

###### Fakes
#
# Only the methods the policy code actually calls. A real hou.Parm raises from
# expression() when it has none, which is the branch that made every naive
# "does this have an expression" check fail, so the fake raises too.


class FakeLanguage:
    """Anything HOM answers with that has a .name(): a language, a parm type."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class FakeTemplate:
    def __init__(self, type_name: str, menu: tuple = ()) -> None:
        self._type = type_name
        self._menu = menu

    def type(self):
        return FakeLanguage(self._type)

    def menuItems(self):  # noqa: N802 - HOM spelling
        return self._menu


class FakeParm:
    def __init__(
        self,
        name: str,
        value=0.0,
        expression: str | None = None,
        keyframes: int = 0,
        locked: bool = False,
        raw=None,
        accepts=(int, float, str, bool),
        template=None,
    ) -> None:
        self._template = template
        self._name = name
        self._value = value
        self._expression = expression
        self._keyframes = keyframes
        self._locked = locked
        self._raw = raw
        self._accepts = accepts
        self.set_calls: list = []
        self.deleted = 0

    def name(self):
        return self._name

    def path(self):
        return f"/obj/fake/{self._name}"

    def eval(self):
        return self._value

    def rawValue(self):  # noqa: N802 - HOM spelling
        if self._raw is not None:
            return self._raw
        if self._expression is not None:
            return self._expression
        return str(self._value)

    def expression(self):
        if self._expression is None:
            raise RuntimeError("parameter has no expression")
        return self._expression

    def expressionLanguage(self):  # noqa: N802 - HOM spelling
        return FakeLanguage("Hscript")

    def parmTemplate(self):  # noqa: N802 - HOM spelling
        if self._template is None:
            # Most fakes decline to say, which is how a real parameter behaves
            # when the template cannot be read: the checks stay silent.
            raise RuntimeError("no template")
        return self._template

    def keyframes(self):
        return tuple(range(self._keyframes))

    def isLocked(self):  # noqa: N802 - HOM spelling
        return self._locked

    def deleteAllKeyframes(self):  # noqa: N802 - HOM spelling
        self.deleted += 1
        self._expression = None
        self._keyframes = 0
        self._raw = None

    # HOM's signature, default included: a write that does not say otherwise
    # goes through a channel reference to whatever it points at.
    def set(self, value, follow_parm_reference=True):  # noqa: FBT002 - HOM's own default
        if not isinstance(value, self._accepts):
            raise TypeError(f"{self._name} does not accept {type(value).__name__}")
        self.set_calls.append(value)
        self.followed = follow_parm_reference
        if self._expression is None:
            self._value = value
            self._raw = None


class StubbornParm(FakeParm):
    """A parameter that accepts a write and goes on evaluating to something else.

    The recorded case: 0 written to a pivot carrying ``$CEX``, receipt said
    success, parameter kept answering with the expression's value.
    """

    def set(self, value, follow_parm_reference=True):  # noqa: FBT002 - HOM's own default
        self.set_calls.append(value)


class ExpandingParm(FakeParm):
    """A string parameter that stores what it was given and evaluates it expanded.

    ``$HIP/geo/x.bgeo`` is a variable, not an expression, and the two are told
    apart by which of rawValue() and expression() answers.
    """

    def set(self, value, follow_parm_reference=True):  # noqa: FBT002 - HOM's own default
        self.set_calls.append(value)
        self._raw = value
        self._value = value.replace("$HIP", "/hip")


class ReferencingParm(FakeParm):
    """A parameter carrying a channel reference, behaving the way HOM does.

    Writing to it with the default follow_parm_reference=True edits the
    parameter it points at — on a node the caller never named. That is the
    behaviour every write here has to opt out of.
    """

    def __init__(self, name: str, source: FakeParm, **kwargs) -> None:
        super().__init__(name, expression=f'ch("{source.path()}")', **kwargs)
        self.source = source

    def eval(self):
        return self.source.eval()

    def set(self, value, follow_parm_reference=True):  # noqa: FBT002 - HOM's own default
        self.set_calls.append(value)
        self.followed = follow_parm_reference
        if follow_parm_reference and self._expression is not None:
            self.source.set(value)
            return
        self._expression = None
        self._value = value
        self._raw = None


class KeptExpressionParm(FakeParm):
    """Clearing reports success and the expression is still there afterwards.

    A parameter inside a take behaves this way, and the read-back is the only
    thing that would notice.
    """

    def deleteAllKeyframes(self):  # noqa: N802 - HOM spelling
        self.deleted += 1


class UnclearableParm(FakeParm):
    """A parameter whose channel refuses to be deleted.

    Rare, and the reason it matters is what happens next: the reference is
    still live, so a write that carried on would land on its source.
    """

    def deleteAllKeyframes(self):  # noqa: N802 - HOM spelling
        raise RuntimeError("the parameter is in a locked take")


class FakeParmTuple:
    def __init__(self, name: str, parms: list) -> None:
        self._name = name
        self._parms = parms

    def name(self):
        return self._name

    def __len__(self):
        return len(self._parms)

    def __iter__(self):
        return iter(self._parms)

    def eval(self):
        return tuple(parm.eval() for parm in self._parms)


class FakeNode:
    def __init__(self, parms: list, tuples: dict | None = None, path: str = "/obj/geo1/box1"):
        self._parms = {parm.name(): parm for parm in parms}
        self._tuples = tuples or {}
        self._path = path

    def path(self):
        return self._path

    def parm(self, name):
        return self._parms.get(name)

    def parmTuple(self, name):  # noqa: N802 - HOM spelling
        return self._tuples.get(name)

    def parms(self):
        return list(self._parms.values())


@pytest.fixture
def hou_stub(monkeypatch):
    """A stub with the few real types the serialiser does isinstance() against."""
    for name in ("Vector2", "Vector3", "Vector4", "Matrix3", "Matrix4", "Ramp"):
        monkeypatch.setattr(hou, name, type(name, (), {}), raising=False)
    monkeypatch.setattr(hou, "frame", lambda: 12.0, raising=False)
    monkeypatch.setattr(hou, "time", lambda: 0.5, raising=False)
    monkeypatch.setattr(hou, "fps", lambda: 24.0, raising=False)
    return hou


def _node(monkeypatch, node):
    monkeypatch.setattr(hou, "node", lambda path: node, raising=False)
    return node


###### A network fake, for the branches only a callback can reach
#
# build_network's "exact" policy exists because an environment auto-wires nodes
# as they are created. That is the one thing a live probe cannot ask for on
# demand, so it is asked for here: createNode runs a hook, and the hook rewires
# whatever the test wants it to.


class FakeConnection:
    def __init__(self, index: int, source, output: int) -> None:
        self._index, self._source, self._output = index, source, output

    def inputIndex(self):  # noqa: N802 - HOM spelling
        return self._index

    def inputNode(self):  # noqa: N802 - HOM spelling
        return self._source

    def outputIndex(self):  # noqa: N802 - HOM spelling
        return self._output


class FakeNodeType:
    def __init__(self, name: str, max_inputs: int = 4) -> None:
        self._name, self._max = name, max_inputs

    def name(self):
        return self._name

    def maxNumInputs(self):  # noqa: N802 - HOM spelling
        return self._max

    def minNumInputs(self):  # noqa: N802 - HOM spelling
        return 0

    def definition(self):
        return None


class FakeSopParm:
    def __init__(self, name: str, node) -> None:
        self._name, self._node = name, node

    def name(self):
        return self._name

    def parmTemplate(self):  # noqa: N802 - HOM spelling
        return FakeTemplate("Float")

    def eval(self):
        return 0.0

    def set(self, value, follow_parm_reference=True):  # noqa: FBT002 - HOM's own default
        hook = self._node._network.on_parm_set
        if hook is not None:
            hook(self._node)


class FakeSop:
    """Enough hou.Node for build_network, plus switches for HOM refusing."""

    def __init__(self, name: str, node_type: FakeNodeType, network) -> None:
        self._name, self._type, self._network = name, node_type, network
        self.inputs: dict[int, tuple] = {}
        self.refuse_disconnect: set = set()
        self.refuse_set: set = set()
        self.ignore_set: set = set()
        self.error_text: list = []
        self.destroyed = False
        self.cooked = 0

    def name(self):
        return self._name

    def path(self):
        return f"{self._network.path()}/{self._name}"

    def type(self):
        return self._type

    def errors(self):
        return tuple(self.error_text)

    def warnings(self):
        return ()

    def isBypassed(self):  # noqa: N802 - HOM spelling
        return False

    # One parameter, "scale", whose set() can run a callback. This is the
    # realistic shape of the hazard: creation callbacks all fire before
    # build_network wires anything, but a parameter callback fires in the
    # middle of the wiring pass and can move a connection already made.
    def parms(self):
        return (FakeSopParm("scale", self),)

    def parm(self, name):
        return FakeSopParm(name, self) if name == "scale" else None

    def parmTuple(self, name):  # noqa: N802 - HOM spelling
        return None

    def parmTuples(self):  # noqa: N802 - HOM spelling
        return ()

    def inputConnections(self):  # noqa: N802 - HOM spelling
        return [
            FakeConnection(index, source, output)
            for index, (source, output) in sorted(self.inputs.items())
        ]

    def setInput(self, index, source, output=0):  # noqa: N802 - HOM spelling
        if source is None:
            if index in self.refuse_disconnect:
                raise RuntimeError(f"input {index} cannot be disconnected")
            self.inputs.pop(index, None)
            return
        if index in self.refuse_set:
            raise RuntimeError(f"input {index} cannot be set")
        if index in self.ignore_set:
            # Accepts the call and keeps the old wiring: "it did not raise" is
            # not evidence that it worked.
            return
        self.inputs[index] = (source, output)

    def cook(self, force=False):  # noqa: FBT002 - HOM's own default
        self.cooked += 1

    def destroy(self):
        self.destroyed = True
        self._network.remove(self)


class FakeNetwork:
    """A /obj/geo1 whose createNode can run a callback, like a real one's does."""

    def __init__(self, path: str = "/obj/geo1") -> None:
        self._path = path
        self._children: list = []
        self.types = {name: FakeNodeType(name) for name in ("box", "merge", "xform", "null")}
        self.on_create = None
        self.on_parm_set = None
        self.counter = 0

    def path(self):
        return self._path

    def childTypeCategory(self):  # noqa: N802 - HOM spelling
        return FakeLanguage("Sop")

    def children(self):
        return list(self._children)

    def node(self, name):
        for child in self._children:
            if child.name() == name or child.path() == name:
                return child
        return None

    def createNode(self, type_name, name=None):  # noqa: N802 - HOM spelling
        self.counter += 1
        node = FakeSop(name or f"{type_name}{self.counter}", self.types[type_name], self)
        self._children.append(node)
        if self.on_create is not None:
            self.on_create(node)
        return node

    def remove(self, node):
        if node in self._children:
            self._children.remove(node)

    def displayNode(self):  # noqa: N802 - HOM spelling
        return self._children[-1] if self._children else None

    def renderNode(self):  # noqa: N802 - HOM spelling
        return None


@pytest.fixture
def network(monkeypatch):
    """A fake /obj/geo1 wired into the hou stub build_network reads."""
    from fxhoudinimcp_server.handlers import graph_handlers

    net = FakeNetwork()
    graph_handlers._TYPE_KNOWLEDGE.clear()

    def _lookup(path):
        if path == net.path():
            return net
        return net.node(path)

    monkeypatch.setattr(hou, "node", _lookup, raising=False)
    monkeypatch.setattr(hou, "selectedNodes", lambda: (), raising=False)
    # hou.preferredNodeType on a MagicMock answers with another MagicMock,
    # which would be accepted as a node type.
    monkeypatch.setattr(
        hou, "preferredNodeType", lambda name: (_ for _ in ()).throw(RuntimeError), raising=False
    )
    monkeypatch.setattr(
        hou, "OperationFailed", type("OperationFailed", (Exception,), {}), raising=False
    )
    category = FakeLanguage("Sop")
    category.nodeTypes = lambda: net.types
    monkeypatch.setattr(net, "childTypeCategory", lambda: category, raising=False)
    return net


###### Text and value budgets


class TestTextAndValueBudgets:
    def test_short_text_is_reported_whole(self):
        receipt = text_receipt("hello", 100)
        assert receipt == {"truncated": False, "chars": 5, "bytes": 5, "text": "hello"}

    def test_long_text_reports_what_is_missing(self):
        receipt = text_receipt("x" * 250, 100)
        assert receipt["truncated"] is True
        assert receipt["chars"] == 250
        assert len(receipt["text"]) == 100
        assert receipt["omitted_chars"] == 150
        assert receipt["sha1"]

    def test_characters_and_bytes_are_different_numbers(self):
        """A receipt that reports only characters is wrong on every non-ASCII run."""
        chinese = "三轮车模型"
        assert utf8_bytes(chinese) == 15
        assert text_receipt(chinese, 100)["chars"] == 5

        emoji = "🚲"
        assert utf8_bytes(emoji) == 4
        assert len(emoji) == 1

    def test_small_value_comes_back_as_itself(self):
        receipt = value_receipt([1.0, 2.0, 3.0], 200)
        assert receipt["complete"] is True
        assert receipt["value"] == [1.0, 2.0, 3.0]

    def test_long_string_becomes_length_hash_and_ends(self):
        receipt = value_receipt("a" * 5000, 200)
        assert receipt["complete"] is False
        assert receipt["kind"] == "string"
        assert receipt["chars"] > 5000
        assert receipt["sha1"]
        assert receipt["head"].startswith("a")
        assert receipt["tail"].endswith("a")
        assert "value" not in receipt

    def test_long_list_is_sampled_at_both_ends(self):
        receipt = value_receipt(list(range(1000)), 200)
        assert receipt["complete"] is False
        assert receipt["count"] == 1000
        assert receipt["sample"][0] == 0
        assert receipt["sample"][-1] == 999

    def test_full_overrides_the_cap(self):
        receipt = value_receipt("a" * 5000, 200, full=True)
        assert receipt["complete"] is True
        assert len(receipt["value"]) == 5000


###### JSON safety


class TestJsonSafe:
    def test_nested_structures_survive_unchanged(self):
        value = {"a": [1, {"b": (2, 3)}], "c": "x"}
        safe, notes = json_safe(value)
        assert safe == {"a": [1, {"b": [2, 3]}], "c": "x"}
        assert notes == {}

    def test_nonfinite_floats_are_named_not_emitted(self):
        safe, notes = json_safe({"nan": float("nan"), "inf": [float("inf")]})
        assert safe["nan"] == "NaN"
        assert safe["inf"] == ["Infinity"]
        assert notes["nonfinite"] == ["$.nan", "$.inf[0]"]
        # The point of the exercise: the result is parseable JSON.
        json.loads(json.dumps(safe, allow_nan=False))

    def test_unserialisable_objects_are_recorded_as_coerced(self):
        safe, notes = json_safe({"node": object()})
        assert isinstance(safe["node"], str)
        assert notes["coerced"] == ["$.node"]

    def test_cycles_do_not_hang(self):
        loop: dict = {}
        loop["self"] = loop
        safe, notes = json_safe(loop)
        assert safe["self"] == "<circular reference>"
        assert notes["circular"]

    def test_bytes_and_sets_are_converted_and_noted(self):
        safe, notes = json_safe({"b": b"hi", "s": {1}})
        assert safe["b"] == "hi"
        assert safe["s"] == [1]
        assert len(notes["coerced"]) == 2


###### build_network input specs


class TestInputSpecs:
    def test_positional_strings_wire_in_order(self):
        parsed, errors = parse_input_entries(["a", "b"], 4, "copy")
        assert errors == []
        assert parsed == [
            {"index": 0, "source": "a", "source_output": 0},
            {"index": 1, "source": "b", "source_output": 0},
        ]

    def test_null_holds_a_position_without_connecting(self):
        parsed, errors = parse_input_entries([None, "b"], 4, "merge")
        assert errors == []
        assert parsed == [{"index": 1, "source": "b", "source_output": 0}]

    def test_dicts_may_be_sparse_and_pick_an_output(self):
        parsed, errors = parse_input_entries(
            [{"index": 3, "source": "a", "source_output": 2}], 4, "n"
        )
        assert errors == []
        assert parsed == [{"index": 3, "source": "a", "source_output": 2}]

    def test_two_entries_on_one_index_is_an_error(self):
        parsed, errors = parse_input_entries(
            [{"index": 0, "source": "a"}, {"index": 0, "source": "b"}], 4, "n"
        )
        assert parsed == [{"index": 0, "source": "a", "source_output": 0}]
        assert "specified twice" in errors[0]

    def test_index_past_the_type_is_an_error(self):
        _, errors = parse_input_entries([{"index": 5, "source": "a"}], 2, "xform")
        assert "exceeds max inputs" in errors[0]

    def test_negative_and_non_integer_indices_are_errors(self):
        _, negative = parse_input_entries([{"index": -1, "source": "a"}], 4, "n")
        assert "negative" in negative[0]
        _, fractional = parse_input_entries([{"index": 1.5, "source": "a"}], 4, "n")
        assert "non-integer" in fractional[0]

    def test_negative_source_output_is_an_error(self):
        _, errors = parse_input_entries([{"source": "a", "source_output": -2}], 4, "n")
        assert "source_output" in errors[0]

    def test_wrong_shapes_are_named_rather_than_crashed_on(self):
        _, not_a_list = parse_input_entries({"source": "a"}, 4, "n")
        assert "must be a list" in not_a_list[0]
        _, wrong_entry = parse_input_entries([17], 4, "n")
        assert "must be a node name" in wrong_entry[0]
        _, empty = parse_input_entries(["  "], 4, "n")
        assert "empty source" in empty[0]

    def test_a_generator_takes_no_inputs(self):
        _, errors = parse_input_entries(["a"], 0, "box")
        assert errors == []  # max_inputs 0 means "unknown", not "forbidden"

    def test_omitted_inputs_parse_to_nothing(self):
        assert parse_input_entries(None, 4, "n") == ([], [])


class TestParmValueChecks:
    FLOAT3 = {"type": "Float", "components": 3, "is_tuple": True}
    FLOAT1 = {"type": "Float", "components": 1, "is_tuple": False}
    STRING = {"type": "String", "components": 1, "is_tuple": False}

    def test_correct_shapes_pass(self):
        assert check_parm_value("b", "box", "size", [1, 2, 3], self.FLOAT3) == []
        assert check_parm_value("b", "box", "scale", 0.5, self.FLOAT1) == []
        assert check_parm_value("b", "box", "size", 0.5, self.FLOAT3) == []  # broadcast
        assert check_parm_value("f", "file", "file", "$HIP/x.bgeo", self.STRING) == []

    def test_component_count_mismatch_is_caught_before_the_build(self):
        errors = check_parm_value("b", "box", "size", [1, 2], self.FLOAT3)
        assert "3 components, got 2" in errors[0]

    def test_a_list_on_a_scalar_parameter_is_caught(self):
        errors = check_parm_value("s", "scatter", "npts", [1, 2], self.FLOAT1)
        assert "single value" in errors[0]

    def test_nested_and_null_components_are_caught(self):
        nested = check_parm_value("b", "box", "size", [[1], 2, 3], self.FLOAT3)
        assert "components are scalars" in nested[0]
        null = check_parm_value("b", "box", "size", [None, 2, 3], self.FLOAT3)
        assert "null" in null[0]

    def test_objects_and_nulls_are_caught(self):
        assert "not an object" in check_parm_value("b", "box", "size", {"x": 1}, self.FLOAT3)[0]
        assert "null" in check_parm_value("b", "box", "scale", None, self.FLOAT1)[0]

    def test_unknown_parameters_are_left_to_the_name_check(self):
        assert check_parm_value("b", "box", "whatever", [1, 2], None) == []


class TestTerminals:
    def test_a_chain_ends_at_its_last_node(self):
        assert terminal_names(["a", "b", "c"], [("b", "a"), ("c", "b")]) == ["c"]

    def test_two_branches_give_two_terminals(self):
        assert terminal_names(["a", "b", "c"], [("b", "a"), ("c", "a")]) == ["b", "c"]

    def test_sources_outside_the_build_do_not_count(self):
        assert terminal_names(["a"], [("a", "/obj/geo1/existing")]) == ["a"]

    def test_everything_consumed_falls_back_to_the_last(self):
        assert terminal_names(["a", "b"], [("a", "b"), ("b", "a")]) == ["b"]
        assert terminal_names([], []) == []


class TestPolicyTokens:
    def test_defaults_and_valid_tokens(self):
        assert resolve_input_policy(None) == "preserve"
        assert resolve_input_policy("EXACT") == "exact"
        assert resolve_expression_policy(None) == "preserve"
        assert resolve_expression_policy("Replace") == "replace"

    def test_unknown_tokens_name_the_alternatives(self):
        with pytest.raises(ValueError, match="exact"):
            resolve_input_policy("strict")
        with pytest.raises(ValueError, match="replace"):
            resolve_expression_policy("keep")
        with pytest.raises(ValueError):
            resolve_expression_policy(1)

    def test_positive_int_rejects_the_usual_disguises(self):
        assert positive_int(5, "n") == 5
        with pytest.raises(ValueError):
            positive_int(True, "n")
        with pytest.raises(ValueError):
            positive_int(0, "n")
        with pytest.raises(ValueError):
            positive_int(1.5, "n")
        with pytest.raises(ValueError, match="exceed"):
            positive_int(10, "n", maximum=9)


###### Parameter state and refusals


class TestParmStateAndRefusals:
    def test_state_of_an_expression_driven_parameter(self):
        state = parm_state(FakeParm("px", value=5.0, expression="$CEX"))
        assert state["expression"] == "$CEX"
        assert state["expression_language"] == "Hscript"
        assert state["value"] == 5.0

    def test_state_of_a_plain_parameter_has_no_expression(self):
        state = parm_state(FakeParm("sizex", value=1.0))
        assert state["expression"] is None
        assert state["keyframes"] == 0

    def test_preserve_refuses_and_names_the_expression(self):
        state = parm_state(FakeParm("px", value=5.0, expression="$CEX"))
        refusal = write_refusal(state, "preserve", "/obj/geo1/xform1/px")
        assert "$CEX" in refusal
        assert "replace" in refusal

    def test_preserve_refuses_keyframes_too(self):
        state = parm_state(FakeParm("ty", keyframes=3))
        assert "keyframe" in write_refusal(state, "preserve", "ty")

    def test_replace_allows_the_write(self):
        state = parm_state(FakeParm("px", expression="$CEX"))
        assert write_refusal(state, "replace", "px") is None

    def test_a_locked_parameter_is_refused_under_both_policies(self):
        state = parm_state(FakeParm("sizex", locked=True))
        assert "locked" in write_refusal(state, "preserve", "sizex")
        assert "locked" in write_refusal(state, "replace", "sizex")

    def test_clearing_reports_what_it_removed(self):
        parm = FakeParm("px", expression="$CEX", keyframes=2)
        outcome = clear_channels(parm)
        assert parm.deleted == 1
        assert outcome["cleared_expression"] is True
        assert outcome["previous_expression"] == "$CEX"
        assert outcome["cleared_keyframes"] == 2

    def test_a_number_spelled_differently_is_not_a_difference(self):
        """A float parm holding 1.0 has a raw value of "1"; that is not news."""
        assert raw_differs("1", 1.0) is False
        assert raw_differs("1.0", 1.0) is False
        assert raw_differs("$CEX", 0.0) is True
        assert raw_differs("$HIP/x.bgeo", "/hip/x.bgeo") is True
        assert raw_differs("2", 1.0) is True
        assert raw_differs(None, 1.0) is False

    def test_clearing_a_plain_parameter_touches_nothing(self):
        parm = FakeParm("sizex")
        assert clear_channels(parm) == {"cleared_expression": False, "cleared_keyframes": 0}
        assert parm.deleted == 0


###### Parameter writes end to end


def _box_node():
    sizes = [FakeParm("sizex", 1.0), FakeParm("sizey", 1.0), FakeParm("sizez", 1.0)]
    return FakeNode(
        [*sizes, FakeParm("scale", 1.0)],
        {"size": FakeParmTuple("size", sizes)},
    )


class TestParameterWrites:
    def test_a_plain_write_reports_the_read_back_value(self, hou_stub, monkeypatch):
        node = _node(monkeypatch, _box_node())
        result = parameter_handlers._set_parameter("/obj/geo1/box1", "scale", 2.5)
        assert result["success"] is True
        assert result["new_value"] == 2.5
        assert result["set"][0]["matches_requested"] is True
        assert result["context"] == {"frame": 12.0, "time": 0.5, "fps": 24.0}
        assert node.parm("scale").eval() == 2.5

    def test_a_list_sets_the_whole_tuple(self, hou_stub, monkeypatch):
        node = _node(monkeypatch, _box_node())
        result = parameter_handlers._set_parameter("/obj/geo1/box1", "size", [1.0, 2.0, 3.0])
        assert result["new_value"] == [1.0, 2.0, 3.0]
        assert [p.eval() for p in node.parmTuple("size")] == [1.0, 2.0, 3.0]

    def test_a_scalar_broadcasts_across_a_tuple(self, hou_stub, monkeypatch):
        node = _node(monkeypatch, _box_node())
        parameter_handlers._set_parameter("/obj/geo1/box1", "size", 4.0)
        assert [p.eval() for p in node.parmTuple("size")] == [4.0, 4.0, 4.0]

    def test_the_wrong_component_count_changes_nothing(self, hou_stub, monkeypatch):
        node = _node(monkeypatch, _box_node())
        with pytest.raises(ValueError, match="components"):
            parameter_handlers._set_parameter("/obj/geo1/box1", "size", [1.0, 2.0])
        assert [p.set_calls for p in node.parmTuple("size")] == [[], [], []]

    def test_an_unknown_name_suggests_the_real_one(self, hou_stub, monkeypatch):
        _node(monkeypatch, _box_node())
        with pytest.raises(ValueError, match="sizex"):
            parameter_handlers._set_parameter("/obj/geo1/box1", "sizx", 1.0)

    def test_preserve_refuses_an_expression_and_leaves_it_alone(self, hou_stub, monkeypatch):
        parm = FakeParm("px", value=5.0, expression="$CEX")
        node = _node(monkeypatch, FakeNode([parm]))
        with pytest.raises(ValueError, match=r"\$CEX"):
            parameter_handlers._set_parameter("/obj/geo1/xform1", "px", 0)
        assert parm.expression() == "$CEX"
        assert parm.set_calls == []
        assert node.parm("px").eval() == 5.0

    def test_replace_clears_the_expression_then_writes(self, hou_stub, monkeypatch):
        parm = FakeParm("px", value=5.0, expression="$CEX")
        _node(monkeypatch, FakeNode([parm]))
        result = parameter_handlers._set_parameter(
            "/obj/geo1/xform1", "px", 0.0, expression_policy="replace"
        )
        assert result["success"] is True
        assert result["set"][0]["expression_cleared"] is True
        assert result["set"][0]["cleared"][0]["previous_expression"] == "$CEX"
        assert result["new_value"] == 0.0
        assert parm.deleted == 1

    def test_a_write_that_does_not_take_is_reported_as_not_matching(
        self, hou_stub, monkeypatch
    ):
        """The $CEX case as it actually behaved: accepted, and still wrong."""
        parm = StubbornParm("px", value=5.0)
        _node(monkeypatch, FakeNode([parm]))
        result = parameter_handlers._set_parameter("/obj/geo1/xform1", "px", 0.0)
        assert parm.set_calls == [0.0]
        assert result["set"][0]["matches_requested"] is False
        assert result["set"][0]["requested"] == 0.0
        assert result["new_value"] == 5.0

    def test_a_float_written_plainly_carries_no_raw_value_noise(self, hou_stub, monkeypatch):
        """Houdini's raw text for 2.0 is "2"; the receipt must not call that news."""
        parm = FakeParm("scale", 1.0, raw="2")
        _node(monkeypatch, FakeNode([parm]))
        entry = parameter_handlers._set_parameter("/obj/geo1/box1", "scale", 2.0)["set"][0]
        assert "raw_value" not in entry
        assert "value_is_expanded" not in entry

    def test_a_variable_in_a_string_is_not_called_an_expression(self, hou_stub, monkeypatch):
        parm = ExpandingParm("file", value="", accepts=(str,))
        _node(monkeypatch, FakeNode([parm]))
        result = parameter_handlers._set_parameter("/obj/geo1/file1", "file", "$HIP/geo/x.bgeo")
        entry = result["set"][0]
        assert entry["new_value"] == "/hip/geo/x.bgeo"
        assert entry["raw_value"] == "$HIP/geo/x.bgeo"
        assert entry["value_is_expanded"] is True
        assert "expression" not in entry
        # The value read back is not the string that was written, and the
        # receipt says so rather than leaving a caller to compare them itself.
        assert entry["matches_requested"] is False

    def test_an_hscript_expression_is_reported_as_one(self, hou_stub, monkeypatch):
        parm = FakeParm("tx", value=10.0, expression="$F * 2")
        _node(monkeypatch, FakeNode([parm]))
        result = parameter_handlers._set_parameters(
            "/obj/geo1/xform1", {"tx": 3.0}, expression_policy="replace"
        )
        assert result["success"] is True
        assert result["set"][0]["cleared"][0]["previous_expression"] == "$F * 2"

    def test_a_locked_parameter_is_refused_not_unlocked(self, hou_stub, monkeypatch):
        parm = FakeParm("sizex", locked=True)
        _node(monkeypatch, FakeNode([parm]))
        result = parameter_handlers._set_parameters(
            "/obj/geo1/box1", {"sizex": 2.0}, expression_policy="replace"
        )
        assert result["success"] is False
        assert "locked" in result["errors"][0]["error"]
        assert parm.set_calls == []

    def test_a_batch_reports_each_failure_and_still_applies_the_rest(
        self, hou_stub, monkeypatch
    ):
        node = _node(monkeypatch, _box_node())
        result = parameter_handlers._set_parameters(
            "/obj/geo1/box1", {"sizex": 2.0, "nope": 1.0, "sizey": 3.0}
        )
        assert result["success"] is False
        assert len(result["errors"]) == 1
        assert {entry["parm_name"] for entry in result["set"]} == {"sizex", "sizey"}
        assert node.parm("sizex").eval() == 2.0

    def test_a_bad_policy_is_refused_before_anything_is_written(self, hou_stub, monkeypatch):
        node = _node(monkeypatch, _box_node())
        with pytest.raises(ValueError, match="expression_policy"):
            parameter_handlers._set_parameters(
                "/obj/geo1/box1", {"sizex": 2.0}, expression_policy="clobber"
            )
        with pytest.raises(ValueError, match="return_values"):
            parameter_handlers._set_parameters(
                "/obj/geo1/box1", {"sizex": 2.0}, return_values="everything"
            )
        assert node.parm("sizex").set_calls == []

    def test_a_component_that_refuses_the_value_names_itself(self, hou_stub, monkeypatch):
        parm = FakeParm("file", value="x", accepts=(str,))
        _node(monkeypatch, FakeNode([parm]))
        result = parameter_handlers._set_parameters("/obj/geo1/file1", {"file": 5})
        assert result["success"] is False
        assert "file" in result["errors"][0]["error"]

    def test_a_long_value_is_summarised_rather_than_clipped(self, hou_stub, monkeypatch):
        parm = FakeParm("snippet", value="", accepts=(str,))
        _node(monkeypatch, FakeNode([parm]))
        long_code = "// vex\n" + "f@x = 1;\n" * 400
        result = parameter_handlers._set_parameter("/obj/geo1/wrangle1", "snippet", long_code)
        entry = result["set"][0]
        assert "new_value" not in entry
        assert entry["new_value_summary"]["complete"] is False
        assert entry["new_value_summary"]["chars"] > 3000
        assert entry["new_value_summary"]["sha1"]

    def test_full_asks_for_the_whole_value(self, hou_stub, monkeypatch):
        parm = FakeParm("snippet", value="", accepts=(str,))
        _node(monkeypatch, FakeNode([parm]))
        long_code = "f@x = 1;\n" * 400
        result = parameter_handlers._set_parameter(
            "/obj/geo1/wrangle1", "snippet", long_code, return_values="full"
        )
        assert result["set"][0]["new_value"] == long_code

    def test_none_returns_names_only(self, hou_stub, monkeypatch):
        _node(monkeypatch, _box_node())
        result = parameter_handlers._set_parameters(
            "/obj/geo1/box1", {"sizex": 2.0}, return_values="none"
        )
        assert result["set"] == [{"parm_name": "sizex"}]

    def test_an_empty_batch_is_refused(self, hou_stub, monkeypatch):
        _node(monkeypatch, _box_node())
        with pytest.raises(ValueError, match="non-empty"):
            parameter_handlers._set_parameters("/obj/geo1/box1", {})


###### execute_python: bounded output, honest metadata


class TestExecutePython:
    def test_a_small_run_is_complete_and_says_so(self):
        result = code_handlers._execute_python("print('hi')")
        assert result["success"] is True
        assert result["executed"] is True
        assert result["execution_success"] is True
        assert result["output_complete"] is True
        assert result["stdout"] == "hi\n"
        assert result["stdout_chars"] == 3
        assert result["stdout_bytes"] == 3
        assert result["stdout_truncated"] is False

    def test_output_past_the_cap_reports_how_much_is_missing(self):
        result = code_handlers._execute_python("print('x' * 200000)")
        assert result["execution_success"] is True
        assert result["output_complete"] is False
        assert result["stdout_truncated"] is True
        assert result["stdout_chars"] == 200001
        assert len(result["stdout"]) == 102400
        assert result["stdout_omitted_chars"] == 200001 - 102400
        assert result["stdout_sha1"]

    def test_chinese_and_emoji_are_counted_in_both_units(self):
        result = code_handlers._execute_python("print('三轮车🚲', end='')")
        # Four characters, thirteen bytes: three-byte CJK and a four-byte emoji.
        assert result["stdout_chars"] == 4
        assert result["stdout_bytes"] == 3 * 3 + 4

    def test_a_nested_json_return_comes_back_whole(self):
        result = code_handlers._execute_python(
            "data = {'a': [1, 2, {'b': 'c'}], 'n': 3}",
            return_expression="data",
        )
        assert result["return_value"] == {"a": [1, 2, {"b": "c"}], "n": 3}
        assert result["return_truncated"] is False
        assert result["output_complete"] is True

    def test_a_large_return_is_summarised_not_clipped(self):
        result = code_handlers._execute_python(
            "rows = list(range(20000))", return_expression="rows"
        )
        assert "return_value" not in result
        assert result["return_truncated"] is True
        assert result["output_complete"] is False
        summary = result["return_value_summary"]
        assert summary["count"] == 20000
        assert summary["sample"][-1] == 19999

    def test_an_exception_keeps_the_output_printed_before_it(self):
        result = code_handlers._execute_python("print('partial'); raise RuntimeError('boom')")
        assert result["execution_success"] is False
        assert result["executed"] is False
        assert result["success"] is False
        assert "boom" in result["error"]
        assert result["stdout"] == "partial\n"

    def test_a_failing_expression_does_not_pretend_the_code_failed(self):
        result = code_handlers._execute_python("x = 1", return_expression="x.nope")
        assert result["execution_success"] is True
        assert result["executed"] is True
        assert result["success"] is False
        assert "AttributeError" in result["eval_error"]
        assert result["return_value"] is None

    def test_the_expressions_own_output_is_captured(self):
        result = code_handlers._execute_python(
            "def f():\n    print('from eval')\n    return 7",
            return_expression="f()",
        )
        assert result["return_value"] == 7
        assert result["stdout"] == "from eval\n"

    def test_nonfinite_results_are_named_and_stay_parseable(self):
        result = code_handlers._execute_python("", return_expression="float('nan')")
        assert result["return_value"] == "NaN"
        assert result["return_nonfinite"] == ["$"]
        json.dumps(result["return_value"], allow_nan=False)

    def test_auto_stringifies_what_json_cannot_hold_and_admits_it(self):
        result = code_handlers._execute_python(
            "class Thing:\n    def __repr__(self):\n        return '<thing>'\n"
            "value = {'t': Thing()}",
            return_expression="value",
        )
        assert result["return_value"]["t"] == "<thing>"
        assert result["return_serializable"] is False
        assert result["return_coerced"] == ["$.t"]
        assert result["success"] is True

    def test_json_format_refuses_instead_of_stringifying(self):
        result = code_handlers._execute_python(
            "value = {'t': object()}", return_expression="value", return_format="json"
        )
        assert result["success"] is False
        assert result["execution_success"] is True
        assert "not JSON" in result["serialization_error"]
        assert result["return_value"] is None

    def test_repr_and_none_formats(self):
        as_repr = code_handlers._execute_python("", return_expression="[1, 2]", return_format="repr")
        assert as_repr["return_value"] == "[1, 2]"
        assert as_repr["return_serialization"] == "repr"

        as_none = code_handlers._execute_python("", return_expression="[1, 2]", return_format="none")
        assert as_none["return_value"] is None
        assert as_none["return_type"] == "list"
        assert as_none["return_omitted"] is True

    def test_an_invalid_limit_is_refused_before_the_code_runs(self, tmp_path):
        marker = tmp_path / "ran.txt"
        code = f"open({str(marker)!r}, 'w').write('ran')"
        for kwargs in (
            {"max_stdout_chars": 0},
            {"max_stdout_chars": "lots"},
            {"max_return_chars": -5},
            {"return_format": "yaml"},
            {"dump_path": "   "},
        ):
            with pytest.raises(ValueError):
                code_handlers._execute_python(code, **kwargs)
        assert not marker.exists(), "the code ran before its arguments were checked"

    def test_the_dump_holds_everything_the_receipt_could_not(self, tmp_path):
        dump = tmp_path / "nested" / "run.json"
        result = code_handlers._execute_python(
            "print('x' * 200000)\nrows = list(range(20000))",
            return_expression="rows",
            dump_path=str(dump),
        )
        assert result["dump"]["written"] is True
        assert result["dump"]["bytes"] > 0
        assert result["full_output_in_dump"] is True
        payload = json.loads(dump.read_text(encoding="utf-8"))
        assert len(payload["stdout"]) == 200001
        assert payload["stdout_chars"] == 200001
        assert payload["return_value"] == list(range(20000))
        assert payload["error"] is None
        assert payload["code_sha1"]
        # The code itself is never echoed, to the receipt or to the dump.
        assert "code" not in payload
        assert "print" not in json.dumps(payload["code_sha1"])

    def test_the_dump_records_both_kinds_of_failure(self, tmp_path):
        dump = tmp_path / "failed.json"
        result = code_handlers._execute_python(
            "print('before')\nraise ValueError('nope')", dump_path=str(dump)
        )
        payload = json.loads(dump.read_text(encoding="utf-8"))
        assert payload["execution_success"] is False
        assert "nope" in payload["error"]
        assert payload["stdout"] == "before\n"
        assert result["execution_success"] is False

    def test_a_dump_that_cannot_be_written_is_not_an_execution_failure(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        result = code_handlers._execute_python(
            "print('done')", dump_path=str(blocker / "run.json")
        )
        assert result["execution_success"] is True
        assert result["success"] is True
        assert result["dump_failed"] is True
        assert result["dump"]["written"] is False
        assert result["dump"]["error"]

    def test_a_dump_of_a_normal_run_is_still_written(self, tmp_path):
        dump = tmp_path / "small.json"
        result = code_handlers._execute_python("print('ok')", dump_path=str(dump))
        assert result["output_complete"] is True
        assert result["dump_streams_complete"] is True
        assert result["full_output_in_dump"] is True
        assert "dump_return_lossless" not in result, "nothing was returned to be lossless"
        assert json.loads(dump.read_text(encoding="utf-8"))["stdout"] == "ok\n"

    def test_nothing_is_retried_after_the_code_has_run(self, tmp_path):
        """A side effect must happen exactly once, however the receipt turns out."""
        counter = tmp_path / "count.txt"
        code = (
            f"p = {str(counter)!r}\n"
            "import os\n"
            "n = int(open(p).read()) if os.path.exists(p) else 0\n"
            "open(p, 'w').write(str(n + 1))\n"
        )
        code_handlers._execute_python(code, return_expression="nope_undefined")
        assert counter.read_text(encoding="utf-8") == "1"


###### The MCP wrappers, which must forward what they advertise


class TestToolWrappers:
    @pytest.mark.asyncio
    async def test_build_network_forwards_policy_and_targets(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.graph import build_network

        await build_network(
            mock_ctx,
            parent_path="/obj/geo1",
            nodes=[{"type": "box", "name": "b"}],
            input_policy="exact",
            inspect_nodes=["b"],
        )
        mock_bridge.execute.assert_called_once_with(
            "graph.build_network",
            {
                "parent_path": "/obj/geo1",
                "nodes": [{"type": "box", "name": "b"}],
                "dry_run": False,
                "layout": True,
                "input_policy": "exact",
                "inspect_nodes": ["b"],
            },
        )

    @pytest.mark.asyncio
    async def test_build_network_omits_unset_inspect_nodes(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.graph import build_network

        await build_network(mock_ctx, parent_path="/obj/geo1", nodes=[{"type": "box"}])
        payload = mock_bridge.execute.call_args[0][1]
        # Sending null would override the Houdini-side default of "the nodes
        # this build produced" with nothing.
        assert "inspect_nodes" not in payload
        assert payload["input_policy"] == "preserve"

    @pytest.mark.asyncio
    async def test_parameter_writes_forward_both_policies(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.parameters import set_parameter, set_parameters

        await set_parameter(
            mock_ctx,
            node_path="/obj/geo1/xform1",
            parm_name="px",
            value=0.0,
            expression_policy="replace",
        )
        assert mock_bridge.execute.call_args[0][1] == {
            "node_path": "/obj/geo1/xform1",
            "parm_name": "px",
            "value": 0.0,
            "expression_policy": "replace",
            "return_values": "summary",
        }

        await set_parameters(
            mock_ctx,
            node_path="/obj/geo1/box1",
            params={"sizex": 2.0},
            return_values="full",
        )
        assert mock_bridge.execute.call_args[0][1] == {
            "node_path": "/obj/geo1/box1",
            "params": {"sizex": 2.0},
            "expression_policy": "preserve",
            "return_values": "full",
        }

    @pytest.mark.asyncio
    async def test_execute_python_forwards_the_dump_and_the_caps(self, mock_ctx, mock_bridge):
        from fxhoudinimcp.tools.code import execute_python

        await execute_python(
            mock_ctx,
            code="print(1)",
            justification="no dedicated tool covers this",
            dump_path="D:/tmp/run.json",
            max_stdout_chars=2000,
            return_format="json",
        )
        assert mock_bridge.execute.call_args[0][1] == {
            "code": "print(1)",
            "max_stdout_chars": 2000,
            "max_return_chars": 8192,
            "return_format": "json",
            "dump_path": "D:/tmp/run.json",
        }


###### Review finding: a summary must not grow with what it summarises


def _size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


class TestSummariesStayBounded:
    """The reported failure: five 100 000-character strings summarised to 400 KB.

    A summary whose size follows its input is not a summary, and this one went
    into the receipt that exists precisely to keep large values out of it.
    """

    def test_a_list_of_huge_strings_summarises_small(self):
        receipt = value_receipt(["X" * 100000] * 5, 512)
        assert receipt["complete"] is False
        assert receipt["count"] == 5
        assert receipt["chars"] > 500000, "the real size must still be reported"
        assert _size(receipt) < 2000, _size(receipt)
        # Each sample says how much of itself is missing rather than carrying it.
        assert "[100000 chars]" in receipt["sample"][0]

    def test_the_cap_governs_the_summary_size(self):
        small = value_receipt(["X" * 100000] * 5, 200)
        large = value_receipt(["X" * 100000] * 5, 4000)
        assert _size(small) < _size(large)
        assert _size(small) < 1200, _size(small)

    def test_nested_lists_and_dicts_are_bounded_at_every_level(self):
        value = [{"rows": [["Y" * 50000] * 20] * 20, "n": 1}] * 8
        receipt = value_receipt(value, 512)
        assert receipt["complete"] is False
        assert receipt["count"] == 8
        assert _size(receipt) < 2000, _size(receipt)

    def test_huge_dictionary_keys_are_bounded(self):
        receipt = value_receipt({"K" * 100000: 1, "L" * 100000: 2}, 512)
        assert receipt["kind"] == "object"
        assert receipt["count"] == 2
        assert _size(receipt) < 2000, _size(receipt)
        assert all(len(key) < 200 for key in receipt["sample_keys"])

    def test_a_huge_dictionary_value_is_bounded(self):
        receipt = value_receipt({"snippet": "V" * 200000, "n": 2}, 512)
        assert _size(receipt) < 2000, _size(receipt)

    def test_a_small_value_is_still_returned_untouched(self):
        receipt = value_receipt({"a": [1, 2, 3]}, 512)
        assert receipt == {"complete": True, "value": {"a": [1, 2, 3]}, "chars": 16}

    def test_full_still_overrides_every_bound(self):
        value = ["X" * 100000] * 5
        assert value_receipt(value, 512, full=True)["value"] == value

    def test_a_value_whose_str_raises_does_not_take_the_receipt_down(self):
        class Hostile:
            def __str__(self):
                raise RuntimeError("no")

            __repr__ = __str__

        receipt = value_receipt(Hostile(), 10)
        assert receipt["complete"] is False
        assert "unrepresentable" in receipt["head"]


class TestSerializationNotesStayBounded:
    def test_thousands_of_coercions_report_a_count_and_a_handful_of_paths(self):
        safe, notes = json_safe([object() for _ in range(5000)])
        assert len(safe) == 5000
        assert notes["coerced_count"] == 5000
        assert len(notes["coerced"]) == 50, "note paths must be capped"
        assert _size(notes) < 4000, _size(notes)

    def test_a_huge_key_does_not_become_a_huge_note_path(self):
        safe, notes = json_safe({"K" * 100000: object()})
        assert notes["coerced_count"] == 1
        assert len(notes["coerced"][0]) < 300, len(notes["coerced"][0])
        assert len(next(iter(safe))) == 100000, "the dump copy keeps the real key"

    def test_an_object_whose_str_raises_is_recorded_not_propagated(self):
        class Hostile:
            def __str__(self):
                raise ValueError("no")

            __repr__ = __str__

        safe, notes = json_safe({"x": Hostile()})
        assert "unrepresentable" in safe["x"]
        assert notes["coerced"] == ["$.x"]


###### Review finding: complete is not the same as lossless


class TestOutputCompleteness:
    def test_a_clean_return_is_both_complete_and_lossless(self):
        result = code_handlers._execute_python("v = {'a': [1, 2]}", return_expression="v")
        assert result["return_lossless"] is True
        assert result["output_complete"] is True

    def test_a_stringified_object_is_valid_json_and_not_a_complete_answer(self):
        result = code_handlers._execute_python(
            "class T:\n    def __repr__(self):\n        return '<t>'\nv = {'t': T()}",
            return_expression="v",
        )
        assert result["execution_success"] is True
        assert result["success"] is True
        assert result["return_lossless"] is False
        assert result["return_serializable"] is False
        assert result["output_complete"] is False, "a str() of a node is not the node"
        assert result["return_coerced_count"] == 1

    def test_nonfinite_and_circular_results_are_not_complete_either(self):
        nonfinite = code_handlers._execute_python("", return_expression="float('inf')")
        assert nonfinite["return_lossless"] is False
        assert nonfinite["output_complete"] is False

        circular = code_handlers._execute_python(
            "loop = {}\nloop['self'] = loop", return_expression="loop"
        )
        assert circular["return_circular"] == ["$.self"]
        assert circular["output_complete"] is False

    def test_strict_json_reports_lossless_false_with_the_execution_intact(self):
        result = code_handlers._execute_python(
            "print('side effect happened')\nv = {'t': object()}",
            return_expression="v",
            return_format="json",
        )
        assert result["execution_success"] is True
        assert result["stdout"] == "side effect happened\n"
        assert result["success"] is False
        assert result["return_lossless"] is False
        assert result["return_serializable"] is False
        assert "not JSON" in result["serialization_error"]

    def test_repr_and_none_are_not_claimed_to_be_lossless(self):
        as_repr = code_handlers._execute_python("", return_expression="1.5", return_format="repr")
        assert as_repr["return_lossless"] is False
        assert as_repr["output_complete"] is False

        as_none = code_handlers._execute_python("", return_expression="1.5", return_format="none")
        assert as_none["return_lossless"] is False
        assert as_none["return_omitted"] is True

    def test_a_run_with_no_return_expression_is_complete(self):
        result = code_handlers._execute_python("print('hi')")
        assert result["output_complete"] is True
        assert "return_lossless" not in result

    def test_thousands_of_coercions_do_not_inflate_the_receipt(self):
        result = code_handlers._execute_python(
            "rows = [object() for _ in range(5000)]", return_expression="rows"
        )
        assert result["return_coerced_count"] == 5000
        assert len(result["return_coerced"]) == 10
        assert _size(result) < 8000, _size(result)


class TestRepresentationFailures:
    """repr() and __str__ run AFTER the code did; they must not eat the evidence."""

    HOSTILE = (
        "class Hostile:\n"
        "    def __repr__(self):\n"
        "        raise RuntimeError('repr exploded')\n"
        "    __str__ = __repr__\n"
    )

    def test_a_raising_repr_keeps_the_execution_verdict_and_the_output(self):
        result = code_handlers._execute_python(
            self.HOSTILE + "print('the code ran')\nv = Hostile()",
            return_expression="v",
            return_format="repr",
        )
        assert result["execution_success"] is True
        assert result["executed"] is True
        assert "error" not in result
        assert result["stdout"] == "the code ran\n"
        assert "repr exploded" in result["serialization_error"]
        assert result["return_value"] is None
        assert result["success"] is False

    def test_a_raising_str_under_auto_is_reported_as_a_coercion(self):
        result = code_handlers._execute_python(
            self.HOSTILE + "print('ran')\nv = {'h': Hostile()}", return_expression="v"
        )
        assert result["execution_success"] is True
        assert result["stdout"] == "ran\n"
        assert "unrepresentable" in result["return_value"]["h"]
        assert result["return_lossless"] is False

    def test_a_raising_repr_still_writes_the_dump(self, tmp_path):
        dump = tmp_path / "hostile.json"
        result = code_handlers._execute_python(
            self.HOSTILE + "v = Hostile()",
            return_expression="v",
            return_format="repr",
            dump_path=str(dump),
        )
        assert result["dump"]["written"] is True
        payload = json.loads(dump.read_text(encoding="utf-8"))
        assert payload["execution_success"] is True
        assert "repr exploded" in payload["serialization_error"]


###### Review finding: "exact" must prove the wiring, not assume it


def _build(network, **kwargs):
    from fxhoudinimcp_server.handlers import graph_handlers

    kwargs.setdefault("layout", False)
    return graph_handlers.build_network(parent_path=network.path(), **kwargs)


class TestExactInputEnforcement:
    """An environment that auto-wires new nodes is the reason "exact" exists."""

    def test_a_callback_added_input_is_disconnected_and_proved_gone(self, network):
        first = None

        def rewire(node):
            # The shape of an auto-connect callback: whatever is created next
            # gets wired to what came before.
            nonlocal first
            if first is not None and node.name() == "solo":
                node.inputs[0] = (first, 0)
            if node.name() == "src":
                first = node

        network.on_create = rewire
        result = _build(
            network,
            input_policy="exact",
            nodes=[{"type": "box", "name": "src"}, {"type": "merge", "name": "solo", "inputs": []}],
        )
        assert result["success"] is True
        assert result["exact_inputs_verified"] is True
        assert result["enforced_inputs"][0]["index"] == 0
        assert network.node("solo").inputs == {}
        solo = next(n for n in result["created"] if n["name"] == "solo")
        assert solo["inputs"] == []

    def _steal(self, network, *, refuse=False, ignore=False):
        """A parameter callback that moves an already-wired input to itself.

        Creation callbacks all fire before build_network wires anything; a
        parameter callback fires in the middle of the wiring pass, which is
        how a connection this command made gets moved behind its back.
        """

        def hook(node):
            if node.name() != "late":
                return
            dst = node._network.node("dst")
            if dst is None:
                return
            dst.inputs[0] = (node, 0)
            if refuse:
                dst.refuse_set.add(0)
            if ignore:
                dst.ignore_set.add(0)

        network.on_parm_set = hook
        return [
            {"type": "box", "name": "src"},
            {"type": "merge", "name": "dst", "inputs": ["src"]},
            {"type": "box", "name": "late", "parms": {"scale": 2.0}},
        ]

    def test_a_callback_that_moves_a_connection_is_put_back(self, network):
        result = _build(network, input_policy="exact", nodes=self._steal(network))
        assert result["success"] is True, result.get("input_mismatches")
        assert result["exact_inputs_verified"] is True
        assert result["enforced_inputs"][0]["rewired_from"].endswith("/late")
        assert result["enforced_inputs"][0]["to"].endswith("/src")
        assert network.node("dst").inputs[0][0].name() == "src"

    def test_a_disconnect_that_fails_rolls_the_build_back(self, network):
        def rewire(node):
            if node.name() == "solo":
                node.inputs[3] = (node._network.node("src"), 0)
                node.refuse_disconnect.add(3)

        network.on_create = rewire
        result = _build(
            network,
            input_policy="exact",
            nodes=[{"type": "box", "name": "src"}, {"type": "merge", "name": "solo", "inputs": []}],
        )
        assert result["success"] is False
        assert result["created"] == []
        assert "disconnecting it failed" in " ".join(result["errors"])
        assert network.children() == [], "a failed enforcement must not leave nodes behind"

    def test_a_reconnect_that_fails_rolls_the_build_back(self, network):
        result = _build(
            network, input_policy="exact", nodes=self._steal(network, refuse=True)
        )
        assert result["success"] is False
        assert "could not be set back" in " ".join(result["errors"])
        assert network.children() == []

    def test_wiring_that_silently_refuses_to_stick_is_not_called_success(self, network):
        """setInput returning without raising is not evidence of anything."""
        result = _build(
            network, input_policy="exact", nodes=self._steal(network, ignore=True)
        )
        assert result["success"] is False
        assert result["exact_inputs_verified"] is False
        mismatch = result["input_mismatches"][0]
        assert mismatch["index"] == 0
        assert mismatch["expected"].endswith("/src")
        assert mismatch["actual"].endswith("/late")
        # Not rolled back: the nodes exist, and the caller is told where.
        assert result["created_paths"] == [n.path() for n in network.children()]

    def test_the_source_output_is_part_of_the_comparison(self, network):
        def hook(node):
            if node.name() != "late":
                return
            dst = node._network.node("dst")
            # Same source, different output: still not what was asked for.
            dst.inputs[0] = (dst.inputs[0][0], 3)
            dst.ignore_set.add(0)

        network.on_parm_set = hook
        result = _build(
            network,
            input_policy="exact",
            nodes=[
                {"type": "box", "name": "src"},
                {"type": "merge", "name": "dst", "inputs": [{"source": "src", "source_output": 0}]},
                {"type": "box", "name": "late", "parms": {"scale": 2.0}},
            ],
        )
        assert result["success"] is False
        assert result["input_mismatches"][0]["actual_output"] == 3
        assert result["input_mismatches"][0]["expected_output"] == 0

    def test_an_omitted_inputs_key_stays_no_opinion(self, network):
        def rewire(node):
            if node.name() == "quiet":
                node.inputs[0] = (node._network.node("src"), 0)

        network.on_create = rewire
        result = _build(
            network,
            input_policy="exact",
            nodes=[{"type": "box", "name": "src"}, {"type": "merge", "name": "quiet"}],
        )
        assert result["success"] is True
        assert "input_mismatches" not in result
        assert 0 in network.node("quiet").inputs, "exact must not touch an unmentioned input"

    def test_preserve_leaves_a_callback_alone_and_reports_it(self, network):
        def rewire(node):
            if node.name() == "solo":
                node.inputs[0] = (node._network.node("src"), 0)

        network.on_create = rewire
        result = _build(
            network,
            nodes=[{"type": "box", "name": "src"}, {"type": "merge", "name": "solo", "inputs": []}],
        )
        assert result["success"] is True
        assert "exact_inputs_verified" not in result
        solo = next(n for n in result["created"] if n["name"] == "solo")
        assert [c["index"] for c in solo["inputs"]] == [0], "preserve still reports the truth"


class TestVerificationScope:
    """"cooked: true" about an empty target list was a confident non-answer."""

    def test_no_targets_means_unknown_not_healthy(self, network):
        result = _build(
            network, inspect_nodes=[], nodes=[{"type": "box", "name": "b"}]
        )
        assert result["cooked"] is None
        assert result["healthy"] is None
        assert result["verification"]["targets_inspected"] == 0
        assert result["verification"]["complete"] is False
        assert "nothing about cooking is known" in result["verification"]["note"]

    def test_unresolved_targets_only_means_unknown(self, network):
        result = _build(
            network, inspect_nodes=["nowhere"], nodes=[{"type": "box", "name": "b"}]
        )
        assert result["cooked"] is None
        assert result["healthy"] is None
        assert result["verification"]["targets_unresolved"] == ["nowhere"]
        assert result["verification"]["targets_requested"] == 1
        assert result["inspect_unresolved"] == ["nowhere"]

    def test_a_truncated_target_list_is_not_a_clean_bill(self, network):
        names = [f"n{i}" for i in range(10)]
        result = _build(
            network,
            inspect_nodes=names,
            nodes=[{"type": "box", "name": name} for name in names],
        )
        assert result["verification"]["targets_inspected"] == 8
        assert result["verification"]["targets_omitted"] == 2
        assert result["verification"]["complete"] is False
        assert result["cooked"] is None, "two targets were never cooked"
        assert result["healthy"] is None
        assert result["inspect_truncated"] == 2

    def test_a_fully_inspected_build_still_says_so(self, network):
        result = _build(network, nodes=[{"type": "box", "name": "b"}])
        assert result["cooked"] is True
        assert result["healthy"] is True
        assert result["verification"] == {
            "scope": "terminals",
            "targets_requested": 1,
            "targets_inspected": 1,
            "targets_omitted": 0,
            "targets_unresolved": [],
            "complete": True,
        }
        assert network.node("b").cooked == 1

    def test_the_scope_names_which_targets_were_chosen(self, network):
        default = _build(network, nodes=[{"type": "box", "name": "b"}])
        assert default["verification"]["scope"] == "terminals"
        requested = _build(
            network, inspect_nodes=["c"], nodes=[{"type": "box", "name": "c"}]
        )
        assert requested["verification"]["scope"] == "requested"

    def test_a_node_error_is_reported_bounded(self, network):
        def shout(node):
            node.error_text = ["E" * 50000, "second", "third", "fourth", "fifth", "sixth"]

        network.on_create = shout
        result = _build(network, nodes=[{"type": "box", "name": "loud"}])
        report = result["created"][0]
        assert result["healthy"] is False
        assert len(report["errors"]) == 5, "at most a handful of messages"
        assert report["error_count"] == 6
        assert len(report["errors"][0]) < 600, "a VEX listing must not land whole"
        assert "50000 chars" in report["errors"][0]


###### Review finding: strict JSON means strict, and the dump's promise is narrow


class TestStrictAndDumpPromises:
    def test_strict_json_rejects_nonfinite(self):
        result = code_handlers._execute_python(
            "print('ran')\nv = {'ratio': float('nan')}",
            return_expression="v",
            return_format="json",
        )
        assert result["execution_success"] is True
        assert result["stdout"] == "ran\n"
        assert result["success"] is False
        assert "nonfinite" in result["serialization_error"]
        assert result["return_value"] is None
        assert result["return_lossless"] is False

    def test_strict_json_rejects_cycles(self):
        result = code_handlers._execute_python(
            "v = {}\nv['self'] = v", return_expression="v", return_format="json"
        )
        assert result["success"] is False
        assert "circular" in result["serialization_error"]
        assert result["execution_success"] is True

    def test_strict_json_still_accepts_a_clean_value(self):
        result = code_handlers._execute_python(
            "v = {'a': [1, 2.5, 'x', None, True]}", return_expression="v", return_format="json"
        )
        assert result["success"] is True
        assert result["return_lossless"] is True
        assert result["return_value"] == {"a": [1, 2.5, "x", None, True]}

    def test_the_dump_does_not_claim_to_hold_a_value_nothing_holds(self, tmp_path):
        dump = tmp_path / "lossy.json"
        result = code_handlers._execute_python(
            "print('x' * 200000)\nv = {'ratio': float('inf'), 'thing': object()}",
            return_expression="v",
            dump_path=str(dump),
        )
        assert result["dump"]["written"] is True
        # The streams really are complete in the file...
        assert result["dump_streams_complete"] is True
        payload = json.loads(dump.read_text(encoding="utf-8"))
        assert payload["stdout_chars"] == 200001
        # ...but the return value in it is the altered copy, so the promise
        # that the dump holds "the full output" is not made.
        assert result["dump_return_lossless"] is False
        assert result["full_output_in_dump"] is False
        assert payload["return_value"]["ratio"] == "Infinity"
        assert payload["serialization_notes"]["nonfinite"] == ["$.ratio"]
        assert payload["return_lossless"] is False


###### Review finding: expressions and tracebacks are output too


class TestBoundedErrorAndExpressionText:
    HUGE = "ch(\"/obj/x/y\") + " * 6000  # ~100 000 characters of expression

    def test_a_refusal_does_not_quote_the_whole_expression(self, hou_stub, monkeypatch):
        parm = FakeParm("px", 5.0, expression=self.HUGE)
        _node(monkeypatch, FakeNode([parm], path="/obj/geo1/xform1"))
        with pytest.raises(ValueError) as caught:
            parameter_handlers._set_parameter("/obj/geo1/xform1", "px", 0.0)
        message = str(caught.value)
        assert len(self.HUGE) > 90000
        assert len(message) < 1000, len(message)
        assert "chars, sha1" in message, "the size and a fingerprint stand in for the rest"

    def test_clearing_reports_an_excerpt_of_what_it_removed(self, hou_stub, monkeypatch):
        parm = FakeParm("px", 5.0, expression=self.HUGE)
        _node(monkeypatch, FakeNode([parm], path="/obj/geo1/xform1"))
        result = parameter_handlers._set_parameter(
            "/obj/geo1/xform1", "px", 0.0, expression_policy="replace"
        )
        cleared = result["set"][0]["cleared"][0]
        assert cleared["previous_expression_chars"] == len(self.HUGE)
        assert cleared["previous_expression_truncated"] is True
        assert cleared["previous_expression_sha1"]
        assert len(cleared["previous_expression"]) <= 400
        assert _size(result) < 3000, _size(result)

    def test_a_surviving_expression_is_read_back_bounded(self, hou_stub, monkeypatch):
        parm = KeptExpressionParm("px", 5.0, expression=self.HUGE)
        _node(monkeypatch, FakeNode([parm], path="/obj/geo1/xform1"))
        result = parameter_handlers._set_parameter(
            "/obj/geo1/xform1", "px", 0.0, expression_policy="replace"
        )
        entry = result["set"][0]
        assert entry["expression_truncated"] is True
        assert len(entry["expression"]) < 600
        assert _size(result) < 4000, _size(result)

    def test_a_write_error_quoting_its_value_is_bounded(self, hou_stub, monkeypatch):
        class LoudParm(FakeParm):
            def set(self, value, follow_parm_reference=True):  # noqa: FBT002
                raise RuntimeError("rejected: " + "V" * 200000)

        _node(monkeypatch, FakeNode([LoudParm("sizex")]))
        result = parameter_handlers._set_parameters("/obj/geo1/box1", {"sizex": 1.0})
        assert result["success"] is False
        assert len(result["errors"][0]["error"]) < 700, len(result["errors"][0]["error"])

    def test_a_huge_exception_message_does_not_land_whole(self, tmp_path):
        dump = tmp_path / "huge_error.json"
        result = code_handlers._execute_python(
            "raise ValueError('M' * 500000)", dump_path=str(dump)
        )
        assert result["execution_success"] is False
        assert result["error_truncated"] is True
        assert result["error_chars"] > 500000
        assert len(result["error"]) < 4200, len(result["error"])
        assert result["error_sha1"]
        # The headline is built from the exception, so it names it even though
        # the message that follows was the enormous part.
        assert result["error_summary"].startswith("ValueError: ")
        assert len(result["error_summary"]) < 600
        assert _size(result) < 6000, _size(result)
        # The whole traceback is still recoverable.
        payload = json.loads(dump.read_text(encoding="utf-8"))
        assert len(payload["error"]) > 500000

    def test_both_ends_of_a_clipped_traceback_survive(self):
        result = code_handlers._execute_python(
            "def a():\n    raise KeyError('K' * 300000)\na()"
        )
        assert result["error"].startswith("Traceback (most recent call last)")
        assert "chars elided" in result["error"]
        assert result["error_summary"].startswith("KeyError: ")

    def test_a_small_error_is_reported_exactly_as_before(self):
        result = code_handlers._execute_python("raise ValueError('boom')")
        assert "boom" in result["error"]
        assert result["error"].startswith("Traceback")
        assert "error_truncated" not in result
        assert result["error_summary"] == "ValueError: boom"


###### Review finding: a write addresses one parameter and no other


class TestChannelReferenceSafety:
    def test_a_write_never_follows_a_channel_reference(self, hou_stub, monkeypatch):
        source = FakeParm("sizex", 7.0)
        referencing = ReferencingParm("sizex", source)
        _node(monkeypatch, FakeNode([referencing], path="/obj/geo1/box2"))
        result = parameter_handlers._set_parameter(
            "/obj/geo1/box2", "sizex", 3.0, expression_policy="replace"
        )
        assert result["success"] is True
        assert referencing.followed is False, "follow_parm_reference must be passed as False"
        # The node the caller never named is untouched.
        assert source.set_calls == []
        assert source.eval() == 7.0

    def test_preserve_refuses_a_referenced_parameter_outright(self, hou_stub, monkeypatch):
        source = FakeParm("sizex", 7.0)
        referencing = ReferencingParm("sizex", source)
        _node(monkeypatch, FakeNode([referencing], path="/obj/geo1/box2"))
        with pytest.raises(ValueError, match=r"ch\("):
            parameter_handlers._set_parameter("/obj/geo1/box2", "sizex", 3.0)
        assert referencing.set_calls == []
        assert source.eval() == 7.0

    def test_a_plain_write_passes_the_flag_too(self, hou_stub, monkeypatch):
        node = _node(monkeypatch, _box_node())
        parameter_handlers._set_parameter("/obj/geo1/box1", "scale", 2.0)
        assert node.parm("scale").followed is False

    def test_a_build_without_the_keyword_refuses_rather_than_following(self):
        class OldHoudiniParm(FakeParm):
            def set(self, value):  # the pre-keyword signature
                self.set_calls.append(value)

        parm = OldHoudiniParm("sizex")
        with pytest.raises(TypeError, match="follow_parm_reference"):
            set_parm_value(parm, 1.0)
        assert parm.set_calls == []


class TestClearingFailures:
    def test_a_channel_that_will_not_clear_stops_the_write(self, hou_stub, monkeypatch):
        source = FakeParm("sizex", 7.0)
        parm = UnclearableParm("sizex", expression=f'ch("{source.path()}")')
        _node(monkeypatch, FakeNode([parm], path="/obj/geo1/box2"))
        result = parameter_handlers._set_parameters(
            "/obj/geo1/box2", {"sizex": 3.0}, expression_policy="replace"
        )
        assert result["success"] is False
        assert "could not clear" in result["errors"][0]["error"]
        # Nothing was written, so nothing could have gone through the reference.
        assert parm.set_calls == []
        assert parm.expression() == f'ch("{source.path()}")'
        assert source.set_calls == []

    def test_one_unclearable_component_stops_that_whole_write(self, hou_stub, monkeypatch):
        good = FakeParm("sizex", 1.0)
        bad = UnclearableParm("sizey", 1.0, expression="$F")
        third = FakeParm("sizez", 1.0)
        node = FakeNode([good, bad, third], {"size": FakeParmTuple("size", [good, bad, third])})
        _node(monkeypatch, node)
        result = parameter_handlers._set_parameters(
            "/obj/geo1/box1", {"size": [2.0, 3.0, 4.0]}, expression_policy="replace"
        )
        assert result["success"] is False
        assert third.set_calls == [], "no component may be written once one refused"


class TestReplacePreflight:
    """Obvious nonsense is refused BEFORE an expression is deleted for it."""

    def test_a_dictionary_value_does_not_cost_the_expression(self, hou_stub, monkeypatch):
        parm = FakeParm("px", 5.0, expression="$CEX")
        _node(monkeypatch, FakeNode([parm], path="/obj/geo1/xform1"))
        result = parameter_handlers._set_parameters(
            "/obj/geo1/xform1", {"px": {"x": 1}}, expression_policy="replace"
        )
        assert result["success"] is False
        assert "scalar" in result["errors"][0]["error"]
        assert parm.deleted == 0
        assert parm.expression() == "$CEX"

    def test_a_null_value_does_not_cost_the_expression(self, hou_stub, monkeypatch):
        parm = FakeParm("px", 5.0, expression="$CEX")
        _node(monkeypatch, FakeNode([parm], path="/obj/geo1/xform1"))
        result = parameter_handlers._set_parameters(
            "/obj/geo1/xform1", {"px": None}, expression_policy="replace"
        )
        assert result["success"] is False
        assert "null" in result["errors"][0]["error"]
        assert parm.deleted == 0

    def test_a_word_written_to_a_numeric_parameter_is_refused_before_clearing(
        self, hou_stub, monkeypatch
    ):
        parm = FakeParm("iterations", 1, expression="$F", template=FakeTemplate("Int"))
        _node(monkeypatch, FakeNode([parm], path="/obj/geo1/subdivide1"))
        result = parameter_handlers._set_parameters(
            "/obj/geo1/subdivide1", {"iterations": "quads"}, expression_policy="replace"
        )
        assert result["success"] is False
        assert "not a number" in result["errors"][0]["error"]
        assert parm.deleted == 0

    def test_a_menu_token_on_an_integer_menu_is_not_refused(self, hou_stub, monkeypatch):
        parm = FakeParm(
            "fillmode", 0, template=FakeTemplate("Int", menu=("none", "quads")), accepts=(str, int)
        )
        _node(monkeypatch, FakeNode([parm], path="/obj/geo1/polyfill1"))
        result = parameter_handlers._set_parameters("/obj/geo1/polyfill1", {"fillmode": "quads"})
        assert result["success"] is True, result["errors"]
        assert parm.set_calls == ["quads"]

    def test_a_numeric_string_on_a_numeric_parameter_is_not_refused(self, hou_stub, monkeypatch):
        parm = FakeParm("scale", 1.0, template=FakeTemplate("Float"), accepts=(str, float, int))
        _node(monkeypatch, FakeNode([parm]))
        result = parameter_handlers._set_parameters("/obj/geo1/box1", {"scale": "2.5"})
        assert result["success"] is True, result["errors"]

    def test_a_tuple_length_error_still_precedes_everything(self, hou_stub, monkeypatch):
        parm = FakeParm("px", 5.0, expression="$CEX")
        node = FakeNode([parm], {"p": FakeParmTuple("p", [parm])}, path="/obj/geo1/xform1")
        _node(monkeypatch, node)
        result = parameter_handlers._set_parameters(
            "/obj/geo1/xform1", {"p": [1.0, 2.0]}, expression_policy="replace"
        )
        assert result["success"] is False
        assert "components" in result["errors"][0]["error"]
        assert parm.deleted == 0
