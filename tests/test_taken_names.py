"""A name the caller asked for is theirs or an error, never a silent "name1".

rename_node, copy_node and the create tools passed the clash to Houdini, which
answered with "name1" and a success reply, so later calls on the requested
name hit the other node.
"""

from __future__ import annotations

# Built-in
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

# Internal
import fxhoudinimcp_server.handlers.node_handlers as nodes  # noqa: E402


def _network(children):
    parent = MagicMock()
    parent.path.return_value = "/obj"
    parent.node.side_effect = children.get
    return parent


def test_a_taken_name_is_refused():
    other = MagicMock(path=MagicMock(return_value="/obj/a"))
    with pytest.raises(ValueError, match="already taken in /obj by /obj/a"):
        nodes._refuse_taken_name(_network({"a": other}), "a")


def test_a_free_name_or_the_node_itself_passes():
    me = MagicMock()
    parent = _network({"a": me})
    nodes._refuse_taken_name(parent, "b")
    nodes._refuse_taken_name(parent, "a", me)


def test_rename_to_a_taken_name_does_not_rename():
    me, other = MagicMock(), MagicMock()
    me.parent.return_value = _network({"a": other})
    with patch.object(nodes, "_get_node", return_value=me), pytest.raises(ValueError):
        nodes.rename_node("/obj/b", "a")
    me.setName.assert_not_called()


def test_copy_to_a_taken_name_copies_nothing():
    source = MagicMock()
    source.parent.return_value = _network({"a": MagicMock()})
    with (
        patch.object(nodes, "_get_node", return_value=source),
        patch.object(nodes.hou, "copyNodesTo") as copy,
        pytest.raises(ValueError),
    ):
        nodes.copy_node("/obj/b", new_name="a")
    copy.assert_not_called()
