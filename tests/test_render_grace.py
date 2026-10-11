"""An out-of-process renderer gets longer than the default to show its file.

A Karma CPU frame appeared ~3 s after render() returned, past the 2 s default,
so start_render reported "nothing was written" for a render that worked.
"""

from __future__ import annotations

# Built-in
import os
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("hou", MagicMock())
sys.modules.setdefault("hdefereval", MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"))

# Third-party
# Internal
import fxhoudinimcp_server.config as houdini_config  # noqa: E402
import fxhoudinimcp_server.handlers.rendering_handlers as rendering  # noqa: E402
import pytest  # noqa: E402


def _node(type_name):
    node = MagicMock()
    node.type.return_value.name.return_value = type_name
    return node


@pytest.fixture(autouse=True)
def default_grace(monkeypatch):
    monkeypatch.setattr(houdini_config.hou, "getenv", lambda name: None)
    monkeypatch.delenv("FXHOUDINIMCP_OUTPUT_GRACE", raising=False)


@pytest.mark.parametrize("type_name", ["karma", "usdrender", "usdrender_rop", "ifd", "karma::1.0"])
def test_husk_style_renderers_get_a_long_window(type_name):
    assert rendering._render_grace(_node(type_name)) == 20.0


@pytest.mark.parametrize("type_name", ["geometry", "alembic", "opengl", "filecache::2.0"])
def test_in_process_writers_keep_the_short_default(type_name):
    assert rendering._render_grace(_node(type_name)) == 2.0


def test_a_larger_configured_grace_still_wins(monkeypatch):
    monkeypatch.setenv("FXHOUDINIMCP_OUTPUT_GRACE", "60")

    assert rendering._render_grace(_node("karma")) == 60.0
