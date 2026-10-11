"""set_viewport_display could not hide the environment background.

The only setting it had was the shading mode. Hiding a dome light's map from
behind the scene took GeometryViewportSettings
.setDisplayEnvironmentBackgroundImage() through execute_python, once per view,
because the flag lives on each view: a quad layout carries four of them. Now
environment_background sets it on every view of the viewer and the reply reads
each one back; display_mode is optional so either can be set alone.

hou is mocked here; the live check ran on Houdini 22.0.429.
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
import fxhoudinimcp_server.handlers.viewport_handlers as viewport  # noqa: E402


class _FakeSettings:
    def __init__(self, shown):
        self.shown = shown
        self.display_set = MagicMock()

    def setDisplayEnvironmentBackgroundImage(self, on):
        self.shown = on

    def displayEnvironmentBackgroundImage(self):
        return self.shown

    def displaySet(self, _kind):
        return self.display_set


class _FakeView:
    def __init__(self, name, shown=True):
        self._name = name
        self._settings = _FakeSettings(shown)

    def name(self):
        return self._name

    def settings(self):
        return self._settings


def _viewer(monkeypatch, names=("right1", "front1", "top1", "persp1")):
    views = [_FakeView(name) for name in names]
    viewer = MagicMock()
    viewer.name.return_value = "panetab1"
    viewer.viewports.return_value = views
    viewer.curViewport.return_value = views[-1]
    monkeypatch.setattr(viewport, "_find_scene_viewer", lambda pane_name=None: viewer)
    return viewer, views


class TestEnvironmentBackground:
    def test_every_view_of_a_quad_layout_is_set(self, monkeypatch):
        _, views = _viewer(monkeypatch)
        result = viewport.set_viewport_display(environment_background=False)
        assert result["environment_background"] == {
            "right1": False,
            "front1": False,
            "top1": False,
            "persp1": False,
        }
        assert all(view.settings().shown is False for view in views)

    def test_the_reply_is_read_back_not_echoed(self, monkeypatch):
        _, views = _viewer(monkeypatch, names=("persp1",))
        # A setter that does not take must show up in the reply.
        views[0]._settings.setDisplayEnvironmentBackgroundImage = lambda on: None
        result = viewport.set_viewport_display(environment_background=False)
        assert result["environment_background"] == {"persp1": True}

    def test_alone_it_leaves_the_shading_mode_untouched(self, monkeypatch):
        _, views = _viewer(monkeypatch)
        result = viewport.set_viewport_display(environment_background=True)
        assert "display_mode" not in result
        for view in views:
            view.settings().display_set.setShadedMode.assert_not_called()

    def test_both_are_applied_in_one_call(self, monkeypatch):
        _, views = _viewer(monkeypatch)
        result = viewport.set_viewport_display(display_mode="smooth", environment_background=True)
        assert result["display_mode"] == "smooth"
        assert set(result["environment_background"].values()) == {True}
        views[-1].settings().display_set.setShadedMode.assert_called_once()


class TestDisplayModeAlone:
    def test_reply_keeps_its_keys(self, monkeypatch):
        _viewer(monkeypatch)
        result = viewport.set_viewport_display(display_mode="wireframe")
        assert result == {"success": True, "pane_name": "panetab1", "display_mode": "wireframe"}

    def test_unknown_mode_still_names_the_choices(self, monkeypatch):
        _viewer(monkeypatch)
        with pytest.raises(ValueError, match="Supported modes"):
            viewport.set_viewport_display(display_mode="glossy")

    def test_nothing_to_set_is_refused(self, monkeypatch):
        _viewer(monkeypatch)
        with pytest.raises(ValueError, match="display_mode, environment_background"):
            viewport.set_viewport_display()


###### Reference plane, particle point size, colour scheme
#
# Each of these took execute_python before. The point size has a reader on
# GeometryViewportSettings (particlePointSize) and no setter: a session looking
# for one read back 3.0 and gave up. It is set with viewdisplay -p on the viewer.


class _FakeScheme:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"viewportColorScheme.{self.name}"


class _Houdini:
    """The viewer-wide state: reference plane, hscript's point size, desktop."""

    def __init__(self, monkeypatch, views):
        self.commands = []
        self.error = ""
        self.takes = True
        self.plane = MagicMock()
        self.plane.isVisible.side_effect = lambda: self.plane.setIsVisible.call_args[0][0]
        for view in views:
            view.settings().particlePointSize = lambda: 3.0
            view.settings().scheme = _FakeScheme("Grey")
            view.settings().setColorScheme = lambda s, _v=view: setattr(_v.settings(), "scheme", s)
            view.settings().colorScheme = lambda _v=view: _v.settings().scheme
        self.views = views
        monkeypatch.setattr(viewport.hou, "hscript", self._hscript, raising=False)
        monkeypatch.setattr(
            viewport.hou,
            "viewportColorScheme",
            MagicMock(**{n: _FakeScheme(n) for n in ("Dark", "DarkGrey", "Grey", "Light")}),
            raising=False,
        )
        desktop = MagicMock()
        desktop.name.return_value = "Build"
        monkeypatch.setattr(viewport.hou.ui, "curDesktop", lambda: desktop, raising=False)

    def _hscript(self, command):
        self.commands.append(command)
        if self.takes and not self.error:
            size = float(command.split()[2])
            for view in self.views:
                view.settings().particlePointSize = lambda _s=size: _s
        return "", self.error


@pytest.fixture
def houdini(monkeypatch):
    viewer, views = _viewer(monkeypatch)
    state = _Houdini(monkeypatch, views)
    viewer.referencePlane.return_value = state.plane
    state.viewer = viewer
    return state


class TestReferencePlane:
    def test_the_grid_is_hidden_and_read_back(self, houdini):
        result = viewport.set_viewport_display(reference_plane=False)
        houdini.plane.setIsVisible.assert_called_once_with(False)
        assert result["reference_plane"] is False
        assert "display_mode" not in result

    def test_the_reply_is_read_back_not_echoed(self, houdini):
        houdini.plane.isVisible.side_effect = lambda: True  # the setter did not take
        result = viewport.set_viewport_display(reference_plane=False)
        assert result["reference_plane"] is True


class TestPointSize:
    def test_it_goes_through_viewdisplay_on_the_viewer(self, houdini):
        viewport.set_viewport_display(point_size=4)
        assert houdini.commands == ["viewdisplay -p 4.0 Build.panetab1.world"]

    def test_in_a_lop_network_it_goes_to_the_solaris_viewer(self, houdini, monkeypatch):
        # .world from /stage changed the object viewer's size, not the one
        # shown (22.0.368).
        monkeypatch.setattr(viewport.hou, "lopNodeTypeCategory", lambda: "Lop", raising=False)
        houdini.viewer.pwd.return_value.childTypeCategory.return_value = "Lop"
        viewport.set_viewport_display(point_size=4)
        assert houdini.commands == ["viewdisplay -p 4.0 Build.panetab1.solaris"]

    def test_every_view_is_read_back(self, houdini):
        result = viewport.set_viewport_display(point_size=4)
        assert result["point_size"] == {"right1": 4.0, "front1": 4.0, "top1": 4.0, "persp1": 4.0}

    def test_the_reply_is_read_back_not_echoed(self, houdini):
        houdini.takes = False  # the command ran and changed nothing
        result = viewport.set_viewport_display(point_size=4)
        assert set(result["point_size"].values()) == {3.0}

    def test_an_hscript_error_is_raised_not_swallowed(self, houdini):
        houdini.error = "Unknown option"
        with pytest.raises(RuntimeError, match="viewdisplay -p failed: Unknown option"):
            viewport.set_viewport_display(point_size=4)

    @pytest.mark.parametrize("bad", [0, -2, "big", True, [4]])
    def test_a_bad_size_is_refused_before_anything_is_set(self, houdini, bad):
        with pytest.raises(ValueError, match="point_size must be a positive number"):
            viewport.set_viewport_display(display_mode="smooth", point_size=bad)
        assert houdini.commands == []
        houdini.views[-1].settings().display_set.setShadedMode.assert_not_called()


class TestColorScheme:
    def test_every_view_is_set_and_read_back(self, houdini):
        result = viewport.set_viewport_display(color_scheme="darkgrey")
        assert result["color_scheme"] == {
            "right1": "DarkGrey",
            "front1": "DarkGrey",
            "top1": "DarkGrey",
            "persp1": "DarkGrey",
        }

    def test_the_reply_is_read_back_not_echoed(self, houdini):
        houdini.views[0].settings().setColorScheme = lambda scheme: None  # did not take
        result = viewport.set_viewport_display(color_scheme="light")
        assert result["color_scheme"]["right1"] == "Grey"
        assert result["color_scheme"]["persp1"] == "Light"

    @pytest.mark.parametrize("spelling", ["Dark Grey", "dark_grey", "DARKGREY"])
    def test_the_spellings_of_one_scheme(self, houdini, spelling):
        result = viewport.set_viewport_display(color_scheme=spelling)
        assert set(result["color_scheme"].values()) == {"DarkGrey"}

    def test_an_unknown_scheme_names_the_choices_before_anything_is_set(self, houdini):
        with pytest.raises(ValueError, match="dark, darkgrey, grey, light"):
            viewport.set_viewport_display(reference_plane=False, color_scheme="midnight")
        houdini.plane.setIsVisible.assert_not_called()


class TestAllInOneCall:
    def test_every_setting_applies_and_each_answers(self, houdini):
        result = viewport.set_viewport_display(
            display_mode="smooth",
            environment_background=False,
            reference_plane=False,
            point_size=4,
            color_scheme="dark",
        )
        assert result["display_mode"] == "smooth"
        assert set(result["environment_background"].values()) == {False}
        assert result["reference_plane"] is False
        assert set(result["point_size"].values()) == {4.0}
        assert set(result["color_scheme"].values()) == {"Dark"}
