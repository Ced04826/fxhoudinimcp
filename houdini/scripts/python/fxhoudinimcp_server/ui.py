"""Saying "this needs a graphical Houdini" instead of crashing about it.

``hou.ui`` does not exist in hython or hbatch. Ten handlers reached for it without
checking, so asking for a viewport screenshot in a headless session answered:

    module 'hou' has no attribute 'ui'

which reads as a broken plugin and sends the caller debugging the server. The real
answer is that the operation needs a graphical session -- a different thing to know,
because it tells them to stop retrying and either open Houdini or pick a tool that
works without one.

Kept separate from the pane-finding helpers because it is the *first* question, not
a detail of the search: with no UI at all there are no panes to look through.
"""

from __future__ import annotations

# Third-party
import hou


def ui_available() -> bool:
    """Whether this Houdini has a UI at all.

    ``hou.isUIAvailable`` is itself missing in some headless builds, so the
    attribute check comes first.
    """
    try:
        return bool(hou.isUIAvailable())
    except AttributeError:
        return hasattr(hou, "ui")


def require_ui(operation: str, *, alternative: str = "") -> None:
    """Raise a readable error when *operation* needs a UI and there is none.

    Args:
        operation: What was being attempted, named as the caller would name it
            ("capture a viewport screenshot").
        alternative: Optional pointer to something that does work headlessly.
            Worth giving whenever one exists -- a refusal that names the way
            forward saves the round trip that a bare refusal costs.
    """
    if ui_available():
        return
    message = (
        f"Cannot {operation}: this Houdini session has no UI (hython/hbatch), "
        f"so there are no panes or viewports. It needs a graphical Houdini."
    )
    if alternative:
        message += f" {alternative}"
    raise hou.OperationFailed(message)


def panes():
    """``hou.ui.paneTabs()``, but explaining itself when there is no UI."""
    require_ui("look at Houdini's panes")
    return hou.ui.paneTabs()


# vieweroption -a: what the viewer draws of the objects you are not inside.
# The same switch as the viewport's Y hotkey; display flags are left alone.
OTHER_OBJECTS = {"hide": 0, "show": 1, "ghost": 2}


def set_other_objects(mode: str | None) -> str | None:
    """Set how every Scene Viewer draws objects other than the current one.

    Returns the mode applied, or None when there was nothing to do: no UI, no
    viewer, or *mode* None. Best effort by design, since isolating the viewer
    is a courtesy to the user and never a reason to fail the call it rides on.
    """
    if mode is None or not ui_available():
        return None
    if mode not in OTHER_OBJECTS:
        raise ValueError(f"other_objects must be one of {sorted(OTHER_OBJECTS)} or None")
    applied = None
    try:
        desktop = hou.ui.curDesktop().name()
        for tab in hou.ui.paneTabs():
            if tab.type() == hou.paneTabType.SceneViewer:
                hou.hscript(f"vieweroption -a {OTHER_OBJECTS[mode]} {desktop}.{tab.name()}.world")
                applied = mode
    except Exception:
        return applied
    return applied
