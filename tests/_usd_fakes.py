"""Minimal pxr stand-ins shared by the USD handler tests."""

from __future__ import annotations

# Built-in
from unittest.mock import MagicMock


class Path:
    """Enough of Sdf.Path: str(), element-wise prefix and element count."""

    def __init__(self, text):
        self.text = text
        self.elements = [e for e in text.split("/") if e]

    @property
    def pathElementCount(self):
        return len(self.elements)

    def HasPrefix(self, other):
        return self.elements[: len(other.elements)] == other.elements

    def __str__(self):
        return self.text


def prim(path, valid=True, type_name="Xform"):
    """A Usd.Prim at *path*; falsy and invalid when *valid* is False."""
    fake = MagicMock()
    fake.GetPath.return_value = Path(path)
    fake.GetTypeName.return_value = type_name
    fake.IsValid.return_value = valid
    fake.__bool__ = lambda self: valid
    return fake
