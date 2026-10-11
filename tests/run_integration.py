#!/usr/bin/env python3
"""Launch the integration test suite inside hython on Windows, macOS, or Linux.

Usage:
    python tests/run_integration.py              # whole integration suite
    python tests/run_integration.py -k pyro      # pytest args pass through

Finds the newest installed Houdini (override with the HYTHON environment
variable pointing at the hython executable) and reuses this interpreter's
pytest installation via PYTHONPATH. Requires a Houdini license seat.
"""

from __future__ import annotations

# Built-in
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

# Internal
from fxhoudinimcp.houdini_discovery import find_all_hython  # noqa: E402, F401
from fxhoudinimcp.houdini_discovery import find_hython as _find_hython  # noqa: E402


def find_hython() -> Path:
    """Return the hython executable: $HYTHON, newest install, or PATH."""
    env_override = os.environ.get("HYTHON")
    if env_override and not Path(env_override).is_file():
        sys.exit(f"HYTHON is set but does not exist: {env_override}")
    hython = _find_hython()
    if hython is None:
        sys.exit(
            "No hython executable found. Install Houdini or set the HYTHON "
            "environment variable to the full path of hython."
        )
    return hython


def main() -> int:
    hython = find_hython()

    try:
        import pytest  # noqa: F401

        site_packages = Path(pytest.__file__).resolve().parent.parent
    except ImportError:
        sys.exit(
            "pytest is not importable from this Python. Install it first: "
            f"{sys.executable} -m pip install pytest"
        )

    env = os.environ.copy()
    python_path = [str(REPO_ROOT / "python"), str(site_packages)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    print(f"Using hython: {hython}")
    command = [
        str(hython),
        "-m",
        "pytest",
        str(REPO_ROOT / "tests" / "integration"),
        "-q",
        "-s",
        "--durations=15",
        *sys.argv[1:],
    ]
    return subprocess.call(command, env=env, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    sys.exit(main())
