"""Spawn a new terminal tab running a given argv list.

Detection priority (most specific → least): tmux, zellij, WezTerm, kitty,
Windows Terminal on WSL, iTerm2 on macOS, Konsole. The parent ``rci`` process
returns immediately — the new tab runs the command independently.

Quoting: every plan builds a list-style argv handed to :mod:`subprocess`. For
the wt.exe path the rci args are joined with :func:`shlex.join` and passed as a
single ``bash -lc`` string, so the chain Linux→WSL-interop→wt.exe→wsl.exe→bash
only has to survive one layer of shell parsing (the final ``bash -lc``).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SpawnPlan:
    """Resolved argv that opens a new tab and runs the target command."""

    kind: str
    argv: list[str]


class NoSupportedTerminal(RuntimeError):
    """No supported terminal multiplexer/emulator detected."""


def _detect_plan(target: Sequence[str], title: str | None) -> SpawnPlan:
    """Return a :class:`SpawnPlan` for the current terminal.

    ``target`` is the argv to run in the new tab (e.g. ``["rci", "shell"]``).
    Raises :class:`NoSupportedTerminal` if no spawner is detected.
    """
    env = os.environ
    target = list(target)

    if env.get("TMUX"):
        argv = ["tmux", "new-window"]
        if title:
            argv.extend(["-n", title])
        argv.append("--")
        argv.extend(target)
        return SpawnPlan("tmux", argv)

    if env.get("ZELLIJ_SESSION_NAME"):
        argv = ["zellij", "action", "new-tab"]
        if title:
            argv.extend(["--name", title])
        argv.append("--")
        argv.extend(target)
        return SpawnPlan("zellij", argv)

    if env.get("WEZTERM_PANE"):
        argv = ["wezterm", "cli", "spawn", "--", *target]
        return SpawnPlan("wezterm", argv)

    if env.get("KITTY_WINDOW_ID"):
        argv = ["kitten", "@", "launch", "--type=tab"]
        if title:
            argv.extend(["--tab-title", title])
        argv.append("--")
        argv.extend(target)
        return SpawnPlan("kitty", argv)

    if env.get("WT_SESSION") and shutil.which("wt.exe"):
        distro = env.get("WSL_DISTRO_NAME", "")
        # Wrap the rci args in ``bash -lc <quoted>`` so .profile runs and
        # ``rci`` is on PATH inside the spawned tab.
        inner = shlex.join(target)
        argv = ["wt.exe", "-w", "0", "nt"]
        if title:
            argv.extend(["--title", title])
        argv.append("wsl.exe")
        if distro:
            argv.extend(["-d", distro])
        argv.extend(["--", "bash", "-lc", inner])
        return SpawnPlan("Windows Terminal", argv)

    if env.get("TERM_PROGRAM") == "iTerm.app":
        # AppleScript: open a new tab in the current window running ``target``.
        cmd = shlex.join(target)
        script = (
            'tell application "iTerm" to tell current window '
            f'to create tab with default profile command "{cmd}"'
        )
        return SpawnPlan("iTerm2", ["osascript", "-e", script])

    if env.get("KONSOLE_VERSION") and shutil.which("konsole"):
        argv = ["konsole", "--new-tab", "-e", *target]
        return SpawnPlan("Konsole", argv)

    raise NoSupportedTerminal(
        "no supported terminal detected — tried tmux, zellij, WezTerm, kitty, "
        "Windows Terminal (wt.exe), iTerm2, Konsole"
    )


def spawn(argv: Sequence[str], *, title: str | None = None) -> tuple[int, str]:
    """Open a new terminal tab running ``argv``. Returns ``(rc, kind)``.

    ``rc`` is the spawner's exit code (0 = tab opened). The new tab runs
    independently of this process.
    """
    plan = _detect_plan(argv, title)
    rc = subprocess.run(plan.argv, check=False).returncode
    return rc, plan.kind


def is_supported() -> bool:
    """True if a supported terminal multiplexer/emulator is detected."""
    try:
        _detect_plan(["true"], None)
    except NoSupportedTerminal:
        return False
    return True
