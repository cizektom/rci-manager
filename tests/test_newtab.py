"""Terminal-detection plan shaping for :mod:`rci_cli.newtab`.

We never actually spawn a tab in tests — only assert the argv list that *would*
be passed to :func:`subprocess.run`. Each test isolates ``os.environ`` to the
variables that select a given terminal.
"""

from __future__ import annotations

import pytest

from rci_cli import newtab


TERMINAL_ENV_KEYS = (
    "TMUX",
    "ZELLIJ_SESSION_NAME",
    "WEZTERM_PANE",
    "KITTY_WINDOW_ID",
    "WT_SESSION",
    "WSL_DISTRO_NAME",
    "TERM_PROGRAM",
    "KONSOLE_VERSION",
)


@pytest.fixture(autouse=True)
def _isolate_terminal_env(monkeypatch):
    """Strip every detection-relevant var so each test sets only what it needs."""
    for key in TERMINAL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# ── per-terminal plans ──────────────────────────────────────────────────────


def test_tmux_plan_uses_new_window_with_name(monkeypatch) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")
    plan = newtab._detect_plan(["rci", "shell"], title="rci-shell")
    assert plan.kind == "tmux"
    assert plan.argv == ["tmux", "new-window", "-n", "rci-shell", "--", "rci", "shell"]


def test_tmux_plan_without_title_omits_name(monkeypatch) -> None:
    monkeypatch.setenv("TMUX", "x")
    plan = newtab._detect_plan(["echo", "hi"], title=None)
    assert plan.argv == ["tmux", "new-window", "--", "echo", "hi"]


def test_zellij_plan(monkeypatch) -> None:
    monkeypatch.setenv("ZELLIJ_SESSION_NAME", "main")
    plan = newtab._detect_plan(["rci", "claude"], title="rci-claude")
    assert plan.kind == "zellij"
    assert plan.argv == ["zellij", "action", "new-tab", "--name", "rci-claude", "--", "rci", "claude"]


def test_wezterm_plan(monkeypatch) -> None:
    monkeypatch.setenv("WEZTERM_PANE", "1")
    plan = newtab._detect_plan(["rci", "shell"], title="ignored")
    assert plan.kind == "wezterm"
    assert plan.argv == ["wezterm", "cli", "spawn", "--", "rci", "shell"]


def test_kitty_plan(monkeypatch) -> None:
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    plan = newtab._detect_plan(["rci", "shell"], title="rci-shell")
    assert plan.kind == "kitty"
    assert plan.argv == [
        "kitten",
        "@",
        "launch",
        "--type=tab",
        "--tab-title",
        "rci-shell",
        "--",
        "rci",
        "shell",
    ]


def test_wt_plan_wraps_in_bash_lc_inside_wsl(monkeypatch) -> None:
    monkeypatch.setenv("WT_SESSION", "abc")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(newtab.shutil, "which", lambda _: "/mnt/c/.../wt.exe")
    plan = newtab._detect_plan(["rci", "shell", "--node", "n01"], title="rci-shell")
    assert plan.kind == "Windows Terminal"
    assert plan.argv == [
        "wt.exe",
        "-w",
        "0",
        "nt",
        "--title",
        "rci-shell",
        "wsl.exe",
        "-d",
        "Ubuntu",
        "--",
        "bash",
        "-lc",
        "rci shell --node n01",
    ]


def test_wt_plan_quotes_args_with_spaces(monkeypatch) -> None:
    """Folder paths with spaces must survive the wsl→bash hop."""
    monkeypatch.setenv("WT_SESSION", "abc")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(newtab.shutil, "which", lambda _: "/mnt/c/.../wt.exe")
    plan = newtab._detect_plan(["rci", "shell", "my folder"], title=None)
    # shlex.join must quote the space-containing arg
    assert plan.argv[-1] == "rci shell 'my folder'"


def test_wt_plan_falls_through_when_wt_exe_missing(monkeypatch) -> None:
    monkeypatch.setenv("WT_SESSION", "abc")
    monkeypatch.setattr(newtab.shutil, "which", lambda _: None)
    with pytest.raises(newtab.NoSupportedTerminal):
        newtab._detect_plan(["rci", "shell"], title=None)


def test_iterm_plan_emits_osascript(monkeypatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    plan = newtab._detect_plan(["rci", "shell"], title=None)
    assert plan.kind == "iTerm2"
    assert plan.argv[0] == "osascript"
    assert plan.argv[1] == "-e"
    assert 'create tab with default profile command "rci shell"' in plan.argv[2]


def test_konsole_plan(monkeypatch) -> None:
    monkeypatch.setenv("KONSOLE_VERSION", "230400")
    monkeypatch.setattr(newtab.shutil, "which", lambda _: "/usr/bin/konsole")
    plan = newtab._detect_plan(["rci", "shell"], title=None)
    assert plan.kind == "Konsole"
    assert plan.argv == ["konsole", "--new-tab", "-e", "rci", "shell"]


# ── detection priority ──────────────────────────────────────────────────────


def test_tmux_wins_over_wt(monkeypatch) -> None:
    """Inside tmux running in Windows Terminal, tmux should win — it gives the
    user a new tab without opening a separate WT tab."""
    monkeypatch.setenv("TMUX", "/tmp/tmux/default,1,0")
    monkeypatch.setenv("WT_SESSION", "abc")
    monkeypatch.setattr(newtab.shutil, "which", lambda _: "/mnt/c/.../wt.exe")
    plan = newtab._detect_plan(["echo"], title=None)
    assert plan.kind == "tmux"


def test_no_terminal_raises() -> None:
    with pytest.raises(newtab.NoSupportedTerminal):
        newtab._detect_plan(["echo"], title=None)


def test_is_supported_false_when_nothing_matches() -> None:
    assert newtab.is_supported() is False


def test_is_supported_true_when_tmux(monkeypatch) -> None:
    monkeypatch.setenv("TMUX", "x")
    assert newtab.is_supported() is True


# ── public spawn() shells out to subprocess.run ─────────────────────────────


def test_spawn_invokes_subprocess(monkeypatch) -> None:
    monkeypatch.setenv("TMUX", "x")
    seen = {}

    class FakeResult:
        returncode = 0

    def fake_run(argv, check=False):
        seen["argv"] = argv
        seen["check"] = check
        return FakeResult()

    monkeypatch.setattr(newtab.subprocess, "run", fake_run)
    rc, kind = newtab.spawn(["rci", "shell"], title="rci-shell")
    assert rc == 0
    assert kind == "tmux"
    assert seen["argv"] == ["tmux", "new-window", "-n", "rci-shell", "--", "rci", "shell"]
    assert seen["check"] is False
