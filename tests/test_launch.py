"""Folder resolution + remote preamble for compute-node commands."""

from __future__ import annotations

import pytest

from rci_cli import launch, newtab
from rci_cli.alloc import Allocation
from rci_cli.config import Config
from rci_cli.launch import _remote_preamble, resolve_folder


def test_resolve_folder_empty_returns_home(cfg: Config) -> None:
    assert resolve_folder("", cfg) == "/home/cizekto2"
    assert resolve_folder(None, cfg) == "/home/cizekto2"


def test_resolve_folder_relative_is_under_home(cfg: Config) -> None:
    assert resolve_folder("sam2rl", cfg) == "/home/cizekto2/sam2rl"


def test_resolve_folder_absolute_passthrough(cfg: Config) -> None:
    assert resolve_folder("/scratch/exp42", cfg) == "/scratch/exp42"


def test_remote_preamble_sources_venv_and_cds(cfg: Config) -> None:
    s = _remote_preamble("/home/cizekto2/sam2rl", cfg)
    assert "cd '/home/cizekto2/sam2rl'" in s
    assert "$HOME/sam2rl/.venv/bin/activate" in s
    assert '$HOME/bin:$HOME/.local/bin:$PATH' in s


# ── tab-spawning launchers ──────────────────────────────────────────────────


@pytest.fixture
def captured_spawn(monkeypatch):
    """Patch :func:`newtab.spawn` and capture its argv + title."""
    seen: dict = {}

    def fake_spawn(argv, *, title=None):
        seen["argv"] = list(argv)
        seen["title"] = title
        return (0, "tmux")

    monkeypatch.setattr(newtab, "spawn", fake_spawn)
    return seen


def test_launch_shell_in_tab_passes_node_and_folder(cfg: Config, captured_spawn) -> None:
    a = Allocation(node="n01", jobid="1234")
    rc = launch.launch_shell_in_tab(a, "sam2rl", cfg)
    assert rc == 0
    assert captured_spawn["argv"] == ["rci", "shell", "sam2rl", "--node", "n01"]
    assert captured_spawn["title"] == "rci shell · n01"


def test_launch_claude_in_tab_without_folder(cfg: Config, captured_spawn) -> None:
    a = Allocation(node="g05", jobid="5555")
    rc = launch.launch_claude_in_tab(a, "", cfg)
    assert rc == 0
    assert captured_spawn["argv"] == ["rci", "claude", "--node", "g05"]
    assert captured_spawn["title"] == "rci claude · g05"


def test_launch_shell_in_tab_returns_2_when_unsupported(cfg: Config, monkeypatch) -> None:
    def boom(argv, *, title=None):
        raise newtab.NoSupportedTerminal("nope")

    monkeypatch.setattr(newtab, "spawn", boom)
    rc = launch.launch_shell_in_tab(Allocation(node="n01", jobid="1"), "", cfg)
    assert rc == 2
