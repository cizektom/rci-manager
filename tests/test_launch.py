"""Folder resolution + remote preamble for compute-node commands."""

from __future__ import annotations

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
