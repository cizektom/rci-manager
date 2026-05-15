"""Folder resolution and the remote command shape for shell / editor launchers.

The launchers themselves shell out through :mod:`rci_cli.ssh`; here we assert
the *shape* of the script that ends up on the wire, not actual ssh execution.
"""

from __future__ import annotations

from rci_cli import launch, ssh
from rci_cli.alloc import Allocation
from rci_cli.config import Config
from rci_cli.launch import _remote_preamble, resolve_folder


# ── folder resolution ──────────────────────────────────────────────────────


def test_resolve_folder_empty_returns_home(cfg: Config) -> None:
    assert resolve_folder("", cfg) == "/home/cizekto2"
    assert resolve_folder(None, cfg) == "/home/cizekto2"


def test_resolve_folder_relative_is_under_home(cfg: Config) -> None:
    assert resolve_folder("sam2rl", cfg) == "/home/cizekto2/sam2rl"


def test_resolve_folder_absolute_passthrough(cfg: Config) -> None:
    assert resolve_folder("/scratch/exp42", cfg) == "/scratch/exp42"


# ── remote preamble ────────────────────────────────────────────────────────


def test_remote_preamble_cds_sources_venv_and_sets_path(cfg: Config) -> None:
    s = _remote_preamble("/home/cizekto2/sam2rl", cfg)
    assert "cd '/home/cizekto2/sam2rl'" in s
    assert cfg.venv_activate in s
    assert "$HOME/bin:$HOME/.local/bin:$PATH" in s


# ── launchers (mock ssh.run / ssh.run_local) ───────────────────────────────


def _patch_ssh(monkeypatch) -> dict:
    captured: dict = {}

    def fake_run(host, cmd="", *, tty=False, check=True, stdin=None):
        captured["host"] = host
        captured["cmd"] = cmd
        captured["tty"] = tty
        return 0

    def fake_run_local(argv, *, check=True):
        captured["local_argv"] = list(argv)
        return 0

    monkeypatch.setattr(ssh, "run", fake_run)
    monkeypatch.setattr(ssh, "run_local", fake_run_local)
    return captured


def test_launch_shell_runs_bash_over_ssh(monkeypatch, cfg: Config) -> None:
    captured = _patch_ssh(monkeypatch)
    rc = launch.launch_shell(Allocation(node="g05", jobid="5555"), "/home/cizekto2", cfg)
    assert rc == 0
    assert captured["host"] == "g05"
    assert captured["tty"] is True
    assert "exec bash -i" in captured["cmd"]


def test_launch_editor_uses_vscode_remote_uri(monkeypatch, cfg: Config) -> None:
    captured = _patch_ssh(monkeypatch)
    launch.launch_editor(Allocation(node="n01", jobid="1234"), "/home/cizekto2/sam2rl", cfg)
    argv = captured["local_argv"]
    # Either ["code", "--folder-uri", URI] or ["cmd.exe", "/c", "code", "--folder-uri", URI]
    assert any("vscode-remote://ssh-remote+n01/home/cizekto2/sam2rl" in a for a in argv)
