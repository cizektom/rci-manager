"""Folder resolution, session naming, and the tmux wrapping shell script.

The launchers themselves shell out through :mod:`rci_cli.ssh`; here we assert
the *shape* of the script that ends up on the wire, not actual ssh execution.
"""

from __future__ import annotations

from rci_cli import launch, ssh
from rci_cli.alloc import Allocation
from rci_cli.config import Config
from rci_cli.launch import _inner_command, _tmux_wrap, resolve_folder, session_name


# ── folder resolution ──────────────────────────────────────────────────────


def test_resolve_folder_empty_returns_home(cfg: Config) -> None:
    assert resolve_folder("", cfg) == "/home/cizekto2"
    assert resolve_folder(None, cfg) == "/home/cizekto2"


def test_resolve_folder_relative_is_under_home(cfg: Config) -> None:
    assert resolve_folder("sam2rl", cfg) == "/home/cizekto2/sam2rl"


def test_resolve_folder_absolute_passthrough(cfg: Config) -> None:
    assert resolve_folder("/scratch/exp42", cfg) == "/scratch/exp42"


# ── session names ──────────────────────────────────────────────────────────


def test_session_name_home_is_special_cased(cfg: Config) -> None:
    assert session_name("claude", "/home/cizekto2", home=cfg.home) == "claude-home"
    assert session_name("shell", "/home/cizekto2/", home=cfg.home) == "shell-home"


def test_session_name_uses_folder_basename(cfg: Config) -> None:
    assert session_name("claude", "/home/cizekto2/sam2rl", home=cfg.home) == "claude-sam2rl"
    assert session_name("shell", "/scratch/exp42", home=cfg.home) == "shell-exp42"


def test_session_name_with_suffix(cfg: Config) -> None:
    assert (
        session_name("claude", "/home/cizekto2/sam2rl", "exp1", home=cfg.home)
        == "claude-sam2rl-exp1"
    )


# ── inner command ─────────────────────────────────────────────────────────


def test_inner_command_cds_sources_venv_and_execs(cfg: Config) -> None:
    s = _inner_command("/home/cizekto2/sam2rl", cfg, exec_target="claude")
    # shlex.quote skips quoting for paths with no shell-special chars.
    assert "cd /home/cizekto2/sam2rl" in s
    assert cfg.venv_activate in s
    assert "$HOME/bin:$HOME/.local/bin:$PATH" in s
    assert "exec claude" in s


def test_inner_command_quotes_path_with_spaces(cfg: Config) -> None:
    s = _inner_command("/scratch/my dir", cfg, exec_target="bash -i")
    assert "'/scratch/my dir'" in s


# ── tmux wrapper ──────────────────────────────────────────────────────────


def test_tmux_wrap_attaches_or_creates(cfg: Config) -> None:
    inner = _inner_command("/home/cizekto2", cfg, exec_target="claude")
    s = _tmux_wrap("claude-home", inner)
    # ``tmux new-session -A -s <name>`` attaches if exists, creates with cmd if not.
    assert "tmux new-session -A -s claude-home" in s
    # Inner command quoted exactly once so shlex.quote round-trips through bash -lc.
    assert "bash -lc" in s


def test_tmux_wrap_falls_back_when_tmux_missing(cfg: Config) -> None:
    inner = _inner_command("/home/cizekto2", cfg, exec_target="bash -i")
    s = _tmux_wrap("shell-home", inner)
    assert "command -v tmux" in s
    # Fallback branch runs the inner command directly so users on tmux-less nodes
    # still get a working session (just no disconnect resilience).
    assert s.count("bash -lc") == 2  # one in the tmux branch, one in the fallback


# ── launchers (mock ssh.run) ──────────────────────────────────────────────


def _patch_ssh(monkeypatch) -> dict:
    captured: dict = {}

    def fake_run(host, cmd="", *, tty=False, check=True, stdin=None):
        captured["host"] = host
        captured["cmd"] = cmd
        captured["tty"] = tty
        return 0

    monkeypatch.setattr(ssh, "run", fake_run)
    return captured


def test_launch_shell_wraps_in_tmux(monkeypatch, cfg: Config) -> None:
    captured = _patch_ssh(monkeypatch)
    rc = launch.launch_shell(Allocation(node="g05", jobid="5555"), "/home/cizekto2", cfg)
    assert rc == 0
    assert captured["host"] == "g05"
    assert captured["tty"] is True
    assert "tmux new-session -A -s shell-home" in captured["cmd"]
    assert "exec bash -i" in captured["cmd"]


def test_launch_shell_with_suffix(monkeypatch, cfg: Config) -> None:
    captured = _patch_ssh(monkeypatch)
    launch.launch_shell(
        Allocation(node="n01", jobid="1"), "/home/cizekto2/sam2rl", cfg, suffix="exp1"
    )
    assert "tmux new-session -A -s shell-sam2rl-exp1" in captured["cmd"]
