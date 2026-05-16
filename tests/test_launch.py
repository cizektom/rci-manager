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


def test_resolve_folder_uses_effective_home_fallback() -> None:
    """When ``home`` is empty but ``user`` is set, ``/home/<user>`` is implied
    — relative paths still resolve against that fallback."""
    cfg = Config(user="alice")  # no explicit home
    assert resolve_folder("", cfg) == "/home/alice"
    assert resolve_folder("proj", cfg) == "/home/alice/proj"


# ── remote preamble ────────────────────────────────────────────────────────


def test_remote_preamble_cds_and_tries_relative_venv() -> None:
    """After ``cd <folder>``, the preamble always tries to source
    ``.venv/bin/activate`` — relative, so it adapts per-project. Folders
    without a venv get a no-op via the ``[ -f ... ]`` guard."""
    s = _remote_preamble("/home/alice/proj")
    assert "cd '/home/alice/proj'" in s
    assert "[ -f .venv/bin/activate ] && . .venv/bin/activate" in s
    assert "$HOME/bin:$HOME/.local/bin:$PATH" in s


def test_remote_preamble_does_not_reference_absolute_venv() -> None:
    """Regression: the venv path is always relative (``.venv/bin/activate``)
    — no ``$HOME/...`` prefix that would tie it to one project."""
    s = _remote_preamble("/home/alice")
    assert "$HOME/" not in s.split("PATH=")[0]  # only the PATH line uses $HOME


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


def test_launch_agent_runs_detached_with_nohup(monkeypatch, cfg: Config) -> None:
    """``launch_agent`` must background claude remote-control so the TUI / CLI
    returns immediately. Pattern: ``nohup … >>log 2>&1 </dev/null & disown``,
    no tty, no ``exec`` (which would replace the shell instead of
    backgrounding)."""
    captured = _patch_ssh(monkeypatch)
    rc = launch.launch_agent(
        Allocation(node="g05", jobid="5555"),
        "/home/cizekto2/sam2rl",
        cfg,
        name="agent-2",
        permission_mode="bypassPermissions",
        spawn_mode="worktree",
        capacity=16,
    )
    assert rc == 0
    assert captured["host"] == "g05"
    assert captured["tty"] is False  # detached — no terminal handover
    cmd = captured["cmd"]
    assert "cd '/home/cizekto2/sam2rl'" in cmd
    assert "exec claude" not in cmd  # would replace the shell, blocking ssh
    assert "nohup claude remote-control" in cmd
    assert "& disown" in cmd
    assert "</dev/null" in cmd
    assert ">>$HOME/.rci/agent-logs/agent-2.log 2>&1" in cmd
    assert "mkdir -p $HOME/.rci/agent-logs" in cmd
    # All four claude flags still flow through verbatim.
    assert "--name agent-2" in cmd
    assert "--permission-mode bypassPermissions" in cmd
    assert "--spawn worktree" in cmd
    assert "--capacity 16" in cmd


def test_launch_agent_shell_quotes_dynamic_strings(monkeypatch, cfg: Config) -> None:
    """Free-text ``name`` must be shell-escaped — a name with a space can't
    spill into the next flag and a name with a quote can't terminate it."""
    captured = _patch_ssh(monkeypatch)
    launch.launch_agent(
        Allocation(node="n01", jobid="1"),
        "/tmp",
        cfg,
        name="hello world",
        permission_mode="default",
        spawn_mode="same-dir",
        capacity=32,
    )
    # ``shlex.quote('hello world')`` → ``'hello world'`` (single-quoted).
    assert "--name 'hello world'" in captured["cmd"]


def test_agent_log_path_namespaces_per_name() -> None:
    """Each agent gets its own log file under ``~/.rci/agent-logs/``."""
    assert launch.agent_log_path("agent-1") == "$HOME/.rci/agent-logs/agent-1.log"
    # Spaces in names get shell-quoted; ``.log`` lives outside the quote and
    # the shell concatenates them into one filename.
    assert launch.agent_log_path("my run") == "$HOME/.rci/agent-logs/'my run'.log"
