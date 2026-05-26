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
    without a venv get a no-op via the ``[ -f ... ]`` guard.

    ``cd`` uses ``shlex.quote`` rather than bare single quotes; for plain
    paths that's a no-op (``shlex.quote('/home/alice/proj')`` returns the
    string unchanged), which is what we assert here. See
    :func:`test_remote_preamble_shell_quotes_pathological_folder` for the
    quoting kicking in."""
    s = _remote_preamble("/home/alice/proj")
    assert "cd /home/alice/proj " in s
    assert "[ -f .venv/bin/activate ] && . .venv/bin/activate" in s
    assert "$HOME/bin:$HOME/.local/bin:$PATH" in s
    # Venv path is always relative to cwd — no $HOME/... prefix.
    assert "$HOME/" not in s.split("PATH=")[0]


def test_remote_preamble_shell_quotes_pathological_folder() -> None:
    """A folder containing ``'`` (or any other shell metachar) must not break
    out of the cd argument — regression for the prior raw f-string interpolation."""
    s = _remote_preamble("/scratch/foo'bar")
    # ``shlex.quote`` produces ``'/scratch/foo'"'"'bar'`` — verify the cd line
    # parses as a single argument by checking the closing-then-reopening pattern.
    assert "cd '/scratch/foo'\"'\"'bar' " in s


# ── launchers (mock ssh.run / ssh.run_local) ───────────────────────────────


def _patch_ssh(monkeypatch) -> dict:
    captured: dict = {}

    def fake_run(host, cmd="", *, tty=False, check=True, stdin=None):
        captured["host"] = host
        captured["cmd"] = cmd
        captured["tty"] = tty
        return 0

    def fake_run_local(argv, *, check=True, quiet=False):
        captured["local_argv"] = list(argv)
        captured["local_quiet"] = quiet
        return 0

    monkeypatch.setattr(ssh, "run", fake_run)
    monkeypatch.setattr(ssh, "run_local", fake_run_local)
    return captured


def test_launch_shell_wraps_with_srun_jobid_overlap(monkeypatch, cfg: Config) -> None:
    """The shell must join the job's step via ``srun --jobid --overlap --pty``
    so GPU cgroup + env (``CUDA_VISIBLE_DEVICES``) apply. Without the wrap
    the user lands outside the allocation and sees every GPU on the node."""
    captured = _patch_ssh(monkeypatch)
    rc = launch.launch_shell(Allocation(node="g05", jobid="5555"), "/home/cizekto2", cfg)
    assert rc == 0
    assert captured["host"] == "g05"
    assert captured["tty"] is True  # need -tt so srun --pty has a PTY
    cmd = captured["cmd"]
    assert "exec srun --jobid=5555 --overlap --pty bash -i" in cmd
    # Bare ``bash -i`` (without srun) would be a regression.
    assert "exec bash -i" not in cmd


def test_launch_editor_uses_vscode_remote_uri(monkeypatch, cfg: Config) -> None:
    captured = _patch_ssh(monkeypatch)
    launch.launch_editor(Allocation(node="n01", jobid="1234"), "/home/cizekto2/sam2rl", cfg)
    argv = captured["local_argv"]
    # Either ["code", "--folder-uri", URI] or ["cmd.exe", "/c", "code", "--folder-uri", URI]
    assert any("vscode-remote://ssh-remote+n01/home/cizekto2/sam2rl" in a for a in argv)
    # ``quiet=True`` — VS Code / cmd.exe stdio would otherwise paint over
    # the live TUI's alt screen on first connect.
    assert captured["local_quiet"] is True


def test_run_local_quiet_sends_stdio_to_devnull(monkeypatch) -> None:
    """``quiet=True`` must detach stdin/stdout/stderr from the parent TTY —
    same fix lineage as ssh.capture's stdin=DEVNULL."""
    import subprocess
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    ssh.run_local(["echo", "hi"], quiet=True)
    assert seen["kwargs"]["stdin"] is subprocess.DEVNULL
    assert seen["kwargs"]["stdout"] is subprocess.DEVNULL
    assert seen["kwargs"]["stderr"] is subprocess.DEVNULL


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
    assert "cd /home/cizekto2/sam2rl " in cmd
    assert "exec claude" not in cmd  # would replace the shell, blocking ssh
    # Wrapped with srun so claude runs as a step of the allocation —
    # gets the right GPUs and lives in the job's cgroup.
    assert "nohup srun --jobid=5555 --overlap claude remote-control" in cmd
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


# ── workspace launcher ─────────────────────────────────────────────────────


def _patch_ssh_multi(monkeypatch) -> list[dict]:
    """ssh.run capturer that records every call (launch_workspace makes two)."""
    calls: list[dict] = []

    def fake_run(host, cmd="", *, tty=False, check=True, stdin=None):
        calls.append({"host": host, "cmd": cmd, "tty": tty, "stdin": stdin})
        return 0

    monkeypatch.setattr(ssh, "run", fake_run)
    return calls


def test_launch_workspace_setup_and_attach_phases(monkeypatch, cfg: Config) -> None:
    """The two-phase shape: non-tty bootstrap over stdin, then tty attach.

    Phase 1 ships the tmux setup as a here-doc to ``bash -s`` (no terminal
    handover yet — we're just spawning the holder srun and waiting for the
    socket). Phase 2 is the foreground ``tmux attach`` that takes over the
    user's terminal.
    """
    calls = _patch_ssh_multi(monkeypatch)
    rc = launch.launch_workspace(
        Allocation(node="g05", jobid="5555"), "/home/cizekto2/sam2rl", cfg
    )
    assert rc == 0
    assert len(calls) == 2

    setup, attach = calls
    # Setup: non-interactive, script over stdin.
    assert setup["host"] == "g05"
    assert setup["tty"] is False
    assert setup["cmd"] == "bash -s"
    assert isinstance(setup["stdin"], str) and setup["stdin"]

    # Attach: foreground tmux client on the same per-job socket.
    assert attach["host"] == "g05"
    assert attach["tty"] is True
    assert attach["cmd"] == "tmux -L rci-ws-5555 attach -t main"
    assert attach["stdin"] is None


def test_launch_workspace_disables_destroy_unattached(monkeypatch, cfg: Config) -> None:
    """Our session must override ``destroy-unattached`` to off: workspace
    lifetime is the holder srun's job, not the client's. A stray detach
    sequence (e.g. emitted by some terminals on mouse-selection release)
    would otherwise destroy the session, exit-empty would take the
    server down, and every pane would vanish at once."""
    calls = _patch_ssh_multi(monkeypatch)
    launch.launch_workspace(
        Allocation(node="g05", jobid="5555"), "/home/cizekto2/sam2rl", cfg
    )
    setup_script = calls[0]["stdin"]
    assert "set-option -t main destroy-unattached off" in setup_script
    # We start the server with ``-f /dev/null`` so the user's prefix binding
    # from ~/.tmux.conf is lost; restore Ctrl-Space explicitly.
    assert "set-option -g prefix C-Space" in setup_script
    # Mouse scrolling: enable ``mouse on`` for wheel scrollback, but unbind
    # the drag handlers — the default ``copy-selection-and-cancel`` path
    # crashes the whole session on this cluster's tmux build.
    assert "set-option -g mouse on" in setup_script
    assert "unbind-key -T root MouseDrag1Pane" in setup_script
    assert "unbind-key -T copy-mode MouseDragEnd1Pane" in setup_script
    assert "unbind-key -T copy-mode-vi MouseDragEnd1Pane" in setup_script


def test_launch_workspace_holder_keeps_cgroup_alive(monkeypatch, cfg: Config) -> None:
    """The tmux daemon must be forked inside ``srun --jobid --overlap`` and
    the step must be held open while the session exists — otherwise slurm
    cleans up the cgroup and the workspace's GPU panes see the wrong GPUs."""
    calls = _patch_ssh_multi(monkeypatch)
    launch.launch_workspace(
        Allocation(node="g05", jobid="5555"), "/home/cizekto2/sam2rl", cfg
    )
    setup_script = calls[0]["stdin"]
    # Holder is detached so the bootstrap ssh returns promptly; the holder
    # itself stays alive on the compute node, owned by init.
    assert "nohup srun --jobid=5555 --overlap" in setup_script
    assert "& disown" in setup_script
    # The hold loop is the cgroup keepalive: srun's primary process must
    # outlive the tmux setup, otherwise slurm tears down the step.
    assert "while tmux -L rci-ws-5555 has-session" in setup_script
    assert "sleep 30" in setup_script
    # Re-runs are no-ops: the has-session guard makes the bootstrap idempotent.
    # (The outer guard uses ``$SOCK`` / ``$SESS`` shell vars assigned at the
    # top of the script — see the SOCK=…/SESS=… header lines.)
    assert "SOCK=rci-ws-5555" in setup_script
    assert "SESS=main" in setup_script
    assert 'if ! tmux -L "$SOCK" has-session -t "$SESS"' in setup_script


def test_launch_workspace_three_pane_layout(monkeypatch, cfg: Config) -> None:
    """Default layout: two claude shells on top, bash on the bottom.

    Each pane is born with its right initial command — we don't post-hoc
    target panes via ``send-keys``, because tmux re-indexes panes by visual
    position after every split (creation-order ≠ final-index, so a literal
    ``-t main.2`` after the third split actually hits the bottom pane,
    not the new top-right one).

    The second split must target ``main.0`` explicitly; without ``-t .0``
    it would split the most-recently-focused pane (pane 1, the bash row),
    yielding an L-shape.
    """
    calls = _patch_ssh_multi(monkeypatch)
    launch.launch_workspace(
        Allocation(node="g05", jobid="5555"), "/home/cizekto2/sam2rl", cfg
    )
    setup_script = calls[0]["stdin"]
    # 1) Initial pane starts as claude (becomes top-left after splits).
    #    ``-f /dev/null`` keeps the workspace tmux server isolated from
    #    user/system tmux configs (one of which kills tmux on mouse
    #    selection in this environment).
    assert "-f /dev/null new-session -d -s main -c /home/cizekto2/sam2rl" in setup_script
    # 2) Bash row at the bottom, 30% tall — splits the whole window.
    assert "split-window -v -p 30 -t main -c /home/cizekto2/sam2rl" in setup_script
    # 3) Halve the top by splitting pane 0 specifically; new pane starts
    #    as claude (the second top-row claude). The post-split
    #    ``select-layout -E`` rebalances widths so the two claudes share
    #    the row evenly.
    assert "split-window -h -t main.0 -c /home/cizekto2/sam2rl" in setup_script
    assert "select-layout -E -t main.0" in setup_script
    # Initial command of each pane is the right thing: two claude payloads
    # (top row) and one bash payload (bottom). ``while true; do bash -i``
    # markers from _pane_cmd's respawn loop appear inside the shlex-quoted
    # pane commands embedded in the setup script — one per pane.
    assert setup_script.count("claude; while true; do bash -i") == 2
    assert setup_script.count("while true; do bash -i; sleep 0.1; done") == 3
    # ``exec bash -i`` from the old, pane-dies-on-bash-exit wrapper must not
    # come back — accidental selection in Windows Terminal trips EOF on bash
    # and would otherwise cascade pane closures into a dead session.
    assert "exec bash -i" not in setup_script
    # Defensive: no send-keys anywhere — that's how we got the L-shape.
    assert "send-keys" not in setup_script
    # No GPU watcher pane in this layout.
    assert "nvidia-smi" not in setup_script
    assert "CUDA_VISIBLE_DEVICES" not in setup_script


def test_launch_workspace_per_job_socket(monkeypatch, cfg: Config) -> None:
    """Per-job tmux socket — two workspaces on different allocs must not
    share a daemon (sharing one would put panes in the wrong cgroup)."""
    calls = _patch_ssh_multi(monkeypatch)
    launch.launch_workspace(Allocation(node="g05", jobid="111"), "/tmp", cfg)
    launch.launch_workspace(Allocation(node="g06", jobid="222"), "/tmp", cfg)
    setup_a, _attach_a, setup_b, _attach_b = calls
    assert "tmux -L rci-ws-111" in setup_a["stdin"]
    assert "tmux -L rci-ws-222" in setup_b["stdin"]
    # Sanity: a setup script must not leak the *other* job's socket.
    assert "rci-ws-222" not in setup_a["stdin"]
    assert "rci-ws-111" not in setup_b["stdin"]


def test_workspace_log_path_namespaces_per_jobid() -> None:
    assert launch.workspace_log_path("5555") == "$HOME/.rci/workspace-logs/5555.log"


def test_launch_workspace_respects_explicit_pane_counts(
    monkeypatch, cfg: Config
) -> None:
    """``agents=3, terminals=2`` ⇒ three claude panes on top, two bash on
    bottom. Each row gets a ``select-layout -E`` to spread its members
    evenly without disturbing the other row."""
    calls = _patch_ssh_multi(monkeypatch)
    launch.launch_workspace(
        Allocation(node="g05", jobid="5555"),
        "/home/cizekto2/sam2rl",
        cfg,
        agents=3,
        terminals=2,
    )
    setup_script = calls[0]["stdin"]
    # 3 claude pane payloads + 2 bash payloads = 5 respawn-loop markers.
    assert setup_script.count("claude; while true; do bash -i") == 3
    assert setup_script.count("while true; do bash -i; sleep 0.1; done") == 5
    # First terminal lives at creation-order pane 1 (vertical split). The
    # second terminal splits pane 1 horizontally.
    assert "split-window -v -p 30 -t main -c" in setup_script
    assert setup_script.count("split-window -h -t main.0") == 2  # two extra agents
    assert setup_script.count("split-window -h -t main.1") == 1  # one extra terminal
    # Both rows get evened.
    assert "select-layout -E -t main.0" in setup_script
    assert "select-layout -E -t main.1" in setup_script


def test_launch_workspace_terminals_only_layout(monkeypatch, cfg: Config) -> None:
    """``agents=0, terminals=2`` ⇒ single row of bash panes, no vertical
    split, no claude payload, no top-row select-layout."""
    calls = _patch_ssh_multi(monkeypatch)
    launch.launch_workspace(
        Allocation(node="g05", jobid="5555"),
        "/home/cizekto2/sam2rl",
        cfg,
        agents=0,
        terminals=2,
    )
    setup_script = calls[0]["stdin"]
    assert "claude" not in setup_script
    # Only bash panes (2 of them).
    assert setup_script.count("while true; do bash -i; sleep 0.1; done") == 2
    # No top/bottom split.
    assert "split-window -v" not in setup_script
    # Second terminal splits the first horizontally (pane 0 is the only pane).
    assert "split-window -h -t main.0" in setup_script
    assert "select-layout -E -t main.0" in setup_script


def test_launch_workspace_agents_only_layout(monkeypatch, cfg: Config) -> None:
    """``agents=2, terminals=0`` ⇒ single row of claude panes, no bash, no
    vertical split."""
    calls = _patch_ssh_multi(monkeypatch)
    launch.launch_workspace(
        Allocation(node="g05", jobid="5555"),
        "/home/cizekto2/sam2rl",
        cfg,
        agents=2,
        terminals=0,
    )
    setup_script = calls[0]["stdin"]
    assert setup_script.count("claude; while true; do bash -i") == 2
    # 2 claude panes, no bare-bash pane.
    assert setup_script.count("while true; do bash -i; sleep 0.1; done") == 2
    assert "split-window -v" not in setup_script
    assert "split-window -h -t main.0" in setup_script


def test_launch_workspace_single_agent_single_terminal(
    monkeypatch, cfg: Config
) -> None:
    """``agents=1, terminals=1`` ⇒ one claude on top + one bash on bottom,
    no extra horizontal splits, no select-layout calls (nothing to even out)."""
    calls = _patch_ssh_multi(monkeypatch)
    launch.launch_workspace(
        Allocation(node="g05", jobid="5555"),
        "/home/cizekto2/sam2rl",
        cfg,
        agents=1,
        terminals=1,
    )
    setup_script = calls[0]["stdin"]
    assert setup_script.count("claude; while true; do bash -i") == 1
    assert setup_script.count("while true; do bash -i; sleep 0.1; done") == 2
    assert "split-window -v -p 30 -t main -c" in setup_script
    assert "split-window -h" not in setup_script
    assert "select-layout -E" not in setup_script


def test_launch_workspace_falls_back_to_cfg_defaults(
    monkeypatch, cfg: Config
) -> None:
    """``agents=None, terminals=None`` ⇒ use ``cfg.workspace_agents`` /
    ``cfg.workspace_terminals``. Confirms the cfg-driven path matches the
    historical default (2 agents + 1 terminal)."""
    calls = _patch_ssh_multi(monkeypatch)
    launch.launch_workspace(
        Allocation(node="g05", jobid="5555"),
        "/home/cizekto2/sam2rl",
        cfg,
    )
    setup_script = calls[0]["stdin"]
    assert setup_script.count("claude; while true; do bash -i") == 2
    assert setup_script.count("while true; do bash -i; sleep 0.1; done") == 3
