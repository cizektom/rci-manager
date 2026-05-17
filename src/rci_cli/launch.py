"""Launch interactive tools on a compute node — shell and VS Code Remote-SSH.

Compute nodes (``n*``/``g*``) don't have zsh installed and their bash doesn't
replicate the login-node setup, so we ``cd`` into the working folder and try
to source a ``.venv/bin/activate`` relative to it — silently skipping if the
project doesn't have a venv there. ``$HOME/bin:$HOME/.local/bin`` is
prepended to PATH the same way every shell-init file does it.

No persistence layer at this level — ssh disconnect ends the session. Wrap
inside the spawned bash with ``tmux`` / ``screen`` if you need survival.
"""

from __future__ import annotations

import os
import shlex

from . import ssh
from .alloc import Allocation
from .config import Config


def resolve_folder(folder: str | None, cfg: Config) -> str:
    """Folder rules shared by shell / editor launchers.

    - empty → ``cfg.effective_home``
    - relative → resolved under ``cfg.effective_home``
    - absolute → returned as-is
    """
    home = cfg.effective_home
    if not folder:
        return home
    if folder.startswith("/"):
        return folder
    return f"{home}/{folder}"


def _remote_preamble(folder: str) -> str:
    """Build the ``cd … && [source .venv] && PATH=…`` prefix for compute-node commands.

    The venv source is conditional on a ``.venv/bin/activate`` existing
    *relative to the folder we just cd'd into* — projects without one (or
    that manage activation elsewhere) get a no-op.
    """
    return (
        f"cd '{folder}' || {{ echo 'rci: {folder} not found on '\"$(hostname)\" >&2; exit 1; }}; "
        f"[ -f .venv/bin/activate ] && . .venv/bin/activate; "
        f'PATH="$HOME/bin:$HOME/.local/bin:$PATH"'
    )


def launch_shell(alloc: Allocation, folder: str, cfg: Config) -> int:
    """Open an interactive bash inside the job's allocation.

    Wrapped with ``srun --jobid=<jobid> --overlap --pty`` so the shell
    joins the job's step and inherits its cgroup + environment
    (``CUDA_VISIBLE_DEVICES``, memory limits, …). A bare
    ``ssh node bash`` would land outside the allocation — fine on
    clusters with ``pam_slurm_adopt`` enforcing GPU cgroups, but on
    sites like RCI it'd expose every GPU on the node instead of just
    the ones you requested.

    ``clear`` runs after the preamble so the MOTD / ``Last login`` noise
    from the ssh hop is wiped before bash takes over — equivalent to
    hitting Ctrl-L the moment the connection lands. Scrollback is
    preserved by ``clear`` (it uses ``\\033[H\\033[2J``, not ``[3J``).
    """
    # ``export PATH`` is needed because the preamble's trailing ``PATH=…``
    # is now a standalone assignment (no longer inline-prefixing ``exec
    # srun``), so without exporting it srun's bash wouldn't inherit the
    # ``$HOME/bin:$HOME/.local/bin`` additions.
    cmd = (
        f"{_remote_preamble(folder)}; "
        f"export PATH; "
        f"clear; "
        f"exec srun --jobid={alloc.jobid} --overlap --pty bash -i"
    )
    return ssh.run(alloc.node, cmd, tty=True, check=False)


AGENT_LOG_DIR = "$HOME/.rci/agent-logs"


def agent_log_path(name: str) -> str:
    """Compute-node path of the log for an agent named ``name``.

    String is shell-safe (the dynamic part is shlex-quoted) but contains
    ``$HOME`` — interpret it by running it through ssh, not by os.path.
    """
    return f"{AGENT_LOG_DIR}/{shlex.quote(name)}.log"


def launch_agent(
    alloc: Allocation,
    folder: str,
    cfg: Config,
    *,
    name: str,
    permission_mode: str,
    spawn_mode: str,
    capacity: int,
) -> int:
    """Start ``claude remote-control`` detached on the compute node.

    The launch returns as soon as ssh posts the background command — the
    remote process keeps running via ``nohup … & disown`` and survives the
    ssh disconnect, so the CLI / TUI returns to the foreground immediately.
    Pair the session from claude.ai/code or the mobile app (no terminal
    output needed once you're signed in there).

    Stdout/stderr append to ``~/.rci/agent-logs/<name>.log`` on the
    compute node — peek there if you ever need to see startup messages.
    """
    safe_name = shlex.quote(name)
    log_path = agent_log_path(name)
    flags = (
        f"--name {safe_name} "
        f"--permission-mode {shlex.quote(permission_mode)} "
        f"--spawn {shlex.quote(spawn_mode)} "
        f"--capacity {int(capacity)}"
    )
    # The remote shell must: cd / source venv / set PATH (preamble),
    # ensure the log dir exists, then background claude with stdio
    # detached from ssh so ssh disconnects right away. The claude call
    # itself is wrapped with ``srun --jobid=<jobid> --overlap`` so it
    # runs as a step of the allocation — gets the right GPUs, lives
    # inside the job's cgroup, and Slurm tracks its lifecycle.
    cmd = (
        f"{_remote_preamble(folder)}; "
        f"mkdir -p {AGENT_LOG_DIR} && "
        f"nohup srun --jobid={alloc.jobid} --overlap "
        f"claude remote-control {flags} "
        f">>{log_path} 2>&1 </dev/null & disown"
    )
    print(f"→ {alloc.node} (job {alloc.jobid}): agent '{name}' launched in background")
    print(f"  log: {alloc.node}:{log_path}")
    return ssh.run(alloc.node, cmd, tty=False, check=False)


def launch_editor(alloc: Allocation, folder: str, cfg: Config) -> int:
    """Open VS Code Remote-SSH against a compute node.

    Named ``editor`` (not ``code``) so we're not painted into a single-IDE corner;
    routing logic stays VS-Code-specific for now. On WSL the local ``code``
    wrapper auto-injects ``--remote wsl+<distro>`` which hijacks the Remote-SSH
    connection — invoke the Windows ``code.cmd`` via ``cmd.exe`` instead.
    """
    print(f"→ {alloc.node} (job {alloc.jobid}): opening editor on {folder}")
    uri = f"vscode-remote://ssh-remote+{alloc.node}{folder}"
    if os.path.exists("/mnt/c"):
        return ssh.run_local(["cmd.exe", "/c", "code", "--folder-uri", uri], check=False)
    return ssh.run_local(["code", "--folder-uri", uri], check=False)


WORKSPACE_LOG_DIR = "$HOME/.rci/workspace-logs"


def workspace_log_path(jobid: str) -> str:
    """Compute-node path of the workspace holder log for ``jobid``."""
    return f"{WORKSPACE_LOG_DIR}/{shlex.quote(jobid)}.log"


def _pane_cmd(folder: str, *, command: str = "") -> str:
    """Shell command for a tmux pane: cd + venv + PATH + optional payload + bash.

    Empty ``command`` ⇒ drops straight into ``exec bash -i``. Non-empty
    runs ``command`` first, then *falls back* to interactive bash —
    important for tmux layout stability: if ``command`` exits (or isn't
    installed on the node, exit 127), the pane stays alive instead of
    collapsing and tmux re-tiling around it.
    """
    pre = _remote_preamble(folder)
    if command:
        inner = f"{pre}; {command}; exec bash -i"
    else:
        inner = f"{pre}; exec bash -i"
    return f"bash -c {shlex.quote(inner)}"


def launch_workspace(alloc: Allocation, folder: str, cfg: Config) -> int:
    """Open (or reattach to) a tmux workspace on the compute node.

    Layout (3 panes):

        +----------+----------+
        | claude 0 | claude 2 |   top row (70%): two claude shells,
        +----------+----------+   auto-launched on session creation
        |       bash 1        |   bottom (30%): bash for ad-hoc commands
        +---------------------+

    Pane indices are creation-order (0 first, then 1 from the vertical
    split, then 2 from the horizontal split of pane 0) — visually that
    reads top-left=0, top-right=2, bottom=1. Use Ctrl-b arrow keys to
    move between them.

    Persistence model. tmux's daemon must live inside the job's cgroup
    (otherwise its panes see every GPU on the node — same trap as
    ``rci editor``). Achieved by starting the daemon inside a long-lived
    background ``srun --jobid --overlap`` step that holds the cgroup open
    via a ``while has-session; sleep`` loop. The user's interactive
    ``tmux attach`` runs as a plain ssh command — it's just a client,
    not where the panes execute, so it doesn't need srun wrapping.

    Per-job socket (``rci-ws-<jobid>``) so workspaces on different
    allocations don't share a daemon (and a wrong cgroup).

    Idempotent: subsequent calls reattach to the existing session, so
    ``w`` after a disconnect is instant. Closing every pane exits the
    session, which exits the holder loop, which ends the slurm step —
    next ``w`` builds fresh.
    """
    sock = f"rci-ws-{alloc.jobid}"
    sess = "main"
    sock_q = shlex.quote(sock)
    sess_q = shlex.quote(sess)
    jid_q = shlex.quote(alloc.jobid)
    folder_q = shlex.quote(folder)
    claude_pane = shlex.quote(_pane_cmd(folder, command="claude"))
    bash_pane = shlex.quote(_pane_cmd(folder))

    # Inner script runs inside the long-lived srun step. Builds the 3-pane
    # layout, then idles in the has-session poll loop — the loop is the
    # cgroup holder. When the user kills the last pane the loop returns
    # false and srun exits cleanly.
    #
    # Layout sequence — each pane is born with its right initial command,
    # so we don't rely on post-hoc ``send-keys`` to target pane indices
    # (tmux renumbers panes by visual position after every split, so
    # creation-order ≠ final-index — easy to send keys to the wrong pane):
    #
    #   1) new-session  with claude → pane (whole window, runs claude)
    #   2) split -v -p 30 with bash → bash row below (30% tall)
    #   3) split -h -p 50 -t .0 with claude → second claude in top-right
    #
    # After step 3 the final positional indices are: 0=top-left,
    # 1=top-right, 2=bottom. select-pane -t .0 focuses the left claude.
    inner_lines = [
        f"tmux -L {sock_q} new-session -d -s {sess_q} -c {folder_q} {claude_pane}",
        f"tmux -L {sock_q} split-window -v -p 30 -t {sess_q} -c {folder_q} {bash_pane}",
        f"tmux -L {sock_q} split-window -h -p 50 -t {sess_q}.0 -c {folder_q} {claude_pane}",
        f"tmux -L {sock_q} select-pane -t {sess_q}.0",
        f"while tmux -L {sock_q} has-session -t {sess_q} 2>/dev/null; do sleep 30; done",
    ]
    inner_q = shlex.quote(" && ".join(inner_lines))

    log_path = workspace_log_path(alloc.jobid)
    setup_script = (
        "set -e\n"
        f"SOCK={sock_q}\n"
        f"SESS={sess_q}\n"
        f"mkdir -p {WORKSPACE_LOG_DIR}\n"
        'if ! tmux -L "$SOCK" has-session -t "$SESS" 2>/dev/null; then\n'
        f"  nohup srun --jobid={jid_q} --overlap --quiet bash -c {inner_q} "
        f">>{log_path} 2>&1 </dev/null & disown\n"
        # Poll briefly for the daemon to come up before we try to attach.
        # tmux daemon-start is sub-second on healthy nodes; bail at ~3s.
        "  for _ in 1 2 3 4 5 6 7 8 9 10; do\n"
        "    sleep 0.3\n"
        '    tmux -L "$SOCK" has-session -t "$SESS" 2>/dev/null && break\n'
        "  done\n"
        "fi\n"
    )

    print(f"→ {alloc.node} (job {alloc.jobid}): opening workspace in {folder}")
    rc = ssh.run(alloc.node, "bash -s", tty=False, check=False, stdin=setup_script)
    if rc != 0:
        return rc
    # Attach as a plain ssh client — the daemon (forked from the holder
    # srun) is already cgroup-correct, so this client doesn't need srun.
    return ssh.run(alloc.node, f"tmux -L {sock_q} attach -t {sess_q}", tty=True, check=False)
