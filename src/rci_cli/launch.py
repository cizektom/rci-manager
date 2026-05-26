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

    ``folder`` is ``shlex.quote``'d so a path containing ``'`` (or any
    other shell metacharacter) doesn't break out of the cd argument.
    """
    safe_folder = shlex.quote(folder)
    return (
        f"cd {safe_folder} || {{ echo 'rci: folder not found on '\"$(hostname)\" >&2; exit 1; }}; "
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

    ``quiet=True`` so the launcher's stdio doesn't paint over the live TUI —
    VS Code on first connect prints Remote-SSH installer chatter, and cmd.exe
    occasionally prints its own status, both of which corrupt the alt screen.
    """
    print(f"→ {alloc.node} (job {alloc.jobid}): opening editor on {folder}")
    uri = f"vscode-remote://ssh-remote+{alloc.node}{folder}"
    if os.path.exists("/mnt/c"):
        return ssh.run_local(
            ["cmd.exe", "/c", "code", "--folder-uri", uri], check=False, quiet=True
        )
    return ssh.run_local(["code", "--folder-uri", uri], check=False, quiet=True)


WORKSPACE_LOG_DIR = "$HOME/.rci/workspace-logs"


def workspace_log_path(jobid: str) -> str:
    """Compute-node path of the workspace holder log for ``jobid``."""
    return f"{WORKSPACE_LOG_DIR}/{shlex.quote(jobid)}.log"


def _pane_cmd(folder: str, *, command: str = "") -> str:
    """Shell command for a tmux pane: cd + venv + PATH + optional payload +
    auto-respawning bash.

    Empty ``command`` ⇒ drops straight into the bash respawn loop. Non-empty
    runs ``command`` first, then falls back to the loop — important for
    tmux layout stability: if ``command`` exits (or isn't installed on the
    node, exit 127), the pane stays alive instead of collapsing and tmux
    re-tiling around it.

    ``while true; do bash -i; sleep 0.1; done`` instead of ``exec bash -i``:
    a plain mouse selection in some terminals (Windows Terminal especially)
    sends escape sequences that reach bash and trip an EOF-like exit — that
    would close the pane, and once every pane closes the holder loop ends
    and the whole tmux session terminates. The respawn loop hides the bash
    exit and the pane silently restarts; the ``sleep 0.1`` is a crash-loop
    guard. Intentional pane kill still works via tmux's ``prefix-x``.
    """
    pre = _remote_preamble(folder)
    loop = "while true; do bash -i; sleep 0.1; done"
    if command:
        inner = f"{pre}; {command}; {loop}"
    else:
        inner = f"{pre}; {loop}"
    return f"bash -c {shlex.quote(inner)}"


def launch_workspace(
    alloc: Allocation,
    folder: str,
    cfg: Config,
    *,
    agents: int | None = None,
    terminals: int | None = None,
) -> int:
    """Open (or reattach to) a tmux workspace on the compute node.

    Layout: ``agents`` claude panes in the top row, ``terminals`` bash
    panes in the bottom row. ``None`` for either falls back to
    ``cfg.workspace_agents`` / ``cfg.workspace_terminals``. Default 2+1
    renders as:

        +----------+----------+
        | claude 0 | claude 2 |   top row (70%): two claude shells,
        +----------+----------+   auto-launched on session creation
        |       bash 1        |   bottom (30%): bash for ad-hoc commands
        +---------------------+

    Pane indices are creation-order. With the default 2+1 layout the
    final positional indices read top-left=0, top-right=2, bottom=1 —
    visual position no longer follows creation order once horizontal
    splits land. Use Ctrl-b arrow keys to move between panes.

    Persistence model. tmux's daemon must live inside the job's cgroup
    (otherwise its panes see every GPU on the node — same trap as
    ``rci editor``). Achieved by starting the daemon inside a long-lived
    background ``srun --jobid --overlap`` step that holds the cgroup open
    via a ``while has-session; sleep`` loop. The user's interactive
    ``tmux attach`` runs as a plain ssh command — it's just a client,
    not where the panes execute, so it doesn't need srun wrapping.

    Per-job socket (``rci-ws-<jobid>``) so workspaces on different
    allocations don't share a daemon (and a wrong cgroup).

    Idempotent: subsequent calls reattach to the existing session — the
    ``agents`` / ``terminals`` knobs only apply when the session is being
    built. Closing every pane exits the session, which exits the holder
    loop, which ends the slurm step — next ``w`` builds fresh.
    """
    if agents is None:
        agents = cfg.workspace_agents
    if terminals is None:
        terminals = cfg.workspace_terminals
    agents = max(0, int(agents))
    terminals = max(0, int(terminals))
    # The session needs at least one pane to hold the cgroup open. If the
    # caller asked for nothing, fall back to a single bash so the user
    # isn't staring at an empty tmux.
    if agents == 0 and terminals == 0:
        terminals = 1

    sock = f"rci-ws-{alloc.jobid}"
    sess = "main"
    sock_q = shlex.quote(sock)
    sess_q = shlex.quote(sess)
    jid_q = shlex.quote(alloc.jobid)
    folder_q = shlex.quote(folder)
    claude_pane = shlex.quote(_pane_cmd(folder, command="claude"))
    bash_pane = shlex.quote(_pane_cmd(folder))

    # Inner script runs inside the long-lived srun step. Builds the panes
    # then idles in the has-session poll loop — the loop is the cgroup
    # holder. When the user kills the last pane the loop returns false
    # and srun exits cleanly.
    #
    # Layout sequence (general case for N agents + M terminals):
    #
    #   1) new-session with first pane (claude if agents>0 else bash) —
    #      creation-order pane 0.
    #   2) if both rows are populated: split -v -p 30 with bash creates the
    #      bottom row (pane 1, 30% tall). Skipped when only one row exists.
    #   3) for each additional agent: split -h -t {.0|.bottom_top} with claude
    #      to add another column to the top row.
    #   4) for each additional terminal: split -h -t {.0|.bottom_first} with
    #      bash to add another column to the bottom row.
    #   5) select-layout -E on each multi-pane row distributes width evenly
    #      among siblings without disturbing the other row.
    #
    # Panes are born with their initial command — no post-hoc ``send-keys``,
    # since tmux renumbers panes by visual position after splits and a
    # literal ``-t .N`` would otherwise hit the wrong cell.
    inner_lines: list[str] = []
    first_pane = claude_pane if agents > 0 else bash_pane
    inner_lines.append(
        f"tmux -L {sock_q} new-session -d -s {sess_q} -c {folder_q} {first_pane}"
    )

    if agents > 0 and terminals > 0:
        # Vertical split for the bottom (terminals) row. ``-p 30`` matches
        # the historic 70/30 top:bottom ratio; tunable in a follow-up if
        # we ever want it config-driven.
        inner_lines.append(
            f"tmux -L {sock_q} split-window -v -p 30 -t {sess_q} -c {folder_q} {bash_pane}"
        )
        bottom_first = "1"  # creation-order index of the first terminal
    else:
        # Single-row workspace: the first (and only) row starts at pane 0;
        # no "bottom row" exists in this layout.
        bottom_first = None

    # First pane is at .0 regardless of agents vs terminals. Extra agents
    # split that pane; extra terminals split bottom_first when present,
    # else .0 (single-row terminals-only layout).
    top_first = "0"

    # Add remaining agents to the top row.
    for _ in range(max(0, agents - 1)):
        inner_lines.append(
            f"tmux -L {sock_q} split-window -h -t {sess_q}.{top_first} "
            f"-c {folder_q} {claude_pane}"
        )
    if agents >= 2:
        # Even the agent row in one shot — ``select-layout -E`` spreads
        # the target pane and its siblings under the same parent split,
        # so the other row's widths stay untouched.
        inner_lines.append(
            f"tmux -L {sock_q} select-layout -E -t {sess_q}.{top_first}"
        )

    # Add remaining terminals to the bottom row (or the only row when
    # agents=0).
    terminal_anchor = bottom_first if bottom_first is not None else top_first
    if terminals >= 1:
        for _ in range(max(0, terminals - 1)):
            inner_lines.append(
                f"tmux -L {sock_q} split-window -h -t {sess_q}.{terminal_anchor} "
                f"-c {folder_q} {bash_pane}"
            )
        if terminals >= 2:
            inner_lines.append(
                f"tmux -L {sock_q} select-layout -E -t {sess_q}.{terminal_anchor}"
            )

    # Focus pane 0 — first agent if present, else first terminal.
    inner_lines.append(f"tmux -L {sock_q} select-pane -t {sess_q}.{top_first}")
    inner_lines.append(
        f"while tmux -L {sock_q} has-session -t {sess_q} 2>/dev/null; do sleep 30; done"
    )
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
