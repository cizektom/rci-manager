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
    """
    print(f"→ {alloc.node} (job {alloc.jobid}): opening shell in {folder}")
    cmd = (
        f"{_remote_preamble(folder)} "
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
