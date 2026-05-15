"""Launch interactive tools on a compute node — claude, VS Code Remote-SSH, plain shell.

Compute nodes (``n*``/``g*``) don't have zsh installed and their bash doesn't
replicate the login-node setup, so we manually source the sam2rl venv and
prepend ``$HOME/bin:$HOME/.local/bin`` to PATH inside each ssh invocation.

claude/shell launches are wrapped in a named tmux session on the compute node
so ssh disconnects don't kill the work. Re-running the same command (same
folder, same optional suffix) attaches back to the running session. If tmux
isn't installed on the node we fall back to running the command directly and
warn the user.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from . import ssh
from .alloc import Allocation
from .config import Config


def resolve_folder(folder: str | None, cfg: Config) -> str:
    """Folder rules shared by claude/code/shell launchers.

    - empty → ``cfg.home``
    - relative → resolved under ``cfg.home``
    - absolute → returned as-is
    """
    if not folder:
        return cfg.home
    if folder.startswith("/"):
        return folder
    return f"{cfg.home}/{folder}"


def session_name(kind: str, folder_abs: str, suffix: str = "", *, home: str) -> str:
    """Build a tmux session name: ``<kind>-<basename>[-<suffix>]``.

    The user's home dir is special-cased to ``home`` so a bare ``rci claude``
    doesn't produce ``claude-cizekto2``. Suffix lets you run parallel sessions
    on the same folder (``rci claude sam2rl exp1`` → ``claude-sam2rl-exp1``).
    """
    if folder_abs.rstrip("/") == home.rstrip("/"):
        base = "home"
    else:
        base = Path(folder_abs).name or "root"
    name = f"{kind}-{base}"
    if suffix:
        name = f"{name}-{suffix}"
    return name


def _inner_command(folder: str, cfg: Config, *, exec_target: str) -> str:
    """The cd + venv + PATH + exec target wrapped into a single bash command."""
    return (
        f"cd {shlex.quote(folder)} || "
        f"{{ echo 'rci: {folder} not found on '\"$(hostname)\" >&2; exit 1; }}; "
        f"[ -f {cfg.venv_activate} ] && . {cfg.venv_activate}; "
        f'PATH="$HOME/bin:$HOME/.local/bin:$PATH" exec {exec_target}'
    )


def _tmux_wrap(session: str, inner: str) -> str:
    """Wrap ``inner`` in a tmux session attach-or-create, with fallback if tmux is missing.

    ``tmux new-session -A`` attaches when the session exists, else creates with
    the given command. The fallback path runs the command directly so the user
    still gets a working shell/claude on tmux-less nodes (just no persistence).
    """
    quoted_inner = shlex.quote(inner)
    quoted_session = shlex.quote(session)
    return (
        f"if command -v tmux >/dev/null 2>&1; then "
        f"exec tmux new-session -A -s {quoted_session} bash -lc {quoted_inner}; "
        f"else "
        f"echo 'rci: tmux not found on '\"$(hostname)\"', running without disconnect resilience.' >&2; "
        f"exec bash -lc {quoted_inner}; "
        f"fi"
    )


def launch_shell(alloc: Allocation, folder: str, cfg: Config, *, suffix: str = "") -> int:
    sess = session_name("shell", folder, suffix, home=cfg.home)
    print(f"→ {alloc.node} (job {alloc.jobid}): shell session '{sess}' in {folder}")
    script = _tmux_wrap(sess, _inner_command(folder, cfg, exec_target="bash -i"))
    return ssh.run(alloc.node, script, tty=True, check=False)


def launch_code(alloc: Allocation, folder: str, cfg: Config) -> int:
    """Open VS Code Remote-SSH against a compute node.

    On WSL the ``code`` wrapper auto-injects ``--remote wsl+<distro>`` which hijacks
    the Remote-SSH connection. Invoke the Windows ``code.cmd`` via ``cmd.exe`` instead.
    """
    print(f"→ {alloc.node} (job {alloc.jobid}): launching VS Code remote on {folder}")
    uri = f"vscode-remote://ssh-remote+{alloc.node}{folder}"
    if os.path.exists("/mnt/c"):
        return ssh.run_local(["cmd.exe", "/c", "code", "--folder-uri", uri], check=False)
    return ssh.run_local(["code", "--folder-uri", uri], check=False)
