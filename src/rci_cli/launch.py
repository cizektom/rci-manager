"""Launch interactive tools on a compute node — shell and VS Code Remote-SSH.

Compute nodes (``n*``/``g*``) don't have zsh installed and their bash doesn't
replicate the login-node setup, so we manually source the sam2rl venv and
prepend ``$HOME/bin:$HOME/.local/bin`` to PATH inside each ssh invocation.

No persistence layer at this level — ssh disconnect ends the session. Wrap
inside the spawned bash with ``tmux`` / ``screen`` if you need survival.
"""

from __future__ import annotations

import os

from . import ssh
from .alloc import Allocation
from .config import Config


def resolve_folder(folder: str | None, cfg: Config) -> str:
    """Folder rules shared by shell / editor launchers.

    - empty → ``cfg.home``
    - relative → resolved under ``cfg.home``
    - absolute → returned as-is
    """
    if not folder:
        return cfg.home
    if folder.startswith("/"):
        return folder
    return f"{cfg.home}/{folder}"


def _remote_preamble(folder: str, cfg: Config) -> str:
    """Build the ``cd … && source venv && PATH=…`` prefix for compute-node commands."""
    return (
        f"cd '{folder}' || {{ echo 'rci: {folder} not found on '\"$(hostname)\" >&2; exit 1; }}; "
        f"[ -f {cfg.venv_activate} ] && . {cfg.venv_activate}; "
        f'PATH="$HOME/bin:$HOME/.local/bin:$PATH"'
    )


def launch_shell(alloc: Allocation, folder: str, cfg: Config) -> int:
    print(f"→ {alloc.node} (job {alloc.jobid}): opening shell in {folder}")
    cmd = f"{_remote_preamble(folder, cfg)} exec bash -i"
    return ssh.run(alloc.node, cmd, tty=True, check=False)


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
