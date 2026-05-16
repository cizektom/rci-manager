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
    print(f"→ {alloc.node} (job {alloc.jobid}): opening shell in {folder}")
    cmd = f"{_remote_preamble(folder)} exec bash -i"
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
