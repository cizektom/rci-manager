"""Launch interactive tools on a compute node — claude, VS Code Remote-SSH, plain shell.

Compute nodes (``n*``/``g*``) don't have zsh installed and their bash doesn't replicate
the login-node setup, so we manually source the sam2rl venv and prepend
``$HOME/bin:$HOME/.local/bin`` to PATH inside each ssh invocation.

The ``*_in_tab`` variants delegate to :mod:`rci_cli.newtab` instead of taking
over the current terminal: they spawn ``rci shell`` / ``rci claude`` in a new
tab of the active terminal, passing ``--node`` so the spawned instance hits
the same compute node we already picked.
"""

from __future__ import annotations

import os

from . import newtab, ssh
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


def _remote_preamble(folder: str, cfg: Config) -> str:
    """Build the ``cd … && source venv && PATH=…`` prefix for compute-node commands."""
    return (
        f"cd '{folder}' || {{ echo 'rci: {folder} not found on \\$(hostname)' >&2; exit 1; }}; "
        f"[ -f {cfg.venv_activate} ] && . {cfg.venv_activate}; "
        f'PATH="$HOME/bin:$HOME/.local/bin:$PATH"'
    )


def _job_tag(alloc: Allocation) -> str:
    """``(job NNN)`` when known, empty string when ``--node`` skipped lookup."""
    return f" (job {alloc.jobid})" if alloc.jobid else ""


def launch_claude(alloc: Allocation, folder: str, cfg: Config) -> int:
    print(f"→ {alloc.node}{_job_tag(alloc)}: launching claude in {folder}")
    cmd = f"{_remote_preamble(folder, cfg)} exec claude"
    return ssh.run(alloc.node, cmd, tty=True, check=False)


def launch_shell(alloc: Allocation, folder: str, cfg: Config) -> int:
    print(f"→ {alloc.node}{_job_tag(alloc)}: opening shell in {folder}")
    cmd = f"{_remote_preamble(folder, cfg)} exec bash -i"
    return ssh.run(alloc.node, cmd, tty=True, check=False)


def launch_code(alloc: Allocation, folder: str, cfg: Config) -> int:
    """Open VS Code Remote-SSH against a compute node.

    On WSL the ``code`` wrapper auto-injects ``--remote wsl+<distro>`` which hijacks
    the Remote-SSH connection. Invoke the Windows ``code.cmd`` via ``cmd.exe`` instead.
    """
    print(f"→ {alloc.node}{_job_tag(alloc)}: launching VS Code remote on {folder}")
    uri = f"vscode-remote://ssh-remote+{alloc.node}{folder}"
    if os.path.exists("/mnt/c"):
        return ssh.run_local(["cmd.exe", "/c", "code", "--folder-uri", uri], check=False)
    return ssh.run_local(["code", "--folder-uri", uri], check=False)


def _tab_args(subcmd: str, alloc: Allocation, folder_arg: str) -> list[str]:
    """Build the ``rci <subcmd>`` argv for a new-tab spawn.

    ``folder_arg`` is the *raw* user-provided folder (empty / relative / absolute);
    re-passing it lets the spawned ``rci`` apply the same resolution.
    """
    argv = ["rci", subcmd]
    if folder_arg:
        argv.append(folder_arg)
    argv.extend(["--node", alloc.node])
    return argv


def launch_shell_in_tab(alloc: Allocation, folder_arg: str, cfg: Config) -> int:
    """Open ``rci shell`` in a new terminal tab. Returns 0 on launch, 2 if unsupported."""
    argv = _tab_args("shell", alloc, folder_arg)
    try:
        rc, kind = newtab.spawn(argv, title=f"rci shell · {alloc.node}")
    except newtab.NoSupportedTerminal as e:
        print(f"rci: {e}")
        return 2
    print(f"→ {alloc.node}{_job_tag(alloc)}: opened {kind} tab → rci shell")
    return rc


def launch_claude_in_tab(alloc: Allocation, folder_arg: str, cfg: Config) -> int:
    """Open ``rci claude`` in a new terminal tab. Returns 0 on launch, 2 if unsupported."""
    argv = _tab_args("claude", alloc, folder_arg)
    try:
        rc, kind = newtab.spawn(argv, title=f"rci claude · {alloc.node}")
    except newtab.NoSupportedTerminal as e:
        print(f"rci: {e}")
        return 2
    print(f"→ {alloc.node}{_job_tag(alloc)}: opened {kind} tab → rci claude")
    return rc
