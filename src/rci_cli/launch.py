"""Launch interactive tools on a compute node — claude, VS Code Remote-SSH, plain shell.

Compute nodes (``n*``/``g*``) don't have zsh installed and their bash doesn't replicate
the login-node setup, so we manually source the sam2rl venv and prepend
``$HOME/bin:$HOME/.local/bin`` to PATH inside each ssh invocation.

claude/shell launches are wrapped in a named zellij session so ssh disconnects
don't kill the work — reconnecting re-attaches to the same session. If zellij
isn't installed on the node we fall back to running the command directly and
print a hint pointing at ``rci install-zellij``.
"""

from __future__ import annotations

import os
import shlex
from string import Template

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


_ZELLIJ_TEMPLATE = Template(
    r"""cd $folder 2>/dev/null || { echo "$label: folder not found on $$(hostname)" >&2; exit 1; }
[ -f $venv ] && . $venv
export PATH="$$HOME/bin:$$HOME/.local/bin:$$PATH"
if ! command -v zellij >/dev/null 2>&1; then
    echo "$label: zellij not found on $$(hostname); running without disconnect resilience." >&2
    echo "             install with: rci install-zellij" >&2
    exec $fallback
fi
running=$$(zellij list-sessions 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -v EXITED | awk '{print $$1}')
if echo "$$running" | grep -qFx $session; then
    exec zellij attach $session
fi
zellij delete-session $session 2>/dev/null || true
exec zellij --session $session $layout_arg
"""
)


def _zellij_script(
    *,
    folder: str,
    session: str,
    label: str,
    fallback: str,
    layout: str | None,
    cfg: Config,
) -> str:
    return _ZELLIJ_TEMPLATE.substitute(
        folder=shlex.quote(folder),
        session=shlex.quote(session),
        label=label,
        fallback=fallback,
        venv=cfg.venv_activate,
        layout_arg=f"--layout {layout}" if layout else "",
    )


def launch_claude(alloc: Allocation, folder: str, session: str, cfg: Config) -> int:
    print(f"→ {alloc.node} (job {alloc.jobid}): claude session '{session}' in {folder}")
    script = _zellij_script(
        folder=folder,
        session=session,
        label="rci-claude",
        fallback="claude",
        layout="claude",
        cfg=cfg,
    )
    return ssh.run(alloc.node, script, tty=True, check=False)


def launch_shell(alloc: Allocation, folder: str, session: str, cfg: Config) -> int:
    print(f"→ {alloc.node} (job {alloc.jobid}): shell session '{session}' in {folder}")
    script = _zellij_script(
        folder=folder,
        session=session,
        label="rci-shell",
        fallback="bash -i",
        layout=None,
        cfg=cfg,
    )
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
