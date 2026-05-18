"""Thin ssh wrappers used by the rest of the package.

Everything goes through the local ``ssh`` binary using the user's ``~/.ssh/config``
(which must define ``Host rci`` and ``Host n* g*`` — see the rci-cli README).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence


def capture(
    host: str,
    remote_cmd: str,
    *,
    check: bool = True,
    merge_stderr: bool = False,
) -> str:
    """Run ``remote_cmd`` on ``host`` over ssh and return its stdout (stripped).

    ``merge_stderr=True`` concatenates stderr after stdout. Needed for
    ``salloc``, which writes ``Granted job allocation NNN`` to stderr by
    default — capturing only stdout would lose the job id.
    """
    proc = subprocess.run(
        ["ssh", host, remote_cmd],
        check=check,
        capture_output=True,
        text=True,
        # Detach stdin from the parent TTY — ``capture_output=True`` only
        # rewires stdout/stderr. With stdin inherited, ssh races the TUI
        # for keystrokes from the same terminal; on every background
        # refresh tick a fraction of typed characters disappears into ssh.
        stdin=subprocess.DEVNULL,
    )
    if merge_stderr:
        return (proc.stdout + proc.stderr).strip()
    return proc.stdout.strip()


def run(
    host: str,
    remote_cmd: str = "",
    *,
    tty: bool = False,
    check: bool = True,
    stdin: str | None = None,
) -> int:
    """Run ``remote_cmd`` on ``host`` over ssh, inheriting the current TTY.

    ``tty=True`` passes ``-tt`` so an interactive program (claude, bash, …) sees
    a real PTY even when the connection traverses ``ProxyJump``.

    ``stdin`` pipes a string into the ssh process — used to send multi-line
    bash scripts to ``ssh host bash`` for one-off remote installs.
    """
    argv = ["ssh"]
    if tty:
        argv.append("-tt")
    argv.append(host)
    if remote_cmd:
        argv.append(remote_cmd)
    if stdin is not None:
        return subprocess.run(argv, check=check, input=stdin, text=True).returncode
    return subprocess.run(argv, check=check).returncode


def port_forward(host: str, local_port: int, remote_port: int) -> int:
    """Block forwarding ``localhost:local_port`` → ``host:remote_port`` until Ctrl-C."""
    return subprocess.run(
        ["ssh", "-N", "-L", f"{local_port}:localhost:{remote_port}", host],
        check=False,
    ).returncode


def run_local(argv: Sequence[str], *, check: bool = True) -> int:
    """Run a local command, inheriting the TTY. Used for the WSL ``cmd.exe`` path."""
    return subprocess.run(list(argv), check=check).returncode
