"""Zellij session naming + management on RCI compute nodes.

The cluster login + compute nodes have only bash, no zellij by default.
``rci install-zellij`` drops a static musl binary into ``~/bin`` (shared
across login and compute nodes via the user's home directory) and a
``claude`` layout that auto-starts claude in a fresh pane.

Sessions are named ``<prefix>-<basename-of-folder>[-<suffix>]`` so different
projects get distinct sessions and parallel sessions on the same folder are
explicit (``rci claude sam2rl exp1`` ↔ session ``claude-sam2rl-exp1``).
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath
from string import Template

from . import ssh as ssh_mod
from .config import Config


def session_name(prefix: str, folder: str, suffix: str = "", *, home: str) -> str:
    """Build ``<prefix>-<basename>[-<suffix>]``. ``$HOME`` collapses to ``home``."""
    base = "home" if folder == home else PurePosixPath(folder).name
    return f"{prefix}-{base}-{suffix}" if suffix else f"{prefix}-{base}"


def list_sessions(node: str) -> list[str]:
    """List non-EXITED zellij session names on ``node``."""
    cmd = (
        r"zellij list-sessions 2>/dev/null "
        r"| sed 's/\x1b\[[0-9;]*m//g' "
        r"| grep -v EXITED || true"
    )
    out = ssh_mod.capture(node, cmd, check=False)
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def kill_session(node: str, name: str) -> int:
    n = shlex.quote(name)
    return ssh_mod.run(
        node,
        f"zellij kill-session {n} 2>&1; zellij delete-session {n} 2>&1",
        check=False,
    )


def kill_all_sessions(node: str) -> int:
    return ssh_mod.run(
        node,
        "zellij kill-all-sessions -y 2>&1; zellij delete-all-sessions -y 2>&1; echo done.",
        check=False,
    )


CLAUDE_LAYOUT = """\
layout {
    pane command="claude"
}
"""


_INSTALL_SCRIPT = Template(
    """\
set -e
mkdir -p "$$HOME/bin" "$$HOME/.config/zellij/layouts"
tmpdir=$$(mktemp -d)
trap 'rm -rf "$$tmpdir"' EXIT
cd "$$tmpdir"
url="https://github.com/zellij-org/zellij/releases/latest/download/zellij-x86_64-unknown-linux-musl.tar.gz"
echo "Downloading $$url ..."
curl -fsSL -o zellij.tar.gz "$$url"
tar -xzf zellij.tar.gz
mv zellij "$$HOME/bin/zellij"
chmod +x "$$HOME/bin/zellij"
cat > "$$HOME/.config/zellij/layouts/claude.kdl" <<'LAYOUT'
$layout
LAYOUT
echo "Installed:"
"$$HOME/bin/zellij" --version
echo "Layout: ~/.config/zellij/layouts/claude.kdl"
"""
)


def install_zellij_script() -> str:
    return _INSTALL_SCRIPT.substitute(layout=CLAUDE_LAYOUT.rstrip())


def install_zellij(cfg: Config) -> int:
    """Install zellij + claude layout on the login node (shared with compute via $HOME)."""
    return ssh_mod.run(cfg.ssh_host, "bash", stdin=install_zellij_script(), check=False)
