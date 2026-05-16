"""First-run setup flow for rci-cli.

Two entry points:

- :func:`run_cli` — interactive prompt for the terminal (``rci setup``, or
  auto-invoked when bare ``rci`` finds an empty config).
- :class:`SetupModal` lives in :mod:`rci_cli.tui` and shares the same field
  set (:data:`rci_cli.config.SETUP_FIELDS`) and writer (:func:`build_cfg`).

The wizard collects the bare minimum (``user``, ``ssh_host``, ``home``);
venv activation on the compute node is hard-coded to ``.venv/bin/activate``
relative to the working folder (see :mod:`rci_cli.launch`), so there's no
prompt for it.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import config as config_mod
from .config import Config


def build_cfg(*, user: str, ssh_host: str, home: str = "") -> Config:
    """Compose a :class:`Config` from the wizard's collected fields.

    ``home`` defaults to ``/home/<user>`` when blank — matches the prompt
    default in :func:`run_cli` and what most clusters use.
    """
    if not user:
        raise ValueError("user is required")
    return Config(
        user=user.strip(),
        ssh_host=(ssh_host or "rci").strip(),
        home=(home.strip() or f"/home/{user.strip()}"),
    )


def run_cli() -> Path:
    """Interactive terminal wizard. Writes the config file and returns its path.

    Re-run safe: existing values prefill each prompt, so the user can press
    Enter through fields they don't want to change.
    """
    existing = config_mod.load()
    typer.echo("rci-cli setup — fill in once, edit ~/.config/rci-cli/config.toml later.")
    typer.echo()

    user = typer.prompt(
        "SSH username on the cluster",
        default=existing.user or None,
    ).strip()
    ssh_host = typer.prompt(
        "SSH host alias (from ~/.ssh/config)",
        default=existing.ssh_host or "rci",
    ).strip()
    default_home = existing.home or f"/home/{user}"
    home = typer.prompt(
        "Home directory on the cluster",
        default=default_home,
    ).strip()

    cfg = build_cfg(user=user, ssh_host=ssh_host, home=home)
    path = config_mod.save(cfg)
    typer.echo()
    typer.echo(f"Saved to {path}.")
    return path
