"""Persistent rci-cli configuration loaded from ``~/.config/rci-cli/config.toml``.

The defaults split into two layers:

- **Personal fields** (``user``, ``home``) — empty by default. The CLI / TUI
  runs a setup flow on first launch to fill them in; see :func:`needs_setup`
  and :func:`save`.
- **Cluster fields** (``ssh_host``, partitions, walltimes, …) — sensible
  defaults that match the RCI CVUT Slurm cluster, overridable per-site by
  editing ``config.toml`` (no code change needed).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # ── personal (filled in by the setup wizard; empty by default) ────────
    # Slurm/SSH username on the cluster — used in ``squeue -u``, ``scancel -u``,
    # and as the implicit owner of jobs the TUI watches. Empty ⇒ not configured;
    # both the CLI and TUI gate on this and run setup.
    user: str = ""
    # Absolute path to the user's home on the cluster — empty falls back to
    # ``/home/<user>`` (see :meth:`effective_home`). Override here when the
    # cluster's home layout differs (e.g. ``/storage/users/<user>``).
    home: str = ""

    # ── cluster-wide defaults (RCI CVUT; override in config.toml) ─────────
    ssh_host: str = "rci"
    cpu_partition: str = "cpufast"
    gpu_partition: str = "gpufast"
    cpu_job_name: str = "dev"
    gpu_job_name: str = "dev-gpu"
    # Conservative debug defaults — schedule fast, don't burn quota if you forget one.
    # Override per-allocation with --cores/--mem/--time or globally in ~/.config/rci-cli/config.toml.
    cpu_defaults: tuple[int, int, str] = (2, 4, "1:00:00")  # cores, memGB, walltime
    gpu_defaults: tuple[int, int, int, str] = (1, 2, 8, "1:00:00")  # gpus, cores, memGB, walltime
    # Partition catalog for the New Instance modal. The full partition name is
    # ``<type><class>`` (e.g. ``gpufast``); the ``(normal)`` class is rendered
    # as a label but maps to an empty suffix. Override these in config.toml to
    # adapt rci-cli to a different Slurm cluster — no code change needed.
    partition_types: tuple[str, ...] = ("cpu", "gpu", "amdgpu", "h200")
    partition_classes: tuple[tuple[str, str], ...] = (
        ("fast", "fast"),
        ("(normal)", ""),
        ("long", "long"),
        ("extralong", "extralong"),
    )
    # Partition types that accept ``--gres=gpu:N``. Used by the modal to
    # show/hide the GPUs field and reject CPU-only partitions with gpus>0.
    gpu_partition_types: tuple[str, ...] = ("gpu", "amdgpu", "h200")

    @property
    def effective_home(self) -> str:
        """Resolve ``home`` with a fallback to ``/home/<user>``.

        The setup wizard pre-fills this with ``/home/<user>`` so the explicit
        value is what's normally on disk; this fallback covers manually-edited
        ``config.toml`` files where the user set only ``user``.
        """
        if self.home:
            return self.home
        if self.user:
            return f"/home/{self.user}"
        return ""


# The minimal field set the wizard collects. ``home`` defaults to
# ``/home/<user>`` in the prompt, but the user can override.
SETUP_FIELDS: tuple[str, ...] = ("user", "ssh_host", "home")


def _config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / "rci-cli" / "config.toml"


def _to_tuples(value: object) -> object:
    """Recursively convert TOML arrays-of-arrays into tuples-of-tuples so the
    frozen dataclass stays hashable. Leaves scalars alone."""
    if isinstance(value, list):
        return tuple(_to_tuples(v) for v in value)
    return value


def load() -> Config:
    """Load config from $XDG_CONFIG_HOME/rci-cli/config.toml, falling back to defaults.

    Any field omitted in the TOML keeps its default. Tuple fields can be supplied as
    arrays in TOML (e.g. ``cpu_defaults = [4, 16, "4:00:00"]``); arrays-of-arrays
    are converted to tuples-of-tuples (e.g. ``partition_classes``).
    """
    path = _config_path()
    if not path.exists():
        return Config()
    with path.open("rb") as f:
        data = tomllib.load(f)
    parsed: dict = {}
    for k, v in data.items():
        if k in Config.__dataclass_fields__:
            parsed[k] = _to_tuples(v)
    return Config(**parsed)


def needs_setup(cfg: Config | None = None) -> bool:
    """True when the CLI/TUI should run the setup wizard before doing cluster work.

    Triggers on either a missing config file or an empty ``user`` field — i.e.
    the wizard hasn't been completed. Other fields can stay at defaults.
    """
    if cfg is None:
        cfg = load()
    return not cfg.user


def _toml_value(value: object) -> str:
    """Serialise a Config field value to a TOML literal.

    Covers the field types actually used in :class:`Config` (str, int, bool,
    sequences of those). Strings get double-quoted with backslash + quote
    escaped — TOML's basic-string rules.
    """
    if isinstance(value, bool):  # bool before int (bool is an int subclass)
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot serialise {type(value).__name__} to TOML")


def save(cfg: Config) -> Path:
    """Persist ``cfg`` to ``_config_path()``, returning the path written.

    Only writes fields that differ from a blank :class:`Config` — keeps the
    file minimal so future default changes can flow through without a rewrite.
    Creates the parent directory if missing.
    """
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = Config()
    lines: list[str] = []
    for f in fields(Config):
        cur = getattr(cfg, f.name)
        if cur == getattr(baseline, f.name):
            continue
        lines.append(f"{f.name} = {_toml_value(cur)}")
    path.write_text(("\n".join(lines) + "\n") if lines else "")
    return path
