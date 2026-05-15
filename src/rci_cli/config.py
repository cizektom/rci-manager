from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    user: str = "cizekto2"
    ssh_host: str = "rci"
    home: str = "/home/cizekto2"
    cpu_partition: str = "cpufast"
    gpu_partition: str = "gpufast"
    cpu_job_name: str = "dev"
    gpu_job_name: str = "dev-gpu"
    # Conservative debug defaults — schedule fast, don't burn quota if you forget one.
    # Override per-allocation with --cores/--mem/--time or globally in ~/.config/rci-cli/config.toml.
    cpu_defaults: tuple[int, int, str] = (2, 4, "1:00:00")  # cores, memGB, walltime
    gpu_defaults: tuple[int, int, int, str] = (1, 2, 8, "1:00:00")  # gpus, cores, memGB, walltime
    venv_activate: str = "$HOME/sam2rl/.venv/bin/activate"


def _config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / "rci-cli" / "config.toml"


def load() -> Config:
    """Load config from $XDG_CONFIG_HOME/rci-cli/config.toml, falling back to defaults.

    Any field omitted in the TOML keeps its default. Tuple fields can be supplied as
    arrays in TOML (e.g. ``cpu_defaults = [4, 16, "4:00:00"]``).
    """
    path = _config_path()
    if not path.exists():
        return Config()
    with path.open("rb") as f:
        data = tomllib.load(f)
    fields: dict = {}
    for k, v in data.items():
        if k in Config.__dataclass_fields__:
            fields[k] = tuple(v) if isinstance(v, list) else v
    return Config(**fields)
