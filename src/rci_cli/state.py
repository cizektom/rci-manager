"""Tiny JSON-backed state store for per-user runtime preferences.

Kept separate from :mod:`config` because config is user-edited TOML and state
is machine-written (e.g. last folder typed into the TUI). Lives under
``$XDG_STATE_HOME/rci-cli/state.json`` (default ``~/.local/state/rci-cli``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _state_path() -> Path:
    xdg = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(xdg) / "rci-cli" / "state.json"


def _load() -> dict:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def get_last_folder() -> str:
    val = _load().get("last_folder", "")
    return val if isinstance(val, str) else ""


def set_last_folder(folder: str) -> None:
    data = _load()
    data["last_folder"] = folder
    _save(data)


# Keys match the NewInstanceModal form fields, not AllocParams, because the
# modal needs the *split* partition (type + class) to repopulate the two
# Selects — assembled "cpufast" can't be cleanly un-split (e.g. "cpu" vs "cpu"+"").
_INSTANCE_KEYS = ("partition_type", "partition_class", "cores", "gpus", "mem_gb", "walltime")


def get_last_instance_params() -> dict | None:
    """Return the last submitted New-Instance params, or ``None`` if nothing saved.

    The caller is responsible for validating values (the modal does — falling
    back to ``Config`` defaults if anything's missing or out of range).
    """
    data = _load().get("last_instance_params")
    if not isinstance(data, dict):
        return None
    return {k: data.get(k) for k in _INSTANCE_KEYS}


def set_last_instance_params(
    *,
    partition_type: str,
    partition_class: str,
    cores: int,
    gpus: int,
    mem_gb: int,
    walltime: str,
) -> None:
    data = _load()
    data["last_instance_params"] = {
        "partition_type": partition_type,
        "partition_class": partition_class,
        "cores": cores,
        "gpus": gpus,
        "mem_gb": mem_gb,
        "walltime": walltime,
    }
    _save(data)


_WORKSPACE_KEYS = ("agents", "terminals")


def get_last_workspace_options() -> dict | None:
    """Return the last submitted Workspace-options values, or ``None`` if unsaved.

    Caller validates (the modal does — falling back to ``Config`` defaults
    for missing/out-of-range entries).
    """
    data = _load().get("last_workspace_options")
    if not isinstance(data, dict):
        return None
    return {k: data.get(k) for k in _WORKSPACE_KEYS}


def set_last_workspace_options(*, agents: int, terminals: int) -> None:
    data = _load()
    data["last_workspace_options"] = {"agents": agents, "terminals": terminals}
    _save(data)
