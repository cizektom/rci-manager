"""Config loading: defaults, full override, partial override, list→tuple coercion."""

from __future__ import annotations

from pathlib import Path

from rci_cli import config as config_mod
from rci_cli.config import Config, load


def test_defaults_match_zsh_helpers() -> None:
    """The defaults must mirror the zsh ``rci-*`` helpers — they're the contract."""
    c = Config()
    assert c.user == "cizekto2"
    assert c.ssh_host == "rci"
    assert c.home == "/home/cizekto2"
    assert c.cpu_partition == "cpufast"
    assert c.gpu_partition == "gpufast"
    assert c.cpu_job_name == "dev"
    assert c.gpu_job_name == "dev-gpu"
    # Conservative debug defaults — see config.py for the rationale.
    assert c.cpu_defaults == (2, 4, "1:00:00")
    assert c.gpu_defaults == (1, 2, 8, "1:00:00")


def test_load_with_no_config_file_returns_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config_mod, "_config_path", lambda: tmp_path / "missing.toml")
    assert load() == Config()


def test_load_partial_override(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('user = "other-user"\nhome = "/home/other-user"\n')
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)
    loaded = load()
    assert loaded.user == "other-user"
    assert loaded.home == "/home/other-user"
    # Untouched fields keep their defaults
    assert loaded.ssh_host == "rci"
    assert loaded.cpu_defaults == (2, 4, "1:00:00")


def test_load_list_fields_become_tuples(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('cpu_defaults = [8, 32, "8:00:00"]\ngpu_defaults = [2, 16, 64, "12:00:00"]\n')
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)
    loaded = load()
    assert loaded.cpu_defaults == (8, 32, "8:00:00")
    assert loaded.gpu_defaults == (2, 16, 64, "12:00:00")


def test_load_ignores_unknown_keys(monkeypatch, tmp_path: Path) -> None:
    """A typo in the config shouldn't crash the CLI — just be silently ignored."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('user = "foo"\nbogus_field = "ignored"\n')
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)
    loaded = load()
    assert loaded.user == "foo"


def test_partition_catalog_defaults_match_rci() -> None:
    """Defaults model the RCI cluster's partition matrix — code reading them
    expects this exact shape until config.toml overrides it."""
    c = Config()
    assert c.partition_types == ("cpu", "gpu", "amdgpu", "h200")
    assert c.partition_classes == (
        ("fast", "fast"),
        ("(normal)", ""),
        ("long", "long"),
        ("extralong", "extralong"),
    )
    assert c.gpu_partition_types == ("gpu", "amdgpu", "h200")


def test_load_overrides_partition_catalog_for_other_cluster(
    monkeypatch, tmp_path: Path
) -> None:
    """A different Slurm site can replace the whole partition matrix from
    config.toml — including the nested array-of-arrays for partition_classes
    (recursive list→tuple coercion handles it)."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        'partition_types = ["compute", "gpu-a100"]\n'
        'partition_classes = [["short", "short"], ["(normal)", ""]]\n'
        'gpu_partition_types = ["gpu-a100"]\n'
    )
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)
    loaded = load()
    assert loaded.partition_types == ("compute", "gpu-a100")
    assert loaded.partition_classes == (
        ("short", "short"),
        ("(normal)", ""),
    )
    assert loaded.gpu_partition_types == ("gpu-a100",)
