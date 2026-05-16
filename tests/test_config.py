"""Config: defaults split, TOML load/save round-trip, needs_setup, list→tuple coercion."""

from __future__ import annotations

from pathlib import Path

import pytest

from rci_cli import config as config_mod
from rci_cli.config import Config, load, needs_setup, save


def test_personal_defaults_are_empty() -> None:
    """No personal info in the source tree — the wizard fills these in on
    first launch. Anything non-empty here would be a regression."""
    c = Config()
    assert c.user == ""
    assert c.home == ""


def test_cluster_defaults_match_rci() -> None:
    """Cluster-wide defaults model the RCI cluster — code reading them
    expects this exact shape until config.toml overrides it."""
    c = Config()
    assert c.ssh_host == "rci"
    assert c.cpu_partition == "cpufast"
    assert c.gpu_partition == "gpufast"
    assert c.cpu_job_name == "dev"
    assert c.gpu_job_name == "dev-gpu"
    assert c.cpu_defaults == (2, 4, "1:00:00")
    assert c.gpu_defaults == (1, 2, 8, "1:00:00")
    assert c.partition_types == ("cpu", "gpu", "amdgpu", "h200")
    assert c.partition_classes == (
        ("fast", "fast"),
        ("(normal)", ""),
        ("long", "long"),
        ("extralong", "extralong"),
    )
    assert c.gpu_partition_types == ("gpu", "amdgpu", "h200")


def test_effective_home_uses_explicit_value_when_set() -> None:
    c = Config(user="alice", home="/scratch/users/alice")
    assert c.effective_home == "/scratch/users/alice"


def test_effective_home_falls_back_to_home_slash_user() -> None:
    c = Config(user="alice")
    assert c.effective_home == "/home/alice"


def test_effective_home_empty_when_unconfigured() -> None:
    assert Config().effective_home == ""


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


# ── needs_setup ─────────────────────────────────────────────────────────────


def test_needs_setup_true_when_user_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config_mod, "_config_path", lambda: tmp_path / "missing.toml")
    assert needs_setup() is True


def test_needs_setup_false_with_user_set(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('user = "alice"\n')
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)
    assert needs_setup() is False


def test_needs_setup_accepts_explicit_cfg() -> None:
    assert needs_setup(Config()) is True
    assert needs_setup(Config(user="alice")) is False


# ── save round-trip ─────────────────────────────────────────────────────────


def test_save_writes_only_non_default_fields(monkeypatch, tmp_path: Path) -> None:
    """``save`` must keep the file minimal so default changes flow through."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)
    written = save(Config(user="alice", home="/home/alice"))
    assert written == cfg_path
    body = cfg_path.read_text()
    # Only the overridden personal fields land in the file.
    assert 'user = "alice"' in body
    assert 'home = "/home/alice"' in body
    # Cluster defaults stayed at default ⇒ NOT serialised.
    assert "ssh_host" not in body
    assert "cpu_partition" not in body
    assert "partition_types" not in body


def test_save_then_load_round_trip(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)
    original = Config(
        user="alice",
        home="/scratch/users/alice",
        cpu_defaults=(4, 16, "2:00:00"),
    )
    save(original)
    loaded = load()
    assert loaded == original


def test_save_quotes_strings_with_special_chars(monkeypatch, tmp_path: Path) -> None:
    """Backslashes and double-quotes must be escaped so the file is reloadable."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)
    save(Config(user='al"ice', home="/home/al\\ice"))
    loaded = load()
    assert loaded.user == 'al"ice'
    assert loaded.home == "/home/al\\ice"


def test_save_creates_parent_dir(monkeypatch, tmp_path: Path) -> None:
    nested = tmp_path / "fresh" / "rci-cli" / "config.toml"
    monkeypatch.setattr(config_mod, "_config_path", lambda: nested)
    save(Config(user="alice"))
    assert nested.exists()


def test_save_writes_empty_when_all_at_default(monkeypatch, tmp_path: Path) -> None:
    """Saving a pristine ``Config()`` writes an empty file — by design.
    All-default config implies the user hasn't set anything yet (unconfigured)
    so subsequent ``needs_setup()`` still returns True."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)
    save(Config())
    assert cfg_path.read_text() == ""
    assert needs_setup() is True


@pytest.mark.parametrize("value,expected", [
    (True, "true"),
    (False, "false"),
    (42, "42"),
    ("hello", '"hello"'),
    ((1, 2, 3), "[1, 2, 3]"),
    (("a", "b"), '["a", "b"]'),
    (((1, 2), (3, 4)), "[[1, 2], [3, 4]]"),
])
def test_toml_value_round_trip(value, expected) -> None:
    """Internal serialiser sanity — covers every type in :class:`Config`."""
    assert config_mod._toml_value(value) == expected
