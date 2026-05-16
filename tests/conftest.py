"""Shared fixtures.

Tests run fully offline — anything that would touch ssh/squeue/scancel is
monkeypatched per-test on the relevant module (``rci_cli.slurm`` or
``rci_cli.ssh``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rci_cli.config import Config


@pytest.fixture
def cfg() -> Config:
    """A Config with concrete personal-field values for tests.

    The real ``Config()`` ships with empty ``user``/``home`` so the setup
    wizard pops on first launch (no hardcoded credentials in the binary).
    Tests need real-looking values to exercise the cluster-touching code
    paths — provide them explicitly here.
    """
    return Config(user="cizekto2", home="/home/cizekto2")


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each test sees a fresh, already-configured rci-cli (XDG_CONFIG_HOME → tmp).

    Without this, tests that hit ``config.load()`` would either:
      - read the developer's real ``~/.config/rci-cli/config.toml`` (flaky), or
      - return an unconfigured ``Config()`` and trigger the SetupModal in
        every TUI test.

    Tests exercising the wizard / ``needs_setup()`` path override
    ``XDG_CONFIG_HOME`` themselves to point at an empty dir.
    """
    cfg_root = tmp_path / "config-home"
    cfg_dir = cfg_root / "rci-cli"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        'user = "cizekto2"\n'
        'home = "/home/cizekto2"\n'
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_root))
