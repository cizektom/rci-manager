"""Shared fixtures.

Tests run fully offline — anything that would touch ssh/squeue/scancel is
monkeypatched per-test on the relevant module (``rci_cli.slurm`` or
``rci_cli.ssh``).
"""

from __future__ import annotations

import pytest

from rci_cli.config import Config


@pytest.fixture
def cfg() -> Config:
    """A pristine Config with the rci-cli defaults."""
    return Config()
