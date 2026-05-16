"""First-run setup: build_cfg field handling and the CLI prompt flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from rci_cli import config as config_mod
from rci_cli import setup as setup_mod
from rci_cli.config import Config
from rci_cli.setup import build_cfg, run_cli


# ── build_cfg ──────────────────────────────────────────────────────────────


def test_build_cfg_uses_provided_values() -> None:
    c = build_cfg(user="alice", ssh_host="rci", home="/scratch/users/alice")
    assert c.user == "alice"
    assert c.ssh_host == "rci"
    assert c.home == "/scratch/users/alice"


def test_build_cfg_defaults_home_to_home_user() -> None:
    """Blank ``home`` is the wizard's UX default — fill with ``/home/<user>``."""
    c = build_cfg(user="alice", ssh_host="rci")
    assert c.home == "/home/alice"


def test_build_cfg_defaults_ssh_host_to_rci() -> None:
    c = build_cfg(user="alice", ssh_host="")
    assert c.ssh_host == "rci"


def test_build_cfg_strips_whitespace() -> None:
    """Users pasting from docs sometimes drag trailing whitespace — be tolerant."""
    c = build_cfg(user="  alice  ", ssh_host="  rci  ", home="  /h/alice  ")
    assert c.user == "alice"
    assert c.ssh_host == "rci"
    assert c.home == "/h/alice"


def test_build_cfg_rejects_empty_user() -> None:
    """Without a user, ``squeue -u`` etc. would silently target the wrong account."""
    with pytest.raises(ValueError, match="user is required"):
        build_cfg(user="", ssh_host="rci")


# ── run_cli ────────────────────────────────────────────────────────────────


def _stub_prompts(monkeypatch, answers: list[str]) -> None:
    """Pop ``answers`` in call order. typer.prompt(default=…) returns the
    default if our stub returns ``""``; we sidestep that by always returning
    a concrete value (the prompt-layer default behaviour is exercised in the
    CliRunner-based tests in test_cli.py)."""
    queue = list(answers)

    def fake_prompt(*_a, **_kw):
        return queue.pop(0)

    monkeypatch.setattr(setup_mod.typer, "prompt", fake_prompt)


def test_run_cli_writes_provided_answers(monkeypatch, tmp_path: Path) -> None:
    """run_cli must funnel prompt answers through build_cfg → save."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)
    _stub_prompts(
        monkeypatch,
        answers=["alice", "rci-other", "/scratch/alice"],
    )
    written = run_cli()
    assert written == cfg_path
    loaded = config_mod.load()
    assert loaded == Config(user="alice", ssh_host="rci-other", home="/scratch/alice")


def test_run_cli_prefills_from_existing_config(monkeypatch, tmp_path: Path) -> None:
    """Re-running setup must read the saved config first so prompt defaults
    reflect the current values (the user can keep them by pressing Enter)."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        'user = "existing"\nssh_host = "old-host"\nhome = "/h/existing"\n'
    )
    monkeypatch.setattr(config_mod, "_config_path", lambda: cfg_path)

    seen_defaults: list = []

    def fake_prompt(prompt, default=None, **_kw):
        seen_defaults.append(default)
        # New values for every field — confirms the existing config gets
        # overwritten cleanly, not merged.
        return {
            "SSH username on the cluster": "new",
            "SSH host alias (from ~/.ssh/config)": "new-host",
            "Home directory on the cluster": "/h/new",
        }[prompt]

    monkeypatch.setattr(setup_mod.typer, "prompt", fake_prompt)
    run_cli()
    # The three prompts surfaced the existing values as defaults.
    assert seen_defaults == ["existing", "old-host", "/h/existing"]
    # Saved cfg reflects the new answers.
    loaded = config_mod.load()
    assert loaded.user == "new"
    assert loaded.ssh_host == "new-host"
    assert loaded.home == "/h/new"
