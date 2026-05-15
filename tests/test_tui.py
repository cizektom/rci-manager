"""Headless TUI smoke tests: app mounts, table accepts data, actions don't crash."""

from __future__ import annotations

import pytest

from rci_cli import alloc as alloc_mod
from rci_cli import launch
from rci_cli import slurm
from rci_cli.alloc import Allocation
from rci_cli.tui import RUNNING_STATES, JobRow, JobsPanel, RciApp


# ── JobRow parser (pure unit) ───────────────────────────────────────────────


def test_jobrow_from_running_line() -> None:
    # squeue's %T emits the long-form state name, e.g. "RUNNING" — not "R".
    line = "   1234567   cpufast    vscode    RUNNING   00:05   04:00:00   1 n01"
    r = JobRow.from_squeue_line(line)
    assert r is not None
    assert r.jobid == "1234567"
    assert r.partition == "cpufast"
    assert r.name == "vscode"
    assert r.state == "RUNNING"
    assert r.node == "n01"


def test_jobrow_from_pending_line_keeps_reason() -> None:
    line = "   1234568   cpufast    vscode    PENDING   0:00   04:00:00   1 (Resources)"
    r = JobRow.from_squeue_line(line)
    assert r is not None
    assert r.state == "PENDING"
    assert r.node == "(Resources)"


def test_running_states_accept_both_long_and_short_form() -> None:
    """Regression: %T yields ``RUNNING`` but some setups use ``R`` — both must pass guards."""
    assert "R" in RUNNING_STATES
    assert "RUNNING" in RUNNING_STATES


def test_jobrow_returns_none_for_garbage() -> None:
    assert JobRow.from_squeue_line("") is None
    assert JobRow.from_squeue_line("just three fields") is None


# ── App mount ───────────────────────────────────────────────────────────────


async def test_app_mounts_with_jobs_tab(monkeypatch) -> None:
    """The app composes a TabbedContent with a JobsPanel; refresh doesn't crash."""
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setattr(alloc_mod, "find_strongest", lambda cfg: None)

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        assert panel is not None
        # Default theme picks up the terminal palette.
        assert app.theme == "ansi-dark"


async def test_refresh_populates_table(monkeypatch) -> None:
    monkeypatch.setattr(
        slurm,
        "list_jobs",
        lambda cfg: (
            "JOBID PARTITION NAME STATE TIME LIMIT NODES NODE\n"
            "1234567 cpufast vscode RUNNING 00:05 04:00:00 1 n01\n"
            "1234568 gpufast vscode-gpu RUNNING 00:10 04:00:00 1 g05\n"
        ),
    )
    monkeypatch.setattr(
        alloc_mod, "find_strongest", lambda cfg: Allocation(node="g05", jobid="1234568")
    )

    app = RciApp()
    async with app.run_test() as pilot:
        # Give the refresh worker a couple of frames to land.
        await pilot.pause()
        for _ in range(5):
            await pilot.pause(0.05)
        panel = app.query_one(JobsPanel)
        assert len(panel._rows) == 2
        assert panel._rows[0].jobid == "1234567"
        assert panel._rows[1].name == "vscode-gpu"


async def test_shell_action_attaches_to_strongest_alloc(monkeypatch) -> None:
    """Pressing `s` attaches to the strongest running allocation via launch_shell."""
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setattr(
        alloc_mod,
        "find_strongest",
        lambda cfg: Allocation(node="n01", jobid="1234567"),
    )

    called: dict[str, object] = {}

    def fake_shell(alloc, folder, cfg):
        called["alloc"] = alloc
        called["folder"] = folder
        return 0

    monkeypatch.setattr(launch, "launch_shell", fake_shell)

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        # ``app.suspend`` would actually flip the terminal; bypass it for the test.
        from contextlib import contextmanager

        @contextmanager
        def noop_suspend():
            yield

        monkeypatch.setattr(app, "suspend", noop_suspend)
        panel.action_shell_into()
    assert called.get("alloc") is not None, "launch_shell was never invoked"
    assert called["alloc"].node == "n01"
    assert called["alloc"].jobid == "1234567"


async def test_shell_action_pops_modal_when_no_alloc(monkeypatch) -> None:
    """When no allocation exists, pressing `s` opens the New Instance modal."""
    from rci_cli.tui import NewInstanceModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setattr(alloc_mod, "find_strongest", lambda cfg: None)
    monkeypatch.setattr(launch, "launch_shell", lambda *a, **kw: 0)

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel.action_shell_into()
        await pilot.pause()
        # The modal is pushed; the active screen should be a NewInstanceModal.
        assert isinstance(app.screen, NewInstanceModal)


async def test_action_log_auto_clears(monkeypatch) -> None:
    """Regression: the inline action log must fade after ``ACTION_FADE_SECONDS``."""
    from rci_cli import tui as tui_mod

    monkeypatch.setattr(tui_mod, "ACTION_FADE_SECONDS", 0.1)
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setattr(alloc_mod, "find_strongest", lambda cfg: None)

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._notify_action("cancelled job 1234")
        assert panel._last_action == "cancelled job 1234"
        # Wait past the fade window
        await pilot.pause(0.25)
        assert panel._last_action == ""


async def test_refresh_handles_squeue_failure_gracefully(monkeypatch) -> None:
    def boom(cfg):
        raise RuntimeError("network blip")

    monkeypatch.setattr(slurm, "list_jobs", boom)
    monkeypatch.setattr(alloc_mod, "find_strongest", lambda cfg: None)

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(5):
            await pilot.pause(0.05)
        # App should still be running — error went to a notification, not a crash.
        assert app.is_running
