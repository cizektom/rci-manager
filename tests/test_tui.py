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
    # %T emits the long-form state; new columns CPUS/MIN_M/TRES_PER appear before NODELIST.
    line = "  1234567  cpufast        dev  RUNNING    00:05  01:00:00     2    4G     N/A n01"
    r = JobRow.from_squeue_line(line)
    assert r is not None
    assert r.jobid == "1234567"
    assert r.partition == "cpufast"
    assert r.name == "dev"
    assert r.state == "RUNNING"
    assert r.cpus == "2"
    assert r.mem == "4G"
    assert r.gres == "N/A"
    assert r.gpu_count == "—"
    assert r.node == "n01"


def test_jobrow_from_gpu_line_parses_gpus() -> None:
    line = "  1234568   gpufast    dev-gpu  RUNNING    00:10  04:00:00     8   32G   gpu:1 g05"
    r = JobRow.from_squeue_line(line)
    assert r is not None
    assert r.name == "dev-gpu"
    assert r.cpus == "8"
    assert r.mem == "32G"
    assert r.gres == "gpu:1"
    assert r.gpu_count == "1"
    assert r.node == "g05"


def test_jobrow_from_pending_line_keeps_reason() -> None:
    line = "  1234569   cpufast        dev  PENDING    0:00   04:00:00     2    4G     N/A (Resources)"
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
            "JOBID PARTITION NAME STATE TIME LIMIT CPU MEM GRES NODE\n"
            "1234567 cpufast dev RUNNING 00:05 01:00:00 2 4G N/A n01\n"
            "1234568 gpufast dev-gpu RUNNING 00:10 04:00:00 8 32G gpu:1 g05\n"
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
        assert panel._rows[1].name == "dev-gpu"


async def test_shell_action_opens_folder_modal(monkeypatch) -> None:
    """Pressing `s` first pops a FolderModal so the user can choose a directory."""
    from rci_cli.tui import FolderModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setattr(
        alloc_mod, "find_strongest",
        lambda cfg: Allocation(node="n01", jobid="1234567"),
    )

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel.action_shell_into()
        await pilot.pause()
        assert isinstance(app.screen, FolderModal)


async def test_shell_attaches_after_folder_with_existing_alloc(monkeypatch) -> None:
    """Folder dismiss with existing alloc → launch_shell with the resolved folder."""
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setattr(
        alloc_mod, "find_strongest",
        lambda cfg: Allocation(node="g05", jobid="9999"),
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
        from contextlib import contextmanager

        @contextmanager
        def noop_suspend():
            yield

        monkeypatch.setattr(app, "suspend", noop_suspend)
        # Drive the folder→attach path directly, simulating modal dismissal with "sam2rl".
        panel._after_folder("shell", "sam2rl")
    assert called.get("alloc") is not None, "launch_shell was never invoked"
    assert called["alloc"].node == "g05"
    assert called["folder"] == "/home/cizekto2/sam2rl"


async def test_after_folder_with_no_alloc_pushes_new_instance_modal(monkeypatch) -> None:
    """Folder dismiss with NO existing alloc → push NewInstanceModal."""
    from rci_cli.tui import NewInstanceModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setattr(alloc_mod, "find_strongest", lambda cfg: None)

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._after_folder("shell", "")
        await pilot.pause()
        assert isinstance(app.screen, NewInstanceModal)


def test_is_gpu_partition_matches_known_prefixes() -> None:
    from rci_cli.tui import _is_gpu_partition

    for ok in ("gpu", "gpufast", "amdgpu", "amdgpufast", "h200", "h200fast", "GPU"):
        assert _is_gpu_partition(ok), ok
    for bad in ("cpu", "cpufast", "long", ""):
        assert not _is_gpu_partition(bad), bad


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
