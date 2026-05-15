"""Headless TUI smoke tests: app mounts, table accepts data, actions don't crash."""

from __future__ import annotations

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
    # Internal field still keeps the reason text — used by the launch-guard
    # check ``row.node.startswith('(')`` — but the table column shows ``—``.
    assert r.node == "(Resources)"
    assert r.node_display == "—"


def test_jobrow_node_display_strips_parens_reason() -> None:
    running = JobRow(
        jobid="1", partition="cpufast", name="dev", state="RUNNING",
        time="0:05", limit="1:00:00", cpus="2", mem="4G", gres="N/A", node="n07",
    )
    pending = JobRow(
        jobid="2", partition="cpufast", name="dev", state="PENDING",
        time="0:00", limit="1:00:00", cpus="2", mem="4G", gres="N/A", node="(Priority)",
    )
    assert running.node_display == "n07"
    assert pending.node_display == "—"


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

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel.action_shell_into()
        await pilot.pause()
        assert isinstance(app.screen, FolderModal)


async def test_shell_attaches_to_selected_running_row(monkeypatch) -> None:
    """Folder dismissed with a running row under the cursor → attach there."""
    monkeypatch.setattr(
        slurm,
        "list_jobs",
        lambda cfg: (
            "JOBID PARTITION NAME STATE TIME LIMIT CPU MEM GRES NODE\n"
            "111 cpufast picked RUNNING 00:05 01:00:00 2 4G N/A n42\n"
        ),
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
        for _ in range(60):
            if panel._rows:
                break
            await pilot.pause(0.05)
        from contextlib import contextmanager

        @contextmanager
        def noop_suspend():
            yield

        monkeypatch.setattr(app, "suspend", noop_suspend)
        panel._after_folder("shell", "sam2rl")
    assert called.get("alloc") is not None, "launch_shell was never invoked"
    assert called["alloc"].node == "n42"
    assert called["alloc"].jobid == "111"
    assert called["folder"] == "/home/cizekto2/sam2rl"


async def test_after_folder_with_no_usable_row_pushes_new_instance_modal(monkeypatch) -> None:
    """Empty table → New Instance modal pops, since there's nothing to attach to."""
    from rci_cli.tui import NewInstanceModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._after_folder("shell", "")
        await pilot.pause()
        assert isinstance(app.screen, NewInstanceModal)


async def test_after_folder_with_pending_row_pushes_new_instance_modal(monkeypatch) -> None:
    """A PENDING row can't be ssh'd into → fall through to the New Instance modal."""
    from rci_cli.tui import NewInstanceModal

    monkeypatch.setattr(
        slurm,
        "list_jobs",
        lambda cfg: (
            "JOBID PARTITION NAME STATE TIME LIMIT CPU MEM GRES NODE\n"
            "222 cpufast queued PENDING 0:00 01:00:00 2 4G N/A (Resources)\n"
        ),
    )

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        for _ in range(60):
            if panel._rows:
                break
            await pilot.pause(0.05)
        panel._after_folder("shell", "")
        await pilot.pause()
        assert isinstance(app.screen, NewInstanceModal)


async def test_folder_modal_prefills_last_folder(monkeypatch, tmp_path) -> None:
    """Re-opening the FolderModal must pre-fill with the previously saved value."""
    from rci_cli import state
    from rci_cli.tui import FolderModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    state.set_last_folder("sam2rl")

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel.action_shell_into()
        await pilot.pause()
        assert isinstance(app.screen, FolderModal)
        from textual.widgets import Input
        folder_input = app.screen.query_one("#folder", Input)
        assert folder_input.value == "sam2rl"


async def test_new_instance_modal_q_cancels(monkeypatch) -> None:
    """Pressing ``q`` on the New-Instance modal must close it with ``None``."""
    from rci_cli.tui import AllocParams, NewInstanceModal
    from rci_cli.config import Config

    captured: dict[str, object] = {"params": "untouched"}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        def remember(p: AllocParams | None) -> None:
            captured["params"] = p

        app.push_screen(NewInstanceModal(Config()), remember)
        await pilot.pause()
        assert isinstance(app.screen, NewInstanceModal)
        # Move focus off the partition Select (where 'q' might not bubble) so
        # the test exercises the modal binding cleanly. The first Select is
        # already a non-text widget, but pressing Tab once lands us on the
        # second one — either way q should reach the screen binding.
        await pilot.press("q")
        await pilot.pause()
    assert captured["params"] is None, "q should dismiss with None"


async def test_new_instance_modal_enter_submits_defaults(monkeypatch) -> None:
    """Regression: pressing Enter on the New-Instance modal must submit the
    form with its default values, even when focus is on the partition Select
    (where Enter would otherwise just toggle the dropdown overlay)."""
    from rci_cli.tui import AllocParams, NewInstanceModal
    from rci_cli.config import Config

    captured: dict[str, AllocParams | None] = {"params": None}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        def remember(p: AllocParams | None) -> None:
            captured["params"] = p

        app.push_screen(NewInstanceModal(Config()), remember)
        await pilot.pause()
        assert isinstance(app.screen, NewInstanceModal)
        await pilot.press("enter")
        await pilot.pause()
    assert captured["params"] is not None, "Enter did not submit the modal"
    # Defaults from Config: cpu_defaults = (2, 4, "1:00:00") + cpu/fast.
    p = captured["params"]
    assert p.partition == "cpufast"
    assert p.cores == 2
    assert p.mem_gb == 4
    assert p.walltime == "1:00:00"
    assert p.gpus == 0
    assert p.kind == "cpu"


def test_state_round_trip(monkeypatch, tmp_path) -> None:
    from rci_cli import state

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert state.get_last_folder() == ""
    state.set_last_folder("/scratch/exp")
    assert state.get_last_folder() == "/scratch/exp"
    # Overwrite works.
    state.set_last_folder("other")
    assert state.get_last_folder() == "other"


def test_assemble_partition_concatenates_type_and_class() -> None:
    from rci_cli.tui import assemble_partition

    assert assemble_partition("cpu", "fast") == "cpufast"
    assert assemble_partition("gpu", "") == "gpu"
    assert assemble_partition("amdgpu", "long") == "amdgpulong"
    assert assemble_partition("h200", "extralong") == "h200extralong"


def test_validate_alloc_rejects_cpu_with_gpus() -> None:
    from rci_cli.tui import validate_alloc

    assert validate_alloc("cpu", 0) is None
    assert validate_alloc("gpu", 1) is None
    assert validate_alloc("amdgpu", 2) is None
    assert validate_alloc("h200", 1) is None
    err = validate_alloc("cpu", 1)
    assert err is not None and "CPU" in err


def test_partition_types_and_classes_cover_known_partitions() -> None:
    """The 16 advertised partitions (gpufast … cpuextralong) must all be assemblable."""
    from rci_cli.tui import PARTITION_CLASSES, PARTITION_TYPES, assemble_partition

    expected = {
        "cpufast", "cpu", "cpulong", "cpuextralong",
        "gpufast", "gpu", "gpulong", "gpuextralong",
        "amdgpufast", "amdgpu", "amdgpulong", "amdgpuextralong",
        "h200fast", "h200", "h200long", "h200extralong",
    }
    assembled = {
        assemble_partition(t, c_value)
        for t in PARTITION_TYPES
        for _, c_value in PARTITION_CLASSES
    }
    assert assembled == expected


async def test_action_log_auto_clears(monkeypatch) -> None:
    """Regression: the inline action log must fade after ``ACTION_FADE_SECONDS``."""
    from rci_cli import tui as tui_mod

    monkeypatch.setattr(tui_mod, "ACTION_FADE_SECONDS", 0.1)
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")

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

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(5):
            await pilot.pause(0.05)
        # App should still be running — error went to a notification, not a crash.
        assert app.is_running
