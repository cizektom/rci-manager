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


def test_parse_jobid_from_salloc_handles_known_shapes() -> None:
    from rci_cli.tui import parse_jobid_from_salloc

    assert parse_jobid_from_salloc("salloc: Granted job allocation 1234567") == "1234567"
    assert parse_jobid_from_salloc("salloc: Pending job allocation 9999999") == "9999999"
    assert parse_jobid_from_salloc(
        "salloc: job 5555 queued and waiting for resources\n"
        "salloc: job 5555 has been allocated resources"
    ) == "5555"
    # No jobid present (salloc errored out) — caller surfaces the message instead.
    assert parse_jobid_from_salloc("salloc: error: Invalid partition: foo") is None
    assert parse_jobid_from_salloc("") is None


def test_last_meaningful_line_returns_final_non_empty_line() -> None:
    from rci_cli.tui import last_meaningful_line

    assert last_meaningful_line("a\nb\nc") == "c"
    assert last_meaningful_line("a\n\n  \nb\n\n") == "b"
    assert last_meaningful_line("") == "(no output)"
    assert last_meaningful_line("   ") == "(no output)"


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


async def test_frontend_action_sshes_to_login_host(monkeypatch) -> None:
    """Pressing ``f`` opens an interactive ssh session to ``cfg.ssh_host`` with
    no folder prompt and no allocation involved."""
    from contextlib import contextmanager

    from rci_cli import ssh as ssh_mod

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")

    calls: list[dict] = []

    def fake_run(host, remote_cmd="", *, tty=False, check=True, stdin=None):
        calls.append({"host": host, "remote_cmd": remote_cmd, "check": check})
        return 0

    monkeypatch.setattr(ssh_mod, "run", fake_run)

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)

        @contextmanager
        def noop_suspend():
            yield

        monkeypatch.setattr(app, "suspend", noop_suspend)
        panel.action_shell_frontend()

    assert len(calls) == 1, calls
    assert calls[0]["host"] == "rci"
    assert calls[0]["remote_cmd"] == ""


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


async def test_new_instance_modal_q_steps_back_when_allow_back(monkeypatch, tmp_path) -> None:
    """With ``allow_back=True`` (folder-flow), q dismisses with the ``"back"``
    sentinel so the caller can re-open the previous modal instead of dropping
    the whole flow."""
    from rci_cli.tui import AllocParams, NewInstanceModal
    from rci_cli.config import Config

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    captured: dict[str, object] = {"result": "untouched"}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        def remember(r: AllocParams | str | None) -> None:
            captured["result"] = r

        app.push_screen(NewInstanceModal(Config(), allow_back=True), remember)
        await pilot.pause()
        assert isinstance(app.screen, NewInstanceModal)
        await pilot.press("q")
        await pilot.pause()

    assert captured["result"] == "back", "q with allow_back should dismiss with 'back'"


async def test_after_new_instance_back_reopens_folder_modal(monkeypatch, tmp_path) -> None:
    """Driving the JobsPanel callback with 'back' must push FolderModal again
    so the user can edit the folder rather than losing the whole flow."""
    from rci_cli.tui import FolderModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._after_new_instance("shell", "sam2rl", "back")
        await pilot.pause()
        assert isinstance(app.screen, FolderModal)


async def test_new_instance_modal_q_cancels(monkeypatch, tmp_path) -> None:
    """``q`` on a modal escapes the modal back to the jobs panel (same as
    Escape) — it does NOT quit the app. The App's ``q``→quit fires only on
    the main screen, since modal bindings shadow it."""
    from rci_cli.tui import AllocParams, NewInstanceModal
    from rci_cli.config import Config

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    quit_calls: list[int] = []
    monkeypatch.setattr(RciApp, "action_quit", lambda self: quit_calls.append(1))

    captured: dict[str, object] = {"params": "untouched"}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        def remember(p: AllocParams | None) -> None:
            captured["params"] = p

        app.push_screen(NewInstanceModal(Config()), remember)
        await pilot.pause()
        assert isinstance(app.screen, NewInstanceModal)
        await pilot.press("q")
        await pilot.pause()

    assert captured["params"] is None, "q on the modal should dismiss with None"
    assert quit_calls == [], "q on the modal must NOT trigger app.action_quit"


async def test_new_instance_modal_q_dismisses_from_walltime(monkeypatch, tmp_path) -> None:
    """The walltime field is a Select (not an Input), so pressing ``q`` while
    it's focused must fall through to the modal's cancel binding rather than
    being typed as a character."""
    from textual.widgets import Select

    from rci_cli.tui import AllocParams, NewInstanceModal
    from rci_cli.config import Config

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    captured: dict[str, object] = {"params": "untouched"}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config()), lambda p: captured.update(params=p))
        await pilot.pause()
        # Move focus onto the walltime Select.
        time_select = app.screen.query_one("#time", Select)
        time_select.focus()
        await pilot.pause()
        assert app.focused is time_select
        await pilot.press("q")
        await pilot.pause()
    assert captured["params"] is None, "q on the walltime Select should dismiss the modal"


async def test_new_instance_modal_initial_focus_lands_in_form(monkeypatch, tmp_path) -> None:
    """Initial focus is on the first form field so Tab walks the form
    naturally (partition-type → partition-class → cores → …)."""
    from textual.widgets import Select

    from rci_cli.tui import NewInstanceModal
    from rci_cli.config import Config

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config()), lambda _p: None)
        await pilot.pause()
        scr = app.screen
        assert app.focused is scr.query_one("#partition-type", Select), (
            f"expected partition-type Select focused, got {app.focused!r}"
        )


async def test_new_instance_modal_enter_on_select_opens_dropdown(monkeypatch, tmp_path) -> None:
    """Enter while focused on a Select must open its dropdown, not submit
    the form — otherwise users can't browse partition options."""
    from textual.widgets import Select

    from rci_cli.tui import NewInstanceModal
    from rci_cli.config import Config

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    captured: list[object] = []

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config()), captured.append)
        await pilot.pause()
        scr = app.screen
        # Focus is on partition-type Select; pressing Enter should open it.
        await pilot.press("enter")
        await pilot.pause()
        # Modal must NOT have been dismissed — still on screen.
        assert isinstance(app.screen, NewInstanceModal), (
            "Enter on a Select must not submit the form"
        )
        # And nothing should have been delivered to the dismiss callback.
        assert captured == []


async def test_new_instance_modal_enter_on_input_submits(monkeypatch, tmp_path) -> None:
    """Enter from an Input field (cores / mem / gpus) submits the form —
    the priority binding routes through ``action_submit`` → ``_do_submit``."""
    from textual.widgets import Input

    from rci_cli.tui import AllocParams, NewInstanceModal
    from rci_cli.config import Config

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    captured: list[object] = []

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config()), captured.append)
        await pilot.pause()
        scr = app.screen
        scr.query_one("#cores", Input).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
    assert len(captured) == 1, "Enter on an Input must submit the form"
    assert isinstance(captured[0], AllocParams)


async def test_new_instance_modal_submit_button_uses_defaults(monkeypatch, tmp_path) -> None:
    """Clicking Submit on a fresh modal sends the Config defaults through.

    Enter while typing in an Input no longer submits the form (would have
    interrupted mid-edit); submission now goes via the Submit button — either
    a click, or focus + Enter.
    """
    from rci_cli.tui import AllocParams, NewInstanceModal
    from rci_cli.config import Config

    # Isolate state so a previously-saved set of instance params doesn't
    # override the Config defaults we're asserting on below.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    captured: dict[str, AllocParams | None] = {"params": None}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        def remember(p: AllocParams | None) -> None:
            captured["params"] = p

        app.push_screen(NewInstanceModal(Config()), remember)
        await pilot.pause()
        assert isinstance(app.screen, NewInstanceModal)
        # ``action_submit`` smart-routes (Select → open dropdown); call the
        # underlying ``_do_submit`` directly to simulate Submit-button press.
        app.screen._do_submit()
        await pilot.pause()
    assert captured["params"] is not None, "Submit did not dispatch"
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


def test_instance_params_round_trip(monkeypatch, tmp_path) -> None:
    from rci_cli import state

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert state.get_last_instance_params() is None
    state.set_last_instance_params(
        partition_type="gpu",
        partition_class="long",
        cores=8,
        gpus=2,
        mem_gb=32,
        walltime="6:00:00",
    )
    got = state.get_last_instance_params()
    assert got == {
        "partition_type": "gpu",
        "partition_class": "long",
        "cores": 8,
        "gpus": 2,
        "mem_gb": 32,
        "walltime": "6:00:00",
    }


async def test_new_instance_modal_prefills_last_params(monkeypatch, tmp_path) -> None:
    """Submitting the modal once must make a follow-up open prefill from saved state."""
    from textual.widgets import Input, Select

    from rci_cli import state
    from rci_cli.config import Config
    from rci_cli.tui import NewInstanceModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    state.set_last_instance_params(
        partition_type="gpu",
        partition_class="long",
        cores=8,
        gpus=2,
        mem_gb=32,
        walltime="6:00:00",
    )

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config()), lambda _p: None)
        await pilot.pause()
        assert isinstance(app.screen, NewInstanceModal)
        scr = app.screen
        assert scr.query_one("#partition-type", Select).value == "gpu"
        assert scr.query_one("#partition-class", Select).value == "long"
        assert scr.query_one("#cores", Input).value == "8"
        assert scr.query_one("#gpus", Input).value == "2"
        assert scr.query_one("#mem", Input).value == "32"
        # #time is a Select (not an Input) so printable keys like ``q`` fall
        # through to the modal cancel binding. Non-preset values from saved
        # state are dynamically added to the options list, so "6:00:00" still
        # round-trips even though it isn't in WALLTIME_PRESETS.
        assert scr.query_one("#time", Select).value == "6:00:00"


async def test_new_instance_modal_submit_persists_params(monkeypatch, tmp_path) -> None:
    """Submitting must write the form values to state so the next open prefills them."""
    from rci_cli import state
    from rci_cli.config import Config
    from rci_cli.tui import NewInstanceModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert state.get_last_instance_params() is None

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config()), lambda _p: None)
        await pilot.pause()
        # action_submit() would smart-route to the Select dropdown; go straight
        # to the underlying submit path.
        app.screen._do_submit()
        await pilot.pause()

    saved = state.get_last_instance_params()
    assert saved is not None
    # Defaults: cpu/fast, cpu_defaults = (2, 4, "1:00:00"), gpus hidden ⇒ 0.
    assert saved["partition_type"] == "cpu"
    assert saved["partition_class"] == "fast"
    assert saved["cores"] == 2
    assert saved["gpus"] == 0
    assert saved["mem_gb"] == 4
    assert saved["walltime"] == "1:00:00"


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


async def test_new_instance_submit_surfaces_salloc_error_not_fake_success(monkeypatch) -> None:
    """Regression: salloc errors must show as ``submit failed: …`` — not the
    earlier bug where they came through as ``submitted: salloc: error: …``."""
    from rci_cli.tui import AllocParams

    error_out = (
        "salloc: error: Job submit/allocate failed: Invalid partition name specified\n"
    )
    monkeypatch.setattr(
        slurm, "submit_cpu",
        lambda cfg, cores, mem, time, partition=None: error_out,
    )
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._do_submit(
            AllocParams(partition="bogus", cores=2, gpus=0, mem_gb=4, walltime="1:00:00")
        )
        # Let the worker thread + call_from_thread dispatch complete.
        for _ in range(20):
            await pilot.pause(0.05)
            if panel._last_action:
                break

    assert "submit failed" in panel._last_action, panel._last_action
    assert "Invalid partition" in panel._last_action, panel._last_action
    # The old bug surfaced "submitted: salloc: error: …" — make sure we don't.
    assert "submitted:" not in panel._last_action, panel._last_action


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
