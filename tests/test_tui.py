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


async def test_after_folder_with_pending_row_notifies_and_aborts(monkeypatch) -> None:
    """Highlighted PENDING row → toast and stop. Don't silently spawn a new job
    behind the user's back (would surprise: they were trying to attach)."""
    from rci_cli.tui import NewInstanceModal

    monkeypatch.setattr(
        slurm,
        "list_jobs",
        lambda cfg: (
            "JOBID PARTITION NAME STATE TIME LIMIT CPU MEM GRES NODE\n"
            "222 cpufast queued PENDING 0:00 01:00:00 2 4G N/A (Resources)\n"
        ),
    )

    notifications: list[tuple[str, str]] = []

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        original_notify = app.notify

        def capture(message, *, severity="information", timeout=5, **kw):
            notifications.append((severity, str(message)))
            return original_notify(message, severity=severity, timeout=timeout, **kw)

        monkeypatch.setattr(app, "notify", capture)

        panel = app.query_one(JobsPanel)
        for _ in range(60):
            if panel._rows:
                break
            await pilot.pause(0.05)
        panel._after_folder("shell", "")
        await pilot.pause()
        assert not isinstance(app.screen, NewInstanceModal), \
            "pending row must not silently fall through to New Instance"
        assert any(
            sev == "warning" and "can't attach" in msg for sev, msg in notifications
        ), notifications


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


async def test_new_instance_modal_rejects_short_walltime_from_state(monkeypatch, tmp_path) -> None:
    """Regression: state.json from an older session may contain a free-form
    walltime like ``"0:00:10"``. Slurm parses that as 10 seconds and rounds up
    to the 1-minute minimum, silently shrinking the user's job. The modal must
    discard such non-preset values and fall back to the config default."""
    from textual.widgets import Select

    from rci_cli import state
    from rci_cli.config import Config
    from rci_cli.tui import NewInstanceModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    state.set_last_instance_params(
        partition_type="cpu",
        partition_class="fast",
        cores=2,
        gpus=0,
        mem_gb=4,
        walltime="0:00:10",  # bogus — 10 seconds, was reachable via old Input
    )

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config()), lambda _p: None)
        await pilot.pause()
        # Falls back to the Config default cpu_defaults[-1] = "1:00:00".
        assert app.screen.query_one("#time", Select).value == "1:00:00"


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


async def test_new_instance_modal_opens_with_no_widget_focused(monkeypatch, tmp_path) -> None:
    """The modal opens with no inner widget focused. Enter submits the
    prefilled defaults; Tab walks into the form starting at partition-type."""
    from rci_cli.tui import NewInstanceModal
    from rci_cli.config import Config

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config()), lambda _p: None)
        await pilot.pause()
        # Either nothing is focused, or focus is the screen itself (Textual's
        # "no inner widget" state). Either way, it must not be any of the
        # modal's form widgets or buttons.
        scr = app.screen
        form_ids = {"job-name", "partition-type", "partition-class", "cores", "gpus", "mem", "time", "ok", "cancel"}
        focused_id = getattr(app.focused, "id", None)
        assert focused_id not in form_ids, (
            f"expected no inner widget focused, got {focused_id}"
        )


async def test_new_instance_modal_initial_enter_submits_defaults(monkeypatch, tmp_path) -> None:
    """Pressing Enter immediately after opening (no inner focus) submits the
    pre-filled defaults — the canonical "I accept these values" path."""
    from rci_cli.tui import AllocParams, NewInstanceModal
    from rci_cli.config import Config

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    captured: list[object] = []

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config()), captured.append)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
    assert len(captured) == 1, "initial Enter must submit"
    assert isinstance(captured[0], AllocParams)


async def test_new_instance_modal_enter_on_select_keeps_modal_open(monkeypatch, tmp_path) -> None:
    """Enter on a Select must not dismiss the modal — it opens the dropdown
    (or otherwise keeps the user in the form to pick an option)."""
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
        scr.query_one("#partition-type", Select).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        # Modal is still up; no AllocParams (or anything else) was returned.
        assert isinstance(app.screen, NewInstanceModal)
        assert captured == []


async def test_new_instance_modal_enter_on_input_advances_focus(monkeypatch, tmp_path) -> None:
    """Enter on a numeric Input (cores/mem/gpus) advances to the next field
    — same affordance as Tab, no dismissal of the modal."""
    from textual.widgets import Input

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
        cores = scr.query_one("#cores", Input)
        cores.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        # Modal still up, nothing dismissed, focus moved off Cores.
        assert isinstance(app.screen, NewInstanceModal)
        assert captured == []
        assert app.focused is not cores, "Enter on Input must advance focus"


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
        # Must be a value in WALLTIME_PRESETS — non-preset walltimes are
        # discarded on load (see test_new_instance_modal_rejects_short_walltime_from_state).
        walltime="8:00:00",
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
        # through to the modal cancel binding.
        assert scr.query_one("#time", Select).value == "8:00:00"


async def test_new_instance_modal_prefills_default_name(monkeypatch, tmp_path) -> None:
    """The Name Input is pre-filled with the caller-supplied suggestion (e.g. ``dev-3``)."""
    from textual.widgets import Input

    from rci_cli.config import Config
    from rci_cli.tui import NewInstanceModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config(), default_name="dev-3"), lambda _p: None)
        await pilot.pause()
        assert app.screen.query_one("#job-name", Input).value == "dev-3"


async def test_new_instance_modal_blank_name_reverts_to_suggestion(
    monkeypatch, tmp_path
) -> None:
    """Blanking the Name field doesn't submit a nameless job — it reverts to
    the suggestion that was prefilled."""
    from textual.widgets import Input

    from rci_cli.config import Config
    from rci_cli.tui import AllocParams, NewInstanceModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    captured: dict[str, AllocParams | None] = {"params": None}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(
            NewInstanceModal(Config(), default_name="editor"),
            lambda p: captured.update(params=p),
        )
        await pilot.pause()
        scr = app.screen
        scr.query_one("#job-name", Input).value = ""  # user blanks it
        scr._do_submit()
        await pilot.pause()

    p = captured["params"]
    assert p is not None
    assert p.job_name == "editor"


async def test_new_instance_modal_custom_name_is_respected(
    monkeypatch, tmp_path
) -> None:
    from textual.widgets import Input

    from rci_cli.config import Config
    from rci_cli.tui import AllocParams, NewInstanceModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    captured: dict[str, AllocParams | None] = {"params": None}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(
            NewInstanceModal(Config(), default_name="dev-1"),
            lambda p: captured.update(params=p),
        )
        await pilot.pause()
        scr = app.screen
        scr.query_one("#job-name", Input).value = "my-experiment"
        scr._do_submit()
        await pilot.pause()

    assert captured["params"].job_name == "my-experiment"


async def test_editor_flow_suggests_singleton_editor_name(monkeypatch, tmp_path) -> None:
    """Empty table + Editor action → New Instance modal pre-fills with ``editor``."""
    from textual.widgets import Input

    from rci_cli.tui import NewInstanceModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._after_folder("editor", "")
        await pilot.pause()
        assert isinstance(app.screen, NewInstanceModal)
        assert app.screen.query_one("#job-name", Input).value == "editor"


async def test_suggest_dev_name_uses_cached_rows_with_gap_reuse(monkeypatch) -> None:
    """``_suggest_dev_name`` reuses the lowest gap in cached rows — no ssh
    round-trip. Cancelled jobs naturally disappear from the cache so their
    numbers come back into play."""
    from rci_cli.config import Config
    from rci_cli.tui import JobRow

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        cfg = Config()
        # dev-1 and dev-3 are taken; dev-2 is the lowest unused.
        panel._rows = [
            JobRow(
                jobid="111", partition="cpufast", name="dev-1", state="RUNNING",
                time="0:05", limit="1:00:00", cpus="2", mem="4G", gres="N/A", node="n01",
            ),
            JobRow(
                jobid="333", partition="cpufast", name="dev-3", state="RUNNING",
                time="0:01", limit="1:00:00", cpus="2", mem="4G", gres="N/A", node="n03",
            ),
        ]
        assert panel._suggest_dev_name(cfg) == "dev-2"
        # Empty cache ⇒ start at dev-1.
        panel._rows = []
        assert panel._suggest_dev_name(cfg) == "dev-1"


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


def test_validate_alloc_rejects_non_gpu_type_with_gpus() -> None:
    from rci_cli.tui import validate_alloc

    gpu_types = ("gpu", "amdgpu", "h200")
    # No GPUs requested → any partition type is fine.
    assert validate_alloc("cpu", 0, gpu_types) is None
    # GPU types accept gpus > 0.
    assert validate_alloc("gpu", 1, gpu_types) is None
    assert validate_alloc("amdgpu", 2, gpu_types) is None
    assert validate_alloc("h200", 1, gpu_types) is None
    # Anything outside the configured gpu_types list rejects gpus > 0.
    err = validate_alloc("cpu", 1, gpu_types)
    assert err is not None and "doesn't accept GPUs" in err
    # A different cluster could allow GPUs only on a single type — works.
    err = validate_alloc("cpu", 1, ("v100",))
    assert err is not None and "v100" in err


def test_partition_types_and_classes_cover_known_partitions() -> None:
    """The 16 advertised partitions (gpufast … cpuextralong) must all be assemblable
    from the ``Config`` defaults — defaults still match the RCI cluster."""
    from rci_cli.config import Config
    from rci_cli.tui import assemble_partition

    cfg = Config()
    expected = {
        "cpufast", "cpu", "cpulong", "cpuextralong",
        "gpufast", "gpu", "gpulong", "gpuextralong",
        "amdgpufast", "amdgpu", "amdgpulong", "amdgpuextralong",
        "h200fast", "h200", "h200long", "h200extralong",
    }
    assembled = {
        assemble_partition(t, c_value)
        for t in cfg.partition_types
        for _, c_value in cfg.partition_classes
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
        lambda cfg, cores, mem, time, *, job_name, partition=None: error_out,
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


async def test_confirm_modal_dangerous_focuses_no(monkeypatch) -> None:
    """Safe default for dangerous=True confirms is the ``No`` button —
    Enter must abort, not destroy."""
    from textual.widgets import Button

    from rci_cli.tui import ConfirmModal

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(ConfirmModal("Really cancel?", dangerous=True), lambda _b: None)
        await pilot.pause()
        scr = app.screen
        assert app.focused is scr.query_one("#no", Button)


async def test_confirm_modal_neutral_also_focuses_no(monkeypatch) -> None:
    """Even non-dangerous confirms default to ``No`` — Enter never confirms
    on reflex. To proceed, the user must Tab to ``Yes`` (or press ``y``)."""
    from textual.widgets import Button

    from rci_cli.tui import ConfirmModal

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(ConfirmModal("Proceed?"), lambda _b: None)
        await pilot.pause()
        scr = app.screen
        assert app.focused is scr.query_one("#no", Button)


async def test_confirm_modal_default_yes_focuses_yes(monkeypatch) -> None:
    """``default_yes=True`` opts a caller out of the safe default — the kill
    dialog uses this so Enter confirms the cancel."""
    from textual.widgets import Button

    from rci_cli.tui import ConfirmModal

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(
            ConfirmModal("Kill?", dangerous=True, default_yes=True), lambda _b: None
        )
        await pilot.pause()
        scr = app.screen
        assert app.focused is scr.query_one("#yes", Button)


# ── SetupModal / first-run flow ─────────────────────────────────────────────


async def test_setup_modal_pops_on_first_run(monkeypatch, tmp_path) -> None:
    """No config file ⇒ ``RciApp.on_mount`` stacks the SetupModal on top of
    the dashboard. Without this the JobsPanel would try to ``squeue -u ''``."""
    from rci_cli.tui import SetupModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    # Override the autouse conftest fixture — point at an empty config dir.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, SetupModal)


async def test_setup_modal_save_writes_config_and_refreshes(
    monkeypatch, tmp_path
) -> None:
    """Filling the modal and pressing Save must (1) write the TOML file and
    (2) trigger a JobsPanel refresh that now sees a valid user."""
    from textual.widgets import Input

    from rci_cli.tui import JobsPanel, SetupModal

    cfg_root = tmp_path / "fresh-config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_root))

    list_jobs_calls: list = []

    def fake_list_jobs(cfg):
        list_jobs_calls.append(cfg.user)
        return ""

    monkeypatch.setattr(slurm, "list_jobs", fake_list_jobs)

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, SetupModal)
        # JobsPanel was composed at mount under the modal — confirm it's there.
        assert app.query_one(JobsPanel) is not None
        scr = app.screen
        scr.query_one("#setup-user", Input).value = "alice"
        # ssh-host pre-fills to "rci"; home blank ⇒ /home/alice; venv blank
        scr._do_save()
        # Give the dismiss + after-setup refresh a chance to land.
        for _ in range(20):
            await pilot.pause(0.05)
            if list_jobs_calls:
                break

        saved = cfg_root / "rci-cli" / "config.toml"
        assert saved.exists()
        assert 'user = "alice"' in saved.read_text()
        # The post-save refresh hit slurm with the freshly-saved user.
        assert list_jobs_calls == ["alice"], list_jobs_calls


async def test_setup_modal_cancel_exits_app(monkeypatch, tmp_path) -> None:
    """Esc / Cancel must end the session — the app can't do anything
    without a configured user."""
    from rci_cli.tui import SetupModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, SetupModal)
        app.screen.dismiss(False)
        # Give the after-setup callback (calls self.exit()) a chance to run.
        for _ in range(10):
            await pilot.pause(0.05)
            if not app.is_running:
                break
    assert not app.is_running


async def test_setup_modal_save_requires_user(monkeypatch, tmp_path) -> None:
    """Empty user must keep the modal open (no silent dismiss)."""
    from textual.widgets import Input

    from rci_cli.tui import SetupModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, SetupModal)
        scr = app.screen
        scr.query_one("#setup-user", Input).value = ""
        scr._do_save()
        await pilot.pause()
        # Still on the SetupModal.
        assert isinstance(app.screen, SetupModal)


async def test_dashboard_skips_setup_when_already_configured(monkeypatch) -> None:
    """With a valid config (autouse conftest), the SetupModal must NOT pop —
    the JobsPanel takes the screen as usual."""
    from rci_cli.tui import JobsPanel, SetupModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not isinstance(app.screen, SetupModal)
        # Dashboard panel is mounted.
        assert app.query_one(JobsPanel) is not None
