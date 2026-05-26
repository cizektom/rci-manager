"""Headless TUI smoke tests: app mounts, table accepts data, actions don't crash.

Pure unit tests for ``slurm.py`` helpers re-exported by ``tui`` live in
``test_slurm.py``; tests here exercise the live app via Textual's pilot.
"""

from __future__ import annotations

import pytest

from rci_cli import alloc as alloc_mod
from rci_cli import launch
from rci_cli import slurm
from rci_cli.alloc import Allocation
from rci_cli.tui import JobRow, JobsPanel, RciApp


# ── JobRow parser (pure unit) ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "line,name,state,cpus,mem,gres,gpu_count,node,node_display",
    [
        # CPU running — node column shows real hostname.
        (
            "  1234567 cpufast dev RUNNING 00:05 01:00:00 2 4G N/A n01",
            "dev", "RUNNING", "2", "4G", "N/A", "—", "n01", "n01",
        ),
        # GPU running — ``gres=gpu:1`` exposes the count in the compact column.
        (
            "  1234568 gpufast dev-gpu RUNNING 00:10 04:00:00 8 32G gpu:1 g05",
            "dev-gpu", "RUNNING", "8", "32G", "gpu:1", "1", "g05", "g05",
        ),
        # Pending — ``%R`` puts the reason in the node column; display dashes it
        # out, but the raw field stays so the launch guard can still see ``(``.
        (
            "  1234569 cpufast dev PENDING 0:00 04:00:00 2 4G N/A (Resources)",
            "dev", "PENDING", "2", "4G", "N/A", "—", "(Resources)", "—",
        ),
    ],
)
def test_jobrow_parses_squeue_line(
    line, name, state, cpus, mem, gres, gpu_count, node, node_display
) -> None:
    r = JobRow.from_squeue_line(line)
    assert r is not None
    assert (r.name, r.state, r.cpus, r.mem, r.gres) == (name, state, cpus, mem, gres)
    assert r.gpu_count == gpu_count
    assert r.node == node
    assert r.node_display == node_display


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
    pre-filled defaults — the canonical "I accept these values" path. Also
    asserts the default values that flow through, so a Config drift doesn't
    quietly produce a submission with stale values."""
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
    p = captured[0]
    assert isinstance(p, AllocParams)
    # Defaults from Config: cpu_defaults = (2, 4, "1:00:00") + cpu/fast.
    assert (p.partition, p.cores, p.gpus, p.mem_gb, p.walltime, p.kind) == (
        "cpufast", 2, 0, 4, "1:00:00", "cpu",
    )


async def test_vim_keys_navigate_open_select_dropdown(monkeypatch, tmp_path) -> None:
    """VimSelect: with the dropdown open, j/k move the OptionList highlight,
    G jumps to the last option. Closed-state j is a no-op."""
    from textual.widgets import OptionList, Select

    from rci_cli.tui import NewInstanceModal, VimSelect
    from rci_cli.config import Config

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NewInstanceModal(Config()), lambda _r: None)
        await pilot.pause()
        scr = app.screen
        # The partition-type Select has at least 2 options so j has somewhere
        # to move; default value puts the highlight on the first one.
        select = scr.query_one("#partition-type", Select)
        assert isinstance(select, VimSelect)

        # Closed Select + j ⇒ no-op (dropdown stays closed, no error).
        select.focus()
        await pilot.pause()
        assert not select.expanded
        await pilot.press("j")
        await pilot.pause()
        assert not select.expanded

        # Open the dropdown; capture the starting highlight then walk it.
        await pilot.press("enter")
        await pilot.pause()
        assert select.expanded
        overlay = select.query_one(OptionList)
        n_options = overlay.option_count
        assert n_options >= 2, "test assumes >=2 partition types"
        start = overlay.highlighted

        await pilot.press("j")
        await pilot.pause()
        assert overlay.highlighted == start + 1

        await pilot.press("k")
        await pilot.pause()
        assert overlay.highlighted == start

        # Capital G jumps to the last option (Shift+g).
        await pilot.press("G")
        await pilot.pause()
        assert overlay.highlighted == n_options - 1

        # Lowercase g jumps back to the first.
        await pilot.press("g")
        await pilot.pause()
        assert overlay.highlighted == 0


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


@pytest.mark.parametrize(
    "typed_name,expected_name",
    [
        # Blank reverts to the suggestion — no way to submit a nameless job.
        ("", "editor"),
        # Custom user-typed name wins.
        ("my-experiment", "my-experiment"),
    ],
)
async def test_new_instance_modal_name_resolution(
    monkeypatch, tmp_path, typed_name, expected_name
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
            NewInstanceModal(Config(), default_name="editor"),
            lambda p: captured.update(params=p),
        )
        await pilot.pause()
        scr = app.screen
        scr.query_one("#job-name", Input).value = typed_name
        scr._do_submit()
        await pilot.pause()

    assert captured["params"] is not None
    assert captured["params"].job_name == expected_name


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


async def test_filter_narrows_visible_rows(monkeypatch) -> None:
    """``/`` filter does substring matching on jobid/name/state/partition;
    Escape (``action_cancel_filter``) clears it and restores all rows."""
    from textual.widgets import Input

    monkeypatch.setattr(
        slurm,
        "list_jobs",
        lambda cfg: (
            "JOBID PARTITION NAME STATE TIME LIMIT CPU MEM GRES NODE\n"
            "1234567 cpufast dev RUNNING 00:05 01:00:00 2 4G N/A n01\n"
            "1234568 gpufast train RUNNING 00:10 04:00:00 8 32G gpu:1 g05\n"
            "1234569 cpu train-llama PENDING 0:00 24:00:00 16 64G N/A (Resources)\n"
        ),
    )
    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(5):
            await pilot.pause(0.05)
        panel = app.query_one(JobsPanel)
        assert len(panel._visible_rows) == 3

        # Setting Input.value triggers the panel's Input.Changed handler via
        # Textual's reactive system; ``pause`` lets the message land.
        panel.action_start_filter()
        inp = panel.query_one("#filter-input", Input)
        inp.value = "train"
        await pilot.pause()
        assert {r.name for r in panel._visible_rows} == {"train", "train-llama"}
        # Full snapshot is untouched — status line still counts all 3.
        assert len(panel._rows) == 3

        panel.action_cancel_filter()
        await pilot.pause()
        assert panel._filter == ""
        assert len(panel._visible_rows) == 3
        assert inp.display is False


async def test_filter_preserves_cursor_on_matching_row(monkeypatch) -> None:
    """If the highlighted job survives the filter, the cursor follows it
    rather than snapping to row 0 of the filtered view."""
    from textual.widgets import DataTable

    monkeypatch.setattr(
        slurm,
        "list_jobs",
        lambda cfg: (
            "JOBID PARTITION NAME STATE TIME LIMIT CPU MEM GRES NODE\n"
            "1 cpufast alpha RUNNING 0:00 1:00:00 2 4G N/A n01\n"
            "2 cpufast train RUNNING 0:00 1:00:00 2 4G N/A n02\n"
            "3 cpufast zulu RUNNING 0:00 1:00:00 2 4G N/A n03\n"
        ),
    )
    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(5):
            await pilot.pause(0.05)
        panel = app.query_one(JobsPanel)
        table = panel.query_one("#jobs-table", DataTable)
        # Highlight ``train`` (row index 1) before filtering.
        table.move_cursor(row=1)
        await pilot.pause()
        assert panel._selected_row().name == "train"

        panel._filter = "train"
        panel._render_rows()
        await pilot.pause()
        # Filtered view has 1 row, cursor stayed on the same job.
        assert panel._selected_row().name == "train"


async def test_cursor_top_and_bottom_jump(monkeypatch) -> None:
    """``g`` / ``G`` jump to the first / last visible row."""
    from textual.widgets import DataTable

    monkeypatch.setattr(
        slurm,
        "list_jobs",
        lambda cfg: (
            "JOBID PARTITION NAME STATE TIME LIMIT CPU MEM GRES NODE\n"
            "1 cpufast a RUNNING 0:00 1:00:00 2 4G N/A n01\n"
            "2 cpufast b RUNNING 0:00 1:00:00 2 4G N/A n02\n"
            "3 cpufast c RUNNING 0:00 1:00:00 2 4G N/A n03\n"
        ),
    )
    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(5):
            await pilot.pause(0.05)
        panel = app.query_one(JobsPanel)
        table = panel.query_one("#jobs-table", DataTable)

        panel.action_cursor_bottom()
        await pilot.pause()
        assert table.cursor_row == 2

        panel.action_cursor_top()
        await pilot.pause()
        assert table.cursor_row == 0


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


@pytest.mark.parametrize(
    "kwargs,expected_id",
    [
        # Default: ``No``. Enter on reflex must abort.
        ({}, "no"),
        # ``dangerous=True`` still defaults to ``No`` — the red Yes button is
        # the affordance, not auto-focus.
        ({"dangerous": True}, "no"),
        # Opt-in: ``default_yes`` is used by the cancel-job dialog so Enter
        # confirms the cancellation.
        ({"dangerous": True, "default_yes": True}, "yes"),
    ],
)
async def test_confirm_modal_default_focus(kwargs, expected_id) -> None:
    from textual.widgets import Button

    from rci_cli.tui import ConfirmModal

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(ConfirmModal("prompt?", **kwargs), lambda _b: None)
        await pilot.pause()
        assert app.focused is app.screen.query_one(f"#{expected_id}", Button)


# ── SetupModal / first-run flow ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "configured,expect_setup",
    [
        # Empty XDG_CONFIG_HOME ⇒ SetupModal pops; otherwise JobsPanel would
        # try to ``squeue -u ''``.
        (False, True),
        # Autouse conftest provides a valid config ⇒ dashboard skips setup.
        (True, False),
    ],
)
async def test_setup_modal_pops_only_when_unconfigured(
    monkeypatch, tmp_path, configured, expect_setup
) -> None:
    from rci_cli.tui import JobsPanel, SetupModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    if not configured:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, SetupModal) is expect_setup
        # JobsPanel is composed under whichever screen is up — must always exist.
        assert app.query_one(JobsPanel) is not None


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


# ── Agent flow ──────────────────────────────────────────────────────────────


async def test_suggest_agent_name_uses_cached_rows_with_gap_reuse(monkeypatch) -> None:
    """``_suggest_agent_name`` mirrors ``_suggest_dev_name`` but for the agent
    pool — separate counter from dev-N."""
    from rci_cli.config import Config
    from rci_cli.tui import JobRow

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        cfg = Config()
        # Mix of dev and agent jobs in the cache. dev-1 must not bump the
        # agent counter; agent-1 is free, agent-2 is taken.
        panel._rows = [
            JobRow(
                jobid="111", partition="cpufast", name="dev-1", state="RUNNING",
                time="0:05", limit="1:00:00", cpus="2", mem="4G", gres="N/A", node="n01",
            ),
            JobRow(
                jobid="222", partition="cpufast", name="agent-2", state="RUNNING",
                time="0:01", limit="1:00:00", cpus="2", mem="4G", gres="N/A", node="n02",
            ),
        ]
        assert panel._suggest_agent_name(cfg) == "agent-1"
        panel._rows = []
        assert panel._suggest_agent_name(cfg) == "agent-1"


async def test_agent_flow_opens_agent_options_modal_first(monkeypatch, tmp_path) -> None:
    """Pressing 'a' (Agent) opens ``AgentOptionsModal`` before the resources
    modal — that's the new 3-step ordering."""
    from textual.widgets import Input, Select

    from rci_cli.tui import AgentOptionsModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._after_folder("agent", "")
        await pilot.pause()
        assert isinstance(app.screen, AgentOptionsModal)
        scr = app.screen
        # Prefilled from cfg defaults.
        assert scr.query_one("#agent-permission-mode", Select).value == "default"
        assert scr.query_one("#agent-spawn-mode", Select).value == "same-dir"
        assert scr.query_one("#agent-capacity", Input).value == "32"


async def test_agent_options_modal_returns_options(monkeypatch, tmp_path) -> None:
    """Submitting the agent-options modal returns an :class:`AgentOptions`
    with the user's choices."""
    from textual.widgets import Input, Select

    from rci_cli.config import Config
    from rci_cli.tui import AgentOptions, AgentOptionsModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    captured: dict[str, AgentOptions | None] = {"opts": None}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(
            AgentOptionsModal(Config()),
            lambda o: captured.update(opts=o),
        )
        await pilot.pause()
        scr = app.screen
        scr.query_one("#agent-permission-mode", Select).value = "bypassPermissions"
        scr.query_one("#agent-spawn-mode", Select).value = "worktree"
        scr.query_one("#agent-capacity", Input).value = "8"
        scr._do_submit()
        await pilot.pause()

    o = captured["opts"]
    assert o is not None
    assert o.permission_mode == "bypassPermissions"
    assert o.spawn_mode == "worktree"
    assert o.capacity == 8


async def test_agent_options_modal_blank_capacity_falls_back_to_cfg(
    monkeypatch, tmp_path
) -> None:
    """Blank/0 capacity ⇒ restore ``cfg.agent_capacity`` rather than sending
    0 to claude."""
    from textual.widgets import Input

    from rci_cli.config import Config
    from rci_cli.tui import AgentOptions, AgentOptionsModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    captured: dict[str, AgentOptions | None] = {"opts": None}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(
            AgentOptionsModal(Config()),
            lambda o: captured.update(opts=o),
        )
        await pilot.pause()
        scr = app.screen
        scr.query_one("#agent-capacity", Input).value = ""
        scr._do_submit()
        await pilot.pause()

    assert captured["opts"].capacity == 32  # cfg.agent_capacity default


async def test_agent_options_step_advances_to_resources_modal(
    monkeypatch, tmp_path
) -> None:
    """After AgentOptionsModal submits, the resources modal opens with the
    suggested ``agent-N`` name and no agent fields in it."""
    from textual.css.query import NoMatches
    from textual.widgets import Input, Select

    from rci_cli.tui import AgentOptions, AgentOptionsModal, NewInstanceModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._after_folder("agent", "")
        await pilot.pause()
        assert isinstance(app.screen, AgentOptionsModal)
        scr = app.screen
        scr.query_one("#agent-permission-mode", Select).value = "bypassPermissions"
        scr.query_one("#agent-capacity", Input).value = "16"
        scr._do_submit()
        await pilot.pause()
        # Next screen is the resources modal — agent fields are absent here.
        assert isinstance(app.screen, NewInstanceModal)
        res = app.screen
        assert res.query_one("#job-name", Input).value == "agent-1"
        import pytest
        with pytest.raises(NoMatches):
            res.query_one("#agent-permission-mode")
        # The panel cached the AgentOptions so step-back can re-prefill them.
        assert panel._pending_agent_opts is not None
        assert panel._pending_agent_opts.permission_mode == "bypassPermissions"
        assert panel._pending_agent_opts.capacity == 16


async def test_agent_options_modal_back_replays_folder_picker(
    monkeypatch, tmp_path
) -> None:
    """q/escape on the agent-options modal (allow_back=True) re-opens the
    folder picker instead of dropping the flow."""
    from rci_cli.tui import AgentOptionsModal, FolderModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._after_folder("agent", "")
        await pilot.pause()
        assert isinstance(app.screen, AgentOptionsModal)
        app.screen.dismiss("back")
        await pilot.pause()
        assert isinstance(app.screen, FolderModal)


async def test_agent_options_modal_prefills_from_pending_opts(
    monkeypatch, tmp_path
) -> None:
    """If the panel has cached AgentOptions (user stepped back from
    resources), the next AgentOptionsModal pre-fills with them."""
    from textual.widgets import Input, Select

    from rci_cli.config import Config
    from rci_cli.tui import AgentOptions, AgentOptionsModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._pending_agent_opts = AgentOptions(
            permission_mode="bypassPermissions",
            spawn_mode="worktree",
            capacity=8,
        )
        panel._after_folder("agent", "")
        await pilot.pause()
        assert isinstance(app.screen, AgentOptionsModal)
        scr = app.screen
        assert scr.query_one("#agent-permission-mode", Select).value == "bypassPermissions"
        assert scr.query_one("#agent-spawn-mode", Select).value == "worktree"
        assert scr.query_one("#agent-capacity", Input).value == "8"


async def test_agent_kind_bypasses_pending_row_guard(monkeypatch, tmp_path) -> None:
    """Agent always creates a new alloc — the pending-row guard that blocks
    shell/editor must not fire for the agent flow."""
    from rci_cli.tui import AgentOptionsModal, JobRow

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        # Even with a pending row highlighted, agent flow must proceed.
        panel._rows = [
            JobRow(
                jobid="999", partition="cpufast", name="dev-1", state="PENDING",
                time="0:00", limit="1:00:00", cpus="2", mem="4G", gres="N/A",
                node="(Resources)",
            )
        ]
        panel._after_folder("agent", "")
        await pilot.pause()
        assert isinstance(app.screen, AgentOptionsModal)


# ── Workspace flow ──────────────────────────────────────────────────────────


async def test_workspace_flow_opens_workspace_options_modal_first(
    monkeypatch, tmp_path
) -> None:
    """Pressing 'w' (Workspace) opens ``WorkspaceOptionsModal`` before the
    resources modal — same 3-step ordering as the agent flow."""
    from textual.widgets import Input

    from rci_cli.tui import WorkspaceOptionsModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._after_folder("workspace", "")
        await pilot.pause()
        assert isinstance(app.screen, WorkspaceOptionsModal)
        scr = app.screen
        # Prefilled from cfg defaults (2 agents + 1 terminal).
        assert scr.query_one("#workspace-agents", Input).value == "2"
        assert scr.query_one("#workspace-terminals", Input).value == "1"


async def test_workspace_options_modal_returns_options(monkeypatch, tmp_path) -> None:
    """Submitting the workspace-options modal returns a :class:`WorkspaceOptions`
    with the user's pane counts."""
    from textual.widgets import Input

    from rci_cli.config import Config
    from rci_cli.tui import WorkspaceOptions, WorkspaceOptionsModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    captured: dict[str, WorkspaceOptions | None] = {"opts": None}

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(
            WorkspaceOptionsModal(Config()),
            lambda o: captured.update(opts=o),
        )
        await pilot.pause()
        scr = app.screen
        scr.query_one("#workspace-agents", Input).value = "3"
        scr.query_one("#workspace-terminals", Input).value = "2"
        scr._do_submit()
        await pilot.pause()

    o = captured["opts"]
    assert o is not None
    assert o.agents == 3
    assert o.terminals == 2


async def test_workspace_options_modal_rejects_all_zero(monkeypatch, tmp_path) -> None:
    """``agents=0, terminals=0`` is rejected with a notification rather than
    silently backfilling — the user almost certainly meant something else."""
    from textual.widgets import Input

    from rci_cli.config import Config
    from rci_cli.tui import WorkspaceOptions, WorkspaceOptionsModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    captured: dict[str, object] = {"called": False}

    def _capture(o: WorkspaceOptions | None) -> None:
        captured["called"] = True

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(WorkspaceOptionsModal(Config()), _capture)
        await pilot.pause()
        scr = app.screen
        scr.query_one("#workspace-agents", Input).value = "0"
        scr.query_one("#workspace-terminals", Input).value = "0"
        scr._do_submit()
        await pilot.pause()
        # Modal stays up — no dismiss callback fired.
        assert isinstance(app.screen, WorkspaceOptionsModal)

    assert captured["called"] is False


async def test_workspace_options_modal_persists_across_sessions(
    monkeypatch, tmp_path
) -> None:
    """Submitting writes the pane counts to state.json; a fresh modal
    prefills with them (remembered defaults across launches)."""
    from textual.widgets import Input

    from rci_cli.config import Config
    from rci_cli.tui import WorkspaceOptionsModal

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(WorkspaceOptionsModal(Config()), lambda _: None)
        await pilot.pause()
        scr = app.screen
        scr.query_one("#workspace-agents", Input).value = "4"
        scr.query_one("#workspace-terminals", Input).value = "2"
        scr._do_submit()
        await pilot.pause()
        # Re-open: should pre-fill 4/2, not the cfg defaults.
        app.push_screen(WorkspaceOptionsModal(Config()), lambda _: None)
        await pilot.pause()
        scr2 = app.screen
        assert scr2.query_one("#workspace-agents", Input).value == "4"
        assert scr2.query_one("#workspace-terminals", Input).value == "2"


async def test_workspace_options_step_advances_to_resources_modal(
    monkeypatch, tmp_path
) -> None:
    """After WorkspaceOptionsModal submits with no running row highlighted,
    the resources modal opens with the suggested ``workspace-N`` name."""
    from textual.widgets import Input

    from rci_cli.tui import NewInstanceModal, WorkspaceOptionsModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._after_folder("workspace", "")
        await pilot.pause()
        assert isinstance(app.screen, WorkspaceOptionsModal)
        scr = app.screen
        scr.query_one("#workspace-agents", Input).value = "3"
        scr.query_one("#workspace-terminals", Input).value = "2"
        scr._do_submit()
        await pilot.pause()
        # Next screen is the resources modal with the workspace suggestion.
        assert isinstance(app.screen, NewInstanceModal)
        res = app.screen
        assert res.query_one("#job-name", Input).value == "workspace-1"
        # Panel cached the WorkspaceOptions so step-back can re-prefill them.
        assert panel._pending_workspace_opts is not None
        assert panel._pending_workspace_opts.agents == 3
        assert panel._pending_workspace_opts.terminals == 2


async def test_workspace_options_modal_back_replays_folder_picker(
    monkeypatch, tmp_path
) -> None:
    """q/escape on the workspace-options modal (allow_back=True) re-opens
    the folder picker instead of dropping the flow."""
    from rci_cli.tui import FolderModal, WorkspaceOptionsModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._after_folder("workspace", "")
        await pilot.pause()
        assert isinstance(app.screen, WorkspaceOptionsModal)
        app.screen.dismiss("back")
        await pilot.pause()
        assert isinstance(app.screen, FolderModal)


async def test_workspace_options_modal_prefills_from_pending_opts(
    monkeypatch, tmp_path
) -> None:
    """If the panel has cached WorkspaceOptions (user stepped back from
    resources), the next WorkspaceOptionsModal pre-fills with them."""
    from textual.widgets import Input

    from rci_cli.tui import WorkspaceOptions, WorkspaceOptionsModal

    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    app = RciApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(JobsPanel)
        panel._pending_workspace_opts = WorkspaceOptions(agents=4, terminals=3)
        panel._after_folder("workspace", "")
        await pilot.pause()
        assert isinstance(app.screen, WorkspaceOptionsModal)
        scr = app.screen
        assert scr.query_one("#workspace-agents", Input).value == "4"
        assert scr.query_one("#workspace-terminals", Input).value == "3"


