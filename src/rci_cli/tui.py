"""Interactive Textual TUI for rci-cli.

Live jobs dashboard with one-key actions: cancel selected, shell into the
compute node, launch claude or VS Code on the allocation, submit fresh CPU /
GPU allocations from a modal. Refresh is threaded so the UI never freezes
while ``squeue`` is in flight; ``App.suspend()`` is used to hand the terminal
over to ssh for shell-in / claude attach, then re-render on return.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)

from . import alloc as alloc_mod
from . import launch, slurm, state
from .config import Config, load

REFRESH_INTERVAL = 5.0
ACTION_FADE_SECONDS = 6.0  # how long the inline action log lingers before auto-clearing

# squeue's ``%T`` long-form yields ``RUNNING``/``PENDING``/…; some setups still use
# the short codes (``R``/``PD``). Accept both so the action guards are robust.
RUNNING_STATES = frozenset({"R", "RUNNING"})


# ──────────────────────────── modals ────────────────────────────────────────


class ConfirmModal(ModalScreen[bool]):
    """Generic yes/no confirmation. Returns True/False via ``dismiss``."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "yes", "Yes", show=False),
        Binding("n,q", "no", "No", show=False),
        Binding("escape", "no", "Cancel", show=False),
        Binding("enter", "yes", "Confirm", show=False),
    ]

    def __init__(self, prompt: str, *, dangerous: bool = False) -> None:
        super().__init__()
        self.prompt = prompt
        self.dangerous = dangerous

    def compose(self) -> ComposeResult:
        with Container(id="modal-box"):
            yield Label(self.prompt, id="modal-prompt")
            with Horizontal(id="modal-buttons"):
                # Keep ``-error`` (red) for dangerous confirms — that's a
                # safety affordance, not just a "primary action" hint. Plain
                # confirms use the default variant so the highlight follows
                # focus rather than locking onto Yes.
                if self.dangerous:
                    yield Button("Yes", variant="error", id="yes")
                else:
                    yield Button("Yes", id="yes")
                yield Button("No", id="no")

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


@dataclass(frozen=True)
class AllocParams:
    """Result of :class:`NewInstanceModal`. Same shape for CPU and GPU jobs."""

    partition: str
    cores: int
    gpus: int  # 0 → CPU job (dispatches to slurm.submit_cpu); else GPU job
    mem_gb: int
    walltime: str

    @property
    def kind(self) -> str:
        return "gpu" if self.gpus > 0 else "cpu"


# Allowed partition components on the RCI cluster. The full partition name is
# ``<type><class>`` (e.g. ``gpufast``, ``cpu``, ``h200extralong``). The
# ``(normal)`` class maps to no suffix.
PARTITION_TYPES: tuple[str, ...] = ("cpu", "gpu", "amdgpu", "h200")
PARTITION_CLASSES: tuple[tuple[str, str], ...] = (
    ("fast", "fast"),
    ("(normal)", ""),
    ("long", "long"),
    ("extralong", "extralong"),
)


def assemble_partition(ptype: str, pclass: str) -> str:
    """Compose the two dropdown values into a Slurm partition name."""
    return f"{ptype}{pclass}"


def validate_alloc(ptype: str, gpus: int) -> str | None:
    """Return an error message if the (type, gpus) combo is invalid, else ``None``."""
    if gpus > 0 and ptype == "cpu":
        return "CPU partition doesn't accept GPUs — pick gpu / amdgpu / h200."
    return None


_JOBID_PATTERNS = (
    re.compile(r"job allocation (\d+)"),         # "Granted/Pending job allocation 1234567"
    re.compile(r"job (\d+) has been allocated"), # "salloc: job 1234567 has been allocated resources"
    re.compile(r"Submitted batch job (\d+)"),    # belt and suspenders — sbatch-style
)


def parse_jobid_from_salloc(output: str) -> str | None:
    """Best-effort jobid extraction from ``salloc --no-shell`` stdout/stderr.

    Slurm formats vary by site config; try a few known shapes before giving up.
    """
    for pat in _JOBID_PATTERNS:
        m = pat.search(output)
        if m:
            return m.group(1)
    return None


def last_meaningful_line(output: str) -> str:
    """Last non-empty line — usually the most informative on salloc errors."""
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return "(no output)"


class NewInstanceModal(ModalScreen["AllocParams | str | None"]):
    """Unified configure-and-submit dialog for both CPU and GPU allocations.

    No CPU/GPU toggle — the *kind* is derived from the ``GPUs`` field
    (``> 0`` ⇒ GPU job, dispatched to ``slurm.submit_gpu``; ``0`` ⇒ CPU
    job, dispatched to ``slurm.submit_cpu``).

    Dismiss values:
      - :class:`AllocParams` → user clicked Submit
      - ``None`` → user clicked the explicit Cancel button (abort)
      - the string ``"back"`` → user pressed ``q``/``escape`` while the caller
        had opted into step-back navigation via ``allow_back=True``. Caller
        re-opens the previous modal (typically :class:`FolderModal`).
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        # ``q`` escapes this window the same way ``escape`` does — closes the
        # modal and returns to the jobs panel. App-level ``q`` (quit) only
        # fires on the main screen because modal-screen bindings shadow it.
        Binding("escape,q", "cancel", "Cancel", show=False),
        # No screen-level Enter binding: Enter while editing an Input would
        # otherwise submit the whole form mid-type. Submitting now requires
        # focus on the Submit button (which consumes Enter natively via
        # Button.Pressed). Enter on a Select still opens its dropdown.
    ]

    def __init__(self, cfg: Config, *, allow_back: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        # When True, ``q``/``escape`` dismisses with the ``"back"`` sentinel
        # so the caller can re-open the previous step (e.g. FolderModal).
        # False for the standalone ``n`` flow where there's nothing to step
        # back to.
        self.allow_back = allow_back
        # Pre-fill from the last submitted params (persisted across sessions),
        # falling back to ``Config`` defaults if nothing's saved or a field
        # got corrupted. Validate enum-like fields so a stale state.json
        # doesn't crash the modal — fall back silently instead.
        last = state.get_last_instance_params() or {}
        cpu_cores, cpu_mem, cpu_time = cfg.cpu_defaults
        valid_classes = {value for _label, value in PARTITION_CLASSES}

        ptype = last.get("partition_type")
        pclass = last.get("partition_class")
        cores = last.get("cores")
        gpus = last.get("gpus")
        mem_gb = last.get("mem_gb")
        walltime = last.get("walltime")

        self._init_ptype = ptype if isinstance(ptype, str) and ptype in PARTITION_TYPES else "cpu"
        self._init_pclass = pclass if isinstance(pclass, str) and pclass in valid_classes else "fast"
        self._init_cores = str(cores) if isinstance(cores, int) and cores > 0 else str(cpu_cores)
        self._init_gpus = str(gpus) if isinstance(gpus, int) and gpus >= 0 else "0"
        self._init_mem = str(mem_gb) if isinstance(mem_gb, int) and mem_gb > 0 else str(cpu_mem)
        self._init_time = walltime if isinstance(walltime, str) and walltime else cpu_time

    def compose(self) -> ComposeResult:
        # VerticalScroll so the modal stays usable on shorter terminals — the
        # form scrolls within the box once it grows past the viewport's height.
        with VerticalScroll(id="modal-box"):
            yield Label("[b]New instance[/b]", id="modal-title")
            yield Label("Partition")
            with Horizontal(id="partition-row"):
                yield Select(
                    [(t, t) for t in PARTITION_TYPES],
                    value=self._init_ptype,
                    id="partition-type",
                    allow_blank=False,
                )
                yield Select(
                    list(PARTITION_CLASSES),
                    value=self._init_pclass,
                    id="partition-class",
                    allow_blank=False,
                )
            yield Label("Cores")
            yield Input(value=self._init_cores, id="cores", type="integer")
            yield Label("GPUs  [dim](0 = CPU job)[/dim]", id="gpus-label")
            yield Input(value=self._init_gpus, id="gpus", type="integer")
            yield Label("Memory (GB)")
            yield Input(value=self._init_mem, id="mem", type="integer")
            yield Label("Walltime (HH:MM:SS)")
            yield Input(value=self._init_time, id="time")
            with Horizontal(id="modal-buttons"):
                # No ``variant="primary"`` — both buttons start neutral and
                # the focused one is highlighted via Button:focus styling, so
                # ←/→ or Tab can shift the visible default.
                yield Button("Submit", id="ok")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        # Sync the GPUs row visibility to the restored partition type — if the
        # user's last submission was a GPU job, the field should already be visible.
        self._apply_gpu_visibility(self._init_ptype)
        # Land focus on Submit so the common case — accept the prefilled
        # defaults — is just one Enter press. Tab/Shift-Tab still walks back
        # to the form fields when the user wants to change something.
        self.query_one("#ok", Button).focus()

    def _apply_gpu_visibility(self, ptype: str) -> None:
        is_gpu_type = ptype != "cpu"
        self.query_one("#gpus-label", Label).display = is_gpu_type
        self.query_one("#gpus", Input).display = is_gpu_type

    @on(Select.Changed, "#partition-type")
    def _type_changed(self, event: Select.Changed) -> None:
        ptype = str(event.value)
        self._apply_gpu_visibility(ptype)
        # Quality-of-life: switching to a GPU type prefills GPUs=1 (if still 0);
        # switching back to CPU resets it to 0 so the validation always passes.
        gpus_input = self.query_one("#gpus", Input)
        if ptype == "cpu":
            gpus_input.value = "0"
        elif gpus_input.value in ("", "0"):
            gpus_input.value = "1"

    @on(Input.Submitted)
    def _advance_focus(self) -> None:
        """Enter in an Input moves focus to the next form field (standard form
        UX). After the last Input it lands on the Submit button, where Enter
        then submits via :meth:`_ok`."""
        self.focus_next()

    @on(Button.Pressed, "#ok")
    def _ok(self) -> None:
        self.action_submit()

    def action_submit(self) -> None:
        try:
            ptype = str(self.query_one("#partition-type", Select).value)
            pclass = str(self.query_one("#partition-class", Select).value)
            cores = int(self.query_one("#cores", Input).value or "0")
            gpus = int(self.query_one("#gpus", Input).value or "0")
            mem_gb = int(self.query_one("#mem", Input).value or "0")
            walltime = self.query_one("#time", Input).value.strip()
        except ValueError:
            self.app.notify("Invalid number", severity="error")
            return
        err = validate_alloc(ptype, gpus)
        if err is not None:
            self.app.notify(err, severity="error", timeout=6)
            return
        # Persist so the next session opens the modal prefilled with these values.
        state.set_last_instance_params(
            partition_type=ptype,
            partition_class=pclass,
            cores=cores,
            gpus=gpus,
            mem_gb=mem_gb,
            walltime=walltime,
        )
        self.dismiss(
            AllocParams(
                partition=assemble_partition(ptype, pclass),
                cores=cores,
                gpus=gpus,
                mem_gb=mem_gb,
                walltime=walltime,
            )
        )

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        # Explicit Cancel button always means "abort this flow", never
        # "step back to the folder picker" — q/escape handle the step-back.
        self.dismiss(None)

    def action_cancel(self) -> None:
        # ``q``/``escape``: step back to the prior modal if the caller is
        # willing to handle it; otherwise behave like the Cancel button.
        self.dismiss("back" if self.allow_back else None)

    def on_key(self, event: events.Key) -> None:
        """Let ←/→ swap focus between the Submit and Cancel buttons.

        Only fires when focus is already on one of the two buttons — Inputs
        and Selects consume their own arrow keys (cursor / dropdown nav)
        before the event bubbles up to the screen, so this stays out of
        their way.
        """
        if event.key not in ("left", "right"):
            return
        focused = self.focused
        if focused is None or focused.id not in ("ok", "cancel"):
            return
        other = "cancel" if focused.id == "ok" else "ok"
        self.query_one(f"#{other}", Button).focus()
        event.stop()


class FolderModal(ModalScreen[str | None]):
    """Quick prompt: which folder on the compute node? Empty string = home."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape,q", "cancel", "Cancel", show=False),
    ]

    def __init__(self, prompt: str = "Folder on compute node") -> None:
        super().__init__()
        self.prompt = prompt
        # Pre-fill with whatever the user typed last time (persisted across
        # sessions via state.json) so the common case is just Enter.
        self.default_folder = state.get_last_folder()

    def compose(self) -> ComposeResult:
        with Container(id="modal-box"):
            yield Label(f"[b]{self.prompt}[/b]  [dim](empty = home)[/dim]", id="modal-title")
            yield Input(
                value=self.default_folder,
                id="folder",
                placeholder="e.g. sam2rl  or  /scratch/exp42",
            )
            with Horizontal(id="modal-buttons"):
                yield Button("Open", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted)
    def _ok(self) -> None:
        folder = self.query_one("#folder", Input).value.strip()
        state.set_last_folder(folder)
        self.dismiss(folder)

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ──────────────────────────── jobs screen ───────────────────────────────────


@dataclass(frozen=True)
class JobRow:
    jobid: str
    partition: str
    name: str
    state: str
    time: str
    limit: str
    cpus: str
    mem: str
    gres: str  # raw squeue %b string — ``gpu:1``, ``N/A``, ``(null)``
    node: str

    @classmethod
    def from_squeue_line(cls, line: str) -> JobRow | None:
        parts = line.split(None, 9)
        if len(parts) < 4:
            return None
        while len(parts) < 10:
            parts.append("")
        return cls(
            jobid=parts[0],
            partition=parts[1],
            name=parts[2],
            state=parts[3],
            time=parts[4],
            limit=parts[5],
            cpus=parts[6],
            mem=parts[7],
            gres=parts[8],
            node=parts[9],
        )

    @property
    def node_display(self) -> str:
        """Node name for running jobs; ``—`` when squeue gave us a ``(Reason)`` instead.

        ``%R`` doubles as node-list (running) and reason (pending) — we keep
        the raw text in ``self.node`` for the action-guard check
        (``startswith('(')``) but show a dash in the table.
        """
        n = (self.node or "").strip()
        if not n or n.startswith("("):
            return "—"
        return n

    @property
    def gpu_count(self) -> str:
        """Compact GPU column: ``1`` for ``gpu:1``, ``—`` for none."""
        g = (self.gres or "").strip()
        if not g or g in ("N/A", "(null)"):
            return "—"
        for part in g.split(","):
            part = part.strip()
            if part.startswith("gpu"):
                # ``gpu:1`` or ``gpu:tesla:1`` — last segment is the count.
                segs = part.split(":")
                if len(segs) >= 2 and segs[-1].isdigit():
                    return segs[-1]
                return "?"
        return "—"


class JobsPanel(Container):
    """Live jobs dashboard with action keys."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("r", "refresh", "Refresh"),
        Binding("c", "cancel_job", "Cancel"),
        Binding("s", "shell_into", "Shell"),
        Binding("e", "editor_into", "Editor"),
        Binding("n", "new_instance", "New instance"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rows: list[JobRow] = []
        self._last_action: str = ""
        self._action_clear_timer = None  # type: ignore[var-annotated]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Loading…", id="alloc-status")
            yield DataTable(id="jobs-table", zebra_stripes=True, cursor_type="row")
            yield Static("", id="job-detail")
            yield Static("", id="last-action")

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.add_columns(
            "JobID",
            "Partition",
            "Name",
            "State",
            "Time",
            "Limit",
            "CPU",
            "Mem",
            "GPU",
            "Node",
        )
        table.focus()
        self.refresh_jobs()
        self.set_interval(REFRESH_INTERVAL, self.refresh_jobs)

    # ----- data refresh (threaded) -----

    @work(thread=True, exclusive=True, group="refresh")
    def refresh_jobs(self) -> None:
        cfg = load()
        try:
            raw = slurm.list_jobs(cfg)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._notify_error, f"Refresh failed: {e}")
            return
        rows: list[JobRow] = []
        for line in raw.splitlines()[1:]:  # squeue's own header
            row = JobRow.from_squeue_line(line)
            if row is not None:
                rows.append(row)
        self.app.call_from_thread(self._apply_rows, rows)

    def _apply_rows(self, rows: list[JobRow]) -> None:
        prior = self._selected_jobid()
        table = self.query_one("#jobs-table", DataTable)
        table.clear()
        for r in rows:
            table.add_row(
                r.jobid,
                r.partition,
                r.name,
                r.state,
                r.time,
                r.limit,
                r.cpus or "—",
                r.mem or "—",
                r.gpu_count,
                r.node_display,
                key=r.jobid,
            )
        # Update the detail line for the (re-)selected row, if any.
        self._refresh_detail()
        self._rows = rows
        if rows:
            target = next((i for i, r in enumerate(rows) if r.jobid == prior), 0)
            try:
                table.move_cursor(row=target)
            except Exception:  # noqa: BLE001
                pass
        # Summarize the table at a glance — there's no single "current"
        # allocation now that names are per-job; we just count states.
        running = sum(1 for r in rows if r.state in RUNNING_STATES)
        pending = len(rows) - running
        status = self.query_one("#alloc-status", Static)
        # Uniform format: always show ``N running``; append ``M pending`` only
        # when non-zero. The footer already advertises ``n New instance`` so a
        # call-to-action here would be redundant.
        parts = [f"[green b]{running}[/] running"]
        if pending:
            parts.append(f"[yellow]{pending}[/] pending")
        status.update("  [dim]·[/dim]  ".join(parts))
        # Clear the "refreshing…" indicator if a manual refresh just completed.
        # Other action messages stay until their own fade timer fires.
        if self._last_action == "refreshing…":
            self._clear_action_log()

    # ----- selection helpers -----

    def _selected_row(self) -> JobRow | None:
        table = self.query_one("#jobs-table", DataTable)
        if not self._rows:
            return None
        try:
            return self._rows[table.cursor_row]
        except IndexError:
            return None

    def _selected_jobid(self) -> str | None:
        r = self._selected_row()
        return r.jobid if r else None

    def _refresh_detail(self) -> None:
        row = self._selected_row()
        widget = self.query_one("#job-detail", Static)
        if row is None:
            widget.update("")
            return
        gpu_seg = f" · [b]{row.gpu_count}[/] GPU" if row.gpu_count not in ("—", "0") else ""
        widget.update(
            f"[b]{row.jobid}[/]  {row.name}  [dim]·[/dim]  "
            f"[b]{row.state}[/]  [dim]on[/]  {row.partition}  [dim]·[/dim]  "
            f"[b]{row.cpus}[/] CPU · [b]{row.mem}[/]{gpu_seg}  [dim]·[/dim]  "
            f"used [b]{row.time}[/] / limit [b]{row.limit}[/]  [dim]·[/dim]  "
            f"node [b]{row.node_display}[/]"
        )

    @on(DataTable.RowHighlighted, "#jobs-table")
    def _row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._refresh_detail()

    def _notify_action(self, msg: str) -> None:
        self._last_action = msg
        self.query_one("#last-action", Static).update(msg)
        # Schedule auto-clear; cancel any prior pending clear so a new message
        # gets the full fade window rather than inheriting the old one's clock.
        if self._action_clear_timer is not None:
            self._action_clear_timer.stop()
            self._action_clear_timer = None
        if msg:
            self._action_clear_timer = self.set_timer(ACTION_FADE_SECONDS, self._clear_action_log)

    def _clear_action_log(self) -> None:
        self._last_action = ""
        try:
            self.query_one("#last-action", Static).update("")
        except Exception:  # noqa: BLE001 — widget may be unmounted on exit
            pass
        if self._action_clear_timer is not None:
            self._action_clear_timer.stop()
            self._action_clear_timer = None

    def _notify_error(self, msg: str) -> None:
        self.app.notify(msg, severity="error", timeout=6)
        self._notify_action(f"[red]{msg}[/]")

    # ----- actions -----

    def action_refresh(self) -> None:
        self._notify_action("refreshing…")
        self.refresh_jobs()

    def action_cancel_job(self) -> None:
        row = self._selected_row()
        if row is None:
            self.app.notify("Nothing selected.", severity="warning")
            return
        prompt = f"Cancel job [b]{row.jobid}[/] ([i]{row.name}[/], {row.state}) on {row.partition}?"
        self.app.push_screen(
            ConfirmModal(prompt, dangerous=True),
            lambda ok: self._do_cancel(row.jobid) if ok else None,
        )

    @work(thread=True, group="action")
    def _do_cancel(self, jobid: str) -> None:
        cfg = load()
        rc = slurm.cancel(cfg, jobid)
        if rc == 0:
            self.app.call_from_thread(self._notify_action, f"cancelled job {jobid}")
            self.app.call_from_thread(lambda: self.app.notify(f"Cancelled {jobid}"))
        else:
            self.app.call_from_thread(self._notify_error, f"scancel {jobid} exited {rc}")
        self.refresh_jobs()

    # ----- shell / editor: pick folder → attach existing alloc OR spawn one -----

    def action_shell_into(self) -> None:
        self._pick_folder_then("shell")

    def action_editor_into(self) -> None:
        self._pick_folder_then("editor")

    def _pick_folder_then(self, kind: str) -> None:
        prompt = f"{kind.capitalize()} into folder"
        self.app.push_screen(
            FolderModal(prompt),
            lambda folder: self._after_folder(kind, folder),
        )

    def _after_folder(self, kind: str, folder: str | None) -> None:
        if folder is None:
            return  # user cancelled the folder prompt
        cfg = load()
        # Use whichever job the cursor's on, if it's a running node. With
        # per-allocation names there's no canonical "the" allocation to
        # auto-pick — the user is expected to highlight the one they want.
        alloc = self._alloc_from_selected_row()
        if alloc is not None:
            self._attach_to(kind, alloc, folder)
            return
        # No usable selection — collect alloc params, submit, attach. Opt
        # into ``allow_back`` so q/escape on the resources modal returns
        # to the folder picker instead of dropping the whole flow.
        self.app.push_screen(
            NewInstanceModal(cfg, allow_back=True),
            lambda result: self._after_new_instance(kind, folder, result),
        )

    def _after_new_instance(
        self, kind: str, folder: str, result: "AllocParams | str | None"
    ) -> None:
        """Dispatch on the resources modal's return: submit / step back / abort."""
        if result is None:
            return  # Cancel button → drop the whole flow
        if result == "back":
            # Step back to the folder picker. FolderModal pre-fills with the
            # persisted last_folder, which is exactly the one the user just
            # submitted — they land where they were and can edit.
            self._pick_folder_then(kind)
            return
        assert isinstance(result, AllocParams)
        self._submit_then_attach(kind, result, folder)

    def _alloc_from_selected_row(self) -> alloc_mod.Allocation | None:
        """Promote the highlighted row to an Allocation if it's running on a node."""
        row = self._selected_row()
        if row is None:
            return None
        if row.state not in RUNNING_STATES:
            return None
        # PENDING rows put the reason (``(Resources)``) into the NODE column;
        # only real hostnames make for a working ssh target.
        if not row.node or row.node.startswith("("):
            return None
        return alloc_mod.Allocation(node=row.node, jobid=row.jobid)

    def _attach_to(
        self, kind: str, alloc: alloc_mod.Allocation, folder_arg: str = ""
    ) -> None:
        cfg = load()
        folder_abs = launch.resolve_folder(folder_arg, cfg)
        if kind == "editor":
            self._notify_action(f"opening editor on {alloc.node} ({folder_abs})")
            launch.launch_editor(alloc, folder_abs, cfg)
            return
        # shell: suspend the TUI so ssh has the terminal.
        self._notify_action(f"attaching to {alloc.node} ({folder_abs})… exit to return")
        with self.app.suspend():
            launch.launch_shell(alloc, folder_abs, cfg)
        self._notify_action(f"returned from {alloc.node}")
        self.refresh_jobs()

    def _submit_then_attach(self, kind: str, params: AllocParams, folder: str) -> None:
        self._notify_action(
            f"submitting {params.kind} ({params.partition}; {params.cores}c/{params.mem_gb}G/{params.walltime})…"
        )
        self._do_submit_and_attach(kind, params, folder)

    @work(thread=True, exclusive=True, group="action")
    def _do_submit_and_attach(self, kind: str, params: AllocParams, folder: str) -> None:
        import time
        cfg = load()
        try:
            if params.gpus > 0:
                out = slurm.submit_gpu(
                    cfg, params.gpus, params.cores, params.mem_gb, params.walltime,
                    partition=params.partition or None,
                )
            else:
                out = slurm.submit_cpu(
                    cfg, params.cores, params.mem_gb, params.walltime,
                    partition=params.partition or None,
                )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._notify_error, f"submit failed: {e}")
            return
        # salloc typically prints ``Granted/Pending job allocation NNN`` but
        # variants exist — :func:`parse_jobid_from_salloc` tries multiple shapes.
        jobid = parse_jobid_from_salloc(out)
        if jobid is None:
            # Surface the actual salloc complaint (last non-empty line is usually
            # the error message) so the user knows whether it was a partition
            # mismatch, quota issue, or invalid time/mem spec.
            self.app.call_from_thread(
                self._notify_error,
                f"submit failed: {last_meaningful_line(out)}",
            )
            return
        first_line = (out.strip().splitlines() or [""])[0]
        self.app.call_from_thread(
            self._notify_action, f"submitted ({first_line}); waiting for node…"
        )
        # Poll for the alloc to land. cpufast/gpufast schedule in seconds.
        deadline = time.time() + 30.0
        node = ""
        while time.time() < deadline:
            node = slurm.node_for(cfg, jobid)
            if node:
                break
            time.sleep(1.0)
        if not node:
            self.app.call_from_thread(
                self._notify_error,
                f"job {jobid} didn't get a node assigned within 30s",
            )
            return
        self.app.call_from_thread(
            self._attach_to, kind, alloc_mod.Allocation(node=node, jobid=jobid), folder
        )

    # ----- new instance only (no auto-attach) -----

    def action_new_instance(self) -> None:
        self.app.push_screen(NewInstanceModal(load()), self._on_new_instance)

    def _on_new_instance(self, params: AllocParams | None) -> None:
        if params is None:
            return
        self._notify_action(
            f"submitting {params.kind} ({params.partition}; {params.cores}c/{params.mem_gb}G/{params.walltime})…"
        )
        self._do_submit(params)

    @work(thread=True, group="action")
    def _do_submit(self, params: AllocParams) -> None:
        cfg = load()
        try:
            if params.kind == "cpu":
                out = slurm.submit_cpu(
                    cfg, params.cores, params.mem_gb, params.walltime,
                    partition=params.partition or None,
                )
            else:
                out = slurm.submit_gpu(
                    cfg, params.gpus, params.cores, params.mem_gb, params.walltime,
                    partition=params.partition or None,
                )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._notify_error, f"submit failed: {e}")
            return
        # salloc runs with check=False, so non-zero exits land here as plain
        # output (e.g. ``salloc: error: Invalid partition: foo``). No jobid in
        # the output ⇒ surface it as an error, not a fake-positive "submitted".
        jobid = parse_jobid_from_salloc(out)
        if jobid is None:
            self.app.call_from_thread(
                self._notify_error,
                f"submit failed: {last_meaningful_line(out)}",
            )
            return
        first_line = (out.strip().splitlines() or [""])[0]
        self.app.call_from_thread(self._notify_action, f"submitted: {first_line}")
        self.refresh_jobs()


# ──────────────────────────── app ───────────────────────────────────────────


CSS = """
/* Inherit the terminal's own palette so the TUI lives in the same
   color world as the user's P10k prompt and lazygit. Theme is set to
   ``ansi-dark`` on the app; widget styling here uses ANSI color names
   (or theme-aware tokens) so it adapts to whatever terminal scheme. */

Screen { background: ansi_default; color: ansi_default; }

Header {
    background: ansi_default;
    color: ansi_bright_white;
}
Header > HeaderTitle { text-style: bold; }

Footer {
    background: ansi_default;
    color: ansi_default;
}
FooterKey > .footer-key--key { color: ansi_cyan; text-style: bold; }

#alloc-status {
    padding: 0 1;
    height: 1;
    color: ansi_default;
    background: ansi_default;
}

#job-detail {
    padding: 0 1;
    height: 1;
    color: ansi_default;
    background: ansi_default;
    border-top: dashed ansi_bright_black;
}

#last-action {
    padding: 0 1;
    height: 1;
    color: ansi_bright_black;
    background: ansi_default;
}

#jobs-table {
    height: 1fr;
    background: ansi_default;
    color: ansi_default;
    border: round ansi_cyan;
    border-title-color: ansi_cyan;
    border-title-style: bold;
    padding: 0;
    scrollbar-color: ansi_cyan;
    scrollbar-color-hover: ansi_bright_cyan;
    scrollbar-color-active: ansi_bright_cyan;
}
#jobs-table > .datatable--header {
    color: ansi_yellow;
    text-style: bold;
    background: ansi_default;
}
#jobs-table > .datatable--cursor {
    background: ansi_blue;
    color: ansi_bright_white;
    text-style: bold;
}
#jobs-table > .datatable--hover { background: ansi_default; }

/* Modals: bordered dialog, centered. Backdrop is a light translucent tint —
   the dashboard stays clearly visible (just slightly dimmed) so you don't
   lose the context of which job is highlighted while the modal is open. */
ConfirmModal, NewInstanceModal, FolderModal {
    align: center middle;
    background: black 25%;
}

#partition-row { height: auto; padding-bottom: 1; }
#partition-row Select { width: 1fr; margin-right: 1; }
#partition-row Select:last-child { margin-right: 0; }

#modal-box {
    background: ansi_default;
    color: ansi_default;
    border: round ansi_cyan;
    padding: 1 2;
    width: 60;
    height: auto;
    /* Cap at 90% of the viewport so the box stays on screen on short
       terminals. VerticalScroll lets the form scroll within. */
    max-height: 90%;
    scrollbar-color: ansi_cyan;
    scrollbar-color-hover: ansi_bright_cyan;
    scrollbar-color-active: ansi_bright_cyan;
}


#modal-title, #modal-prompt {
    padding-bottom: 1;
    color: ansi_bright_white;
    text-style: bold;
}

Input {
    background: ansi_default;
    color: ansi_default;
    border: round ansi_bright_black;
    margin-bottom: 1;
}
Input:focus { border: round ansi_cyan; }

#modal-buttons {
    height: auto;
    padding-top: 1;
    align-horizontal: right;
}
#modal-buttons Button { margin-left: 1; min-width: 12; }

Button {
    background: ansi_default;
    color: ansi_default;
    border: round ansi_bright_black;
}
Button:hover, Button:focus { border: round ansi_cyan; color: ansi_cyan; }
Button.-primary {
    background: ansi_blue;
    color: ansi_bright_white;
    border: round ansi_blue;
}
Button.-primary:hover, Button.-primary:focus { border: round ansi_bright_blue; }
Button.-error {
    background: ansi_red;
    color: ansi_bright_white;
    border: round ansi_red;
}
Button.-error:hover, Button.-error:focus { border: round ansi_bright_red; }

Label { color: ansi_default; }

Toast {
    background: ansi_default;
    color: ansi_default;
    border: round ansi_cyan;
}
"""


class RciApp(App):
    """Top-level Textual app — live Slurm jobs dashboard."""

    CSS = CSS
    TITLE = "RCI Cluster Manager"
    # Drop the default ``^p palette`` footer entry — we don't expose any
    # actions through it, so the Ctrl-prefixed key only adds clutter next to
    # the single-letter bindings.
    ENABLE_COMMAND_PALETTE = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        # Hide Textual's default ``^q`` / ``^c`` footer entries — ``q`` alone
        # is the documented exit key, and the Ctrl-prefixed duplicates just
        # add noise next to the single-letter bindings.
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("t", "toggle_theme", "Theme", show=False),
    ]

    def on_mount(self) -> None:
        # Use the terminal's own palette so the TUI sits in the same color
        # world as the user's shell prompt and lazygit. ``t`` cycles to dark.
        self.theme = "ansi-dark"
        # Textual's Header ships a ``⭘`` icon in the top-left — clickable toggle
        # for the clock that just adds visual noise here. Blank it out.
        self.query_one(Header).icon = ""

    def action_toggle_theme(self) -> None:
        # Minimal cycle: stick with the terminal palette by default, fall back
        # to Textual's own dark theme if ansi colors look bad on this terminal.
        order = ["ansi-dark", "textual-dark"]
        try:
            idx = order.index(self.theme)
        except ValueError:
            idx = -1
        self.theme = order[(idx + 1) % len(order)]
        self.notify(f"theme: {self.theme}", timeout=2)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield JobsPanel(id="jobs-panel")
        yield Footer()
