"""Interactive Textual TUI for rci-cli.

Live jobs dashboard with one-key actions: cancel selected, shell into the
compute node, launch claude or VS Code on the allocation, submit fresh CPU /
GPU allocations from a modal. Refresh is threaded so the UI never freezes
while ``squeue`` is in flight; ``App.suspend()`` is used to hand the terminal
over to ssh for shell-in / claude attach, then re-render on return.

The screen is wrapped in ``TabbedContent`` so additional tabs (an "Agents"
panel for managing claude agents on the cluster — coming later) can plug in
without restructuring the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
    TabbedContent,
    TabPane,
)

from . import alloc as alloc_mod
from . import launch, slurm
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
        Binding("n", "no", "No", show=False),
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
                yield Button(
                    "Yes",
                    variant="error" if self.dangerous else "primary",
                    id="yes",
                )
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
    """Result of :class:`NewInstanceModal`. Carries everything ``slurm.submit_*`` needs."""

    kind: str  # "cpu" or "gpu"
    partition: str
    cores: int
    mem_gb: int
    walltime: str
    gpus: int = 0  # only meaningful when kind == "gpu"


class NewInstanceModal(ModalScreen[AllocParams | None]):
    """Single configure-and-submit modal for both CPU and GPU allocations.

    Top radio toggles kind; switching kind swaps the partition default and
    shows/hides the GPUs field. All numeric values prefilled from the relevant
    ``cfg.cpu_defaults`` / ``cfg.gpu_defaults`` tuple.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, cfg: Config, *, initial_kind: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        self._kind = initial_kind

    def compose(self) -> ComposeResult:
        cpu_cores, cpu_mem, cpu_time = self.cfg.cpu_defaults
        gpu_count, gpu_cores, gpu_mem, gpu_time = self.cfg.gpu_defaults
        with Container(id="modal-box"):
            yield Label("[b]New instance[/b]", id="modal-title")
            with RadioSet(id="kind"):
                yield RadioButton("CPU", value=(self._kind == "cpu"), id="kind-cpu")
                yield RadioButton("GPU", value=(self._kind == "gpu"), id="kind-gpu")
            yield Label("Partition")
            yield Input(
                value=self.cfg.cpu_partition if self._kind == "cpu" else self.cfg.gpu_partition,
                id="partition",
            )
            # GPU count — only relevant when kind == "gpu"; we toggle the row's
            # display on radio change rather than re-composing.
            yield Label("GPUs", id="gpus-label")
            yield Input(value=str(gpu_count), id="gpus", type="integer")
            yield Label("Cores")
            yield Input(
                value=str(cpu_cores if self._kind == "cpu" else gpu_cores),
                id="cores",
                type="integer",
            )
            yield Label("Memory (GB)")
            yield Input(
                value=str(cpu_mem if self._kind == "cpu" else gpu_mem),
                id="mem",
                type="integer",
            )
            yield Label("Walltime (HH:MM:SS)")
            yield Input(value=cpu_time if self._kind == "cpu" else gpu_time, id="time")
            with Horizontal(id="modal-buttons"):
                yield Button("Submit", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self._apply_kind_visibility()

    def _apply_kind_visibility(self) -> None:
        is_gpu = self._kind == "gpu"
        self.query_one("#gpus-label", Label).display = is_gpu
        self.query_one("#gpus", Input).display = is_gpu

    @on(RadioSet.Changed, "#kind")
    def _kind_changed(self, event: RadioSet.Changed) -> None:
        new_kind = "gpu" if event.pressed.id == "kind-gpu" else "cpu"
        if new_kind == self._kind:
            return
        self._kind = new_kind
        # Swap partition + numeric defaults to whichever side the user just picked.
        cpu_cores, cpu_mem, cpu_time = self.cfg.cpu_defaults
        gpu_count, gpu_cores, gpu_mem, gpu_time = self.cfg.gpu_defaults
        self.query_one("#partition", Input).value = (
            self.cfg.gpu_partition if new_kind == "gpu" else self.cfg.cpu_partition
        )
        self.query_one("#cores", Input).value = str(gpu_cores if new_kind == "gpu" else cpu_cores)
        self.query_one("#mem", Input).value = str(gpu_mem if new_kind == "gpu" else cpu_mem)
        self.query_one("#time", Input).value = gpu_time if new_kind == "gpu" else cpu_time
        self.query_one("#gpus", Input).value = str(gpu_count)
        self._apply_kind_visibility()

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted)
    def _ok(self) -> None:
        try:
            params = AllocParams(
                kind=self._kind,
                partition=self.query_one("#partition", Input).value.strip(),
                cores=int(self.query_one("#cores", Input).value or "0"),
                mem_gb=int(self.query_one("#mem", Input).value or "0"),
                walltime=self.query_one("#time", Input).value.strip(),
                gpus=int(self.query_one("#gpus", Input).value or "0") if self._kind == "gpu" else 0,
            )
        except ValueError:
            self.app.notify("Invalid number", severity="error")
            return
        self.dismiss(params)

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
    node: str

    @classmethod
    def from_squeue_line(cls, line: str) -> JobRow | None:
        parts = line.split(None, 7)
        if len(parts) < 4:
            return None
        while len(parts) < 8:
            parts.append("")
        return cls(
            jobid=parts[0],
            partition=parts[1],
            name=parts[2],
            state=parts[3],
            time=parts[4],
            limit=parts[5],
            node=parts[7] or parts[6],  # %N is last in our format string
        )


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
            yield Static("", id="last-action")

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.add_columns("JobID", "Partition", "Name", "State", "Time", "Limit", "Node / Reason")
        table.focus()
        self.refresh_jobs()
        self.set_interval(REFRESH_INTERVAL, self.refresh_jobs)

    # ----- data refresh (threaded) -----

    @work(thread=True, exclusive=True, group="refresh")
    def refresh_jobs(self) -> None:
        cfg = load()
        try:
            raw = slurm.list_jobs(cfg)
            alloc = alloc_mod.find_strongest(cfg)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._notify_error, f"Refresh failed: {e}")
            return
        rows: list[JobRow] = []
        for line in raw.splitlines()[1:]:  # squeue's own header
            row = JobRow.from_squeue_line(line)
            if row is not None:
                rows.append(row)
        self.app.call_from_thread(self._apply_rows, rows, alloc)

    def _apply_rows(self, rows: list[JobRow], alloc: alloc_mod.Allocation | None) -> None:
        prior = self._selected_jobid()
        table = self.query_one("#jobs-table", DataTable)
        table.clear()
        for r in rows:
            table.add_row(r.jobid, r.partition, r.name, r.state, r.time, r.limit, r.node, key=r.jobid)
        self._rows = rows
        if rows:
            target = next((i for i, r in enumerate(rows) if r.jobid == prior), 0)
            try:
                table.move_cursor(row=target)
            except Exception:  # noqa: BLE001
                pass
        status = self.query_one("#alloc-status", Static)
        if alloc:
            status.update(
                f"[green b]●[/] allocation: [b]{alloc.node}[/]  job [b]{alloc.jobid}[/]"
            )
        else:
            status.update("[yellow]○[/yellow] no running vscode allocation — press [b]n[/] or [b]g[/] to submit")
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

    # ----- shell / editor: auto-attach, or pop the New Instance modal -----

    def action_shell_into(self) -> None:
        self._attach_or_spawn("shell")

    def action_editor_into(self) -> None:
        self._attach_or_spawn("editor")

    def _attach_or_spawn(self, kind: str) -> None:
        """If a vscode allocation is already running, attach to it. Else pop the modal."""
        cfg = load()
        alloc = alloc_mod.find_strongest(cfg)
        if alloc is not None:
            self._attach_to(kind, alloc)
            return
        # No allocation — ask the user to configure one, then attach.
        self.app.push_screen(
            NewInstanceModal(cfg),
            lambda params: self._submit_then_attach(kind, params) if params else None,
        )

    def _attach_to(self, kind: str, alloc: alloc_mod.Allocation) -> None:
        cfg = load()
        folder = launch.resolve_folder("", cfg)
        if kind == "editor":
            self._notify_action(f"opening editor on {alloc.node}")
            launch.launch_editor(alloc, folder, cfg)
            return
        # shell: suspend the TUI so ssh has the terminal.
        self._notify_action(f"attaching to {alloc.node} (shell)… exit to return")
        with self.app.suspend():
            launch.launch_shell(alloc, folder, cfg)
        self._notify_action(f"returned from {alloc.node}")
        self.refresh_jobs()

    def _submit_then_attach(self, kind: str, params: AllocParams) -> None:
        self._notify_action(
            f"submitting {params.kind} ({params.partition}; {params.cores}c/{params.mem_gb}G/{params.walltime})…"
        )
        self._do_submit_and_attach(kind, params)

    @work(thread=True, exclusive=True, group="action")
    def _do_submit_and_attach(self, kind: str, params: AllocParams) -> None:
        import time
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
        first_line = (out.strip().splitlines() or [""])[0]
        self.app.call_from_thread(self._notify_action, f"submitted ({first_line}); waiting for node…")
        # Poll for the alloc to land. cpufast/gpufast schedule in seconds.
        deadline = time.time() + 30.0
        alloc: alloc_mod.Allocation | None = None
        while time.time() < deadline:
            alloc = alloc_mod.find_strongest(cfg)
            if alloc is not None:
                break
            time.sleep(1.0)
        if alloc is None:
            self.app.call_from_thread(
                self._notify_error, "submission didn't produce a running allocation within 30s"
            )
            return
        self.app.call_from_thread(self._attach_to, kind, alloc)

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

TabbedContent { height: 1fr; }
Tabs Underline { color: ansi_cyan; }
Tab { color: ansi_default; }
Tab.-active { color: ansi_cyan; text-style: bold; }

#alloc-status {
    padding: 0 1;
    height: 1;
    color: ansi_default;
    background: ansi_default;
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

/* Modals: bordered dialog, lazygit-style centered popup. */
ConfirmModal, SubmitCpuModal, SubmitGpuModal { align: center middle; }

#modal-box {
    background: ansi_default;
    color: ansi_default;
    border: round ansi_cyan;
    padding: 1 2;
    width: 60;
    height: auto;
    max-height: 24;
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
    """Top-level Textual app. Single 'Jobs' tab for now; extension point for future tabs."""

    CSS = CSS
    TITLE = "rci"
    SUB_TITLE = "RCI CVUT Slurm cluster"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("t", "toggle_theme", "Theme", show=False),
    ]

    def on_mount(self) -> None:
        # Use the terminal's own palette so the TUI sits in the same color
        # world as the user's shell prompt and lazygit. ``t`` cycles to dark.
        self.theme = "ansi-dark"

    def action_toggle_theme(self) -> None:
        # Quick escape hatch if the ansi palette looks bad on a given terminal.
        order = ["ansi-dark", "gruvbox", "nord", "monokai", "textual-dark"]
        try:
            idx = order.index(self.theme)
        except ValueError:
            idx = -1
        self.theme = order[(idx + 1) % len(order)]
        self.notify(f"theme: {self.theme}", timeout=2)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="tab-jobs"):
            with TabPane("Jobs", id="tab-jobs"):
                yield JobsPanel(id="jobs-panel")
            # Future: TabPane("Agents", id="tab-agents") for claude-agent management.
        yield Footer()
