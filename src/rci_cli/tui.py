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
class CpuParams:
    cores: int
    mem: int
    walltime: str


@dataclass(frozen=True)
class GpuParams:
    gpus: int
    cores: int
    mem: int
    walltime: str


class SubmitCpuModal(ModalScreen[CpuParams | None]):
    """Submit-CPU dialog. Inputs prefilled from ``cfg.cpu_defaults``."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        cores, mem, time = self.cfg.cpu_defaults
        with Container(id="modal-box"):
            yield Label("[b]Submit CPU allocation[/b] (partition: cpufast)", id="modal-title")
            yield Label("Cores")
            yield Input(value=str(cores), id="cores", type="integer")
            yield Label("Memory (GB)")
            yield Input(value=str(mem), id="mem", type="integer")
            yield Label("Walltime (HH:MM:SS)")
            yield Input(value=time, id="time")
            with Horizontal(id="modal-buttons"):
                yield Button("Submit", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted)
    def _ok(self) -> None:
        try:
            params = CpuParams(
                cores=int(self.query_one("#cores", Input).value or "0"),
                mem=int(self.query_one("#mem", Input).value or "0"),
                walltime=self.query_one("#time", Input).value.strip(),
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


class SubmitGpuModal(ModalScreen[GpuParams | None]):
    """Submit-GPU dialog. Inputs prefilled from ``cfg.gpu_defaults``."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        gpus, cores, mem, time = self.cfg.gpu_defaults
        with Container(id="modal-box"):
            yield Label("[b]Submit GPU allocation[/b] (partition: gpufast)", id="modal-title")
            yield Label("GPUs")
            yield Input(value=str(gpus), id="gpus", type="integer")
            yield Label("Cores")
            yield Input(value=str(cores), id="cores", type="integer")
            yield Label("Memory (GB)")
            yield Input(value=str(mem), id="mem", type="integer")
            yield Label("Walltime (HH:MM:SS)")
            yield Input(value=time, id="time")
            with Horizontal(id="modal-buttons"):
                yield Button("Submit", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted)
    def _ok(self) -> None:
        try:
            params = GpuParams(
                gpus=int(self.query_one("#gpus", Input).value or "0"),
                cores=int(self.query_one("#cores", Input).value or "0"),
                mem=int(self.query_one("#mem", Input).value or "0"),
                walltime=self.query_one("#time", Input).value.strip(),
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
        Binding("S", "shell_into_tab", "Shell tab"),
        Binding("l", "claude_into", "Claude"),
        Binding("L", "claude_into_tab", "Claude tab"),
        Binding("o", "code_into", "VS Code"),
        Binding("n", "submit_cpu", "+ CPU"),
        Binding("g", "submit_gpu", "+ GPU"),
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

    def action_shell_into(self) -> None:
        self._launch_on_selected("shell")

    def action_shell_into_tab(self) -> None:
        self._launch_on_selected("shell", in_tab=True)

    def action_claude_into(self) -> None:
        self._launch_on_selected("claude")

    def action_claude_into_tab(self) -> None:
        self._launch_on_selected("claude", in_tab=True)

    def action_code_into(self) -> None:
        self._launch_on_selected("code")

    def _launch_on_selected(self, kind: str, *, in_tab: bool = False) -> None:
        row = self._selected_row()
        if row is None:
            self.app.notify("Nothing selected.", severity="warning")
            return
        if row.state not in RUNNING_STATES:
            self.app.notify(f"Job {row.jobid} is not running ({row.state}).", severity="warning")
            return
        if not row.node or row.node.startswith("("):
            self.app.notify(
                f"Selected job has no assigned node yet ({row.node}).", severity="warning"
            )
            return
        cfg = load()
        a = alloc_mod.Allocation(node=row.node, jobid=row.jobid)
        folder = launch.resolve_folder("", cfg)
        if kind == "code":
            # No suspend needed — code launches a windowed app or returns immediately.
            self._notify_action(f"opening VS Code on {row.node}")
            launch.launch_code(a, folder, cfg)
            return
        if in_tab:
            # No suspend needed — spawner returns immediately and the TUI keeps running.
            rc = (
                launch.launch_shell_in_tab(a, "", cfg)
                if kind == "shell"
                else launch.launch_claude_in_tab(a, "", cfg)
            )
            if rc == 2:
                self.app.notify(
                    "No supported terminal for new-tab spawn — try inside tmux, "
                    "Windows Terminal, WezTerm, kitty, or iTerm2.",
                    severity="warning",
                    timeout=6,
                )
            else:
                self._notify_action(f"opened new tab → {row.node} ({kind})")
            return
        # shell / claude: suspend the TUI so ssh has the terminal.
        self._notify_action(f"attaching to {row.node} ({kind})… press the TUI's [b]Ctrl+C[/] to return")
        with self.app.suspend():
            if kind == "shell":
                launch.launch_shell(a, folder, cfg)
            else:
                launch.launch_claude(a, folder, cfg)
        self._notify_action(f"returned from {row.node}")
        self.refresh_jobs()

    def action_submit_cpu(self) -> None:
        self.app.push_screen(SubmitCpuModal(load()), self._on_cpu_submit)

    def _on_cpu_submit(self, params: CpuParams | None) -> None:
        if params is None:
            return
        self._notify_action(f"submitting CPU ({params.cores}c / {params.mem}G / {params.walltime})…")
        self._do_submit_cpu(params)

    @work(thread=True, group="action")
    def _do_submit_cpu(self, params: CpuParams) -> None:
        cfg = load()
        try:
            out = slurm.submit_cpu(cfg, params.cores, params.mem, params.walltime)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._notify_error, f"submit failed: {e}")
            return
        first_line = (out.strip().splitlines() or [""])[0]
        self.app.call_from_thread(self._notify_action, f"submitted: {first_line}")
        self.refresh_jobs()

    def action_submit_gpu(self) -> None:
        self.app.push_screen(SubmitGpuModal(load()), self._on_gpu_submit)

    def _on_gpu_submit(self, params: GpuParams | None) -> None:
        if params is None:
            return
        self._notify_action(
            f"submitting GPU ({params.gpus}× / {params.cores}c / {params.mem}G / {params.walltime})…"
        )
        self._do_submit_gpu(params)

    @work(thread=True, group="action")
    def _do_submit_gpu(self, params: GpuParams) -> None:
        cfg = load()
        try:
            out = slurm.submit_gpu(cfg, params.gpus, params.cores, params.mem, params.walltime)
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
