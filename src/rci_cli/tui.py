"""Interactive Textual TUI — skeleton.

This is the home for "extended capabilities" beyond the one-shot CLI subcommands.
Day-one scope: a live jobs dashboard with one-key actions for cancel, attach a
shell, launch claude, and submit a new allocation. Future screens (log tailing,
GPU/CPU utilization, batch script editor) plug in here.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import DataTable, Footer, Header, Static

from . import slurm
from .config import load


class JobsScreen(Container):
    """Live ``squeue`` table for the configured user."""

    def compose(self) -> ComposeResult:
        yield Static("Jobs (press [b]r[/b] to refresh, [b]q[/b] to quit)", id="jobs-hint")
        yield DataTable(id="jobs-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table: DataTable = self.query_one("#jobs-table", DataTable)
        table.add_columns("JobID", "Partition", "Name", "State", "Time", "Limit", "Nodes", "Reason/Nodelist")
        self.refresh_jobs()

    def refresh_jobs(self) -> None:
        cfg = load()
        table: DataTable = self.query_one("#jobs-table", DataTable)
        table.clear()
        out = slurm.list_jobs(cfg)
        for line in out.splitlines()[1:]:  # skip header from squeue's own format
            parts = line.split(None, 7)
            if len(parts) >= 4:
                while len(parts) < 8:
                    parts.append("")
                table.add_row(*parts[:8])


class RciApp(App):
    """Top-level Textual app. Future: tabs for jobs, allocations, logs, settings."""

    CSS = """
    Screen { layout: vertical; }
    #jobs-hint { padding: 1 2; color: $accent; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield JobsScreen(id="jobs-screen")
        yield Footer()

    def action_refresh(self) -> None:
        screen = self.query_one("#jobs-screen", JobsScreen)
        screen.refresh_jobs()
