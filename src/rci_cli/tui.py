"""Interactive Textual TUI for rci-cli.

Live jobs dashboard with one-key actions: cancel selected, shell into the
compute node, launch claude or VS Code on the allocation, submit fresh CPU /
GPU allocations from a modal. Refresh is threaded so the UI never freezes
while ``squeue`` is in flight; ``App.suspend()`` is used to hand the terminal
over to ssh for shell-in / claude attach, then re-render on return.
"""

from __future__ import annotations

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
    OptionList,
    Select,
    Static,
)

from . import alloc as alloc_mod
from . import config as config_mod
from . import launch, setup as setup_mod, slurm, ssh, state
from .config import Config, load

REFRESH_INTERVAL = 5.0
ACTION_FADE_SECONDS = 6.0  # how long the inline action log lingers before auto-clearing

# squeue's ``%T`` long-form yields ``RUNNING``/``PENDING``/…; some setups still use
# the short codes (``R``/``PD``). Accept both so the action guards are robust.
RUNNING_STATES = frozenset({"R", "RUNNING"})


# ──────────────────────────── widgets ───────────────────────────────────────


class VimSelect(Select):
    """``Select`` with vim-style navigation inside the open dropdown.

    Adds ``j``/``k``/``g``/``G`` as cursor-down/up/first/last while the
    dropdown is open. When the Select is collapsed the actions are no-ops,
    so j/k don't fire from a closed-but-focused Select (a vim user pressing
    j there expects nothing — they haven't opened the menu yet).

    Trade-off: Textual's stock ``SelectOverlay`` consumes every printable
    key for type-to-search (typing "g" jumps to the first option matching
    "g"). That consumption stops printable keys from reaching our bindings,
    so type-to-search is disabled here (``type_to_search=False``). Acceptable
    for this TUI because every Select has a short fixed list (≤ 5 options)
    where Tab + arrow keys / j/k are faster than partial typing anyway.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("j", "vim_down", show=False),
        Binding("k", "vim_up", show=False),
        Binding("g", "vim_top", show=False),
        Binding("G", "vim_bottom", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        # Force type-to-search off so the overlay doesn't swallow ``j``/``k``
        # via its printable-key handler before our bindings see them.
        kwargs.setdefault("type_to_search", False)
        super().__init__(*args, **kwargs)

    def _overlay(self) -> OptionList | None:
        # The expanded dropdown is an ``OptionList`` child (``SelectOverlay``
        # subclasses it). Avoid importing the private ``SelectOverlay`` name
        # by matching on the public base class instead.
        return next(iter(self.query(OptionList)), None)

    def action_vim_down(self) -> None:
        if self.expanded and (ov := self._overlay()) is not None:
            ov.action_cursor_down()

    def action_vim_up(self) -> None:
        if self.expanded and (ov := self._overlay()) is not None:
            ov.action_cursor_up()

    def action_vim_top(self) -> None:
        if self.expanded and (ov := self._overlay()) is not None:
            ov.action_first()

    def action_vim_bottom(self) -> None:
        if self.expanded and (ov := self._overlay()) is not None:
            ov.action_last()


# ──────────────────────────── modals ────────────────────────────────────────


class ConfirmModal(ModalScreen[bool]):
    """Generic yes/no confirmation. Returns True/False via ``dismiss``.

    Defaults focus to ``No`` so a reflex Enter never confirms. Pass
    ``default_yes=True`` for dialogs where Enter-to-proceed is the desired
    affordance (the caller is asserting the reflex risk is acceptable).
    """

    # Disable Textual's "auto-focus first focusable" — we pick the
    # default ourselves in :meth:`on_mount`.
    AUTO_FOCUS: ClassVar[str] = ""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "yes", "Yes", show=False),
        Binding("n,q", "no", "No", show=False),
        Binding("escape", "no", "Cancel", show=False),
    ]

    def __init__(
        self, prompt: str, *, dangerous: bool = False, default_yes: bool = False
    ) -> None:
        super().__init__()
        self.prompt = prompt
        self.dangerous = dangerous
        self.default_yes = default_yes

    def compose(self) -> ComposeResult:
        # Plain Container is non-focusable by default, so the first Tab walks
        # straight between the Yes/No buttons without an intermediate stop.
        with Container(id="modal-box"):
            yield Label(self.prompt, id="modal-prompt")
            with Horizontal(id="modal-buttons"):
                # Keep ``-error`` (red) for dangerous confirms — that's a
                # safety affordance, not just a "primary action" hint.
                if self.dangerous:
                    yield Button("Yes", variant="error", id="yes")
                else:
                    yield Button("Yes", id="yes")
                yield Button("No", id="no")

    def on_mount(self) -> None:
        target = "#yes" if self.default_yes else "#no"
        self.query_one(target, Button).focus()

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
    """Result of :class:`NewInstanceModal`. Same shape for CPU and GPU jobs.

    The agent-specific fields (``permission_mode``, ``spawn_mode``, ``capacity``)
    are populated downstream when the Agent flow merges its
    :class:`AgentOptions` into the resources params before submission. Other
    flows leave them at their defaults and the agent launcher (irrelevant
    there) falls back to ``cfg.agent_*``.
    """

    partition: str
    cores: int
    gpus: int  # 0 → CPU job (dispatches to slurm.submit_cpu); else GPU job
    mem_gb: int
    walltime: str
    job_name: str = ""
    permission_mode: str = ""
    spawn_mode: str = ""
    capacity: int = 0

    @property
    def kind(self) -> str:
        return "gpu" if self.gpus > 0 else "cpu"


@dataclass(frozen=True)
class AgentOptions:
    """Result of :class:`AgentOptionsModal` — the ``claude remote-control`` flags.

    Merged into :class:`AllocParams` by the panel before submission so the
    Slurm and claude sides flow through the same plumbing as every other
    submit path.
    """

    permission_mode: str
    spawn_mode: str
    capacity: int


# ``claude remote-control --permission-mode`` choices.
PERMISSION_MODES: tuple[str, ...] = (
    "default",
    "acceptEdits",
    "auto",
    "bypassPermissions",
    "dontAsk",
    "plan",
)
# ``claude remote-control --spawn`` choices.
SPAWN_MODES: tuple[str, ...] = ("same-dir", "worktree", "session")


# Walltime is a Select (not an Input) so printable keys like ``q`` fall through
# to the modal-level cancel binding instead of being typed into the field.
# These covers the common cluster ranges (fast caps at 4h, normal at 24h, long
# at ~4d, extralong at 7d); custom values via config.toml / saved state are
# still respected — see :class:`NewInstanceModal.__init__`.
WALLTIME_PRESETS: tuple[str, ...] = (
    "0:30:00",
    "1:00:00",
    "2:00:00",
    "4:00:00",
    "8:00:00",
    "12:00:00",
    "1-00:00:00",
    "2-00:00:00",
    "4-00:00:00",
    "7-00:00:00",
)


def assemble_partition(ptype: str, pclass: str) -> str:
    """Compose the two dropdown values into a Slurm partition name."""
    return f"{ptype}{pclass}"


def validate_alloc(ptype: str, gpus: int, gpu_types: tuple[str, ...]) -> str | None:
    """Return an error message if the (type, gpus) combo is invalid, else ``None``.

    ``gpu_types`` is the list of partition types that accept ``--gres=gpu:N``
    on this cluster (from :attr:`Config.gpu_partition_types`). Anything not in
    that list rejects ``gpus > 0``.
    """
    if gpus > 0 and ptype not in gpu_types:
        choices = ", ".join(gpu_types) if gpu_types else "(none configured)"
        return f"Partition type '{ptype}' doesn't accept GPUs — pick one of: {choices}."
    return None


# Re-exported for callers that still import from ``rci_cli.tui``. The actual
# definitions live in :mod:`rci_cli.slurm` so the non-TUI CLI can use them
# without importing Textual.
parse_jobid_from_salloc = slurm.parse_jobid_from_salloc
last_meaningful_line = slurm.last_meaningful_line


class NewInstanceModal(ModalScreen["AllocParams | str | None"]):
    # Disable Textual's default "auto-focus the first focusable widget on
    # mount". The modal opens with NO inner focus: Enter submits defaults,
    # Tab walks into the form starting at partition-type.
    AUTO_FOCUS: ClassVar[str] = ""
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
        # ``q``/``escape`` close the modal from anywhere — priority=True so the
        # binding fires even when an Input has focus (otherwise integer inputs
        # for cores/mem/gpus would silently swallow ``q`` as a rejected
        # character and the user couldn't back out of the form).
        Binding("escape,q", "cancel", "Cancel", show=False, priority=True),
        # Enter is *not* priority — focused widgets must keep their native
        # Enter handling so the open Select overlay (a Widget, not a Screen)
        # can still confirm the highlighted option. The screen binding only
        # fires when nothing inside the modal has focus, i.e. the "modal just
        # opened" state, which is exactly when we want to submit the
        # prefilled defaults. Inputs handle Enter natively by emitting
        # ``Input.Submitted`` — see :meth:`_advance_focus` for the
        # advance-to-next-field behavior.
        # overlay confirms the highlighted option (its own screen handles it).
        Binding("enter", "submit", "Submit", show=False),
    ]

    def __init__(
        self,
        cfg: Config,
        *,
        allow_back: bool = False,
        default_name: str = "",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # When True, ``q``/``escape`` dismisses with the ``"back"`` sentinel
        # so the caller can re-open the previous step (e.g. FolderModal).
        # False for the standalone ``n`` flow where there's nothing to step
        # back to.
        self.allow_back = allow_back
        # Suggested job name (e.g. ``dev-3`` or ``editor``). Shown in the Name
        # input pre-filled; the user can override or blank it (blank reverts
        # to this suggestion on submit). Empty string falls back to
        # ``cfg.dev_job_name`` so the modal still has *some* name to submit.
        self._default_name = default_name or cfg.dev_job_name
        # Pre-fill from the last submitted params (persisted across sessions),
        # falling back to ``Config`` defaults if nothing's saved or a field
        # got corrupted. Validate enum-like fields so a stale state.json
        # doesn't crash the modal — fall back silently instead.
        last = state.get_last_instance_params() or {}
        cpu_cores, cpu_mem, cpu_time = cfg.cpu_defaults
        valid_classes = {value for _label, value in cfg.partition_classes}
        default_ptype = cfg.partition_types[0] if cfg.partition_types else "cpu"
        default_pclass = (
            cfg.partition_classes[0][1] if cfg.partition_classes else ""
        )

        ptype = last.get("partition_type")
        pclass = last.get("partition_class")
        cores = last.get("cores")
        gpus = last.get("gpus")
        mem_gb = last.get("mem_gb")
        walltime = last.get("walltime")

        self._init_ptype = (
            ptype if isinstance(ptype, str) and ptype in cfg.partition_types else default_ptype
        )
        self._init_pclass = (
            pclass if isinstance(pclass, str) and pclass in valid_classes else default_pclass
        )
        self._init_cores = str(cores) if isinstance(cores, int) and cores > 0 else str(cpu_cores)
        self._init_gpus = str(gpus) if isinstance(gpus, int) and gpus >= 0 else "0"
        self._init_mem = str(mem_gb) if isinstance(mem_gb, int) and mem_gb > 0 else str(cpu_mem)
        # Accept saved walltime only if it's a known-good value (preset or the
        # current config default). Drops garbage like ``"0:00:10"`` left over
        # from when this field was a free-form Input — Slurm parses ``H:MM:SS``
        # so 10 seconds rounds up to the 1-minute minimum, silently shrinking
        # the user's intended limit. Anything unrecognised falls back to the
        # config default and won't be re-persisted on submit.
        valid_times = set(WALLTIME_PRESETS) | {cpu_time}
        self._init_time = walltime if isinstance(walltime, str) and walltime in valid_times else cpu_time

    def compose(self) -> ComposeResult:
        # VerticalScroll so the modal stays usable on shorter terminals — the
        # form scrolls within the box once it grows past the viewport's height.
        # ``can_focus=False`` so Tab from the no-focus initial state skips
        # over the scroll container and lands directly on partition-type
        # (the user otherwise had to Tab twice). The form is short enough
        # that arrow-key scrolling inside the container isn't needed.
        with VerticalScroll(id="modal-box", can_focus=False):
            yield Label("[b]Resources[/b]", id="modal-title")
            yield Label("Job name")
            yield Input(
                value=self._default_name,
                id="job-name",
                placeholder=self._default_name,
            )
            yield Label("Partition")
            with Horizontal(id="partition-row"):
                yield VimSelect(
                    [(t, t) for t in self.cfg.partition_types],
                    value=self._init_ptype,
                    id="partition-type",
                    allow_blank=False,
                )
                yield VimSelect(
                    list(self.cfg.partition_classes),
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
            yield Label("Walltime")
            # Select rather than Input so printable keys (e.g. ``q``) don't
            # get typed into the field and instead fall through to the modal
            # cancel binding. Custom values from config.toml / saved state
            # land at the top of the list so they're still selectable.
            time_options = list(WALLTIME_PRESETS)
            if self._init_time and self._init_time not in time_options:
                time_options.insert(0, self._init_time)
            yield VimSelect(
                [(t, t) for t in time_options],
                value=self._init_time,
                id="time",
                allow_blank=False,
            )
            with Horizontal(id="modal-buttons"):
                # Visual order: Submit on the left (primary action),
                # Cancel on the right. Compose order matches because Tab
                # naturally walks form → buttons in reading order.
                yield Button("Submit", id="ok")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        # Sync the GPUs row visibility to the restored partition type — if the
        # user's last submission was a GPU job, the field should already be visible.
        self._apply_gpu_visibility(self._init_ptype)
        # No widget focus on open (AUTO_FOCUS = ""). Enter submits the
        # prefilled defaults; Tab walks into the form starting at
        # partition-type. Once inside the form, Enter advances field-by-field
        # — see :meth:`action_submit`.

    def _apply_gpu_visibility(self, ptype: str) -> None:
        is_gpu_type = ptype in self.cfg.gpu_partition_types
        self.query_one("#gpus-label", Label).display = is_gpu_type
        self.query_one("#gpus", Input).display = is_gpu_type

    @on(Select.Changed, "#partition-type")
    def _type_changed(self, event: Select.Changed) -> None:
        ptype = str(event.value)
        self._apply_gpu_visibility(ptype)
        # Quality-of-life: switching to a GPU-capable type prefills GPUs=1
        # (if still 0); switching to a non-GPU type resets it to 0 so the
        # validation always passes.
        gpus_input = self.query_one("#gpus", Input)
        if ptype not in self.cfg.gpu_partition_types:
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
        self._do_submit()

    def action_submit(self) -> None:
        # Without ``priority=True`` on the Enter binding, this action only
        # fires when no inner widget claims Enter — i.e. the "modal just
        # opened, no focus" state. Other Enter cases (Select dropdown,
        # Input.Submitted, Button.Pressed) are handled natively by the
        # focused widget. So this is straightforward: submit the defaults.
        self._do_submit()
        # Input (numeric fields) → step to the next focusable widget.
        self.focus_next()

    def _do_submit(self) -> None:
        try:
            ptype = str(self.query_one("#partition-type", Select).value)
            pclass = str(self.query_one("#partition-class", Select).value)
            cores = int(self.query_one("#cores", Input).value or "0")
            gpus = int(self.query_one("#gpus", Input).value or "0")
            mem_gb = int(self.query_one("#mem", Input).value or "0")
            walltime = str(self.query_one("#time", Select).value).strip()
        except ValueError:
            self.app.notify("Invalid number", severity="error")
            return
        # Job name: free-text Input; blank reverts to the suggestion so
        # there's no way to submit a nameless job.
        name = self.query_one("#job-name", Input).value.strip() or self._default_name
        err = validate_alloc(ptype, gpus, self.cfg.gpu_partition_types)
        if err is not None:
            self.app.notify(err, severity="error", timeout=6)
            return
        # Persist so the next session opens the modal prefilled with these values.
        # ``job_name`` is intentionally *not* persisted — it should re-suggest
        # the next free index on each open, not stick on the last value.
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
                job_name=name,
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


class AgentOptionsModal(ModalScreen["AgentOptions | str | None"]):
    """Configure the ``claude remote-control`` flags before picking resources.

    Step 2 of the Agent flow (folder → **agent options** → resources →
    submit). Keeping the claude knobs in a dedicated modal means the
    resources step looks identical for every flow.

    Dismiss values:
      - :class:`AgentOptions` → user clicked Submit
      - ``None`` → user clicked the explicit Cancel button (abort)
      - the string ``"back"`` → user pressed ``q``/``escape`` while
        ``allow_back=True`` so the caller can re-open the folder picker.
    """

    AUTO_FOCUS: ClassVar[str] = ""

    BINDINGS: ClassVar[list[BindingType]] = [
        # ``priority=True`` so q/escape always backs out even from inside
        # the capacity Input — mirrors NewInstanceModal's binding.
        Binding("escape,q", "cancel", "Cancel", show=False, priority=True),
        Binding("enter", "submit", "Submit", show=False),
    ]

    def __init__(
        self,
        cfg: Config,
        *,
        allow_back: bool = False,
        prefill: AgentOptions | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.allow_back = allow_back
        # Prefill takes priority when stepping back from the resources modal
        # — we want the user's prior choices visible. Otherwise fall back to
        # the cfg-default values.
        self._init_permission = (
            prefill.permission_mode if prefill else cfg.agent_permission_mode
        )
        self._init_spawn = prefill.spawn_mode if prefill else cfg.agent_spawn_mode
        self._init_capacity = str(prefill.capacity if prefill else cfg.agent_capacity)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="modal-box", can_focus=False):
            yield Label("[b]Agent options[/b]", id="modal-title")
            yield Label("Permission mode")
            yield VimSelect(
                [(m, m) for m in PERMISSION_MODES],
                value=self._init_permission,
                id="agent-permission-mode",
                allow_blank=False,
            )
            yield Label("Spawn")
            yield VimSelect(
                [(m, m) for m in SPAWN_MODES],
                value=self._init_spawn,
                id="agent-spawn-mode",
                allow_blank=False,
            )
            yield Label("Capacity")
            yield Input(
                value=self._init_capacity,
                id="agent-capacity",
                type="integer",
            )
            with Horizontal(id="modal-buttons"):
                yield Button("Next", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Input.Submitted)
    def _advance_focus(self) -> None:
        self.focus_next()

    @on(Button.Pressed, "#ok")
    def _ok(self) -> None:
        self._do_submit()

    def action_submit(self) -> None:
        self._do_submit()

    def _do_submit(self) -> None:
        permission_mode = str(self.query_one("#agent-permission-mode", Select).value)
        spawn_mode = str(self.query_one("#agent-spawn-mode", Select).value)
        try:
            capacity = int(self.query_one("#agent-capacity", Input).value or "0")
        except ValueError:
            self.app.notify("Invalid capacity", severity="error")
            return
        # 0/blank → cfg default rather than failing — claude rejects 0
        # outright, and the user almost certainly meant "leave it alone".
        if capacity <= 0:
            capacity = self.cfg.agent_capacity
        self.dismiss(
            AgentOptions(
                permission_mode=permission_mode,
                spawn_mode=spawn_mode,
                capacity=capacity,
            )
        )

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss("back" if self.allow_back else None)


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
                placeholder="e.g. myproj  or  /scratch/exp42",
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


class SetupModal(ModalScreen[bool]):
    """First-run wizard: collects the minimum fields needed to talk to a cluster
    and writes them to ``~/.config/rci-cli/config.toml``.

    Dismisses with ``True`` on save, ``False`` on cancel. Re-running with a
    populated config prefills every field — same widget doubles as ``,`` /
    edit-config (future enhancement, currently only triggered on first launch).
    """

    # First focusable widget (the user field) gets focus on mount. Unlike
    # NewInstanceModal we *want* the user to start typing immediately.
    AUTO_FOCUS: ClassVar[str] = "#setup-user"

    BINDINGS: ClassVar[list[BindingType]] = [
        # Priority so Esc backs out even while an Input has focus. There's no
        # ``q`` binding because every field is a free-text Input where ``q`` is
        # a legitimate character.
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        # Prefill from whatever's currently saved so re-running setup doesn't
        # start from blank fields. On a true first run, all fields are empty
        # and the ssh-host field defaults to the Config-level "rci".
        self._existing = config_mod.load()

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="modal-box", can_focus=False):
            yield Label("[b]rci-cli setup[/b]", id="modal-title")
            yield Static(
                "Fill in once; edit ~/.config/rci-cli/config.toml later to change.",
                id="setup-blurb",
            )
            yield Label("SSH username on the cluster")
            yield Input(
                value=self._existing.user,
                id="setup-user",
                placeholder="e.g. jdoe2",
            )
            yield Label("SSH host alias (from ~/.ssh/config)")
            yield Input(
                value=self._existing.ssh_host or "rci",
                id="setup-host",
            )
            yield Label("Home directory on the cluster")
            yield Input(
                value=self._existing.home,
                id="setup-home",
                placeholder="blank → /home/<user>",
            )
            with Horizontal(id="modal-buttons"):
                yield Button("Save", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#ok")
    def _on_ok(self) -> None:
        self._do_save()

    @on(Input.Submitted)
    def _on_input_submitted(self) -> None:
        # Enter on an Input advances field-by-field; on the last field it lands
        # on the Save button, where the user presses Enter again to commit.
        self.focus_next()

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def _do_save(self) -> None:
        user = self.query_one("#setup-user", Input).value.strip()
        if not user:
            self.app.notify("SSH username is required.", severity="error")
            self.query_one("#setup-user", Input).focus()
            return
        try:
            cfg = setup_mod.build_cfg(
                user=user,
                ssh_host=self.query_one("#setup-host", Input).value.strip(),
                home=self.query_one("#setup-home", Input).value.strip(),
            )
        except ValueError as e:
            self.app.notify(str(e), severity="error")
            return
        try:
            path = config_mod.save(cfg)
        except OSError as e:
            self.app.notify(f"Failed to save config: {e}", severity="error", timeout=8)
            return
        self.app.notify(f"Saved to {path}", timeout=4)
        self.dismiss(True)


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

    # Order = footer order. Reads as a workflow story left-to-right: get onto
    # the cluster (Frontend) → submit a job (Submit) → connect to it (Connect)
    # → open in an editor (Editor) → spawn agent (Agent) → tmux workspace
    # (Workspace) → maintenance (Refresh) → destructive (Delete). Every label
    # is cued by its key letter
    # so the footer reads as a menu. ``hjkl`` are kept off the action list
    # (`k`/`l` would collide with vim-style row movement) and bound below as
    # hidden navigation shortcuts.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("f", "shell_frontend", "Frontend"),
        Binding("s", "new_instance", "Submit"),
        Binding("c", "shell_into", "Connect"),
        Binding("e", "editor_into", "Editor"),
        Binding("a", "agent_into", "Agent"),
        Binding("w", "workspace_into", "Workspace"),
        Binding("r", "refresh", "Refresh"),
        Binding("d", "cancel_job", "Delete"),
        # vim-style row navigation, hidden from the footer to keep the action
        # menu compact. j/k move the row cursor; h/l scroll horizontally when
        # the table is wider than the viewport (no-op otherwise in row mode);
        # g/G jump to the first/last row (single-key form, not the gg chord —
        # consistent with k9s/lazygit and avoids tracking key state).
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
        Binding("h", "cursor_left", show=False),
        Binding("l", "cursor_right", show=False),
        Binding("g", "cursor_top", show=False),
        Binding("G", "cursor_bottom", show=False),
        # Incremental filter: ``/`` opens an inline Input below the table.
        # Typing filters live (substring match on jobid/name/state/partition);
        # Enter commits and hands focus back to the table with the filter
        # still active; Escape clears it. ``escape`` is priority so it fires
        # while the Input itself has focus.
        Binding("/", "start_filter", show=False),
        Binding("escape", "cancel_filter", show=False, priority=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # ``_rows`` is the full squeue snapshot; ``_visible_rows`` is what's
        # currently rendered after applying ``_filter``. Keeping both lets
        # the status line count the full set ("3 running") while the table
        # and the cursor-selection helpers operate on the filtered view.
        self._rows: list[JobRow] = []
        self._visible_rows: list[JobRow] = []
        self._filter: str = ""
        self._last_action: str = ""
        self._action_clear_timer = None  # type: ignore[var-annotated]
        # Carries the AgentOptions across the agent flow's 3 modal hops so
        # step-back from the resources modal can re-open the options modal
        # with the user's prior choices intact. Cleared after submission /
        # outright cancel.
        self._pending_agent_opts: AgentOptions | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Loading…", id="alloc-status")
            yield DataTable(id="jobs-table", zebra_stripes=True, cursor_type="row")
            yield Static("", id="job-detail")
            # Hidden until the user presses ``/``. Placeholder doubles as
            # in-place help so the keys are discoverable on first open.
            yield Input(placeholder="filter…  enter=apply  esc=clear", id="filter-input")
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
        # Filter input is hidden until ``/`` reveals it; toggling ``display``
        # collapses it out of the layout entirely so the panel looks identical
        # to the pre-filter layout when no search is active.
        self.query_one("#filter-input", Input).display = False
        self.refresh_jobs()
        self.set_interval(REFRESH_INTERVAL, self.refresh_jobs)

    # ----- data refresh (threaded) -----

    @work(thread=True, exclusive=True, group="refresh")
    def refresh_jobs(self) -> None:
        cfg = load()
        # Setup hasn't run yet — the SetupModal is still up on top of the
        # dashboard. Skip silently; the modal's save callback will fire a
        # one-off refresh once the user is set.
        if not cfg.user:
            return
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
        """Stash the latest squeue snapshot and trigger a render.

        Kept narrow so the refresh worker has a single tiny call site; all
        of the table-mutation work lives in :meth:`_render_rows`, which is
        also what the filter handlers call to re-render without a new
        squeue round-trip.
        """
        self._rows = rows
        self._render_rows()
        # Status line counts the *full* snapshot, not the filtered view —
        # ``N running`` shouldn't shrink just because you searched for one
        # name. (Filter is incremental and constantly changing; a moving
        # count would be noisy and uninformative.)
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

    def _render_rows(self) -> None:
        """Repaint the table from ``_rows`` honoring the active ``_filter``.

        Preserves the highlighted job across re-renders by jobid (rather
        than row index) so that filtering doesn't snap the cursor to an
        unrelated row when the visible subset shifts.
        """
        prior = self._selected_jobid()
        table = self.query_one("#jobs-table", DataTable)
        table.clear()
        self._visible_rows = self._apply_filter(self._rows)
        for r in self._visible_rows:
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
        if self._visible_rows:
            target = next(
                (i for i, r in enumerate(self._visible_rows) if r.jobid == prior),
                0,
            )
            try:
                table.move_cursor(row=target)
            except Exception:  # noqa: BLE001
                pass
        # Update the detail line for the (re-)selected row, if any.
        self._refresh_detail()

    def _apply_filter(self, rows: list[JobRow]) -> list[JobRow]:
        """Substring match against the columns a user is most likely to
        search by — jobid, name, state, partition. Case-insensitive."""
        if not self._filter:
            return list(rows)
        q = self._filter.lower()
        return [
            r for r in rows
            if q in r.jobid.lower()
            or q in r.name.lower()
            or q in r.state.lower()
            or q in r.partition.lower()
        ]

    # ----- selection helpers -----

    def _selected_row(self) -> JobRow | None:
        # Indexes ``_visible_rows`` — what's actually rendered — so that an
        # active filter doesn't desync the cursor position from the JobRow
        # the user sees highlighted.
        table = self.query_one("#jobs-table", DataTable)
        if not self._visible_rows:
            return None
        try:
            return self._visible_rows[table.cursor_row]
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

    # vim-key delegates to the inner DataTable. Bindings live on the panel
    # (not the table) so they keep firing while modal callbacks momentarily
    # steal focus, and so the screen-level vim mapping survives even if the
    # table widget is rebuilt by ``_apply_rows``.
    def action_cursor_down(self) -> None:
        self.query_one("#jobs-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#jobs-table", DataTable).action_cursor_up()

    def action_cursor_left(self) -> None:
        self.query_one("#jobs-table", DataTable).action_cursor_left()

    def action_cursor_right(self) -> None:
        self.query_one("#jobs-table", DataTable).action_cursor_right()

    def action_cursor_top(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        if table.row_count:
            table.move_cursor(row=0)

    def action_cursor_bottom(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        if table.row_count:
            table.move_cursor(row=table.row_count - 1)

    # ----- filter (``/``) -----

    def action_start_filter(self) -> None:
        """Reveal the filter Input and hand it focus.

        Pre-populates with the current filter so pressing ``/`` again is a
        re-edit (vim's ``/`` reopens the last search), not a reset.
        """
        inp = self.query_one("#filter-input", Input)
        inp.display = True
        inp.value = self._filter
        inp.focus()

    def action_cancel_filter(self) -> None:
        """Escape: clear any active filter and hide the Input.

        Bound at priority=True on the panel so it fires even when the Input
        itself is focused. When no filter is active and the Input is hidden,
        this is a no-op (the binding still fires but does nothing visible).
        """
        inp = self.query_one("#filter-input", Input)
        if not inp.display and not self._filter:
            return
        self._filter = ""
        inp.value = ""
        inp.display = False
        self.query_one("#jobs-table", DataTable).focus()
        self._render_rows()
        self._notify_action("filter cleared")

    @on(Input.Changed, "#filter-input")
    def _on_filter_changed(self, event: Input.Changed) -> None:
        # Live filter as the user types — instant feedback beats requiring
        # Enter for each preview. The cost is one re-render per keystroke,
        # which is cheap for the typical dozens-of-jobs case.
        self._filter = event.value
        self._render_rows()

    @on(Input.Submitted, "#filter-input")
    def _on_filter_submitted(self, event: Input.Submitted) -> None:
        # Enter commits: hide the input but keep the filter active so the
        # user can navigate the filtered view with j/k/d/c/etc. Escape (or
        # ``/`` then clearing) is the way to drop the filter.
        inp = self.query_one("#filter-input", Input)
        inp.display = False
        self.query_one("#jobs-table", DataTable).focus()
        if self._filter:
            self._notify_action(
                f"filter: {self._filter}  ({len(self._visible_rows)}/{len(self._rows)})"
            )

    def action_cancel_job(self) -> None:
        row = self._selected_row()
        if row is None:
            self.app.notify("Nothing selected.", severity="warning")
            return
        prompt = f"Cancel job [b]{row.jobid}[/] ([i]{row.name}[/], {row.state}) on {row.partition}?"
        self.app.push_screen(
            ConfirmModal(prompt, dangerous=True, default_yes=True),
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

    def action_agent_into(self) -> None:
        # Agent always spawns a new alloc, but we still ask for the working
        # folder first so ``claude remote-control`` starts in the right place.
        self._pick_folder_then("agent")

    def action_workspace_into(self) -> None:
        # Workspace mirrors the shell flow — pick folder, reuse an existing
        # dev alloc when there is one, else spawn dev-N. The tmux session
        # itself is keyed by the alloc's jobid, so subsequent presses on the
        # same row reattach to the same workspace.
        self._pick_folder_then("workspace")

    def action_shell_frontend(self) -> None:
        """Open an interactive ssh session to the login host (``cfg.ssh_host``).

        No folder prompt, no allocation — this is the bare ``ssh rci`` you'd
        type by hand. Useful for browsing files, running ``squeue``/``sacct``,
        or kicking off submissions outside the TUI.
        """
        cfg = load()
        self._notify_action(f"ssh → {cfg.ssh_host}… exit to return")
        with self.app.suspend():
            ssh.run(cfg.ssh_host, check=False)
        self._notify_action(f"returned from {cfg.ssh_host}")
        self.refresh_jobs()

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
        # Agent gets its own 3-step flow: folder → agent options → resources.
        # Always spawns a fresh ``agent-N`` (no reuse, no row-state check).
        if kind == "agent":
            self.app.push_screen(
                AgentOptionsModal(
                    cfg,
                    allow_back=True,
                    prefill=self._pending_agent_opts,
                ),
                lambda result: self._after_agent_options(folder, result),
            )
            return
        # Block early when the highlighted row exists but isn't connectable
        # (pending) — silently falling through to the New Instance modal made
        # it look like the action was ignored.
        row = self._selected_row()
        if row is not None and row.state not in RUNNING_STATES:
            self.app.notify(
                f"Job {row.jobid} is {row.state.lower()} — can't attach yet.",
                severity="warning",
                timeout=4,
            )
            return
        alloc = self._alloc_from_selected_row()
        if alloc is not None:
            self._attach_to(kind, alloc, folder)
            return
        # No usable selection — collect alloc params, submit, attach. Opt
        # into ``allow_back`` so q/escape on the resources modal returns
        # to the folder picker instead of dropping the whole flow.
        # The Editor flow spawns the singleton ``editor`` (only ever one);
        # everything else spawns ``dev-N`` with N = lowest unused suffix
        # among current jobs.
        if kind == "editor":
            default_name = cfg.editor_job_name
        else:
            default_name = self._suggest_dev_name(cfg)
        self.app.push_screen(
            NewInstanceModal(cfg, allow_back=True, default_name=default_name),
            lambda result: self._after_new_instance(kind, folder, result),
        )

    def _suggest_dev_name(self, cfg: Config) -> str:
        """Compute the next ``dev-N`` from the cached row set — no ssh round-trip."""
        n = slurm.lowest_unused_index((r.name for r in self._rows), cfg.dev_job_name)
        return f"{cfg.dev_job_name}-{n}"

    def _suggest_agent_name(self, cfg: Config) -> str:
        """Same gap-reuse logic as ``_suggest_dev_name``, but for the agent pool."""
        n = slurm.lowest_unused_index((r.name for r in self._rows), cfg.agent_job_name)
        return f"{cfg.agent_job_name}-{n}"

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

    # ----- agent flow (folder → agent options → resources → submit) ----------

    def _after_agent_options(
        self, folder: str, result: "AgentOptions | str | None"
    ) -> None:
        """Step 2 callback: route to resources modal or step back to folder."""
        if result is None:
            # Cancel: drop the whole flow and forget any cached options so
            # the next start opens fresh.
            self._pending_agent_opts = None
            return
        if result == "back":
            self._pick_folder_then("agent")
            return
        assert isinstance(result, AgentOptions)
        self._pending_agent_opts = result
        cfg = load()
        self.app.push_screen(
            NewInstanceModal(
                cfg,
                allow_back=True,
                default_name=self._suggest_agent_name(cfg),
            ),
            lambda r: self._after_agent_resources(folder, result, r),
        )

    def _after_agent_resources(
        self,
        folder: str,
        opts: AgentOptions,
        result: "AllocParams | str | None",
    ) -> None:
        """Step 3 callback: merge agent opts into AllocParams and submit, or
        step back to the agent-options modal with prefill intact."""
        if result is None:
            self._pending_agent_opts = None
            return
        if result == "back":
            cfg = load()
            self.app.push_screen(
                AgentOptionsModal(cfg, allow_back=True, prefill=opts),
                lambda r: self._after_agent_options(folder, r),
            )
            return
        assert isinstance(result, AllocParams)
        from dataclasses import replace as _replace
        merged = _replace(
            result,
            permission_mode=opts.permission_mode,
            spawn_mode=opts.spawn_mode,
            capacity=opts.capacity,
        )
        self._pending_agent_opts = None
        self._submit_then_attach("agent", merged, folder)

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
        self,
        kind: str,
        alloc: alloc_mod.Allocation,
        folder_arg: str = "",
        params: AllocParams | None = None,
    ) -> None:
        # The agent flow launches detached straight from the worker thread
        # (no UI suspend, no terminal handover) so it never reaches here —
        # only shell / editor do.
        cfg = load()
        folder_abs = launch.resolve_folder(folder_arg, cfg)
        if kind == "editor":
            self._notify_action(f"opening editor on {alloc.node} ({folder_abs})")
            launch.launch_editor(alloc, folder_abs, cfg)
            return
        if kind == "workspace":
            self._notify_action(
                f"opening workspace on {alloc.node} ({folder_abs})… detach with Ctrl-b d"
            )
            with self.app.suspend():
                launch.launch_workspace(alloc, folder_abs, cfg)
            self._notify_action(f"returned from {alloc.node} (workspace still running)")
            self.refresh_jobs()
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
        name = params.job_name or cfg.dev_job_name
        try:
            if params.gpus > 0:
                out = slurm.submit_gpu(
                    cfg, params.gpus, params.cores, params.mem_gb, params.walltime,
                    job_name=name,
                    partition=params.partition or None,
                )
            else:
                out = slurm.submit_cpu(
                    cfg, params.cores, params.mem_gb, params.walltime,
                    job_name=name,
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
        if kind == "agent":
            # Detached launch — stays in this worker thread (the ssh call
            # returns as soon as the bg cmd is posted) so the UI never
            # suspends. Just notify and refresh on the way out.
            assert params is not None
            folder_abs = launch.resolve_folder(folder, cfg)
            name = params.job_name or cfg.agent_job_name
            try:
                launch.launch_agent(
                    alloc_mod.Allocation(node=node, jobid=jobid),
                    folder_abs,
                    cfg,
                    name=name,
                    permission_mode=params.permission_mode or cfg.agent_permission_mode,
                    spawn_mode=params.spawn_mode or cfg.agent_spawn_mode,
                    capacity=params.capacity or cfg.agent_capacity,
                )
            except Exception as e:  # noqa: BLE001
                self.app.call_from_thread(
                    self._notify_error, f"agent launch failed: {e}"
                )
                return
            self.app.call_from_thread(
                self._notify_action,
                f"agent '{name}' launched on {node} — pair from claude.ai/code",
            )
            self.refresh_jobs()
            return
        self.app.call_from_thread(
            self._attach_to,
            kind,
            alloc_mod.Allocation(node=node, jobid=jobid),
            folder,
            params,
        )

    # ----- new instance only (no auto-attach) -----

    def action_new_instance(self) -> None:
        cfg = load()
        self.app.push_screen(
            NewInstanceModal(cfg, default_name=self._suggest_dev_name(cfg)),
            self._on_new_instance,
        )

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
        name = params.job_name or cfg.dev_job_name
        try:
            if params.kind == "cpu":
                out = slurm.submit_cpu(
                    cfg, params.cores, params.mem_gb, params.walltime,
                    job_name=name,
                    partition=params.partition or None,
                )
            else:
                out = slurm.submit_gpu(
                    cfg, params.gpus, params.cores, params.mem_gb, params.walltime,
                    job_name=name,
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

/* Inline filter bar — appears between #job-detail and #last-action when
   the user hits ``/``. Visually a thin yellow-cued strip so it reads as
   "interactive input area" rather than just another status line. */
#filter-input {
    height: 1;
    padding: 0 1;
    border: none;
    background: ansi_default;
    color: ansi_default;
}
#filter-input:focus { color: ansi_bright_white; }

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

/* Modals: bordered dialog, centered, fully transparent backdrop — the
   dashboard underneath shows through at full brightness around the modal
   box. The modal-box itself is opaque (default bg + cyan border) so its
   content stays crisp; only the area outside the box is "see-through". */
ConfirmModal, NewInstanceModal, AgentOptionsModal, FolderModal, SetupModal {
    align: center middle;
    background: transparent;
}

#setup-blurb {
    padding-bottom: 1;
    color: ansi_bright_black;
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

/* Match Select to Input. Textual draws Select's visible border on the inner
   ``SelectCurrent`` (not on Select itself), so we have to target that to swap
   the default ``tall`` border for the rounded grey/cyan look Inputs use. */
SelectCurrent {
    background: ansi_default;
    color: ansi_default;
    border: round ansi_bright_black;
}
Select:focus > SelectCurrent {
    border: round ansi_cyan;
}
/* Vertical spacing below the standalone walltime Select to mirror Input's
   margin-bottom. The two Selects inside #partition-row already get spacing
   from the row's own padding-bottom — overriding that here would double up. */
#time { margin-bottom: 1; }

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
    TITLE = "RCI Job Manager"
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
        # First-run: stack the setup wizard on top of the dashboard. The
        # JobsPanel's threaded refresh early-returns on empty user, so the
        # background tick stays silent until the modal saves.
        if config_mod.needs_setup():
            self.push_screen(SetupModal(), self._after_setup)

    def _after_setup(self, saved: bool) -> None:
        """Callback for :class:`SetupModal` — either kick off the first refresh
        with the freshly-saved config, or exit if the user cancelled (the
        dashboard can't do anything without a configured ``user``)."""
        if not saved:
            self.exit()
            return
        # New config — fire an immediate refresh; the interval tick continues normally.
        self.query_one(JobsPanel).refresh_jobs()

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
