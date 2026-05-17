"""Typer entry point for the ``rci`` CLI.

Subcommands wrap the Slurm primitives (``squeue`` / ``salloc`` / ``scancel``)
and add an interactive ``tui`` and a first-run ``setup`` flow.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich import print as rprint

from . import alloc as alloc_mod
from . import config as config_mod
from . import launch, setup as setup_mod, slurm
from . import ssh as ssh_mod
from .config import Config, load

app = typer.Typer(
    no_args_is_help=False,
    help="CLI + TUI for the RCI CVUT Slurm cluster. Bare `rci` opens the TUI.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Bare ``rci`` (no subcommand) launches the TUI. Pipe to fall back to help."""
    if ctx.invoked_subcommand is not None:
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        typer.echo(ctx.get_help())
        return
    # First-run: walk the user through setup before opening the TUI so the
    # dashboard's first refresh has somewhere to talk to.
    if config_mod.needs_setup():
        setup_mod.run_cli()
    from .tui import RciApp

    RciApp().run()


def _cfg() -> Config:
    """Return the active config, exiting with a setup hint if not yet configured."""
    cfg = load()
    if config_mod.needs_setup(cfg):
        rprint("[yellow]rci-cli isn't configured yet.[/yellow] Run [b]rci setup[/b] first.")
        raise typer.Exit(code=2)
    return cfg


def _require_alloc(
    cfg: Config,
    *,
    require_gpu: bool = False,
    spawn_name: str | None = None,
) -> alloc_mod.Allocation:
    """Either pick an existing allocation or submit one. Exits on failure."""
    try:
        return alloc_mod.select_or_submit(
            cfg, require_gpu=require_gpu, spawn_name=spawn_name
        )
    except alloc_mod.AllocationError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def setup() -> None:
    """Run the first-run setup wizard (or edit ``~/.config/rci-cli/config.toml`` by hand)."""
    setup_mod.run_cli()


@app.command()
def ssh() -> None:
    """Open an interactive shell on the cluster login host (``ssh <ssh_host>``)."""
    cfg = _cfg()
    sys.exit(ssh_mod.run(cfg.ssh_host, "", tty=False, check=False))


@app.command()
def jobs() -> None:
    """List your jobs (``squeue -u <user>``)."""
    cfg = _cfg()
    typer.echo(slurm.list_jobs(cfg))


@app.command()
def cpu(
    cores: Annotated[int, typer.Option("-c", "--cores")] = -1,
    mem: Annotated[int, typer.Option("-m", "--mem", help="memory in GB")] = -1,
    time: Annotated[str, typer.Option("-t", "--time", help="walltime, e.g. 4:00:00")] = "",
) -> None:
    """Submit a CPU allocation. Names it ``dev-N`` (lowest unused N)."""
    cfg = _cfg()
    d_cores, d_mem, d_time = cfg.cpu_defaults
    out = slurm.submit_cpu(
        cfg,
        cores if cores > 0 else d_cores,
        mem if mem > 0 else d_mem,
        time or d_time,
        job_name=slurm.next_indexed_name(cfg, cfg.dev_job_name),
    )
    typer.echo(out)
    typer.echo()
    typer.echo(slurm.list_jobs(cfg))


@app.command()
def gpu(
    gpus: Annotated[int, typer.Option("-g", "--gpus")] = -1,
    cores: Annotated[int, typer.Option("-c", "--cores")] = -1,
    mem: Annotated[int, typer.Option("-m", "--mem", help="memory in GB")] = -1,
    time: Annotated[str, typer.Option("-t", "--time", help="walltime, e.g. 4:00:00")] = "",
) -> None:
    """Submit a GPU allocation. Names it ``dev-N`` (same pool as ``rci cpu``)."""
    cfg = _cfg()
    d_gpus, d_cores, d_mem, d_time = cfg.gpu_defaults
    out = slurm.submit_gpu(
        cfg,
        gpus if gpus > 0 else d_gpus,
        cores if cores > 0 else d_cores,
        mem if mem > 0 else d_mem,
        time or d_time,
        job_name=slurm.next_indexed_name(cfg, cfg.dev_job_name),
    )
    typer.echo(out)
    typer.echo()
    typer.echo(slurm.list_jobs(cfg))


def _confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return False
    ans = input(prompt).strip().lower()
    return ans in {"y", "yes"}


@app.command()
def cancel(jobid: Annotated[str, typer.Argument(help="Slurm job id to cancel")]) -> None:
    """Cancel a specific job by ID."""
    cfg = _cfg()
    info = slurm.describe(cfg, jobid)
    if not info:
        rprint(f"[yellow]Job {jobid} not found (already finished?).[/yellow]")
        raise typer.Exit(code=1)
    rc = slurm.cancel(cfg, jobid)
    if rc == 0:
        rprint(f"[green]Cancelled:[/green] {info}")
    raise typer.Exit(code=rc)


@app.command("cancel-all")
def cancel_all() -> None:
    """Cancel ALL of your jobs (asks for confirmation)."""
    cfg = _cfg()
    jobs_out = slurm.list_jobs_brief(cfg)
    if not jobs_out:
        rprint("No jobs to cancel.")
        return
    rprint("[bold]Will cancel ALL of your jobs:[/bold]")
    for line in jobs_out.splitlines():
        rprint(f"  {line}")
    if not _confirm("Proceed? [y/N] "):
        rprint("Aborted.")
        raise typer.Exit(code=1)
    rc = slurm.cancel_all(cfg)
    rprint("Done." if rc == 0 else f"[red]scancel exited {rc}[/red]")
    raise typer.Exit(code=rc)


@app.command("cancel-dev")
def cancel_dev() -> None:
    """Cancel all rci-cli managed allocations (``dev*`` + ``editor`` + ``agent*``)."""
    cfg = _cfg()
    jobs = (
        slurm.jobs_by_prefix(cfg, cfg.dev_job_name)
        + slurm.jobs_by_prefix(cfg, cfg.editor_job_name)
        + slurm.jobs_by_prefix(cfg, cfg.agent_job_name)
    )
    if not jobs:
        rprint("No rci-managed allocations to cancel.")
        return
    rprint("[bold]Will cancel these allocations:[/bold]")
    for j in jobs:
        rprint(f"  {j.jobid}  {j.name}")
    if not _confirm("Proceed? [y/N] "):
        rprint("Aborted.")
        raise typer.Exit(code=1)
    rc = slurm.cancel_jobids(cfg, [j.jobid for j in jobs])
    rprint("Done." if rc == 0 else f"[red]scancel exited {rc}[/red]")
    raise typer.Exit(code=rc)


@app.command()
def editor(
    folder: Annotated[str, typer.Argument()] = "",
    gpu: Annotated[bool, typer.Option("--gpu", help="require a GPU allocation")] = False,
) -> None:
    """Open the configured editor (VS Code Remote-SSH) on the strongest existing allocation.

    Spawns a singleton ``editor`` allocation when none is available — subsequent
    invocations reuse it.
    """
    cfg = _cfg()
    a = _require_alloc(cfg, require_gpu=gpu, spawn_name=cfg.editor_job_name)
    sys.exit(launch.launch_editor(a, launch.resolve_folder(folder, cfg), cfg))


@app.command()
def agent(
    folder: Annotated[str, typer.Argument()] = "",
    gpu_alloc: Annotated[bool, typer.Option("--gpu", help="request a GPU allocation")] = False,
    cores: Annotated[int, typer.Option("-c", "--cores")] = -1,
    mem: Annotated[int, typer.Option("-m", "--mem", help="memory in GB")] = -1,
    time_: Annotated[str, typer.Option("-t", "--time", help="walltime, e.g. 4:00:00")] = "",
    gpus_n: Annotated[int, typer.Option("-g", "--gpus")] = -1,
    name: Annotated[str, typer.Option("--name", help="claude session name (defaults to job name)")] = "",
    permission_mode: Annotated[
        str, typer.Option("--permission-mode", help="acceptEdits|auto|bypassPermissions|default|dontAsk|plan")
    ] = "",
    spawn: Annotated[str, typer.Option("--spawn", help="same-dir|worktree|session")] = "",
    capacity: Annotated[int, typer.Option("--capacity", help="max concurrent sessions")] = -1,
) -> None:
    """Spawn a fresh ``agent-N`` allocation and run ``claude remote-control`` on it.

    Always submits a new allocation — each ``claude remote-control`` run is an
    independent server you pair with from claude.ai/code or the mobile app.
    Defaults for ``--permission-mode`` / ``--spawn`` / ``--capacity`` come from
    ``agent_*`` fields in ``~/.config/rci-cli/config.toml``.
    """
    import time as _time

    cfg = _cfg()
    job_name = slurm.next_indexed_name(cfg, cfg.agent_job_name)
    if gpu_alloc:
        d_g, d_c, d_m, d_t = cfg.gpu_defaults
        out = slurm.submit_gpu(
            cfg,
            gpus_n if gpus_n > 0 else d_g,
            cores if cores > 0 else d_c,
            mem if mem > 0 else d_m,
            time_ or d_t,
            job_name=job_name,
        )
    else:
        d_c, d_m, d_t = cfg.cpu_defaults
        out = slurm.submit_cpu(
            cfg,
            cores if cores > 0 else d_c,
            mem if mem > 0 else d_m,
            time_ or d_t,
            job_name=job_name,
        )
    typer.echo(out)
    jobid = slurm.parse_jobid_from_salloc(out)
    if jobid is None:
        rprint(f"[red]submit failed:[/red] {slurm.last_meaningful_line(out)}")
        raise typer.Exit(code=1)
    # Poll for node assignment — cpufast/gpufast usually schedule in seconds.
    deadline = _time.time() + 30.0
    node = ""
    while _time.time() < deadline:
        node = slurm.node_for(cfg, jobid)
        if node:
            break
        _time.sleep(1.0)
    if not node:
        rprint(f"[red]job {jobid} didn't get a node assigned within 30s[/red]")
        raise typer.Exit(code=1)
    rc = launch.launch_agent(
        alloc_mod.Allocation(node=node, jobid=jobid),
        launch.resolve_folder(folder, cfg),
        cfg,
        name=name or job_name,
        permission_mode=permission_mode or cfg.agent_permission_mode,
        spawn_mode=spawn or cfg.agent_spawn_mode,
        capacity=capacity if capacity > 0 else cfg.agent_capacity,
    )
    raise typer.Exit(code=rc)


@app.command()
def shell(
    folder: Annotated[str, typer.Argument()] = "",
    gpu: Annotated[bool, typer.Option("--gpu", help="require a GPU allocation")] = False,
) -> None:
    """Interactive bash on the compute node."""
    cfg = _cfg()
    a = _require_alloc(cfg, require_gpu=gpu)
    sys.exit(launch.launch_shell(a, launch.resolve_folder(folder, cfg), cfg))


@app.command()
def workspace(
    folder: Annotated[str, typer.Argument()] = "",
    gpu: Annotated[bool, typer.Option("--gpu", help="require a GPU allocation")] = False,
) -> None:
    """Open a tmux workspace on the compute node (2 claude panes + bash).

    Reuses an existing rci-managed allocation if one is running, otherwise
    spawns a ``dev-N`` (same pool as ``rci shell``). Subsequent invocations
    against the same job reattach to the live session — disconnect-safe.
    """
    cfg = _cfg()
    a = _require_alloc(cfg, require_gpu=gpu)
    sys.exit(launch.launch_workspace(a, launch.resolve_folder(folder, cfg), cfg))


@app.command()
def alloc(
    gpu: Annotated[bool, typer.Option("--gpu", help="require a GPU allocation")] = False,
) -> None:
    """Print ``<node> <jobid>`` of an existing allocation, submitting one if needed.

    Scripting-friendly.
    """
    cfg = _cfg()
    a = _require_alloc(cfg, require_gpu=gpu)
    typer.echo(f"{a.node} {a.jobid}")


@app.command()
def port(
    spec: Annotated[str, typer.Argument(help="<local-port>[:<remote-port>] — e.g. 8888 or 8888:9000")],
) -> None:
    """Forward a local port to the compute node (Jupyter, Tensorboard, …). Ctrl-C to stop."""
    if ":" in spec:
        local_s, remote_s = spec.split(":", 1)
    else:
        local_s = remote_s = spec
    try:
        local_p, remote_p = int(local_s), int(remote_s)
    except ValueError:
        rprint(f"[red]Invalid port spec '{spec}': expected <local>[:<remote>], integers only.[/red]")
        raise typer.Exit(code=1) from None
    cfg = _cfg()
    a = _require_alloc(cfg, require_gpu=False)
    rprint(f"→ {a.node} (job {a.jobid}): forwarding localhost:{local_p} → {a.node}:{remote_p}")
    rprint("   Ctrl-C to stop.")
    sys.exit(ssh_mod.port_forward(a.node, local_p, remote_p))


@app.command()
def tui() -> None:
    """Launch the interactive TUI (Textual)."""
    from .tui import RciApp

    RciApp().run()


@app.command()
def version() -> None:
    """Print the rci-cli version."""
    from . import __version__

    typer.echo(__version__)
