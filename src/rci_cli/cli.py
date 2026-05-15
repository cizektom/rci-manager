"""Typer entry point for the ``rci`` CLI.

Subcommands mirror the existing zsh ``rci-*`` helpers and add an interactive ``tui``.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich import print as rprint

from . import alloc as alloc_mod
from . import launch, slurm
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
    from .tui import RciApp

    RciApp().run()


def _cfg() -> Config:
    return load()


def _require_alloc(cfg: Config, *, require_gpu: bool = False) -> alloc_mod.Allocation:
    """Either pick an existing allocation or submit one. Exits on failure."""
    try:
        return alloc_mod.select_or_submit(cfg, require_gpu=require_gpu)
    except alloc_mod.AllocationError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def ssh() -> None:
    """Open an interactive shell on the RCI login host (``ssh rci``)."""
    cfg = _cfg()
    sys.exit(ssh_mod.run(cfg.ssh_host, "", tty=False, check=False))


@app.command()
def jobs() -> None:
    """List your jobs on RCI (``squeue -u <user>``)."""
    cfg = _cfg()
    typer.echo(slurm.list_jobs(cfg))


@app.command()
def cpu(
    cores: Annotated[int, typer.Option("-c", "--cores")] = -1,
    mem: Annotated[int, typer.Option("-m", "--mem", help="memory in GB")] = -1,
    time: Annotated[str, typer.Option("-t", "--time", help="walltime, e.g. 4:00:00")] = "",
) -> None:
    """Submit a CPU ``vscode`` allocation."""
    cfg = _cfg()
    d_cores, d_mem, d_time = cfg.cpu_defaults
    out = slurm.submit_cpu(
        cfg,
        cores if cores > 0 else d_cores,
        mem if mem > 0 else d_mem,
        time or d_time,
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
    """Submit a GPU ``vscode-gpu`` allocation."""
    cfg = _cfg()
    d_gpus, d_cores, d_mem, d_time = cfg.gpu_defaults
    out = slurm.submit_gpu(
        cfg,
        gpus if gpus > 0 else d_gpus,
        cores if cores > 0 else d_cores,
        mem if mem > 0 else d_mem,
        time or d_time,
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


@app.command("cancel-vscode")
def cancel_vscode() -> None:
    """Cancel all ``vscode`` / ``vscode-gpu`` allocations (asks for confirmation)."""
    cfg = _cfg()
    ids = slurm.jobs_by_name(cfg, cfg.cpu_job_name, state="RUNNING") + slurm.jobs_by_name(
        cfg, cfg.gpu_job_name, state="RUNNING"
    )
    if not ids:
        rprint("No VS Code allocations to cancel.")
        return
    rprint("[bold]Will cancel these VS Code jobs:[/bold]")
    for jid in ids:
        rprint(f"  {jid}")
    if not _confirm("Proceed? [y/N] "):
        rprint("Aborted.")
        raise typer.Exit(code=1)
    rc = slurm.cancel_by_names(cfg, [cfg.cpu_job_name, cfg.gpu_job_name])
    rprint("Done." if rc == 0 else f"[red]scancel exited {rc}[/red]")
    raise typer.Exit(code=rc)


@app.command()
def claude(
    folder: Annotated[str, typer.Argument(help="folder on the compute node")] = "",
    suffix: Annotated[
        str,
        typer.Argument(help="optional suffix for the tmux session name (parallel sessions)"),
    ] = "",
    gpu: Annotated[bool, typer.Option("--gpu", help="require a GPU allocation")] = False,
) -> None:
    """Run ``claude`` on a compute node inside a persistent tmux session.

    Re-running the same command (same folder, same suffix) attaches back to the
    running session — ssh disconnects don't kill the work.
    """
    cfg = _cfg()
    a = _require_alloc(cfg, require_gpu=gpu)
    sys.exit(launch.launch_claude(a, launch.resolve_folder(folder, cfg), cfg, suffix=suffix))


@app.command()
def code(
    folder: Annotated[str, typer.Argument()] = "",
    gpu: Annotated[bool, typer.Option("--gpu", help="require a GPU allocation")] = False,
) -> None:
    """Open VS Code Remote-SSH on the strongest existing vscode allocation."""
    cfg = _cfg()
    a = _require_alloc(cfg, require_gpu=gpu)
    sys.exit(launch.launch_code(a, launch.resolve_folder(folder, cfg), cfg))


@app.command()
def shell(
    folder: Annotated[str, typer.Argument()] = "",
    suffix: Annotated[
        str,
        typer.Argument(help="optional suffix for the tmux session name (parallel sessions)"),
    ] = "",
    gpu: Annotated[bool, typer.Option("--gpu", help="require a GPU allocation")] = False,
) -> None:
    """Interactive bash on the compute node inside a persistent tmux session.

    Re-running the same command attaches back to the running session.
    """
    cfg = _cfg()
    a = _require_alloc(cfg, require_gpu=gpu)
    sys.exit(launch.launch_shell(a, launch.resolve_folder(folder, cfg), cfg, suffix=suffix))


@app.command()
def alloc(
    gpu: Annotated[bool, typer.Option("--gpu", help="require a GPU allocation")] = False,
) -> None:
    """Print ``<node> <jobid>`` of an existing allocation, submitting one if needed.

    Scripting-friendly equivalent of the zsh ``_rci_alloc`` helper.
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
    """Launch the interactive TUI (Textual). Skeleton — extended capabilities WIP."""
    from .tui import RciApp

    RciApp().run()


@app.command()
def version() -> None:
    """Print the rci-cli version."""
    from . import __version__

    typer.echo(__version__)
