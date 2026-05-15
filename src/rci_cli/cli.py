"""Typer entry point for the ``rci`` CLI.

Subcommands mirror the existing zsh ``rci-*`` helpers and add an interactive ``tui``.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich import print as rprint

from . import alloc as alloc_mod
from . import launch, session, slurm
from . import ssh as ssh_mod
from .config import Config, load

app = typer.Typer(
    no_args_is_help=True,
    help="CLI + TUI for the RCI CVUT Slurm cluster.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _cfg() -> Config:
    return load()


def _require_alloc(cfg: Config, *, require_gpu: bool = False) -> alloc_mod.Allocation:
    """Either pick an existing allocation or submit one. Exits on failure."""
    try:
        return alloc_mod.select_or_submit(cfg, require_gpu=require_gpu)
    except alloc_mod.AllocationError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e


def _require_existing_alloc(cfg: Config) -> alloc_mod.Allocation:
    """Read-only: pick the strongest existing allocation; exit if none."""
    a = alloc_mod.find_strongest(cfg)
    if a is None:
        rprint("[yellow]No running vscode allocation.[/yellow]")
        raise typer.Exit(code=1)
    return a


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
    suffix: Annotated[str, typer.Argument(help="optional session-name suffix")] = "",
    gpu: Annotated[bool, typer.Option("--gpu", help="require a GPU allocation")] = False,
) -> None:
    """Run ``claude`` on a compute node, inside a persistent zellij session.

    Reconnect to the same folder and the session is re-attached automatically.
    Pass a ``suffix`` to run multiple parallel sessions on the same folder.
    """
    cfg = _cfg()
    folder_abs = launch.resolve_folder(folder, cfg)
    sess = session.session_name("claude", folder_abs, suffix, home=cfg.home)
    a = _require_alloc(cfg, require_gpu=gpu)
    sys.exit(launch.launch_claude(a, folder_abs, sess, cfg))


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
    suffix: Annotated[str, typer.Argument(help="optional session-name suffix")] = "",
    gpu: Annotated[bool, typer.Option("--gpu", help="require a GPU allocation")] = False,
) -> None:
    """Interactive bash on the compute node, inside a persistent zellij session."""
    cfg = _cfg()
    folder_abs = launch.resolve_folder(folder, cfg)
    sess = session.session_name("shell", folder_abs, suffix, home=cfg.home)
    a = _require_alloc(cfg, require_gpu=gpu)
    sys.exit(launch.launch_shell(a, folder_abs, sess, cfg))


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


@app.command(name="install-zellij")
def install_zellij_cmd() -> None:
    """Install zellij (static musl) + claude layout into ``~/bin`` on the login node.

    Shared with compute nodes via your home directory — they pick it up on PATH
    once ``rci claude`` / ``rci shell`` prepend ``$HOME/bin``.
    """
    cfg = _cfg()
    sys.exit(session.install_zellij(cfg))


@app.command()
def sessions() -> None:
    """List zellij sessions on the strongest existing allocation (read-only)."""
    cfg = _cfg()
    a = _require_existing_alloc(cfg)
    rprint(f"→ {a.node} (job {a.jobid}):")
    names = session.list_sessions(a.node)
    if not names:
        rprint("  (no zellij sessions)")
        return
    for n in names:
        rprint(f"  {n}")


@app.command(name="kill-session")
def kill_session_cmd(
    name: Annotated[str, typer.Argument(help="session name (omit when using --all)")] = "",
    all_: Annotated[bool, typer.Option("--all", help="kill every zellij session on the node")] = False,
) -> None:
    """Kill a zellij session by name (or all of them) without attaching first."""
    if not name and not all_:
        rprint("Usage: rci kill-session <name> | --all  (see `rci sessions`)")
        raise typer.Exit(code=1)
    cfg = _cfg()
    a = _require_existing_alloc(cfg)
    if all_:
        rprint(f"→ {a.node}: killing all zellij sessions")
        sys.exit(session.kill_all_sessions(a.node))
    rprint(f"→ {a.node}: killing zellij session '{name}'")
    sys.exit(session.kill_session(a.node, name))


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
