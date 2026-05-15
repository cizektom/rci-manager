"""Slurm primitives — submit allocations, list/cancel jobs, look up node assignments.

All commands run on the login host via :mod:`rci_cli.ssh`. Output is parsed into
plain Python structures so callers (CLI/TUI) can render however they want.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ssh
from .config import Config

SQUEUE_LIST_FORMAT = "%.10i %.9P %.10j %.8T %.10M %.10l %.6D %R"


@dataclass(frozen=True)
class Job:
    jobid: str
    partition: str
    name: str
    state: str
    time_used: str = ""
    time_limit: str = ""
    nodes: str = ""
    reason_or_nodelist: str = ""


def submit_cpu(
    cfg: Config,
    cores: int,
    mem_gb: int,
    walltime: str,
    *,
    partition: str | None = None,
) -> str:
    """Submit a vscode CPU allocation. Returns the salloc output (job id printout).

    ``partition`` overrides ``cfg.cpu_partition`` when set — used by the TUI's
    New Instance modal where the user picks the partition explicitly.
    """
    cmd = (
        f"salloc --no-shell --partition={partition or cfg.cpu_partition} "
        f"--job-name={cfg.cpu_job_name} "
        f"--cpus-per-task={cores} --mem={mem_gb}G --time={walltime}"
    )
    return ssh.capture(cfg.ssh_host, cmd, check=False)


def submit_gpu(
    cfg: Config,
    gpus: int,
    cores: int,
    mem_gb: int,
    walltime: str,
    *,
    partition: str | None = None,
) -> str:
    """Submit a vscode-gpu GPU allocation. ``partition`` overrides cfg default."""
    cmd = (
        f"salloc --no-shell --partition={partition or cfg.gpu_partition} "
        f"--job-name={cfg.gpu_job_name} "
        f"--gres=gpu:{gpus} --cpus-per-task={cores} --mem={mem_gb}G --time={walltime}"
    )
    return ssh.capture(cfg.ssh_host, cmd, check=False)


def list_jobs(cfg: Config) -> str:
    """Return the raw ``squeue`` table for the configured user."""
    cmd = f"squeue -u {cfg.user} -o '{SQUEUE_LIST_FORMAT}'"
    return ssh.capture(cfg.ssh_host, cmd)


def jobs_by_name(cfg: Config, name: str, *, state: str = "RUNNING") -> list[str]:
    """Return job ids matching ``name`` in ``state``. Empty list if none."""
    cmd = f"squeue -u {cfg.user} -h -n {name} -t {state} -o '%i'"
    out = ssh.capture(cfg.ssh_host, cmd, check=False)
    return [line.strip() for line in out.splitlines() if line.strip()]


def node_for(cfg: Config, jobid: str) -> str:
    """Return the assigned node name for ``jobid``, or empty string if not yet assigned."""
    out = ssh.capture(cfg.ssh_host, f"squeue -j {jobid} -h -o '%N'", check=False)
    return out.strip()


def describe(cfg: Config, jobid: str) -> str:
    """Short human description of a job: ``id partition name state``."""
    return ssh.capture(cfg.ssh_host, f"squeue -j {jobid} -h -o '%i %P %j %T'", check=False)


def cancel(cfg: Config, jobid: str) -> int:
    return ssh.run(cfg.ssh_host, f"scancel {jobid}", check=False)


def cancel_all(cfg: Config) -> int:
    return ssh.run(cfg.ssh_host, f"scancel -u {cfg.user}", check=False)


def cancel_by_names(cfg: Config, names: list[str]) -> int:
    """``scancel`` doesn't take a comma-separated --name; run one pass per name."""
    parts = [f"scancel --name={n} -u {cfg.user}" for n in names]
    return ssh.run(cfg.ssh_host, "; ".join(parts), check=False)


def list_jobs_brief(cfg: Config) -> str:
    cmd = f"squeue -u {cfg.user} -h -o '%.10i %.9P %.10j %.8T'"
    return ssh.capture(cfg.ssh_host, cmd, check=False)
