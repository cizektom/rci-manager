"""Slurm primitives — submit allocations, list/cancel jobs, look up node assignments.

All commands run on the login host via :mod:`rci_cli.ssh`. Output is parsed into
plain Python structures so callers (CLI/TUI) can render however they want.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import ssh
from .config import Config

# Columns: jobid, partition, name, state, time-used, time-limit,
#          cpus-requested, memory-min-per-node, generic-resources (gres), node/reason.
# Reads as a single squeue call so we don't have to follow up with scontrol per job.
SQUEUE_LIST_FORMAT = "%.10i %.9P %.12j %.8T %.10M %.10l %.5C %.6m %.8b %R"


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
    gres: str = ""


def submit_cpu(
    cfg: Config,
    cores: int,
    mem_gb: int,
    walltime: str,
    *,
    job_name: str,
    partition: str | None = None,
) -> str:
    """Submit a CPU allocation. Returns the salloc output (job id printout).

    ``job_name`` is required — callers compute it via :func:`next_indexed_name`
    (or pass a fixed singleton like ``"editor"``). ``partition`` overrides
    ``cfg.cpu_partition`` when set.
    """
    cmd = (
        f"salloc --no-shell --partition={partition or cfg.cpu_partition} "
        f"--job-name={job_name} "
        f"--cpus-per-task={cores} --mem={mem_gb}G --time={walltime}"
    )
    # salloc writes "Granted job allocation NNN" to stderr — merge so the caller
    # can parse the job id out of the returned string.
    return ssh.capture(cfg.ssh_host, cmd, check=False, merge_stderr=True)


def submit_gpu(
    cfg: Config,
    gpus: int,
    cores: int,
    mem_gb: int,
    walltime: str,
    *,
    job_name: str,
    partition: str | None = None,
) -> str:
    """Submit a GPU allocation. ``job_name`` is required; ``partition`` overrides cfg default."""
    cmd = (
        f"salloc --no-shell --partition={partition or cfg.gpu_partition} "
        f"--job-name={job_name} "
        f"--gres=gpu:{gpus} --cpus-per-task={cores} --mem={mem_gb}G --time={walltime}"
    )
    return ssh.capture(cfg.ssh_host, cmd, check=False, merge_stderr=True)


def list_jobs(cfg: Config) -> str:
    """Return the raw ``squeue`` table for the configured user."""
    cmd = f"squeue -u {cfg.user} -o '{SQUEUE_LIST_FORMAT}'"
    return ssh.capture(cfg.ssh_host, cmd)


def jobs_by_prefix(
    cfg: Config, prefix: str, *, state: str = "RUNNING"
) -> list[Job]:
    """Return rci-managed jobs whose name is ``prefix`` or starts with ``prefix-``.

    Matches both the singleton form (``"editor"``) and the indexed form
    (``"dev-1"``, ``"dev-2"``, …). Returns ``Job`` objects so callers can
    inspect ``gres`` to distinguish CPU vs GPU allocations.
    """
    fmt = "%i %P %j %T %M %l %b %R"
    cmd = f"squeue -u {cfg.user} -h -t {state} -o '{fmt}'"
    out = ssh.capture(cfg.ssh_host, cmd, check=False)
    rows: list[Job] = []
    for line in out.splitlines():
        parts = line.split(None, 7)
        if len(parts) < 4:
            continue
        while len(parts) < 8:
            parts.append("")
        name = parts[2]
        if name != prefix and not name.startswith(f"{prefix}-"):
            continue
        rows.append(
            Job(
                jobid=parts[0],
                partition=parts[1],
                name=name,
                state=parts[3],
                time_used=parts[4],
                time_limit=parts[5],
                gres=parts[6],
                reason_or_nodelist=parts[7],
            )
        )
    return rows


def lowest_unused_index(names: Iterable[str], prefix: str) -> int:
    """Lowest integer ≥ 1 not used as ``<prefix>-N`` in ``names``.

    Pure helper — same logic the CLI uses on a squeue query and the TUI uses
    on its cached row set. Cancelled/finished jobs aren't in either source,
    so their numbers naturally become reusable.
    """
    used = set()
    head = f"{prefix}-"
    for n in names:
        if n.startswith(head):
            suffix = n[len(head):]
            if suffix.isdigit():
                used.add(int(suffix))
    n = 1
    while n in used:
        n += 1
    return n


def next_indexed_name(cfg: Config, prefix: str) -> str:
    """Compute ``<prefix>-N`` for the next available N by querying squeue.

    Used by the CLI side (``rci cpu`` / ``rci gpu`` / ``rci shell``). The TUI
    avoids the round-trip by computing from its cached rows.
    """
    cmd = f"squeue -u {cfg.user} -h -o '%j'"
    out = ssh.capture(cfg.ssh_host, cmd, check=False)
    names = (line.strip() for line in out.splitlines() if line.strip())
    return f"{prefix}-{lowest_unused_index(names, prefix)}"


def has_gpu(job: Job) -> bool:
    """True iff the job's ``gres`` advertises any GPU."""
    g = (job.gres or "").strip()
    if not g or g in ("N/A", "(null)"):
        return False
    for part in g.split(","):
        if part.strip().startswith("gpu"):
            return True
    return False


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


def cancel_jobids(cfg: Config, jobids: list[str]) -> int:
    """Cancel a batch of job ids in a single ``scancel`` call."""
    if not jobids:
        return 0
    return ssh.run(cfg.ssh_host, "scancel " + " ".join(jobids), check=False)


def list_jobs_brief(cfg: Config) -> str:
    cmd = f"squeue -u {cfg.user} -h -o '%.10i %.9P %.10j %.8T'"
    return ssh.capture(cfg.ssh_host, cmd, check=False)
