"""Slurm primitives — submit allocations, list/cancel jobs, look up node assignments.

All commands run on the login host via :mod:`rci_cli.ssh`. Output is parsed into
plain Python structures so callers (CLI/TUI) can render however they want.
"""

from __future__ import annotations

import re
import shlex
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
    # ``job_name`` is user-controllable (TUI free-text Input). ``partition``
    # is enum-controlled today but still quoted for defence-in-depth: if a
    # future Config field plumbs through anything funky the salloc args
    # stay one argv element each, no shell-metachar break-outs.
    cmd = (
        f"salloc --no-shell --partition={shlex.quote(partition or cfg.cpu_partition)} "
        f"--job-name={shlex.quote(job_name)} "
        f"--cpus-per-task={cores} --mem={mem_gb}G --time={shlex.quote(walltime)}"
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
        f"salloc --no-shell --partition={shlex.quote(partition or cfg.gpu_partition)} "
        f"--job-name={shlex.quote(job_name)} "
        f"--gres=gpu:{gpus} --cpus-per-task={cores} --mem={mem_gb}G --time={shlex.quote(walltime)}"
    )
    return ssh.capture(cfg.ssh_host, cmd, check=False, merge_stderr=True)


def list_jobs(cfg: Config) -> str:
    """Return the raw ``squeue`` table for the configured user."""
    cmd = f"squeue -u {shlex.quote(cfg.user)} -o '{SQUEUE_LIST_FORMAT}'"
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
    cmd = f"squeue -u {shlex.quote(cfg.user)} -h -t {shlex.quote(state)} -o '{fmt}'"
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
    cmd = f"squeue -u {shlex.quote(cfg.user)} -h -o '%j'"
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
    out = ssh.capture(cfg.ssh_host, f"squeue -j {shlex.quote(jobid)} -h -o '%N'", check=False)
    return out.strip()


def describe(cfg: Config, jobid: str) -> str:
    """Short human description of a job: ``id partition name state``."""
    return ssh.capture(
        cfg.ssh_host, f"squeue -j {shlex.quote(jobid)} -h -o '%i %P %j %T'", check=False
    )


def cancel(cfg: Config, jobid: str) -> int:
    # ``jobid`` comes from a typer.Argument — quote so ``rci cancel '1; …'``
    # can't smuggle a second command into the remote shell.
    return ssh.run(cfg.ssh_host, f"scancel {shlex.quote(jobid)}", check=False)


def cancel_all(cfg: Config) -> int:
    return ssh.run(cfg.ssh_host, f"scancel -u {shlex.quote(cfg.user)}", check=False)


def cancel_jobids(cfg: Config, jobids: list[str]) -> int:
    """Cancel a batch of job ids in a single ``scancel`` call."""
    if not jobids:
        return 0
    return ssh.run(cfg.ssh_host, "scancel " + " ".join(shlex.quote(j) for j in jobids), check=False)


def list_jobs_brief(cfg: Config) -> str:
    cmd = f"squeue -u {shlex.quote(cfg.user)} -h -o '%.10i %.9P %.10j %.8T'"
    return ssh.capture(cfg.ssh_host, cmd, check=False)


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
