"""Allocation selection: reuse a running rci-managed job, or submit one.

An rci-managed job is anything whose name matches ``dev`` / ``dev-N`` / ``editor``
(per :attr:`Config.dev_job_name` / :attr:`Config.editor_job_name`). When
:func:`select_or_submit` has to spawn a new allocation, the default name is
``dev-N`` (lowest unused N); callers can override with ``spawn_name=`` —
e.g. the Editor CLI passes ``"editor"`` so the singleton stays singleton.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import slurm
from .config import Config


@dataclass(frozen=True)
class Allocation:
    node: str
    jobid: str


class AllocationError(RuntimeError):
    pass


def _running_managed(cfg: Config) -> list[slurm.Job]:
    """Every running rci-managed allocation, both dev-N and editor."""
    return slurm.jobs_by_prefix(cfg, cfg.dev_job_name) + slurm.jobs_by_prefix(
        cfg, cfg.editor_job_name
    )


def _pick(jobs: list[slurm.Job], *, gpu: bool | None) -> slurm.Job | None:
    """Pick the first job matching the GPU filter. ``None`` ⇒ either is fine."""
    for j in jobs:
        if gpu is None or slurm.has_gpu(j) == gpu:
            return j
    return None


def select_or_submit(
    cfg: Config,
    *,
    require_gpu: bool = False,
    spawn_name: str | None = None,
) -> Allocation:
    """Return the allocation to use, submitting one if necessary.

    Reuse preference: prefer a GPU allocation when one exists (it's the
    strongest), otherwise take any running dev-N / editor. ``require_gpu``
    narrows reuse to GPU jobs only and submits a GPU allocation when missing.

    ``spawn_name`` overrides the default ``dev-N`` for the spawn case — used
    by ``rci editor`` so a fresh editor allocation is named ``editor``.

    Submission output is printed so the caller sees salloc's job-id line.
    """
    running = _running_managed(cfg)

    if require_gpu:
        pick = _pick(running, gpu=True)
    else:
        # Prefer a GPU job if one is running — it's the strongest resource,
        # so reusing it avoids spawning a second allocation. Fall back to any.
        pick = _pick(running, gpu=True) or _pick(running, gpu=None)

    if pick is None:
        name = spawn_name or slurm.next_indexed_name(cfg, cfg.dev_job_name)
        if require_gpu:
            print(f"No running GPU allocation — submitting {name}.")
            gpus, cores, mem, walltime = cfg.gpu_defaults
            print(slurm.submit_gpu(cfg, gpus, cores, mem, walltime, job_name=name))
        else:
            print(f"No running rci allocation — submitting CPU {name}.")
            cores, mem, walltime = cfg.cpu_defaults
            print(slurm.submit_cpu(cfg, cores, mem, walltime, job_name=name))
        # Re-query to find the newly submitted job by name.
        new_jobs = [
            j for j in _running_managed(cfg) if j.name == name
        ]
        if not new_jobs:
            raise AllocationError(
                f"Could not find a running {name} job after submission."
            )
        pick = new_jobs[0]

    node = slurm.node_for(cfg, pick.jobid)
    if not node:
        raise AllocationError(f"Job {pick.jobid} has no node assigned yet.")
    return Allocation(node=node, jobid=pick.jobid)
