"""Allocation selection: pick the strongest running vscode allocation, or submit one.

Mirrors the ``_rci_alloc`` shell helper:

- default mode: prefer a running ``vscode-gpu``; else a running ``vscode``;
  else submit a new CPU allocation and wait for it.
- gpu mode: require a running ``vscode-gpu``; submit one if none exists.
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


def _first_running(cfg: Config, name: str) -> str | None:
    ids = slurm.jobs_by_name(cfg, name, state="RUNNING")
    return ids[0] if ids else None


def select_or_submit(cfg: Config, *, require_gpu: bool = False) -> Allocation:
    """Return the allocation to use, submitting one if necessary.

    Submission output is printed to stdout (salloc echoes the new job id),
    so the caller sees the same feedback they'd get from the zsh helpers.
    """
    jobid = _first_running(cfg, cfg.gpu_job_name)

    if jobid is None and require_gpu:
        print(f"No running {cfg.gpu_job_name} allocation — submitting one.")
        gpus, cores, mem, walltime = cfg.gpu_defaults
        print(slurm.submit_gpu(cfg, gpus, cores, mem, walltime))
        jobid = _first_running(cfg, cfg.gpu_job_name)
        if jobid is None:
            raise AllocationError(
                f"Could not find a running {cfg.gpu_job_name} job after submission."
            )
    elif jobid is None:
        jobid = _first_running(cfg, cfg.cpu_job_name)
        if jobid is None:
            print(f"No running {cfg.cpu_job_name} allocation — submitting a CPU one.")
            cores, mem, walltime = cfg.cpu_defaults
            print(slurm.submit_cpu(cfg, cores, mem, walltime))
            jobid = _first_running(cfg, cfg.cpu_job_name)
            if jobid is None:
                raise AllocationError(
                    f"Could not find a running {cfg.cpu_job_name} job after submission."
                )

    node = slurm.node_for(cfg, jobid)
    if not node:
        raise AllocationError(f"Job {jobid} has no node assigned yet.")
    return Allocation(node=node, jobid=jobid)
