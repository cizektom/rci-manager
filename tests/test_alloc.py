"""Allocation selection — reuse a running dev-*/editor or spawn a new one."""

from __future__ import annotations

import pytest

from rci_cli import slurm
from rci_cli.alloc import Allocation, AllocationError, select_or_submit
from rci_cli.config import Config


def _job(jobid: str, name: str, *, gres: str = "N/A") -> slurm.Job:
    return slurm.Job(
        jobid=jobid, partition="x", name=name, state="RUNNING", gres=gres
    )


def _patch_running(monkeypatch, by_prefix: dict[str, list[slurm.Job]]) -> None:
    """``jobs_by_prefix(cfg, p)`` → ``by_prefix[p]`` (default empty)."""

    def fake(cfg: Config, prefix: str, *, state: str = "RUNNING") -> list[slurm.Job]:
        return list(by_prefix.get(prefix, []))

    monkeypatch.setattr(slurm, "jobs_by_prefix", fake)


def _patch_node(monkeypatch, mapping: dict[str, str]) -> None:
    monkeypatch.setattr(slurm, "node_for", lambda cfg, jid: mapping.get(jid, ""))


# ── reuse paths ─────────────────────────────────────────────────────────────


def test_reuses_existing_gpu_dev_job(monkeypatch, cfg: Config) -> None:
    _patch_running(monkeypatch, {"dev": [_job("2222", "dev-1", gres="gpu:1")]})
    _patch_node(monkeypatch, {"2222": "g05"})
    monkeypatch.setattr(slurm, "submit_cpu", lambda *a, **k: pytest.fail("no submit"))
    monkeypatch.setattr(slurm, "submit_gpu", lambda *a, **k: pytest.fail("no submit"))
    assert select_or_submit(cfg) == Allocation(node="g05", jobid="2222")


def test_reuses_existing_cpu_when_not_requiring_gpu(monkeypatch, cfg: Config) -> None:
    _patch_running(monkeypatch, {"dev": [_job("1111", "dev-1")]})
    _patch_node(monkeypatch, {"1111": "n01"})
    monkeypatch.setattr(slurm, "submit_cpu", lambda *a, **k: pytest.fail("no submit"))
    assert select_or_submit(cfg, require_gpu=False) == Allocation(
        node="n01", jobid="1111"
    )


def test_prefers_gpu_over_cpu_when_both_running(monkeypatch, cfg: Config) -> None:
    """A GPU allocation is stronger; reuse it before any CPU dev when no --gpu."""
    _patch_running(
        monkeypatch,
        {
            "dev": [
                _job("1111", "dev-1"),
                _job("2222", "dev-2", gres="gpu:1"),
            ]
        },
    )
    _patch_node(monkeypatch, {"1111": "n01", "2222": "g05"})
    assert select_or_submit(cfg) == Allocation(node="g05", jobid="2222")


def test_reuses_existing_editor(monkeypatch, cfg: Config) -> None:
    """Singleton ``editor`` is just another running rci-managed job — reuse it."""
    _patch_running(monkeypatch, {"editor": [_job("3333", "editor")]})
    _patch_node(monkeypatch, {"3333": "n07"})
    assert select_or_submit(cfg) == Allocation(node="n07", jobid="3333")


# ── spawn paths ─────────────────────────────────────────────────────────────


def test_submits_cpu_when_nothing_running(monkeypatch, cfg: Config) -> None:
    state = {"submitted": False, "name": ""}
    store: dict[str, list[slurm.Job]] = {}

    def fake_jobs(cfg, prefix, *, state="RUNNING"):
        return list(store.get(prefix, []))

    def fake_submit_cpu(_cfg, cores, mem, walltime, *, job_name, partition=None):
        state["submitted"] = True
        state["name"] = job_name
        store["dev"] = [_job("3333", job_name)]
        return f"Granted job allocation 3333"

    monkeypatch.setattr(slurm, "jobs_by_prefix", fake_jobs)
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-1")
    monkeypatch.setattr(slurm, "submit_cpu", fake_submit_cpu)
    _patch_node(monkeypatch, {"3333": "n07"})

    alloc = select_or_submit(cfg)
    assert state == {"submitted": True, "name": "dev-1"}
    assert alloc == Allocation(node="n07", jobid="3333")


def test_require_gpu_submits_gpu_when_missing(monkeypatch, cfg: Config) -> None:
    state = {"submitted": False, "name": ""}
    store: dict[str, list[slurm.Job]] = {}

    def fake_jobs(cfg, prefix, *, state="RUNNING"):
        return list(store.get(prefix, []))

    def fake_submit_gpu(_cfg, gpus, cores, mem, walltime, *, job_name, partition=None):
        state["submitted"] = True
        state["name"] = job_name
        store["dev"] = [_job("4444", job_name, gres="gpu:1")]
        return "Granted job allocation 4444"

    monkeypatch.setattr(slurm, "jobs_by_prefix", fake_jobs)
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-1")
    monkeypatch.setattr(slurm, "submit_gpu", fake_submit_gpu)
    monkeypatch.setattr(
        slurm, "submit_cpu", lambda *a, **k: pytest.fail("should not submit CPU")
    )
    _patch_node(monkeypatch, {"4444": "g09"})

    alloc = select_or_submit(cfg, require_gpu=True)
    assert state == {"submitted": True, "name": "dev-1"}
    assert alloc == Allocation(node="g09", jobid="4444")


def test_require_gpu_skips_cpu_dev_and_spawns(monkeypatch, cfg: Config) -> None:
    """A running CPU dev-1 is no good for --gpu — spawn a fresh GPU allocation."""
    state = {"submitted": False}
    store: dict[str, list[slurm.Job]] = {"dev": [_job("1111", "dev-1")]}

    def fake_jobs(cfg, prefix, *, state="RUNNING"):
        return list(store.get(prefix, []))

    def fake_submit_gpu(_cfg, gpus, cores, mem, walltime, *, job_name, partition=None):
        state["submitted"] = True
        store["dev"].append(_job("4444", job_name, gres="gpu:1"))
        return "Granted job allocation 4444"

    monkeypatch.setattr(slurm, "jobs_by_prefix", fake_jobs)
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-2")
    monkeypatch.setattr(slurm, "submit_gpu", fake_submit_gpu)
    _patch_node(monkeypatch, {"1111": "n01", "4444": "g09"})

    alloc = select_or_submit(cfg, require_gpu=True)
    assert state["submitted"] is True
    assert alloc.jobid == "4444"


def test_spawn_name_override_is_honored(monkeypatch, cfg: Config) -> None:
    """``rci editor`` passes ``spawn_name="editor"`` so the singleton stays singleton."""
    state = {"name": ""}
    store: dict[str, list[slurm.Job]] = {}

    def fake_jobs(cfg, prefix, *, state="RUNNING"):
        return list(store.get(prefix, []))

    def fake_submit_cpu(_cfg, cores, mem, walltime, *, job_name, partition=None):
        state["name"] = job_name
        store["editor"] = [_job("5555", job_name)]
        return "Granted job allocation 5555"

    monkeypatch.setattr(slurm, "jobs_by_prefix", fake_jobs)
    monkeypatch.setattr(
        slurm,
        "next_indexed_name",
        lambda cfg, pfx: pytest.fail("should not auto-name when spawn_name given"),
    )
    monkeypatch.setattr(slurm, "submit_cpu", fake_submit_cpu)
    _patch_node(monkeypatch, {"5555": "n07"})

    alloc = select_or_submit(cfg, spawn_name="editor")
    assert state["name"] == "editor"
    assert alloc == Allocation(node="n07", jobid="5555")


def test_raises_when_submission_doesnt_take(monkeypatch, cfg: Config) -> None:
    _patch_running(monkeypatch, {})  # nothing before, nothing after
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-1")
    monkeypatch.setattr(slurm, "submit_cpu", lambda *a, **k: "")
    _patch_node(monkeypatch, {})
    with pytest.raises(AllocationError):
        select_or_submit(cfg)
