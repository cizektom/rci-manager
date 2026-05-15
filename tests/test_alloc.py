"""Allocation selection — the GPU > CPU > submit logic that mirrors ``_rci_alloc``."""

from __future__ import annotations

import pytest

from rci_cli import alloc as alloc_mod
from rci_cli import slurm
from rci_cli.alloc import Allocation, AllocationError, find_strongest, select_or_submit
from rci_cli.config import Config


def _patch_jobs(monkeypatch, by_name: dict[str, list[str]]) -> None:
    """Make ``slurm.jobs_by_name(cfg, name)`` return ``by_name[name]``."""

    def fake(cfg: Config, name: str, *, state: str = "RUNNING") -> list[str]:
        return list(by_name.get(name, []))

    monkeypatch.setattr(slurm, "jobs_by_name", fake)


def _patch_node(monkeypatch, mapping: dict[str, str]) -> None:
    monkeypatch.setattr(slurm, "node_for", lambda cfg, jobid: mapping.get(jobid, ""))


# ── find_strongest ──────────────────────────────────────────────────────────


def test_find_strongest_returns_gpu_when_present(monkeypatch, cfg: Config) -> None:
    _patch_jobs(monkeypatch, {"vscode-gpu": ["2222"], "vscode": ["1111"]})
    _patch_node(monkeypatch, {"2222": "g05"})
    assert find_strongest(cfg) == Allocation(node="g05", jobid="2222")


def test_find_strongest_falls_back_to_cpu_when_no_gpu(monkeypatch, cfg: Config) -> None:
    _patch_jobs(monkeypatch, {"vscode": ["1111"]})
    _patch_node(monkeypatch, {"1111": "n01"})
    assert find_strongest(cfg) == Allocation(node="n01", jobid="1111")


def test_find_strongest_returns_none_when_no_jobs(monkeypatch, cfg: Config) -> None:
    _patch_jobs(monkeypatch, {})
    _patch_node(monkeypatch, {})
    assert find_strongest(cfg) is None


def test_find_strongest_returns_none_when_node_not_assigned(monkeypatch, cfg: Config) -> None:
    _patch_jobs(monkeypatch, {"vscode": ["1111"]})
    _patch_node(monkeypatch, {})  # node_for returns ""
    assert find_strongest(cfg) is None


# ── select_or_submit ────────────────────────────────────────────────────────


def test_select_or_submit_reuses_existing_gpu(monkeypatch, cfg: Config) -> None:
    _patch_jobs(monkeypatch, {"vscode-gpu": ["2222"]})
    _patch_node(monkeypatch, {"2222": "g05"})
    monkeypatch.setattr(slurm, "submit_cpu", lambda *a, **k: pytest.fail("should not submit"))
    monkeypatch.setattr(slurm, "submit_gpu", lambda *a, **k: pytest.fail("should not submit"))
    assert select_or_submit(cfg) == Allocation(node="g05", jobid="2222")


def test_select_or_submit_reuses_existing_cpu_when_not_requiring_gpu(
    monkeypatch, cfg: Config
) -> None:
    _patch_jobs(monkeypatch, {"vscode": ["1111"]})
    _patch_node(monkeypatch, {"1111": "n01"})
    monkeypatch.setattr(slurm, "submit_cpu", lambda *a, **k: pytest.fail("should not submit"))
    assert select_or_submit(cfg, require_gpu=False) == Allocation(node="n01", jobid="1111")


def test_select_or_submit_submits_cpu_when_nothing_running(monkeypatch, cfg: Config) -> None:
    state = {"submitted": False}
    job_state: dict[str, list[str]] = {}

    def fake_submit_cpu(_cfg, cores, mem, walltime):
        # After submit, a running vscode job appears.
        state["submitted"] = True
        job_state["vscode"] = ["3333"]
        return "Granted job allocation 3333"

    _patch_jobs_dynamic(monkeypatch, job_state)
    _patch_node(monkeypatch, {"3333": "n07"})
    monkeypatch.setattr(slurm, "submit_cpu", fake_submit_cpu)

    alloc = select_or_submit(cfg)
    assert state["submitted"] is True
    assert alloc == Allocation(node="n07", jobid="3333")


def test_select_or_submit_require_gpu_submits_gpu_when_missing(
    monkeypatch, cfg: Config
) -> None:
    state = {"submitted": False}
    job_state: dict[str, list[str]] = {}

    def fake_submit_gpu(_cfg, gpus, cores, mem, walltime):
        state["submitted"] = True
        job_state["vscode-gpu"] = ["4444"]
        return "Granted job allocation 4444"

    _patch_jobs_dynamic(monkeypatch, job_state)
    _patch_node(monkeypatch, {"4444": "g09"})
    monkeypatch.setattr(slurm, "submit_gpu", fake_submit_gpu)
    monkeypatch.setattr(slurm, "submit_cpu", lambda *a, **k: pytest.fail("should not submit CPU"))

    alloc = select_or_submit(cfg, require_gpu=True)
    assert state["submitted"] is True
    assert alloc == Allocation(node="g09", jobid="4444")


def test_select_or_submit_raises_when_submission_doesnt_take(monkeypatch, cfg: Config) -> None:
    _patch_jobs(monkeypatch, {})  # job never appears
    _patch_node(monkeypatch, {})
    monkeypatch.setattr(slurm, "submit_cpu", lambda *a, **k: "")
    with pytest.raises(AllocationError):
        select_or_submit(cfg)


# ── helper: dynamic jobs_by_name backed by a mutable dict ───────────────────


def _patch_jobs_dynamic(monkeypatch, store: dict[str, list[str]]) -> None:
    """Like ``_patch_jobs`` but the mapping can mutate between calls (post-submit)."""

    def fake(cfg: Config, name: str, *, state: str = "RUNNING") -> list[str]:
        return list(store.get(name, []))

    monkeypatch.setattr(slurm, "jobs_by_name", fake)
