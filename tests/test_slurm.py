"""Slurm command-string builders. Mocks ``ssh.capture/run`` to assert what gets sent."""

from __future__ import annotations

from typing import Any

import pytest

from rci_cli import slurm, ssh
from rci_cli.config import Config


@pytest.fixture
def captured(monkeypatch) -> dict[str, Any]:
    """Capture the most recent ``ssh.capture`` / ``ssh.run`` invocation."""
    box: dict[str, Any] = {"capture": [], "run": []}

    def fake_capture(host: str, cmd: str, **kw) -> str:
        box["capture"].append({"host": host, "cmd": cmd, **kw})
        return box.get("capture_return", "")

    def fake_run(host: str, cmd: str = "", *, tty: bool = False, check: bool = True, stdin=None) -> int:
        box["run"].append({"host": host, "cmd": cmd, "tty": tty, "check": check, "stdin": stdin})
        return box.get("run_return", 0)

    monkeypatch.setattr(ssh, "capture", fake_capture)
    monkeypatch.setattr(ssh, "run", fake_run)
    return box


def test_submit_cpu_command_string(captured, cfg: Config) -> None:
    slurm.submit_cpu(cfg, cores=8, mem_gb=32, walltime="2:00:00", job_name="dev-3")
    cmd = captured["capture"][0]["cmd"]
    assert "--partition=cpufast" in cmd
    assert "--job-name=dev-3" in cmd
    assert "--cpus-per-task=8" in cmd
    assert "--mem=32G" in cmd
    assert "--time=2:00:00" in cmd
    assert "salloc --no-shell" in cmd


def test_submit_gpu_command_string(captured, cfg: Config) -> None:
    slurm.submit_gpu(cfg, gpus=2, cores=16, mem_gb=64, walltime="8:00:00", job_name="dev-5")
    cmd = captured["capture"][0]["cmd"]
    assert "--partition=gpufast" in cmd
    assert "--job-name=dev-5" in cmd
    assert "--gres=gpu:2" in cmd
    assert "--cpus-per-task=16" in cmd
    assert "--mem=64G" in cmd
    assert "--time=8:00:00" in cmd


def test_submit_cpu_accepts_singleton_editor_name(captured, cfg: Config) -> None:
    """The Editor flow spawns a fixed ``editor`` name (no numbering)."""
    slurm.submit_cpu(cfg, cores=2, mem_gb=4, walltime="1:00:00", job_name="editor")
    cmd = captured["capture"][0]["cmd"]
    assert "--job-name=editor" in cmd


def test_list_jobs_uses_friendly_format(captured, cfg: Config) -> None:
    slurm.list_jobs(cfg)
    cmd = captured["capture"][0]["cmd"]
    assert cmd.startswith("squeue -u cizekto2 -o ")
    # Padded format specifiers — jobid / partition / name / cpus / mem / gres.
    for spec in ("%.10i", "%.9P", "%.12j", "%.5C", "%.6m", "%.8b"):
        assert spec in cmd, spec


def test_jobs_by_prefix_matches_singleton_and_indexed(captured, cfg: Config) -> None:
    """``editor`` matches exactly; ``dev`` matches ``dev``, ``dev-1``, ``dev-2``."""
    captured["capture_return"] = (
        "111 cpufast dev RUNNING 0:05 1:00:00 N/A n01\n"
        "222 cpufast dev-1 RUNNING 0:01 1:00:00 N/A n02\n"
        "333 gpufast dev-2 RUNNING 0:02 1:00:00 gpu:1 g05\n"
        "444 cpufast other RUNNING 0:03 1:00:00 N/A n03\n"
        "555 cpufast editor RUNNING 0:04 1:00:00 N/A n04\n"
    )
    dev = slurm.jobs_by_prefix(cfg, "dev")
    assert [j.jobid for j in dev] == ["111", "222", "333"]
    assert [j.name for j in dev] == ["dev", "dev-1", "dev-2"]
    # gres is captured so callers can detect GPU jobs.
    assert dev[2].gres == "gpu:1"

    cmd = captured["capture"][0]["cmd"]
    assert "-u cizekto2" in cmd
    assert "-t RUNNING" in cmd

    captured["capture_return"] = "555 cpufast editor RUNNING 0:04 1:00:00 N/A n04\n"
    ed = slurm.jobs_by_prefix(cfg, "editor")
    assert [j.jobid for j in ed] == ["555"]


def test_jobs_by_prefix_handles_empty_output(captured, cfg: Config) -> None:
    captured["capture_return"] = ""
    assert slurm.jobs_by_prefix(cfg, "dev") == []


def test_jobs_by_prefix_does_not_match_unrelated_prefix(captured, cfg: Config) -> None:
    """``dev`` must not match ``develop`` or ``devops`` — only exact + ``dev-…``."""
    captured["capture_return"] = (
        "111 cpufast develop RUNNING 0:05 1:00:00 N/A n01\n"
        "222 cpufast devops RUNNING 0:01 1:00:00 N/A n02\n"
    )
    assert slurm.jobs_by_prefix(cfg, "dev") == []


def test_has_gpu_distinguishes_gres_strings() -> None:
    def j(gres: str) -> slurm.Job:
        return slurm.Job(jobid="1", partition="x", name="dev", state="RUNNING", gres=gres)

    assert slurm.has_gpu(j("gpu:1"))
    assert slurm.has_gpu(j("gpu:tesla:2"))
    assert not slurm.has_gpu(j("N/A"))
    assert not slurm.has_gpu(j("(null)"))
    assert not slurm.has_gpu(j(""))


def test_lowest_unused_index_finds_first_gap() -> None:
    assert slurm.lowest_unused_index([], "dev") == 1
    assert slurm.lowest_unused_index(["dev-1", "dev-2", "dev-3"], "dev") == 4
    # gap reuse — cancelled "dev-2" → next spawn picks 2, not 4.
    assert slurm.lowest_unused_index(["dev-1", "dev-3"], "dev") == 2
    # Non-numeric suffixes ignored.
    assert slurm.lowest_unused_index(["dev-foo", "dev-1"], "dev") == 2
    # Other prefixes ignored.
    assert slurm.lowest_unused_index(["editor", "other-1"], "dev") == 1


def test_next_indexed_name_queries_squeue_and_picks_gap(captured, cfg: Config) -> None:
    captured["capture_return"] = "dev-1\ndev-3\neditor\nunrelated\n"
    assert slurm.next_indexed_name(cfg, "dev") == "dev-2"
    cmd = captured["capture"][0]["cmd"]
    assert cmd == "squeue -u cizekto2 -h -o '%j'"


def test_cancel_uses_scancel(captured, cfg: Config) -> None:
    slurm.cancel(cfg, "1234")
    assert captured["run"][0]["cmd"] == "scancel 1234"


def test_cancel_all_uses_user_flag(captured, cfg: Config) -> None:
    slurm.cancel_all(cfg)
    assert captured["run"][0]["cmd"] == "scancel -u cizekto2"


def test_cancel_jobids_batches_into_single_scancel(captured, cfg: Config) -> None:
    slurm.cancel_jobids(cfg, ["111", "222", "333"])
    assert captured["run"][0]["cmd"] == "scancel 111 222 333"


def test_cancel_jobids_with_empty_list_is_noop(captured, cfg: Config) -> None:
    slurm.cancel_jobids(cfg, [])
    assert captured["run"] == []
