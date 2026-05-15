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

    def fake_capture(host: str, cmd: str, *, check: bool = True) -> str:
        box["capture"].append({"host": host, "cmd": cmd, "check": check})
        return box.get("capture_return", "")

    def fake_run(host: str, cmd: str = "", *, tty: bool = False, check: bool = True, stdin=None) -> int:
        box["run"].append({"host": host, "cmd": cmd, "tty": tty, "check": check, "stdin": stdin})
        return box.get("run_return", 0)

    monkeypatch.setattr(ssh, "capture", fake_capture)
    monkeypatch.setattr(ssh, "run", fake_run)
    return box


def test_submit_cpu_command_string(captured, cfg: Config) -> None:
    slurm.submit_cpu(cfg, cores=8, mem_gb=32, walltime="2:00:00")
    cmd = captured["capture"][0]["cmd"]
    assert "--partition=cpufast" in cmd
    assert "--job-name=dev" in cmd
    assert "--cpus-per-task=8" in cmd
    assert "--mem=32G" in cmd
    assert "--time=2:00:00" in cmd
    assert "salloc --no-shell" in cmd


def test_submit_gpu_command_string(captured, cfg: Config) -> None:
    slurm.submit_gpu(cfg, gpus=2, cores=16, mem_gb=64, walltime="8:00:00")
    cmd = captured["capture"][0]["cmd"]
    assert "--partition=gpufast" in cmd
    assert "--job-name=dev-gpu" in cmd
    assert "--gres=gpu:2" in cmd
    assert "--cpus-per-task=16" in cmd
    assert "--mem=64G" in cmd
    assert "--time=8:00:00" in cmd


def test_list_jobs_uses_friendly_format(captured, cfg: Config) -> None:
    slurm.list_jobs(cfg)
    cmd = captured["capture"][0]["cmd"]
    assert cmd.startswith("squeue -u cizekto2 -o ")
    # Padded format specifiers — jobid / partition / name / cpus / mem / gres.
    for spec in ("%.10i", "%.9P", "%.12j", "%.5C", "%.6m", "%.8b"):
        assert spec in cmd, spec


def test_jobs_by_name_parses_one_per_line(captured, cfg: Config) -> None:
    captured["capture_return"] = "1111\n2222\n3333\n"
    out = slurm.jobs_by_name(cfg, "dev")
    assert out == ["1111", "2222", "3333"]
    cmd = captured["capture"][0]["cmd"]
    assert "-n dev" in cmd
    assert "-t RUNNING" in cmd


def test_jobs_by_name_handles_empty_output(captured, cfg: Config) -> None:
    captured["capture_return"] = ""
    assert slurm.jobs_by_name(cfg, "dev") == []


def test_cancel_uses_scancel(captured, cfg: Config) -> None:
    slurm.cancel(cfg, "1234")
    assert captured["run"][0]["cmd"] == "scancel 1234"


def test_cancel_all_uses_user_flag(captured, cfg: Config) -> None:
    slurm.cancel_all(cfg)
    assert captured["run"][0]["cmd"] == "scancel -u cizekto2"


def test_cancel_by_names_uses_separate_passes(captured, cfg: Config) -> None:
    slurm.cancel_by_names(cfg, ["dev", "dev-gpu"])
    cmd = captured["run"][0]["cmd"]
    # Slurm scancel doesn't accept comma-separated --name; we issue two passes.
    assert "scancel --name=dev -u cizekto2" in cmd
    assert "scancel --name=dev-gpu -u cizekto2" in cmd
