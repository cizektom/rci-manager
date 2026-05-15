"""End-to-end smoke tests via Typer's CliRunner. Mocks ssh/slurm primitives."""

from __future__ import annotations

from typer.testing import CliRunner

from rci_cli import alloc as alloc_mod
from rci_cli import launch, slurm
from rci_cli import ssh as ssh_mod
from rci_cli.alloc import Allocation
from rci_cli.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_no_args_in_non_tty_falls_back_to_help() -> None:
    """Bare ``rci`` opens the TUI only when stdin+stdout are TTYs; CliRunner isn't."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "Commands" in result.stdout


def test_jobs_calls_list_jobs(monkeypatch) -> None:
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "JOBID PARTITION NAME\n1234 cpufast vscode")
    result = runner.invoke(app, ["jobs"])
    assert result.exit_code == 0
    assert "1234" in result.stdout
    assert "vscode" in result.stdout


def test_cancel_with_existing_job(monkeypatch) -> None:
    monkeypatch.setattr(slurm, "describe", lambda cfg, jid: "1234 cpufast vscode R")
    monkeypatch.setattr(slurm, "cancel", lambda cfg, jid: 0)
    result = runner.invoke(app, ["cancel", "1234"])
    assert result.exit_code == 0


def test_cancel_with_missing_job_fails(monkeypatch) -> None:
    monkeypatch.setattr(slurm, "describe", lambda cfg, jid: "")
    result = runner.invoke(app, ["cancel", "9999"])
    assert result.exit_code == 1


def test_cpu_passes_defaults_to_submit(monkeypatch) -> None:
    seen = {}

    def fake_submit(cfg, cores, mem, walltime):
        seen.update({"cores": cores, "mem": mem, "walltime": walltime})
        return "Granted job allocation 7777"

    monkeypatch.setattr(slurm, "submit_cpu", fake_submit)
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    result = runner.invoke(app, ["cpu"])
    assert result.exit_code == 0
    assert seen == {"cores": 2, "mem": 4, "walltime": "1:00:00"}


def test_cpu_flags_override_defaults(monkeypatch) -> None:
    seen = {}

    def fake_submit(cfg, cores, mem, walltime):
        seen.update({"cores": cores, "mem": mem, "walltime": walltime})
        return ""

    monkeypatch.setattr(slurm, "submit_cpu", fake_submit)
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    result = runner.invoke(app, ["cpu", "--cores", "16", "--mem", "64", "--time", "2:00:00"])
    assert result.exit_code == 0
    assert seen == {"cores": 16, "mem": 64, "walltime": "2:00:00"}


def test_gpu_defaults(monkeypatch) -> None:
    seen = {}

    def fake_submit(cfg, gpus, cores, mem, walltime):
        seen.update({"gpus": gpus, "cores": cores, "mem": mem, "walltime": walltime})
        return ""

    monkeypatch.setattr(slurm, "submit_gpu", fake_submit)
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    result = runner.invoke(app, ["gpu"])
    assert result.exit_code == 0
    assert seen == {"gpus": 1, "cores": 2, "mem": 8, "walltime": "1:00:00"}


def test_alloc_prints_node_and_jobid(monkeypatch) -> None:
    monkeypatch.setattr(slurm, "jobs_by_name", lambda cfg, name, state="RUNNING": ["5555"] if name == "vscode-gpu" else [])
    monkeypatch.setattr(slurm, "node_for", lambda cfg, jid: "g05")
    result = runner.invoke(app, ["alloc"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "g05 5555"


def test_port_rejects_garbage() -> None:
    result = runner.invoke(app, ["port", "not-a-number"])
    assert result.exit_code == 1
    assert "Invalid port spec" in result.stdout


def test_shell_with_node_skips_squeue_lookup(monkeypatch) -> None:
    """``--node`` bypasses :func:`alloc.select_or_submit` — no squeue calls allowed."""
    captured: dict = {}

    def fake_launch_shell(a, folder, cfg):
        captured.update({"node": a.node, "jobid": a.jobid, "folder": folder})
        return 0

    def boom(*args, **kwargs):  # any allocation lookup is a bug here
        raise AssertionError("--node should skip allocation lookup")

    monkeypatch.setattr(launch, "launch_shell", fake_launch_shell)
    monkeypatch.setattr(alloc_mod, "select_or_submit", boom)
    result = runner.invoke(app, ["shell", "sam2rl", "--node", "n07"])
    assert result.exit_code == 0
    assert captured == {"node": "n07", "jobid": "", "folder": "/home/cizekto2/sam2rl"}


def test_shell_tab_dispatches_to_in_tab_launcher(monkeypatch) -> None:
    """``--tab`` routes through :func:`launch_shell_in_tab` instead of the inline path."""
    captured: dict = {}

    def fake_in_tab(a, folder_arg, cfg):
        captured.update({"node": a.node, "folder_arg": folder_arg})
        return 0

    def inline_boom(*args, **kwargs):
        raise AssertionError("--tab should not call inline launcher")

    monkeypatch.setattr(launch, "launch_shell_in_tab", fake_in_tab)
    monkeypatch.setattr(launch, "launch_shell", inline_boom)
    result = runner.invoke(app, ["shell", "sam2rl", "--tab", "--node", "n07"])
    assert result.exit_code == 0
    assert captured == {"node": "n07", "folder_arg": "sam2rl"}


def test_claude_tab_dispatches_to_in_tab_launcher(monkeypatch) -> None:
    captured: dict = {}

    def fake_in_tab(a, folder_arg, cfg):
        captured.update({"node": a.node, "folder_arg": folder_arg})
        return 0

    monkeypatch.setattr(launch, "launch_claude_in_tab", fake_in_tab)
    result = runner.invoke(app, ["claude", "--tab", "--node", "g05"])
    assert result.exit_code == 0
    assert captured == {"node": "g05", "folder_arg": ""}
