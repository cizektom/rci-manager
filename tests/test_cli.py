"""End-to-end smoke tests via Typer's CliRunner. Mocks ssh/slurm primitives."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from rci_cli import alloc as alloc_mod
from rci_cli import cli as cli_mod
from rci_cli import config as config_mod
from rci_cli import launch, setup as setup_mod, slurm
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

    def fake_submit(cfg, cores, mem, walltime, *, job_name, partition=None):
        seen.update({"cores": cores, "mem": mem, "walltime": walltime, "job_name": job_name})
        return "Granted job allocation 7777"

    monkeypatch.setattr(slurm, "submit_cpu", fake_submit)
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-1")
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    result = runner.invoke(app, ["cpu"])
    assert result.exit_code == 0
    assert seen == {"cores": 2, "mem": 4, "walltime": "1:00:00", "job_name": "dev-1"}


def test_cpu_flags_override_defaults(monkeypatch) -> None:
    seen = {}

    def fake_submit(cfg, cores, mem, walltime, *, job_name, partition=None):
        seen.update({"cores": cores, "mem": mem, "walltime": walltime})
        return ""

    monkeypatch.setattr(slurm, "submit_cpu", fake_submit)
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-1")
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    result = runner.invoke(app, ["cpu", "--cores", "16", "--mem", "64", "--time", "2:00:00"])
    assert result.exit_code == 0
    assert seen == {"cores": 16, "mem": 64, "walltime": "2:00:00"}


def test_gpu_defaults(monkeypatch) -> None:
    seen = {}

    def fake_submit(cfg, gpus, cores, mem, walltime, *, job_name, partition=None):
        seen.update({
            "gpus": gpus, "cores": cores, "mem": mem,
            "walltime": walltime, "job_name": job_name,
        })
        return ""

    monkeypatch.setattr(slurm, "submit_gpu", fake_submit)
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-2")
    monkeypatch.setattr(slurm, "list_jobs", lambda cfg: "")
    result = runner.invoke(app, ["gpu"])
    assert result.exit_code == 0
    # Shared dev pool with rci cpu — number comes from next_indexed_name.
    assert seen == {
        "gpus": 1, "cores": 2, "mem": 8,
        "walltime": "1:00:00", "job_name": "dev-2",
    }


def test_alloc_prints_node_and_jobid(monkeypatch) -> None:
    """`rci alloc` reuses a running rci-managed job, prefers GPU."""
    gpu_job = slurm.Job(
        jobid="5555", partition="gpufast", name="dev-1",
        state="RUNNING", gres="gpu:1",
    )
    monkeypatch.setattr(
        slurm, "jobs_by_prefix",
        lambda cfg, prefix, state="RUNNING": [gpu_job] if prefix == "dev" else [],
    )
    monkeypatch.setattr(slurm, "node_for", lambda cfg, jid: "g05")
    result = runner.invoke(app, ["alloc"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "g05 5555"


def test_agent_submits_new_and_launches(monkeypatch) -> None:
    """``rci agent`` always spawns a new ``agent-N`` (no reuse) and runs
    ``launch_agent`` with config-default claude flags when none are passed."""
    submitted: dict = {}
    launched: dict = {}

    def fake_submit_cpu(cfg, cores, mem, walltime, *, job_name, partition=None):
        submitted.update({
            "cores": cores, "mem": mem, "walltime": walltime, "job_name": job_name,
        })
        return "Granted job allocation 7777"

    def fake_launch_agent(a, folder, cfg, *, name, permission_mode, spawn_mode, capacity):
        launched.update({
            "node": a.node, "jobid": a.jobid, "folder": folder,
            "name": name, "permission_mode": permission_mode,
            "spawn_mode": spawn_mode, "capacity": capacity,
        })
        return 0

    monkeypatch.setattr(slurm, "submit_cpu", fake_submit_cpu)
    monkeypatch.setattr(slurm, "submit_gpu", lambda *a, **k: pytest_fail("no gpu"))
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-3")
    monkeypatch.setattr(slurm, "node_for", lambda cfg, jid: "n07")
    monkeypatch.setattr(launch, "launch_agent", fake_launch_agent)

    result = runner.invoke(app, ["agent", "sam2rl"])
    assert result.exit_code == 0, result.stdout
    assert submitted == {"cores": 2, "mem": 4, "walltime": "1:00:00", "job_name": "agent-3"}
    assert launched == {
        "node": "n07", "jobid": "7777", "folder": "/home/cizekto2/sam2rl",
        "name": "agent-3",                # falls back to job name
        "permission_mode": "default",     # cfg default
        "spawn_mode": "same-dir",         # cfg default
        "capacity": 32,                   # cfg default
    }


def test_agent_cli_flags_override_config_defaults(monkeypatch) -> None:
    """Explicit ``--permission-mode`` / ``--spawn`` / ``--capacity`` win over cfg."""
    launched: dict = {}

    monkeypatch.setattr(slurm, "submit_cpu", lambda *a, **k: "Granted job allocation 9")
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-1")
    monkeypatch.setattr(slurm, "node_for", lambda cfg, jid: "n01")
    monkeypatch.setattr(
        launch, "launch_agent",
        lambda a, folder, cfg, *, name, permission_mode, spawn_mode, capacity:
            launched.update({
                "name": name, "permission_mode": permission_mode,
                "spawn_mode": spawn_mode, "capacity": capacity,
            }) or 0,
    )
    result = runner.invoke(app, [
        "agent",
        "--name", "alpha",
        "--permission-mode", "bypassPermissions",
        "--spawn", "worktree",
        "--capacity", "8",
    ])
    assert result.exit_code == 0, result.stdout
    assert launched == {
        "name": "alpha",
        "permission_mode": "bypassPermissions",
        "spawn_mode": "worktree",
        "capacity": 8,
    }


def test_agent_with_gpu_flag_uses_submit_gpu(monkeypatch) -> None:
    submitted: dict = {}

    def fake_submit_gpu(cfg, gpus, cores, mem, walltime, *, job_name, partition=None):
        submitted.update({
            "gpus": gpus, "cores": cores, "mem": mem, "walltime": walltime,
            "job_name": job_name,
        })
        return "Granted job allocation 4242"

    monkeypatch.setattr(slurm, "submit_gpu", fake_submit_gpu)
    monkeypatch.setattr(slurm, "submit_cpu", lambda *a, **k: pytest_fail("no cpu"))
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-1")
    monkeypatch.setattr(slurm, "node_for", lambda cfg, jid: "g09")
    monkeypatch.setattr(
        launch, "launch_agent",
        lambda *a, **kw: 0,
    )

    result = runner.invoke(app, ["agent", "--gpu"])
    assert result.exit_code == 0, result.stdout
    # gpu_defaults = (1, 2, 8, "1:00:00")
    assert submitted == {
        "gpus": 1, "cores": 2, "mem": 8, "walltime": "1:00:00",
        "job_name": "agent-1",
    }


def test_agent_fails_on_unparseable_salloc_output(monkeypatch) -> None:
    monkeypatch.setattr(slurm, "submit_cpu", lambda *a, **k: "salloc: error: Invalid partition: foo")
    monkeypatch.setattr(slurm, "next_indexed_name", lambda cfg, pfx: f"{pfx}-1")
    monkeypatch.setattr(launch, "launch_agent", lambda *a, **k: pytest_fail("must not launch"))
    result = runner.invoke(app, ["agent"])
    assert result.exit_code == 1
    assert "submit failed" in result.stdout


def test_cancel_dev_sweeps_dev_editor_and_agent(monkeypatch) -> None:
    """The cleanup command must include all three rci-managed prefixes."""
    called: list[str] = []

    def fake_jobs_by_prefix(cfg, prefix, *, state="RUNNING"):
        called.append(prefix)
        return [
            slurm.Job(
                jobid=f"{prefix}1", partition="x", name=f"{prefix}-1",
                state="RUNNING", gres="N/A",
            )
        ]

    cancelled: list[list[str]] = []
    monkeypatch.setattr(slurm, "jobs_by_prefix", fake_jobs_by_prefix)
    monkeypatch.setattr(
        slurm, "cancel_jobids",
        lambda cfg, ids: cancelled.append(list(ids)) or 0,
    )
    # ``_confirm`` short-circuits to False in non-TTY (CliRunner) — bypass
    # it so the cancellation actually runs.
    monkeypatch.setattr(cli_mod, "_confirm", lambda prompt: True)

    result = runner.invoke(app, ["cancel-dev"])
    assert result.exit_code == 0, result.stdout
    assert called == ["dev", "editor", "agent"]
    assert cancelled == [["dev1", "editor1", "agent1"]]


def pytest_fail(msg: str):  # tiny shim used in monkeypatched fakes
    raise AssertionError(msg)


def test_port_rejects_garbage() -> None:
    result = runner.invoke(app, ["port", "not-a-number"])
    assert result.exit_code == 1
    assert "Invalid port spec" in result.stdout


def test_shell_uses_strongest_alloc(monkeypatch) -> None:
    captured: dict = {}

    def fake_launch_shell(a, folder, cfg):
        captured.update({"node": a.node, "folder": folder})
        return 0

    monkeypatch.setattr(launch, "launch_shell", fake_launch_shell)
    monkeypatch.setattr(
        alloc_mod, "select_or_submit",
        lambda cfg, **kw: alloc_mod.Allocation(node="g05", jobid="9999"),
    )
    result = runner.invoke(app, ["shell", "sam2rl"])
    assert result.exit_code == 0
    assert captured == {"node": "g05", "folder": "/home/cizekto2/sam2rl"}


def test_editor_runs_for_known_alloc(monkeypatch) -> None:
    captured: dict = {}

    def fake_launch_editor(a, folder, cfg):
        captured.update({"node": a.node, "folder": folder})
        return 0

    monkeypatch.setattr(launch, "launch_editor", fake_launch_editor)
    monkeypatch.setattr(
        alloc_mod, "select_or_submit",
        lambda cfg, **kw: alloc_mod.Allocation(node="g05", jobid="9999"),
    )
    result = runner.invoke(app, ["editor", "sam2rl"])
    assert result.exit_code == 0
    assert captured == {"node": "g05", "folder": "/home/cizekto2/sam2rl"}


# ── first-run setup gating ──────────────────────────────────────────────────


def test_subcommand_exits_with_setup_hint_when_unconfigured(
    monkeypatch, tmp_path: Path
) -> None:
    """Any cluster-touching command must refuse to run with an empty config and
    point the user at ``rci setup`` instead of crashing on ``squeue -u ''``."""
    # Override the autouse XDG isolation: empty config dir ⇒ needs_setup() == True.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    result = runner.invoke(app, ["jobs"])
    assert result.exit_code == 2
    assert "rci setup" in result.stdout


def test_setup_subcommand_writes_config(monkeypatch, tmp_path: Path) -> None:
    """``rci setup`` runs the wizard non-interactively here (typer.prompt
    stubbed), and the resulting TOML file lands under XDG_CONFIG_HOME."""
    cfg_root = tmp_path / "fresh"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_root))

    answers = iter(["alice", "rci", "/home/alice"])
    monkeypatch.setattr(setup_mod.typer, "prompt", lambda *a, **kw: next(answers))

    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0, result.stdout

    written = cfg_root / "rci-cli" / "config.toml"
    assert written.exists()
    loaded = config_mod.load()
    assert loaded.user == "alice"
    assert loaded.home == "/home/alice"


def test_version_works_without_setup(monkeypatch, tmp_path: Path) -> None:
    """``version`` shouldn't be gated on the wizard — it's a pure stdout call."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
