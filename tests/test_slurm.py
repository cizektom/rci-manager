"""Slurm command-string builders. Mocks ``ssh.capture/run`` to assert what gets sent.

Beyond the happy path, this asserts that user-controllable strings
(``job_name`` from the TUI Input, ``jobid`` from the CLI argument) are
shell-quoted before going through ``ssh host '<cmd>'`` — otherwise a value
like ``foo;rm -rf /`` would smuggle a second command into the remote shell.
"""

from __future__ import annotations

from typing import Any

import pytest

from rci_cli import slurm, ssh
from rci_cli.config import Config


@pytest.fixture
def captured(monkeypatch) -> dict[str, Any]:
    """Capture every ``ssh.capture`` / ``ssh.run`` call."""
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


# ── submit_cpu / submit_gpu command shape ───────────────────────────────────


def test_submit_cpu_command_string(captured, cfg: Config) -> None:
    slurm.submit_cpu(cfg, cores=8, mem_gb=32, walltime="2:00:00", job_name="dev-3")
    cmd = captured["capture"][0]["cmd"]
    assert cmd.startswith("salloc --no-shell ")
    assert "--partition=cpufast" in cmd
    assert "--job-name=dev-3" in cmd
    assert "--cpus-per-task=8" in cmd
    assert "--mem=32G" in cmd
    assert "--time=2:00:00" in cmd


def test_submit_gpu_command_string(captured, cfg: Config) -> None:
    slurm.submit_gpu(cfg, gpus=2, cores=16, mem_gb=64, walltime="8:00:00", job_name="dev-5")
    cmd = captured["capture"][0]["cmd"]
    assert "--partition=gpufast" in cmd
    assert "--job-name=dev-5" in cmd
    assert "--gres=gpu:2" in cmd


# ── shell-injection defence: user-controllable strings get shlex-quoted ────


def test_submit_cpu_shell_quotes_malicious_job_name(captured, cfg: Config) -> None:
    """The TUI's #job-name is a free-text Input — quote the value so a name
    like ``foo;rm -rf /`` can't smuggle a second command into the remote shell."""
    slurm.submit_cpu(cfg, cores=2, mem_gb=4, walltime="1:00:00", job_name="foo;rm -rf /")
    cmd = captured["capture"][0]["cmd"]
    # Quoted form: `--job-name='foo;rm -rf /'`. ``;`` lives inside the quotes
    # so the remote shell never splits on it.
    assert "--job-name='foo;rm -rf /'" in cmd
    assert "rm -rf /" in cmd  # still present — but harmlessly, inside the quote
    # Sanity: the ``;`` is NOT followed by an unquoted ``rm`` after the
    # closing quote, which is what would actually be dangerous.
    assert "; rm" not in cmd


def test_submit_gpu_shell_quotes_partition(captured, cfg: Config) -> None:
    """Same defence for ``partition`` — quoted even though it's enum-derived,
    so a future Config override or a typo in config.toml can't break things."""
    slurm.submit_gpu(
        cfg, gpus=1, cores=2, mem_gb=4, walltime="1:00:00",
        job_name="dev-1", partition="weird name",
    )
    cmd = captured["capture"][0]["cmd"]
    assert "--partition='weird name'" in cmd


# ── squeue/scancel: jobid + user quoting ────────────────────────────────────


def test_list_jobs_uses_friendly_format(captured, cfg: Config) -> None:
    slurm.list_jobs(cfg)
    cmd = captured["capture"][0]["cmd"]
    assert cmd.startswith("squeue -u cizekto2 -o ")
    # The format string padding spec — load-bearing for jobs_by_prefix parsing.
    for spec in ("%.10i", "%.9P", "%.12j", "%.5C", "%.6m", "%.8b"):
        assert spec in cmd, spec


def test_cancel_shell_quotes_jobid(captured, cfg: Config) -> None:
    """``rci cancel`` accepts the jobid as a typer.Argument — user-controllable."""
    slurm.cancel(cfg, "1234")
    assert captured["run"][0]["cmd"] == "scancel 1234"
    captured["run"].clear()

    slurm.cancel(cfg, "12; rm -rf /")
    assert captured["run"][0]["cmd"] == "scancel '12; rm -rf /'"


def test_cancel_all_quotes_user(captured, cfg: Config) -> None:
    slurm.cancel_all(cfg)
    assert captured["run"][0]["cmd"] == "scancel -u cizekto2"


def test_cancel_jobids_batches_and_quotes(captured, cfg: Config) -> None:
    slurm.cancel_jobids(cfg, ["111", "222", "333"])
    assert captured["run"][0]["cmd"] == "scancel 111 222 333"
    captured["run"].clear()

    # Empty list is a no-op (no ssh round-trip).
    slurm.cancel_jobids(cfg, [])
    assert captured["run"] == []


def test_node_for_quotes_jobid(captured, cfg: Config) -> None:
    captured["capture_return"] = "g05\n"
    slurm.node_for(cfg, "5; evil")
    assert "squeue -j '5; evil'" in captured["capture"][0]["cmd"]


# ── jobs_by_prefix parsing ──────────────────────────────────────────────────


def test_jobs_by_prefix_matches_singleton_and_indexed(captured, cfg: Config) -> None:
    """``editor`` matches exactly; ``dev`` matches ``dev``, ``dev-1``, ``dev-2`` —
    but never ``develop``/``devops``."""
    captured["capture_return"] = (
        "111 cpufast dev RUNNING 0:05 1:00:00 N/A n01\n"
        "222 cpufast dev-1 RUNNING 0:01 1:00:00 N/A n02\n"
        "333 gpufast dev-2 RUNNING 0:02 1:00:00 gpu:1 g05\n"
        "444 cpufast develop RUNNING 0:03 1:00:00 N/A n03\n"
        "555 cpufast devops RUNNING 0:03 1:00:00 N/A n03\n"
        "666 cpufast editor RUNNING 0:04 1:00:00 N/A n04\n"
    )
    dev = slurm.jobs_by_prefix(cfg, "dev")
    assert [j.jobid for j in dev] == ["111", "222", "333"]
    assert dev[2].gres == "gpu:1"

    captured["capture_return"] = "666 cpufast editor RUNNING 0:04 1:00:00 N/A n04\n"
    assert [j.jobid for j in slurm.jobs_by_prefix(cfg, "editor")] == ["666"]


def test_jobs_by_prefix_handles_empty_output(captured, cfg: Config) -> None:
    captured["capture_return"] = ""
    assert slurm.jobs_by_prefix(cfg, "dev") == []


@pytest.mark.parametrize(
    "gres,expected",
    [
        ("gpu:1", True),
        ("gpu:tesla:2", True),
        ("N/A", False),
        ("(null)", False),
        ("", False),
    ],
)
def test_has_gpu_distinguishes_gres_strings(gres, expected) -> None:
    j = slurm.Job(jobid="1", partition="x", name="dev", state="RUNNING", gres=gres)
    assert slurm.has_gpu(j) is expected


# ── name allocation ─────────────────────────────────────────────────────────


def test_lowest_unused_index_finds_first_gap() -> None:
    assert slurm.lowest_unused_index([], "dev") == 1
    assert slurm.lowest_unused_index(["dev-1", "dev-2", "dev-3"], "dev") == 4
    # Gap reuse — cancelled "dev-2" becomes available again.
    assert slurm.lowest_unused_index(["dev-1", "dev-3"], "dev") == 2
    # Non-numeric suffixes ignored.
    assert slurm.lowest_unused_index(["dev-foo", "dev-1"], "dev") == 2
    # Unrelated prefixes don't bump the counter.
    assert slurm.lowest_unused_index(["editor", "other-1"], "dev") == 1


def test_next_indexed_name_queries_squeue_and_picks_gap(captured, cfg: Config) -> None:
    captured["capture_return"] = "dev-1\ndev-3\neditor\nunrelated\n"
    assert slurm.next_indexed_name(cfg, "dev") == "dev-2"
    assert captured["capture"][0]["cmd"] == "squeue -u cizekto2 -h -o '%j'"


# ── salloc output helpers (moved from test_tui.py — these are slurm.py logic) ─


def test_parse_jobid_from_salloc_handles_known_shapes() -> None:
    assert slurm.parse_jobid_from_salloc("salloc: Granted job allocation 1234567") == "1234567"
    assert slurm.parse_jobid_from_salloc("salloc: Pending job allocation 9999999") == "9999999"
    assert slurm.parse_jobid_from_salloc(
        "salloc: job 5555 queued and waiting for resources\n"
        "salloc: job 5555 has been allocated resources"
    ) == "5555"
    # No jobid present (salloc errored out) — caller surfaces the message.
    assert slurm.parse_jobid_from_salloc("salloc: error: Invalid partition: foo") is None
    assert slurm.parse_jobid_from_salloc("") is None


def test_last_meaningful_line_returns_final_non_empty_line() -> None:
    assert slurm.last_meaningful_line("a\nb\nc") == "c"
    assert slurm.last_meaningful_line("a\n\n  \nb\n\n") == "b"
    assert slurm.last_meaningful_line("") == "(no output)"
    assert slurm.last_meaningful_line("   ") == "(no output)"
