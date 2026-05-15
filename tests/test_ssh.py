"""Argv shape of the ssh wrappers. Mocks subprocess.run so nothing actually executes."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from rci_cli import ssh


@pytest.fixture
def captured(monkeypatch) -> dict[str, Any]:
    box: dict[str, Any] = {"runs": []}

    def fake_run(argv, **kwargs):
        box["runs"].append({"argv": list(argv), "kwargs": kwargs})
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return box


def test_run_no_command_just_host(captured) -> None:
    ssh.run("rci")
    assert captured["runs"][0]["argv"] == ["ssh", "rci"]


def test_run_with_command(captured) -> None:
    ssh.run("rci", "squeue -u me")
    assert captured["runs"][0]["argv"] == ["ssh", "rci", "squeue -u me"]


def test_run_tty_adds_dash_tt(captured) -> None:
    ssh.run("g05", "exec bash -i", tty=True)
    assert captured["runs"][0]["argv"] == ["ssh", "-tt", "g05", "exec bash -i"]


def test_run_stdin_pipes_input(captured) -> None:
    ssh.run("rci", "bash", stdin="echo hi")
    assert captured["runs"][0]["kwargs"]["input"] == "echo hi"
    assert captured["runs"][0]["kwargs"]["text"] is True


def test_port_forward_argv(captured) -> None:
    ssh.port_forward("g05", 8080, 9000)
    assert captured["runs"][0]["argv"] == ["ssh", "-N", "-L", "8080:localhost:9000", "g05"]


def test_quote_remote_handles_spaces() -> None:
    assert ssh.quote_remote("hello world") == "'hello world'"
    assert ssh.quote_remote("simple") == "simple"
