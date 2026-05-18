"""Argv shape AND stdin handling of the ssh wrappers.

Mocks ``subprocess.run`` so nothing actually executes. Stdin assertions are
the load-bearing ones — see the long incident in
[[feedback-diagnose-dont-speculate]]: when ``ssh.capture`` left stdin
inherited from the parent TTY, the background refresh worker raced the
foreground TUI for keystrokes and ate every other character the user typed.
The original test_ssh.py asserted argv only and silently passed.
"""

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


# ── capture: stdout/stderr round-trip + DEVNULL stdin (regression) ──────────


def test_capture_detaches_stdin_from_parent_tty(captured) -> None:
    """The bug: ``capture_output=True`` only rewires stdout/stderr — without
    an explicit ``stdin``, ssh inherits the TTY and races a parent TUI for
    keystrokes on every refresh tick. ``stdin=DEVNULL`` is the fix."""
    ssh.capture("rci", "squeue -u me")
    kwargs = captured["runs"][0]["kwargs"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs.get("capture_output") is True


def test_capture_returns_stripped_stdout(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="hello\n", stderr=""),
    )
    assert ssh.capture("rci", "echo hi") == "hello"


def test_capture_merge_stderr_concatenates_stderr(monkeypatch) -> None:
    """``salloc`` writes the job-id line to stderr — caller needs both."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, stdout="stdout-line\n", stderr="Granted job allocation 42\n"
        ),
    )
    assert ssh.capture("rci", "salloc …", merge_stderr=True) == (
        "stdout-line\nGranted job allocation 42"
    )


# ── run: argv shape + stdin handling per mode ───────────────────────────────


def test_run_no_command_just_host_inherits_stdin(captured) -> None:
    """``ssh rci`` opens a remote shell — needs the user's stdin to type into."""
    ssh.run("rci")
    call = captured["runs"][0]
    assert call["argv"] == ["ssh", "rci"]
    assert "stdin" not in call["kwargs"]  # inherited


def test_run_tty_inherits_stdin(captured) -> None:
    """``-tt`` is for interactive remote programs (bash, claude) — keep stdin."""
    ssh.run("g05", "exec bash -i", tty=True)
    call = captured["runs"][0]
    assert call["argv"] == ["ssh", "-tt", "g05", "exec bash -i"]
    assert "stdin" not in call["kwargs"]


def test_run_batch_command_detaches_stdin(captured) -> None:
    """Same race class as the ``ssh.capture`` bug: a non-interactive ``ssh
    host cmd`` from a TUI worker thread (e.g. ``scancel``, ``launch_agent``)
    would otherwise inherit the foreground TTY's stdin and steal keys."""
    ssh.run("rci", "scancel 1234")
    call = captured["runs"][0]
    assert call["argv"] == ["ssh", "rci", "scancel 1234"]
    assert call["kwargs"]["stdin"] is subprocess.DEVNULL


def test_run_stdin_pipes_input(captured) -> None:
    """A string ``stdin`` opens a pipe — used to feed multi-line bash scripts."""
    ssh.run("rci", "bash -s", stdin="echo hi")
    kwargs = captured["runs"][0]["kwargs"]
    assert kwargs["input"] == "echo hi"
    assert kwargs["text"] is True


def test_port_forward_argv(captured) -> None:
    ssh.port_forward("g05", 8080, 9000)
    assert captured["runs"][0]["argv"] == ["ssh", "-N", "-L", "8080:localhost:9000", "g05"]
