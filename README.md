# rci-cli

CLI + interactive TUI for the **RCI CVUT Slurm cluster**. Drop-in replacement
for the `rci-*` zsh helpers in [zsh-setup](https://github.com/cizektom/zsh-setup),
with a path toward richer capabilities (live job dashboard, log tailing,
allocation management, batch script editing) in a Textual TUI.

```sh
git clone git@github.com:cizektom/rci-cli.git ~/rci-cli
uv tool install --python 3.11 ~/rci-cli   # or: pipx install ~/rci-cli
rci --help
```

The binary is named `rci`. If you previously used the
[zsh-setup](https://github.com/cizektom/zsh-setup) `alias rci='ssh rci'`,
remove it — `rci ssh` now does the same thing (and the alias would otherwise
shadow this binary).

---

## Prerequisites

1. **SSH config** with `Host rci` (and `Host n* g*` via `ProxyJump rci`) — see
   [zsh-setup/ssh/rci.conf](https://github.com/cizektom/zsh-setup/blob/main/ssh/rci.conf).
2. **Python ≥ 3.11**.
3. **`pipx`** or any `pip` install method (`pip install -e .` in a venv also works).

---

## Commands

| command                  | replaces zsh helper       | notes                                              |
| ------------------------ | ------------------------- | -------------------------------------------------- |
| `rci ssh`                          | `rci` (zsh alias `ssh rci`) | interactive shell on the login host        |
| `rci jobs`                         | `rci-list`               | `squeue -u $USER` with the friendly format    |
| `rci cpu`                          | `rci-cpu`                | `--cores N --mem GB --time HH:MM:SS`          |
| `rci gpu`                          | `rci-gpu`                | `--gpus N --cores N --mem GB --time HH:MM:SS` |
| `rci cancel JOBID`                 | `rci-cancel`             |                                               |
| `rci cancel-all`                   | `rci-cancel-all`         | confirms first                                |
| `rci cancel-vscode`                | `rci-cancel-vscode`      | confirms first                                |
| `rci claude [DIR] [SUFFIX] [--gpu]` | `rci-claude` / `-gpu`   | persistent zellij session, claude auto-starts |
| `rci shell [DIR] [SUFFIX] [--gpu]`  | `rci-shell` / `-gpu`    | persistent zellij session, plain bash         |
| `rci code [DIR] [--gpu]`           | `rci-code` / `-gpu`      | WSL → Windows `code.cmd` via `cmd.exe`        |
| `rci alloc [--gpu]`                | `_rci_alloc`             | prints `<node> <jobid>` — scripting-friendly  |
| `rci install-zellij`               | `rci-install-zellij`     | drop static zellij + claude layout into `~/bin` |
| `rci sessions`                     | `rci-sessions`           | list zellij sessions on existing allocation   |
| `rci kill-session NAME` / `--all`  | `rci-kill-session`       | kill zellij session(s) without attaching      |
| `rci port LOCAL[:REMOTE]`          | `rci-port`               | local → compute-node port forward (Ctrl-C)    |
| `rci tui`                          | *(new)*                  | Textual TUI — live jobs dashboard (skeleton)  |
| `rci version`                      | -                        |                                               |

**Folder argument rules** (applies to `claude`, `code`, `shell`):

- omitted → `/home/cizekto2`
- relative → resolved under `/home/cizekto2` (`rci claude sam2rl` → `/home/cizekto2/sam2rl`)
- absolute → used as-is

**Zellij sessions** (`claude` and `shell`): each launch is wrapped in a named
zellij session — `<prefix>-<basename-of-folder>[-<suffix>]`:

- `rci claude` → folder `~`, session `claude-home`
- `rci claude sam2rl` → folder `~/sam2rl`, session `claude-sam2rl`
- `rci claude sam2rl exp1` → folder `~/sam2rl`, session `claude-sam2rl-exp1` (parallel)

ssh disconnects don't kill the session; re-running the same command attaches
back to the running session. Run `rci install-zellij` once per cluster
account to get the binary onto compute nodes (it's a static musl build that
works without library hassles).

---

## Configuration

Defaults match the existing zsh helpers. Override any field in
`~/.config/rci-cli/config.toml`:

```toml
user = "cizekto2"
ssh_host = "rci"
home = "/home/cizekto2"
cpu_partition = "cpufast"
gpu_partition = "gpufast"
cpu_job_name = "vscode"
gpu_job_name = "vscode-gpu"
cpu_defaults = [2, 4, "1:00:00"]             # cores, memGB, walltime (conservative debug-friendly)
gpu_defaults = [1, 2, 8, "1:00:00"]          # gpus, cores, memGB, walltime
venv_activate = "$HOME/sam2rl/.venv/bin/activate"
```

Anything you leave out keeps its default.

---

## Why the `*fast` partitions?

The cluster rejects interactive `salloc` (anything `--no-shell` counts) outside
`cpufast` / `gpufast`. The CLI hard-defaults to those — don't change the
partition unless you've also switched from `salloc` to `sbatch`.

---

## Architecture

Single Python package, `src/` layout:

```
src/rci_cli/
├── cli.py        # Typer app — subcommand routing
├── config.py     # Defaults + TOML overrides
├── ssh.py        # ssh wrappers (capture/run/run_local)
├── slurm.py      # squeue / salloc / scancel primitives
├── alloc.py      # select_or_submit() — vscode-gpu > vscode > submit CPU
├── launch.py     # claude / code / shell launchers on compute nodes
└── tui.py        # Textual app (skeleton)
```

Everything that touches Slurm goes through `slurm.py`. The CLI and TUI both
consume the same primitives, so anything you can do in one will (eventually)
work in the other.

---

## Roadmap

- [x] 1:1 parity with the existing zsh helpers.
- [x] TUI skeleton with live jobs dashboard.
- [ ] In-TUI cancel / submit / attach actions.
- [ ] Log tailing for running jobs (`sattach` or remote `tail -F`).
- [ ] GPU/CPU utilization snapshot on the active node.
- [ ] Saved allocation profiles (`rci profile use ml-train`).
- [ ] Batch script editor (`sbatch` flow alongside the salloc flow).
- [ ] Optional non-RCI Slurm clusters via config profile switching.

---

## Development

```sh
git clone git@github.com:cizektom/rci-cli.git ~/rci-cli
cd ~/rci-cli
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
rci --help
pytest                # 48 tests, ~1.5s — runs fully offline
```

Run the package without install:

```sh
python -m rci_cli --help
```

### Test layout

```
tests/
├── conftest.py     # cfg fixture
├── test_config.py  # TOML load + defaults + partial overrides
├── test_alloc.py   # find_strongest + select_or_submit (GPU > CPU > submit)
├── test_launch.py  # resolve_folder + remote-preamble templating
├── test_slurm.py   # salloc / squeue / scancel command-string assertions
├── test_ssh.py     # ssh argv shape (subprocess.run mocked)
├── test_cli.py     # Typer CliRunner — version, jobs, cancel, cpu, gpu, alloc, port
└── test_tui.py     # headless Textual mount + JobRow parser + worker-error path
```

Everything that would otherwise touch the network is monkeypatched on
`rci_cli.ssh`, `rci_cli.slurm`, or `rci_cli.alloc` — tests run with no RCI
connectivity required.
