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

| command                              | replaces zsh helper            | notes                                              |
| ------------------------------------ | ------------------------------ | -------------------------------------------------- |
| `rci ssh`                            | `rci` (zsh alias `ssh rci`)    | interactive shell on the login host                |
| `rci jobs`                           | `rci-list`                     | `squeue -u $USER` with the friendly format         |
| `rci cpu`                            | `rci-cpu`                      | `--cores N --mem GB --time HH:MM:SS`               |
| `rci gpu`                            | `rci-gpu`                      | `--gpus N --cores N --mem GB --time HH:MM:SS`      |
| `rci cancel JOBID`                   | `rci-cancel`                   |                                                    |
| `rci cancel-all`                     | `rci-cancel-all`               | confirms first                                     |
| `rci cancel-dev`                     | `rci-cancel-vscode`            | cancel all rci-managed (`dev` + `dev-gpu`) allocs  |
| `rci shell  [DIR] [--gpu]`           | `rci-shell` / `-gpu`           | interactive bash on the compute node               |
| `rci editor [DIR] [--gpu]`           | `rci-code` / `-gpu`            | VS Code Remote-SSH (WSL → Windows `code.cmd`)      |
| `rci alloc  [--gpu]`                 | `_rci_alloc`                   | prints `<node> <jobid>` — scripting-friendly       |
| `rci port LOCAL[:REMOTE]`            | -                              | local → compute-node port forward (Ctrl-C to stop) |
| `rci tui`                            | *(new)*                        | Textual TUI dashboard (also: bare `rci`)           |
| `rci version`                        | -                              |                                                    |

**Folder argument rules** (applies to `editor`, `shell`):

- omitted → `/home/cizekto2`
- relative → resolved under `/home/cizekto2` (`rci shell sam2rl` → `/home/cizekto2/sam2rl`)
- absolute → used as-is

Run `claude` (or any other tool) directly from inside `rci shell`. Persistence
across ssh disconnect isn't wrapped at the rci-cli layer right now — re-introduce
when the rest of the UX is settled.

### TUI dashboard

Bare `rci` opens the Jobs dashboard. One-key actions:

| key | what it does |
|-----|--------------|
| `s` | Shell into the strongest running allocation. **If none exists, opens the New Instance modal** — configure (CPU/GPU, partition, cores, mem, walltime), submit, then auto-attaches once the job starts. |
| `e` | Same as `s` but launches the editor (VS Code Remote-SSH) instead. |
| `n` | Opens the New Instance modal without auto-attaching — just submit and return to the dashboard. |
| `c` | Cancel the highlighted job (confirmation modal). |
| `r` | Force-refresh the table (also auto-refreshes every 5s). |
| `t` | Cycle theme (escape hatch if `ansi-dark` looks bad on a given terminal). |
| `q` | Quit. |

The New Instance modal has a CPU/GPU toggle at the top — switching swaps the
partition default (`cpufast` ↔ `gpufast`), shows/hides the GPUs field, and
re-prefills the numeric defaults from `cfg.cpu_defaults` / `cfg.gpu_defaults`.

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
cpu_job_name = "dev"
gpu_job_name = "dev-gpu"
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
