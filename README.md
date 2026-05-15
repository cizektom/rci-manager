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
| `rci ssh`            | `rci` (zsh alias for `ssh rci`) | interactive shell on the login host          |
| `rci jobs`               | `rci-list`                | `squeue -u $USER` with the friendly format         |
| `rci cpu`                | `rci-cpu`                 | `--cores N --mem GB --time HH:MM:SS`               |
| `rci gpu`                | `rci-gpu`                 | `--gpus N --cores N --mem GB --time HH:MM:SS`      |
| `rci cancel JOBID`       | `rci-cancel`              |                                                    |
| `rci cancel-all`         | `rci-cancel-all`          | confirms first                                     |
| `rci cancel-vscode`      | `rci-cancel-vscode`       | confirms first                                     |
| `rci claude [DIR] [--gpu]` | `rci-claude` / `-gpu`   | folder rules below                                 |
| `rci code [DIR] [--gpu]`   | `rci-code` / `-gpu`     | WSL → Windows `code.cmd` via `cmd.exe`             |
| `rci shell [DIR] [--gpu]`  | *(new)*                 | interactive bash on the compute node               |
| `rci alloc [--gpu]`        | `_rci_alloc`            | prints `<node> <jobid>` — scripting-friendly       |
| `rci tui`                  | *(new)*                 | Textual TUI — live jobs dashboard (skeleton)       |
| `rci version`              | -                       |                                                    |

**Folder argument rules** (applies to `claude`, `code`, `shell`):

- omitted → `/home/cizekto2`
- relative → resolved under `/home/cizekto2` (`rci claude sam2rl` → `/home/cizekto2/sam2rl`)
- absolute → used as-is

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
cpu_defaults = [4, 16, "4:00:00"]            # cores, memGB, walltime
gpu_defaults = [1, 8, 32, "4:00:00"]         # gpus, cores, memGB, walltime
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
pip install -e .
rci --help
```

Run the package without install:

```sh
python -m rci_cli --help
```
