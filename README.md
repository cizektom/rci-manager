# rci-cli

A CLI + Textual TUI for the **RCI CVUT Slurm cluster**. Submit allocations,
shell into the assigned compute node, run VS Code Remote-SSH, cancel jobs,
forward ports — all from one tool with a dashboard you can leave open.

```sh
git clone git@github.com:cizektom/rci-cli.git ~/rci-cli
uv tool install --python 3.11 ~/rci-cli   # or: pipx install ~/rci-cli
rci --help
```

The binary is named `rci`. If you previously used the
[zsh-setup](https://github.com/cizektom/zsh-setup) `alias rci='ssh rci'`,
remove it — `rci ssh` does the same and the alias would otherwise shadow
this binary.

---

## Prerequisites

1. **SSH config** with `Host rci` (and `Host n* g*` via `ProxyJump rci`) — see
   [zsh-setup/ssh/rci.conf](https://github.com/cizektom/zsh-setup/blob/main/ssh/rci.conf).
2. **Python ≥ 3.11**.
3. **`uv`** (recommended), `pipx`, or any `pip` install method.

---

## Commands

Bare `rci` opens the TUI in a real terminal; piped or non-TTY invocations
fall back to `--help`.

| command                              | notes                                              |
| ------------------------------------ | -------------------------------------------------- |
| `rci`                                | TUI dashboard (same as `rci tui`)                  |
| `rci ssh`                            | interactive shell on the login host                |
| `rci jobs`                           | `squeue -u $USER` with the friendly format         |
| `rci cpu`                            | submit a CPU `dev` allocation — `-c -m -t` flags   |
| `rci gpu`                            | submit a GPU `dev-gpu` allocation — `-g -c -m -t`  |
| `rci cancel JOBID`                   | cancel a single job (confirms first if running)    |
| `rci cancel-all`                     | cancel ALL your jobs (confirms)                    |
| `rci cancel-dev`                     | cancel all rci-managed `dev` / `dev-gpu` jobs      |
| `rci shell  [DIR] [--gpu]`           | interactive bash on the compute node               |
| `rci editor [DIR] [--gpu]`           | VS Code Remote-SSH (WSL → Windows `code.cmd`)      |
| `rci alloc  [--gpu]`                 | prints `<node> <jobid>` — scripting-friendly       |
| `rci port LOCAL[:REMOTE]`            | local → compute-node port forward (Ctrl-C to stop) |
| `rci tui`                            | Textual TUI dashboard                              |
| `rci version`                        | prints the rci-cli version                         |

**Folder argument rules** (applies to `editor`, `shell`):

- omitted → `cfg.home` (default `/home/cizekto2`)
- relative → resolved under `cfg.home` (`rci shell sam2rl` → `/home/cizekto2/sam2rl`)
- absolute → used as-is

Persistence across ssh disconnect isn't wrapped at the rci-cli layer — run
`tmux` or `screen` inside `rci shell` if you need it. The bare `rci` TUI
itself runs locally and survives any ssh drop.

---

## The TUI

Bare `rci` opens the dashboard:

```
┌─ rci · RCI CVUT Slurm cluster ─────────────────────────  12:34 ┐
│ Jobs                                                            │
├────────────────────────────────────────────────────────────────┤
│ 2 running  ·  1 pending                                         │
│ JobID    Partition Name    State   Time  Limit  CPU Mem GPU Node│
│ 1234567  cpufast   dev     RUNNING 00:05 01:00  2   4G  —   n01 │
│ 1234568  gpufast   dev-gpu RUNNING 00:10 04:00  8   32G 1   g05 │
│ 1234569  cpu       train   PENDING 0:00  24:00  16  64G —   —   │
├────────────────────────────────────────────────────────────────┤
│ 1234568  dev-gpu  ·  RUNNING  on  gpufast  ·  8 CPU · 32G · 1 GPU
│ ·  used 00:10 / limit 04:00:00  ·  node g05                     │
├────────────────────────────────────────────────────────────────┤
│ l Login  s Submit  c Connect  e Editor  k Kill  r Refresh  q Quit
└────────────────────────────────────────────────────────────────┘
```

**Key bindings** (Jobs panel):

| key | action |
|-----|--------|
| `l` | **Login** — open a shell on the RCI login host |
| `s` | **Submit** — open the New Instance modal to spawn a new allocation |
| `c` | **Connect** — shell into the highlighted job's compute node (prompts for folder first) |
| `e` | **Editor** — VS Code Remote-SSH against the highlighted job (prompts for folder first) |
| `k` | **Kill** — cancel the highlighted job (confirmation modal, default ✕ No) |
| `r` | force-refresh the table (also auto-refreshes every 5s) |
| `↑/↓` | navigate rows; the detail line updates live |
| `t` | cycle theme (`ansi-dark` ↔ `textual-dark`) |
| `q` / `Ctrl+C` | quit |

**New Instance modal** (`s`):

```
┌─ New instance ────────────────────────────────┐
│ Partition                                     │
│ [cpu     ▾]  [fast    ▾]                      │
│ Cores      [2     ]                           │
│ GPUs       [0     ]   (shown only for gpu*)   │
│ Memory     [4 GB  ]                           │
│ Walltime   [1:00:00 ▾]                        │
│                          [Submit] [Cancel]    │
└───────────────────────────────────────────────┘
```

- **No widget is focused on open.** Press **Enter** to submit the prefilled
  defaults (last-used params, or `cfg.cpu_defaults` / `gpu_defaults`).
- **Tab** walks the form starting at partition-type. Inside the form, **Enter**
  opens dropdowns (Select) or advances to the next field (Input).
- The partition is composed of two Selects: a **type** (`cpu` / `gpu` /
  `amdgpu` / `h200`) and a **class** (`fast` / `(normal)` / `long` /
  `extralong`). All 16 combinations match what the RCI cluster offers.
- The **GPUs** field shows only for GPU-capable partition types. Setting it >0
  with `cpu` selected is rejected with a clear toast.
- `q` / Esc close the modal from anywhere.

**Confirmation modal** (Kill, etc.):

- Always opens with focus on **No** — pressing Enter on reflex never confirms
  a destructive action.
- Tab to **Yes** (or press `y` from anywhere) to proceed.

---

## Configuration

Defaults match the original zsh helpers and the RCI cluster's actual
partitions. Override any field in `~/.config/rci-cli/config.toml`:

```toml
user = "cizekto2"
ssh_host = "rci"
home = "/home/cizekto2"

# Default partitions for ``rci cpu`` / ``rci gpu`` (CLI). The modal lets you
# pick any partition; these are the auto-defaults.
cpu_partition = "cpufast"
gpu_partition = "gpufast"

# Job names used by ``rci cancel-dev`` and ``alloc.select_or_submit``.
cpu_job_name = "dev"
gpu_job_name = "dev-gpu"

# Resource defaults — kept conservative so a forgotten allocation doesn't
# burn quota. Override per-call with --cores / --mem / --time / --gpus.
cpu_defaults = [2, 4, "1:00:00"]      # cores, memGB, walltime
gpu_defaults = [1, 2, 8, "1:00:00"]   # gpus, cores, memGB, walltime

# Compute-node venv to source before claude / editor / shell launches.
venv_activate = "$HOME/sam2rl/.venv/bin/activate"

# Partition catalog — what the modal's two dropdowns offer. Override these
# to adapt rci-cli to a different Slurm cluster without changing any code.
partition_types = ["cpu", "gpu", "amdgpu", "h200"]
partition_classes = [
    ["fast", "fast"],
    ["(normal)", ""],
    ["long", "long"],
    ["extralong", "extralong"],
]
# Partition type prefixes that accept ``--gres=gpu:N``. Anything outside
# this list rejects ``gpus > 0`` in the modal.
gpu_partition_types = ["gpu", "amdgpu", "h200"]
```

Anything you leave out keeps its default. List-of-list fields
(`partition_classes`) are coerced to tuples-of-tuples on load so the frozen
Config dataclass stays hashable.

---

## Architecture

Single Python package, `src/` layout:

```
src/rci_cli/
├── cli.py        # Typer app — subcommand routing
├── config.py     # Defaults + TOML overrides
├── state.py      # Persistent state (last folder, last modal params) — JSON
├── ssh.py        # ssh wrappers (capture / run / run_local / port_forward)
├── slurm.py      # squeue / salloc / scancel primitives
├── alloc.py      # select_or_submit() — re-use or spawn an allocation
├── launch.py     # shell / editor launchers on compute nodes
└── tui.py        # Textual app, modals, jobs dashboard
```

Everything that touches Slurm goes through `slurm.py`; everything that touches
ssh goes through `ssh.py`. The CLI and TUI both consume the same primitives,
so the contract surface is small and well-tested.

---

## Roadmap

- [x] 1:1 parity with the original zsh `rci-*` helpers.
- [x] Live jobs dashboard (TUI) with one-key actions.
- [x] In-TUI cancel / submit / attach.
- [x] Per-call partition picker (cluster portability via config catalog).
- [ ] Log tailing for running jobs (`sattach` or remote `tail -F`).
- [ ] GPU/CPU utilization snapshot on the active node.
- [ ] Saved allocation profiles (`rci profile use ml-train`).
- [ ] Batch script editor (`sbatch` flow alongside the salloc flow).
- [ ] Persistent (tmux/zellij) wrapping around `rci shell` — when it's
      proven worth the complexity.

---

## Development

```sh
git clone git@github.com:cizektom/rci-cli.git ~/rci-cli
cd ~/rci-cli
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
rci --help
pytest               # runs fully offline; ssh/slurm calls are monkeypatched
```

Run the package without install:

```sh
python -m rci_cli --help
```

After source edits, refresh the installed binary:

```sh
uv tool install --reinstall --python 3.11 ~/rci-cli
```

### Test layout

```
tests/
├── conftest.py     # cfg fixture
├── test_config.py  # TOML load + defaults + partition-catalog overrides
├── test_alloc.py   # select_or_submit (re-use existing, submit when missing)
├── test_launch.py  # resolve_folder + remote-preamble + launchers
├── test_slurm.py   # salloc / squeue / scancel command-string assertions
├── test_ssh.py     # ssh argv shape (subprocess.run mocked)
├── test_cli.py     # Typer CliRunner — version / jobs / cancel / cpu / gpu / alloc / port / shell / editor
└── test_tui.py     # headless Textual mount + modals + JobRow parser + state persistence
```

All network-touching primitives are monkeypatched per-test (`rci_cli.ssh`,
`rci_cli.slurm`, `rci_cli.alloc`) — the suite runs with zero RCI connectivity.

### Shell completion (zsh)

A hand-crafted completion lives in `completions/_rci`. Either let `rci`
install its own (Typer-generated) version:

```sh
rci --install-completion
```

…or use the tracked version (grouped sections, per-subcommand flag/value
completion):

```sh
cp completions/_rci ~/.zfunc/_rci
# ensure ``fpath=(~/.zfunc $fpath)`` runs before ``compinit`` in your .zshrc
```
