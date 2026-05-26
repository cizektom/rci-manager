# rci-cli

A CLI + Textual TUI for the **RCI CVUT Slurm cluster** (and, with one config
edit, any other Slurm site). Submit allocations, shell into the assigned
compute node, run VS Code Remote-SSH, cancel jobs, forward ports — all from
one tool with a dashboard you can leave open.

```sh
git clone https://github.com/cizektom/rci-manager.git ~/rci-manager
uv tool install --python 3.11 ~/rci-manager   # or: pipx install ~/rci-manager
rci         # first run pops a setup wizard
```

---

## First-run setup

`rci-cli` ships **without** any cluster-specific personal info — the binary
walks you through a one-time setup on first launch (TUI modal) or via
`rci setup` (terminal prompt). Three fields:

| field      | what it is                            | default        |
| ---------- | ------------------------------------- | -------------- |
| `user`     | your SSH username on the cluster      | (required)     |
| `ssh_host` | host alias from `~/.ssh/config`       | `rci`          |
| `home`     | absolute home dir on the cluster      | `/home/<user>` |

The wizard writes `~/.config/rci-cli/config.toml`. Edit by hand any time, or
re-run `rci setup` to repopulate from the existing values.

When you `rci shell <folder>` (or `rci editor <folder>`), the launcher
`cd`s into the folder on the compute node and — if a `.venv/bin/activate`
exists there — sources it. No config needed; projects without a `.venv` are
left alone.

---

## Prerequisites

1. **SSH config** with a `Host` entry matching `ssh_host` above, plus
   `ProxyJump` entries for the cluster's compute nodes — example for RCI:

   ```sshconfig
   Host rci
       HostName login3.rci.cvut.cz
       User <your-username>

   Host n0* n1* n2* n3* n4* n5* n6* n7* n8* n9* g0* g1* g2* g3* g4* g5* g6* g7* g8* g9*
       ProxyJump rci
       User <your-username>
   ```

   The digit after `n`/`g` is deliberate: a bare `Host n* g*` would also
   match `github.com`, `gitlab.com`, `ghcr.io`, `nginx.*`, etc. and try to
   `ProxyJump rci` through them — breaking `git push` whenever the cluster
   isn't reachable.

   For other clusters, point `Host` at your login node and add `ProxyJump`
   entries that match your compute-node naming.

2. **Python ≥ 3.11**.
3. **`uv`** (recommended), `pipx`, or any `pip` install method.

---

## Commands

Bare `rci` opens the TUI in a real terminal; piped or non-TTY invocations
fall back to `--help`.

| command                              | notes                                              |
| ------------------------------------ | -------------------------------------------------- |
| `rci`                                | TUI dashboard (same as `rci tui`)                  |
| `rci setup`                          | (re-)run the configuration wizard                  |
| `rci ssh`                            | interactive shell on the login host                |
| `rci jobs`                           | `squeue -u $USER` with the friendly format         |
| `rci cpu`                            | submit a CPU `dev` allocation — `-c -m -t` flags   |
| `rci gpu`                            | submit a GPU `dev-gpu` allocation — `-g -c -m -t`  |
| `rci cancel JOBID`                   | cancel a single job (confirms first if running)    |
| `rci cancel-all`                     | cancel ALL your jobs (confirms)                    |
| `rci cancel-dev`                     | cancel all rci-managed jobs (`dev*` + `editor` + `agent*`) |
| `rci shell  [DIR] [--gpu]`           | interactive bash on the compute node               |
| `rci editor [DIR] [--gpu]`           | VS Code Remote-SSH (WSL → Windows `code.cmd`)      |
| `rci agent  [DIR] [--gpu] [...]`     | spawn `agent-N` + run `claude remote-control`      |
| `rci workspace [DIR] [--gpu] [-a N] [-T M]` | tmux workspace (N claude panes + M bash panes) |
| `rci alloc  [--gpu]`                 | prints `<node> <jobid>` — scripting-friendly       |
| `rci port LOCAL[:REMOTE]`            | local → compute-node port forward (Ctrl-C to stop) |
| `rci tui`                            | Textual TUI dashboard                              |
| `rci version`                        | prints the rci-cli version                         |

**Folder argument rules** (applies to `editor`, `shell`, `agent`, `workspace`):

- omitted → `cfg.home` (your cluster home dir)
- relative → resolved under `cfg.home` (`rci shell myproj` → `<home>/myproj`)
- absolute → used as-is

**Job-step wrapping.** `rci shell` and `rci agent` enter the allocation via
`srun --jobid=<id> --overlap` after the ssh hop, so the spawned process
joins the job's cgroup and inherits its environment (`CUDA_VISIBLE_DEVICES`,
CPU affinity, memory limits, etc.). Without this wrap you'd land on the
node *outside* the allocation and `nvidia-smi` would show every GPU on
the host — meaning you could grab one that belongs to another user's job.

**`rci editor` caveat.** VS Code Remote-SSH owns its SSH command end-to-end,
so we can't inject `srun` — `vscode-server` runs *outside* the job's cgroup.
For pure editing (files, language servers, git) this is harmless; vscode-server
itself doesn't touch the GPU. But anything you launch *from inside* VS Code —
integrated-terminal commands, Jupyter notebook cells, the "Run Python File"
button, build tasks — inherits the un-cgrouped env and would see every GPU
on the node. If you run ML/GPU code from VS Code, either wrap it manually
(`srun --jobid=$SLURM_JOB_ID --overlap …` in the terminal, or set
`CUDA_VISIBLE_DEVICES=0` for one-off scripts) or run it via `rci shell`
in a separate window.

Persistence across ssh disconnect isn't wrapped at the rci-cli layer for
`rci shell` — run `tmux` or `screen` inside if you need it. (`rci workspace`
*is* tmux-wrapped — see below.) The bare `rci` TUI itself runs locally and
survives any ssh drop.

### Workspace — predefined tmux cockpit, disconnect-safe

`rci workspace` (TUI key `w`) opens a tmux session on the compute node with
a default 2-or-3-pane layout, reusing an existing rci-managed allocation when
one is running (same alloc pool as `rci shell`).

```sh
rci workspace                  # CPU or strongest existing alloc, $home
rci workspace sam2rl           # under <home>/sam2rl
rci workspace sam2rl --gpu     # require/spawn a GPU alloc
rci workspace -a 3 -T 2        # 3 claude panes on top, 2 bash on bottom
```

Default layout (2 agents + 1 terminal):

```
+----------+----------+    panes 0 & 2: claude (auto-launched)
|          |          |
| claude 0 | claude 2 |
|          |          |
+----------+----------+
|       bash 1        |    pane 1: bash for ad-hoc commands
+---------------------+
```

Both claude panes start running on session creation — no Enter required.
Pane indices are creation-order (top-left=0, bottom=1, top-right=2). Use
Ctrl-b arrow keys to move between them — the numbers only matter if you're
scripting tmux.

**Pane counts.** `--agents`/`-a` and `--terminals`/`-T` (or the TUI's
Workspace options popup) control how many claude panes fill the top row
and how many bash panes fill the bottom row. Defaults
(`workspace_agents` / `workspace_terminals` in `~/.config/rci-cli/config.toml`)
ship at 2 + 1; the TUI also remembers your last choice across sessions.
Pane counts only matter the first time the session is built — reattaches
inherit the existing layout.

**Disconnect-safe.** The tmux daemon lives in a long-lived
`srun --jobid --overlap` step that holds the job's cgroup open. Detach with
Ctrl-b d, your local terminal returns; the session keeps running on the
compute node. Re-press `w` (or rerun `rci workspace`) on the same alloc to
reattach instantly. Closing every pane exits the session, which exits the
holder step, which Slurm cleans up — next workspace launch builds fresh.

**Cgroup-correct by construction.** The tmux *daemon* is forked from inside
the cgroup-wrapped srun (the holder); panes inherit the daemon's cgroup, so
`nvidia-smi` inside a pane sees only your allocated GPUs — no leak like the
`rci editor` caveat above. The interactive `tmux attach` client doesn't need
srun wrapping (clients only proxy keystrokes; the daemon does the work).

Stdout/stderr from the holder step append to
`~/.rci/workspace-logs/<jobid>.log` on the compute node if you need to debug
a failed setup. Per-job tmux socket (`rci-ws-<jobid>`) means multiple
workspaces on different allocations don't share a daemon.

### Agent — control Claude from your phone

`rci agent` always spawns a **new** allocation (`agent-N`, gap-reused like
`dev-N`), then runs [`claude remote-control`](https://docs.claude.com/) on
the compute node so the session is pairable from claude.ai/code or the
Claude mobile app. Pass-through flags map 1:1 to the `claude` ones:

```sh
rci agent                                       # CPU agent in $home
rci agent sam2rl --gpu                          # GPU agent in <home>/sam2rl
rci agent --permission-mode bypassPermissions   # auto-accept tool calls
rci agent --spawn worktree --capacity 16        # worktree-isolated sessions
rci agent --name nightly-train                  # custom claude.ai/code label
```

Defaults for `--permission-mode` / `--spawn` / `--capacity` come from the
`agent_*` keys in `~/.config/rci-cli/config.toml`; the `--name` defaults to
the job name (`agent-N`). Run `claude remote-control --help` for the meaning
of each option. Each invocation is a separate server — there's no reuse, by
design.

**Detached launch.** The `claude remote-control` server starts in the
background via `nohup … & disown`, so `rci agent` (and the TUI) returns
immediately — no terminal handover, the session keeps running after you
disconnect. Stdout/stderr append to
`~/.rci/agent-logs/<name>.log` on the compute node — `rci shell` to it
and `tail -f` if you ever need to peek at startup output. Pairing itself
needs no link: once you're signed into claude.ai/code or the mobile app,
the new session shows up there automatically.

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
│ f Frontend  s Submit  c Connect  e Editor  a Agent  w Workspace  r Refresh  d Delete  q Quit
└────────────────────────────────────────────────────────────────┘
```

**Key bindings** (Jobs panel):

| key | action |
|-----|--------|
| `f` | **Frontend** — open a shell on the cluster login host |
| `s` | **Submit** — open the New Instance modal to spawn a new allocation |
| `c` | **Connect** — shell into the highlighted job's compute node (prompts for folder first) |
| `e` | **Editor** — VS Code Remote-SSH against the highlighted job (prompts for folder first) |
| `a` | **Agent** — spawn a new `agent-N` and run `claude remote-control` (folder → agent options → resources) |
| `w` | **Workspace** — open (or reattach to) a tmux session on the highlighted job. Asks for pane counts (claude on top, bash on bottom; defaults 2 + 1, remembered across sessions). Detach with `Ctrl-b d` |
| `d` | **Delete** — cancel the highlighted job (confirmation modal, default ✕ No) |
| `r` | force-refresh the table (also auto-refreshes every 5s) |
| `↑/↓` or `j`/`k` | navigate rows; the detail line updates live |
| `←/→` or `h`/`l` | scroll the table horizontally when columns overflow |
| `g` / `G` | jump to first / last row (single-key, no `gg` chord) |
| `/` | filter rows live — substring match on jobid/name/state/partition. `Enter` commits the filter (keeps it active, returns focus to the table); `Esc` clears |
| `t` | cycle theme (`ansi-dark` ↔ `textual-dark`) |
| `q` / `Ctrl+C` | quit |

**New Instance modal** (`s`):

```
┌─ New instance ────────────────────────────────┐
│ Job name   [dev-2          ]                  │
│ Partition                                     │
│ [cpu     ▾]  [fast    ▾]                      │
│ Cores      [2     ]                           │
│ GPUs       [0     ]   (shown only for gpu*)   │
│ Memory     [4 GB  ]                           │
│ Walltime   [1:00:00 ▾]                        │
│                          [Submit] [Cancel]    │
└───────────────────────────────────────────────┘
```

- **Job name** is pre-filled with a suggestion: `dev-N` for Submit/Connect
  (lowest unused N — cancelled numbers come back into play), `editor` for
  the Editor flow (singleton, no number), or `agent-N` for the Agent flow.
  Blanking the field reverts to the suggestion; otherwise type whatever you
  like (e.g. `train-llama`).

**Agent flow** (`a`) is a 3-step wizard so the claude knobs stay separate
from the Slurm ones:

1. **Folder** — `FolderModal` (same as Connect/Editor).
2. **Agent options** — `AgentOptionsModal`: permission mode, spawn mode,
   capacity. Defaults come from `cfg.agent_*`; q/escape backs out to (1).
3. **Resources** — the standard New Instance modal with the suggested
   `agent-N` name; q/escape backs to (2) and your option choices are
   preserved.

- **No widget is focused on open.** Press **Enter** to submit the prefilled
  defaults (last-used params, or `cfg.cpu_defaults` / `gpu_defaults`).
- **Tab** walks the form starting at partition-type. Inside the form, **Enter**
  opens dropdowns (Select) or advances to the next field (Input).
- Inside an **open dropdown**: `↑`/`↓` or `j`/`k` to move the highlight,
  `g`/`G` to jump to the first/last option, **Enter** to commit, **Esc**
  to close without selecting. (Type-to-search is off — every list is short.)
- The partition is composed of two Selects: a **type** (`cpu` / `gpu` /
  `amdgpu` / `h200`) and a **class** (`fast` / `(normal)` / `long` /
  `extralong`). All 16 combinations match what the RCI cluster offers.
- The **GPUs** field shows only for GPU-capable partition types. Setting it >0
  with `cpu` selected is rejected with a clear toast.
- `q` / Esc close the modal from anywhere.

**Confirmation modal** (Delete, etc.):

- Always opens with focus on **No** — pressing Enter on reflex never confirms
  a destructive action.
- Tab to **Yes** (or press `y` from anywhere) to proceed.

---

## Configuration

The setup wizard writes the **personal** fields (`user`, `ssh_host`, `home`).
Every other field has a cluster-wide default matching the RCI cluster — edit
`~/.config/rci-cli/config.toml` to override:

```toml
# personal — filled in by `rci setup`, change at any time
user = "jdoe2"
ssh_host = "rci"
home = "/home/jdoe2"

# Default partitions for ``rci cpu`` / ``rci gpu`` (CLI). The modal lets you
# pick any partition; these are the auto-defaults.
cpu_partition = "cpufast"
gpu_partition = "gpufast"

# Job-name prefixes for rci-managed allocations. Anything spawned by Submit /
# Connect / ``rci cpu`` / ``rci gpu`` becomes ``<dev_job_name>-N`` (lowest
# unused N — cancelled numbers are reused). The Editor flow spawns a singleton
# ``<editor_job_name>`` (no number) since there's only ever one at a time.
# The Agent flow always spawns a fresh ``<agent_job_name>-N``; ``rci cancel-dev``
# sweeps all three prefixes.
dev_job_name = "dev"
editor_job_name = "editor"
agent_job_name = "agent"

# Defaults for ``claude remote-control`` flags used by ``rci agent`` and the
# TUI's Agent flow. Override per-call with --permission-mode / --spawn /
# --capacity. See ``claude remote-control --help`` for accepted values.
agent_permission_mode = "default"   # acceptEdits | auto | bypassPermissions | default | dontAsk | plan
agent_spawn_mode = "same-dir"       # same-dir | worktree | session
agent_capacity = 32                 # max concurrent sessions per server

# Resource defaults — kept conservative so a forgotten allocation doesn't
# burn quota. Override per-call with --cores / --mem / --time / --gpus.
cpu_defaults = [2, 4, "1:00:00"]      # cores, memGB, walltime
gpu_defaults = [1, 2, 8, "1:00:00"]   # gpus, cores, memGB, walltime

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
├── config.py     # Defaults + TOML overrides + save() for the wizard
├── setup.py      # First-run wizard (CLI side; TUI modal lives in tui.py)
├── state.py      # Persistent state (last folder, last modal params) — JSON
├── ssh.py        # ssh wrappers (capture / run / run_local / port_forward)
├── slurm.py      # squeue / salloc / scancel primitives
├── alloc.py      # select_or_submit() — re-use or spawn an allocation
├── launch.py     # shell / editor launchers on compute nodes
└── tui.py        # Textual app, modals, jobs dashboard, SetupModal
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
- [x] First-run setup wizard (CLI + TUI) — no personal info in the source tree.
- [ ] Log tailing for running jobs (`sattach` or remote `tail -F`).
- [ ] GPU/CPU utilization snapshot on the active node.
- [ ] Saved allocation profiles (`rci profile use ml-train`).
- [ ] Batch script editor (`sbatch` flow alongside the salloc flow).
- [x] Persistent tmux wrapping (see `rci workspace` / TUI `w` — cgroup-correct,
      disconnect-safe, predefined layout).

---

## Development

```sh
git clone https://github.com/cizektom/rci-manager.git ~/rci-manager
cd ~/rci-manager
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
uv tool install --reinstall --python 3.11 ~/rci-manager
```

### Test layout

```
tests/
├── conftest.py      # cfg fixture + autouse XDG isolation
├── test_config.py   # TOML load + defaults + save() + needs_setup()
├── test_setup.py    # wizard: build_cfg, run_cli prompts
├── test_alloc.py    # select_or_submit (re-use existing, submit when missing)
├── test_launch.py   # resolve_folder + remote-preamble + launchers
├── test_slurm.py    # salloc / squeue / scancel command-string assertions
├── test_ssh.py      # ssh argv shape (subprocess.run mocked)
├── test_cli.py      # Typer CliRunner — every subcommand
└── test_tui.py      # headless Textual mount + modals + JobRow parser + state persistence
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

---

## License

MIT — see [LICENSE](LICENSE).
