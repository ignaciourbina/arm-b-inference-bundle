# Sprint 15 — Handoff

**Written:** 2026-08-15 · **Updated:** 2026-08-17 · **Status:** P0 RESOLVED —
programme collecting, currently PAUSED at cell 1 (8/15), resume-ready.

## One-line state

P0 is fixed and the programme runs on the GPU. The NVIDIA driver had TWO faults:
(1) a stale kernel module (fixed by reloading to 580.173.02), and (2) after
that reload NVIDIA **Vulkan** still enumerated zero GPUs while `nvidia-smi`
worked — a live-reload GPU-init state problem. `fix-gpu-vulkan.sh` (repo root)
cleared #2 with a full module teardown + device-node recreation + `modeset=1`
reload, no reboot. GPU benches **67.8 tok/s** cold; `run_ablation_programme.sh`
collects at ~6 min/run, GPU 80–87% util, ~9 h for all 6 cells.

**As of 2026-08-17 it is PAUSED** (checkpoint-stopped by user request) with
cell 1 `prompt_anti-repetition` at 8/15 runs, checkpoints intact. Resume with
the command in "Pausing and resuming" below.

## Pausing and resuming (checkpoint method — any duration, frees the GPU)

Resume is idempotent: each run is launched with `--resume`, completed runs
replay from checkpoint in ~1s, an in-flight run resumes from its last round,
and a config-mismatch guard refuses to resume across incompatible settings.

**To STOP cleanly — kill the cell DRIVER first, or it respawns the next seed.**
The nohup'd bash parent does NOT cascade to its children, so killing only the
programme PID leaves `run_ablation_cell.py` orphaned and it keeps spawning
runners. Kill by explicit PID, not `pkill -f <pattern>` — the pattern matches
your own shell's command line and kills it (exit 144). Order:

```bash
# 1. find the PIDs (read-only; safe)
pgrep -af 'run_ablation_cell[.]py'      # the DRIVER (respawner) — note its PID
pgrep -af 'llm[.]townhall[.]runner'     # the in-flight run
pgrep -x  llama-server                  # the server holding the GPU
# 2. kill by number, driver first, then runner, then server
kill <driver_pid> ; sleep 2 ; kill <runner_pid> ; sleep 2 ; kill <server_pid>
```

**To RESUME** (GPU must be up — re-run `fix-gpu-vulkan.sh --verify` first if the
box was rebooted):

```bash
cd ~/Dropbox/Workbench/research-anvil/01-Active/simulating-open-democracy
nohup bash llm/run_ablation_programme.sh > llm/traces/logs/programme_resume.out 2>&1 &
```

**Short interruptions only** (seconds–minutes, keeps VRAM warm) can instead use
`kill -STOP -<pgid>` / `kill -CONT -<pgid>` on the process group — but a long
freeze can time out the in-flight HTTP call, so prefer the checkpoint method.

## P0 — what fixed it (for next time)

- `nvidia-smi` passing is NOT the gate. The real gate is that **Vulkan**
  enumerates the card: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  llama-server --list-devices` must show `Vulkan0: ... RTX 2060`, not `llvmpipe`.
- A live `modprobe` swap fixes the version but can leave Vulkan blind. The
  remedy is `bash fix-gpu-vulkan.sh` (teardown all 4 modules → `nvidia-modprobe
  -c 0 -u -m` → `modprobe nvidia_drm modeset=1`). `--verify` re-checks without
  changing anything. Needs sudo; touches no disk/data.
- The bench guard (`run_ablation_cell.py`) now warms the graph before measuring,
  because the first request after any server restart reads ~10 tok/s cold even
  at 67 tok/s steady-state — that false-negative would otherwise abort each
  phase. Uncommitted change; commit with `fix-gpu-vulkan.sh` when ready.

## Original blocked-state note (superseded, kept for context)

Everything that can be built without the GPU is built, tested, and committed.
The ablation programme is staged in a single command and will run the whole
24-hour budget unattended — but it cannot start until the NVIDIA driver is
reloaded, which needs the user's sudo password.

## Where each plan item stands

| Item | State | Evidence |
|---|---|---|
| Immutable plan | ✅ committed | `f833f36` |
| P1 — prompt variants, runner `--prompt-variant`, provenance capture, cell driver, scorer | ✅ committed, 9 tests pass | `0e8beea`, `cee44eb` |
| P5 — LaTeX delta table + decision-memo generator | ✅ committed (awaits cell data) | `36c9311` |
| **P0 — load on-disk driver 580.173.02** | ⛔ **BLOCKED — user sudo only** | see below |
| P2 — prompt cells (3) | ⛔ gated on P0 | — |
| P3 — reasoning cells (2 + calibration) | ⛔ gated on P0 | — |
| P4 — quantization cell (1) | ⛔ gated on P0 | — |

## The blocker (P0), precisely

- **Loaded kernel module:** `580.159.03` (stale). **On-disk DKMS build:**
  `580.173.02`, already compiled for the running kernel `6.14.0-36-generic`.
- `nvidia-smi` → `Failed to initialize NVML: Driver/library version mismatch`.
- Backend benches at **~3.75 tok/s** = CPU fallback. GPU (Vulkan) is ~43 tok/s.
  The plan's exit threshold is **>30 tok/s**.
- **A reboot is NOT required.** As of the last check `nvidia_drm` had **0 users**,
  no display manager was active, and **0** processes held `/dev/nvidia*` — so a
  live module swap works. This matters because the box has **LUKS full-disk
  encryption**; a reboot would halt at the passphrase prompt and the user may
  have no physical access.
- Sudo is password-protected. Claude does not hold the password and will not
  ask for it. This step is the user's to run.

## What the user must run (P0)

```bash
pkill -x llama-server
sudo systemctl stop nvidia-persistenced 2>/dev/null
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
nvidia-smi ; cat /proc/driver/nvidia/version   # expect RTX 2060 SUPER + 580.173.02
```

If `modprobe -r` reports *"Module nvidia is in use"*, something grabbed the card
since the last check:

```bash
sudo lsof /dev/nvidia* 2>/dev/null   # find & kill the holder, then retry
```

## What the next session should do

1. **Re-verify P0** — do not trust a successful load alone. Bench the backend
   and confirm **>30 tok/s** (start the p6-cache server first if it is down):
   ```bash
   cat /proc/driver/nvidia/version                       # 580.173.02?
   agora/.venv/bin/python llm/run_ablation_cell.py \
       --cell _p0check --dry-run                          # prints benched tok/s, exits
   ```
2. **Launch P2–P5** — one command, self-guards against the CPU-fallback path
   (aborts in seconds if the GPU is not really up), ~19h of the 24h budget:
   ```bash
   cd ~/Dropbox/Workbench/research-anvil/01-Active/simulating-open-democracy
   bash llm/run_ablation_programme.sh
   ```
   Prefer detached so a tunnel drop doesn't kill it:
   ```bash
   nohup bash llm/run_ablation_programme.sh > llm/traces/logs/programme.out 2>&1 &
   ```
3. **P5 is automatic at the end** of the programme, but can be re-run any time on
   whatever cells exist:
   ```bash
   CELLS=$(ls llm/traces/ablation)
   agora/.venv/bin/python llm/score_ablation_cells.py --cells $CELLS
   agora/.venv/bin/python llm/report_ablation.py
   ```
   Output: `pipeline/output/reports/ablation/ablation_deltas.tex` (paired
   delta table) and `agora/analysis/sprint-15-ablation-prep/sprint-15-decision-memo.md`
   (numbers auto-filled; **the configuration choice is left to the author** —
   the generator deliberately does not auto-conclude from 5 seeds/composition).

## Key files

- `agora/analysis/sprint-15-ablation-prep/plan.md` — the immutable plan (P0–P5,
  budget table, exit criteria) + 2 addenda.
- `llm/run_ablation_programme.sh` — the whole budget in one command (P2/P3/P4/P5).
- `llm/run_ablation_cell.py` — one cell, paired seeds, with the >20 tok/s
  CPU-fallback guard (`--allow-slow` to override, `--dry-run` to just bench).
- `llm/prompt_variants.py` — 4 variants; `control` is proved byte-identical to
  `BASELINE_PROMPT_BUILDER` so cells stay comparable with the 390-run collection.
- `llm/score_ablation_cells.py`, `llm/report_ablation.py` — P5.
- `llm/tests/test_prompt_variants.py` — 9 tests guarding variant integrity.

## Environment notes for whoever runs this

- **asus-desktop is operated remotely** via a VS Code tunnel
  (`code-tunnel.service`, user systemd). It survives reboots only because
  `Linger=yes` (set 2026-08-05) and auth is file-based (`~/.vscode/cli/token.json`).
  Its `ExecStart` must point at `/snap/code/current/...` — a stale
  `/snap/code/227/...` path previously caused 215,698 crash-loops.
- Control cell is free: seeds 1–5 of the validated 390-run production collection
  in `llm/traces/beta_local/arm_b_local_p6cache/` are the `control` for every
  paired comparison. Do not overwrite them.
- Caveat carried into the memo: 5 seeds/composition detects large paired effects
  and nothing subtler; a null is "not detected at this power," not "absent."

## Not part of Sprint 15 (open, but separate)

- Arm-A `dri_orthogonal` full reproduction (Sprint 14 leftover).
- Possible Arm-B extension 390 → 1200 runs (~65 GPU h) — this is the run the
  ablation is meant to *configure*, not part of the ablation itself.
