# Storage constraints and priorities — SeaWulf

Frozen 2026-08-30. Verified with the cluster's own `myquota`, not `df`
(`df` on GPFS reports *fileset* size, not your quota — it will tell you there
are 20 TB free in your home directory, which is wrong and dangerous).

---

## 1. The quotas (measured, `myquota`)

| Space | Path | Used | **Quota** | Backed up? | Auto-purge? |
|---|---|---:|---:|---|---|
| **home** | `/gpfs/home/$USER` | 11 GB | **20 GB** | **YES** | no |
| **scratch** | `/gpfs/scratch/$USER` | 51 GB | **20 TB** | **NO** | **YES — 30 days** |
| SOC595 | `/gpfs/projects/SOC595` | 4 GB | 1 TB | — | — |

File-count limits also apply: home 204,800 files; scratch 10,000,000.
Home is currently at 77,705 files — **38% of the file-count quota**, which will
likely bind before the GB quota does. Watch both.

**`SOC595` is a course allocation and is NOT ours to use.** Do not put project
data there, however much space it appears to have.

---

## 2. The priority that governs every placement decision

> **The collected traces are the irreplaceable artifact.** They cost GPU-days to
> produce and cannot be regenerated cheaply. Everything else — model weights,
> container images, build trees, caches — is re-downloadable or rebuildable in
> minutes to hours.

Therefore:

**Irreplaceable → home (backed up).**
**Regenerable → scratch (fast, huge, disposable).**

Never invert this under space pressure. Moving traces to scratch to free home
would put months of collection behind a 30-day delete timer on an un-backed-up
filesystem. If home fills, compress or prune *within* home, or ask RCI for more
— do not relocate the data.

---

## 3. Current placement (correct as of freeze)

### On scratch (regenerable — safe to lose)

```
/gpfs/scratch/$USER/
  models/          all GGUF weights (E2B, E4B, 26B-A4B)   ~20 GB, re-downloadable
  images/          Singularity SIFs (cuda124, llamacpp, ollama)  ~9 GB, re-pullable
  sing-cache/      Singularity build cache                ~19 GB, PURGEABLE ANY TIME
  src/llama.cpp/   build trees (build, build-cpu, build-multi)   ~1.2 GB, rebuildable
  envs/, mamba/, conda-pkgs/, opt/                        ~1.6 GB, rebuildable
  polis-data/      raw Polis CSVs                         re-downloadable
```

`~/.cache/llama-models` is a **symlink** to `/gpfs/scratch/$USER/models`, so the
repo's hardcoded model paths resolve without any weights touching home.

### In home (irreplaceable — backed up)

```
~/projects/arm-b-inference-bundle/
  llm/traces/collection/     THE RESEARCH DATA
    finals  (townhall_*.json, not _checkpoint/_trace_)   ~50 MB  <- essential
    checkpoints                                          ~40 MB  <- resume state
    raw call traces (townhall_*_trace_*.json)            ~2.9 GB <- debug logs
  polis-analysis/output/     derived Ising profiles + provenance
  docs/                      memos
```

---

## 4. The space actually at risk, and what it is

**97% of home's project usage is raw LLM call logs** (`*_trace_*.json` — every
prompt and response), not results. The research data is ~50 MB; the debug logs
are ~2.9 GB.

Those logs **compress 13.3x** (5.8 MB -> 445 KB), so 2.9 GB -> ~0.22 GB.

**Do not gzip them yet.** Six scripts open them as plain JSON and would break
silently:

```
llm/inventory_traces.py          llm/analyze_ablation_reliability.py
llm/plot_ablation_trajectories.py  llm/bench_local_config.py
llm/run_collection_parallel.py     llm/tests/test_client_trace.py
```

`run_ablation_analysis.sh` depends on two of them. Compression is the right
eventual move, but it must ship **with `.gz` support in those readers**, not
before.

---

## 5. Trigger points

| Condition | Action |
|---|---|
| home < 5 GB free **or** > 160k files | add `.gz` support to the six readers, then compress `*_trace_*.json` for completed cells |
| still tight after that | prune raw traces of superseded cells (keep finals + checkpoints always) |
| still tight | request additional storage from RCI — do **not** move traces to scratch |
| scratch anything unused > 2 weeks | purge freely; `sing-cache/` (~19 GB) is always safe to delete |

Rough budget: a 300-run collection costs ~50 MB of finals + ~1.6 GB of raw
traces. Home has ~9 GB free, so roughly **5 more full collections** before the
GB quota binds — but check the file count too.

---

## 6. Commands

```bash
myquota                                   # authoritative — use this, not df
du -sh ~/projects/arm-b-inference-bundle/llm/traces/*
find llm/traces -name '*_trace_*.json' -printf '%s\n' | awk '{s+=$1} END{print s/1e9" GB"}'
rm -rf /gpfs/scratch/$USER/sing-cache     # always safe; SIFs are already built
```

**Reminder:** scratch is wiped after 30 days of inactivity and is not backed up.
Anything there must be reproducible from a script or a URL.

---

## 7. Trace compression policy (measured 2026-08-30, not yet applied)

Raw call traces (`townhall_*_trace_*.json`) are ~97% of the project's home
usage and compress extremely well. Measured on `sl_lanes_L6` (70 MB of traces):

| Strategy | ratio | random access | notes |
|---|---:|---|---|
| per-file `.json.gz` | **23.3x** | **KEPT** | `gzip.open()` is a drop-in for readers |
| single `.zip` | 23.3x | lost | **strictly dominated** — same ratio, no random access |
| **`tar.zst -19` per cell** | **67.4x** | lost | shared dictionary across near-identical files |

**Do not use zip.** It compresses each entry independently, so it matches
per-file gzip's ratio while giving up per-file access for nothing. The gain in
`tar.zst` comes from cross-file redundancy, which zip cannot exploit.

### The two-tier policy

**Tier 1 — cells still being analysed: per-file `.json.gz` (23x).**
Random access is preserved, so tooling keeps working after a one-line change
per reader. Applied to today's 2.9 GB: **-> ~125 MB**.

*Prerequisite (blocking):* six scripts open these as plain JSON and would break
silently —

```
llm/inventory_traces.py            llm/analyze_ablation_reliability.py
llm/plot_ablation_trajectories.py  llm/bench_local_config.py
llm/run_collection_parallel.py     llm/tests/test_client_trace.py
```

`run_ablation_analysis.sh` depends on two of them. The fix is small — swap
`open(p)` for a helper that picks `gzip.open` on a `.gz` suffix — but it must
land **before** any compression, and `llm/tests/` must pass afterwards.

**Tier 2 — superseded / archived cells: `tar.zst -19` per cell (67x).**
For cells that are no longer read (e.g. `sl_*`, `cd_*`, `vo_*` once the memo is
final, and anything under `superseded-traces/`). Applied to today's 2.9 GB:
**-> ~43 MB**. Extraction is required to read them, which is the correct
trade for material nobody is analysing.

Keep **finals and checkpoints uncompressed always** — they are only ~90 MB
together, they are the actual research data, and they are what the validator
and scorer read on every run.

### Order of operations

1. Add `.gz`-aware reading to the six scripts; `pytest llm/tests/` green.
2. Tier-1 compress raw traces of completed cells.
3. Tier-2 archive superseded cells to `tar.zst`.
4. Only then reconsider whether anything needs to leave home at all — at 67x,
   ~2.9 GB becomes ~43 MB and the space problem disappears entirely.

Note this makes relocating traces to scratch unnecessary, which is the
preferred outcome: compression keeps the irreplaceable data in backed-up home
rather than on a filesystem that deletes after 30 days.
