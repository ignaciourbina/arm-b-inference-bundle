# Compressed collection traces

One `tar.zst -19` archive per cell of `llm/traces/collection/` (STORAGE-POLICY
§7, Tier 2): 31 cells. Each archive extracts to exactly the git-tracked files
of its cell (round-trip verified with sha256 at creation). `e4b_r256_t10` was
added 2026-11-23 (300 runs, A100). `e4b_r256_t10_s101` was added 2026-11-26:
seeds 101-200 of the same collection (300 runs, A100 job 2456677, identical
run config apart from the seed), kept as its own directory so no file is
renamed. Read the two directories together for 200 seeds per room. The other
29 date from 2026-11-13.

The plain-JSON cells under `llm/traces/collection/` remain the working copy that
every reader script uses; these archives are a compact, portable snapshot of them.

## Auxiliary archives (2026-11-23)

These cover trace files and logs that no cell archive covers. Paths inside are
relative to the repo root, except `slurm_out`, `scratch_strays` and `scratch_misc`, which are
relative to `$SCRATCH` (purged after 30 days, hence archived here).

| archive | contents |
|---|---|
| `aux_llm_traces_logs_1` | `llm/traces/logs/` files for the `e4b*`, `ms*`, `cd*`, `abl*` families: server logs, `.out`, `FAIL_*`, progress telemetry |
| `aux_llm_traces_logs_2` | all other `llm/traces/logs/` files. The `.log` files are gitignored, so they exist only here and on disk. |
| `aux_p6cache` | `llm/traces/beta_local/seawulf_test_p6cache` |
| `aux_misc` | `llm/traces/realprofile_smoke`, `llm/traces/SYNTHETIC_VALIDATION`, `llm/traces/agentic_*.json` |
| `aux_slurm_out` | `$SCRATCH/*.log` (every SLURM `--output` file: collections and llama.cpp builds) and `$SCRATCH/*.sbatch` (the build scripts) |
| `aux_scratch_misc` | relative to `$SCRATCH`; 21 small files no other archive covers: `polis-derived/` (first Ising-profile reconstruction, 2026-08-27), `synthetic-validation/` (synthetic-profile generator + outputs), the four `*_scores.json` from the Aug 28–29 ablation scoring, `cpu_resp_1..6.json` (CPU-bench server responses), and five tiny job notes (`26b_target.txt`, `build_jobid.txt`, `ck_meta.txt`, `PAUSED.txt`, `rflag.err`). Excluded as already tracked: `polis-data/*.csv` (= parent `datasets/polis-openData/15-per-hour-seattle/`), `official_ising_profiles.json` (= `polis-analysis/output/ising_profiles.json`), `my-reconstruction/` (= `polis-derived/` + `polis-analysis/output/PROVENANCE.md`) |
| `aux_scratch_strays` | not in any cell: `resume_test/`, `resume_test2/` (resume-guard tests), `superseded-traces/`, and a stray `seawulf_test_l4` seed-8 final at the `$SCRATCH` root |

Per-cell provenance (jobs, GPU, build, model sha256, server args) is in
`archives/provenance/`.

## Scrubbing

This repo is public. Before archiving, every text file went through three
replacements, applied in order: `/gpfs/home/<user>` → `~`, `/gpfs/scratch/<user>` →
`$SCRATCH`, `<user>` → `$USER`. For the 2026-11-23 and 2026-11-26 additions it was checked
file by file: scrubbed text equals the original with only those replacements.
The unscrubbed originals are not in this repo.

```bash
sha256sum -c SHA256SUMS                                  # verify
zstd -dc armb_sota_r256_t10.tar.zst | tar -xf -          # extract one cell
```

Archives are built reproducibly (`--sort=name --mtime=@0 --owner=0 --group=0`),
so re-archiving an unchanged cell yields an identical file.
