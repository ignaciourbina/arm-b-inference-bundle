# A letter to the Claude running the collection on SeaWulf

From: the Claude working in the private source repo (`simulating-open-democracy`).
Written: 2026-08-27. If you're the agent driving the Arm-B collection on the V100s
out of `~/projects/arm-b-inference-bundle`, this is for you.

You're doing well — the FAIL-log catch was sharp and correct. This letter closes
three gaps I can see from my side that you can't see from yours, in priority order.

---

## 1. Pull, first — you've been flying with stale + incomplete docs

You cloned this **bundle**, which until today shipped only the optimistic
short-prompt *probe* projections ("~434 tok/s at 24-way, no plateau, ~2.5 days,
~3–5×") in code comments and README — and **none** of the empirical corrections.
That's on us, not you. The corrections and the full evidence are now pushed. Get
them before any more tuning decisions:

```bash
cd ~/projects/arm-b-inference-bundle
git checkout -- llm/run_collection_parallel.py   # drop your inline FAIL-log hack
git pull                                          # clean upstream fix + all docs
```

Your inline patch is functionally equivalent to what's now upstream (`run_one`
writes `llm/traces/logs/FAIL_<tag>.log` and adds a stderr tail + log path to the
`run_fail`/`retry_fail` telemetry), so discarding it loses nothing. After the
pull, **read `llm/SCALING-FINDINGS.md`** — it's the authoritative,
measured-on-real-runs guide and it *wins over any contradicting code comment*.
The full memos are under `docs/analysis/` (sprint-15 ablation, sprint-16
scale-out).

## 2. Drop to `--lanes 4`, and prefer one server *per GPU*

`run_collection_parallel.py` **defaults to `--lanes 8`** — that is above the
measured ceiling. Sprint-16 P4a, on real 3–4k-ctx runs, found:

- 3 lanes = 1.72× macro; **4 lanes ≈ 1.9–2.1×** (per-run pace degrades ~1.69×
  under contention). More lanes past ~4 buys **no makespan** and costs
  reliability.
- The lever is **cards, not lanes**: each GPU is an independent decode-bandwidth
  domain, so scaling across the 2 V100s on your node is ~linear where lanes are
  sublinear.

So the intended topology is **one llama-server + one orchestrator per GPU**,
`CUDA_VISIBLE_DEVICES` pinned, distinct ports, **disjoint seed slices**, per-card
cells, **`--lanes 4` each** — not one 8-lane orchestrator against a tensor-split
server. If you're currently tensor-splitting one server across both cards: the
model fits in ~3 GB of a 32 GB card, so splitting buys nothing and adds cross-GPU
sync. Concretely:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 PORT=20434 N_PARALLEL=12 CTX=49152 CACHE_REUSE=256 \
  bash pipeline/runpod/scripts/start_llama_server_local.sh &
python llm/run_collection_parallel.py --cell arm_b_gpu0 --seeds 1-200 \
  --lanes 4 --run-parallel 6 --base-url http://localhost:20434 &

# GPU 1 — different port, disjoint seeds, its own cell
CUDA_VISIBLE_DEVICES=1 PORT=20435 N_PARALLEL=12 CTX=49152 CACHE_REUSE=256 \
  bash pipeline/runpod/scripts/start_llama_server_local.sh &
python llm/run_collection_parallel.py --cell arm_b_gpu1 --seeds 201-400 \
  --lanes 4 --run-parallel 6 --base-url http://localhost:20435 &
```

**Keep `--reasoning off`.** Do NOT use `run_collection_server.sh` to get more
slots — it flips `--reasoning on`, which changes generation semantics and makes
the traces non-comparable to the pinned set. Slot count / ctx is throughput-only
and safe; reasoning is not. That distinction is the whole reason you don't need
the "p24 reasoning-on" config for speed — you get the speedup from more cards at
reasoning-off `p12`.

## 3. Your ~19% failure rate is probably *partly the 8 lanes* — now you can prove it

The failures concentrating in **REFLECT** is the tell. Under heavy concurrent
load, reasoning-off gemma-4-E2B truncates / mis-terminates the REFLECT tool-call
more often — consistent with the known llama-server parse-500 quirk. Note
`llm/client.py` already has `_perturb_retry`: on a 500 it perturbs the sampling
seed and doubles the token budget on the retry (first attempts stay
deterministic; only retries diverge). So some REFLECT flakiness is expected and
self-heals.

19% sits at the **top** of your Sprint-15 0–20% Q8 band — not alarming, but it
smells like contention, not just a model floor. The clean experiment, now that
the FAIL log exists: after this pilot, re-run one cell at **one-server-per-GPU +
`--lanes 4`** and compare the REFLECT failure rate. If it drops, it was the
lanes. Read a couple of `FAIL_*.log` files to confirm the failure mode (timeout
vs. parse-500 vs. truncated tool-call) before concluding anything.

---

## Housekeeping you already have right

- `--resume` idempotency + the end-of-run retry pass will recover the 5
  checkpointed failures — no action needed.
- Model contract: the client requests the served GGUF filename
  `gemma-4-E2B-it-Q8_0.gguf`, and that's what's recorded in traces — they match
  by design. Don't "fix" a mismatch that isn't one.
- On SeaWulf specifically (24 h walltime, QoS cap, sm_70 build, K80 nodes
  unusable): the SBU-specific runbook + a self-resubmitting sbatch live in the
  **private** repo (`ops/seawulf-slurm-runbook.md`,
  `ops/seawulf-collect-arm-b.sbatch`) — ask IU to hand them over if you need the
  Slurm wiring; they're intentionally not in this public bundle.

You've got good instincts. The main thing was just missing context — that's
fixed now. — C.
