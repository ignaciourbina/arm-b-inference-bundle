# SeaWulf cluster policy — job and node limits

Reference for planning Arm-B collections on SBU SeaWulf. Retrieved **2026-08-28**.

Everything in the **Verbatim policy** sections below is quoted word-for-word from
the official Research Computing documentation, with the source URL given. The
**Implications** section is *our own analysis and measurements* and is not
policy — it is clearly separated for that reason.

---

## 1. Global user limits (verbatim)

> "users cannot use more than 32 nodes at one time unless running jobs in one of the large queues"

> "the maximum number of jobs that a user can have queued at any given time is 100"

Source: <https://rci.stonybrook.edu/HPC/faqs/seawulf-queues>

---

## 2. Queue limits (as printed in the documentation)

### Queues reachable from `login1` / `login2` (Haswell, 28-core)

| Queue | Max Runtime | Max Nodes | Max Jobs/User |
|-------|-------------|-----------|---------------|
| debug-28core | 1 hour | 8 | n/a |
| short-28core | 4 hours | 12 | 8 |
| medium-28core | 12 hours | 24 | 2 |
| long-28core | 48 hours | 8 | 6 |
| extended-28core | 7 days | 2 | 6 |
| large-28core | 8 hours | 80 | 1 |
| gpu | 8 hours | 2 | 2 |
| gpu-long | 48 hours | 1 | 2 |
| gpu-large | 8 hours | 4 | 1 |
| p100 | 24 hours | 1 | 1 |
| **v100** | **24 hours** | **1** | **1** |

### Queues reachable from `milan1` / `milan2` (selected)

| Queue | Max Runtime | Max Nodes | Max Jobs/User |
|-------|-------------|-----------|---------------|
| debug-40core | 1 hour | 8 | n/a |
| short-96core | 4 hours | 8 | 4 |
| long-96core | 48 hours | 6 | 3 |
| extended-96core | 7 days | 2 | 3 |
| large-96core | 8 hours | 38 | 1 |
| hbm-1tb-long-96core | 48 hours | 1 | 1 |
| **a100-long** | **48 hours** | **1** | **2** |

Source: <https://rci.stonybrook.edu/HPC/faqs/seawulf-queues>

---

## 3. Fairshare and priority (verbatim)

> "users who have consumed fewer resources recently receive a higher fairshare score"

> "Scheduling priority - determines the order in which jobs are considered for scheduling based on factors such as queue, requested resources, walltime, and fairshare"

Source: <https://rci.stonybrook.edu/HPC/docs/policies/fairshare>

---

## 4. Recommended practice / etiquette (verbatim)

> "Submit smaller jobs when possible; multiple smaller jobs often start faster than one very large job."

> "Use appropriate queues that match your job's walltime and node requirements."

> "Stagger job submissions to avoid sending many jobs to the queue simultaneously."

> "Use job arrays for repeated tasks to reduce queue overhead and improve scheduling efficiency."

> "Monitor your fairshare score using `squeue -u <username>` or other SLURM commands."

Source: <https://rci.stonybrook.edu/HPC/docs/policies/fairshare>

### Data handling (verbatim)

> "never upload, process, or store PHI on SeaWulf under any circumstances"

Source: <https://rci.stonybrook.edu/HPC/faqs>

---

## 5. Implications for Arm-B collections (OUR ANALYSIS — not policy)

### The binding constraint is the v100 queue, not the hardware

`v100` allows **1 node, 1 job per user**. `sn-nvda[1,2]` each hold 2× V100, so a
single user can never exceed **2 V100 GPUs at once**, however idle the other node
looks. Any plan quoting "4 GPUs" is unreachable for one user. `p100` is a
*separate* queue with its own 1-job allowance, so a concurrent p100 job legally
adds 2× P100 (~1 V100-equivalent) — but our llama.cpp build is
`CMAKE_CUDA_ARCHITECTURES=70-real` (sm_70 only) and would need `60-real` added
before it runs there.

### CPU capacity is the only way past 2 GPUs

Measured on one 28-core node (Xeon E5-2683 v3, gemma-4-E2B Q8_0, 6 slots):

| | aggregate decode | per run (~168k decode tok) | runs/h |
|---|---:|---:|---:|
| 1× CPU node (28 cores) | 37.6 tok/s | ~4,470 s (1.24 h) | 0.81 |
| 1× V100 node (2 GPUs) | ~249 tok/s per GPU | 675 s per GPU | 10.7 |

One CPU node ≈ **1/13** of the V100 node. But the 32-node global cap means a CPU
fleet can reach **~26 runs/h ≈ 2.4× the entire V100 node**, because the v100
queue caps you at one node while CPU does not.

### Shape a CPU fleet to the rules

- `long-28core` (48 h, 8 nodes/job, 6 jobs/user) is the best fit — long enough
  for 1.24 h/run, and 6 jobs × 8 nodes reaches the 32-node cap.
- `extended-28core`'s 7-day walltime is tempting but caps at **2 nodes**.
- `large-28core` offers 80 nodes but only 8 h and 1 job — thin margin for
  1.24 h runs.
- Use a **job array**, and **stagger** submissions, per the quoted guidance.
- Fairshare is a real cost: heavy CPU use lowers priority for *all* subsequent
  jobs, including the v100 node this project depends on.

### Untested opportunity: A100

`a100-long` (48 h, 1 node, **2 jobs/user**) exists on the milan side. A100 is
substantially faster than V100 and the per-user job allowance is 2× that of
`v100`. Requires: access from `milan1`/`milan2`, and a rebuild with
`CMAKE_CUDA_ARCHITECTURES=80-real` (our binary is sm_70-only). Not yet
investigated — likely the single biggest available speedup.

---

## Sources

- SeaWulf queues (limits table, global limits): <https://rci.stonybrook.edu/HPC/faqs/seawulf-queues>
- Fairshare and job priority: <https://rci.stonybrook.edu/HPC/docs/policies/fairshare>
- SeaWulf FAQs: <https://rci.stonybrook.edu/HPC/faqs>
- Shared queues / partial-node jobs: <https://it.stonybrook.edu/help/kb/how-to-use-seawulfs-shared-queues>
- Understanding SeaWulf: <https://rci.stonybrook.edu/HPC/understanding-SeaWulf>

Retrieved 2026-08-28. Limits change — re-check before scaling out.
