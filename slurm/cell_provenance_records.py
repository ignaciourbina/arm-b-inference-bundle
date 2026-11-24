#!/usr/bin/env python3
"""One provenance record per SeaWulf collection cell (written 2026-11-23).

Writes archives/provenance/<cell>.json and archives/provenance/CELLS.md.

Every field is tagged with how it is known:
  recorded      written at run time: by the traces' `config`, the orchestrator
                telemetry (llm/traces/logs/<cell>_progress.jsonl), or the SLURM /
                server log of the job
  reconstructed worked out afterwards, from log file names, sbatch scripts, file
                mtimes and git history. Treat these as best evidence, not fact.
  unknown       no evidence survives

Why so much is reconstructed: the V100 jobs ran on the old SeaWulf scheduler
(login1/2, Slurm 17.11). Its accounting is not reachable from milan1/2, and its
job IDs collide with unrelated seawulf3 jobs, so `sacct -j 79595` here returns
someone else's job. Server logs carry no llama.cpp build stamp, and runs never
recorded the repo commit.
"""
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "slurm"))
from provenance import fingerprint  # noqa: E402

COLL = f"{ROOT}/llm/traces/collection"
LOGS = f"{ROOT}/llm/traces/logs"
OUT = f"{ROOT}/archives/provenance"
FINAL_RE = re.compile(r"_(polarized|symmetric|three_clusters)_n6_s(\d+)_\d+\.json$")

# llama.cpp: one checkout (18443257a30c, 2026-08-27) was built three ways. The
# build used is the PATH the sbatch exports; build dates are from build logs.
BUILDS = {
    "build":       {"cuda_arch": "70-real",         "build_job": "79441", "built": "2026-08-27"},
    "build-multi": {"cuda_arch": "60-real;70-real", "build_job": "79721", "built": "2026-08-28"},
    "build-a100":  {"cuda_arch": "80-real",         "build_job": "2434147", "built": "2026-11-17"},
}
LLAMA_COMMIT = "18443257a30c884d5332abb8e7dc43c7ffe42fda"
COMMON_FLAGS = ("-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON, built inside "
                "cuda124-devel.sif via Singularity; see "
                "$SCRATCH/build_llama*.sbatch (archived in slurm_out)")
MODEL_SHA256 = {
    "gemma-4-E2B-it-Q8_0.gguf": "996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a",
    "gemma-4-E2B-it-Q4_0.gguf": "8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52",
    "gemma-4-E4B-it-Q8_0.gguf": "34be82b17b4942d389b9b527170c4b058027abdd32531fda063d3d97dd8ce80a",
    "gemma-4-26B-A4B-it-Q8_0.gguf": "b5108fd13147d1c866bb595295bc9d56f5fe744d7209f18421031d0cc47009c6",
}

# cell -> (sbatch, [jobs], scratch log prefix, gpu, node, note). Reconstructed
# from the job IDs in llm/traces/logs/*_<job>.log and $SCRATCH/<prefix>_<job>.log,
# and from which sbatch passes that --cell. Node is printed by only some logs.
V100 = "Tesla V100-PCIE-32GB"
JOBS = {
    "seawulf_test_l4": ("armb_collection_l4.sbatch, armb_collection_2gpu.sbatch, armb_finish_one.sbatch, armb_fix_s8.sbatch, armb_fix_s8_p24.sbatch",
                        ["79464", "79476", "79484", "79485", "79486"], "armb_l4/armb_2gpu/armb_finish/armb_s8/armb_s8b", V100, "unknown",
                        "seed 8 was re-collected by the fix_s8 jobs; one stray seed-8 final also sits at $SCRATCH root"),
    "seawulf_test_p6cache": ("armb_collection.sbatch", ["79448", "79449", "79452", "79454"], "armb_test30", V100, "unknown",
                             "lives in llm/traces/beta_local/, not collection/"),
    "lanesweep_solo": ("lane_sweep.sbatch", ["79460"], "lane_sweep", V100, "sn-nvda2",
                       "solo 1-lane baseline (seed 900, 4 rounds, reasoning off) of the lane sweep; job cancelled before its first run finished"),
    "sota_r256_t10": ("sota_pilot.sbatch", ["79512"], "sota_pilot", V100, "unknown", "GPU 0 of the pilot"),
    "sota_r256_t03": ("sota_pilot.sbatch", ["79512"], "sota_pilot", V100, "unknown", "GPU 1 of the pilot"),
    **{c: ("scaling_law.sbatch", ["79537"], "scaling_law", V100, "unknown", "")
       for c in ["sl_lanes_L1", "sl_lanes_L2", "sl_lanes_L4", "sl_lanes_L6",
                 "sl_slots_p6", "sl_slots_p12", "sl_slots_p24", "sl_slots_p48"]},
    "vo_L4_p6": ("verify_optimum.sbatch", ["79544"], "verify_opt", V100, "unknown", ""),
    "vo_L6_p6": ("verify_optimum.sbatch", ["79544"], "verify_opt", V100, "unknown", ""),
    "abl_off_t10": ("abl_off_t10.sbatch", ["79595"], "abl_off_t10", V100, "sn-nvda2",
                    "reasoning off; cancelled 2026-08-28T14:07:52 when reasoning-on was locked"),
    "armb_sota_r256_t10": ("armb_sota_collection.sbatch", ["79645", "79894"], "armb_sota", V100, "sn-nvda2", "2-link chain"),
    "abl_q8_b128": ("abl_quant_budget.sbatch", ["79783"], "abl_qb", "Tesla P100-PCIE-16GB", "cn-nvidia", "p100 partition, not v100"),
    "abl_q4_b256": ("abl_quant_budget.sbatch", ["79783"], "abl_qb", "Tesla P100-PCIE-16GB", "cn-nvidia", "p100 partition, not v100"),
    **{c: ("axes_cd.sbatch", ["79903"], "axes_cd", V100, "unknown", "")
       for c in ["cd_cache_0", "cd_cache_256", "cd_cache_1024", "cd_cache_4096",
                 "cd_slots_p6", "cd_slots_p12", "cd_slots_p24"]},
    "ms_E4B": ("model_scale_pilot.sbatch", ["80086", "80088", "80089"], "model_scale", V100, "unknown", ""),
    "ms_26B_a4b": ("model_scale_pilot.sbatch", ["80086", "80088", "80089"], "model_scale", V100, "unknown", ""),
    "ms2_E4B": ("model_scale_rerun.sbatch", ["80172"], "ms_rerun", V100, "unknown", ""),
    "ms2_26B_a4b": ("model_scale_rerun.sbatch", ["80172"], "ms_rerun", V100, "unknown", ""),
    "e4b_r256_t10": ("e4b_collection_a100.sbatch", ["2433689"], "e4b_a100", "NVIDIA A100 80GB PCIe", "a100-02",
                     "V100 attempt (e4b_collection.sbatch, jobs 80816/81118/81119/81122, 2026-08-31) produced 0 runs: "
                     "every link aborted on the disk-space guard. All 300 runs come from A100 job 2433689"),
}
BUILD_OF_SBATCH = {}
for f in glob.glob(f"{ROOT}/slurm/*.sbatch"):
    m = re.search(r"llama\.cpp/(build[\w-]*)", open(f).read())
    BUILD_OF_SBATCH[os.path.basename(f)] = m.group(1) if m else None


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()


def cell_dir(cell):
    d = f"{COLL}/{cell}"
    return d if os.path.isdir(d) else f"{ROOT}/llm/traces/beta_local/{cell}"


def record(cell):
    d = cell_dir(cell)
    files = os.listdir(d)
    finals = [f for f in files if FINAL_RE.search(f)]
    pairs = defaultdict(list)
    fps = []
    for f in finals:
        m = FINAL_RE.search(f)
        pairs[m.group(1)].append(int(m.group(2)))
        fps.append(fingerprint(json.load(open(os.path.join(d, f)))))
    cfg = {}
    for k in sorted({k for fp in fps for k in fp}):
        vals = Counter(str(fp.get(k)) for fp in fps)
        cfg[k] = vals.most_common(1)[0][0] if len(vals) == 1 else dict(vals)

    ev = []
    pj = f"{LOGS}/{cell}_progress.jsonl"
    if os.path.exists(pj):
        ev = [json.loads(l) for l in open(pj) if l.strip()]
    ts = [e["ts"] for e in ev if "ts" in e]
    evc = Counter(e["event"] for e in ev)

    sb, jobs, prefix, gpu, node, note = JOBS.get(cell, ("unknown", [], "", "unknown", "unknown", ""))
    first_sb = sb.split(",")[0].strip()
    build = BUILD_OF_SBATCH.get(first_sb)
    launch_commit = git("log", "-1", "--format=%h %cI", f"--before={iso(min(ts))}") if ts else "unknown"
    # The server-reported model_path is what was loaded; config.model is the
    # orchestrator's label and can lag it (abl_q4_b256: label Q8_0, loaded Q4_0).
    model = cfg.get("model_path") if cfg.get("model_path") not in (None, "(unrecorded)") else cfg.get("model")
    srv_logs = sorted(os.path.basename(p) for j in jobs for p in glob.glob(f"{LOGS}/*_{j}.log"))

    R, C, U = "recorded", "reconstructed", "unknown"
    return {
        "cell": cell,
        "trace_dir": os.path.relpath(d, ROOT),
        "counts": {"value": {"finals": len(finals),
                             "checkpoints": sum("_checkpoint" in f for f in files),
                             "call_traces": sum("_trace_" in f for f in files)}, "how": R},
        "grid": {"value": {c: f"{len(s)} seeds {min(s)}-{max(s)}" for c, s in sorted(pairs.items())}, "how": R},
        "sbatch": {"value": sb, "how": C},
        "launch_repo_commit": {"value": launch_commit if ts else U, "how": C if ts else U,
                               "note": "last commit before the first telemetry event; the working tree may have been dirty"},
        "slurm_jobs": {"value": jobs, "how": C},
        "times_utc": {"value": {"first_event": iso(min(ts)), "last_event": iso(max(ts))} if ts else U,
                      "how": R if ts else U,
                      "note": "orchestrator telemetry; SLURM submit/start/end are in sacct only for e4b_r256_t10"},
        "sacct": {"value": SACCT.get(cell, U), "how": R if cell in SACCT else U},
        "node": {"value": node, "how": R if node != U else U},
        "gpu": {"value": cfg.get("infra.gpu", gpu), "how": R if "infra.gpu" in cfg or gpu != U else U},
        "llama_cpp": {"value": {"commit": LLAMA_COMMIT, "build_dir": build, **BUILDS.get(build or "", {}),
                                "flags": COMMON_FLAGS}, "how": C},
        "model": {"value": {"file": model, "sha256": MODEL_SHA256.get(model, U)}, "how": R,
                  "note": "file name recorded at run time (server model_path); sha256 measured 2026-11-23 on the file at the same path"
                  + ("" if model == cfg.get("model") else f"; config.model label says {cfg.get('model')}, the server loaded {model}")},
        "server": {"value": {k: cfg.get(k) for k in ("total_slots", "n_ctx_slot", "infra.server", "infra.budget",
                                                       "infra.reasoning", "infra.lanes", "infra.quant") if k in cfg},
                   "how": R, "note": "absent keys predate ablation_infra; see the sbatch for the launch line"},
        "sampling": {"value": {"temperature": cfg.get("temperature"), "harness_profile": cfg.get("harness_profile"),
                               "prompt_variant": cfg.get("prompt_variant"),
                               "reflect_decision_policy": cfg.get("reflect_decision_policy")}, "how": R},
        "telemetry_events": {"value": dict(evc), "how": R if ev else U},
        "logs": {"value": {"server_logs": srv_logs,
                           "slurm_out": [f"{p}_{j}.log" for j in jobs for p in prefix.split("/")
                                         if os.path.exists(f"/gpfs/scratch/{os.environ['USER']}/{p}_{j}.log")],
                           "archived_in": "archives/traces/aux_llm_traces_logs.tar.zst, aux_slurm_out.tar.zst"}, "how": R},
        "note": note,
        "config_all_fields": cfg,
    }


SACCT = {}
try:
    out = subprocess.run(["sacct", "-j", "2433689", "-X", "-n", "-P", "-o",
                          "jobid,jobname,partition,submit,start,end,elapsed,state,exitcode,nodelist,alloctres"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        SACCT["e4b_r256_t10"] = out
except FileNotFoundError:
    pass

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cells = sorted(os.listdir(COLL)) + ["seawulf_test_p6cache"]
    rows = []
    for c in cells:
        r = record(c)
        json.dump(r, open(f"{OUT}/{c}.json", "w"), indent=1, sort_keys=False)
        t = r["times_utc"]["value"]
        rows.append(f"| {c} | {r['counts']['value']['finals']} | {r['sbatch']['value'].split(',')[0]} | "
                    f"{', '.join(r['slurm_jobs']['value'])} | {t['first_event'][:10] if t != 'unknown' else 'unknown'} | "
                    f"{r['gpu']['value']} | {r['model']['value']['file']} | {r['llama_cpp']['value']['build_dir']} |")
    with open(f"{OUT}/CELLS.md", "w") as f:
        f.write("# Per-cell provenance\n\nGenerated by `slurm/cell_provenance_records.py`. One JSON per cell "
                "beside this file; each field says whether it was recorded at run time or reconstructed "
                "afterwards.\n\n| cell | finals | sbatch | SLURM jobs | first event | GPU | model | build |\n"
                "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    print(f"{len(cells)} records -> {os.path.relpath(OUT, ROOT)}")
