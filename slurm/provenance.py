#!/usr/bin/env python3
"""Config provenance: is a cell internally consistent, and is a comparison legitimate?

The expensive failures on this project were not crashes but *invalid
comparisons that looked valid*: one collection had seed 8 collected at p6 while
the other 29 ran at p24, and nothing flagged it. This makes that question a
command instead of a judgement call.

Two modes:

  provenance.py CELL...            fingerprint each cell; FAIL if the runs
                                   inside it disagree on anything but seed and
                                   composition; report grid coverage
  provenance.py --compare A B      what differs between two cells, and whether
                                   their seeds actually overlap enough to pair

What the traces do and do not record. `config` carries the model, temperature,
prompt variant and decision policy; `config.server_props` carries total_slots,
n_ctx_slot and the resolved model path. Lane count, cache-reuse, reasoning
budget and GPU model appear ONLY in `config.ablation_infra`, which was added
partway through the project -- so for the sl_*, cd_* and vo_* families those
live only in the sbatch script and the directory name. Fields that are simply
absent are reported as `(unrecorded)`, never silently treated as equal.
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "llm/traces/collection")
FINAL_RE = re.compile(r"_(?P<comp>polarized|symmetric|three_clusters)_n6_s(?P<seed>\d+)_\d+\.json$")

# Fields that MUST be identical across every run in a cell. seed and
# composition are the two that are supposed to vary.
CONFIG_FIELDS = ("model", "temperature", "n_agents", "n_rounds", "condition",
                 "prompt_variant", "reflect_decision_policy",
                 "confirmation_bias", "ablation_infra")  # infra is unpacked, not reported raw
SERVER_FIELDS = ("total_slots", "n_ctx_slot", "model_path")
MISSING = "(unrecorded)"

# Several fields re-encode the same fact: temperature also appears inside
# ablation_infra, the model also appears as model_path. Declaring TREATMENT=
# "temperature" should silence all of its spellings, otherwise the tool nags
# about the very thing you just declared and gets ignored.
ALIASES = {
    "temperature": ("infra.temp",),
    "model": ("model_path", "infra.model"),
    "total_slots": ("infra.server",),
    "n_ctx_slot": ("infra.server",),
}


def fingerprint(rec):
    cfg = rec.get("config", {})
    sp = cfg.get("server_props") or {}
    fp = {k: cfg.get(k, MISSING) for k in CONFIG_FIELDS}
    fp.update({k: sp.get(k, MISSING) for k in SERVER_FIELDS})
    if fp.get("model_path") not in (MISSING, None):
        fp["model_path"] = os.path.basename(str(fp["model_path"]))
    # ablation_infra is a semicolon k=v string; unpack so a single differing
    # key (lanes, budget, gpu) shows up as that key rather than one opaque blob.
    infra = fp.pop("ablation_infra", "") or ""
    for part in str(infra).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            fp[f"infra.{k.strip()}"] = v.strip()
    return fp


def scan(cell):
    runs = {}
    for f in glob.glob(f"{ROOT}/{cell}/townhall_*.json"):
        if "_checkpoint" in f or "_trace_" in f:
            continue
        m = FINAL_RE.search(os.path.basename(f))
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        key = (m.group("comp"), int(m.group("seed"))) if m else (os.path.basename(f), -1)
        runs[key] = fingerprint(rec)
    return runs


def report_cell(cell):
    runs = scan(cell)
    if not runs:
        print(f"=== {cell}: no finals ===\n")
        return True
    fields = sorted({k for fp in runs.values() for k in fp})
    split = {}
    for k in fields:
        vals = Counter(str(fp.get(k, MISSING)) for fp in runs.values())
        if len(vals) > 1:
            split[k] = vals
    ok = not split
    print(f"=== {cell}  ({len(runs)} runs) — {'UNIFORM' if ok else 'NOT UNIFORM'} ===")
    for k in fields:
        if k not in split:
            print(f"  {k:26} {list(runs.values())[0].get(k, MISSING)}")
    if split:
        print("  --- fields that DISAGREE across runs in this cell ---")
        for k, vals in split.items():
            print(f"  {k:26} " + "  ".join(f"{v!r}x{n}" for v, n in vals.most_common()))
        print("  A cell whose runs disagree is not one experimental condition.")

    comps = defaultdict(set)
    for comp, seed in runs:
        comps[comp].add(seed)
    print(f"  --- grid ---")
    for comp in sorted(comps):
        seeds = sorted(comps[comp])
        gaps = ""
        if seeds and seeds[-1] - seeds[0] + 1 != len(seeds):
            missing = sorted(set(range(seeds[0], seeds[-1] + 1)) - set(seeds))
            gaps = f"  gaps: {missing[:8]}{'...' if len(missing) > 8 else ''}"
        print(f"  {comp:26} {len(seeds)} seeds [{seeds[0]}..{seeds[-1]}]{gaps}")
    sizes = {len(v) for v in comps.values()}
    if len(sizes) > 1:
        print("  compositions have UNEQUAL seed counts — the design is unbalanced.")
    print()
    return ok


def report_compare(a, b):
    ra, rb = scan(a), scan(b)
    if not ra or not rb:
        print(f"cannot compare: {a if not ra else b} has no finals")
        return False
    fa, fb = list(ra.values())[0], list(rb.values())[0]
    fields = sorted(set(fa) | set(fb))
    diff = [k for k in fields if str(fa.get(k, MISSING)) != str(fb.get(k, MISSING))]
    w = max(len(a), len(b), 24)
    print(f"=== {a}  vs  {b} ===")
    print(f"  {'field':26} {a:<{w}}  {b}")
    for k in fields:
        va, vb = str(fa.get(k, MISSING)), str(fb.get(k, MISSING))
        mark = "  <-- DIFFERS" if k in diff else ""
        print(f"  {k:26} {va:<{w}}  {vb}{mark}")

    sa, sb = set(ra), set(rb)
    both = sa & sb
    print(f"\n  --- pairing ---")
    print(f"  {a}: {len(sa)} runs · {b}: {len(sb)} runs · shared (comp,seed): {len(both)}")
    if not both:
        print("  NO SHARED CELLS — a paired comparison is impossible; any 'paired'")
        print("  score against these two is really unpaired.")
    elif len(both) < min(len(sa), len(sb)):
        print(f"  only {len(both)} of {min(len(sa), len(sb))} can pair; the rest are dropped.")

    treatment = [t.strip() for t in os.environ.get("TREATMENT", "").split(",") if t.strip()]
    declared = set(treatment)
    for t_ in treatment:
        declared.update(ALIASES.get(t_, ()))
    confounds = [k for k in diff if k not in declared]
    print(f"\n  --- verdict ---")
    if treatment:
        print(f"  declared treatment: {', '.join(treatment)}")
    if confounds:
        print(f"  {len(confounds)} field(s) differ beyond the declared treatment:")
        for k in confounds:
            print(f"    {k}: {fa.get(k, MISSING)!r} vs {fb.get(k, MISSING)!r}")
        print("  Each is a potential confound. Declare intended ones with")
        print("  TREATMENT=\"model,infra.quant\" to silence them deliberately.")
    else:
        print("  no differences beyond the declared treatment.")
    print("\n  Note: a field reading (unrecorded) is NOT evidence of sameness.")
    print("  Lanes, cache-reuse, budget and GPU are only in ablation_infra, which")
    print("  the sl_*/cd_*/vo_* cells predate — check their sbatch scripts.")
    return not confounds


def main():
    args = sys.argv[1:]
    if args and args[0] == "--compare":
        if len(args) != 3:
            print("usage: provenance.py --compare CELL_A CELL_B")
            return 2
        return 0 if report_compare(args[1], args[2]) else 1
    cells = args or sorted(os.path.basename(p) for p in glob.glob(f"{ROOT}/*")
                           if os.path.isdir(p))
    return 0 if all([report_cell(c) for c in cells]) else 1


if __name__ == "__main__":
    sys.exit(main())
