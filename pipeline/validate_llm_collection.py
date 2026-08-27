#!/usr/bin/env python3
"""Validate the Arm-B (AgenticLLMEngine) trace collection.

Sibling of validate_rule_based_collection.py, adapted to the LLM arm
(config schema from llm/townhall/runner.py, plus LLM-call quality checks
read from each run's sibling `*_trace_<ts>.json`).

  A. Completeness — exactly one final trace per (composition x seed 1..N),
     no duplicates, no strays. (`--allow-partial` downgrades missing cells
     to informational for mid-collection / pilot validation.)
  B. Schema — parses; config matches filename (seed, composition) and the
     fixed design (n_agents=6, n_rounds=8, baseline, gemma-4 Q8, cb=0.3,
     reflect policy v2); T+1 snapshots; n_rounds round records; profiles
     aligned; per-round voices/evaluations/reflections consistent.
  C. Physical validity — weights and opinions finite in [-1,1]; every
     profile carries a valid non-empty salience_prior.
  D. Config uniformity — one config signature collection-wide.
  E. LLM-call quality (per sibling trace file) — zero recorded call errors;
     max prompt+gen tokens within the 4096/slot budget (WARN > 3600, FAIL
     >= 4096); tool-loop retry (`loop_rounds`) counts; evaluate-fallback
     share (influence_likert == 50 can hide INFLUENCE_LIKERT_FALLBACK —
     WARN above threshold, indistinguishable in-record).

The server config is not recorded in trace configs (the runner only logs
the model name), so the collection's server signature is recorded here and
embedded in the manifest for provenance.

Usage:
    python pipeline/validate_llm_collection.py \
        [--traces-dir llm/traces/beta_local/arm_b_local_p6cache] \
        [--prefix arm_b_local_p6cache] [--reps 130] [--allow-partial] \
        [--logs PATH ...] [--report PATH.md] [--manifest PATH.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

COMPOSITIONS = ("polarized", "symmetric", "three_clusters")

CONFIG_VARIABLE_FIELDS = {"seed", "composition"}

# The Arm-B design (llm/townhall/runner.py config block).
REQUIRED_CONFIG = {
    "topic": "minimum_wage_seattle",
    "topic_description": "Should Seattle implement a $15/hour minimum wage?",
    "n_agents": 6,
    "n_rounds": 8,
    "condition": "baseline",
    "model": "gemma-4-E2B-it-Q8_0.gguf",
    "confirmation_bias": 0.3,
    "reflect_decision_policy": "explicit_update_or_no_update_v2",
}

# Server-side collection signature — NOT in the trace configs; recorded for
# provenance (see llm/run_arm_b_local_collection.sh and the readiness addendum).
SERVER_SIGNATURE = {
    "backend": "llama-server (Vulkan, RTX 2060 SUPER 8GB, local :20434)",
    "n_parallel": 6,
    "ctx_size": 24576,
    "ctx_per_slot": 4096,
    "cache_reuse": 256,
    "ubatch": 512,
    "flags": "--jinja --reasoning off --flash-attn on --cache-type-k/v q8_0",
}

TOKEN_WARN = 3600   # prompt+gen tokens; per-slot ctx is 4096
TOKEN_FAIL = 4096
FALLBACK_LIKERT = 50          # INFLUENCE_LIKERT_FALLBACK (llm/influence_scale.py)
FALLBACK_WARN_SHARE = 0.20

LOG_PATTERNS = (
    "[warn] voice failed",
    "[warn] eval failed",
    "HookLoopError",
    "chat got 500",
    "no progress",
)


def finite_in_unit(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and -1.0 <= x <= 1.0


def validate_record(rec: dict, comp: str, seed: int,
                    errors: list[str], warns: list[str]) -> None:
    cfg = rec.get("config", {})
    if cfg.get("seed") != seed:
        errors.append(f"config.seed={cfg.get('seed')} != filename seed {seed}")
    if cfg.get("composition") != f"{comp}_n6":
        errors.append(f"config.composition={cfg.get('composition')} != {comp}_n6")
    for k, v in REQUIRED_CONFIG.items():
        if cfg.get(k) != v:
            errors.append(f"config.{k}={cfg.get(k)!r} != required {v!r}")

    n_agents = cfg.get("n_agents", 6)
    n_rounds = cfg.get("n_rounds", 8)

    rounds = rec.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != n_rounds:
        errors.append(f"rounds len={len(rounds) if isinstance(rounds, list) else 'n/a'} != {n_rounds}")
        rounds = rounds if isinstance(rounds, list) else []
    for rr in rounds:
        rn = rr.get("round_num")
        voices = rr.get("voices") or []
        evals = rr.get("evaluations") or []
        refls = rr.get("reflections") or []
        if len(voices) > n_agents:
            errors.append(f"round {rn}: {len(voices)} voices > {n_agents}")
        if len(voices) == 0:
            warns.append(f"round {rn}: 0 voices (all agents skipped)")
        if len(evals) != (n_agents - 1) * len(voices):
            errors.append(f"round {rn}: {len(evals)} evaluations != "
                          f"{(n_agents - 1) * len(voices)} (= {n_agents - 1} x voices)")
        if len(refls) != n_agents:
            errors.append(f"round {rn}: {len(refls)} reflections != {n_agents}")
        if not rr.get("llm_calls"):
            errors.append(f"round {rn}: llm_calls={rr.get('llm_calls')} (expected > 0)")

    snaps = rec.get("snapshots")
    if not isinstance(snaps, list) or len(snaps) != n_rounds + 1:
        errors.append(f"snapshots len={len(snaps) if isinstance(snaps, list) else 'n/a'} != {n_rounds + 1}")
        return  # downstream checks need snapshots
    for r_i, snap in enumerate(snaps):
        if len(snap) != n_agents:
            errors.append(f"snapshot[{r_i}] has {len(snap)} agents != {n_agents}")
            continue
        for a in snap:
            for cid, w in a.get("weights", {}).items():
                if not finite_in_unit(w):
                    errors.append(f"snapshot[{r_i}] agent {a.get('id')} weight {cid}={w} out of [-1,1]")
                    break
            if not finite_in_unit(a.get("opinion")):
                errors.append(f"snapshot[{r_i}] agent {a.get('id')} opinion={a.get('opinion')} invalid")
        if any(not a.get("weights") for a in snap):
            errors.append(f"snapshot[{r_i}] has agent with empty repertoire")

    profiles = rec.get("profiles") or []
    if len(profiles) != n_agents:
        errors.append(f"profiles len={len(profiles)} != {n_agents}")
    for i, prof in enumerate(profiles):
        sp = (prof or {}).get("salience_prior")
        if not sp:
            errors.append(f"profile[{i}] missing/empty salience_prior")
            continue
        bad = [(c, v) for c, v in sp.items() if not finite_in_unit(v)]
        if bad:
            errors.append(f"profile[{i}] salience_prior invalid entries: {bad[:3]}")


def validate_llm_trace(trace_path: Path, errors: list[str],
                       warns: list[str]) -> dict:
    """Quality checks on the sibling LLM-call trace. Returns stats."""
    stats = {"n_calls": 0, "n_errors": 0, "max_tokens": 0, "n_loop_retries": 0,
             "fallback_share": 0.0}
    if not trace_path.exists():
        errors.append(f"missing sibling LLM trace {trace_path.name}")
        return stats
    try:
        tr = json.loads(trace_path.read_bytes())
    except json.JSONDecodeError as e:
        errors.append(f"trace JSON parse error: {e}")
        return stats
    calls = tr.get("calls") or []
    stats["n_calls"] = len(calls)
    for c in calls:
        if c.get("error") is not None:
            stats["n_errors"] += 1
        tok = (c.get("prompt_tokens") or 0) + (c.get("gen_tokens") or 0)
        stats["max_tokens"] = max(stats["max_tokens"], tok)
        if c.get("loop_rounds", 1) > 1:
            stats["n_loop_retries"] += 1
    if stats["n_errors"]:
        errors.append(f"{stats['n_errors']} LLM call(s) with recorded error")
    if stats["max_tokens"] >= TOKEN_FAIL:
        errors.append(f"max prompt+gen tokens {stats['max_tokens']} >= {TOKEN_FAIL} (per-slot ctx)")
    elif stats["max_tokens"] > TOKEN_WARN:
        warns.append(f"max prompt+gen tokens {stats['max_tokens']} > {TOKEN_WARN}")
    if not calls:
        warns.append("trace has 0 calls (resumed run? cross-check supervisor log)")
    return stats


def fallback_share(rec: dict) -> float:
    """Share of evaluate events at the fallback likert value (50)."""
    n = tot = 0
    for rr in rec.get("rounds") or []:
        for ev in rr.get("evaluations") or []:
            tot += 1
            if ev.get("influence_likert") == FALLBACK_LIKERT:
                n += 1
    return (n / tot) if tot else 0.0


def scan_logs(paths: list[Path]) -> dict[str, int]:
    counts = {p: 0 for p in LOG_PATTERNS}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        for pat in LOG_PATTERNS:
            counts[pat] += text.count(pat)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces-dir", type=Path,
                    default=BASE_DIR / "llm/traces/beta_local/arm_b_local_p6cache")
    ap.add_argument("--prefix", default="arm_b_local_p6cache")
    ap.add_argument("--reps", type=int, default=130)
    ap.add_argument("--allow-partial", action="store_true",
                    help="Missing cells are informational (pilot / mid-collection).")
    ap.add_argument("--logs", type=Path, nargs="*", default=[],
                    help="Supervisor/sweep logs to scan for warn patterns.")
    ap.add_argument("--report", type=Path,
                    default=BASE_DIR / "pipeline/output/llm_collection/validation-report.md")
    ap.add_argument("--manifest", type=Path,
                    default=BASE_DIR / "pipeline/output/llm_collection/manifest.json")
    args = ap.parse_args()

    name_re = re.compile(
        rf"townhall_minimum_wage_seattle_{re.escape(args.prefix)}_baseline_"
        rf"(?P<comp>polarized|symmetric|three_clusters)_n6_s(?P<seed>\d+)_(?P<ts>\d+)\.json$"
    )

    finals: dict[tuple[str, int], list[Path]] = defaultdict(list)
    strays: list[str] = []
    n_checkpoints = n_traces = 0
    if not args.traces_dir.exists():
        print(f"traces dir {args.traces_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    for p in sorted(args.traces_dir.iterdir()):
        if p.name.endswith("_checkpoint.json"):
            n_checkpoints += 1
            continue
        if "_trace_" in p.name:  # *_trace_<ts>.json and *_trace_live.json
            n_traces += 1
            continue
        m = name_re.match(p.name)
        if not m:
            strays.append(p.name)
            continue
        finals[(m.group("comp"), int(m.group("seed")))].append(p)

    # A. Completeness
    expected = {(c, s) for c in COMPOSITIONS for s in range(1, args.reps + 1)}
    missing = sorted(expected - set(finals))
    extras = sorted(set(finals) - expected)
    dupes = sorted(k for k, v in finals.items() if len(v) > 1)

    # B/C/D/E per-file
    per_file_errors: dict[str, list[str]] = {}
    per_file_warns: dict[str, list[str]] = {}
    manifest_entries: list[dict] = []
    config_signatures: dict[str, list[str]] = defaultdict(list)
    quality_totals = {"n_calls": 0, "n_errors": 0, "max_tokens": 0, "n_loop_retries": 0}
    high_fallback: list[str] = []
    mean_abs_shifts: list[float] = []
    for (comp, seed), paths in sorted(finals.items()):
        for p in paths:
            raw = p.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            errs: list[str] = []
            warns: list[str] = []
            stats: dict = {}
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                errs.append(f"JSON parse error: {e}")
                rec = None
            if rec is not None:
                validate_record(rec, comp, seed, errs, warns)
                ts = name_re.match(p.name).group("ts")  # type: ignore[union-attr]
                trace_name = p.name.replace(f"_{ts}.json", f"_trace_{ts}.json")
                stats = validate_llm_trace(p.parent / trace_name, errs, warns)
                stats["fallback_share"] = round(fallback_share(rec), 4)
                if stats["fallback_share"] > FALLBACK_WARN_SHARE:
                    warns.append(f"evaluate fallback-likert share {stats['fallback_share']:.0%} "
                                 f"> {FALLBACK_WARN_SHARE:.0%}")
                    high_fallback.append(p.name)
                for k in ("n_calls", "n_errors", "n_loop_retries"):
                    quality_totals[k] += stats.get(k, 0)
                quality_totals["max_tokens"] = max(quality_totals["max_tokens"],
                                                   stats.get("max_tokens", 0))
                if isinstance(rec.get("summary"), dict):
                    v = rec["summary"].get("mean_abs_shift")
                    if isinstance(v, (int, float)):
                        mean_abs_shifts.append(v)
                sig_cfg = {k: v for k, v in rec.get("config", {}).items()
                           if k not in CONFIG_VARIABLE_FIELDS}
                config_signatures[json.dumps(sig_cfg, sort_keys=True)].append(p.name)
            if errs:
                per_file_errors[p.name] = errs
            if warns:
                per_file_warns[p.name] = warns
            manifest_entries.append({
                "file": p.name, "sha256": sha, "bytes": len(raw),
                "composition": comp, "seed": seed,
                "valid": not errs, **{f"q_{k}": v for k, v in stats.items()},
            })

    log_counts = scan_logs(list(args.logs))

    n_files = sum(len(v) for v in finals.values())
    completeness_ok = not missing or args.allow_partial
    ok = bool(finals) and completeness_ok \
        and not (extras or dupes or strays or per_file_errors) \
        and len(config_signatures) == 1

    # ---- report ----
    verdict = "PASS" if ok else "FAIL"
    if ok and args.allow_partial and missing:
        verdict = "PASS (partial)"
    lines = [
        "# Arm-B (AgenticLLMEngine) collection — validation report",
        "",
        f"Traces dir: `{args.traces_dir}`  ·  design: {args.reps} seeds × "
        f"{len(COMPOSITIONS)} compositions (n=6, t=8, baseline, AgenticLLMEngine, "
        f"{REQUIRED_CONFIG['model']})",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| A. Completeness | {n_files}/{len(expected)} finals; missing={len(missing)}"
        f"{' (allowed: partial)' if args.allow_partial else ''}, extras={len(extras)}, "
        f"duplicates={len(dupes)}, strays={len(strays)}; checkpoints={n_checkpoints}, "
        f"llm-traces={n_traces} |",
        f"| B/C. Schema + physical validity | {len(per_file_errors)} file(s) with errors |",
        f"| D. Config uniformity | {len(config_signatures)} distinct config signature(s) (must be 1) |",
        f"| E. LLM-call quality | {quality_totals['n_calls']} calls, "
        f"{quality_totals['n_errors']} errors, max prompt+gen={quality_totals['max_tokens']} tok "
        f"(fail ≥{TOKEN_FAIL}), {quality_totals['n_loop_retries']} tool-loop retries, "
        f"{len(high_fallback)} file(s) over fallback-share warn |",
    ]
    if mean_abs_shifts:
        nonzero = sum(1 for v in mean_abs_shifts if v > 0)
        lines.append(
            f"| Opinion movement (info) | mean of mean_abs_shift = "
            f"{sum(mean_abs_shifts)/len(mean_abs_shifts):.4f}; "
            f"{nonzero}/{len(mean_abs_shifts)} runs with movement > 0 |")
    if args.logs:
        lines.append("| Log scan | " + "; ".join(
            f"`{k}`×{v}" for k, v in log_counts.items()) + " |")
    lines.append("")
    if missing:
        lines += ["### Missing cells", "", ", ".join(f"{c}/s{s}" for c, s in missing[:60]),
                  f"... ({len(missing)} total)" if len(missing) > 60 else "", ""]
    if extras:
        lines += ["### Unexpected cells", "", ", ".join(f"{c}/s{s}" for c, s in extras[:50]), ""]
    if dupes:
        lines += ["### Duplicate finals", "", ", ".join(f"{c}/s{s}" for c, s in dupes[:50]), ""]
    if strays:
        lines += ["### Stray files", "", "\n".join(strays[:50]), ""]
    if per_file_errors:
        lines += ["### Per-file errors", ""]
        for name, errs in list(per_file_errors.items())[:40]:
            lines.append(f"- `{name}`")
            lines += [f"  - {e}" for e in errs[:5]]
        lines.append("")
    if per_file_warns:
        lines += ["### Per-file warnings (non-blocking)", ""]
        for name, warns in list(per_file_warns.items())[:40]:
            lines.append(f"- `{name}`")
            lines += [f"  - {w}" for w in warns[:5]]
        lines.append("")
    if len(config_signatures) > 1:
        lines += ["### Config signature split", ""]
        for sig, names in config_signatures.items():
            lines.append(f"- {len(names)} file(s): first `{names[0]}`")
            lines.append(f"  `{sig[:300]}`")
        lines.append("")
    if len(config_signatures) == 1:
        sig = next(iter(config_signatures))
        lines += ["### Uniform engine config (collection-wide)", "", "```json",
                  json.dumps(json.loads(sig), indent=2), "```", ""]
    lines += ["### Recorded server signature (external provenance)", "", "```json",
              json.dumps(SERVER_SIGNATURE, indent=2), "```", ""]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "traces_dir": str(args.traces_dir),
        "n_finals": n_files, "n_checkpoints": n_checkpoints, "n_llm_traces": n_traces,
        "design": {"reps": args.reps, "compositions": list(COMPOSITIONS),
                   **REQUIRED_CONFIG},
        "server_signature": SERVER_SIGNATURE,
        "verdict": verdict,
        "log_scan": {k: v for k, v in log_counts.items()} if args.logs else None,
        "files": manifest_entries,
    }, indent=1))
    print("\n".join(lines[:24]))
    print(f"\nwrote {args.report}\nwrote {args.manifest}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
