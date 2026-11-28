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

# The shared deliberation design — identical across arms by construction.
# `model` is NOT here: it is the one required field that legitimately differs
# between arms, so it is supplied per-run from --expect-model (Sprint 17).
REQUIRED_CONFIG_BASE = {
    "topic": "minimum_wage_seattle",
    "topic_description": "Should Seattle implement a $15/hour minimum wage?",
    "n_agents": 6,
    "n_rounds": 8,
    "condition": "baseline",
    "confirmation_bias": 0.3,
    "reflect_decision_policy": "explicit_update_or_no_update_v2",
}

# Arm-B's model, and the --expect-model default, so existing invocations keep
# validating exactly as before.
DEFAULT_EXPECT_MODEL = "gemma-4-E2B-it-Q8_0.gguf"


def required_config(expect_model: str) -> dict:
    return {**REQUIRED_CONFIG_BASE, "model": expect_model}


# Server-side collection signature — NOT in the trace configs; recorded for
# provenance (see llm/run_arm_b_local_collection.sh and the readiness addendum).
SERVER_SIGNATURES = {
    "local-llama": {
        "backend": "llama-server (Vulkan, RTX 2060 SUPER 8GB, local :20434)",
        "n_parallel": 6,
        "ctx_size": 24576,
        "ctx_per_slot": 4096,
        "cache_reuse": 256,
        "ubatch": 512,
        "flags": "--jinja --reasoning off --flash-attn on --cache-type-k/v q8_0",
    },
    # Arm C (Sprint 17). A managed endpoint exposes no serving internals, so
    # this records what is actually pinnable: the API surface and the region.
    "bedrock": {
        "backend": "AWS Bedrock Converse (managed, on-demand)",
        "api": "bedrock-runtime.converse",
        "region": "us-east-1",
        "tool_transport": "native toolConfig / toolUse blocks",
        "flags": "temperature=0.3, maxTokens>=2048 when tools present",
    },
    # B2 (ledger RUNS.md): SeaWulf V100 collection armb_sota_r256_t10, read
    # from the traces' ablation_infra / server_props (Sprint 20 validation).
    "seawulf-v100-r256-t10": {
        "backend": "llama-server on SeaWulf V100 (arm-b-inference-bundle)",
        "n_parallel": 6,
        "ctx_size": 24576,
        "ctx_per_slot": 4096,
        "lanes": 4,
        "flags": "reasoning=on;budget=256;quant=Q8_0;temp=1.0",
    },
    # e4b_r256_t10 (ledger RUNS.md): SeaWulf A100 collection, same server
    # signature as B2 on different hardware (slurm/e4b_collection_a100.sbatch).
    "seawulf-a100-e4b-r256-t10": {
        "backend": "llama-server on SeaWulf 2x A100-80GB (arm-b-inference-bundle)",
        "n_parallel": 6,
        "ctx_size": 24576,
        "ctx_per_slot": 4096,
        "lanes": 4,
        "flags": "reasoning=on;budget=256;quant=Q8_0;temp=1.0;harness=format",
    },
    # armb_sota_r256_t10_a100: B2 seeds 101-200, same server signature as
    # B2 on A100 hardware (slurm/b2_collection_a100.sbatch, job 2456679).
    "seawulf-a100-b2-r256-t10": {
        "backend": "llama-server on SeaWulf 2x A100-80GB (arm-b-inference-bundle)",
        "n_parallel": 6,
        "ctx_size": 24576,
        "ctx_per_slot": 4096,
        "lanes": 4,
        "flags": "reasoning=on;budget=256;quant=Q8_0;temp=1.0;harness=strict",
    },
    # B3 (ledger RUNS.md): desktop Sprint-16 collection_main, refreshed
    # (Jul-15) E2B weights, 12 slots; reasoning on (budget not stamped).
    "local-llama-r0715-p12": {
        "backend": "llama-server (Vulkan, RTX 2060 SUPER 8GB, local), refreshed r0715 weights",
        "n_parallel": 12,
        "ctx_per_slot": 4096,
        "flags": "reasoning on (budget not stamped in traces; Sprint 16 plan: 128|256)",
    },
    # Sprint 21 demo: OpenAI gpt-6-luna through llm/openai_adapter.py
    # (llm/run_openai_cloud_collection.sh). A managed endpoint: what is
    # pinnable is the API surface and the request settings.
    "openai-cloud-luna": {
        "backend": "OpenAI Chat Completions (api.openai.com), default tier",
        "model": "gpt-6-luna",
        "flags": "reasoning_effort=none, temperature=0.3, strict tools "
                 "(optional fields required-but-nullable), max_completion_tokens=512",
    },
    # Arm D (Sprint 19) on the NUC's Intel Arc iGPU. These are the launcher
    # defaults (llm/run_arm_d_local_collection.sh, per MODEL_PROFILE); confirm
    # or update n_parallel / ctx_per_slot / max_tokens / tool_call_format after
    # QA, and pass --token-warn/--token-fail to match the per-slot context.
    "k2-horizon-local-nuc": {
        "backend": "llama-server (Vulkan, Intel Arc iGPU MTL, NUC, local :20434)",
        "model": "IFM/K2-Horizon-7B-GGUF Q4_K_M @ 7a75dbd",
        "n_parallel": 2,
        "ctx_per_slot": 8192,
        "cache_reuse": 256,
        "ubatch": 512,
        "max_tokens": 2048,
        "flags": "--jinja --chat-template-kwargs reasoning_effort=low,tool_call_format=xml "
                 "--flash-attn on --cache-type-k/v q8_0",
    },
    "inkling-local-nuc": {
        "backend": "llama-server (Vulkan, Intel Arc iGPU MTL, NUC, local :20434)",
        "model": "ggml-org/Inkling-2-Mini-GGUF IQ4_NL @ e8a2cec",
        "n_parallel": 2,
        "ctx_per_slot": 8192,
        "cache_reuse": 256,
        "ubatch": 512,
        "max_tokens": 2048,
        "flags": "--jinja --chat-template-kwargs reasoning_effort=low "
                 "--flash-attn on --cache-type-k/v q8_0",
    },
}

# Arm-B ceilings: the LOCAL llama-server per-slot context is 4096, so a run
# approaching it is losing history. A managed endpoint has its own, far
# larger window, so these are overridable per arm (--token-warn/--token-fail)
# rather than global truths. Defaults keep Arm-B validation identical.
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


def recorded_prompt_variant(cfg: dict) -> str:
    """The prompt variant a run was produced under.

    The field was added in Sprint 15; the July Arm-B collection predates it and
    ran the production prompt, so a missing key means `control`.
    """
    return cfg.get("prompt_variant", "control")


def validate_record(rec: dict, comp: str, seed: int,
                    errors: list[str], warns: list[str],
                    required_cfg: dict | None = None,
                    expect_prompt_variant: str = "control") -> None:
    cfg = rec.get("config", {})
    if cfg.get("seed") != seed:
        errors.append(f"config.seed={cfg.get('seed')} != filename seed {seed}")
    if cfg.get("composition") != f"{comp}_n6":
        errors.append(f"config.composition={cfg.get('composition')} != {comp}_n6")
    for k, v in (required_cfg or required_config(DEFAULT_EXPECT_MODEL)).items():
        if cfg.get(k) != v:
            errors.append(f"config.{k}={cfg.get(k)!r} != required {v!r}")
    variant = recorded_prompt_variant(cfg)
    if variant != expect_prompt_variant:
        errors.append(f"config.prompt_variant={variant!r} != expected "
                      f"{expect_prompt_variant!r}")

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
        empty_ids = [a.get("id") for a in snap if not a.get("weights")]
        if empty_ids:
            # A repertoire that empties and then refills is a transient state,
            # not a corrupted record: the agent rejected every consideration it
            # held that round and re-adopted later, and its opinion stays
            # finite throughout. A repertoire still empty in the FINAL snapshot
            # is fatal, because the run's end-state opinion is then undefined.
            # Observed in Arm C (GLM-4.7-Flash) in 8/450 runs, always transient
            # and never at the final snapshot; never observed in Arm B.
            final_snap = snaps[-1]
            still_empty = [a.get("id") for a in final_snap
                           if a.get("id") in empty_ids and not a.get("weights")]
            if still_empty:
                errors.append(f"snapshot[{r_i}] agent(s) {still_empty} have an "
                              f"empty repertoire that is STILL empty at the "
                              f"final snapshot")
            else:
                warns.append(f"snapshot[{r_i}] agent(s) {empty_ids} transiently "
                             f"emptied their repertoire (refilled by the end)")

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
                       warns: list[str], token_warn: int = TOKEN_WARN,
                       token_fail: int = TOKEN_FAIL) -> dict:
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
    if stats["max_tokens"] >= token_fail:
        errors.append(f"max prompt+gen tokens {stats['max_tokens']} >= {token_fail} (ctx ceiling)")
    elif stats["max_tokens"] > token_warn:
        warns.append(f"max prompt+gen tokens {stats['max_tokens']} > {token_warn}")
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
    ap.add_argument("--expect-model", default=DEFAULT_EXPECT_MODEL,
                    help="config.model every run must report. Defaults to the "
                         "Arm-B local model so existing invocations are "
                         "unchanged; set it for a cloud arm (Sprint 17).")
    ap.add_argument("--expect-prompt-variant", default="control",
                    help="config.prompt_variant every run must report (a missing "
                         "field counts as control, as in the July collection). "
                         "Set it for a prompt-variant collection (Sprint 18).")
    ap.add_argument("--backend", default="local-llama",
                    choices=sorted(SERVER_SIGNATURES),
                    help="which server signature to record as provenance")
    ap.add_argument("--arm-label", default="Arm-B (AgenticLLMEngine)",
                    help="report title prefix")
    ap.add_argument("--token-warn", type=int, default=TOKEN_WARN,
                    help="warn above this prompt+gen total (default: the local "
                         "per-slot ctx budget)")
    ap.add_argument("--token-fail", type=int, default=TOKEN_FAIL,
                    help="fail at or above this prompt+gen total; raise it for "
                         "a backend with a larger context window. NOTE: a "
                         "managed endpoint rejects an over-context request "
                         "outright rather than truncating silently, so on "
                         "those backends this is drift detection and the "
                         "zero-call-errors check is what proves no truncation")
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

    # Per-arm expectations. Defaults reproduce the Arm-B behaviour exactly, so
    # existing invocations are unaffected by the Sprint-17 parameterisation.
    req_cfg = required_config(args.expect_model)
    server_sig = SERVER_SIGNATURES[args.backend]

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
                validate_record(rec, comp, seed, errs, warns, req_cfg,
                                expect_prompt_variant=args.expect_prompt_variant)
                ts = name_re.match(p.name).group("ts")  # type: ignore[union-attr]
                trace_name = p.name.replace(f"_{ts}.json", f"_trace_{ts}.json")
                stats = validate_llm_trace(p.parent / trace_name, errs, warns,
                                           args.token_warn, args.token_fail)
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
        f"# {args.arm_label} collection — validation report",
        "",
        f"Traces dir: `{args.traces_dir}`  ·  design: {args.reps} seeds × "
        f"{len(COMPOSITIONS)} compositions (n=6, t=8, baseline, AgenticLLMEngine, "
        f"{req_cfg['model']})",
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
        f"(fail ≥{args.token_fail}), {quality_totals['n_loop_retries']} tool-loop retries, "
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
              json.dumps(server_sig, indent=2), "```", ""]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "traces_dir": str(args.traces_dir),
        "n_finals": n_files, "n_checkpoints": n_checkpoints, "n_llm_traces": n_traces,
        "design": {"reps": args.reps, "compositions": list(COMPOSITIONS),
                   **req_cfg},
        "server_signature": server_sig,
        "verdict": verdict,
        "log_scan": {k: v for k, v in log_counts.items()} if args.logs else None,
        "files": manifest_entries,
    }, indent=1))
    print("\n".join(lines[:24]))
    print(f"\nwrote {args.report}\nwrote {args.manifest}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
