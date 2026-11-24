# Arm-B inference bundle — workflow commands for SeaWulf.
#
#   make help          every target, grouped
#   make doctor        verify the environment before submitting anything
#   make status        what is on the cluster and what is on disk
#
# Conventions this file encodes so you never have to remember them:
#   * PYTHONSAFEPATH=1 is mandatory (llm/types.py shadows stdlib `types`)
#   * slurm lives behind `module load slurm` and is absent from a bare shell
#   * v100 is 1 node / 1 job per user -- submissions are guarded, not trusted
#   * traces are irreplaceable; every destructive target asks or refuses
#
# See CLAUDE.md for the doc index. llm/Makefile is the upstream bundle's own
# Makefile (smoke tests, sweeps) and is left untouched; `make smoke` delegates.

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
.DEFAULT_GOAL := help

ROOT       := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PY         ?= agora/.venv/bin/python
SCRATCH    ?= /gpfs/scratch/$(USER)
COLLECTION := llm/traces/collection
LOGS       := llm/traces/logs
MODELS     := $(HOME)/.cache/llama-models

export PYTHONPATH     := .:agora/src
export PYTHONSAFEPATH := 1

# slurm is a module, not a PATH entry. Every recipe line is its own shell, so
# the load has to be inlined per line rather than done once at the top.
SLURM := source /etc/profile.d/modules.sh >/dev/null 2>&1; module load slurm >/dev/null 2>&1;

# Analysis defaults -- override on the command line, e.g. make validate CELL=ms_E4B
CELL   ?= armb_sota_r256_t10
REPS   ?= 100
# validator expectations; the defaults validate B2 (armb_sota_r256_t10)
MODEL  ?= gemma-4-E2B-it-Q8_0.gguf
BACKEND ?= seawulf-v100-r256-t10
SEEDS  ?= 1 2 3 4 5
CELLS  ?= armb_sota_r256_t10
# score_ablation_cells.py's own defaults point at the old bundle layout
# (llm/traces/{beta_local/arm_b_local_p6cache,ablation}); neither exists here.
CONTROL ?= armb_sota_r256_t10
# Reasoning ON is locked, so the default curve is the model-scale axis only.
# Name a reasoning-off cell explicitly if you need the historical baseline.
GROWTH_CELLS ?= armb_sota_r256_t10 ms_E4B ms_26B_a4b

.PHONY: help env doctor \
        sota pilot scaling axes-cd quant-budget verify-optimum lane-sweep submit \
        cancel cancel-job \
        status queue cells progress watch log srvlog gpu \
        provenance provenance-all compare tool-usage \
        validate score scaling-report growth inventory report \
        quota du gz-check compress-dry compress archive \
        test lint typecheck smoke clean

##@ Setup & diagnostics

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; \
	    printf "\n\033[1mArm-B inference bundle\033[0m — make <target> [VAR=value]\n"} \
	  /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
	  /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' \
	  $(MAKEFILE_LIST)
	@printf "\n\033[1mCommon variables\033[0m\n"
	@printf "  %-16s %s\n" "CELL=" "cell to act on (default $(CELL))" \
	    "CELLS=" "space-separated cells for score/growth" \
	    "REPS=" "seeds per composition, for validate (default $(REPS))" \
	    "PY=" "python interpreter (default $(PY))"
	@printf "\nDocs: CLAUDE.md · docs/SOTA-CONFIG-MEMO.md · docs/STORAGE-POLICY.md\n\n"

env:  ## Print the resolved environment
	@echo "ROOT        $(ROOT)"
	@echo "PY          $(PY)  ($$($(PY) -V 2>&1))"
	@echo "PYTHONPATH  $(PYTHONPATH)"
	@echo "SAFEPATH    $(PYTHONSAFEPATH)"
	@echo "SCRATCH     $(SCRATCH)"
	@echo "MODELS      $(MODELS) -> $$(readlink -f $(MODELS) 2>/dev/null || echo MISSING)"
	@echo "llama-server $$(ls -l $(HOME)/src/llama.cpp/build/bin/llama-server 2>/dev/null | awk '{print $$NF}' || echo MISSING)"

doctor:  ## Preflight: verify every invariant a job depends on
	@ok=0; \
	chk() { if eval "$$2" >/dev/null 2>&1; then printf "  \033[32mok  \033[0m %s\n" "$$1"; \
	        else printf "  \033[31mFAIL\033[0m %s\n" "$$1"; ok=1; fi; }; \
	echo "--- environment ---"; \
	chk "python venv present"            "test -x $(PY)"; \
	chk "PYTHONSAFEPATH exported"        "test '$(PYTHONSAFEPATH)' = 1"; \
	chk "llm package imports"            "$(PY) -c 'import llm.townhall.runner'"; \
	chk "model symlink resolves"         "test -d $(MODELS)"; \
	chk "E2B Q8_0 weights present"       "test -s $(MODELS)/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf"; \
	chk "llama-server shim executable"   "test -x $(HOME)/src/llama.cpp/build/bin/llama-server"; \
	chk "singularity available"          "command -v singularity"; \
	chk "slurm loadable"                 "$(SLURM) command -v sbatch"; \
	chk "scenario file present"          "test -s llm/scenarios/minimum_wage_seattle_crossover.json"; \
	chk "ising profiles present"         "test -s polis-analysis/output/ising_profiles.json"; \
	echo "--- storage ---"; \
	myquota 2>/dev/null | sed 's/^/  /' || echo "  myquota unavailable"; \
	echo "--- exit ---"; \
	if [ $$ok -eq 0 ]; then echo "  all checks passed"; else echo "  one or more checks FAILED — do not submit"; fi; \
	exit $$ok

##@ Submit jobs   (v100 = 1 node / 1 job per user; submissions are guarded)

# Refuses to queue a second v100 job. The cluster policy is a hard limit, not
# advice: a second submission sits PENDING forever and blocks the chain.
define guard_v100
	@$(SLURM) if ! OUT=$$(timeout 45 squeue -u $(USER) -h -p v100 -o "%i %j %T" 2>/dev/null); then \
	  echo "REFUSED: cannot reach the Slurm controller, so cannot tell whether a"; \
	  echo "job is already queued. This guard fails CLOSED on purpose -- an earlier"; \
	  echo "version counted lines from a failing squeue as zero and let a duplicate"; \
	  echo "through. Wait for the controller and retry."; \
	  exit 1; \
	fi; \
	if echo "$$OUT" | grep -q .; then \
	  echo "REFUSED: a v100 job is already queued/running (limit is 1):"; \
	  echo "$$OUT" | sed 's/^/  /'; \
	  echo "Cancel it first (make cancel) or wait. See docs/CLUSTER_POLICY_JOBS.md"; \
	  exit 1; \
	fi
endef

sota:  ## Submit the 300-run SoTA collection (self-resubmitting chain)
	$(guard_v100)
	@$(SLURM) id=$$(sbatch --parsable slurm/armb_sota_collection.sbatch) && \
	  echo "submitted armb-sota  job=$$id  log=$(SCRATCH)/armb_sota_$$id.log"

pilot:  ## Submit the model-scale pilot (26B-A4B + E4B) — gated on complete weights
	@f=$(MODELS)/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q8_0.gguf; \
	t=$$(cat $(SCRATCH)/26b_target.txt 2>/dev/null || echo 0); \
	s=$$(stat -c%s "$$f" 2>/dev/null || echo 0); \
	if [ "$$t" -le 0 ]; then echo "REFUSED: no target size at $(SCRATCH)/26b_target.txt"; exit 1; fi; \
	if [ "$$s" -lt "$$t" ]; then \
	  awk -v s=$$s -v t=$$t 'BEGIN{printf "REFUSED: 26B weights incomplete — %.1f/%.1f GB (%.1f%%)\n", s/1e9, t/1e9, 100*s/t}'; \
	  echo "A truncated GGUF fails at load and burns the allocation. Wait for the download."; \
	  exit 1; \
	fi; \
	echo "26B weights complete ($$s bytes)."
	$(guard_v100)
	@$(SLURM) id=$$(sbatch --parsable slurm/model_scale_pilot.sbatch) && \
	  echo "submitted model-scale  job=$$id  log=$(SCRATCH)/model_scale_$$id.log"

scaling:         ## Submit the wall-clock scaling-law sweep
	$(guard_v100)
	@$(SLURM) echo "submitted job=$$(sbatch --parsable slurm/scaling_law.sbatch)"

axes-cd:         ## Submit axes (c) cache-reuse and (d) slot-curve cells
	$(guard_v100)
	@$(SLURM) echo "submitted job=$$(sbatch --parsable slurm/axes_cd.sbatch)"

quant-budget:    ## Submit the Q4-vs-Q8 / b128-vs-b256 ablation
	$(guard_v100)
	@$(SLURM) echo "submitted job=$$(sbatch --parsable slurm/abl_quant_budget.sbatch)"

verify-optimum:  ## Submit the confirmation run at the recommended operating point
	$(guard_v100)
	@$(SLURM) echo "submitted job=$$(sbatch --parsable slurm/verify_optimum.sbatch)"

lane-sweep:      ## Submit the concurrent-lane sweep
	$(guard_v100)
	@$(SLURM) echo "submitted job=$$(sbatch --parsable slurm/lane_sweep.sbatch)"

submit:  ## Submit an arbitrary script: make submit JOB=slurm/foo.sbatch
	@test -n "$(JOB)" || { echo "usage: make submit JOB=slurm/<file>.sbatch"; exit 1; }
	@test -f "$(JOB)" || { echo "no such script: $(JOB)"; exit 1; }
	$(guard_v100)
	@$(SLURM) ID=$$(timeout 60 sbatch --parsable $(JOB) 2>&1); RC=$$?; \
	case "$$ID" in \
	  ''|*[!0-9]*) \
	    echo "SUBMISSION STATE UNKNOWN (rc=$$RC): $$ID"; \
	    echo "A socket timeout is reported on the REPLY, so the job may well have"; \
	    echo "been accepted. Run 'make queue' before retrying -- retrying blind is"; \
	    echo "how two jobs ended up queued against the same cell."; \
	    exit 1;; \
	  *) echo "submitted job=$$ID";; \
	esac

cancel:  ## Cancel ALL of your jobs (asks first)
	@$(SLURM) squeue -u $(USER) -o "%.9i %.16j %.9T %.11M %R" || true; \
	read -r -p "cancel every job above? [y/N] " a; \
	if [ "$$a" = y ] || [ "$$a" = Y ]; then scancel -u $(USER) && echo cancelled; \
	else echo "aborted"; fi

cancel-job:  ## Cancel one job: make cancel-job ID=12345
	@test -n "$(ID)" || { echo "usage: make cancel-job ID=<jobid>"; exit 1; }
	@$(SLURM) scancel $(ID) && echo "cancelled $(ID)"

##@ Monitor

status: queue cells  ## Queue + on-disk cell status (the one to run first)

queue:  ## Slurm queue for this user
	@echo "=== slurm ==="
	@$(SLURM) squeue -u $(USER) -o "%.9i %.16j %.9P %.9T %.11M %.11l %.4D %R" 2>/dev/null \
	  | { grep . || echo "(no jobs queued or running)"; }

cells:  ## On-disk status of every collection cell
	@echo; echo "=== cells ==="; $(PY) slurm/cell_status.py

progress:  ## Detailed status of one cell: make progress CELL=<name>
	@$(PY) slurm/cell_status.py $(CELL)
	@echo; echo "=== last 8 telemetry events ($(CELL)) ==="
	@tail -8 $(LOGS)/$(CELL)_progress.jsonl 2>/dev/null || echo "(no progress log)"

watch:  ## Live status, refreshed every 60s (Ctrl-C to stop)
	@while :; do clear; date; $(MAKE) --no-print-directory status; sleep 60; done

log:  ## Tail the newest slurm job log
	@f=$$(ls -t $(SCRATCH)/*_*.log 2>/dev/null | head -1); \
	if [ -z "$$f" ]; then echo "no slurm logs in $(SCRATCH)"; else \
	  echo "=== $$f ==="; tail -n $${N:-40} "$$f"; fi

srvlog:  ## Tail the newest llama-server log
	@f=$$(ls -t $(LOGS)/*srv*.log $(LOGS)/llama_server_*.log 2>/dev/null | head -1); \
	if [ -z "$$f" ]; then echo "no server logs in $(LOGS)"; else \
	  echo "=== $$f ==="; tail -n $${N:-40} "$$f"; fi

gpu:  ## GPU state on the node running your job
	@$(SLURM) j=$$(squeue -u $(USER) -h -t RUNNING -o "%i" | head -1); \
	if [ -z "$$j" ]; then echo "no running job"; else \
	  srun --jobid="$$j" nvidia-smi \
	    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
	    --format=csv,noheader 2>/dev/null || echo "could not attach to job $$j"; fi

##@ Analysis

provenance:  ## Config fingerprint + grid coverage: make provenance [CELL=x]
	$(PY) slurm/provenance.py $(if $(filter-out armb_sota_r256_t10,$(CELL)),$(CELL),$(CELLS))

provenance-all:  ## Fingerprint every cell; non-zero exit if any cell is not uniform
	$(PY) slurm/provenance.py

compare:  ## Is a comparison legitimate? make compare A=cell1 B=cell2 [TREATMENT=model]
	@test -n "$(A)" -a -n "$(B)" || { echo 'usage: make compare A=<cell> B=<cell> [TREATMENT="model,infra.quant"]'; exit 1; }
	@TREATMENT="$(TREATMENT)" $(PY) slurm/provenance.py --compare $(A) $(B)

validate:  ## Validator over a cell: make validate CELL=<name> [REPS=100] [MODEL=x.gguf] [BACKEND=sig]
	@mkdir -p pipeline/output/llm_collection/$(CELL)
	$(PY) pipeline/validate_llm_collection.py \
	  --traces-dir $(COLLECTION)/$(CELL) --prefix $(CELL) --reps $(REPS) \
	  --expect-model $(MODEL) --backend $(BACKEND) \
	  --report pipeline/output/llm_collection/$(CELL)/validation-report.md \
	  --manifest pipeline/output/llm_collection/$(CELL)/manifest.json

score:  ## Paired scoring vs control: make score CELLS="a b" [CONTROL=x] [SEEDS="1 2 3"]
	$(PY) llm/score_ablation_cells.py --cells $(CELLS) --seeds $(SEEDS) \
	  --control-dir $(COLLECTION)/$(CONTROL) --ablation-root $(COLLECTION)

scaling-report:  ## Measured wall-clock scaling law + projections
	$(PY) slurm/scaling_law_report.py

growth:  ## Deliberation growth curves: make growth [GROWTH_CELLS="a b"]
	$(PY) slurm/growth_pattern_compare.py $(GROWTH_CELLS)

tool-usage:  ## Elective tool use per run + its relation to outcomes: make tool-usage CELL=x
	$(PY) slurm/tool_usage.py $(CELL)

inventory:  ## Inventory result and trace files
	$(PY) llm/inventory_traces.py

report: scaling-report growth  ## Everything that reads only from disk

##@ Storage   (policy: docs/STORAGE-POLICY.md — traces stay in backed-up home)

quota:  ## Authoritative quota (myquota, never df — df lies on GPFS)
	@myquota

du:  ## Where the space actually goes
	@echo "=== home / project ==="
	@du -sh $(COLLECTION) llm/traces/beta_local docs polis-analysis 2>/dev/null
	@echo "=== raw call traces (the 97%) ==="
	@find llm/traces -name '*_trace_*.json' -printf '%s\n' 2>/dev/null | \
	  awk '{s+=$$1; n++} END{printf "  %d files, %.2f GB\n", n, s/1e9}'
	@echo "=== scratch ==="
	@du -sh $(SCRATCH)/* 2>/dev/null | sort -h | tail -8

gz-check:  ## Are the six trace readers .gz-aware yet? (blocks compression)
	@echo "STORAGE-POLICY §7: compression must not land before reader support."; \
	miss=0; \
	for f in llm/inventory_traces.py llm/plot_ablation_trajectories.py \
	         llm/analyze_ablation_reliability.py llm/bench_local_config.py \
	         llm/run_collection_parallel.py llm/tests/test_client_trace.py; do \
	  if grep -q "gzip" "$$f" 2>/dev/null; then printf "  \033[32mgz  \033[0m %s\n" "$$f"; \
	  else printf "  \033[31mplain\033[0m %s\n" "$$f"; miss=1; fi; \
	done; \
	if [ $$miss -eq 1 ]; then \
	  echo "NOT READY — add gzip.open() support, run 'make test', then compress."; \
	else echo "READY — 'make compress' is safe."; fi; \
	exit $$miss

compress-dry:  ## Show what Tier-1 gzip would save (reads nothing, writes nothing)
	@find $(COLLECTION) -name '*_trace_*.json' -printf '%s\n' 2>/dev/null | \
	  awk '{s+=$$1; n++} END{printf "Tier 1 (per-file gzip, 23.3x, random access KEPT)\n  now:   %d files, %.2f GB\n  after: %.3f GB\n", n, s/1e9, s/1e9/23.3}'
	@find $(COLLECTION) -name '*_trace_*.json' -printf '%s\n' 2>/dev/null | \
	  awk '{s+=$$1} END{printf "Tier 2 (tar.zst -19 per cell, 67.4x, extraction required)\n  after: %.3f GB\n", s/1e9/67.4}'

compress: gz-check  ## Tier-1 gzip the raw traces of one cell: make compress CELL=<name>
	@test -d $(COLLECTION)/$(CELL) || { echo "no such cell: $(CELL)"; exit 1; }
	@n=$$(find $(COLLECTION)/$(CELL) -name '*_trace_*.json' | wc -l); \
	echo "gzip $$n raw trace files in $(CELL) (finals and checkpoints untouched)"; \
	read -r -p "proceed? [y/N] " a; [ "$$a" = y ] || [ "$$a" = Y ] || { echo aborted; exit 1; }; \
	find $(COLLECTION)/$(CELL) -name '*_trace_*.json' -print0 | xargs -0 -P4 gzip -9 && \
	echo "done — re-run 'make validate CELL=$(CELL)' to confirm nothing broke"

archive:  ## Tier-2 archive a superseded cell to tar.zst: make archive CELL=<name>
	@test -d $(COLLECTION)/$(CELL) || { echo "no such cell: $(CELL)"; exit 1; }
	@command -v zstd >/dev/null || { echo "zstd not on PATH"; exit 1; }
	@echo "Tier 2 archives a cell nobody is analysing. Reading it again needs extraction."
	@read -r -p "archive $(CELL) to $(COLLECTION)/$(CELL).tar.zst? [y/N] " a; \
	[ "$$a" = y ] || [ "$$a" = Y ] || { echo aborted; exit 1; }; \
	tar -C $(COLLECTION) -cf - $(CELL) | zstd -19 -T4 -o $(COLLECTION)/$(CELL).tar.zst && \
	echo "written. The source dir is intentionally LEFT IN PLACE — verify, then rm -rf it yourself."

##@ Development

test:  ## Run the test suite (mandatory after any code change)
	$(PY) -m pytest llm/tests/ -q

lint:  ## ruff
	@agora/.venv/bin/ruff check llm/ pipeline/ || true

typecheck:  ## mypy over llm/
	@agora/.venv/bin/mypy llm/ || true

smoke:  ## Upstream bundle smoke test (delegates to llm/Makefile)
	$(MAKE) -f llm/Makefile townhall-smoke

clean:  ## Remove __pycache__ only. Never touches traces.
	@find . -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "pycache cleared (traces untouched — see docs/STORAGE-POLICY.md)"
