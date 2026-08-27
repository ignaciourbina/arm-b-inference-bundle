SHELL := /bin/bash
# Canonical venv is agora/.venv (CLAUDE.md); a local llm/.venv is honored if
# someone created one. Override with PYTHON=... on the command line.
PYTHON ?= $(shell test -x llm/.venv/bin/python && echo llm/.venv/bin/python || echo agora/.venv/bin/python)
PYTHONPATH := .:agora/src
LLM_BASE_URL ?= http://localhost:20434
LLM_API_FLAVOR ?= openai
LLM_MODEL ?= gemma-4-E2B-it-Q8_0.gguf

# Smoke-test knobs. Override them on the make command line, for example:
#   make -f llm/Makefile townhall-smoke
# Do not place NAME=value between -f and the Makefile path; make will treat it
# as another file/target instead of a variable override.
SMOKE_TOPIC ?= minimum_wage_seattle
SMOKE_SCENARIO ?= llm/scenarios/minimum_wage_seattle_crossover.json
SMOKE_AGENTS ?= 6
SMOKE_ROUNDS ?= 2
SMOKE_SEED ?= 42
SMOKE_CONDITION ?= baseline
SMOKE_COMPOSITION ?= symmetric_n6
SMOKE_COMPOSITIONS ?= symmetric_n6 polarized_n6 three_clusters_n6
SMOKE_PARALLEL ?= 1
SMOKE_RUN_TAG ?= smoke_make_$(shell date +%Y%m%d_%H%M%S)
SMOKE_RUN_TAG_ARG := $(if $(strip $(SMOKE_RUN_TAG)),--run-tag $(SMOKE_RUN_TAG),)
SMOKE_RESUME ?= false
SMOKE_RESUME_ARG := $(if $(filter 1 true TRUE yes YES,$(SMOKE_RESUME)),--resume,)
SMOKE_OUTPUT_DIR ?=
SMOKE_OUTPUT_DIR_ARG := $(if $(strip $(SMOKE_OUTPUT_DIR)),--output-dir $(SMOKE_OUTPUT_DIR),)
SMOKE_PROFILES ?= polis-analysis/output/ising_profiles.json
SMOKE_THETA ?= polis-analysis/output/irt_ising_theta.json

BETA_LOCAL_CONDITION ?= baseline
BETA_LOCAL_ROUNDS ?= 8
BETA_LOCAL_PARALLEL ?= 1
BETA_LOCAL_COMPOSITIONS ?= symmetric_n6 polarized_n6 three_clusters_n6
BETA_LOCAL_SEEDS ?= 326 380 366 386 327 318 383 308 334 310 309 332 335 356 323 307 325 368 361 320
BETA_LOCAL_RUN_TAG_PREFIX ?= beta_local_$(shell date +%Y%m%d_%H%M%S)
BETA_LOCAL_OUTPUT_DIR ?= outputs/llm_engine/beta_local/$(BETA_LOCAL_RUN_TAG_PREFIX)
BETA_LOCAL_RESUME ?= true
BETA_LOCAL_SKIP_EXISTING ?= true
BETA_TRACE_ROOT ?= outputs/llm_engine/beta_local
CLOUD_LLM_BASE_URL ?= http://localhost:21435
CLOUD_LLM_API_FLAVOR ?= openai
CLOUD_LLM_MODEL ?= gemma-4-E2B-it-Q8_0.gguf
CLOUD_BETA_PARALLEL ?= 7
SUPERVISOR_STALL_TIMEOUT_S ?= 600
SUPERVISOR_POLL_INTERVAL_S ?= 15

export PYTHONPATH LLM_BASE_URL LLM_API_FLAVOR LLM_MODEL

.PHONY: help venv server stop health test benchmark trace-view clean aws-launch aws-terminate aws-ssh townhall-smoke townhall-smoke-compositions townhall-beta-local-sweep townhall-beta-cloud-sweep townhall-beta-local-retry-loop townhall-beta-cloud-retry-loop

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

venv: ## Ensure llm deps in the canonical venv (agora/.venv)
	agora/.venv/bin/pip install -q aiohttp numpy scipy pytest mypy ruff

server: ## Print backend startup reminder
	@echo "Start llama-server separately and ensure it is reachable at $(LLM_BASE_URL)."
	@echo "See CLAUDE.md for the supported backend and model identifier."

stop: ## Print backend shutdown reminder
	@echo "Backend shutdown is managed outside this Makefile."

health: ## Check server + GPU status
	@curl -sf $(LLM_BASE_URL)/v1/models | python3 -c \
		"import sys,json; m=json.load(sys.stdin).get('data', []); print(f'Models: {[x.get(\"id\") for x in m]}')" \
		2>/dev/null || echo "LLM backend NOT running"
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
		--format=csv,noheader 2>/dev/null || echo "No GPU detected"

test: ## Run minimal 2-agent test (1 round)
	$(PYTHON) llm/tests/minimal_round.py

# Example overrides:
#   make -f llm/Makefile townhall-smoke SMOKE_TOPIC=minimum_wage_seattle SMOKE_ROUNDS=3
#   make -f llm/Makefile townhall-smoke SMOKE_RUN_TAG=my_probe
#   make -f llm/Makefile townhall-beta-local-sweep \
#     LLM_BASE_URL=http://localhost:21435 \
#     LLM_API_FLAVOR=openai \
#     LLM_MODEL=gemma-4-E2B-it-Q8_0.gguf
# If SMOKE_RUN_TAG is empty or unset, the runner will use its own default tag.
townhall-smoke: ## Run a real townhall deliberation smoke test
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm.townhall.runner \
		--topic $(SMOKE_TOPIC) \
		--scenario-path $(SMOKE_SCENARIO) \
		--agents $(SMOKE_AGENTS) \
		--rounds $(SMOKE_ROUNDS) \
		--seed $(SMOKE_SEED) \
		--condition $(SMOKE_CONDITION) \
		--empirical-init \
		--profiles-path $(SMOKE_PROFILES) \
		--theta-path $(SMOKE_THETA) \
		--composition $(SMOKE_COMPOSITION) --parallel $(SMOKE_PARALLEL) \
		$(SMOKE_RUN_TAG_ARG) $(SMOKE_OUTPUT_DIR_ARG) $(SMOKE_RESUME_ARG)

townhall-smoke-compositions: ## Run the quick smoke suite across the three local composition presets
	@for composition in $(SMOKE_COMPOSITIONS); do \
		echo "==> $$composition"; \
		$(MAKE) -f llm/Makefile townhall-smoke \
			SMOKE_COMPOSITION=$$composition \
			SMOKE_RUN_TAG=$(SMOKE_RUN_TAG)_$$composition || exit $$?; \
	done

townhall-beta-local-sweep: ## Run the local beta sweep with the current local defaults
	@trace_root="$(BETA_TRACE_ROOT)"; \
	skip_existing="$(BETA_LOCAL_SKIP_EXISTING)"; \
	find_existing() { \
		local cond="$$1" comp="$$2" seed="$$3"; \
		find "$$trace_root" -type f -name '*.json' \
			! -name '*_checkpoint.json' \
			! -name '*_trace_live.json' \
			! -name '*_trace_*.json' \
			-print 2>/dev/null | \
			grep -E "_$${cond}_$${comp}_s$${seed}(_|\\.json$$)" | head -n 1; \
	}; \
	for seed in $(BETA_LOCAL_SEEDS); do \
		for composition in $(BETA_LOCAL_COMPOSITIONS); do \
			tag=$(BETA_LOCAL_RUN_TAG_PREFIX)_$(BETA_LOCAL_CONDITION)_$${composition}_s$${seed}; \
			if [[ "$$skip_existing" =~ ^(1|true|TRUE|yes|YES)$$ ]]; then \
				existing="$$(find_existing "$(BETA_LOCAL_CONDITION)" "$$composition" "$$seed")"; \
				if [[ -n "$$existing" ]]; then \
					echo "[skip-existing] $$tag -> $$existing"; \
					continue; \
				fi; \
			fi; \
			echo "==> $$tag"; \
			$(MAKE) -f llm/Makefile townhall-smoke \
				SMOKE_CONDITION=$(BETA_LOCAL_CONDITION) \
				SMOKE_COMPOSITION=$$composition \
				SMOKE_ROUNDS=$(BETA_LOCAL_ROUNDS) \
				SMOKE_PARALLEL=$(BETA_LOCAL_PARALLEL) \
				SMOKE_SEED=$$seed \
				SMOKE_OUTPUT_DIR=$(BETA_LOCAL_OUTPUT_DIR) \
				SMOKE_RESUME=$(BETA_LOCAL_RESUME) \
				SMOKE_RUN_TAG=$$tag || exit $$?; \
		done; \
	done

townhall-beta-cloud-sweep: ## Run the beta sweep locally while sending inference through the tunneled cloud backend
	$(MAKE) -f llm/Makefile townhall-beta-local-sweep \
		LLM_BASE_URL=$(CLOUD_LLM_BASE_URL) \
		LLM_API_FLAVOR=$(CLOUD_LLM_API_FLAVOR) \
		LLM_MODEL=$(CLOUD_LLM_MODEL) \
		BETA_LOCAL_CONDITION=$(BETA_LOCAL_CONDITION) \
		BETA_LOCAL_ROUNDS=$(BETA_LOCAL_ROUNDS) \
		BETA_LOCAL_PARALLEL=$(CLOUD_BETA_PARALLEL) \
		BETA_LOCAL_COMPOSITIONS="$(BETA_LOCAL_COMPOSITIONS)" \
		BETA_LOCAL_SEEDS="$(BETA_LOCAL_SEEDS)" \
		BETA_LOCAL_RUN_TAG_PREFIX=$(BETA_LOCAL_RUN_TAG_PREFIX) \
		BETA_LOCAL_OUTPUT_DIR=$(BETA_LOCAL_OUTPUT_DIR) \
		BETA_LOCAL_RESUME=$(BETA_LOCAL_RESUME)

townhall-beta-local-retry-loop: ## Supervise the local beta sweep and restart on failure with resume
	RUN_TAG_PREFIX=$(BETA_LOCAL_RUN_TAG_PREFIX) \
	SEEDS="$(BETA_LOCAL_SEEDS)" \
	COMPOSITIONS="$(BETA_LOCAL_COMPOSITIONS)" \
	BETA_LOCAL_ROUNDS=$(BETA_LOCAL_ROUNDS) \
	BETA_LOCAL_PARALLEL=$(BETA_LOCAL_PARALLEL) \
	OUTPUT_DIR=$(BETA_LOCAL_OUTPUT_DIR) \
	BETA_LOCAL_RESUME=$(BETA_LOCAL_RESUME) \
	STALL_TIMEOUT_S=$(SUPERVISOR_STALL_TIMEOUT_S) \
	POLL_INTERVAL_S=$(SUPERVISOR_POLL_INTERVAL_S) \
	SWEEP_TARGET=townhall-beta-local-sweep \
	$(PYTHON) llm/run_beta_local_retry_loop.py

townhall-beta-cloud-retry-loop: ## Supervise the cloud-backed beta sweep and restart on failure with resume
	RUN_TAG_PREFIX=$(BETA_LOCAL_RUN_TAG_PREFIX) \
	SEEDS="$(BETA_LOCAL_SEEDS)" \
	COMPOSITIONS="$(BETA_LOCAL_COMPOSITIONS)" \
	BETA_LOCAL_ROUNDS=$(BETA_LOCAL_ROUNDS) \
	CLOUD_BETA_PARALLEL=$(CLOUD_BETA_PARALLEL) \
	OUTPUT_DIR=$(BETA_LOCAL_OUTPUT_DIR) \
	BETA_LOCAL_RESUME=$(BETA_LOCAL_RESUME) \
	STALL_TIMEOUT_S=$(SUPERVISOR_STALL_TIMEOUT_S) \
	POLL_INTERVAL_S=$(SUPERVISOR_POLL_INTERVAL_S) \
	SWEEP_TARGET=townhall-beta-cloud-sweep \
	LLM_BASE_URL=$(CLOUD_LLM_BASE_URL) \
	LLM_API_FLAVOR=$(CLOUD_LLM_API_FLAVOR) \
	LLM_MODEL=$(CLOUD_LLM_MODEL) \
	ON_SUCCESS_CMD='./pipeline/runpod/mpsa pod-stop' \
	$(PYTHON) llm/run_beta_local_retry_loop.py

benchmark: ## Run sequential vs parallel throughput benchmark
	$(PYTHON) -m llm.benchmark

trace-view: ## Pretty-print the latest trace file
	@find outputs/llm_engine -type f -name 'minimal_trace_*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- | \
		xargs -I{} python3 -c "\
import json,sys; d=json.load(open('{}')); \
print(f'Model: {d[\"model\"]}'); \
print(f'Calls: {d[\"n_calls\"]}'); \
print(f'Total: {d[\"total_elapsed_s\"]:.1f}s'); \
print(); \
[print(f'{c[\"seq\"]:2d} [{c[\"hook\"]:8s}] {c[\"agent_id\"]:4s} {c[\"elapsed_s\"]:5.1f}s | {json.dumps(c[\"response_parsed\"])}') for c in d['calls']]"

aws-launch: ## Launch legacy GPU deploy script on AWS
	bash llm/deploy/aws_launch.sh

aws-terminate: ## Terminate the legacy AWS deploy instance
	@cat llm/deploy/instance.json 2>/dev/null | python3 -c \
		"import sys,json; d=json.load(sys.stdin); print(f'Terminating {d[\"instance_id\"]}...')" && \
	aws ec2 terminate-instances --instance-ids \
		$$(python3 -c "import json; print(json.load(open('llm/deploy/instance.json'))['instance_id'])") \
		--region us-east-1 --output text || echo "No instance found"

aws-ssh: ## SSH into the legacy AWS deploy instance
	@eval $$(python3 -c "import json; print(json.load(open('llm/deploy/instance.json'))['ssh_cmd'])")

trace-view-agentic: ## Pretty-print the latest agentic trace file
	@find outputs/llm_engine -type f -name 'agentic_trace_*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- | \
		xargs -I{} python3 -c "\
import json,sys; d=json.load(open('{}')); \
print(f'Model: {d[\"model\"]}'); \
print(f'Calls: {d[\"n_calls\"]}'); \
print(f'Total: {d[\"total_elapsed_s\"]:.1f}s'); \
print(); \
[print(f'{c[\"seq\"]:2d} [{c[\"hook\"]:8s}] {c[\"agent_id\"]:4s} {c[\"elapsed_s\"]:5.1f}s | {json.dumps(c[\"response_parsed\"], default=str)[:120]}') for c in d['calls']]"

clean: ## Remove traces and __pycache__
	rm -rf outputs/llm_engine/*.json
	find llm -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
