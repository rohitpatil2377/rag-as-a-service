# RAG as a Service — developer entrypoints. See README.md for the quickstart.
# All targets assume `uv` (https://docs.astral.sh/uv/) is installed.

.DEFAULT_GOAL := help
# The dev stack is the BASE and is SELF-CONTAINED: `make up` on a fresh checkout needs no
# secrets, no model endpoint and no GPU. The rest are optional overlays a real deployment adds —
# model endpoints (deploy), secrets (override), federation nodes (mesh), box-local tuning (local).
# Each is included only when the file exists; each ships as a `.example.yml` you COPY to the name
# below to activate, and the activated copies are gitignored because they are box-specific.
#
# This ordering is not decorative. Running `docker compose -f dev.yml` ALONE on a box that HAS a
# deploy overlay silently drops RAG_EMBED_MODEL/RAG_EMBED_DIM back to the local 384-dim default,
# and every query then fails against a 1024-dim collection with "expected dim: 1024, got 384".
# `make compose-files` prints the set these targets use; recover the set a running container was
# actually built from with:
#   docker inspect <container> --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}'
COMPOSE_FILES := infra/compose/docker-compose.dev.yml \
                 $(wildcard infra/compose/docker-compose.deploy.yml) \
                 $(wildcard infra/compose/docker-compose.override.yml) \
                 $(wildcard infra/compose/docker-compose.mesh.yml) \
                 $(wildcard infra/compose/docker-compose.local.yml)
COMPOSE := docker compose $(foreach f,$(COMPOSE_FILES),-f $(f))

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Sync the uv workspace (all libs + apps + dev tools)
	uv sync --all-packages --group dev

.PHONY: install-chem
install-chem: ## ...plus the optional RDKit extra (unskips the 5 chem-normalization tests)
	uv sync --all-packages --group dev --extra chem

.PHONY: lint
lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

.PHONY: fmt
fmt: ## Auto-format with ruff
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: typecheck
typecheck: ## mypy strict type-check
	uv run mypy libs services apps sdks/python

.PHONY: test
test: ## Run unit tests (excludes integration/e2e)
	uv run pytest -m "not integration and not e2e and not eval"

.PHONY: test-integration
test-integration: ## Run integration tests (requires `make up`)
	uv run pytest -m integration

.PHONY: eval
eval: ## Run the golden-set release gate (requires stack up — needs the real cross-encoder)
	uv run pytest -m eval

.PHONY: calibrate
calibrate: ## Sweep abstain+faithfulness thresholds vs the golden set (stack up; --judge-url for a real judge)
	uv run python scripts/calibrate.py --kb $(KB) $(ARGS)

.PHONY: ragas
ragas: ## RAGAS governance gate: certify KBs against the quality contract (stack up). make ragas ARGS="--kbs acme__hr,acme__finance"
	uv run python scripts/ragas_gate.py $(ARGS)

.PHONY: up
up: ## Start the local sovereign store + model stack (docker-compose)
	$(COMPOSE) up -d --build
	# Run the readiness probe INSIDE the compose network. The stores publish no host ports, so
	# from the host their names do not resolve and `make up` exited non-zero on a healthy stack.
	$(COMPOSE) exec -T api /app/.venv/bin/python scripts/wait_for_stores.py

.PHONY: redeploy
redeploy: ## Rebuild + restart ONE service after a code change (SVC=api, default api+console)
	$(COMPOSE) up -d --build $(or $(SVC),api console)
	@echo "redeployed: $(or $(SVC),api console)"

.PHONY: compose-files
compose-files: ## Print the compose files these targets use (what a running container should match)
	@echo $(COMPOSE_FILES) | tr ' ' '\n'

.PHONY: down
down: ## Stop the stack and remove volumes
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail stack logs
	$(COMPOSE) logs -f

.PHONY: migrate
migrate: ## Apply Alembic migrations (schema-per-tenant discipline)
	uv run alembic -c libs/clients/alembic.ini upgrade head

.PHONY: demo
demo: ## Walking skeleton: ingest -> grounded cited answer -> abstain. ARGS=--stores for real stores
	uv run python -m demo_cli.run --seed --ask "What is the platform's uptime target?" $(ARGS)

.PHONY: ci
ci: lint typecheck test ## What CI runs on every PR

.PHONY: hooks
hooks: ## Install the pre-push gate (runs `make ci` before every push)
	git config core.hooksPath .githooks
	@echo "pre-push gate installed. The GitLab remote runs no pipeline, so this is the only gate."

backup: ## Back up the stateful stores to backups/<timestamp>
	@bash scripts/backup.sh

restore: ## Restore from a backup dir: make restore BACKUP=backups/<timestamp>
	@bash scripts/restore.sh "$(BACKUP)"

.PHONY: airgap-bundle
airgap-bundle: ## Build a self-contained air-gapped install bundle (run on a connected machine)
	bash scripts/airgap-bundle.sh $(ARGS)

.PHONY: bench
bench: ## Benchmark federated scatter-gather scaling / vector backends (stack up): make bench ARGS="scaling --kb caller"
	uv run python scripts/benchmark.py $(ARGS)
