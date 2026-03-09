# ============================================================
# VLM Document Linker — shorthand commands
#
# Usage: make <target>
# Run "make help" to see all available commands.
# ============================================================

DC = docker compose

.PHONY: help up down restart web web-rebuild web-logs \
        models worker worker-rebuild worker-logs \
        enqueue export status \
        gpu0 gpu1 gpu-multi \
        ollama-shell-gpu0 ollama-shell-gpu1 ollama-shell-multi \
        queue-clear redis-cli logs ps

# -- Help ---------------------------------------------------

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# -- Full stack ---------------------------------------------

up: ## Start everything (models + worker + web)
	$(DC) --profile models --profile worker up -d

down: ## Stop everything
	$(DC) --profile models --profile worker down

restart: ## Restart everything
	$(DC) --profile models --profile worker restart

ps: ## Show running containers
	$(DC) --profile models --profile worker ps

logs: ## Tail all logs
	$(DC) --profile models --profile worker logs -f --tail=50

# -- Web dashboard ------------------------------------------

web: ## Start web dashboard
	$(DC) up -d web

web-rebuild: ## Rebuild and restart web dashboard
	$(DC) up -d --build web

web-logs: ## Tail web logs
	$(DC) logs -f --tail=50 web

# -- Models (Ollama) ----------------------------------------

models: ## Start all Ollama GPU services
	$(DC) --profile models up -d

gpu0: ## Start ollama-gpu0 only (RTX 3080)
	$(DC) --profile models up -d ollama-gpu0

gpu1: ## Start ollama-gpu1 only (RTX 3060 Ti)
	$(DC) --profile models up -d ollama-gpu1

gpu-multi: ## Start ollama-multi only (3x RTX 3060 Ti)
	$(DC) --profile models up -d ollama-multi

# -- Worker -------------------------------------------------

worker: ## Start pipeline worker
	$(DC) --profile worker up -d

worker-rebuild: ## Rebuild and restart worker
	$(DC) --profile worker up -d --build worker

worker-logs: ## Tail worker logs
	$(DC) --profile worker logs -f --tail=50 worker

# -- One-shot jobs ------------------------------------------

enqueue: ## Scan input/ and enqueue new files
	$(DC) --profile jobs run --rm enqueuer

export: ## Export results to CSV
	$(DC) --profile jobs run --rm exporter

# -- Ollama shells ------------------------------------------

ollama-shell-gpu0: ## Open bash in ollama-gpu0 (RTX 3080)
	docker exec -it vdl-ollama-gpu0 bash

ollama-shell-gpu1: ## Open bash in ollama-gpu1 (RTX 3060 Ti)
	docker exec -it vdl-ollama-gpu1 bash

ollama-shell-multi: ## Open bash in ollama-multi (3x RTX 3060 Ti)
	docker exec -it vdl-ollama-multi bash

# -- Redis --------------------------------------------------

redis-cli: ## Open Redis CLI
	docker exec -it vdl-redis redis-cli

queue-clear: ## Clear the pending queue
	docker exec vdl-redis redis-cli DEL queue:pending

status: ## Show queue depth and file counts
	@echo "--- Queue ---"
	@docker exec vdl-redis redis-cli LLEN queue:pending
	@echo "--- Files ---"
	@docker exec vdl-redis redis-cli HLEN files
	@echo "--- Failures ---"
	@docker exec vdl-redis redis-cli HLEN failures
	@echo "--- Processing ---"
	@docker exec vdl-redis redis-cli HLEN processing
