.PHONY: help setup sync data feast-apply lint format check typecheck test test-cov \
        notebooks run lab clean reset ci dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-extras
	uv run python -m ipykernel install --user --name feature-store-101
	@echo "\n✅ Ready. Run 'make data' then 'make test'."

sync: ## Sync deps
	uv sync --all-extras

data: ## Generate the synthetic event streams (local parquet)
	uv run python -m src.datasets.events
	@echo "✅ Event streams written to data/"

feast-apply: ## Apply the local Feast repo (provider: local, sqlite + file; no server)
	cd src/feast_demo/feature_repo && uv run feast apply
	@echo "✅ Local Feast registry applied (no server, no account)."

lint: format check typecheck ## All linters
format: ## ruff format
	uv run ruff format src/ tests/ app/
check: ## ruff check
	uv run ruff check --fix src/ tests/ app/
typecheck: ## ty check
	uv run ty check src/

test: ## pytest
	uv run pytest
test-cov: ## pytest with coverage
	uv run pytest --cov=src --cov-report=term-missing

notebooks: data ## Execute all notebooks
	@for nb in notebooks/0*.ipynb; do \
		echo "▶ Executing $$nb ..."; \
		uv run jupyter nbconvert --to notebook --execute \
			--ExecutePreprocessor.timeout=300 "$$nb" || exit 1; \
	done
	@echo "\n✅ All notebooks executed."

run: ## Streamlit app
	uv run streamlit run app/streamlit_app.py --server.port 8501
lab: ## JupyterLab
	uv run jupyter lab --no-browser --port 8888

clean: ## Clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache htmlcov .coverage .pytest_cache .ruff_cache
	@echo "🧹 Cleaned."
reset: clean ## Full reset
	rm -rf .venv data/events

ci: sync lint test ## CI
dev: lint test ## Fast loop
