.PHONY: help setup install lint format test check clean run-baseline run-inverse run-ensemble run-all figs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)PINN Passivity Paper - Makefile Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

setup: ## Install dependencies and setup environment (Poetry)
	@echo "$(BLUE)Setting up environment...$(NC)"
	@command -v poetry >/dev/null 2>&1 || { echo "$(RED)Poetry not found. Installing...$(NC)"; pip install poetry; }
	poetry install
	poetry run pre-commit install
	@echo "$(GREEN)Setup complete!$(NC)"

install: setup ## Alias for setup

lint: ## Run linting (ruff)
	@echo "$(BLUE)Running linter...$(NC)"
	poetry run ruff check src/
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code (black + ruff)
	@echo "$(BLUE)Formatting code...$(NC)"
	poetry run black src/
	poetry run ruff check --fix src/
	@echo "$(GREEN)Formatting complete!$(NC)"

test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	pytest src/tests/ -v
	@echo "$(GREEN)Tests complete!$(NC)"

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	poetry run pytest src/tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

check: lint test ## Run all checks (lint + test)
	@echo "$(GREEN)All checks passed!$(NC)"

clean: ## Clean generated files and caches
	@echo "$(BLUE)Cleaning...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-outputs: ## Clean output directories
	@echo "$(BLUE)Cleaning outputs...$(NC)"
	rm -rf outputs/
	@echo "$(GREEN)Outputs cleaned!$(NC)"

run-baseline: ## Run baseline experiment
	@echo "$(BLUE)Running baseline experiment...$(NC)"
	python -m src.experiments.exp_baseline \
		--output-dir outputs/baseline \
		--g 9.81 \
		--L 1.0 \
		--m 1.0 \
		--t-max 10.0 \
		--n-points 10000
	@echo "$(GREEN)Baseline experiment complete!$(NC)"

run-inverse: ## Run single inverse PINN experiment
	@echo "$(BLUE)Running inverse PINN experiment...$(NC)"
	python -m src.experiments.exp_inverse_single \
		--dissipation viscous \
		--n-epochs 5000 \
		--output-dir outputs/inverse_single
	@echo "$(GREEN)Inverse PINN experiment complete!$(NC)"

run-inverse-passivity: ## Run single inverse PINN with passivity constraint
	@echo "$(BLUE)Running inverse PINN with passivity...$(NC)"
	poetry run python -m src.experiments.exp_inverse_single \
		--seed 1337 \
		--use-passivity \
		--weight-passivity 1.0 \
		--n-epochs 5000 \
		--output-dir outputs/inverse_passivity
	@echo "$(GREEN)Inverse PINN with passivity complete!$(NC)"

run-ensemble: ## Run ensemble experiment
	@echo "$(BLUE)Running ensemble experiment...$(NC)"
	python -m src.experiments.exp_inverse_ens \
		--n-models 5 \
		--theta0 30.0 \
		--damping 0.05 \
		--n-sparse 20 \
		--noise 0.01 \
		--dissipation viscous \
		--n-epochs 2000 \
		--output-dir outputs/ensemble \
		--seed 1337
	@echo "$(GREEN)Ensemble experiment complete!$(NC)"

run-all: ## Run all experiments (baseline + inverse + ensemble)
	@echo "$(BLUE)Running all experiments...$(NC)"
	bash scripts/run_all.sh
	@echo "$(GREEN)All experiments complete!$(NC)"

figs: ## Generate all figures from existing results
	@echo "$(BLUE)Generating figures...$(NC)"
	@echo "$(YELLOW)Note: Ensure experiments have been run first$(NC)"
	@echo "$(GREEN)Figures are generated during experiment runs$(NC)"

tensorboard: ## Launch tensorboard
	@echo "$(BLUE)Launching tensorboard...$(NC)"
	poetry run tensorboard --logdir outputs/

dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	poetry shell

requirements: ## Generate requirements.txt from pyproject.toml
	@echo "$(BLUE)Generating requirements.txt...$(NC)"
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	@echo "$(GREEN)requirements.txt generated!$(NC)"

info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "  Name:    pinn-passivity-paper"
	@echo "  Python:  >=3.10"
	@echo "  Seed:    1337 (default)"
	@echo ""
	@echo "$(BLUE)Key Dependencies:$(NC)"
	@poetry show --tree | head -20

# Development helpers
watch-test: ## Watch tests (requires pytest-watch)
	@echo "$(BLUE)Watching tests...$(NC)"
	poetry run ptw src/tests/

pre-commit: ## Run pre-commit hooks manually
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	poetry run pre-commit run --all-files
	@echo "$(GREEN)Pre-commit complete!$(NC)"

