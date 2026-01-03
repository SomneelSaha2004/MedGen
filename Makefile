# =============================================================================
# MedGen Makefile
# =============================================================================
# Common development commands for the MedGen project
# =============================================================================

.PHONY: help install dev backend frontend test lint format clean docker docker-down

# Default target
help:
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║                    🏥 MedGen Commands                         ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  make install      - Install all dependencies"
	@echo "  make dev          - Start development servers (backend + frontend)"
	@echo "  make backend      - Start backend server only"
	@echo "  make frontend     - Start frontend server only"
	@echo "  make test         - Run all tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean generated files"
	@echo "  make docker       - Build and run with Docker"
	@echo "  make docker-down  - Stop Docker containers"
	@echo "  make eval         - Run evaluation pipeline"
	@echo "  make privacy      - Run privacy assessment"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing Python dependencies..."
	pip install uv
	uv sync
	@echo "📦 Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✅ Installation complete!"

# Development mode - run both servers
dev:
	@echo "🚀 Starting development servers..."
	./run.sh

# Backend only
backend:
	@echo "🚀 Starting backend server..."
	uv run python backend.py

# Frontend only
frontend:
	@echo "🎨 Starting frontend server..."
	cd frontend && npm start

# Run tests
test:
	@echo "🧪 Running backend tests..."
	uv run pytest tests/ -v
	@echo "🧪 Running frontend tests..."
	cd frontend && npm test -- --watchAll=false

# Lint code
lint:
	@echo "🔍 Linting Python code..."
	uv run flake8 *.py evals/
	@echo "🔍 Linting frontend code..."
	cd frontend && npm run lint

# Format code
format:
	@echo "✨ Formatting Python code..."
	uv run black *.py evals/
	uv run isort *.py evals/
	@echo "✨ Formatting frontend code..."
	cd frontend && npm run format

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf data/chroma_db data/generated
	rm -rf results multi_dataset_results
	rm -rf frontend/build frontend/node_modules/.cache
	rm -f *.log *.csv !datasets/*.csv
	rm -f evals/models/*.png
	@echo "✅ Cleanup complete!"

# Docker commands
docker:
	@echo "🐳 Building and starting Docker containers..."
	docker-compose up --build -d
	@echo "✅ Containers started!"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:5000"

docker-down:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down
	@echo "✅ Containers stopped!"

# Run evaluation pipeline
eval:
	@echo "🔬 Running evaluation pipeline..."
	uv run python basic_eval_pipeline.py

# Run multi-dataset evaluation
eval-multi:
	@echo "🔬 Running multi-dataset evaluation..."
	uv run python multi_dataset_pipeline.py

# Run privacy assessment
privacy:
	@echo "🔒 Running privacy assessment..."
	uv run python anonymeter_privacy_eval.py

# Create required directories
setup-dirs:
	@mkdir -p data/features data/chroma_db data/generated results
	@echo "✅ Directories created!"

# Check environment
check-env:
	@if [ ! -f .env ]; then \
		echo "⚠️  No .env file found. Creating from template..."; \
		cp .env.example .env; \
		echo "📝 Please edit .env and add your OPENAI_API_KEY"; \
	else \
		echo "✅ .env file exists"; \
	fi
