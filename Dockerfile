# =============================================================================
# MedGen Backend Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build stage
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# -----------------------------------------------------------------------------
# Stage 2: Production stage
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r medgen && useradd -r -g medgen medgen

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY backend.py .
COPY generate_data.py .
COPY rag.py .
COPY preprocess.py .
COPY dquery.py .
COPY basic_eval_pipeline.py .
COPY anonymeter_privacy_eval.py .
COPY multi_dataset_pipeline.py .
COPY scaling_pipeline.py .

# Copy evaluation models
COPY evals/ ./evals/

# Copy anonymeter submodule if needed
COPY anonymeter/ ./anonymeter/

# Create necessary directories
RUN mkdir -p data/features data/chroma_db data/generated results

# Set ownership
RUN chown -R medgen:medgen /app

# Switch to non-root user
USER medgen

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "backend.py"]
