# ════════════════════════════════════════════════════════════════════════════
#  QuantGuard — Dockerfile
#  Quantum-Enhanced Fraud Detection for Green Bharat
# ════════════════════════════════════════════════════════════════════════════
#
#  Multi-stage build:
#    Stage 1 (builder) — install heavy deps, compile wheels
#    Stage 2 (runtime) — slim image with only runtime deps
#
#  Usage:
#    docker build -t quantguard .
#    docker run -p 8000:8000 --env-file .env quantguard
#
#  Or via docker-compose:
#    docker compose up --build
# ════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# System dependencies for building native wheels + unstructured parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libmagic-dev poppler-utils tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install "pathway[xpack-llm]==0.14.7" && \
    pip install --no-cache-dir --prefix=/install sentence-transformers>=2.2.0

# ── Stage 2: Runtime ───────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="QuantGuard Team"
LABEL description="Quantum-Enhanced Fraud Detection — Hack For Green Bharat"

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Runtime libraries needed by unstructured / xPack parsers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 poppler-utils tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Create data directories
RUN mkdir -p /app/data/policies /app/static /app/data/persistence

# Copy application source
COPY *.py ./
COPY static/ ./static/
COPY data/policies/ ./data/policies/
COPY data/transactions.jsonl ./data/transactions.jsonl

# Create empty alert file
RUN touch /app/data/high_risk_alerts.jsonl

# Environment defaults (override with --env-file or -e)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QUANTGUARD_HOST=0.0.0.0 \
    QUANTGUARD_PORT=8000 \
    PATHWAY_PERSISTENCE_DIR=/app/data/persistence

# Expose API port + HTTP REST connector port (market data)
EXPOSE 8000 9090

# Health check against the API
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run the API server (which starts the Pathway engine internally)
CMD ["python", "main_api.py"]
