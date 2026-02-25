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

# Install ALL build + runtime dependencies upfront
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran \
    libopenblas-dev liblapack-dev libgomp1 \
    libmagic-dev poppler-utils tesseract-ocr \
    git curl wget && \
    rm -rf /var/lib/apt/lists/*

# Use minimal requirements to avoid OpenTelemetry resolution issues  
COPY requirements-minimal.txt ./

# Upgrade pip FIRST
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core deps from minimal requirements
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements-minimal.txt

# Try to install pathway + sentence-transformers (optional, graceful fail)
RUN pip install --no-cache-dir --default-timeout=1000 \
    "pathway[xpack-llm]>=0.14.0" 2>/dev/null || \
    pip install --no-cache-dir "pathway>=0.14.0" 2>/dev/null || \
    echo "⚠ Pathway unavailable - will use compat layer"

RUN pip install --no-cache-dir --default-timeout=2000 \
    "sentence-transformers>=2.2.0" 2>/dev/null || \
    echo "⚠ Sentence-transformers unavailable - will use BM25 fallback"

# ── Stage 2: Runtime ───────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="QuantGuard Team"
LABEL description="Quantum-Enhanced Fraud Detection — Hack For Green Bharat"

WORKDIR /app

# Copy ALL site-packages from builder (where pip installed everything)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install ONLY runtime libraries (not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 poppler-utils tesseract-ocr \
    libopenblas0 liblapack3 libgomp1 && \
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
