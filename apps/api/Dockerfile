# RAG as a Service — modular-monolith API image.
# Sovereign build: base image + wheels come from Acme internal mirrors in prod
# (no external registries). uv gives reproducible installs from the committed lockfile.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# uv (pinned in prod via internal mirror; latest here for the dev image).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# OCR engine (FR-B1) — tesseract binary + poppler (pdf2image rasterises scanned PDFs).
# eng + hin language data for Indic scanned documents. Enable at runtime with
# RAG_OCR_BACKEND=tesseract; the seam falls back to noop if this layer is absent.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-hin poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the whole workspace (monorepo) so path-based workspace deps resolve.
COPY pyproject.toml uv.lock ./
COPY libs ./libs
COPY services ./services
COPY apps ./apps
# Operational tooling (readiness probe, re-index, calibration) — run inside the network.
COPY scripts ./scripts

# Install only the API app + its transitive workspace deps.
RUN uv sync --frozen --package rag-api --no-dev
# OCR Python bindings (the rag-svc-doc-intelligence[ocr] extra) for the tesseract seam.
RUN uv pip install --no-cache pytesseract pdf2image pillow

# Scale-out coordinator client (the rag-common[scale] extra). rag_common.coordination
# import-guards `import redis`: absent ⇒ InMemoryCoordinator (single-process, today's
# default); present + RAG_REDIS_URL set ⇒ RedisCoordinator does cross-process leader
# election, so running >1 uvicorn worker is SAFE (exactly one worker drains ingest /
# fires schedulers). Baked here so a scaled deployment never fails at runtime on a
# missing module. Redis is a self-hosted internal store (like Postgres) — NOT egress-
# guarded, so this carries no sovereignty change.
RUN uv pip install --no-cache "redis>=5.0"

# NL→SQL database drivers (D10). Pure-python, sovereign-friendly DBAPIs so a tenant can connect
# ANY common RDBMS without a per-DB code change: PyMySQL (MySQL/MariaDB), pymssql (SQL Server,
# manylinux wheel bundles FreeTDS — no apt), python-oracledb (Oracle, thin mode — no client libs).
# psycopg (PostgreSQL) + the stdlib sqlite3 already ship with the base install.
RUN uv pip install --no-cache pymysql pymssql oracledb

# Azure Blob (FR-A2b, the `azure` extra of rag-clients). Installed here rather than made a core
# dependency so the library stays usable without it -- rag_clients.azure_fetcher raises a
# ConfigError naming this extra when it is absent, instead of an ImportError from a worker
# thread. The image carries it because a deployment that offers the connector in its console
# should not fail the first time someone uses it.
RUN uv pip install --no-cache "azure-storage-blob>=12.30"

# Chemical-structure canonicalization (Slice 6 Phase C, the `chem` extra of rag-clients). RDKit
# canonicalizes SMILES so an equivalent-but-differently-drawn molecule matches at retrieval on a
# chemistry-profile KB. BSD-licensed, pure-local, no runtime egress (sovereign). Installed here
# rather than made a core dependency so the library stays usable without it -- the runtime seam
# (rag_clients.chem_normalize) falls back to a no-op NullMoleculeNormalizer when absent, so a
# deployment that never enables RAG_CHEM_NORMALIZATION_ENABLED carries no behavioural change. The
# image carries it because a chemistry KB with the flag on needs it to actually normalize.
RUN uv pip install --no-cache "rdkit>=2024.3.1"

# Optional: PaddleOCR (PP-OCRv4) — stronger sovereign CPU OCR for low-quality scans + Indic
# scripts. OFF by default (paddlepaddle is large). Enable with:
#   docker build --build-arg INSTALL_PADDLEOCR=true ...
# NOTE: pin 2.7.x (2.8+ needs a C toolchain); paddle needs setuptools. Use a paddlepaddle
# wheel matching the TARGET host's glibc/CPU (or the official paddle base image), and
# pre-mirror the PP-OCR models into RAG_OCR_MODEL_DIR for an air-gapped deployment.
ARG INSTALL_PADDLEOCR=false
RUN if [ "$INSTALL_PADDLEOCR" = "true" ]; then \
      uv pip install --no-cache setuptools numpy "paddleocr>=2.7,<2.8" "paddlepaddle>=2.6,<3"; \
    fi

# Sovereign-clean image: the third-party MCP SDK ships a Claude-Desktop install helper
# (mcp/cli/claude.py) that this service never imports — it uses only mcp.server.*. Remove
# it so the delivered image carries no assistant-tooling references. Harmless to runtime.
RUN find /app/.venv -path '*/mcp/cli/claude.py' -delete 2>/dev/null || true \
 && find /app/.venv -path '*/mcp/cli/__pycache__/claude*.pyc' -delete 2>/dev/null || true

EXPOSE 8000

# Non-root runtime.
RUN useradd --create-home --uid 10001 rag && chown -R rag:rag /app
USER rag

CMD ["uv", "run", "--no-dev", "rag-api"]
