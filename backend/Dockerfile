# Atlas backend, the FastAPI edge (chat + auth) over the LangGraph agent.
# Build context is the repo ROOT (needs testing/harness/ + committed cassettes + pyproject).
#   docker build -f backend/Dockerfile -t atlas-api .
# Runtime deps only (+ the ollama group so ATLAS_MODE=live works). No pytest, no provider keys.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH=/app/backend:/app/testing/harness:/app \
    TZ=UTC LC_ALL=C PYTHONHASHSEED=0 OMP_NUM_THREADS=1

# Install deps first (cache layer), no project install, pythonpath layout.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --group ollama

# Source the server needs: the app package, the harness (gateway/determinism/models), cassettes.
COPY backend/atlas ./backend/atlas
COPY testing/harness ./testing/harness

EXPOSE 8000
# ATLAS_MODE=replay (default, zero egress) | live (Ollama via OLLAMA_BASE_URL).
CMD ["uv", "run", "--no-sync", "uvicorn", "atlas.server:app", "--host", "0.0.0.0", "--port", "8000"]
