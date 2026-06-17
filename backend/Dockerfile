# Atlas backend, the FastAPI edge (chat + auth) over the LangGraph agent.
# Build context is the repo ROOT (needs testing/harness/ + committed cassettes + pyproject).
#   docker build -f backend/Dockerfile --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t atlas-api .
# Runtime deps only (+ the ollama group so ATLAS_MODE=live works, + the observability group so
# ATLAS_TRACING=otel can actually export over OTLP, SP6 task 3). No pytest, no provider keys.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH=/app/backend:/app/testing/harness:/app \
    TZ=UTC LC_ALL=C PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
    UV_CACHE_DIR=/app/.uv-cache
    # The `atlas` user below is created with --no-create-home, so its passwd $HOME (/home/atlas)
    # exists as a path but not as a directory it can write to; `uv run` defaults its cache dir under
    # $HOME and fails closed with a permission error on first use otherwise (found live: SP3 task 7,
    # the first task to actually boot this image's CMD via `docker compose up`). Anchored under /app
    # (not /tmp) so the existing `chown -R atlas:atlas /app` below also covers the cache dir the
    # build-time `uv sync` (run as root) creates -- /tmp alone still 403s: the root-owned cache dir
    # `uv sync` leaves behind there is not writable by `atlas` at runtime either.

# Install deps first (cache layer), no project install, pythonpath layout.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --group ollama --group observability

# GIT_SHA changes on every commit, so it lives BELOW the dependency layer: a new sha invalidates only
# the app COPY below it, never the expensive `uv sync` cache above. (Put it above the RUN and every
# build would re-sync all deps for a one-line label change.)
ARG GIT_SHA
ENV GIT_SHA=${GIT_SHA:-dev}

# Source the server needs: the app package, the harness (gateway/determinism/models), cassettes.
COPY backend/atlas ./backend/atlas
COPY testing/harness ./testing/harness
# SP6 task 3, found live: trace_translation.py reads contracts/trace/span_inventory.json and
# contracts/trace/schema.json at import time (its own fail closed check and the trace_version stamp),
# anchored on ITS OWN file path (parents[3] resolves to /app here), not testing/harness/ or
# backend/atlas/ -- ATLAS_TRACING=otel crashed the whole container at import without this, since
# nothing had ever actually built and run this image with tracing on before this task.
COPY contracts ./contracts

# Drop root before serving traffic.
RUN useradd --system --no-create-home --uid 10001 atlas && chown -R atlas:atlas /app
USER atlas

EXPOSE 8000
# GET /healthz via urllib, not curl: not present in the slim base image.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)"]
# ATLAS_MODE=replay (default, zero egress) | live (Ollama via OLLAMA_BASE_URL).
# --proxy-headers: nginx sets X-Forwarded-For in front of this container; trust it for client IPs.
CMD ["uv", "run", "--no-sync", "uvicorn", "atlas.server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
