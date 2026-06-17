"""ASGI entrypoint for the Atlas product API, wired for the hermetic/replay lane.

`uv run uvicorn atlas.server:app` serves the chat + auth edge against the replayed gateway and an
in memory actions backend: no keys, no egress, the app the Playwright E2E lane boots.
The container runs `NullTracer` by default; `ATLAS_TRACING=otel` opts into the real OTel backed
adapter (`_tracer`), shared by BOTH `build_atlas_graph` and `make_chat_app` (SP6 task 2: one tracer
instance per process, so the graph's own span tree and chat_app's ttft mark land on the same trace).
State is in process: a restart drops pending confirmations, and more than one worker breaks resume.
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from contract_tools.loader import contract_versions
from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory, fixture_kit
from replay.gateway import CassetteMiss, GatewayChatModel
from tracing import NullTracer

from atlas import metrics
from atlas.adapters.label_store import LabelStore
from atlas.chat_app import make_chat_app
from atlas.config import AtlasSettings
from atlas.domain.accounts import apply_write
from atlas.domain.actions import ActionsBackend
from atlas.label_routes import build_label_router
from atlas.logging_setup import configure_logging
from atlas.mcp_servers.tool_surface import mcp_tool_surface
from atlas.orchestration.atlas_graph import build_atlas_graph, select_retriever
from atlas.persistence.checkpointer import checkpointer_kind, open_postgres_checkpointer, postgres_dsn

_log = logging.getLogger("atlas.server")

# SP8 Task 4 (label collection half, pulled early): the HITL adjudication page's own backend route
# needs two local paths, neither yet threaded through `AtlasSettings` (D10's dataclass grows in
# place as more of its scope lands, per that module's own docstring; this is a "captured but not
# threaded" seam for now, the same boundary `_fallback_gateway`'s own `ATLAS_FALLBACK_MODEL` read
# draws until a later task needs it in `config_hash()`'s identity). `ATLAS_LABEL_ITEMS_PATH`
# defaults to the committed, clearly marked fixture (`label_items.fixture.jsonl`, `"source":
# "fixture"` on every row) so a fresh `docker compose up` can label something immediately; an
# operator points it at the REAL generated set (`testing/harness/labeling`, `task label:generate`)
# to run an actual session. `ATLAS_LABEL_STORE_PATH` is the local path standing in for the S3 label
# prefix (D30; real S3/R2 sync is late binding, never a prerequisite) -- `var/` is gitignored, never
# committed, since it is runtime state, not corpus/index data.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LABEL_ITEMS = _REPO_ROOT / "testing" / "harness" / "dataset_tools" / "label_items.fixture.jsonl"
_DEFAULT_LABEL_STORE = _REPO_ROOT / "var" / "labels" / "adjudicator_labels.jsonl"


# The "atlas" logger tree attachment used to live here directly (a plain text StreamHandler,
# SP3/SP4); SP6 task 4 moved it into `atlas.logging_setup.configure_logging` (structured JSON to
# stdout, trace/span correlation, redaction of structured extras) -- see that module's own docstring
# for the full design. This call site (below, in `create_app`) is unchanged: still the ONE place the
# "atlas" logger tree is configured, still idempotent, still not root.

# The default cassette dir now lives on AtlasSettings (atlas.config, anchored the same way, on that
# module's own file), the one place `ATLAS_CASSETTES` is read.

# ATLAS_MODE maps 1:1 to GatewayMode, no translation, so the env value never lies about behaviour:
#   replay (default): zero egress, only committed prompts answer (a miss hard fails)
#   record          : answer via the live provider (default Ollama) AND persist to the cassette dir
#   live            : answer via the live provider, persist nothing (the eval lane)
_KNOWN_MODES = ("replay", "record", "live")


def _resolve_mode(mode: str) -> str:
    """Validates `AtlasSettings.atlas_mode` (`ATLAS_MODE`) against the known set. `settings` is now
    the one place that reads the env var; this only keeps the fail fast validation a typo needs, so
    it does not fall through to the branch that reaches a live provider."""
    if mode not in _KNOWN_MODES:
        raise RuntimeError(f"unknown ATLAS_MODE={mode!r}; expected one of {'|'.join(_KNOWN_MODES)}")
    return mode


# provider -> (SDK module that must import, dependency group that installs it). The container image
# syncs only the `ollama` group (see docker-compose.yml / backend/Dockerfile), so switching
# MODEL_PROVIDER to anthropic/openai without syncing its group is a config error, not a code one.
_PROVIDER_SDK = {
    "ollama": ("langchain_ollama", "ollama"),
    "anthropic": ("langchain_anthropic", "anthropic"),
    "openai": ("langchain_openai", "openai"),
}


def _require_provider_sdk(provider: str) -> None:
    """In live/record mode, fail fast at startup with ONE actionable line if the configured
    MODEL_PROVIDER's SDK is not importable. Without this, an unavailable provider surfaces as a raw
    ImportError traceback on the first chat turn, hanging the web edge instead of refusing to boot.
    Mirrors providers.py's default (ollama) and its provider set. `provider` is
    `AtlasSettings.model_provider` (`MODEL_PROVIDER`), read once by `settings` now."""
    import importlib.util

    if provider not in _PROVIDER_SDK:
        raise RuntimeError(
            f"unknown MODEL_PROVIDER={provider!r}; expected one of {'|'.join(_PROVIDER_SDK)}"
        )
    module, group = _PROVIDER_SDK[provider]
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(
            f"MODEL_PROVIDER={provider!r} needs the {module!r} SDK, which is not installed. "
            f"Sync its dependency group first: `uv sync --group {group}` "
            f"(the container image ships only the 'ollama' group)."
        )


def _gateway(mode: str, cassettes: Path) -> GatewayChatModel:
    if mode == "replay":
        return GatewayChatModel(model_id="claude-test", cassette_dir=cassettes, mode="replay")
    from replay.providers import build_chat_model, provider_tag

    return GatewayChatModel(model_id=provider_tag(), cassette_dir=cassettes, mode=mode, inner=build_chat_model())


def _fallback_gateway(mode: str, cassettes: Path) -> GatewayChatModel | None:
    """SP4 final fix wave (F2): the ladder's `provider_fallback` rung (`atlas_graph.py`'s
    `fallback_model`) has a real producer only when `ATLAS_FALLBACK_MODEL` names a second model,
    `provider:model_id` -- the SAME compact shape `replay.providers.provider_tag` already builds
    for the cassette key, e.g. `ATLAS_FALLBACK_MODEL=anthropic:claude-haiku-4-5-20251001`. Unset
    (the default) keeps `fallback_model=None`, today's behaviour unchanged, so `provider_fallback`
    still never fires without an explicit opt in. Meaningless in replay (a cassette miss is
    `CassetteMiss`, never routed through the ladder), so this returns None there without even
    reading the env var. Fails fast at startup, mirroring `_require_provider_sdk`'s own discipline,
    rather than surfacing a raw ImportError on the turn the primary model first needs the fallback."""
    if mode == "replay":
        return None
    raw = os.environ.get("ATLAS_FALLBACK_MODEL")
    if not raw:
        return None
    provider, sep, model_id = raw.partition(":")
    if not sep or not model_id:
        raise RuntimeError(
            f"ATLAS_FALLBACK_MODEL={raw!r} must be 'provider:model_id' (e.g. "
            "'anthropic:claude-haiku-4-5-20251001'), the same compact shape replay.providers.provider_tag builds."
        )
    if provider not in _PROVIDER_SDK:
        raise RuntimeError(
            f"ATLAS_FALLBACK_MODEL names an unknown provider {provider!r}; expected one of {'|'.join(_PROVIDER_SDK)}"
        )
    import importlib.util

    module, group = _PROVIDER_SDK[provider]
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(
            f"ATLAS_FALLBACK_MODEL={raw!r} needs the {module!r} SDK, which is not installed. "
            f"Sync its dependency group first: `uv sync --group {group}`."
        )
    from replay.providers import build_chat_model

    return GatewayChatModel(
        model_id=f"{provider}:{model_id}", cassette_dir=cassettes, mode=mode,
        inner=build_chat_model(provider, model_id),
    )


def _resolve_mcp_tools(mode: str, retriever) -> dict[str, dict] | None:
    """SP4 task 5: the MCP tool surface `atlas_graph.build_atlas_graph`'s `agent` node binds onto
    the model in live/record mode (`atlas_graph._tool_bindable`/`_generate_message`). Building it is
    cheap (four in process FastMCP servers, list their tools, no I/O) but is still skipped entirely
    in replay/hermetic mode, mirroring `select_retriever`'s own "unset stays untouched" discipline,
    so `mode == "replay"` is a genuine "never even builds it" proof (see test_chat_app.py), not just
    an unused value the graph happens to ignore."""
    if mode == "replay":
        return None
    return mcp_tool_surface(retriever)


def _otlp_exporter(endpoint: str):
    """SP6 task 3: the real OTLP HTTP wire exporter, against `endpoint + "/v1/traces"` (the
    collector's OTLP HTTP receiver path, `infra/observability/otel-collector.yaml`'s own
    `receivers.otlp.protocols.http`). `endpoint` is always the explicit `ATLAS_OTEL_ENDPOINT` value
    threaded in by `_tracer` below -- this never reads `OTEL_EXPORTER_OTLP_*` auto configuration
    (the SP6 global constraint every tracing entry point in this repo holds itself to).

    Local import, guarded: `opentelemetry-exporter-otlp-proto-http` lives ONLY in the optional
    `observability` dependency group (pyproject.toml), never installed by `task test`/`uv sync`. The
    hermetic lane's own `ATLAS_TRACING=otel` gate tests (test_otel_tracer.py) exercise this function
    for real (they monkeypatch `OtelTracer` itself, not this helper), so it must degrade gracefully,
    never raise, when the group is absent: caught here and demoted to `None`, which leaves
    `OtelTracer.__init__`'s own `exporter or ConsoleSpanExporter()` fallback in charge, exactly
    Task 1's already tested "console when nothing injected" default. When the group IS installed
    (backend/Dockerfile installs it alongside `ollama`), this constructs the real wire exporter, so
    an operator's `ATLAS_TRACING=otel` actually leaves the process, unlike Task 1/2's console only
    adapter."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        _log.warning(
            "ATLAS_TRACING=otel but the 'observability' dependency group is not installed; "
            "falling back to console export (`uv sync --group observability` to export over OTLP)."
        )
        return None
    return OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")


def _tracer(settings: AtlasSettings):
    """SP6 task 1's opt in gate: `ATLAS_TRACING=otel` (`settings.tracing`) constructs the real OTel
    backed adapter (recording only for now; the gen_ai/atlas.* translation lands in Task 2). ANY
    other value -- unset, a typo, anything else -- keeps `NullTracer`, the default everywhere in
    this codebase, deliberately permissive so a mistyped flag never accidentally opts into a real,
    network capable adapter (unlike `_resolve_mode`/`checkpointer_kind`/`select_retriever`, which
    fail fast on a typo because picking the WRONG concrete adapter there is the danger; here picking
    NO adapter is always the safe default). Constructed HERE ONLY, with a local import, so no
    hermetic/collected test path -- none of which ever sets ATLAS_TRACING -- constructs one (SP6's
    determinism constraint: the OTel adapter never enters the hermetic lane's exercised paths).

    SP6 task 3: `exporter=_otlp_exporter(...)` decides the WIRE, `OtelTracer` itself decides nothing
    about transport (Task 1's own scope boundary, unchanged) -- the real OTLP endpoint only ever
    reaches this one call site.

    SP6 task 7 (the v1 freeze): `corpus_version`/`index_build_id` thread straight from `settings`
    (already resolved off the active index's own `build_manifest.json`, task 6) to `OtelTracer`,
    which stamps them on every turn span alongside `atlas.config.hash` -- this function is the one
    place a settings sourced identity fact reaches the tracer, unchanged in shape from task 3."""
    if settings.tracing != "otel":
        return NullTracer()
    from atlas.adapters.otel_tracer import OtelTracer

    return OtelTracer(
        endpoint=settings.otel_endpoint, config_hash=settings.config_hash(),
        corpus_version=settings.corpus_version, index_build_id=settings.index_build_id,
        exporter=_otlp_exporter(settings.otel_endpoint),
    )


def create_app():
    configure_logging()
    settings = AtlasSettings.from_env()
    mode = _resolve_mode(settings.atlas_mode)
    cassettes = Path(settings.cassette_dir)
    if mode == "replay" and not cassettes.is_dir():
        # fail at startup, not on the first turn: replay with no cassettes can never answer
        raise RuntimeError(f"replay mode needs an existing cassette dir, got cassettes={cassettes}")
    if mode != "replay":
        # live/record reach a real provider: fail fast now if its SDK group was never synced
        _require_provider_sdk(settings.model_provider)
    # ATLAS_CHECKPOINTER (SP4 task 2): validated up front, same fail fast timing as mode/provider
    # above, so a typo'd value is a startup error, never a surprise on the first turn.
    ck_kind = checkpointer_kind(settings.checkpointer_kind)
    _log.info(
        "startup mode=%s cassettes=%s checkpointer=%s git_sha=%s", mode, cassettes, ck_kind, settings.git_sha
    )

    kit = fixture_kit()
    # write through: a confirmed action mutates account state, so a later read reflects it
    backend = ActionsBackend(IdFactory("ref"), writer=apply_write)
    # ATLAS_RETRIEVER (D36 tier 2, SP3 task 7): unset/"inmemory" keeps the hermetic adapter untouched;
    # "pgvector" builds the real hybrid adapter ONCE for this app's whole lifetime (never per
    # request), closed below at shutdown -- the client lifecycle a long lived httpx.Client needs.
    retriever = select_retriever(settings.retriever_kind)
    # The graph is ALWAYS compiled with the in memory saver first (identical to the unset env
    # default), even when ck_kind == "postgres": AsyncPostgresSaver captures the running event loop
    # at construction (persistence/checkpointer.py's module docstring), which does not exist yet at
    # this point in a plain `create_app()` call. The lifespan below swaps it in once uvicorn's loop
    # is actually running, before any request is served (CompiledStateGraph.checkpointer is a plain
    # instance attribute read fresh per invoke, so the swap is safe post compile).
    # SP6 task 2: constructed exactly ONCE and shared with `make_chat_app` below (never a second,
    # independent tracer) -- `OtelTracer` keys its span tree by its own per instance seq counter, so
    # chat_app's ttft mark and the graph's own turn/agent/guard spans must share the SAME instance
    # to land on the SAME trace.
    tracer = _tracer(settings)
    graph = build_atlas_graph(
        _gateway(mode, cassettes), IdFactory("idem"), backend, new_checkpointer(),
        retriever=retriever, mcp_tools=_resolve_mcp_tools(mode, retriever),
        fallback_model=_fallback_gateway(mode, cassettes), tracer=tracer,
    )
    app = make_chat_app(kit.clock, graph, tracer=tracer, cors_origins=["http://localhost:5173"])

    # SP8 Task 4 (label collection half, pulled early): the HITL adjudication page's own backend
    # route, registered on the SAME app the product chat routes live on (one FastAPI process, one
    # /healthz, one /metrics). `kit.clock` is the SAME frozen/real clock the rest of this process
    # already shares (`fixture_kit()`, above) -- `created_at` on every label is never a second,
    # independent clock read.
    label_items_path = Path(os.environ.get("ATLAS_LABEL_ITEMS_PATH", _DEFAULT_LABEL_ITEMS))
    label_store_path = Path(os.environ.get("ATLAS_LABEL_STORE_PATH", _DEFAULT_LABEL_STORE))
    app.include_router(
        build_label_router(items_path=label_items_path, store=LabelStore(label_store_path, kit.clock))
    )

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        pg_conn = None
        if ck_kind == "postgres":
            saver = await open_postgres_checkpointer(postgres_dsn())
            graph.checkpointer = saver
            pg_conn = saver.conn
        yield
        # shutdown, duck typed: InMemoryRetriever (the default) holds no resource and has no close();
        # only PgvectorRetriever's two httpx clients need this. `make_chat_app`'s signature stays
        # unchanged (no lifespan parameter): the app it returns has no lifespan wired yet, so setting
        # `router.lifespan_context` here is how the ASGI lifespan protocol is read (Starlette resolves
        # it per lifespan cycle, not only at Router construction; see Router.lifespan()).
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
        if pg_conn is not None:
            await pg_conn.close()

    app.router.lifespan_context = _lifespan

    # SP6 task 5: the ONLY writer `atlas.metrics`'s request counter ever has, wrapping every route
    # (healthz/metrics included) so `atlas_http_requests_total` reflects the whole edge, not just
    # /chat. An exception that escapes `call_next` unhandled (no registered exception handler, e.g.
    # CassetteMiss below IS handled and never reaches this except) is recorded as 5xx before
    # reraising, so it still counts toward the D29 error rate rule instead of vanishing silently.
    @app.middleware("http")
    async def _count_requests(request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            metrics.record_request(500)
            raise
        metrics.record_request(response.status_code)
        return response

    @app.get("/healthz")
    def healthz() -> dict:
        # fixed contract: CI readiness, the compose healthcheck, and depends_on all read this shape
        return {
            "status": "ok",
            "mode": mode,
            "cassettes": cassettes.is_dir(),
            "git_sha": settings.git_sha,
        }

    @app.get("/version")
    def version() -> dict:
        # Release identity (D37, SP6 task 6): every field here is SURFACED, never computed in this
        # handler. `git_sha` is the SAME `settings.git_sha` read `/healthz` already returns (one
        # GIT_SHA env read, resolved once by AtlasSettings.from_env()). `contracts` is a direct call
        # to `contract_tools.loader.contract_versions()` (`{family: x-contract-version}` for all
        # four families, read fresh from contracts/*/schema.json). `corpus_version`/
        # `index_build_id` are `AtlasSettings`' own fields, derived once at startup from the active
        # index's `build_manifest.json` off `ATLAS_INDEX_DIR` (`config.py::_read_index_manifest`) --
        # never a second, independent read of that file here.
        return {
            "git_sha": settings.git_sha,
            "contracts": contract_versions(),
            "corpus_version": settings.corpus_version,
            "index_build_id": settings.index_build_id,
        }

    @app.get("/metrics")
    def metrics_endpoint() -> PlainTextResponse:
        # `retriever.breaker` (pgvector_retriever.py's own inspection only property) is absent on
        # InMemoryRetriever (the hermetic default): `getattr(..., None)` reads that as "no breaker
        # to report," never a crash. Read fresh on every scrape, never cached (metrics.render's own
        # docstring), the same "no wall clock, no memoized state" discipline this codebase holds its
        # tracing spans to.
        body = metrics.render(
            breaker=getattr(retriever, "breaker", None),
            registry_version=settings.registry_version,
            index_dir=settings.index_dir,
        )
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4")

    async def _cassette_miss(request, exc: CassetteMiss) -> JSONResponse:
        # a miss in replay is a serving config problem (missing recording), not a client error
        return JSONResponse(status_code=503, content={"error": f"replay cassette miss: {exc}"})

    app.add_exception_handler(CassetteMiss, _cassette_miss)
    return app


app = create_app()
