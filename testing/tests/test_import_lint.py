"""The import lint, itself a test.

The hexagon only holds if it is enforced. ``atlas/domain`` and ``atlas/ports`` are pure: they
import no framework and no client, and they never import an outer ring. This is the test that
keeps "architecture is a testing lever" true across every later phase instead of eroding silently.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PURE_LAYERS = ("backend/atlas/domain", "backend/atlas/ports")

# One named file inside `backend/atlas/adapters` (an otherwise framework heavy layer, so it is not
# folded into PURE_LAYERS above) whose OWN docstring claims purity ("No framework import (no
# langgraph/mcp/fastapi/opentelemetry/httpx/psycopg)"): before this fix round nothing machine
# checked that claim, only PURE_LAYERS' domain/ports coverage did (SP6 task 2 review, Important 3).
PURE_FILES = ("backend/atlas/adapters/trace_translation.py",)

# Frameworks and clients the pure rings must never touch (the re layered allowlist).
FORBIDDEN_TOPLEVEL = {
    "langgraph", "langchain", "langchain_core", "mcp", "anthropic", "openai",
    "psycopg", "fastapi", "sqlalchemy", "httpx",
}

# `opentelemetry` is the one framework FORBIDDEN_TOPLEVEL above never had to name (no pure layer
# needed it before the OTel adapter existed); trace_translation.py's own docstring lists it
# explicitly as forbidden (its sibling, `otel_tracer.py`, is the file that legitimately imports it),
# so PURE_FILES' gate must check for it too.
FORBIDDEN_PURE_FILE_TOPLEVEL = FORBIDDEN_TOPLEVEL | {"opentelemetry"}
# Outer rings a pure layer must not depend on (dependencies point inward only).
FORBIDDEN_ATLAS_PREFIXES = ("atlas.orchestration", "atlas.adapters", "atlas.mcp_servers")

# The seam between the two harnesses (property 5): the agent harness is the product and must
# never import the eval harness. The eval harness reads the agent through its ports, and the
# dependency is one way. A backend import of `evals` (the lane package: evalkit/drift/inference_oracle) or any
# `testing.*` would wire the test rig into the runtime, the conflation this boundary guards against.
FORBIDDEN_EVAL_TOPLEVEL = ("evals", "evalkit", "drift", "inference_oracle", "testing")


def _resolve_relative(path: pathlib.Path, level: int, module: str | None) -> str:
    """Resolve a relative import (``from .`` / ``from ..``) to its absolute dotted path.

    `from ..orchestration import x` inside ``backend/atlas/domain/foo.py`` resolves to
    ``atlas.orchestration``, so the SAME outer-ring / framework checks apply. Without this the
    relative form would be silently skipped and a pure layer could reach outward through it."""
    pkg = list(path.relative_to(ROOT / "backend").with_suffix("").parts)[:-1]  # the file's package
    up = pkg[: len(pkg) - (level - 1)]                                          # climb one per extra dot
    return ".".join([*up, module]) if module else ".".join(up)


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    yield node.module, node.lineno
            else:  # a relative import: resolve to absolute so the layering rules are not evaded
                yield _resolve_relative(path, node.level, node.module), node.lineno


def _pure_files():
    for layer in PURE_LAYERS:
        yield from (ROOT / layer).rglob("*.py")


def _backend_files():
    yield from (ROOT / "backend/atlas").rglob("*.py")


def test_pure_layers_import_no_framework_or_client():
    violations = []
    for path in _pure_files():
        for module, lineno in _imports(path):
            if module.split(".")[0] in FORBIDDEN_TOPLEVEL:
                violations.append(f"{path.relative_to(ROOT)}:{lineno} imports {module}")
    assert not violations, "pure layer leaked a framework/client import:\n" + "\n".join(violations)


def test_trace_translation_module_stays_pure():
    """`trace_translation.py` claims, in its own module docstring, to import no framework at all.
    PURE_LAYERS above intentionally does not cover `backend/atlas/adapters` (most adapters legitimately
    import frameworks, e.g. `otel_tracer.py` imports opentelemetry), so without this dedicated,
    single file check that purity claim rested on discipline alone, not a gate (SP6 task 2 review,
    Important 3). Machine checked here the same way the domain/ports layers already are."""
    violations = []
    for rel in PURE_FILES:
        path = ROOT / rel
        for module, lineno in _imports(path):
            if module.split(".")[0] in FORBIDDEN_PURE_FILE_TOPLEVEL:
                violations.append(f"{path.relative_to(ROOT)}:{lineno} imports {module}")
    assert not violations, "trace_translation.py leaked a framework import:\n" + "\n".join(violations)


def test_pure_layers_do_not_import_outer_rings():
    violations = []
    for path in _pure_files():
        for module, lineno in _imports(path):
            if module.startswith(FORBIDDEN_ATLAS_PREFIXES):
                violations.append(f"{path.relative_to(ROOT)}:{lineno} imports {module}")
    assert not violations, "pure layer depends on an outer ring (deps must point inward):\n" + "\n".join(violations)


def test_agent_harness_never_imports_the_eval_harness():
    """The product must not import the test rig (the seam between the two harnesses, property 5)."""
    violations = []
    for path in _backend_files():
        for module, lineno in _imports(path):
            if module.split(".")[0] in FORBIDDEN_EVAL_TOPLEVEL:
                violations.append(f"{path.relative_to(ROOT)}:{lineno} imports {module}")
    assert not violations, "agent harness leaked an eval-harness import:\n" + "\n".join(violations)


def test_relative_imports_resolve_to_absolute_so_the_layers_are_not_evaded():
    # The blind spot this closes: a pure-layer file reaching outward via `from ..` would be skipped by
    # a level==0-only matcher. Resolution maps it back to the absolute path the layering rules catch.
    dom = ROOT / "backend/atlas/domain/example.py"
    assert _resolve_relative(dom, 1, "guard") == "atlas.domain.guard"
    assert _resolve_relative(dom, 2, "orchestration.atlas_graph") == "atlas.orchestration.atlas_graph"
    assert _resolve_relative(dom, 2, "orchestration").startswith(FORBIDDEN_ATLAS_PREFIXES)
