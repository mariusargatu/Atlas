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
# dependency is one way. A backend import of any harness package would wire the test rig into the
# runtime, the conflation this boundary guards against.
#
# Derived from the harness tree itself (`_harness_packages`), never a hand kept list. The list this
# replaced named five modules, three of which (`evalkit`, `drift`, `inference_oracle`) are
# subpackages of `evals/`, so `module.split(".")[0]` was always `evals` and they could never match;
# meanwhile every OTHER top level harness package (`rag_tools`, `corpus_tools`, `quality`, `matrix`,
# ...) was unguarded, which is exactly the direction three adapter docstrings cite this rule to
# justify duplicating code against.
#
# `SHARED_HARNESS_PACKAGES` is the explicit, small allowlist of harness packages the product
# legitimately depends on: determinism sources/canonicalisation, the replay gateway, the tracer
# protocol, and contract loading. These are shared machinery, not the eval rig. Anything else under
# `testing/harness/` is forbidden to `backend/atlas`; add to this set deliberately, never by
# accident.
SHARED_HARNESS_PACKAGES = frozenset({"contract_tools", "determinism", "replay", "tracing"})


def _harness_packages() -> frozenset[str]:
    """Every importable top level package under `testing/harness/` (the directories carrying an
    `__init__.py`, which is what `PYTHONPATH=testing/harness` actually exposes)."""
    harness = ROOT / "testing/harness"
    return frozenset(p.name for p in harness.iterdir() if p.is_dir() and (p / "__init__.py").is_file())


# `testing` itself is added on top of the derived set: `testing.tests` is importable under
# `PYTHONPATH=.` and is never shared machinery.
FORBIDDEN_EVAL_TOPLEVEL = (_harness_packages() - SHARED_HARNESS_PACKAGES) | {"testing"}


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


def test_the_harness_seam_covers_every_harness_package_it_does_not_allowlist():
    """The gate above is only as good as the set it matches against. This pins the derivation
    itself: every top level package under `testing/harness/` is either forbidden to the backend or
    named in `SHARED_HARNESS_PACKAGES`, with nothing falling through the gap. The hand kept tuple
    this replaced left `rag_tools`, `corpus_tools`, `quality`, `matrix` and eight more unguarded
    while three adapter docstrings cited the rule as the reason they duplicate code."""
    packages = _harness_packages()
    assert "rag_tools" in FORBIDDEN_EVAL_TOPLEVEL, "the seam must cover rag_tools (it did not before)"
    assert "corpus_tools" in FORBIDDEN_EVAL_TOPLEVEL
    assert SHARED_HARNESS_PACKAGES <= packages, "an allowlisted package no longer exists in the harness"
    assert packages <= (FORBIDDEN_EVAL_TOPLEVEL | SHARED_HARNESS_PACKAGES), "a harness package is unguarded"


def test_the_allowlist_names_only_packages_the_backend_actually_imports():
    """An allowlist that outlives its need is a hole. Every name in `SHARED_HARNESS_PACKAGES` must
    be justified by a real backend import; drop one that no longer is."""
    imported = set()
    for path in _backend_files():
        for module, _ in _imports(path):
            imported.add(module.split(".")[0])
    unused = SHARED_HARNESS_PACKAGES - imported
    assert not unused, f"allowlisted but never imported by the backend, so drop it: {sorted(unused)}"


def test_relative_imports_resolve_to_absolute_so_the_layers_are_not_evaded():
    # The blind spot this closes: a pure-layer file reaching outward via `from ..` would be skipped by
    # a level==0-only matcher. Resolution maps it back to the absolute path the layering rules catch.
    dom = ROOT / "backend/atlas/domain/example.py"
    assert _resolve_relative(dom, 1, "guard") == "atlas.domain.guard"
    assert _resolve_relative(dom, 2, "orchestration.atlas_graph") == "atlas.orchestration.atlas_graph"
    assert _resolve_relative(dom, 2, "orchestration").startswith(FORBIDDEN_ATLAS_PREFIXES)
