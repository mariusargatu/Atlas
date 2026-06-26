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

# Frameworks and clients the pure rings must never touch (the re layered allowlist).
FORBIDDEN_TOPLEVEL = {
    "langgraph", "langchain", "langchain_core", "mcp", "anthropic", "openai",
    "psycopg", "fastapi", "sqlalchemy", "httpx",
}
# Outer rings a pure layer must not depend on (dependencies point inward only).
FORBIDDEN_ATLAS_PREFIXES = ("atlas.orchestration", "atlas.adapters", "atlas.mcp_servers")

# The seam between the two harnesses (property 5): the agent harness is the product and must
# never import the eval harness. The eval harness reads the agent through its ports, and the
# dependency is one way. A backend import of `evals` (the lane package: evalkit/drift/inference_oracle) or any
# `testing.*` would wire the test rig into the runtime, the conflation the article warns about.
FORBIDDEN_EVAL_TOPLEVEL = ("evals", "evalkit", "drift", "inference_oracle", "testing")


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module, node.lineno


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
