from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def offending_imports(source: str, where: str) -> list[str]:
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                 else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        if any(name == "evals" or name.startswith("evals.") for name in names):
            offenders.append(f"{where}:{node.lineno}")
    return offenders


def test_the_import_walker_sees_a_module_that_imports_the_measuring_package() -> None:
    # The rule below can pass vacuously, so this hands the walker something it must flag.
    assert offending_imports("import evals", "a") == ["a:1"]
    assert offending_imports("import evals.stats as s", "b") == ["b:1"]
    assert offending_imports("from evals import ir_metrics", "c") == ["c:1"]
    assert offending_imports("from evals.stats import interval", "d") == ["d:1"]
    assert offending_imports("import atlas.contracts", "e") == []


def test_the_pipeline_package_never_imports_the_measuring_package() -> None:
    # The dependency arrow points one way only, so `evals` measures a pipeline that cannot
    # see it. Otherwise a scoring function reachable from the pipeline acquires an input it
    # was designed not to have.
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "atlas").rglob("*.py")):
        offenders.extend(
            offending_imports(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))
        )
    assert offenders == [], f"pipeline imports the measuring apparatus at {offenders}"
