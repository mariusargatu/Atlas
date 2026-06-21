"""Shared write and report epilogue for the eval studies (judge, benchmark, ...).

Every study's `__main__.py` writes its rendered text to a committed artifact and prints
where it landed. Centralizing that stops the mkdir/write/print incantation from drifting
between studies the way the graph wiring incantation would without `scaffold.py`.
"""
from __future__ import annotations

from pathlib import Path


def write_artifacts(paths_and_content: list[tuple[Path, str]], *, echo: str) -> None:
    """Write each (path, content) pair (creating parent dirs), print `echo`, then report where they landed."""
    rels = []
    for path, content in paths_and_content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        rels.append(path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path)
    print(echo)
    print(f"\n(committed artifact written to {' and '.join(str(r) for r in rels)})")


__all__ = ["write_artifacts"]
