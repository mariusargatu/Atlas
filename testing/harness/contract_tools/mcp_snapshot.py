"""D11: golden byte diffs of every MCP server's advertised tool schema.

This is a DIFFERENT, simpler mechanism than `contract_tools.diff` (the SchemaVer evolution rule
engine `contracts/trace|dataset|manifest|sse/schema.json` use): D11 and D22 both ask for a literal
byte for byte snapshot of what a server currently advertises, not a versioned, evolvable schema
family. `contracts/mcp_snapshots/<server>.json` is that snapshot, one file per server
(account/actions/catalog/knowledge), each a JSON array of `{"name", "description", "parameters"}`
objects (the SAME shape `atlas.mcp_servers.tool_surface.mcp_tool_surface` builds for bind_tools),
sorted by tool name, keys sorted, two space indent, trailing newline.

Regeneration: an intentional schema change (a new tool, a renamed parameter, additionalProperties
newly required somewhere it was not) is expected to change these files. When it does:

    uv run python -m contract_tools.mcp_snapshot --write

then review the resulting `git diff contracts/mcp_snapshots/` in the SAME change: that diff IS the
review signal (`testing/tests/test_mcp_snapshots.py` fails loud on any drift the committed files do
not already reflect). `--check` (the default; also what the hermetic test itself calls, via
`render_snapshot`, not this CLI) prints a diff without writing, for local inspection.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from atlas.mcp_servers.tool_surface import mcp_tool_surface

# Anchored on this file, not the CWD: `task test`/pytest may run from anywhere, but `uv run python
# -m contract_tools.mcp_snapshot` is meant to be run from the repo root either way (mirrors
# `contract_tools.diff`'s own `--git-ref` resolution note).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = REPO_ROOT / "contracts" / "mcp_snapshots"


def server_names() -> tuple[str, ...]:
    """The four servers, in the same order `mcp_tool_surface` builds them (SP4 task 5's own MCP
    hardening inventory: knowledge, account, catalog, actions per the existing servers) -- alphabetical
    here so `sorted(server_names())` in a test reads as a stable, human ordered list."""
    return ("account", "actions", "catalog", "knowledge")


# Every tool this server declares, by name -- single sourced from `domain.binding` so this file
# never hand keeps a second copy of the same four sets `atlas_graph.py`'s routing already uses.
def _server_tool_names() -> dict[str, frozenset[str]]:
    from atlas.domain.binding import CATALOG_TOOLS, KNOWLEDGE_TOOLS, READ_TOOLS, WRITE_TOOLS

    return {
        "account": frozenset(READ_TOOLS),
        "actions": frozenset(WRITE_TOOLS),
        "catalog": frozenset(CATALOG_TOOLS),
        "knowledge": frozenset(KNOWLEDGE_TOOLS),
    }


def dump_server_schema(name: str) -> list[dict]:
    """Every tool `name`'s server owns, from the ONE aggregated surface (`mcp_tool_surface`, which
    already applies `hardening.harden_tool_schemas` at each server's own construction), sorted by
    tool name for a byte stable snapshot."""
    if name not in server_names():
        raise ValueError(f"unknown MCP server {name!r}; expected one of {server_names()}")
    surface = mcp_tool_surface()
    owned = _server_tool_names()[name]
    return sorted((spec for tool_name, spec in surface.items() if tool_name in owned), key=lambda spec: spec["name"])


def render_snapshot(name: str) -> str:
    """The exact bytes a committed `contracts/mcp_snapshots/<name>.json` file holds (and what
    `--write` writes): sorted keys, two space indent, one trailing newline. `test_mcp_snapshots.py`
    calls this directly (never implements the rendering a second time), so "byte diffs the committed
    file" means exactly that, not a comparison that is only semantically equal but formatted
    differently."""
    return json.dumps(dump_server_schema(name), indent=2, sort_keys=True) + "\n"


def _snapshot_path(name: str) -> pathlib.Path:
    return SNAPSHOT_DIR / f"{name}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the current schemas to contracts/mcp_snapshots/")
    args = parser.parse_args(argv)

    if args.write:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for name in server_names():
            _snapshot_path(name).write_text(render_snapshot(name))
            print(f"wrote {_snapshot_path(name)}")
        return 0

    exit_code = 0
    for name in server_names():
        fresh = render_snapshot(name)
        path = _snapshot_path(name)
        committed = path.read_text() if path.is_file() else None
        if committed == fresh:
            print(f"{name}: unchanged")
            continue
        exit_code = 1
        print(f"{name}: DRIFTED from {path} (run with --write to regenerate, then review the diff)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
