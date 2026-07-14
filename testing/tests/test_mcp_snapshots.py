"""D11: hermetic byte diff of every MCP server's advertised tool schema against its committed
golden snapshot (`contracts/mcp_snapshots/<server>.json`).

Regeneration is documented in `testing/harness/contract_tools/mcp_snapshot.py`'s own header
comment: `uv run python -m contract_tools.mcp_snapshot --write`, then review the resulting
`git diff contracts/mcp_snapshots/` -- that diff IS the review signal an intentional schema change
leaves behind, and this test is what makes an UNINTENTIONAL drift (a FastMCP upgrade quietly
changing schema generation, a stray added parameter) fail loud instead of shipping silently.
"""
from __future__ import annotations

import pytest
from contract_tools.mcp_snapshot import SNAPSHOT_DIR, render_snapshot, server_names


@pytest.mark.parametrize("name", sorted(server_names()))
def test_committed_snapshot_is_byte_identical_to_a_fresh_dump(name):
    fresh = render_snapshot(name)
    path = SNAPSHOT_DIR / f"{name}.json"
    assert path.is_file(), f"no committed snapshot for {name!r} at {path}"
    committed = path.read_text()
    assert committed == fresh, (
        f"{name} MCP tool schema drifted from the committed snapshot ({path}). Regenerate via "
        "`uv run python -m contract_tools.mcp_snapshot --write` and review the diff before "
        "committing, if the change is intentional."
    )


def test_every_declared_server_has_exactly_one_committed_snapshot_file():
    committed = {p.stem for p in SNAPSHOT_DIR.glob("*.json")}
    assert committed == set(server_names())


def test_snapshot_rendering_is_deterministic_across_repeated_calls():
    # a regression here (e.g. dict iteration order leaking into the dump) would make the byte diff
    # test above flaky rather than a reliable gate; pin the determinism claim directly.
    for name in server_names():
        assert render_snapshot(name) == render_snapshot(name)
