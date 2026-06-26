"""Operator entrypoint: `uv run python -m labeling [...]` (`task label:generate` /
`task label:generate-live`).

Defaults to the fully hermetic path: `--mode replay` against the committed
`FIXTURE_SEED_CASES`/`FIXTURE_CASSETTE_DIR`, so running this with no flags at all reproduces the
committed fixture (`label_items.fixture.jsonl`) byte for byte, zero keys, zero network. The REAL
label set needs `--seed-cases` pointed at SP7's real seed set (`SEED_CASES`), a real retriever
(`--retriever pgvector`, needs `docker compose up`), and, for anything beyond replay, a real
provider (`--mode record`, keys in `.env`) -- `task label:generate-live` wires exactly this; see
this task's own report for the full explanation of what it needs.

Not part of the hermetic gate (`pyproject.toml`'s coverage omit list names this file the same
"operator entrypoint, not gated" way it already names `judge/live_provisional.py`); the
functions it calls (`load_seed_cases`, `generate_label_items`, `retrieved_chunks_from_messages`,
`write_label_items`) ARE gated, by `testing/tests/test_generate_label_set.py`.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from labeling.generate_label_set import (
    FIXTURE_CASSETTE_DIR,
    FIXTURE_OUT,
    FIXTURE_SEED_CASES,
    build_generation_graph,
    generate_label_items,
    load_seed_cases,
    write_label_items,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "record", "live"), default="replay")
    parser.add_argument("--cassette-dir", type=Path, default=FIXTURE_CASSETTE_DIR)
    parser.add_argument("--seed-cases", type=Path, default=FIXTURE_SEED_CASES)
    parser.add_argument(
        "--retriever", default=None,
        help="unset keeps the hermetic InMemoryRetriever; 'pgvector' needs a running TEI stack "
        "(docker compose up), the real label generation path",
    )
    parser.add_argument("--limit", type=int, default=None, help="take the first N cases, fixed seed order")
    parser.add_argument("--out", type=Path, default=FIXTURE_OUT)
    parser.add_argument(
        "--source", default=None,
        help="stamped on every item as \"source\" (e.g. 'fixture'); omitted by default, the real "
        "generation path's own honest shape",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    retriever = None
    if args.retriever:
        from atlas.orchestration.atlas_graph import select_retriever

        retriever = select_retriever(args.retriever)
    cases = load_seed_cases(args.seed_cases, args.limit)
    graph, _tracer = build_generation_graph(args.mode, args.cassette_dir, retriever=retriever)
    items = asyncio.run(generate_label_items(graph, cases, source=args.source))
    write_label_items(items, args.out)
    skipped = len(cases) - len(items)
    print(f"wrote {len(items)} label items to {args.out} ({skipped} case(s) skipped: no final answer)")


if __name__ == "__main__":
    main()
