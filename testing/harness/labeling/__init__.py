"""The batch label generation lane (SP8 Task 4, label collection half, pulled early).

`generate_label_set.py` holds the gated, unit tested functions (`load_seed_cases`,
`retrieved_chunks_from_messages`, `build_generation_graph`, `generate_label_items`,
`write_label_items`); `__main__.py` is the thin, operator run CLI entrypoint (`task
label:generate`), covered by the same "operator entrypoint, not gated" precedent
`pyproject.toml`'s coverage omit list already applies to `judge/live_provisional.py` and its
siblings.
"""
from __future__ import annotations
