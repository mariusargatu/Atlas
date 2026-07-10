"""The golden prompt corpus (SP9 task 6, D31): a small, FIXED set of prompts with controlled token
counts, committed to `prompt_corpus.json` next to this module -- the load lane's own precondition
for a fair comparison across stepped concurrency. A freshly sampled question at each concurrency step would
confound "did the system get slower" with "did this particular question need more generation," so
every step in `k6/chat_sse_load.js` cycles through the SAME fixed corpus instead
(`prompt_for_iteration`, deterministic modulo cycling, never a random pick).

`approx_tokens` is declared per prompt and cross checked here against a live whitespace recount
(`_approx_token_count`) within a small tolerance band -- NOT a real BPE tokenizer count (no
tokenizer dependency in the hermetic lane), just an honest guard against the committed corpus's own
token profile silently drifting out from under a later edit to the prompt text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROMPT_CORPUS_PATH = Path(__file__).resolve().parent / "prompt_corpus.json"

VALID_BUCKETS = frozenset({"short", "medium", "long"})

# Words, not real tokens: a whitespace split is a coarse approximation, wide enough that a small,
# honest wording tweak never trips this, narrow enough that a genuinely different prompt (the kind
# of accidental copy paste drift this guard exists for) still does.
_TOKEN_COUNT_TOLERANCE = 2


@dataclass(frozen=True)
class GoldenPrompt:
    """One fixed load lane prompt. `bucket` names its own controlled size class (short/medium/long)
    so a report can slice latency by prompt size, not just by concurrency step."""

    prompt_id: str
    text: str
    bucket: str
    approx_tokens: int


def _approx_token_count(text: str) -> int:
    return len(text.split())


def load_prompt_corpus(path: Path = DEFAULT_PROMPT_CORPUS_PATH) -> tuple[GoldenPrompt, ...]:
    """Order preserved from the file, never resorted through a set or dict -- a stable prompt order
    is what makes `prompt_for_iteration`'s own modulo cycling deterministic across two runs."""
    path = Path(path)
    entries = json.loads(path.read_text())
    if not entries:
        raise ValueError(f"{path}: the golden prompt corpus is empty; the load lane needs at least one prompt")
    prompts: list[GoldenPrompt] = []
    seen_ids: set[str] = set()
    for entry in entries:
        prompt_id = entry["prompt_id"]
        if prompt_id in seen_ids:
            raise ValueError(f"{path}: duplicate prompt_id {prompt_id!r}")
        seen_ids.add(prompt_id)
        bucket = entry["bucket"]
        if bucket not in VALID_BUCKETS:
            raise ValueError(
                f"{prompt_id!r}: unrecognized bucket {bucket!r}, expected one of {sorted(VALID_BUCKETS)}"
            )
        text = entry["text"]
        declared = entry["approx_tokens"]
        measured = _approx_token_count(text)
        if abs(measured - declared) > _TOKEN_COUNT_TOLERANCE:
            raise ValueError(
                f"{prompt_id!r}: declared approx_tokens={declared} drifted from a live recount "
                f"({measured}); the golden corpus's own controlled token counts are the load "
                f"lane's whole point (SP9 task 6) -- update the committed value, never silently "
                f"trust a stale one"
            )
        prompts.append(GoldenPrompt(prompt_id=prompt_id, text=text, bucket=bucket, approx_tokens=declared))
    return tuple(prompts)


def prompt_for_iteration(prompts: tuple[GoldenPrompt, ...], iteration: int) -> GoldenPrompt:
    """Deterministic modulo cycling: iteration N always maps to the SAME prompt across runs, so a
    saturation knee is never explained away by "step 8 just happened to draw the long prompts."""
    if not prompts:
        raise ValueError("prompt_for_iteration needs at least one prompt")
    return prompts[iteration % len(prompts)]


__all__ = [
    "DEFAULT_PROMPT_CORPUS_PATH",
    "VALID_BUCKETS",
    "GoldenPrompt",
    "load_prompt_corpus",
    "prompt_for_iteration",
]
