"""LLM paraphrase machinery for the seed dataset (SP7 Task 6): grows phrasing volume around an
existing case's registry anchored ground truth, never invents ground truth. Flag gated behind a
worded provider key check (the standing CODE FIRST directive): the machinery below is built and
hermetically tested in this same commit, but a real paraphrase run needs a real provider key
(`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) exported first and is never invoked by `task test` or any
other gate. An operator can later grow phrasing volume by running this module directly once a key
is set; its absence never blocks anything this sub project ships.

Ground truth never varies: every paraphrase of a case keeps `expected_facts`, `expected_doc_ids`,
`expected_tool_calls`, `answerable`, `adversarial_class`, and `intent` identical to the base case,
the same discipline `dataset_tools.counterfactual.generate_cohort` already applies for its own
varying axis (persona instead of phrasing); only `turns[0]['user']` (the wording) and `case_id`
(suffixed, kept unique) differ. A paraphrase is a PROPOSAL: nothing here validates or commits it
into the seed set automatically, a human curator reviews the output first (the "operator decision
later" the plan names), matching the module docstring's own no grading rule below.

No reference free faithfulness, judge, or rubric anything lives here: SP8's boundary (the 04/05
grader boundary this repo's CLAUDE.md names), out of scope for SP7 entirely. This module also never
GRADES a paraphrase's quality, only proposes phrasing text.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# Checked in this order but combined with "any": either provider key licenses a paraphrase run,
# mirroring judge.live_provisional's own per provider key convention (_KEY_ENV there), never
# a single hardcoded provider assumption.
_KEY_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

_SYSTEM = (
    "You paraphrase a single customer support question into N distinct, natural alternative "
    "phrasings that ask for EXACTLY the same information, in the same register a real customer "
    "would use. Never add or remove a fact, a name, a number, or a claim; only reword. Reply with "
    "a JSON array of N strings, nothing else."
)


class ParaphraseKeyMissingError(RuntimeError):
    """Raised when no provider key is configured. This is the worded gate itself (SP7 plan Task 6,
    the CODE FIRST directive): paraphrase machinery is built and tested hermetically, but never
    RUNS, not from `task test`, not from any other gate, without an operator explicitly exporting
    a real provider key first. Checked before constructing any provider client, so this module
    imports and unit tests cleanly under the hermetic, keyless, networkless lane."""


def paraphrase_key_configured(env: Mapping[str, str] | None = None) -> bool:
    """True only when a real provider key is present. Never guesses: `Mapping.get` is falsy for
    both an unset AND an empty string value, so an environment that exports the variable NAME with
    no real value still gates closed, never treated as configured."""
    source = env if env is not None else os.environ
    return any(source.get(name) for name in _KEY_ENV)


def require_paraphrase_key(env: Mapping[str, str] | None = None) -> None:
    """Raises `ParaphraseKeyMissingError` when no provider key is configured; returns silently
    otherwise. Every entry point below (CLI `main`, and any future programmatic caller) calls this
    FIRST, before touching argparse or constructing a provider client, so a keyless invocation
    fails fast on this module's own worded message rather than a provider SDK's less legible one."""
    if not paraphrase_key_configured(env):
        raise ParaphraseKeyMissingError(
            f"no provider key configured ({' or '.join(_KEY_ENV)}); paraphrase machinery is flag "
            "gated and never runs without one, per the SP7 Task 6 code first directive"
        )


def _parse_variants(text: str, n: int) -> tuple[str, ...]:
    """Best effort JSON array parse; anything else is a hard failure. No silent fallback to the
    base phrasing repeated `n` times: that would look like `n` successful paraphrases when it is
    zero, exactly the kind of quiet failure this codebase's own doctrine (fail closed, declared
    never silent) rejects elsewhere."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"paraphrase model did not return valid JSON: {text!r}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"paraphrase model did not return a JSON array of strings: {text!r}")
    return tuple(parsed[:n])


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b for b in content if isinstance(b, str)]
        parts += [str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return " ".join(parts)
    return ""


def paraphrase_text(question: str, *, n: int, model: BaseChatModel) -> tuple[str, ...]:
    """`n` alternative phrasings of `question` from a live model. Raises whatever the model call
    itself raises (a provider error, a malformed response); never silently returns fewer than
    requested, a caller that wants fault tolerance handles that itself at the curation step."""
    messages = [SystemMessage(_SYSTEM), HumanMessage(f"N={n}\nQuestion: {question}")]
    result = model.invoke(messages)
    return _parse_variants(_content_text(result.content), n)


def paraphrase_case(case: Mapping[str, object], *, n: int, model: BaseChatModel) -> tuple[dict, ...]:
    """`n` new case dicts, one per paraphrase, identical to `case` in every field except `case_id`
    (suffixed `-para-{i}`, kept unique) and `turns[0]['user']` (the new phrasing). Registry ground
    truth is copied unchanged, per the module docstring. Single turn cases only: a multi turn
    case's later turns are written to follow turn 1, so paraphrasing only `turns[0]` would silently
    break the trajectory after turn 1; this raises instead of doing that quietly."""
    turns = case.get("turns") or ()
    if len(turns) != 1:
        raise ValueError("paraphrase_case only supports single turn cases (see module docstring)")
    base_user = str(turns[0]["user"])
    variants = paraphrase_text(base_user, n=n, model=model)
    return tuple(
        {**case, "case_id": f"{case['case_id']}-para-{i}", "turns": [{"user": variant}]}
        for i, variant in enumerate(variants, start=1)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: paraphrase one case's phrasing `n` ways, print the resulting cases as JSONL (a human
    curator reviews and hand picks before anything here enters a committed seed set). Checks the
    worded key gate FIRST, before argparse or constructing a provider client, and prints a clear
    message with a nonzero exit code instead of a bare traceback when the gate is closed. This is
    the ONE function in this module a gate could ever reach, and even it refuses to run keyless."""
    import argparse

    try:
        require_paraphrase_key()
    except ParaphraseKeyMissingError as exc:
        print(f"error: {exc}")
        return 1

    from replay.providers import build_chat_model

    parser = argparse.ArgumentParser(prog="dataset_tools.paraphrase")
    parser.add_argument("--case", required=True, help="a single case JSON object, one line")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args(argv)

    case = json.loads(args.case)
    model = build_chat_model()
    for new_case in paraphrase_case(case, n=args.n, model=model):
        print(json.dumps(new_case, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
