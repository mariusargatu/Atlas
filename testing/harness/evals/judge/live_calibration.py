"""LIVE judge-calibration probe: the real model, real account facts, real cost (`task judge-live`).

`task judge` (`__main__.py`) is a REPLAY demo: every verdict is a recorded cassette, so the kappa
reproduces byte for byte and costs nothing. It is explicit about what it cannot prove: the V2
rubric's prompt only ever carries the question and the answer, never the account facts an SME
applied when writing the `corrected` column in `evals/datasets/judge_calibration.py`, so a live judge
grading only that text has no way to earn the same kappa. This script closes that gap for real,
against a live model, at the cost of an actual API call:

1. A new rubric (`RUBRIC_V3`) that states the pass condition against account facts the prompt now
   actually includes (plan, term, fee, cap, usage, bill), looked up live from the same domain the
   agent itself reads (`atlas.domain.accounts` / `catalog`). This is a new instrument: per the judge
   contract (`contract.py`), a changed prompt voids any prior calibration, so it earns its own kappa.
2. A tiered cost/quality sweep, cheapest tier first within EACH provider (OpenAI: Luna -> Terra ->
   Sol; Anthropic: Haiku -> Sonnet -> Opus), because the right model for a judge is whichever
   cheapest tier clears the 0.6 automation bar, not reflexively the frontier one. Settle on the first
   tier per provider that licenses.
3. A real panel (`judge/panel.py`'s `panel_vote`) over the two providers' winning tiers: `panel.py`
   is otherwise only ever called with hand-typed integers in a unit test, never on an actual judge's
   readings. This wires it into real data already collected above, at no extra cost, and reports
   where the two judges split, the actual point of a panel (the split cases are the ambiguous,
   information-dense ones worth a human's attention, not the agreement rate).

Two providers, on purpose, not one winner: Atlas's agent runs on Claude (Anthropic), and the whole
point of the cross-family rule (ADR-004, ``judge/llm_judge.py``) is that the judge should NOT share
the agent's family, the strongest defence against self-enhancement bias. So the OpenAI sweep is the
one whose winner is safe to actually deploy as the judge. The Anthropic sweep runs too, because
"what would the cheapest same-family judge cost and score" is a real comparison worth having on
record, but its winner is reported as same-family information, not a same-family recommendation.

Never wired into `task judge`: this is `GatewayMode.LIVE` (call live, persist nothing), needs
provider keys and network egress, and is never part of the hermetic PR lane. The 14-case REPLAY
fixture in `datasets/judge_calibration.py` is untouched; this is a second, independent measurement,
not a replacement for it.

Meant to be RUN AGAIN, not run once: a judge contract calibrated in March is a historical fact by
June (the-lying-judge, "recalibrate monthly, because both ends of the comparison drift"), and a
new model generation ships every few months regardless. Each run writes a dated snapshot under
`judge/artifacts/live_calibration/` alongside a `latest.md`, so the next recalibration is a diff
against real history, not a fresh guess. When a new tier ships, add it to `_PROVIDER_TIERS` (cheapest
first) and rerun; the sweep and the panel need no other change.
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from atlas.domain import accounts, catalog

from evals.artifacts import write_artifacts
from evals.datasets.judge_calibration import CALIBRATION, case_ids, human_labels
from evals.judge.calibration import calibrate
from evals.judge.contract import JudgeContract
from evals.judge.panel import panel_vote
from evals.judge.rubric import Rubric, template_hash
from replay.gateway import GatewayChatModel
from replay.providers import build_chat_model

_ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "live_calibration"

# Pricing snapshot 2026-07-10, USD per 1M tokens (input, output). Recheck before trusting the cost
# math: this is exactly the kind of hardcoded assumption that goes stale on the vendor's schedule,
# not yours. Cheapest first per provider, so the sweep below settles on the first tier that
# licenses. Anthropic's Sonnet 5 is at introductory pricing through 2026-08-31 ($3.00/$15.00 after).
_PROVIDER_TIERS = {
    "openai": [
        ("gpt-5.6-luna", 1.00, 6.00),
        ("gpt-5.6-terra", 2.50, 15.00),
        ("gpt-5.6-sol", 5.00, 30.00),
    ],
    "anthropic": [
        ("claude-haiku-4-5-20251001", 1.00, 5.00),
        ("claude-sonnet-5", 2.00, 10.00),
        ("claude-opus-4-8", 5.00, 25.00),
    ],
}
_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

_VERDICT = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)

RUBRIC_V3 = Rubric(
    version="account-truth-v3-live-facts",
    system=(
        "You score a customer-support answer for CORRECTNESS against the ACCOUNT FACTS given below, "
        "not against your own assumptions or any document. Reply PASS only if every claim in the "
        "answer is TRUE for these account facts. A confident, fluent, well-written answer that "
        "contradicts a fact given (a term, a fee, a data cap, a usage figure) must FAIL. "
        "Answer with exactly one word: PASS or FAIL."
    ),
)


def _account_facts(customer_id: str) -> str:
    """The account truth an SME checks by hand, handed to the judge instead of assumed."""
    acct = accounts.get_account(customer_id)
    plan = catalog.get_plan(acct.plan_id)
    cap = f"{plan.data_cap_gb} GB" if plan.data_cap_gb is not None else "uncapped"
    return (
        f"plan={plan.name}; minimum_term={'yes' if plan.has_term else 'no'}; "
        f"early_termination_fee=£{plan.early_termination_fee}; data_cap={cap}; "
        f"usage_this_period={acct.usage.gigabytes_used} GB; bill=£{acct.bill.amount}"
    )


def _prompt(question: str, answer: str, facts: str) -> list[BaseMessage]:
    return [
        SystemMessage(RUBRIC_V3.system),
        HumanMessage(
            f"Account facts: {facts}\nQuestion: {question}\nAnswer under review: {answer}\n"
            "Verdict (PASS or FAIL):"
        ),
    ]


def _content_text(content: object) -> str:
    """Flatten a message's content to plain text. Some models (e.g. claude-sonnet-5) return a list of
    content blocks (text/thinking/...) instead of a bare string; join just the text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return " ".join(parts)
    return ""


def _parse_label(text: str) -> int:
    """First standalone PASS/FAIL token decides; anything unparseable fails closed to FAIL."""
    match = _VERDICT.search(text)
    return 1 if match and match.group(1).upper() == "PASS" else 0


def _invoke_with_retry(gateway: GatewayChatModel, messages: list[BaseMessage], *, attempts: int = 5):
    """The very first call a fresh process makes to a model (seen live, repeatedly, against a model
    that GA'd a day earlier) occasionally throws a spurious AuthenticationError that never recurs on
    retry. Retry with backoff before surfacing it as real; a live probe should not corrode into a
    flaky sweep over one bad request."""
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            return gateway.invoke(messages)
        except Exception as err:
            last_err = err
            if attempt < attempts - 1:
                time.sleep(3.0 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _run_tier(provider: str, model_id: str) -> tuple[list[int], JudgeContract]:
    gateway = GatewayChatModel(
        model_id=f"{provider}:{model_id}", mode="live", inner=build_chat_model(provider, model_id)
    )
    labels = [
        _parse_label(
            _content_text(
                getattr(
                    _invoke_with_retry(gateway, _prompt(c.question, c.answer, _account_facts(c.customer_id))),
                    "content",
                    "",
                )
            )
        )
        for c in CALIBRATION
    ]
    contract = JudgeContract(f"{provider}:{model_id}", RUBRIC_V3.version, template_hash(RUBRIC_V3))
    return labels, contract


def _sweep(out: list[str], provider: str, ids: list[str], humans: list[int]) -> tuple | None:
    """Run every tier of one provider cheapest-first; return the first that licenses, or None.

    One tier's hard failure (a real incompatibility, e.g. an unsupported parameter for that specific
    model) should not erase a cheaper tier's already-good result: report the failure and keep going,
    the same "a rate, never bare, never silently dropped" discipline the rest of the harness holds to.
    """
    chosen = None
    for model_id, in_price, out_price in _PROVIDER_TIERS[provider]:
        out.append(f"## {provider}:{model_id}  (${in_price:.2f} / ${out_price:.2f} per 1M tokens, in/out)")
        try:
            labels, contract = _run_tier(provider, model_id)
        except Exception as err:
            out.append(f"FAILED: {type(err).__name__}: {err}\n")
            continue
        report = calibrate(contract, ids, humans, labels)
        out.append(report.render())
        out.append("")
        if report.licensed and chosen is None:
            chosen = (provider, model_id, report, labels)
    return chosen


def _warm_up(provider: str) -> None:
    """Absorb the first-call-in-a-fresh-process flakiness (see `_invoke_with_retry`) on a throwaway
    call before the scored sweep starts, so a real case is never the one that eats the bad first try."""
    cheapest_model_id = _PROVIDER_TIERS[provider][0][0]
    gateway = GatewayChatModel(
        model_id=f"{provider}:{cheapest_model_id}:warmup",
        mode="live",
        inner=build_chat_model(provider, cheapest_model_id),
    )
    try:
        _invoke_with_retry(gateway, [HumanMessage("say OK")])
    except Exception:
        pass  # a real failure here just means the sweep below hits it first instead; not fatal


def _panel_section(out: list[str], ids: list[str], cross_family: tuple, same_family: tuple) -> None:
    """A real panel of 2 disjoint-family judges (Verga et al., "Replacing Judges with Juries"):
    the two cheapest tiers that each independently licensed, voting per case. The aggregate is not
    the point (`panel.py`'s own docstring); the cases where they SPLIT are, because those are exactly
    the ambiguous ones worth a human's scarce attention. This is real data already paid for above, not
    a second sweep: wiring `panel_vote` into an actual judge comparison instead of leaving it exercised
    only by a unit test on hand-typed integers."""
    cf_provider, cf_model, _, cf_labels = cross_family
    sf_provider, sf_model, _, sf_labels = same_family
    out.append(f"\n# Panel: {cf_provider}:{cf_model}  +  {sf_provider}:{sf_model}\n")
    votes = [panel_vote([a, b]) for a, b in zip(cf_labels, sf_labels)]
    splits = [(case_id, v) for case_id, v in zip(ids, votes) if v.disagreed]
    out.append(f"{len(splits)}/{len(ids)} cases split between the two judges.")
    for case_id, v in splits:
        out.append(f"  SPLIT {case_id:<28} votes={v.votes} -> majority={v.label} (tie fails closed to 0)")
    if not splits:
        out.append("  no splits: both judges agreed on every case.")


def _run() -> str:
    """Run the full sweep and return the rendered report (also the artifact content)."""
    ids, humans = case_ids(), human_labels()
    out = [
        "# Live judge-calibration probe (real models, real account facts, real cost)",
        "Cheapest tier first per provider; settling on the first that clears the 0.6 bar.\n",
    ]

    results = {}
    for provider in _PROVIDER_TIERS:
        if not os.environ.get(_KEY_ENV[provider]):
            out.append(f"(skipping {provider}: {_KEY_ENV[provider]} not set)\n")
            continue
        _warm_up(provider)
        out.append(f"# {provider}\n")
        results[provider] = _sweep(out, provider, ids, humans)

    cross_family = results.get("openai")  # Atlas's agent is Claude; OpenAI is the safe cross-family pick
    same_family = results.get("anthropic")

    if cross_family is not None:
        provider, model_id, report, _ = cross_family
        out.append(
            f"RECOMMENDATION (cross-family, safe to deploy as the judge): {provider}:{model_id} "
            f"is the cheapest tier that clears the automation bar live "
            f"(kappa={report.kappa:.2f} >= {report.bar:.2f})."
        )
    else:
        out.append("RECOMMENDATION (cross-family): no OpenAI tier cleared the automation bar live "
                    "(or OPENAI_API_KEY was not set); keep this judge manual or revise the rubric further.")

    if same_family is not None:
        provider, model_id, report, _ = same_family
        out.append(
            f"FOR COMPARISON ONLY (same-family as the agent, reintroduces self-enhancement bias "
            f"risk per ADR-004): {provider}:{model_id} clears the bar too "
            f"(kappa={report.kappa:.2f})."
        )

    if cross_family is not None and same_family is not None:
        _panel_section(out, ids, cross_family, same_family)

    return "\n".join(out)


def main() -> None:
    report = _run()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    write_artifacts(
        [
            (_ARTIFACT_DIR / "latest.md", report),
            (_ARTIFACT_DIR / f"{stamp}.md", report),
        ],
        echo=report,
    )


if __name__ == "__main__":
    main()
