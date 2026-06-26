"""LIVE judge provisional calibration probe: real models, real cost (`task judge-live`).

Retires `evals/judge/live_calibration.py` (SP8 task 2 left it in place for this task to rewrite,
per the planning digest's own disposition: keep the sweep and settle SHAPE, discard
RUBRIC_V4/_account_facts, which graded against account state, the wrong ground truth for a
groundedness judge). This keeps the shape (provider tiers cheapest first, a warm up call absorbing
first request flakiness, a cross family plus same family comparison) and rewrites the content
against `judge.provisional`'s manufactured failures (ground truth by construction from SP7's
registry contradictions) and the groundedness rubric, never account facts.

Produces the SAME two provisional numbers `judge.provisional` computes hermetically, against REAL
model responses instead of seeded cassette fixtures: registry truth agreement per tier (does the
judge get the manufactured cases right), and judge vs judge kappa between the cheapest cross family
and same family tiers that ran without error. PROVISIONAL ONLY, per `judge.provisional`'s own KAPPA
HONESTY discipline: neither number licenses a production deployment, and this module never imports
the deployment gate either.

Cadence: recalibrate monthly, same as before. Each run writes a dated snapshot under
`judge/artifacts/live_provisional/` plus `latest.md`. Needs `OPENAI_API_KEY` and/or
`ANTHROPIC_API_KEY`; never the PR lane (`pyproject.toml`'s coverage omit list names this file the
same "operator entrypoint, not gated" way it named the file this retires).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage

from evals.artifacts import write_artifacts

from judge.contract import JudgeContract
from judge.llm_judge import judge_label
from judge.provisional import (
    ManufacturedCase,
    judge_vs_judge_kappa,
    manufactured_cases,
    provisional_calibration_artifact,
    registry_truth_agreement,
)
from judge.rubric import RUBRIC_GROUNDEDNESS, template_hash
from replay.gateway import GatewayChatModel
from replay.providers import build_chat_model

_ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "live_provisional"

# Pricing snapshot 2026-07-10, USD per 1M tokens (input, output). Recheck before trusting the cost
# math: this is exactly the kind of hardcoded assumption that goes stale on the vendor's schedule,
# not yours. Cheapest first per provider, so the sweep below settles on the first tier that runs.
# Anthropic's Sonnet 5 is at introductory pricing through 2026-08-31 ($3.00/$15.00 after).
_PROVIDER_TIERS = {
    "openai": [
        ("gpt-5.4-nano", 0.10, 0.40),
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


def _invoke_with_retry(fn, *, attempts: int = 5):
    """The very first call a fresh process makes to a model occasionally throws a spurious error
    that never recurs on retry (seen live, repeatedly). Retry with backoff before surfacing it as
    real; a live probe should not corrode into a flaky sweep over one bad request."""
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as err:
            last_err = err
            if attempt < attempts - 1:
                time.sleep(3.0 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _warm_up(provider: str) -> None:
    """Absorb the first call flakiness on a throwaway call before the scored sweep starts, so a
    real manufactured case is never the one that eats the bad first try."""
    cheapest_model_id = _PROVIDER_TIERS[provider][0][0]
    gateway = GatewayChatModel(
        model_id=f"{provider}:{cheapest_model_id}:warmup",
        mode="live",
        inner=build_chat_model(provider, cheapest_model_id),
    )
    try:
        _invoke_with_retry(lambda: gateway.invoke([HumanMessage("say OK")]))
    except Exception:
        pass  # a real failure here just means the sweep below hits it first instead; not fatal


def _run_tier(
    provider: str, model_id: str, cases: tuple[ManufacturedCase, ...]
) -> tuple[list[int], JudgeContract]:
    gateway = GatewayChatModel(
        model_id=f"{provider}:{model_id}", mode="live", inner=build_chat_model(provider, model_id)
    )
    labels = [
        _invoke_with_retry(
            lambda case=case: judge_label(
                gateway, RUBRIC_GROUNDEDNESS, case.question, case.answer, case.context
            )
        )
        for case in cases
    ]
    contract = JudgeContract(
        f"{provider}:{model_id}", RUBRIC_GROUNDEDNESS.version, template_hash(RUBRIC_GROUNDEDNESS)
    )
    return labels, contract


def _sweep(out: list[str], provider: str, cases: tuple[ManufacturedCase, ...]) -> tuple | None:
    """Run every tier of one provider cheapest first; feature the first that ran without error.

    One tier's hard failure (a real incompatibility, e.g. an unsupported parameter for that
    specific model) should not erase a cheaper tier's already computed reading: report the failure
    and keep going, the same "a rate, never bare, never silently dropped" discipline the rest of
    the harness holds to. There is no bar to settle on here (the digest's own point): every tier
    that runs gets its registry truth agreement reported, and the CHEAPEST one that ran is simply
    the one featured in the cross family / same family comparison below.
    """
    chosen = None
    for model_id, in_price, out_price in _PROVIDER_TIERS[provider]:
        out.append(f"## {provider}:{model_id}  (${in_price:.2f} / ${out_price:.2f} per 1M tokens, in/out)")
        try:
            labels, contract = _run_tier(provider, model_id, cases)
        except Exception as err:
            out.append(f"FAILED: {type(err).__name__}: {err}\n")
            continue
        report = registry_truth_agreement(
            contract, list(cases), labels, generated_at=datetime.now(timezone.utc)
        )
        out.append(report.render())
        out.append("")
        if chosen is None:
            chosen = (provider, model_id, report, labels, contract)
    return chosen


def _run() -> str:
    """Run the full sweep and return the rendered report (also the artifact content)."""
    cases = manufactured_cases()
    out = [
        "# Live judge provisional calibration probe (real models, registry truth + judge vs judge)",
        "Cheapest tier first per provider; the cheapest tier that ran without error is featured "
        "below. PROVISIONAL ONLY: see the honesty statement at the end, this never licenses a "
        "deployment.\n",
    ]

    results: dict[str, tuple | None] = {}
    for provider in _PROVIDER_TIERS:
        if not os.environ.get(_KEY_ENV[provider]):
            out.append(f"(skipping {provider}: {_KEY_ENV[provider]} not set)\n")
            continue
        _warm_up(provider)
        out.append(f"# {provider}\n")
        results[provider] = _sweep(out, provider, cases)

    cross_family = results.get("openai")  # Atlas's agent is Claude; OpenAI is the safe cross family pick
    same_family = results.get("anthropic")

    if cross_family is None or same_family is None:
        out.append(
            "\nBoth OPENAI_API_KEY and ANTHROPIC_API_KEY are needed to compute the judge vs judge "
            "kappa below (it compares two judges on the same set); at least one provider produced "
            "no successful tier. Only the registry truth agreement readings above (where present) "
            "come out of this run."
        )
        return "\n".join(out)

    _, _, cf_report, cf_labels, cf_contract = cross_family
    _, _, _, sf_labels, sf_contract = same_family
    jvj = judge_vs_judge_kappa(
        cf_contract,
        sf_contract,
        [c.case_id for c in cases],
        cf_labels,
        sf_labels,
        generated_at=datetime.now(timezone.utc),
    )
    artifact = provisional_calibration_artifact(cf_report, jvj, generated_at=datetime.now(timezone.utc))
    out.append("")
    out.append(artifact.render())
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
