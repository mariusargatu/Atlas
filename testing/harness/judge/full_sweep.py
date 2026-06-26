"""SP10 task 3: the Full sweep lane's own driver (`.github/workflows/full-sweep.yml`, push to main
only). WIRES existing mechanics; builds nothing that already exists elsewhere:

  - case loading + the real live agent turn: `labeling.generate_label_set` (SP8 task 4's batch
    answer generation step) and `judge.live_pr_lane.build_live_agent` (SP10 task 2), BOTH reused
    UNCHANGED -- the SAME "run the real Atlas graph over SP7's 76 case seed set" wiring the Live PR
    lane already established, never a second copy of it.
  - the calibrated groundedness rubric (`judge.rubric.RUBRIC_GROUNDEDNESS`, SP8's one calibrated
    rubric, D15) plus `judge.llm_judge.judge_label`, graded against the SAME cited chunks
    `labeling.generate_label_set.retrieved_chunks_from_messages` already extracted onto each item --
    `judge.live_pr_lane.judge_context`/`build_live_judge_gateway` reused unchanged for that join and
    gateway construction, never a second copy of either.
  - the report + trend shape: `evals.evalkit.report`'s own `EvalReport`/`build_report` (paired with
    `evals.evalkit.runner`'s own `CaseResult`/`TrialResult` and `evals.evalkit.graders.Verdict`) and
    `append_trend_row` (evalkit's own append only JSONL mechanic), pointed at a NEW path
    (`judge/artifacts/full_sweep/trend.jsonl`) rather than evalkit's own demo trend file
    (`evalkit/artifacts/trend.jsonl`, `task eval`'s own file, which IS committed) -- the SAME row
    shape, a distinct stream, so a Live PR lane row (were one ever also appended to a trend file)
    could never be mistaken for a Full sweep one. UNLIKE the evalkit demo file, this path is NOT
    committed to the repo: the workflow uploads whatever this driver writes as a per run workflow
    artifact, so every single run's trend.jsonl is exactly one row; cross run accumulation would
    need a maintainer to commit appended rows back, a discipline this lane does not implement (see
    D16's own "same set, different lane" disclosure below, and the NEVER GATES note).

THE JUDGE TIER IS FRONTIER, NOT CHEAP (D15's "calibrated frontier judge", the one deliberate
difference from the Live PR lane's own judge choice): the top tier per provider family (the SAME
model ids `judge.live_provisional._PROVIDER_TIERS` names as each family's most capable entry),
because a report that gates nothing (see NEVER GATES below) is exactly the lane that can afford the
frontier tier's own cost on every push to main, unlike the Live PR lane's cheap tier which runs on
every matching pull request. Named here explicitly rather than importing that module's private
constant (ordered cheapest first) and reaching for its last entry, which would silently break if a
provider's own tier list is ever reordered or appended to.

NEVER GATES (D18): unlike the Live PR lane, this module has no floor and no `sys.exit(1)` path tied
to any rate in this report at all -- report and an appended trend row, nothing else. The workflow's
own upload step runs `if: always()` for the identical reason `live-pr.yml`'s does.

76 CASE HONESTY (D16, matching SP7/SP9/SP10 task 2's own disclosures): this is the SAME
`dataset_tools/seed_cases.jsonl` 76 case seed set the Live PR lane also draws on -- not yet D16's
50-80 PR smoke / 300-500 full sweep distinctly sized targets. Both lanes run the identical set today
at different judge tiers, not genuinely different sized slices. See
docs/measurements/sp7-datasets-metrics.md.

DEPENDENCY INJECTION for hermetic testability (the SAME discipline `judge.live_pr_lane` itself
holds to): `run()` takes an already built graph and an already built judge gateway, so the ENTIRE
grading/report computation here is proven end to end in `testing/tests/test_judge_full_sweep.py`
with a REPLAY agent gateway and a REPLAY judge gateway against seeded cassettes -- zero keys, zero
egress. Only `main()` reaches for a real live provider or a real Postgres/TEI retriever, and only
when this file is run directly (`task full-sweep:run`, push to main only, never the PR lane).
`build_live_agent` and `build_live_judge_gateway` are `judge.live_pr_lane`'s own live wiring,
imported and reused unchanged, never rebuilt a second time here.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from evals.artifacts import write_artifacts
from evals.evalkit.graders import Verdict
from evals.evalkit.report import EvalReport, append_trend_row, build_report
from evals.evalkit.runner import CaseResult, TrialResult

from judge.live_pr_lane import build_live_agent, build_live_judge_gateway, judge_context
from judge.llm_judge import judge_label
from judge.rubric import RUBRIC_GROUNDEDNESS

from labeling.generate_label_set import SEED_CASES, _GENERATION_CUSTOMER, generate_label_items, load_seed_cases

from matrix.live_driver import MissingEnvVarError

_ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "full_sweep"
TREND_PATH = _ARTIFACT_DIR / "trend.jsonl"

# The frontier tier per provider family (D15's "calibrated frontier judge"): each family's own most
# capable entry in `judge.live_provisional._PROVIDER_TIERS` today (2026-07-10 pricing snapshot,
# recheck before trusting any cost math derived from these two ids -- the same staleness warning
# that module's own pricing table already carries).
_FRONTIER_JUDGE_TIERS = (
    ("openai", "gpt-5.6-sol"),
    ("anthropic", "claude-opus-4-8"),
)


def select_frontier_judge_tier() -> tuple[str, str]:
    """Whichever provider key is configured, OpenAI checked first (an arbitrary but deterministic
    tie break when both are set, the SAME provider precedence `judge.live_pr_lane.select_judge_tier`
    already uses for its own cheap tier pick -- kept consistent rather than invented a second way to
    break the tie). Raises `MissingEnvVarError` -- the SAME fail closed exception type every other
    live entrypoint in this repo raises (`matrix.live_driver.require_env`) -- if neither key is set,
    checked BEFORE any gateway or retriever is built."""
    if os.environ.get("OPENAI_API_KEY"):
        return _FRONTIER_JUDGE_TIERS[0]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _FRONTIER_JUDGE_TIERS[1]
    raise MissingEnvVarError(
        "neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set: the calibrated frontier judge tier "
        "needs at least one of them to grade the 76 case set's live answers. Refusing to attempt "
        "any live call with neither configured."
    )


def grade_case(case: dict, item: dict, *, judge_gateway) -> CaseResult:
    """Grade one item's live answer for groundedness against its own cited context (the calibrated
    rubric, D15), then wrap the single judge trial into evalkit's own `CaseResult` shape so this
    lane's report and trend row are built from the SAME machinery `evals.evalkit` already ships,
    never a second report shape invented for this lane alone. A per item exception (the same class
    `judge.live_pr_lane.judge_the_items` already guards against) is recorded as a failed trial with
    the item still counted in the sample, never silently dropped."""
    try:
        label = judge_label(
            judge_gateway, RUBRIC_GROUNDEDNESS, item["question"], item.get("answer") or "", judge_context(item)
        )
    except Exception:
        label = 0
    passed = bool(label)
    verdict = Verdict(
        "groundedness", passed=passed, reason="grounded" if passed else "ungrounded or unsupported by cited context"
    )
    trial = TrialResult(index=0, passed=passed, verdicts=(verdict,))
    return CaseResult(
        case_id=case["case_id"],
        passes=int(passed),
        k=1,
        trials=(trial,),
        name=case.get("intent") or "",
        risk=case.get("adversarial_class") or "none",
    )


def run(
    cases: list[dict],
    *,
    graph,
    judge_gateway,
    judge_provider: str,
    judge_model_id: str,
    customer_id: str = _GENERATION_CUSTOMER,
) -> EvalReport:
    """The whole Full sweep lane, dependency injected: given an already built agent `graph` and an
    already built `judge_gateway`, drives every case once (`generate_label_items`, reused
    unchanged), then grades every answer with the calibrated frontier judge tier and folds the
    results into evalkit's own `EvalReport` shape (`lane="live"`, `model_id` stamped with the
    provider:model pair actually used, per `EvalReport.as_dict()`'s own provenance discipline).
    Hermetically testable end to end with a REPLAY agent graph and a REPLAY judge gateway
    (`testing/tests/test_judge_full_sweep.py`); `main()` below supplies live ones."""
    items = asyncio.run(generate_label_items(graph, cases, customer_id=customer_id))
    cases_by_id = {c["case_id"]: c for c in cases}
    results = tuple(
        grade_case(cases_by_id[item["case_id"]], item, judge_gateway=judge_gateway)
        for item in items
        if item["case_id"] in cases_by_id
    )
    return build_report(results, lane="live", model_id=f"{judge_provider}:{judge_model_id}")


def render_report(report: EvalReport, *, judge_provider: str, judge_model_id: str) -> str:
    """Wrap evalkit's own generic `EvalReport.render()` with this lane's disclosures, rather than
    inventing a second report dataclass the way `judge.live_pr_lane.LivePrLaneReport` needed to (that
    module also carries two gating floors this one does not have)."""
    lines = [
        "# Full sweep lane: full current golden set, calibrated frontier judge (D15), report + trend only",
        f"judge: {judge_provider}:{judge_model_id}",
        "",
        report.render(),
        "",
        "NEVER GATES (D18): this lane reports and appends a trend row only -- nothing here blocks a merge.",
        "(76 case honesty note: this is the SAME 76 case seed set the Live PR lane also draws on today "
        "-- D16's own 50-80 PR smoke / 300-500 full sweep sizing is not yet met by two distinctly sized "
        "slices; see docs/measurements/sp7-datasets-metrics.md.)",
    ]
    return "\n".join(lines)


def main() -> None:
    provider, model_id = select_frontier_judge_tier()  # fail closed before any live call, key or retriever
    cases = load_seed_cases(SEED_CASES)
    graph, _tracer = build_live_agent()
    judge_gateway = build_live_judge_gateway(provider, model_id)
    report = run(cases, graph=graph, judge_gateway=judge_gateway, judge_provider=provider, judge_model_id=model_id)

    rendered = render_report(report, judge_provider=provider, judge_model_id=model_id)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    write_artifacts(
        [(_ARTIFACT_DIR / "latest.md", rendered), (_ARTIFACT_DIR / f"{stamp}.md", rendered)], echo=rendered
    )
    row = {**report.as_dict(), "date": stamp}
    json_path = _ARTIFACT_DIR / f"{stamp}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(row, indent=2, sort_keys=True))
    append_trend_row(TREND_PATH, row)
    print(f"\ntrend row appended: date={stamp} lane={report.lane} model_id={report.model_id} path={TREND_PATH}")
    print("\n(this lane never gates -- D18 -- the process exit code below reflects a real script error only.)")


if __name__ == "__main__":
    main()


__all__ = [
    "TREND_PATH",
    "grade_case",
    "render_report",
    "run",
    "select_frontier_judge_tier",
]
