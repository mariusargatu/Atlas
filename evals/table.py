"""A recorded run, rendered as the block a chapter pastes.

Systems as rows and metrics as columns, because that is the shape every comparison in
this repository actually wants: what did each arm score, and what did a ranking that
ignores the question score. The two null rows are not optional (`RunRow` refuses to
exist without them) and the ceiling row is what stops a correct number being read as a
broken one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from evals.results import NULL_SYSTEMS, RunRow
from evals.stats import smallest_resolvable_difference

# Rendered left to right in this order when present, so two runs' tables line up.
_METRIC_ORDER = ("recall@k", "precision@k", "MRR@k", "nDCG@k", "success@k", "graded nDCG@k")
_SYSTEM_ORDER = (*NULL_SYSTEMS, "vector", "keyword", "fused", "reranked")


def _cell(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def _ordered(names: set[str], preferred: Sequence[str]) -> list[str]:
    known = [n for n in preferred if n in names]
    return known + sorted(names - set(known))


def retrieval_table(row: RunRow, headline: str = "reranked") -> str:
    """The table itself, plus the ceiling, what the headline system attained of it, and
    how many questions each column speaks for."""
    metrics = _ordered({m.metric for m in row.metrics}, _METRIC_ORDER)
    systems = _ordered({m.system for m in row.metrics}, _SYSTEM_ORDER)
    by_key = {(m.system, m.metric): m for m in row.metrics}

    def line(label: str, cells: list[str]) -> str:
        return f"| {label} | " + " | ".join(cells) + " |"

    out = [
        f"### Retrieval over {row.questions} tau2 questions at k={row.k}, run `{row.run_id}`",
        "",
        line("system", [m.replace("@k", f"@{row.k}") for m in metrics]),
        line("---", ["---"] * len(metrics)),
    ]
    for system in systems:
        out.append(line(system, [
            _cell(by_key[(system, m)].value) if (system, m) in by_key else "n/a" for m in metrics
        ]))

    # The headline arm's ceiling only, labelled as such: a ceiling is per-arm once k
    # exceeds what an arm returns, and printing all six would bury the table.
    ceilings = [by_key[(headline, m)].ceiling if (headline, m) in by_key else math.nan
                for m in metrics]
    out.append(line(f"**ceiling ({headline})**", [f"**{_cell(c)}**" for c in ceilings]))
    attained = [by_key[(headline, m)].bounded.attained if (headline, m) in by_key else math.nan
                for m in metrics]
    out.append(line(f"{headline} attained", [
        "n/a" if math.isnan(a) else f"{a:.0%}" for a in attained
    ]))
    # graded nDCG speaks for only 22 of the 97 tasks; without this row it reads as if it
    # covered the full set.
    out.append(line("questions", [
        str(by_key[(headline, m)].questions) if (headline, m) in by_key else "n/a" for m in metrics
    ]))

    # Per column: a single "at 97 questions" bar would misapply to a graded nDCG column
    # actually measured over 22, where the true bar is much wider.
    counts = sorted({by_key[(headline, m)].questions for m in metrics if (headline, m) in by_key})
    resolvable = smallest_resolvable_difference(row.questions)
    per_column = ""
    if len(counts) > 1:
        per_column = (
            " Columns measured over fewer questions have a wider bar: "
            + ", ".join(f"{n} questions → **{smallest_resolvable_difference(n):.3f}**"
                        for n in counts if n != row.questions)
            + "."
        )
    out += [
        "",
        f"Smallest resolvable difference at {row.questions} questions: **{resolvable:.3f}**.{per_column} "
        "That is a worst-case bar on the gap between two rows: it assumes the per-question "
        "difference is as variable as a bounded difference can be. Two rows closer together "
        "than that are not shown to differ by this argument alone. A paired bootstrap over "
        "the same questions is usually much tighter, and is what a real claim rests on.",
        "",
        f"`{NULL_SYSTEMS[1]}` is the ten chunks appearing in the most questions' gold sets: "
        "the best a ranking that never reads the question can do here, not an arbitrary one. "
        "A system has to beat it to have shown anything at all.",
    ]
    return "\n".join(out)


def worked_seconds(row: RunRow) -> float:
    """Roughly how long the measured work took, from the per-stage means the row already
    carries. Not a replacement for `wall_seconds`: it counts only time inside a stage and
    collapses concurrency across questions incorrectly. Its one job is distinguishing a
    run that did the work from one that read answers back from the ledger.
    """
    per_question = sum(figures.get("mean", 0.0) for figures in row.latency_ms.values())
    return per_question * row.questions / 1000.0


def cost_line(row: RunRow) -> str:
    """What the run spent, with two asymmetries worth knowing about.

    `answer` and `judge` are read back from the ledger, so a re-recorded run reports what
    was actually paid rather than zero. `embedding` is what *this* process paid, and is
    $0.0000 on a warm `.cache/vectors`. So a reader comparing two rows at the same
    settings should expect the embedding entry to differ and the other two not to.

    `cost_usd` covers every process that ever paid into this run's ledger; `wall_seconds`
    covers only the one that wrote the row. The two can differ by orders of magnitude on a
    resumed run, so this line does not weld them together as "Spent $X in Ys", which would
    assert a rate that only holds for a run that did the work itself.
    """
    total = sum(row.cost_usd.values())
    parts = ", ".join(f"{stage} ${amount:.4f}" for stage, amount in sorted(row.cost_usd.items()))
    line = f"Spent ${total:.4f} ({parts})."
    worked = worked_seconds(row)
    # A margin, not an equality: `worked` misses gaps between stages, so an honest run
    # always shows a little more wall time than work. Only an order-of-magnitude gap means
    # the money was spent by a different process.
    if total > 0.0 and worked > row.wall_seconds * 2.0:
        return (
            f"{line} This process ran for {row.wall_seconds:.1f}s and did not do that "
            f"work: it resumed from the ledger, where answers costing that much had "
            f"already been bought. The stage timings below total {worked:.0f}s, which is "
            f"what the run itself took."
        )
    return f"{line} This process ran for {row.wall_seconds:.1f}s."


def latency_table(row: RunRow) -> str:
    if not row.latency_ms:
        return ""
    out = ["| stage | median ms | slowest tenth ms | mean ms |", "|---|---|---|---|"]
    for stage, figures in row.latency_ms.items():
        out.append(
            f"| {stage} | {figures['median']:.1f} | {figures['slowest_tenth']:.1f} | "
            f"{figures['mean']:.1f} |"
        )
    return "\n".join(out)


def contrast_table(row: RunRow) -> str:
    """What the industry standard scorers said, beside this repository's own judge.

    The column to read is `kappa`, not `mean`: raw agreement reads high for a scorer that
    rates everything above threshold, the same trap an always-approving judge sets.
    """
    if not row.contrast:
        return ""
    out = [
        "### The same answers, scored by deepeval",
        "",
        "| metric | mean | cut point | raw agreement with the judge | kappa | 95% interval | questions |",
        "|---|---|---|---|---|---|---|",
    ]
    stale = False
    for contrast in row.contrast:
        if contrast.low is None or contrast.high is None:
            interval = "not recorded"
            stale = True
        else:
            flag = " ⚠" if contrast.underpowered else ""
            interval = f"[{contrast.low:+.3f}, {contrast.high:+.3f}]{flag}"
        out.append(
            f"| {contrast.metric} | {contrast.mean:.3f} | {contrast.threshold:.2f} | "
            f"{contrast.raw_agreement:.3f} | {contrast.chance_corrected:.3f} | "
            f"{interval} | {contrast.questions} |"
        )
    out += [
        "",
        "A kappa is a point estimate and this table prints its interval beside it, for the "
        "same reason every recall figure above carries the ceiling it was measured "
        "against: the estimate alone cannot distinguish a scorer measured to agree no "
        "better than chance from one the sample was too small to say anything about. ⚠ "
        "marks a row whose interval is too wide, or too degenerate, for the point estimate "
        "to be read on its own.",
    ]
    if stale:
        out += [
            "",
            "*not recorded* is the honest answer for this run and not a rendering fault. "
            "`ContrastRow` gained the interval after these figures were recorded, and they "
            "cannot be recomputed: it needs each question's deepeval score, and "
            "`.cache/report/` keeps one row per question, so the retrieval-only re-records "
            "that came afterwards overwrote all but three of them. Re-run "
            "`just report --contrast` to record a row that carries it. That is the "
            "expensive command, which is the reason this one has not been re-run here.",
        ]
    noise_aware = [c for c in row.contrast if c.low_with_judge_noise is not None]
    if noise_aware:
        out += [
            "",
            "The judge's own measured self-disagreement (`data/noise_floor.json`) folded "
            "into the same resampling pulls every interval above toward zero, because a "
            "reference that unstable caps how much agreement any comparison against it "
            "can show. Toward zero, not wider: flipping a reference verdict biases a "
            "kappa down rather than only adding spread to it, so the interval below can "
            "be narrower than the plain one as well as lower, and on the recorded run it "
            "is. Read the two as a point estimate and its ceiling, not as a tight "
            "estimate and a loose one:",
            "",
        ]
        for contrast in noise_aware:
            out.append(
                f"- **{contrast.metric}**: [{contrast.low_with_judge_noise:+.3f}, "
                f"{contrast.high_with_judge_noise:+.3f}] against the plain "
                f"[{contrast.low:+.3f}, {contrast.high:+.3f}] above"
            )
    out += [
        "",
        "Contextual precision and recall are absent on purpose: both need a reference "
        "answer, which tau2 does not supply. Atlas computes those two above, from the "
        "gold set, as arithmetic rather than as an opinion.",
        "",
        "**The cost line above does not include this table.** These scorers spend inside "
        "the library's own client, which never reaches this repository's token accounting, "
        "so the figure covers the answer and the verdict only. Four scorers per question, "
        "several calls each.",
    ]
    return "\n".join(out)


def settings_block(row: RunRow) -> str:
    """Every choice that produced the numbers above, because `run_id` is a hash and a
    hash on its own tells a reader nothing about what was run."""
    lines = [f"Settings behind `{row.run_id}` (recorded {row.recorded}, commit `{row.commit}`):", ""]
    for group, values in sorted(row.settings.items()):
        if isinstance(values, dict):
            inner = ", ".join(f"{k}={v!r}" for k, v in sorted(values.items()))
            lines.append(f"- **{group}**: {inner}")
        else:
            lines.append(f"- **{group}**: {values!r}")
    return "\n".join(lines)


def chapter_block(row: RunRow) -> str:
    parts = [retrieval_table(row), "", cost_line(row), ""]
    latency = latency_table(row)
    if latency:
        parts += [latency, ""]
    contrast = contrast_table(row)
    if contrast:
        parts += [contrast, ""]
    parts.append(settings_block(row))
    return "\n".join(parts)


def run_index(rows: Sequence[RunRow]) -> str:
    """One line per recorded run, so a reader can see what exists before asking for one."""
    if not rows:
        return "No runs recorded yet. `just report` writes the first one."
    out = ["| run | recorded | questions | k | generated | judged | headline nDCG |",
           "|---|---|---|---|---|---|---|"]
    for row in rows:
        headline = next(
            (m.value for m in row.metrics if m.system == "reranked" and m.metric == "nDCG@k"),
            math.nan,
        )
        out.append(
            f"| `{row.run_id}` | {row.recorded} | {row.questions} | {row.k} | "
            f"{'yes' if row.generated else 'no'} | {'yes' if row.judged else 'no'} | "
            f"{_cell(headline)} |"
        )
    return "\n".join(out)
