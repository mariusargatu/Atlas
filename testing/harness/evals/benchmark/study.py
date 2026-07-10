"""Compute the worked example: marginal CIs, the paired delta, the release gate, and the verdict.

Pure analysis over the committed paired vectors. Everything here is a real call into
`quality.stats` and `quality.gate`. Nothing is asserted. The result is a plain dict so `__main__`
can render it to both a human readable artifact and the committed JSON.
"""
from __future__ import annotations

from evals.benchmark.dataset import (
    A_FAIL_B_PASS,
    A_PASS_B_FAIL,
    N,
    SEED,
    paired_vectors,
)
from quality.gate import GateDecision, gate_on_lower_bound
from quality.stats import (
    intervals_overlap,
    mcnemar_exact,
    paired_bootstrap_diff,
    paired_permutation_test,
    wilson_interval,
)

RESAMPLES = 10000  # high stakes: well past the >=1000 floor, cheap in pure Python at this n

# The release gate worked example: the candidate v_b must clear this bar to ship, and the
# rule is "gate on the lower bound of the interval, never the point". The budget is the
# CI width the release decision tolerates. Wider than this is a quarantine, not a verdict.
RELEASE_THRESHOLD = 0.80
VARIANCE_BUDGET = 0.20


def _gate_summary(candidate: str, gate: GateDecision) -> dict:
    """The release_gate artifact block, every number read off the decision itself.

    Sourcing threshold/budget from the `GateDecision` (not the module constants) means the
    artifact reports what the gate was actually asked, and cannot desync if a caller ever
    feeds the gate a computed bar.
    """
    return {
        "candidate": candidate,
        "threshold": gate.threshold,
        "variance_budget": gate.variance_budget,
        "lower_bound": gate.lower_bound,
        "width": gate.width,
        "verdict": gate.verdict.value,
        "reason": gate.reason,
    }


def run() -> dict:
    """Return every number the worked example reports, computed live from the fixture."""
    a, b = paired_vectors()
    passes_a, passes_b = sum(a), sum(b)

    ci_a = wilson_interval(passes_a, N)
    ci_b = wilson_interval(passes_b, N)
    overlap = intervals_overlap(ci_a, ci_b)

    diff, diff_lo, diff_hi = paired_bootstrap_diff(a, b, seed=SEED, n_resamples=RESAMPLES)
    perm_p = paired_permutation_test(a, b, seed=SEED, n_resamples=RESAMPLES)
    mcnemar_p = mcnemar_exact(A_PASS_B_FAIL, A_FAIL_B_PASS)

    diff_ci_excludes_zero = not (diff_lo <= 0.0 <= diff_hi)
    # All three paired tests must agree: the seeded permutation p value is a Monte Carlo
    # approximation of the exact McNemar test on the same discordant pairs, so requiring
    # both catches the case where sampling noise puts one on either side of 0.05.
    significant = perm_p < 0.05 and mcnemar_p < 0.05 and diff_ci_excludes_zero
    # Direction is read off the sign of diff (mean(a) - mean(b)), never hardcoded, so the
    # verdict names whichever version actually underperforms.
    leader, trailer = ("v_a", "v_b") if diff > 0 else ("v_b", "v_a")
    # The candidate v_b against the release bar: its point (0.81) sits above 0.80, its
    # Wilson floor does not, and the gate reads the floor.
    gate = gate_on_lower_bound(ci_b, threshold=RELEASE_THRESHOLD, variance_budget=VARIANCE_BUDGET)
    return {
        "n": N,
        "seed": SEED,
        "resamples": RESAMPLES,
        "v_a": {"passes": passes_a, "rate": passes_a / N, "ci95": list(ci_a)},
        "v_b": {"passes": passes_b, "rate": passes_b / N, "ci95": list(ci_b)},
        "marginal_intervals_overlap": overlap,
        "paired": {
            "diff": diff,
            "diff_ci95": [diff_lo, diff_hi],
            "diff_ci_excludes_zero": diff_ci_excludes_zero,
            "permutation_p": perm_p,
            "mcnemar_p": mcnemar_p,
            "discordant": {"a_pass_b_fail": A_PASS_B_FAIL, "a_fail_b_pass": A_FAIL_B_PASS},
        },
        "release_gate": _gate_summary("v_b", gate),
        "significant": significant,
        "verdict": (
            f"regression confirmed: {trailer} is worse"
            if significant
            else f"no regression: the gap sits inside the noise, you cannot ship \"{leader} is better\""
        ),
    }


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _ci(pair) -> str:
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def render(r: dict) -> str:
    """Render `run()`'s result to the human readable report the committed artifact embeds.

    Lives here (not `__main__.py`, which is an operator entrypoint omitted from coverage) so
    it is covered and directly testable by a unit test, the same way judge's `CalibrationReport.render()`
    lives in `calibration.py` rather than in `judge/__main__.py`.
    """
    a, b, p, g = r["v_a"], r["v_b"], r["paired"], r["release_gate"]
    return "\n".join([
        "# Honest benchmark study (the regression that wasn't)",
        "",
        "Two Atlas model versions over the answer golden set, the SAME 100 items, so the",
        "comparison is paired. The point estimates say v_a beat v_b by three points. The",
        "statistics say you cannot tell yet.",
        "",
        "## Marginal scores, each with its interval",
        "```",
        f"v_a  {a['passes']}/{r['n']}  rate {_pct(a['rate'])}  Wilson 95% CI {_ci(a['ci95'])}",
        f"v_b  {b['passes']}/{r['n']}  rate {_pct(b['rate'])}  Wilson 95% CI {_ci(b['ci95'])}",
        f"intervals overlap: {r['marginal_intervals_overlap']}",
        "```",
        "",
        "## The paired comparison (the test that belongs on paired data)",
        "```",
        f"discordant pairs: v_a pass / v_b fail = {p['discordant']['a_pass_b_fail']}, "
        f"v_a fail / v_b pass = {p['discordant']['a_fail_b_pass']}",
        f"paired difference (mean v_a - mean v_b): {p['diff']:+.3f}",
        f"paired bootstrap 95% CI on the difference: {_ci(p['diff_ci95'])}  "
        f"(excludes zero: {p['diff_ci_excludes_zero']})",
        f"paired permutation p: {p['permutation_p']:.3f}",
        f"exact McNemar p: {p['mcnemar_p']:.3f}",
        "```",
        "",
        f"seed: {hex(r['seed'])}   resamples: {r['resamples']}",
        "",
        "## The release gate (gate on the floor, never the point)",
        "```",
        f"candidate {g['candidate']}  point {_pct(b['rate'])}  floor {g['lower_bound']:.3f}  "
        f"bar {g['threshold']:.2f}  budget {g['variance_budget']:.2f}",
        f"verdict: {g['verdict'].upper()}  ({g['reason']})",
        "```",
        "",
        "The candidate's best guess sits above the bar and its honest floor sits below it;",
        "shipping on the best guess is shipping on optimism, so the gate fails closed.",
        "",
        "## The verdict",
        "",
        f"**{r['verdict']}.**",
        "",
        "The difference interval contains zero and the paired test returns p well above 0.05.",
        "There is no regression here to find, only a smaller sample than the question deserved.",
        "Gate a release on the lower bound of the interval, never the point.",
        "",
    ])


__all__ = ["RELEASE_THRESHOLD", "RESAMPLES", "VARIANCE_BUDGET", "render", "run"]
