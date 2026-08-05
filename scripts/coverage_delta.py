"""Whether coverage regressed since the last recorded baseline, not whether it clears an
absolute floor.

Compares today's total against a number committed the last time somebody looked, and
fails only when today's total is genuinely lower by more than run-to-run noise allows
for. See docs/test-coverage.md for why there is no absolute floor.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "data" / "coverage_baseline.json"
COVERAGE_JSON_PATH = ROOT / "coverage.json"
# Wide on purpose: room for coverage.py's own run-to-run noise, not a quality bar.
_TOLERANCE_PERCENT = 1.0


def measure() -> float:
    subprocess.run(
        ["uv", "run", "pytest", "--cov=src/atlas", "--cov=evals", "--cov-report=json"],
        cwd=ROOT, check=True,
    )
    payload = json.loads(COVERAGE_JSON_PATH.read_text(encoding="utf-8"))
    return float(payload["totals"]["percent_covered"])


def verdict(current: float, baseline: float | None) -> tuple[int, str]:
    if baseline is None:
        return 0, (
            f"no baseline recorded yet. Current total: {current:.2f}%. Commit "
            f'{{"total_percent": {current:.2f}}} to '
            f"{BASELINE_PATH.relative_to(ROOT)} to start tracking regressions from here."
        )
    delta = current - baseline
    header = f"coverage: {current:.2f}% (baseline {baseline:.2f}%, change {delta:+.2f}%)"
    if delta < -_TOLERANCE_PERCENT:
        return 1, header + (
            f"\nREGRESSION: coverage dropped by more than {_TOLERANCE_PERCENT} percentage "
            "point(s) since the recorded baseline. If the drop is expected (dead code "
            f"removed, a module deliberately left untested), update "
            f"{BASELINE_PATH.relative_to(ROOT)} in the same change."
        )
    if delta > _TOLERANCE_PERCENT:
        return 0, header + (
            f"\ncoverage rose by more than {_TOLERANCE_PERCENT} percentage point(s). Not a "
            f"failure, but worth updating {BASELINE_PATH.relative_to(ROOT)} in the same "
            "change so the next regression check compares against where things actually are."
        )
    return 0, header


def main() -> int:
    current = measure()
    baseline = (
        float(json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["total_percent"])
        if BASELINE_PATH.exists() else None
    )
    code, message = verdict(current, baseline)
    print(message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
