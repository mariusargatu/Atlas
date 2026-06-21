"""Freshness check for the judge live provisional calibration artifact, as a TESTED pure function
plus a thin CLI.

The staleness GitHub workflow used to parse dated filenames and diff them against the wall clock
inline in bash: untested, and unreachable from the hermetic suite. This module is that logic as a
pure function that takes `today` as DATA (no clock inside, so the suite pins it), plus a CLI the
workflow runs passing `$(date -u +%F)`. Pure stdlib, no keys, no third-party imports, so
`python3 -m evals.staleness` runs on a bare runner with nothing synced (it lives directly under
`evals`, whose `__init__` imports nothing; the judge now lives at `testing/harness/judge/`, whose
own `__init__` imports nothing either, so this module's placement under `evals` is a naming
convenience, not a dependency firewall against a package that used to pull in the judge stack --
that package, the pre rewrite `evals.judge`, was retired by SP8's rewrite). Notification, not a
gate: the hermetic PR lane never depends on the wall clock; this is the weekly cron only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

# A dated calibration artifact is exactly `YYYY-MM-DD.md`. `latest.md` (the committed pointer) and any
# other name is not a dated artifact and is ignored, not an error.
_DATED = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.md$")

_NO_ARTIFACT = "no dated calibration artifact found"


@dataclass(frozen=True)
class StalenessVerdict:
    ok: bool
    message: str
    newest: str | None = None
    age_days: int | None = None


def check_staleness(filenames, today: date, max_age_days: int) -> StalenessVerdict:
    """Verdict on the freshness of the newest `YYYY-MM-DD.md` in `filenames`, as of `today`.

    Fails (ok=False) when there is no dated artifact at all (a DISTINCT message: an empty or absent
    directory is a different failure from a stale one, and the operator reads them differently), or
    when the newest dated artifact is more than `max_age_days` old. Non-dated names and a
    shape-but-not-a-real-date (`2026-13-40.md`) are ignored rather than crashing the check.
    """
    names = list(filenames)  # materialise once: `filenames` may be a one-shot iterator
    dated: list[tuple[date, str]] = []
    for name in names:
        m = _DATED.match(name)
        if not m:
            continue
        try:
            dated.append((date(int(m[1]), int(m[2]), int(m[3])), name))
        except ValueError:
            continue  # matches the shape but is not a real calendar date; skip, do not crash
    if not dated:
        return StalenessVerdict(False, f"{_NO_ARTIFACT} among {len(names)} file(s)")
    newest_date, newest_name = max(dated)
    age = (today - newest_date).days
    if age > max_age_days:
        return StalenessVerdict(
            False,
            f"STALE: {newest_name} is {age} days old (>{max_age_days}); "
            "run 'task judge-live' and commit a fresh dated artifact",
            newest_name,
            age,
        )
    return StalenessVerdict(True, f"OK: {newest_name} is {age} days old (<={max_age_days})", newest_name, age)


def main(argv: list[str] | None = None) -> int:
    """CLI: `python -m evals.staleness <artifact-dir> <today-YYYY-MM-DD> <max-age-days>`. Prints the
    verdict message and returns 0 (fresh) or 1 (stale / no artifact), 2 on bad usage. A missing dir is
    treated as no artifacts (the distinct 'no dated artifact' failure), never a traceback."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print("usage: python -m evals.staleness <artifact-dir> <today-YYYY-MM-DD> <max-age-days>", file=sys.stderr)
        return 2
    artifact_dir, today_str, max_age_str = args
    directory = Path(artifact_dir)
    filenames = [p.name for p in directory.iterdir()] if directory.is_dir() else []
    verdict = check_staleness(filenames, date.fromisoformat(today_str), int(max_age_str))
    print(verdict.message)
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
