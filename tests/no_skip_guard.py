from __future__ import annotations

from collections.abc import Generator

import pytest

SKIPPED = pytest.StashKey[tuple[str, ...]]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    # An expected failure also reports as skipped, so `wasxfail` is what keeps the markers
    # the single registry of allowed skips. The record lives on the session rather than the
    # module because a `pytester` run shares this module with the run that started it.
    if report.skipped and not hasattr(report, "wasxfail"):
        item.session.stash[SKIPPED] = (*item.session.stash.get(SKIPPED, ()), report.nodeid)
    return report


# Known gap: `pytest_runtest_makereport` only fires for tests that reached the run phase,
# so a module skipped at import time with `allow_module_level=True` disappears silently.
# `pytest_collectreport` would see it but receives no session to record against, and both
# alternatives leak state across the `pytester` runs this plugin is tested through. The
# invariant is held elsewhere too: conftest raises UsageError rather than skipping when the
# key is unset, a job collecting nothing exits 5, and tests/EXPECTED_MIN_TESTS floors fail
# the repo contract when a file stops being collected.


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    recorded = session.stash.get(SKIPPED, ())
    if recorded:
        session.exitstatus = 1
        print(f"skipped in a blocking job: {', '.join(recorded)}")
