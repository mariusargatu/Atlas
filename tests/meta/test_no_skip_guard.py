from __future__ import annotations

from pathlib import Path

from tests.no_skip_guard import SKIPPED

ROOT = Path(__file__).resolve().parents[2]

SKIPPING_FILE = """
    import pytest

    def test_that_skips():
        pytest.skip("collection fixture unavailable")
"""


def test_a_skip_turns_a_blocking_run_red_and_only_the_guard_makes_that_happen(pytester) -> None:
    # The control run is the point: without it, a guard that failed to import would look like
    # a guard that worked, since any non zero exit would satisfy the check.
    pytester.syspathinsert(ROOT)
    pytester.makepyfile(SKIPPING_FILE)
    without_guard = pytester.runpytest()
    without_guard.assert_outcomes(skipped=1)
    assert without_guard.ret == 0, "a skip already fails without the guard, so this proves nothing"

    pytester.makeconftest('pytest_plugins = ("tests.no_skip_guard",)')
    with_guard = pytester.runpytest()
    with_guard.assert_outcomes(skipped=1)
    assert with_guard.ret == 1, (
        f"exit status {with_guard.ret}, so something other than the guard ended this run"
    )
    with_guard.stdout.fnmatch_lines(["*skipped in a blocking job*test_that_skips*"])


def test_a_registered_expected_failure_is_not_counted_as_a_skip(pytester) -> None:
    # An expected failure also reports as skipped.
    pytester.syspathinsert(ROOT)
    pytester.makeconftest('pytest_plugins = ("tests.no_skip_guard",)')
    pytester.makepyfile("""
        import pytest

        @pytest.mark.xfail(strict=True, reason="a fixed size chunker divides a fact")
        def test_a_fact_shorter_than_the_limit_is_never_divided():
            assert False
    """)
    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)
    assert result.ret == 0, "the guard is treating a registered expected failure as a skip"


def test_the_skip_guard_records_nothing_for_the_session_that_is_running_it(pytester, request) -> None:
    # The inner run and the run starting it share one imported module, so a record held on
    # the module would leak: either this test's deliberate skip reddens the whole suite, or
    # an inner run wipes a real skip and the suite goes green with a skipped test in it.
    pytester.syspathinsert(ROOT)
    pytester.makeconftest('pytest_plugins = ("tests.no_skip_guard",)')
    pytester.makepyfile(SKIPPING_FILE)
    assert pytester.runpytest().ret == 1
    assert request.session.stash.get(SKIPPED, ()) == (), (
        "the inner run's skip was recorded against the session running this test"
    )
