"""The one deliberate checkpoint between the real catalog and the test suite's own, independently
maintained expectations of it. If this fails, the real catalog changed and
testing/tests/fixtures/catalog_expectations.py was not updated to match on purpose. Fix the
fixture, never this test's assertions.
"""
from __future__ import annotations

from atlas.domain.catalog import CATALOG

from testing.tests.fixtures.catalog_expectations import EXPECTED_CURRENT_PLAN, EXPECTED_LEGACY_PLAN


def test_current_plan_matches_the_fixture_the_test_suite_owns():
    assert CATALOG["plan_current_fast"] == EXPECTED_CURRENT_PLAN


def test_legacy_plan_matches_the_fixture_the_test_suite_owns():
    assert CATALOG["plan_legacy_value"] == EXPECTED_LEGACY_PLAN
