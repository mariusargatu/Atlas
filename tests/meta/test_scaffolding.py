from __future__ import annotations

import json
from pathlib import Path

from hypothesis import settings as property_settings

ROOT = Path(__file__).resolve().parents[2]


def test_the_scaffolding_holds_together(pytestconfig) -> None:
    # Shape, not exact values: every later piece bumps its own job's entry in the floor file,
    # so a pinned snapshot would fail for reasons unrelated to the scaffolding.
    floor = json.loads((ROOT / "tests" / "EXPECTED_MIN_TESTS").read_text(encoding="utf-8"))
    assert floor.keys() >= {"suite_integrity"}
    assert all(isinstance(count, int) and count > 0 for count in floor.values())

    checks = property_settings.get_profile("checks")
    development = property_settings.get_profile("development")
    thorough = property_settings.get_profile("thorough")
    assert (checks.max_examples, development.max_examples, thorough.max_examples) == (200, 25, 2000)
    assert property_settings().max_examples == checks.max_examples, "checks is not active by default"

    assert pytestconfig.getoption("strict_markers") is True, "an unregistered marker would pass silently"
    assert "error" in (pytestconfig.getoption("pythonwarnings") or []), "warnings are not errors here"
    assert not pytestconfig.pluginmanager.hasplugin("randomly"), "pytest-randomly is loaded despite addopts"
