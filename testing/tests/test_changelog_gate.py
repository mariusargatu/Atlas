"""The CHANGELOG gate (SP6 task 6, D37): a contract touching change with no CHANGELOG entry must
fail. Two halves, per the plan's own constraint (no git history walking in the hermetic lane):

  - The hermetic half (this file, folded into `task test`): `check_consistency` is a cheap FILE
    consistency check, no git involved. CHANGELOG.md's recorded ```contract-versions block must
    equal `contract_tools.loader.contract_versions()` exactly. It cannot detect a schema bump that
    forgot to update the block (that IS the failure it catches) nor a same version contract edit
    with no CHANGELOG entry at all (the git aware half's job); it only proves the repo's two
    self declared sources of truth agree right now.
  - The git aware half (`git_gate`, `task contracts:changelog-gate`, the push activated CI
    workflow) walks git history and is deliberately NOT exercised here.
"""
from __future__ import annotations

import pytest
from contract_tools import changelog_gate, loader


def test_repo_changelog_records_the_current_contract_tuple():
    """The committed CHANGELOG.md itself must already pass: this is the actual gate `task test`
    runs, not a fixture standing in for it."""
    ok, why = changelog_gate.check_consistency()
    assert ok, why


def test_check_consistency_passes_when_the_recorded_tuple_matches(tmp_path):
    current = loader.contract_versions()
    changelog = tmp_path / "CHANGELOG.md"
    block = "\n".join(f"{family}: {version}" for family, version in current.items())
    changelog.write_text(f"# Changelog\n\n```contract-versions\n{block}\n```\n")
    ok, why = changelog_gate.check_consistency(changelog)
    assert ok, why


def test_check_consistency_fails_on_a_stale_recorded_tuple(tmp_path):
    current = loader.contract_versions()
    changelog = tmp_path / "CHANGELOG.md"
    block = "\n".join(f"{family}: 0.0.0-stale" for family in current)
    changelog.write_text(f"# Changelog\n\n```contract-versions\n{block}\n```\n")
    ok, why = changelog_gate.check_consistency(changelog)
    assert not ok
    assert "does not match" in why


def test_check_consistency_fails_when_a_family_is_missing_from_the_recorded_block(tmp_path):
    current = loader.contract_versions()
    changelog = tmp_path / "CHANGELOG.md"
    families = list(current)[:-1]  # drop the last family entirely
    block = "\n".join(f"{family}: {current[family]}" for family in families)
    changelog.write_text(f"```contract-versions\n{block}\n```\n")
    ok, why = changelog_gate.check_consistency(changelog)
    assert not ok
    assert "does not match" in why


def test_parse_recorded_tuple_raises_a_worded_error_when_the_block_is_missing():
    with pytest.raises(changelog_gate.ChangelogFormatError, match="contract-versions"):
        changelog_gate.parse_recorded_tuple("no fenced block here at all")


def test_parse_recorded_tuple_raises_a_worded_error_on_a_malformed_line():
    with pytest.raises(changelog_gate.ChangelogFormatError, match="family: version"):
        changelog_gate.parse_recorded_tuple("```contract-versions\nnot a colon line\n```\n")


def test_parse_recorded_tuple_reads_family_version_pairs():
    text = "```contract-versions\ntrace: 0.1.0\ndataset: 0.2.0\n```\n"
    assert changelog_gate.parse_recorded_tuple(text) == {"trace": "0.1.0", "dataset": "0.2.0"}


def test_check_consistency_reports_a_missing_changelog_file(tmp_path):
    with pytest.raises(changelog_gate.ChangelogFormatError, match="no such file"):
        changelog_gate.check_consistency(tmp_path / "nope.md")
