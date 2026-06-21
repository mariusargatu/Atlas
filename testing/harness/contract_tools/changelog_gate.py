"""The CHANGELOG gate (SP6 task 6, D37): "a contract touching change without a CHANGELOG entry
fails" (the plan's own words). Split into two halves per the plan's own hermetic constraint (no git
history walking in the hermetic lane):

  - `check_consistency` (this module's hermetic half, `testing/tests/test_changelog_gate.py`, folded
    into `task test`): a cheap FILE consistency check, no git involved. CHANGELOG.md carries a
    ```contract-versions fenced block recording the contract tuple it was last updated against; this
    function compares that recorded tuple to `contract_tools.loader.contract_versions()`, the
    schemas as they actually sit in the working tree right now. A mismatch means CHANGELOG.md is
    stale relative to a contract version bump that already landed.
  - `git_gate` (this module's git aware half, `task contracts:changelog-gate`, wired into the push
    activated CI workflow ONLY -- never a hermetically collected test path): walks git history
    (`git diff --name-only`) to catch the broader case the file check cannot see on its own, a
    contracts/*/schema.json change landing in the same push with NO CHANGELOG.md change at all
    (version bumped or not). Needs a real git history to diff against, so it is never imported by a
    test `task test` collects.

Mirrors `contract_tools/diff.py`'s own shape: pure/hermetic functions and a git aware CLI entry
point coexist in one module, and only the pure half is exercised by the hermetic test suite (see
that module's own `_load_from_git`/`_compare_against_ref`, never unit tested either).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from contract_tools import loader

CHANGELOG_PATH = loader.CONTRACTS_DIR.parent / "CHANGELOG.md"

_BLOCK_RE = re.compile(r"```contract-versions\n(.*?)\n```", re.DOTALL)


class ChangelogFormatError(Exception):
    """CHANGELOG.md is missing, or has no parseable ```contract-versions fenced block."""


def parse_recorded_tuple(text: str) -> dict[str, str]:
    """Extract the `{family: version}` tuple from a ```contract-versions fenced code block: one
    `family: version` pair per line, blank lines ignored. Raises `ChangelogFormatError`, never
    returns a partial or guessed result, on a missing block or a line with no colon."""
    match = _BLOCK_RE.search(text)
    if not match:
        raise ChangelogFormatError(
            "CHANGELOG.md has no ```contract-versions fenced block recording the current contract "
            "tuple. Add one (see contract_tools.changelog_gate's own module docstring for the "
            "format: one 'family: version' line per contract family, inside a "
            "```contract-versions ... ``` fence)."
        )
    recorded: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        family, sep, version = line.partition(":")
        if not sep:
            raise ChangelogFormatError(
                f"CHANGELOG.md's contract-versions block has a line that is not 'family: version': {line!r}"
            )
        recorded[family.strip()] = version.strip()
    return recorded


def check_consistency(changelog_path: Path = CHANGELOG_PATH) -> tuple[bool, str]:
    """The hermetic half: CHANGELOG.md's recorded contract tuple must equal
    `contract_tools.loader.contract_versions()` exactly, family for family. No git involved, no
    history walking: this only compares two files already sitting in the working tree, so it is
    safe inside the hermetic lane (no keys, no network, no git subprocess)."""
    try:
        text = changelog_path.read_text()
    except FileNotFoundError as exc:
        raise ChangelogFormatError(f"cannot read {changelog_path}: no such file") from exc
    recorded = parse_recorded_tuple(text)
    current = loader.contract_versions()
    if recorded == current:
        return True, "CHANGELOG.md's recorded contract tuple matches contract_versions()"
    return False, (
        f"CHANGELOG.md's recorded contract tuple {recorded} does not match "
        f"contract_tools.loader.contract_versions() {current}. Update the ```contract-versions "
        "block in CHANGELOG.md in the SAME commit as the contracts/*/schema.json change."
    )


# ---- git aware half: subprocess based, never imported by a hermetically collected test ----------


class GitDiffError(Exception):
    """Raised when `git diff --name-only` itself fails (a bad ref, no git history, ...)."""


def _changed_paths(git_ref: str) -> list[str]:
    """Paths that differ between `git_ref` and the working tree (`git diff --name-only`, the same
    subprocess shape `contract_tools.diff._load_from_git` already uses for a single file). Only
    ever called by `git_gate`, itself only ever called by `task contracts:changelog-gate` and the
    push activated CI workflow: never a path `task test` collects."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{git_ref}...HEAD"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise GitDiffError(result.stderr.strip())
    return [line for line in result.stdout.splitlines() if line]


def git_gate(git_ref: str = "main") -> int:
    """The full gate: fails if any `contracts/*/schema.json` changed since `git_ref` with no
    `CHANGELOG.md` change in the same diff (the plan's own "a contract touching change without a
    CHANGELOG entry fails"), then re runs the hermetic file consistency check so a git aware CI run
    also catches a same commit CHANGELOG edit that still recorded the wrong tuple."""
    try:
        changed = _changed_paths(git_ref)
    except GitDiffError as exc:
        print(f"error: cannot diff against {git_ref!r}: {exc}")
        return 2
    contract_changed = [p for p in changed if p.startswith("contracts/") and p.endswith("schema.json")]
    changelog_changed = "CHANGELOG.md" in changed
    if contract_changed and not changelog_changed:
        print(
            f"error: contract schema changed since {git_ref} with no CHANGELOG.md entry: {contract_changed}"
        )
        return 1
    ok, why = check_consistency()
    print(why)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="contract_tools.changelog_gate")
    parser.add_argument("--git-ref", default="main", help="compare against this git ref (default main)")
    args = parser.parse_args(argv)
    return git_gate(args.git_ref)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CHANGELOG_PATH",
    "ChangelogFormatError",
    "GitDiffError",
    "check_consistency",
    "git_gate",
    "main",
    "parse_recorded_tuple",
]
