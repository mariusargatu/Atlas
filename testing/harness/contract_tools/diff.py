"""Classify schema changes against the evolution rules (HLD D25) and check the version bump.

Pure functions on schema dicts. Path separator is "/" because attribute
names themselves contain dots (atlas.stage.embed_ms).

--git-ref mode resolves both the git-show path and the working tree path relative to the
repo root, since `task contracts:diff` (Taskfile.yml) always runs from there.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from contract_tools import loader

LEVELS = ("patch", "minor", "major")


@dataclass(frozen=True)
class ChangeReport:
    level: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"invalid ChangeReport level {self.level!r}: expected one of {LEVELS}")


def flatten_properties(schema: dict, prefix: str = "") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, spec in schema.get("properties", {}).items():
        path = f"{prefix}{name}"
        out[path] = spec
        child = spec.get("items") if spec.get("type") == "array" else spec
        if isinstance(child, dict) and child.get("properties"):
            out.update(flatten_properties(child, prefix=f"{path}/"))
    for name, def_schema in schema.get("$defs", {}).items():
        out.update(flatten_properties(def_schema, prefix=f"{prefix}$defs/{name}/"))
    return out


def _required_at(schema: dict, prefix: str = "") -> frozenset[str]:
    found = {f"{prefix}{name}" for name in schema.get("required", [])}
    for name, spec in schema.get("properties", {}).items():
        child = spec.get("items") if spec.get("type") == "array" else spec
        if isinstance(child, dict) and (child.get("properties") or child.get("required")):
            found |= _required_at(child, prefix=f"{prefix}{name}/")
    for name, def_schema in schema.get("$defs", {}).items():
        found |= _required_at(def_schema, prefix=f"{prefix}$defs/{name}/")
    return frozenset(found)


def required_paths(schema: dict) -> frozenset[str]:
    return _required_at(schema)


# Keys already modeled by dedicated logic in classify_change: type/enum have their own
# reasons, properties/required/items drive recursion and the required-paths diff. Anything
# else that differs between two specs at the same path is an unmodeled constraint change.
_MODELED_KEYWORDS = frozenset({"type", "enum", "properties", "required", "items"})

# Purely descriptive keywords: a change limited to this set carries no schema-validation
# consequence, so it must not fall through the unmodeled-change MINOR floor.
_METADATA_KEYWORDS = frozenset({"title", "description", "$comment", "examples", "deprecated"})


def _changed_keywords(old_spec: dict, new_spec: dict) -> list[str]:
    keys = (set(old_spec) | set(new_spec)) - _MODELED_KEYWORDS - _METADATA_KEYWORDS
    return sorted(key for key in keys if old_spec.get(key) != new_spec.get(key))


def _type_of(spec: dict) -> object:
    """A JSON Schema "type" is either a string or a union list; list order is not semantic."""
    value = spec.get("type")
    return set(value) if isinstance(value, list) else value


def classify_change(old: dict, new: dict) -> ChangeReport:
    major: list[str] = []
    minor: list[str] = []
    old_flat, new_flat = flatten_properties(old), flatten_properties(new)

    for path in sorted(set(old_flat) - set(new_flat)):
        major.append(f"removed property: {path}")
    for path in sorted(set(new_flat) - set(old_flat)):
        minor.append(f"added optional property: {path}")

    for path in sorted(set(old_flat) & set(new_flat)):
        old_spec, new_spec = old_flat[path], new_flat[path]
        old_type, new_type = _type_of(old_spec), _type_of(new_spec)
        old_enum, new_enum = old_spec.get("enum"), new_spec.get("enum")
        enum_dropped = bool(old_enum) and not new_enum

        # Explicit-on-both-sides is a modeled retype. Explicit-on-one-side is a symmetric
        # widen/narrow: gaining a type where none existed is a narrowing (an open field now
        # rejects everything outside the type), losing one is a widening. The one exception:
        # if the enum was dropped on this same path, the enum already constrained the
        # accepted values, and the new explicit type is wider than (or equal to) what the
        # enum allowed, so this is not a separate narrowing on top of the enum-drop widening.
        if old_type is not None and new_type is not None:
            if old_type != new_type:
                major.append(f"retyped property: {path}")
        elif old_type is None and new_type is not None:
            if not enum_dropped:
                major.append(f"narrowed: type constraint added on {path}")
        elif old_type is not None and new_type is None:
            minor.append(f"widened: type constraint dropped on {path}")

        if old_enum and new_enum:
            removed = sorted(set(old_enum) - set(new_enum))
            added = sorted(set(new_enum) - set(old_enum))
            if removed:
                major.append(f"narrowed enum on {path}: removed {removed}")
            if added:
                minor.append(f"widened enum on {path}: added {added}")
        elif enum_dropped:
            minor.append(f"widened: enum constraint dropped on {path}")
        elif not old_enum and new_enum:
            major.append(f"narrowed: enum constraint added on {path}")

        changed_keywords = _changed_keywords(old_spec, new_spec)
        if changed_keywords:
            minor.append(f"unmodeled change on {path}: {changed_keywords}")

    for path in sorted(required_paths(old) - required_paths(new)):
        if path in old_flat and path in new_flat:
            minor.append(f"widened: {path} no longer required")

    newly_required = sorted(required_paths(new) - required_paths(old))
    if newly_required:
        major.append(f"new required: {newly_required}")

    if major:
        return ChangeReport("major", tuple(major + minor))
    if minor:
        return ChangeReport("minor", tuple(minor))
    if old != new:
        return ChangeReport("patch", ("metadata only change",))
    return ChangeReport("patch", ("no change",))


def _parse_version(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError as exc:
        raise ValueError(
            f"invalid version {version!r}: expected MAJOR.MINOR.PATCH with integer components"
        ) from exc


def required_bump(old_version: str, new_version: str, report: ChangeReport) -> tuple[bool, str]:
    old_v = _parse_version(old_version)
    new_v = _parse_version(new_version)
    if report.level == "patch" and report.reasons == ("no change",):
        return (old_v == new_v, "version must not change when the schema is unchanged")
    if new_v <= old_v:
        return (False, "version must increase")
    if report.level == "major":
        if old_v[0] == 0:
            ok = new_v[0] > 0 or new_v[1] > old_v[1]
            return (ok, "ok" if ok else "breaking change in 0.x requires at least a MINOR bump")
        ok = new_v[0] == old_v[0] + 1
        return (ok, "ok" if ok else "breaking change requires a MAJOR bump")
    if report.level == "minor":
        ok = new_v[0] > old_v[0] or new_v[1] > old_v[1]
        return (ok, "ok" if ok else "additive change requires at least a MINOR bump")
    return (True, "ok")


class GitShowError(Exception):
    """Raised when `git show REF:PATH` fails: a bad ref, or a path missing at that ref."""


class LoadError(Exception):
    """Raised when a local schema file is missing or is not valid JSON."""


def _load_json(path: str) -> dict:
    try:
        text = Path(path).read_text()
    except FileNotFoundError as exc:
        raise LoadError(f"cannot read {path}: no such file") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LoadError(f"cannot read {path}: invalid JSON ({exc})") from exc


def _load_from_git(ref: str, repo_path: str) -> dict:
    shown = subprocess.run(
        ["git", "show", f"{ref}:{repo_path}"], capture_output=True, text=True
    )
    if shown.returncode != 0:
        raise GitShowError(shown.stderr.strip())
    return json.loads(shown.stdout)


def family_schema_paths() -> list[str]:
    """The schema path for every contract family, derived from loader.FAMILIES."""
    return [f"contracts/{family}/schema.json" for family in loader.FAMILIES]


def _report(old_schema: dict, new_schema: dict) -> int:
    report = classify_change(old_schema, new_schema)
    try:
        ok, why = required_bump(
            old_schema.get("x-contract-version", "0.0.0"),
            new_schema.get("x-contract-version", "0.0.0"),
            report,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(f"change level: {report.level}")
    for reason in report.reasons:
        print(f"  - {reason}")
    print(f"version: {old_schema.get('x-contract-version')} -> {new_schema.get('x-contract-version')}: {why}")
    return 0 if ok else 1


def _compare_against_ref(path: str, ref: str) -> int:
    """Compare the working tree copy of `path` against `ref`. Exit 2 on a git/read failure."""
    try:
        old_schema = _load_from_git(ref, path)
        new_schema = _load_json(path)
    except GitShowError as exc:
        print(f"error: cannot read {path} at ref {ref}: {exc}")
        return 2
    except LoadError as exc:
        print(f"error: {exc}")
        return 2
    return _report(old_schema, new_schema)


def _compare_files(old_path: str, new_path: str) -> int:
    try:
        old_schema = _load_json(old_path)
        new_schema = _load_json(new_path)
    except LoadError as exc:
        print(f"error: {exc}")
        return 2
    return _report(old_schema, new_schema)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="contract_tools.diff")
    parser.add_argument("old", nargs="?", help="old schema file, or the schema path when --git-ref is used")
    parser.add_argument("new", nargs="?", help="new schema file (omit with --git-ref)")
    parser.add_argument("--git-ref", help="compare the working tree file against this git ref")
    parser.add_argument(
        "--all-families",
        action="store_true",
        help="compare every contract family's schema.json against --git-ref (requires --git-ref)",
    )
    args = parser.parse_args(argv)

    if args.all_families:
        if not args.git_ref:
            parser.error("--all-families requires --git-ref")
        worst = 0
        for path in family_schema_paths():
            print(f"== {path}")
            worst = max(worst, _compare_against_ref(path, args.git_ref))
        return worst

    if args.git_ref:
        if not args.old:
            parser.error("--git-ref requires a schema path")
        return _compare_against_ref(args.old, args.git_ref)

    if args.old and args.new:
        return _compare_files(args.old, args.new)

    parser.print_usage()
    return 2


if __name__ == "__main__":
    sys.exit(main())
