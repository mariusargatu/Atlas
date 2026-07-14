"""The pipeline: one command, verified output, byte identical rebuilds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from corpus_tools import build, registry, verify

COMMITTED = Path("corpus/rendered/corpus-0.1.1")


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_build_twice_is_byte_identical(tmp_path: Path) -> None:
    first = build.build_corpus("corpus-t", tmp_path / "a", seed=7)
    second = build.build_corpus("corpus-t", tmp_path / "b", seed=7)
    manifest_a = json.loads((first / "manifest.json").read_text())
    manifest_b = json.loads((second / "manifest.json").read_text())
    assert manifest_a["content_hash"] == manifest_b["content_hash"]


def test_committed_corpus_matches_its_manifest() -> None:
    manifest = json.loads((COMMITTED / "manifest.json").read_text())
    assert manifest["corpus_version"] == "corpus-0.1.1"
    assert manifest["doc_count"] >= 40
    assert set(manifest["never_rendered"]) >= {"plan-quantum-5g", "fee-teleport-setup"}
    for doc_id, digest in manifest["docs"].items():
        import hashlib

        actual = hashlib.sha256((COMMITTED / "docs" / f"{doc_id}.txt").read_bytes()).hexdigest()
        assert actual == digest, f"{doc_id} drifted from its manifest hash"


def test_committed_corpus_is_fresh_against_the_registry(tmp_path: Path) -> None:
    """The staleness gate: the committed corpus-0.1.1 must be exactly what the current
    registry and templates produce, not a snapshot that has drifted out from under them.

    Two independent checks, because they catch different drift: rebuilding and comparing
    content_hash only proves again what is actually rendered (never_rendered fields are, by
    design, never rendered); a direct sha256 comparison against the registry files pins the
    registry byte for byte, including fields on hidden entities that no rebuild would ever
    surface as a content_hash difference.
    """
    manifest = json.loads((COMMITTED / "manifest.json").read_text())

    fresh_dir = build.build_corpus(manifest["corpus_version"], tmp_path, seed=manifest["seed"])
    fresh_manifest = json.loads((fresh_dir / "manifest.json").read_text())
    assert fresh_manifest["content_hash"] == manifest["content_hash"], (
        "rebuilding corpus-0.1.1 from the current registry and templates no longer reproduces "
        "the committed rendered docs; run task corpus:build and commit the regenerated corpus"
    )

    registry_dir = Path("corpus/registry")
    for name, committed_sha256 in manifest["registry_sha256"].items():
        actual_sha256 = hashlib.sha256((registry_dir / name).read_bytes()).hexdigest()
        assert actual_sha256 == committed_sha256, (
            f"{name} has changed since corpus-0.1.1 was built (sha256 no longer matches the "
            "committed manifest); rebuild and recommit the corpus"
        )


def test_provenance_sidecars_exist_for_every_doc() -> None:
    manifest = json.loads((COMMITTED / "manifest.json").read_text())
    for doc_id in manifest["docs"]:
        sidecar = COMMITTED / "provenance" / f"{doc_id}.json"
        assert sidecar.exists()
        assert json.loads(sidecar.read_text())["placements"]


def test_failing_rebuild_leaves_prior_corpus_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_root = tmp_path / "out"
    corpus_dir = build.build_corpus("corpus-t", out_root, seed=7)
    before = _snapshot(corpus_dir)

    monkeypatch.setattr(verify, "verify_corpus", lambda docs, reg: ("forced violation for the test",))

    with pytest.raises(SystemExit):
        build.build_corpus("corpus-t", out_root, seed=7)

    assert _snapshot(corpus_dir) == before
    assert (corpus_dir / "manifest.json").exists()


def test_integrity_violation_surfaces_before_render_ever_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An edgeless promotion is both an integrity_report violation (compile.py) and, until
    # guarded, a bare StopIteration out of render.py's _promo_context. The gate order matters:
    # build_corpus must hard fail on the integrity violation before render_corpus ever runs, so
    # the failure a caller sees is the named integrity message, not a renderer crash.
    bad_registry = registry.Registry(
        entities=(
            registry.Entity(id="promotion-x", kind="promotion", render=True, fields={"discount": "5"}),
        ),
        edges=(),
        contradictions=(),
    )
    monkeypatch.setattr(registry, "load_registry", lambda paths: bad_registry)

    with pytest.raises(SystemExit) as exc_info:
        build.build_corpus("corpus-t", tmp_path / "out", seed=7)

    message = str(exc_info.value)
    assert "promotion-x" in message
    assert "no applies_to edge" in message
