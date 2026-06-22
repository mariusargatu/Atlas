"""The corpus_version build pipeline: one command, verified output, byte identical rebuilds.

Chains the upstream stages end to end: load the registry (core + generated variants), compile it
to a throwaway SQLite so the registry's own integrity gates run (compiler.integrity_report),
render with the given seed, then verify the rendered docs against the registry (verify.verify_corpus).
Any violation from either gate is a hard fail (SystemExit), never a partial write: the corpus dir
either lands complete and verified, or not at all.

Stage then rename (not write-in-place): everything is built into a temp staging directory inside
out_root, and the combined violations check runs against that staging tree BEFORE the pre-existing
corpus_dir is touched. Only once verification is confirmed clean does the old corpus_dir (if any)
get removed and the staging dir renamed into place. A failing build must leave any prior committed
corpus byte for byte untouched, not wiped ahead of the gate that was supposed to protect it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from corpus_tools import compile as reg_compile
from corpus_tools import expand, registry, render, verify

CORE = Path("corpus/registry/core.yaml")
GENERATED = Path("corpus/registry/generated_variants.yaml")
TEMPLATES = Path("corpus/templates")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_corpus(corpus_version: str, out_root: Path, seed: int) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    corpus_dir = out_root / corpus_version

    reg = registry.load_registry([CORE, GENERATED])

    with tempfile.TemporaryDirectory(dir=out_root) as tmp_dir:
        tmp_root = Path(tmp_dir)

        db_path = tmp_root / "registry.sqlite"
        reg_compile.compile_registry(reg, db_path)
        integrity_violations = reg_compile.integrity_report(db_path)

        # Gate BEFORE render_corpus even runs: a registry integrity violation (e.g. an edgeless
        # promotion) is a fact authoring error the compiler can name precisely. Rendering it
        # anyway would surface a renderer crash instead of a diagnosable message, and it wastes
        # work on a registry the build is about to reject either way.
        if integrity_violations:
            raise SystemExit(f"corpus integrity failed: {integrity_violations}")

        docs = render.render_corpus(reg, TEMPLATES, seed=seed)
        verify_violations = verify.verify_corpus(docs, reg)

        # Gate BEFORE anything touches corpus_dir: a failing build must leave any pre existing
        # corpus byte for byte untouched, not wiped ahead of the check that was meant to protect it.
        if verify_violations:
            raise SystemExit(f"corpus verification failed: {verify_violations}")

        staging_dir = tmp_root / "staged"
        docs_dir = staging_dir / "docs"
        provenance_dir = staging_dir / "provenance"
        docs_dir.mkdir(parents=True)
        provenance_dir.mkdir(parents=True)

        doc_hashes: dict[str, str] = {}
        doc_type_counts: dict[str, int] = {}
        for doc in docs:
            (docs_dir / f"{doc.doc_id}.txt").write_text(doc.text)
            doc_hashes[doc.doc_id] = _sha256_bytes(doc.text.encode())
            doc_type_counts[doc.doc_type] = doc_type_counts.get(doc.doc_type, 0) + 1

            placement_entries = []
            for placement in doc.placements:
                start, end = placement.span
                sliced = doc.text[start:end]
                entry = {
                    "fact_ref": placement.fact_ref,
                    "value": placement.value,
                    "span": [start, end],
                }
                # A span whose slice does not literally contain the raw value is exactly a
                # prose branch (contract_months=0 rendering as "No contract. Cancel any time.",
                # which never contains the digit "0" as a token): record the rendered clause
                # alongside the value rather than only the value's own, unlocatable occurrence.
                # Literal placements omit this field; there is nothing beyond the value to add.
                if placement.value not in sliced:
                    entry["clause"] = sliced
                placement_entries.append(entry)

            sidecar = {
                "doc_id": doc.doc_id,
                "doc_type": doc.doc_type,
                "placements": placement_entries,
            }
            (provenance_dir / f"{doc.doc_id}.json").write_text(
                json.dumps(sidecar, sort_keys=True, indent=2) + "\n"
            )

        registry_sha256 = {
            path.name: _sha256_bytes(path.read_bytes()) for path in (CORE, GENERATED)
        }
        never_rendered = sorted(e.id for e in reg.entities if not e.render)
        contradiction_ids = sorted(c.id for c in reg.contradictions)
        content_hash = _sha256_bytes("\n".join(sorted(doc_hashes.values())).encode())

        manifest = {
            "corpus_version": corpus_version,
            "seed": seed,
            "registry_sha256": registry_sha256,
            "doc_count": len(docs),
            "doc_type_counts": doc_type_counts,
            "docs": doc_hashes,
            "never_rendered": never_rendered,
            "contradiction_ids": contradiction_ids,
            "content_hash": content_hash,
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )

        # Only now, with a fully built and verified staging tree in hand, replace any prior
        # corpus_dir: remove-then-rename, both on the same filesystem (staging_dir lives under
        # out_root), so the window where corpus_dir is absent is as short as a single os.rename.
        if corpus_dir.exists():
            shutil.rmtree(corpus_dir)
        shutil.move(str(staging_dir), str(corpus_dir))

    return corpus_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="corpus_tools.build")
    parser.add_argument("--version", required=True, help="corpus_version, e.g. corpus-0.1.1")
    parser.add_argument("--seed", type=int, default=expand.DEFAULT_SEED, help="render seed")
    parser.add_argument("--out", default="corpus/rendered", help="output root directory")
    args = parser.parse_args(argv)

    corpus_dir = build_corpus(args.version, Path(args.out), seed=args.seed)
    print(f"wrote {corpus_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
