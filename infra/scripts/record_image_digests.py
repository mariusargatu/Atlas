#!/usr/bin/env python3
"""Rewrite the marker delimited generated image digest block in a values file (D37: `task k3d:up`
records the real digest it just built and pushed into the local values file, per the SP5 plan's
own instruction). Kept as a small standalone script, not inlined in the bash orchestrator
(`infra/scripts/k3d-up.sh`), so the one piece of this pipeline that edits a human authored,
commented values file in place is easy to read and test in isolation.

Only the text between the BEGIN/END marker comments is replaced; everything else in the file
(comments, unrelated keys) is left byte identical.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

BEGIN_MARKER = "# BEGIN k3d:up generated image digests"
END_MARKER = "# END k3d:up generated image digests"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")


def render_block(*, registry_host: str, pgvector_digest: str, backend_digest: str, web_digest: str) -> str:
    return (
        f"{BEGIN_MARKER} -- task k3d:up rewrites this block on every image build; do\n"
        "# not hand edit. Placeholder (all zero sha256) until the first `task k3d:up` run records the real\n"
        "# digests pushed to the k3d local registry (D37: repo@sha256:..., never a floating tag).\n"
        f"# {registry_host} is the registry's CLUSTER INTERNAL address (a k3s containerd registry mirror\n"
        '# to the "atlas-registry" docker container on port 5000, verified against this cluster\'s own\n'
        "# /etc/rancher/k3s/registries.yaml); the HOST VISIBLE push address is the SAME container's\n"
        "# separately published, randomly assigned host port, which `task k3d:up` resolves at build time\n"
        '# (see infra/README.md\'s "Registry addressing" note).\n'
        "images:\n"
        "  postgresPgvector:\n"
        f"    repository: {registry_host}/atlas-postgres-pgvector\n"
        f'    digest: "{pgvector_digest}"\n'
        "  backend:\n"
        f"    repository: {registry_host}/atlas-backend\n"
        f'    digest: "{backend_digest}"\n'
        "  web:\n"
        f"    repository: {registry_host}/atlas-web\n"
        f'    digest: "{web_digest}"\n'
        f"{END_MARKER}"
    )


def rewrite(
    values_path: pathlib.Path, *, registry_host: str, pgvector_digest: str, backend_digest: str, web_digest: str
) -> None:
    text = values_path.read_text()
    if BEGIN_MARKER not in text or END_MARKER not in text:
        raise SystemExit(
            f"{values_path}: missing the generated image digest marker block "
            f"({BEGIN_MARKER!r} .. {END_MARKER!r}). Was it hand edited away? Restore it from "
            "infra/environments/local/values.yaml's own history for the expected shape."
        )
    pattern = re.compile(re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
    new_block = render_block(
        registry_host=registry_host, pgvector_digest=pgvector_digest, backend_digest=backend_digest,
        web_digest=web_digest,
    )
    new_text, count = pattern.subn(new_block, text, count=1)
    if count != 1:
        raise SystemExit(f"{values_path}: expected exactly one marker block, found {count}")
    values_path.write_text(new_text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values-file", required=True, type=pathlib.Path)
    parser.add_argument(
        "--registry-host", default="atlas-registry:5000", help="cluster internal registry host:port"
    )
    parser.add_argument("--pgvector-digest", required=True, help="sha256:<64 hex>, from `docker push`")
    parser.add_argument("--backend-digest", required=True, help="sha256:<64 hex>, from `docker push`")
    parser.add_argument("--web-digest", required=True, help="sha256:<64 hex>, from `docker push` (SP5 task 4)")
    args = parser.parse_args(argv)

    for label, digest in (
        ("pgvector", args.pgvector_digest), ("backend", args.backend_digest), ("web", args.web_digest),
    ):
        if not _DIGEST_RE.fullmatch(digest):
            raise SystemExit(f"--{label}-digest {digest!r} is not digest shaped (sha256:<64 hex>)")

    rewrite(
        args.values_file,
        registry_host=args.registry_host,
        pgvector_digest=args.pgvector_digest,
        backend_digest=args.backend_digest,
        web_digest=args.web_digest,
    )
    print(f"{args.values_file}: recorded {args.registry_host}/atlas-postgres-pgvector@{args.pgvector_digest}")
    print(f"{args.values_file}: recorded {args.registry_host}/atlas-backend@{args.backend_digest}")
    print(f"{args.values_file}: recorded {args.registry_host}/atlas-web@{args.web_digest}")


if __name__ == "__main__":
    main(sys.argv[1:])
