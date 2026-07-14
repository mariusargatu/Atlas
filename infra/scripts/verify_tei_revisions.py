#!/usr/bin/env python3
"""Live /info revision check for the local tier's external TEI endpoint (SP5 task 3): confirms the
endpoint named by ATLAS_TEI_EMBED_URL/ATLAS_TEI_RERANK_URL actually SERVES the model revision
models.lock pins, not just that some endpoint answers -- a URL can point at the right host but a
stale or differently configured TEI process; this is the runtime side of that promise, a real HTTP
call against the live endpoint's own /info route, never mocked.

This checks reachability from THIS operator's own host. infra/scripts/k3d-up.sh's own step 3.5 (and
`task k3d:verify` below, which runs it again) separately proves the IN CLUSTER half (a pod, not just
this host, can reach the same endpoint) via the connectivity-check Job
(infra/charts/tei/templates/connectivity-check-job.yaml) that `tei.mode: external` renders.

Kept as a small standalone script, not inlined in the bash orchestrator
(infra/scripts/k3d-verify.sh), matching this directory's own precedent (record_image_digests.py):
the one piece of this pipeline that parses JSON and compares it against a committed file is easier
to read, and to unit test in isolation, as plain Python than as bash + jq string plumbing.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_LOCK = ROOT / "models.lock"


def fetch_info(url: str) -> dict:
    """A direct HTTP GET against `<url>/info`, no proxy, no cluster hop."""
    request = urllib.request.Request(f"{url.rstrip('/')}/info")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"GET {url}/info failed: {error}") from error


def check_revision(label: str, url: str, expected_model_id: str, expected_revision: str) -> bool:
    info = fetch_info(url)
    actual_model_id = info.get("model_id")
    actual_revision = info.get("model_sha")
    ok = actual_model_id == expected_model_id and actual_revision == expected_revision
    status = "OK" if ok else "MISMATCH"
    print(
        f"{label} ({url}): expected {expected_model_id}@{expected_revision}, "
        f"got {actual_model_id}@{actual_revision} [{status}]"
    )
    return ok


def main(argv: list[str] | None = None) -> None:
    del argv  # no CLI args today; kept for parity with this directory's other scripts
    embed_url = os.environ.get("ATLAS_TEI_EMBED_URL")
    rerank_url = os.environ.get("ATLAS_TEI_RERANK_URL")
    if not embed_url or not rerank_url:
        print(
            "ATLAS_TEI_EMBED_URL and ATLAS_TEI_RERANK_URL must both be set (source .env.fastlane "
            "first, see infra/README.md's k3d tier section).",
            file=sys.stderr,
        )
        sys.exit(1)

    lock = json.loads(MODELS_LOCK.read_text())
    embed = lock["embedding"][0]
    rerank = lock["reranker"][0]

    try:
        embed_ok = check_revision("tei-embed", embed_url, embed["model_id"], embed["revision"])
        rerank_ok = check_revision("tei-rerank", rerank_url, rerank["model_id"], rerank["revision"])
    except RuntimeError as error:
        print(f"live /info check failed: {error}", file=sys.stderr)
        sys.exit(1)

    if not (embed_ok and rerank_ok):
        print("one or more TEI /info revisions do not match models.lock", file=sys.stderr)
        sys.exit(1)
    print("both external TEI endpoints' /info revisions match models.lock")


if __name__ == "__main__":
    main(sys.argv[1:])
