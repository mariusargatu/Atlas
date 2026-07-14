"""SP10 task 5's sentinel go or no go gate: the thin CI level wiring `.github/workflows/
burst-benchmark.yml` needs (D29/D41, HLD section 7.3's Burst benchmark row) -- one synchronous
invocation of `sentinel.probe`'s existing known-answer/known-refusal/known-injection queries
against the freshly provisioned burst endpoint, HARD FAILING the job before any paid step (the xk6-sse
build is free; `task matrix:live`/`task load:k6` spend real provider and burst money) on a single red
class, never averaged, never partially admitted.

REUSES `sentinel.probe.run_probe`/`PROBE_QUERIES` UNCHANGED (the plan's own "reuse the module, never
reimplement" instruction): this file adds no new query class, no new grading rule, nothing `probe.py`
does not already own. The one genuine difference from `sentinel.probe.main` (the already deployed
CronJob's own entrypoint, SP6 task 5): `main` here never calls `push_to_gateway`. The in cluster
Pushgateway (`http://pushgateway:9091`, `infra/charts/atlas-monitoring`) is a `ClusterIP` Service,
unreachable from a GitHub hosted runner outside the cluster network, and pushing the SAME
`atlas_probe_success` gauge series from a one shot CI invocation would double write a metric the
standing CronJob already owns as its one writer. This gate's own job is narrower: GO or NO GO for THIS
one workflow run, reported to the job log (and the workflow's `GITHUB_STEP_SUMMARY`, wired by the
calling workflow, not this module), never Prometheus.

Reaching the endpoint: `ATLAS_PROBE_BASE_URL` must name wherever the caller already arranged to reach
the freshly provisioned backend (`burst-benchmark.yml`'s own step opens a `kubectl port-forward` to
the in cluster `backend` Service, exactly like `infra/scripts/k3d-smoke.sh` already does for CNPG's
`atlas-pg-rw` Service, and points this gate at the forwarded local port). Unlike
`sentinel.probe.main`, which defaults to `http://backend:8000` (a hostname that only resolves inside
the cluster network the CronJob itself runs in), this gate has NO implicit default: a CI runner
reaching the same Service needs an explicit URL named on purpose every time, never a default that
could silently probe nothing (or the wrong endpoint) without the caller noticing.
"""
from __future__ import annotations

import os
import sys

import httpx

from sentinel.probe import PROBE_QUERIES, run_probe

DEFAULT_CUSTOMER_ID = "cust_current"  # Sarah, the SEED "happy path" account (domain/accounts.py)


def gate(client: httpx.Client, *, customer_id: str = DEFAULT_CUSTOMER_ID) -> tuple[bool, dict[str, bool]]:
    """GO (`True`) only if every `PROBE_QUERIES` class passed; NO GO (`False`) if any single class
    failed -- a single red query fails the whole gate, the same "no partial credit" reading
    `sentinel.probe.main`'s own `all(results.values())` return already encodes. Surfaced here as its
    own named boolean (rather than an exit code alone) so a caller, or a test, can assert on the
    decision directly, alongside the full per class breakdown for the job log."""
    results = run_probe(client, customer_id=customer_id)
    return all(results.values()), results


def main(
    *,
    base_url: str | None = None,
    customer_id: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """The CLI entrypoint (`python -m sentinel.ci_gate`, `task sentinel:gate`). Every parameter
    defaults to reading the matching `ATLAS_PROBE_*` env var, unset only for tests, which inject a
    `transport` (`httpx.MockTransport`) instead of ever touching a real socket -- the same
    fully injectable shape `sentinel.probe.main` already establishes. Returns 0 (GO) only if every
    probe class passed, 1 (NO GO, or a missing endpoint) otherwise."""
    base_url = base_url or os.environ.get("ATLAS_PROBE_BASE_URL")
    if not base_url:
        print(
            "ATLAS_PROBE_BASE_URL is not set: refusing to run the sentinel gate against an "
            "unspecified endpoint (unlike sentinel.probe.main's own in cluster default, this CI "
            "gate has no implicit base URL -- name the freshly provisioned burst endpoint, e.g. "
            "the local end of a kubectl port-forward to the backend Service, explicitly).",
            file=sys.stderr,
        )
        return 1
    customer_id = customer_id or os.environ.get("ATLAS_PROBE_CUSTOMER_ID", DEFAULT_CUSTOMER_ID)

    with httpx.Client(base_url=base_url, timeout=30.0, transport=transport) as client:
        passed, results = gate(client, customer_id=customer_id)

    for query in PROBE_QUERIES:
        status = "PASS" if results.get(query.probe_class) else "FAIL"
        print(f"{query.probe_class} ({query.registry_id}): {status}")

    if not passed:
        print(
            "SENTINEL GATE: NO GO -- at least one probe class failed against the freshly "
            "provisioned endpoint. Refusing to proceed to any paid step (task matrix:live, "
            "task load:k6, task load:join).",
            file=sys.stderr,
        )
        return 1

    print("SENTINEL GATE: GO -- every probe class passed against the freshly provisioned endpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
