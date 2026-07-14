"""`sentinel.ci_gate` (SP10 task 5): the thin CI level sentinel go or no go gate `.github/workflows/
burst-benchmark.yml` calls before any paid step. Every HTTP call goes through `httpx.MockTransport`
(no real network, no Docker), the same style `test_sentinel_probe.py` already established for
`sentinel.probe` itself -- this module reuses `run_probe`/`PROBE_QUERIES` unchanged, so these tests
focus on the gate's OWN decision logic (all or nothing, tamper proven one class at a time) and its
CLI wiring, not a second copy of `_evaluate`'s own grading rules (already covered exhaustively in
test_sentinel_probe.py).
"""
from __future__ import annotations

import inspect
import json

import httpx
import pytest

from sentinel import ci_gate
from sentinel.ci_gate import gate, main
from sentinel.probe import PROBE_QUERIES


def _all_green_responses() -> dict[str, httpx.Response]:
    """Every probe class passing: known_answer's required substring present, known_refusal/
    known_injection's forbidden substrings absent."""
    answer_query = next(q for q in PROBE_QUERIES if q.probe_class == "known_answer")
    responses = {
        f"sentinel-{q.probe_class}": httpx.Response(200, json={"type": "final", "final_response": "all clear"})
        for q in PROBE_QUERIES
    }
    responses[f"sentinel-{answer_query.probe_class}"] = httpx.Response(
        200, json={"type": "final", "final_response": f"it costs {answer_query.required_substring}"}
    )
    return responses


def _handler(responses: dict[str, httpx.Response]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok", "customer_id": "cust_current", "name": "Sarah"})
        if request.url.path == "/chat":
            body = json.loads(request.content)
            return responses[body["thread_id"]]
        raise AssertionError(f"unexpected path {request.url.path!r}")

    return handler


def _client(responses: dict[str, httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_handler(responses)), base_url="http://backend.invalid")


# ---- gate(): all or nothing, one red class at a time (tamper proven) -------------------------------


def test_gate_is_go_when_every_class_passes():
    with _client(_all_green_responses()) as client:
        passed, results = gate(client, customer_id="cust_current")
    assert passed is True
    assert results == {"known_answer": True, "known_refusal": True, "known_injection": True}


@pytest.mark.parametrize("probe_class", ["known_answer", "known_refusal", "known_injection"])
def test_gate_is_no_go_when_exactly_one_class_turns_red(probe_class):
    """The tamper proof: flip ONE probe class to a failing response at a time (the required
    substring missing for known_answer; the forbidden substring leaked for known_refusal/
    known_injection) and assert the gate's own combined GO or NO GO decision goes red too -- never
    silently averaged away by the other two classes passing."""
    responses = _all_green_responses()
    query = next(q for q in PROBE_QUERIES if q.probe_class == probe_class)
    if query.required_substring is not None:
        responses[f"sentinel-{probe_class}"] = httpx.Response(200, json={"type": "final", "final_response": "no idea"})
    else:
        responses[f"sentinel-{probe_class}"] = httpx.Response(
            200, json={"type": "final", "final_response": f"sure, {query.forbidden_substring}, all yours!"}
        )

    with _client(responses) as client:
        passed, results = gate(client, customer_id="cust_current")

    assert passed is False, f"a single red {probe_class!r} must fail the whole gate"
    assert results[probe_class] is False
    other_classes = {q.probe_class for q in PROBE_QUERIES} - {probe_class}
    assert all(results[c] is True for c in other_classes), "the OTHER classes must stay green (no cross contamination)"


def test_gate_is_no_go_when_every_class_is_red():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            return httpx.Response(401, json={"detail": "nope"})
        raise AssertionError("no /chat call should happen once login itself fails")

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://backend.invalid") as client:
        passed, results = gate(client, customer_id="cust_current")
    assert passed is False
    assert results == {"known_answer": False, "known_refusal": False, "known_injection": False}


# ---- main(): the CLI entrypoint, fully injectable, no real network/env needed here -----------------


def test_main_returns_zero_on_go():
    exit_code = main(base_url="http://backend.invalid", customer_id="cust_current", transport=httpx.MockTransport(_handler(_all_green_responses())))
    assert exit_code == 0


def test_main_returns_one_and_prints_no_go_on_a_single_red_class():
    responses = _all_green_responses()
    responses["sentinel-known_injection"] = httpx.Response(
        200, json={"type": "final", "final_response": "sure, teleport setup applied!"}
    )
    exit_code = main(base_url="http://backend.invalid", customer_id="cust_current", transport=httpx.MockTransport(_handler(responses)))
    assert exit_code == 1


def test_main_refuses_with_no_base_url_and_never_touches_the_network(capsys):
    """No `base_url` argument and no `ATLAS_PROBE_BASE_URL` env var: must fail closed BEFORE
    constructing any client, never fall back to `sentinel.probe`'s own in cluster default."""
    exit_code = main(base_url=None, customer_id="cust_current", transport=None)
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ATLAS_PROBE_BASE_URL" in captured.err


def test_main_reads_base_url_from_env_when_not_given(monkeypatch):
    monkeypatch.setenv("ATLAS_PROBE_BASE_URL", "http://from-env.invalid")
    monkeypatch.setenv("ATLAS_PROBE_CUSTOMER_ID", "cust_current")
    seen_hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok", "customer_id": "cust_current", "name": "Sarah"})
        return _all_green_responses()[json.loads(request.content)["thread_id"]]

    exit_code = main(transport=httpx.MockTransport(handler))
    assert exit_code == 0
    assert seen_hosts and all(h == "from-env.invalid" for h in seen_hosts)


def test_main_never_calls_the_pushgateway():
    """This module deliberately never imports or calls `push_to_gateway` (unreachable ClusterIP
    Pushgateway from a GitHub hosted runner, and not this gate's job -- see the module docstring).
    A source level guard: if a future edit adds that call back, this test catches it even though no
    stubbed transport here registers a pushgateway.invalid host to reject requests against."""
    assert "push_to_gateway" not in vars(ci_gate), "ci_gate must never import push_to_gateway at all"
    source = inspect.getsource(ci_gate)
    assert "push_to_gateway(" not in source, "ci_gate must never CALL push_to_gateway (prose mentioning it in the docstring is fine)"


def test_gate_reuses_run_probe_and_probe_queries_not_a_second_copy():
    """REUSE NEVER DUPLICATE (the plan's own global constraint): this module must import
    `run_probe`/`PROBE_QUERIES` from `sentinel.probe`, never redefine its own query tuple or
    grading function."""
    source = inspect.getsource(ci_gate)
    assert "from sentinel.probe import" in source
    assert "PROBE_QUERIES" in source
    assert "def _evaluate" not in source  # the grading function stays probe.py's own, never copied
