"""`sentinel.probe` (SP6 task 5): the CronJob script that drives the DEPLOYED service over the
network (`/auth/login` then `/chat`, a real authenticated session, never an in process graph
import -- SP6 planning digest design question 7) with three query classes pinned by their
`corpus/registry/core.yaml` entity ids, so a corpus bump that renames or removes one fails this
hermetic suite loudly rather than the CronJob silently probing a stale question in production.

Every HTTP call in these tests goes through `httpx.MockTransport` (built into httpx, no new test
dependency), the same stubbed HTTP layer style `test_pgvector_adapter.py` already established for
TEI doubles. No real network, no Docker.
"""
from __future__ import annotations

import json

import httpx
from corpus_tools.registry import load_registry

from sentinel.probe import (
    PROBE_QUERIES,
    ProbeQuery,
    _evaluate,
    main,
    render_pushgateway_payload,
    run_probe,
)
from .fixtures.corpus_expectations import CORE


# ---- the pin itself: a corpus bump that removes a probed entity fails loudly -----------------------


def test_every_pinned_registry_id_still_exists_in_the_fact_registry():
    registry = load_registry([CORE])
    existing_ids = {e.id for e in registry.entities}
    for query in PROBE_QUERIES:
        assert query.registry_id in existing_ids, (
            f"{query.probe_class!r} pins registry id {query.registry_id!r}, which no longer exists "
            f"in {CORE} -- update the probe's pinned id (and question) to match the corpus bump."
        )


def test_three_distinct_query_classes_are_pinned():
    assert {q.probe_class for q in PROBE_QUERIES} == {"known_answer", "known_refusal", "known_injection"}
    assert len(PROBE_QUERIES) == 3


def test_the_never_rendered_pool_entries_back_refusal_and_injection_not_answer():
    """`plan-quantum-5g`/`fee-teleport-setup` are the registry's own `render: false` "never rendered
    pool" (corpus/registry/core.yaml's own comment: "must never surface in a customer facing
    document"); the known_answer class must pin a real, rendered, customer facing fact instead."""
    by_class = {q.probe_class: q for q in PROBE_QUERIES}
    registry = load_registry([CORE])
    assert registry.entity(by_class["known_answer"].registry_id).render is True
    assert registry.entity(by_class["known_refusal"].registry_id).render is False
    assert registry.entity(by_class["known_injection"].registry_id).render is False


# ---- _evaluate: the pure grading function, both the 200 and the documented 503 shapes --------------


def _query(**overrides) -> ProbeQuery:
    base = dict(
        probe_class="known_answer", registry_id="plan-fiber-100", question="q",
        required_substring=None, forbidden_substring=None,
    )
    base.update(overrides)
    return ProbeQuery(**base)


def test_evaluate_passes_on_200_when_the_required_substring_is_present():
    q = _query(required_substring="29.99")
    assert _evaluate(200, {"final_response": "Fiber 100 is $29.99 a month."}, q) is True


def test_evaluate_fails_on_200_when_the_required_substring_is_absent():
    q = _query(required_substring="29.99")
    assert _evaluate(200, {"final_response": "I'm not sure."}, q) is False


def test_evaluate_passes_on_200_when_the_forbidden_substring_is_absent():
    q = _query(required_substring=None, forbidden_substring="79.99")
    assert _evaluate(200, {"final_response": "That plan is not available yet."}, q) is True


def test_evaluate_fails_on_200_when_the_forbidden_substring_leaks_case_insensitively():
    q = _query(required_substring=None, forbidden_substring="79.99")
    assert _evaluate(200, {"final_response": "Sure, that's $79.99/month, sign up now!"}, q) is False


def test_evaluate_treats_a_documented_cassette_miss_as_the_service_alive_not_a_failure():
    """The deployed reference system defaults to replay mode (environments/base/values.yaml's own
    `backend.mode: replay`); a fresh question with no committed cassette 503s by DESIGN
    (server.py's own `_cassette_miss` handler), the same "expected, proves the wiring" reasoning
    infra/scripts/k3d_smoke.py's own chat endpoint half already documents. A probe MUST NOT page on
    this, or a replay mode demo deployment would never stop paging."""
    q = _query()
    assert _evaluate(503, {"error": "replay cassette miss: no cassette for this request"}, q) is True


def test_evaluate_fails_on_an_undocumented_503():
    """A 503 that does NOT carry the cassette miss shape is a real outage signal, not the documented
    replay boundary -- must not be swallowed as a false pass."""
    q = _query()
    assert _evaluate(503, {"error": "upstream timeout"}, q) is False
    assert _evaluate(503, None, q) is False


def test_evaluate_fails_on_any_other_status_code():
    q = _query()
    assert _evaluate(500, {"final_response": "..."}, q) is False
    assert _evaluate(401, None, q) is False


# ---- run_probe: one login, three /chat calls, over a stubbed transport -----------------------------


def _handler(*, login_status=200, chat_by_thread: dict[str, httpx.Response] | None = None):
    chat_by_thread = chat_by_thread or {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            if login_status != 200:
                return httpx.Response(login_status, json={"detail": "nope"})
            return httpx.Response(200, json={"access_token": "tok", "customer_id": "cust_current", "name": "Sarah"})
        if request.url.path == "/chat":
            body = json.loads(request.content)
            thread_id = body["thread_id"]
            assert request.headers["authorization"] == "Bearer tok"
            return chat_by_thread[thread_id]
        raise AssertionError(f"unexpected path {request.url.path!r}")

    return handler


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://backend.invalid")


def test_run_probe_logs_in_once_and_grades_each_class_independently():
    responses = {
        f"sentinel-{q.probe_class}": httpx.Response(200, json={"type": "final", "final_response": "x"})
        for q in PROBE_QUERIES
    }
    # known_answer needs its required substring to actually be present to pass
    answer_query = next(q for q in PROBE_QUERIES if q.probe_class == "known_answer")
    responses[f"sentinel-{answer_query.probe_class}"] = httpx.Response(
        200, json={"type": "final", "final_response": f"the price is {answer_query.required_substring}"}
    )
    with _client(_handler(chat_by_thread=responses)) as client:
        results = run_probe(client, customer_id="cust_current")
    assert results == {"known_answer": True, "known_refusal": True, "known_injection": True}


def test_run_probe_fails_every_class_when_login_itself_fails():
    with _client(_handler(login_status=401)) as client:
        results = run_probe(client, customer_id="cust_current")
    assert results == {q.probe_class: False for q in PROBE_QUERIES}
    assert set(results) == {"known_answer", "known_refusal", "known_injection"}


def test_run_probe_reports_a_leaked_forbidden_fact_as_a_failure():
    refusal_query = next(q for q in PROBE_QUERIES if q.probe_class == "known_refusal")
    responses = {
        f"sentinel-{q.probe_class}": httpx.Response(200, json={"type": "final", "final_response": "fine"})
        for q in PROBE_QUERIES
    }
    responses[f"sentinel-{refusal_query.probe_class}"] = httpx.Response(
        200,
        json={"type": "final", "final_response": f"sure, it's {refusal_query.forbidden_substring}/mo, enrolled!"},
    )
    with _client(_handler(chat_by_thread=responses)) as client:
        results = run_probe(client, customer_id="cust_current")
    assert results["known_refusal"] is False


# ---- render_pushgateway_payload + push_to_gateway ---------------------------------------------------


def test_render_pushgateway_payload_carries_one_gauge_line_per_class():
    payload = render_pushgateway_payload({"known_answer": True, "known_refusal": False, "known_injection": True})
    assert '# TYPE atlas_probe_success gauge' in payload
    assert 'atlas_probe_success{class="known_answer"} 1' in payload
    assert 'atlas_probe_success{class="known_refusal"} 0' in payload
    assert 'atlas_probe_success{class="known_injection"} 1' in payload


# ---- main(): the CLI entrypoint, fully injectable so no real network/env is ever needed here --------


def test_main_returns_zero_and_pushes_when_every_class_passes():
    answer_query = next(q for q in PROBE_QUERIES if q.probe_class == "known_answer")
    responses = {
        f"sentinel-{q.probe_class}": httpx.Response(200, json={"type": "final", "final_response": "ok"})
        for q in PROBE_QUERIES
    }
    responses[f"sentinel-{answer_query.probe_class}"] = httpx.Response(
        200, json={"type": "final", "final_response": f"it's {answer_query.required_substring}"}
    )
    pushed: dict[str, httpx.Request] = {}

    def app_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "backend.invalid":
            if request.url.path == "/auth/login":
                return httpx.Response(200, json={"access_token": "tok", "customer_id": "cust_current", "name": "s"})
            body = json.loads(request.content)
            return responses[body["thread_id"]]
        if request.url.host == "pushgateway.invalid":
            pushed["request"] = request
            return httpx.Response(200)
        raise AssertionError(f"unexpected host {request.url.host!r}")

    exit_code = main(
        base_url="http://backend.invalid", pushgateway_url="http://pushgateway.invalid",
        customer_id="cust_current", transport=httpx.MockTransport(app_handler),
    )
    assert exit_code == 0
    assert "request" in pushed
    assert pushed["request"].method == "PUT"
    assert "/metrics/job/atlas_sentinel_probe" in str(pushed["request"].url)
    assert b'atlas_probe_success{class="known_answer"} 1' in pushed["request"].content


def test_main_returns_one_when_any_class_fails():
    def app_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok", "customer_id": "c", "name": "n"})
        if request.url.path == "/chat":
            return httpx.Response(503, json={"error": "upstream unavailable"})
        return httpx.Response(200)

    exit_code = main(
        base_url="http://backend.invalid", pushgateway_url="http://pushgateway.invalid",
        customer_id="cust_current", transport=httpx.MockTransport(app_handler),
    )
    assert exit_code == 1


def test_main_reads_defaults_from_env_when_not_given(monkeypatch):
    monkeypatch.setenv("ATLAS_PROBE_BASE_URL", "http://from-env.invalid")
    monkeypatch.setenv("ATLAS_PROBE_PUSHGATEWAY_URL", "http://from-env-pg.invalid")
    monkeypatch.setenv("ATLAS_PROBE_CUSTOMER_ID", "cust_neighbor")
    seen_hosts = []

    def app_handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.path == "/auth/login":
            body = json.loads(request.content)
            assert body["customer_id"] == "cust_neighbor"
            return httpx.Response(200, json={"access_token": "tok", "customer_id": "cust_neighbor", "name": "n"})
        if request.url.path == "/chat":
            return httpx.Response(200, json={"type": "final", "final_response": "ok"})
        return httpx.Response(200)

    main(transport=httpx.MockTransport(app_handler))
    assert "from-env.invalid" in seen_hosts
    assert "from-env-pg.invalid" in seen_hosts
