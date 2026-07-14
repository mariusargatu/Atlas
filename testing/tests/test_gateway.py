"""Gateway record/replay/live tests, the determinism keystone, exercised hermetically.

Replay needs no network and no provider SDK: a seeded cassette stands in for the recorded model
response. Record and live are proven against a tiny stub provider, so the capture path is covered
in CI too (it used to be reachable only through the Ollama make targets). Proves replay returns the
recorded answer, a miss hard fails, replay is byte stable, record persists, live never persists,
binding tools shifts the key, and the wiring is validated at construction.
"""
from __future__ import annotations

from typing import Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from replay.cassette import Cassette, build_request, cassette_key
from replay.cassette_store import CassetteMiss, InMemoryCassetteStore
from replay.gateway import GatewayChatModel, GatewayMode


def _seed(store, model_id, messages, response, **req_extra):
    """Seed a cassette through the public schema + store, the same path the gateway reads."""
    request = build_request(model_id, messages, req_extra)
    store.save(Cassette(model_id=model_id, request=request, response=response))
    return cassette_key(request)


class _StubProvider(BaseChatModel):
    """A deterministic stand in for a live provider, so record/live paths run with no network.

    `usage_metadata` (SP9 task 5): optional, mirrors a real provider's own `AIMessage.usage_metadata`
    (langchain_core's typed `input_tokens`/`output_tokens`/`total_tokens` shape) so a RECORD mode
    test can exercise the cost column's real capture path with no network at all -- `None` (the
    default) keeps every EXISTING test in this file byte identical, a stub with no usage story."""

    reply: str = "stub reply"
    usage_metadata: Optional[dict] = None

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = AIMessage(content=self.reply, usage_metadata=self.usage_metadata)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.asyncio
async def test_replay_returns_recorded_response():
    store = InMemoryCassetteStore()
    gw = GatewayChatModel(model_id="claude-test", store=store, mode="replay")
    messages = [HumanMessage("Is my plan contract-free?")]
    _seed(store, "claude-test", messages, {"content": "No contract.", "tool_calls": []})
    result = await gw._agenerate(messages)
    assert result.generations[0].message.content == "No contract."


@pytest.mark.asyncio
async def test_cassette_miss_hard_fails():
    gw = GatewayChatModel(model_id="claude-test", store=InMemoryCassetteStore(), mode="replay")
    with pytest.raises(CassetteMiss):
        await gw._agenerate([HumanMessage("unrecorded question")])


@pytest.mark.asyncio
async def test_replay_is_byte_stable():
    store = InMemoryCassetteStore()
    gw = GatewayChatModel(model_id="claude-test", store=store, mode="replay")
    messages = [HumanMessage("hello")]
    _seed(store, "claude-test", messages, {"content": "hi", "tool_calls": []})
    r1 = await gw._agenerate(messages)
    r2 = await gw._agenerate(messages)
    assert r1.generations[0].message.content == r2.generations[0].message.content == "hi"


@pytest.mark.asyncio
async def test_replay_returns_recorded_tool_calls():
    store = InMemoryCassetteStore()
    gw = GatewayChatModel(model_id="claude-test", store=store, mode="replay")
    messages = [HumanMessage("what's my bill?")]
    _seed(
        store, "claude-test", messages,
        {"content": "", "tool_calls": [{"name": "get_bill", "args": {}, "id": "call-1"}]},
    )
    result = await gw._agenerate(messages)
    calls = result.generations[0].message.tool_calls
    assert calls and calls[0]["name"] == "get_bill"


@pytest.mark.asyncio
async def test_record_persists_then_replays():
    """Record mode calls the provider and stores a cassette. The same key then replays with no provider."""
    store = InMemoryCassetteStore()
    messages = [HumanMessage("what is a data cap?")]
    rec = GatewayChatModel(model_id="stub", store=store, mode="record", inner=_StubProvider(reply="A data cap is a limit."))
    out = await rec._agenerate(messages)
    assert out.generations[0].message.content == "A data cap is a limit."

    rep = GatewayChatModel(model_id="stub", store=store, mode="replay")  # no inner, no network
    replayed = await rep._agenerate(messages)
    assert replayed.generations[0].message.content == "A data cap is a limit."


@pytest.mark.asyncio
async def test_live_does_not_persist():
    """The eval lane: live calls the provider but writes no cassette (variance is the measurement)."""
    store = InMemoryCassetteStore()
    messages = [HumanMessage("anything")]
    gw = GatewayChatModel(model_id="stub", store=store, mode="live", inner=_StubProvider(reply="live"))
    out = await gw._agenerate(messages)
    assert out.generations[0].message.content == "live"
    assert store.load(cassette_key(build_request("stub", messages))) is None  # nothing recorded


def test_replay_without_a_store_is_a_construction_error():
    with pytest.raises(ValueError):
        GatewayChatModel(model_id="m", mode="replay")  # no store, no cassette_dir


def test_record_without_a_provider_is_a_construction_error():
    with pytest.raises(ValueError):
        GatewayChatModel(model_id="m", store=InMemoryCassetteStore(), mode="record")  # no inner


def test_string_mode_coerces_to_enum():
    gw = GatewayChatModel(model_id="m", store=InMemoryCassetteStore(), mode="replay")
    assert gw.mode is GatewayMode.REPLAY


def test_binding_tools_shifts_the_cassette_key():
    messages = [HumanMessage("hi")]
    without = cassette_key(build_request("m", messages, {}))
    with_tools = cassette_key(build_request("m", messages, {"tools": [{"name": "search_knowledge"}]}))
    assert without != with_tools


@pytest.mark.asyncio
async def test_stop_sequence_records_without_crashing_and_shapes_the_key():
    """Regression: `stop` must reach the provider exactly once (not as both stop= and **kwargs),
    and it must still shape the cassette key so record/replay stay symmetric."""
    store = InMemoryCassetteStore()
    messages = [HumanMessage("hi")]
    rec = GatewayChatModel(model_id="stub", store=store, mode="record", inner=_StubProvider(reply="ok"))
    out = await rec._agenerate(messages, stop=["STOP"])  # used to raise TypeError (double passed stop)
    assert out.generations[0].message.content == "ok"
    assert store.load(cassette_key(build_request("stub", messages, {"stop": ["STOP"]}))) is not None
    assert store.load(cassette_key(build_request("stub", messages))) is None  # stop less is a different key


def test_stop_sequence_sync_record_does_not_crash():
    store = InMemoryCassetteStore()
    rec = GatewayChatModel(model_id="stub", store=store, mode="record", inner=_StubProvider(reply="ok"))
    out = rec._generate([HumanMessage("hi")], stop=["STOP"])
    assert out.generations[0].message.content == "ok"


def test_request_kwargs_are_a_subset_of_the_digest_allow_list():
    """The two halves of the key contract cannot drift: every field build_request copies must be one
    the digest actually hashes, else the field would silently fall out of the key."""
    from determinism.canonical import REQUEST_ALLOW
    from replay.cassette import _REQUEST_KWARGS

    assert set(_REQUEST_KWARGS) <= set(REQUEST_ALLOW)


# ---- SP9 task 5: backward compatible usage_metadata capture (the cost column) --------------------


@pytest.mark.asyncio
async def test_record_persists_usage_metadata_when_the_provider_returns_it():
    """A RECORD mode call whose live provider returns real token counts persists them onto the
    cassette -- the concrete unlock ADR-029's cost trio amendment names."""
    store = InMemoryCassetteStore()
    messages = [HumanMessage("what is a data cap?")]
    usage = {"input_tokens": 42, "output_tokens": 17, "total_tokens": 59}
    rec = GatewayChatModel(
        model_id="stub", store=store, mode="record",
        inner=_StubProvider(reply="A data cap is a limit.", usage_metadata=usage),
    )
    await rec._agenerate(messages)
    cassette = store.load(cassette_key(build_request("stub", messages)))
    assert cassette is not None
    assert cassette.response["usage_metadata"] == usage


@pytest.mark.asyncio
async def test_record_then_replay_roundtrip_carries_usage_metadata():
    """The full backward compatible roundtrip: RECORD captures usage_metadata, REPLAY rehydrates
    the SAME AIMessage.usage_metadata with no provider and no network."""
    store = InMemoryCassetteStore()
    messages = [HumanMessage("what is a data cap?")]
    usage = {"input_tokens": 42, "output_tokens": 17, "total_tokens": 59}
    rec = GatewayChatModel(
        model_id="stub", store=store, mode="record",
        inner=_StubProvider(reply="A data cap is a limit.", usage_metadata=usage),
    )
    await rec._agenerate(messages)

    rep = GatewayChatModel(model_id="stub", store=store, mode="replay")  # no inner, no network
    replayed = await rep._agenerate(messages)
    message = replayed.generations[0].message
    assert message.content == "A data cap is a limit."
    assert message.usage_metadata == usage


@pytest.mark.asyncio
async def test_record_without_usage_metadata_never_persists_the_key_at_all():
    """A provider that returns NO usage_metadata (langchain_core's own `None` default) must not
    grow a spurious `usage_metadata` key on the cassette -- the cassette shape for a response with
    no usage story stays exactly what it was before this task."""
    store = InMemoryCassetteStore()
    messages = [HumanMessage("hi")]
    rec = GatewayChatModel(model_id="stub", store=store, mode="record", inner=_StubProvider(reply="hi back"))
    await rec._agenerate(messages)
    cassette = store.load(cassette_key(build_request("stub", messages)))
    assert cassette is not None
    assert "usage_metadata" not in cassette.response


@pytest.mark.asyncio
async def test_an_old_cassette_with_no_usage_metadata_still_replays_green():
    """CRITICAL backward compatibility: a cassette recorded BEFORE this task (no `usage_metadata`
    key in its response dict at all) still replays cleanly; the rehydrated message's own
    `usage_metadata` is `None` (cost reported as unavailable), never a KeyError or a hard fail."""
    store = InMemoryCassetteStore()
    messages = [HumanMessage("is my plan contract-free?")]
    key = _seed(store, "claude-test", messages, {"content": "No contract.", "tool_calls": []})  # no usage_metadata key
    gw = GatewayChatModel(model_id="claude-test", store=store, mode="replay")
    result = await gw._agenerate(messages)
    message = result.generations[0].message
    assert message.content == "No contract."
    assert message.usage_metadata is None
    assert store.load(key).response.get("usage_metadata") is None


def test_cost_from_usage_reports_unavailable_for_the_old_cassette_read_path():
    """The cost column's own honesty contract (matrix.spend_gate.cost_from_usage), exercised
    directly against the exact shape an old cassette's rehydrated message carries."""
    from matrix.spend_gate import cost_from_usage

    assert cost_from_usage("anthropic", None) is None
