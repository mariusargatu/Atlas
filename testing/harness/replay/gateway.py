"""The model gateway: a LangChain `BaseChatModel` that records and replays (ADR-007).

The agent core has exactly one nondeterministic node, the call to the model. The gateway is the
seam that pins it, implemented as the very chat model the LangGraph graph already calls, so nothing
upstream knows it is being recorded. It has three modes, and they are the seam between the two
machineries the harness keeps apart:

  REPLAY: the regression lane. Cassette only, zero egress, a miss is a hard fail. Deterministic by
           construction: same request, same decision, every run. This is what gates a merge.
  RECORD: capture once. Call the live provider AND persist the cassette, so REPLAY has something
           to serve. Run deliberately, with keys, never in the PR lane.
  LIVE:   the eval lane. Call the live provider and persist NOTHING, because the eval measures
           variance against the real model and a cassette would freeze the thing being measured.

Responsibilities are split on purpose: the cassette *shape* lives in `cassette.py`, *where* it is
stored lives in `cassette_store.py`, and this class only adapts those to LangChain's sync/async
generate protocol. The only logic duplicated between `_generate` and `_agenerate` is the six line
skeleton that LangChain's separate sync/async provider methods force. The policy (replay, persist,
miss) is shared.
"""
from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import model_validator

from replay.cassette import Cassette, build_request, cassette_key
from replay.cassette_store import CassetteMiss, CassetteStore, FileCassetteStore

# Re exported so existing call sites keep `from replay.gateway import CassetteMiss` working.
__all__ = ["CassetteMiss", "GatewayChatModel", "GatewayMode"]


class GatewayMode(str, enum.Enum):
    """How the gateway sources a response, the seam between the regression and eval machineries.

    A `str` enum so a plain `mode="replay"` from a caller still coerces to a member, while the code
    that branches on it gets exhaustive, typo proof identity checks instead of string compares.
    """

    REPLAY = "replay"
    RECORD = "record"
    LIVE = "live"


class GatewayChatModel(BaseChatModel):
    """Record/replay/live around a provider model. REPLAY needs a store and no network. RECORD and
    LIVE need a live `inner` provider, and the wiring is checked once, at construction, not per call.

    A store can be injected directly (`store=`) or, for the common case, named by directory
    (`cassette_dir=`) and built lazily into a `FileCassetteStore`.
    """

    model_id: str
    mode: GatewayMode = GatewayMode.REPLAY
    cassette_dir: Optional[Path] = None
    store: Optional[CassetteStore] = None
    inner: Optional[BaseChatModel] = None

    model_config = {"protected_namespaces": (), "arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _check_wiring(self) -> "GatewayChatModel":
        """Fail fast on a misconfigured gateway: a missing store or provider is a construction
        error, not a surprise on the first turn."""
        if self.mode in (GatewayMode.REPLAY, GatewayMode.RECORD) and self._resolved_store() is None:
            raise ValueError(f"{self.mode.value} mode needs a cassette store (pass `cassette_dir` or `store`)")
        if self.mode in (GatewayMode.RECORD, GatewayMode.LIVE) and self.inner is None:
            raise ValueError(f"{self.mode.value} mode needs a live `inner` provider model")
        return self

    @property
    def _llm_type(self) -> str:
        return "gateway"

    def _resolved_store(self) -> Optional[CassetteStore]:
        """The store to use: an injected one wins, otherwise build a file store from the directory."""
        if self.store is not None:
            return self.store
        if self.cassette_dir is not None:
            return FileCassetteStore(self.cassette_dir)
        return None

    # ---- shared policy (both sync and async funnel through these) ----

    def _request(self, messages: list[BaseMessage], stop: Optional[list[str]], kwargs: dict[str, Any]) -> dict[str, Any]:
        """The digest request. `stop` is folded in for the key WITHOUT mutating `kwargs`, so it is
        never both an explicit `stop=` argument and a `**kwargs` entry on the inner call."""
        merged = {**kwargs, "stop": stop} if stop is not None else kwargs
        return build_request(self.model_id, messages, merged)

    def _replay(self, request: dict[str, Any]) -> ChatResult:
        store = self._resolved_store()
        assert store is not None  # guaranteed by _check_wiring for REPLAY
        key = cassette_key(request)
        cassette = store.load(key)
        if cassette is None:
            raise CassetteMiss(self._miss_message(request, key))
        return cassette.to_chat_result()

    def _persist(self, request: dict[str, Any], result: ChatResult) -> None:
        store = self._resolved_store()
        assert store is not None  # guaranteed by _check_wiring for RECORD
        store.save(Cassette.from_result(result, request, self.model_id))

    def _miss_message(self, request: dict[str, Any], key: str) -> str:
        roles = "→".join(m.get("type", "?") for m in request.get("messages", []))
        return (
            f"cassette miss for model_id={self.model_id!r} key={key} "
            f"(messages: {roles or 'none'}). No recording matches this request — re-record in "
            f"RECORD mode, or check the request drifted from a committed cassette."
        )

    # ---- LangChain sync/async generate protocol ----

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = self._request(messages, stop, kwargs)
        if self.mode is GatewayMode.REPLAY:
            return self._replay(request)
        inner = self.inner
        assert inner is not None  # guaranteed by _check_wiring for RECORD/LIVE
        result = inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if self.mode is GatewayMode.RECORD:
            self._persist(request, result)
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = self._request(messages, stop, kwargs)
        if self.mode is GatewayMode.REPLAY:
            return self._replay(request)
        inner = self.inner
        assert inner is not None  # guaranteed by _check_wiring for RECORD/LIVE
        result = await inner._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if self.mode is GatewayMode.RECORD:
            self._persist(request, result)
        return result
