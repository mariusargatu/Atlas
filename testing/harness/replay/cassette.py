"""The cassette: the typed on disk contract for one recorded model call (ADR-007).

The gateway records and replays *cassettes*. This module owns their shape: what a cassette is,
how it is built from a LangChain request/response, and how it round trips to a plain dict for
JSON storage. Keeping the schema here (rather than inline in the gateway and again in the seed
scripts, where it had already drifted) gives the on disk format one source of truth and an
explicit version, so a reader and a writer can never disagree.

The cassette *key* is the allow list request digest from `canonical.py`. The same bytes always
map to the same key, so a recording made today is found by an identical request tomorrow. The
cassette *body* additionally carries the human readable request, which is what makes a replay
miss debuggable instead of just a bare hash.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from determinism.canonical import REQUEST_ALLOW, REQUEST_STRUCTURAL, request_digest

#: Bumped only on a breaking change to the on disk shape. `from_dict` stays tolerant of older bodies.
CASSETTE_VERSION = 1

#: Caller supplied kwargs allowed to shape a cassette (and therefore its key). DERIVED from
#: `canonical.REQUEST_ALLOW`: the structural fields the gateway sets itself are removed, leaving the
#: sampling kwargs forwarded from the model call. Deriving (rather than re listing) makes it
#: impossible to add a key to the digest allow list without `build_request` also copying it, which
#: would otherwise silently drop the field from the key.
_REQUEST_KWARGS: tuple[str, ...] = tuple(k for k in REQUEST_ALLOW if k not in REQUEST_STRUCTURAL)


def normalize_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """A stable, provider agnostic view of the message list, the basis of the cassette key."""
    out: list[dict[str, Any]] = []
    for m in messages:
        entry: dict[str, Any] = {"type": m.type, "content": m.content}
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc["name"], "args": tc["args"], "id": tc.get("id")} for tc in tool_calls
            ]
        if getattr(m, "tool_call_id", None):
            entry["tool_call_id"] = m.tool_call_id
        if getattr(m, "name", None):
            entry["name"] = m.name
        out.append(entry)
    return out


def build_request(
    model_id: str, messages: Sequence[BaseMessage], kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The allow listed request the cassette key is computed from (ADR-007).

    Only `model_id`, the normalized messages, and the allow listed sampling kwargs survive, so two
    requests that differ only in framework plumbing produce the same key.
    """
    kwargs = kwargs or {}
    request: dict[str, Any] = {"model_id": model_id, "messages": normalize_messages(messages)}
    for k in _REQUEST_KWARGS:
        v = kwargs.get(k)
        if v is not None:
            request[k] = v
    return request


def cassette_key(request: dict[str, Any]) -> str:
    """The content addressed key for a request, the cassette filename and the replay lookup key."""
    return request_digest(request)


@dataclass(frozen=True)
class Cassette:
    """One recorded model call: the request that produced it, the response, and a schema version.

    Immutable by construction (frozen). Recordings are facts, not state. `key` is derived from the
    request, so the cassette is content addressed and a body can never be filed under the wrong key.
    """

    model_id: str
    request: dict[str, Any]
    response: dict[str, Any]
    version: int = CASSETTE_VERSION

    @property
    def key(self) -> str:
        return cassette_key(self.request)

    @classmethod
    def from_result(cls, result: ChatResult, request: dict[str, Any], model_id: str) -> Cassette:
        """Capture a live provider result into a cassette (record mode).

        SP9 task 5 (the cost column, backward compatible): a RECORD mode call also persists the
        provider's own `usage_metadata` (langchain_core's `input_tokens`/`output_tokens`/
        `total_tokens` shape) WHEN the provider actually returned one -- `getattr(..., None)` because
        not every `BaseChatModel` even carries the attribute, and a falsy value (`None`, `{}`) is
        never written at all, so a response with no usage story keeps a cassette shape byte identical to before
        this task (no spurious `"usage_metadata": null` key). `dict(...)` copies it: `usage_metadata`
        is typically a `TypedDict`/`Mapping`, and a cassette's own response body must be a plain,
        JSON round trippable `dict`, never a foreign mapping type."""
        message = result.generations[0].message
        response: dict[str, Any] = {
            "content": message.content,
            "tool_calls": list(getattr(message, "tool_calls", []) or []),
        }
        usage = getattr(message, "usage_metadata", None)
        if usage:
            response["usage_metadata"] = dict(usage)
        return cls(model_id=model_id, request=request, response=response)

    def to_chat_result(self) -> ChatResult:
        """Rehydrate the recorded response into the `ChatResult` the graph expects (replay mode).

        `content` and `tool_calls` are always persisted (see `from_result`), which is everything the
        graph routes on, and `AIMessage` defaults the rest, so we do not read fields we never wrote.
        `usage_metadata` (SP9 task 5) is the ONE additive, optional exception: `.get(...)` defaults
        to `None` for every cassette recorded before this task (CRITICAL backward compatibility --
        an old cassette has no `"usage_metadata"` key at all, and `AIMessage(usage_metadata=None)` is
        byte identical to never passing the kwarg, confirmed against langchain_core directly), so an
        old cassette still replays green, cost simply unavailable rather than a KeyError."""
        message = AIMessage(
            content=self.response.get("content", ""),
            tool_calls=self.response.get("tool_calls", []),
            usage_metadata=self.response.get("usage_metadata"),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def to_dict(self) -> dict[str, Any]:
        """The JSON serializable body written to disk."""
        return {
            "version": self.version,
            "model_id": self.model_id,
            "request": self.request,
            "response": self.response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cassette:
        """Parse a stored body. `model_id` is cosmetic on replay (the key comes from the request),
        so a body that omits it still parses."""
        return cls(
            model_id=data.get("model_id", ""),
            request=data.get("request", {}),
            response=data["response"],
            version=data.get("version", CASSETTE_VERSION),
        )


__all__ = ["CASSETTE_VERSION", "Cassette", "build_request", "cassette_key", "normalize_messages"]
