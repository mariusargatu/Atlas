"""The real model clients, and the seam that chooses between providers.

Provider choice reads MODEL_PROVIDER. Only generation is a choice: embeddings are always
OpenAI, since Anthropic publishes no embeddings endpoint. See
docs/why-every-run-uses-real-models.md.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, get_args

import numpy as np

from atlas.config import AnswerSettings, EmbeddingSettings, JudgeSettings
from atlas.contracts import ChunkId, Outcome, Usage
from atlas.models.generate import ModelClient
from atlas.models.pricing import cost_usd, require_priced
from atlas.models.schemas import ANSWER_SCHEMA, JUDGE_SCHEMA

# SDKs are imported inside the functions that need them, never at module scope: this
# module is imported for `selected_provider`/`require_priced` alone by entry points
# that construct no client, and importing either SDK costs real time.
if TYPE_CHECKING:
    from anthropic.types import OutputConfigParam
    from openai import OpenAI
    from openai.types.chat import ChatCompletionUserMessageParam

Provider = Literal["openai", "anthropic"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# The endpoint caps a request at 2048 inputs and 300,000 tokens; stays well under both.
_EMBED_BATCH_SIZE = 96

# Both SDKs default to a 600s read timeout. Deliberate, not a setting: a timeout is
# operational and stays out of run_id.
_REQUEST_TIMEOUT_SECONDS = 60.0
_MAX_RETRIES = 3

_OUTCOMES = frozenset(get_args(Outcome))


def _openai_client() -> OpenAI:
    """Every OpenAI client this module builds, transport policy set in one place."""
    from openai import OpenAI

    return OpenAI(timeout=_REQUEST_TIMEOUT_SECONDS, max_retries=_MAX_RETRIES)


def selected_provider() -> Provider:
    raw = os.environ.get("MODEL_PROVIDER", "openai").strip().lower()
    if raw == "anthropic":
        return "anthropic"
    if raw == "openai":
        return "openai"
    raise ValueError(
        f"MODEL_PROVIDER is {raw!r}; expected 'openai' or 'anthropic'. Generation runs on "
        f"either; embeddings are always OpenAI because Anthropic has no embeddings API."
    )


def selected_judge_provider() -> Provider:
    """The judge's own provider switch, defaulting to `selected_provider()`. An env var
    rather than a `JudgeSettings` field so `run_id` stays stable while trying both
    providers on the judge alone."""
    raw = os.environ.get("JUDGE_MODEL_PROVIDER")
    if raw is None:
        return selected_provider()
    raw = raw.strip().lower()
    if raw == "anthropic":
        return "anthropic"
    if raw == "openai":
        return "openai"
    raise ValueError(f"JUDGE_MODEL_PROVIDER is {raw!r}; expected 'openai' or 'anthropic'.")


@dataclass(frozen=True, slots=True)
class ProviderReply:
    """Satisfies generate.StructuredReply."""

    text: str
    cited: tuple[ChunkId, ...]
    outcome: Outcome
    reason: str
    usage: Usage


def _chunk_name(raw: str) -> ChunkId:
    # The prompt lists each passage as `[name] text`; a model copying the label
    # verbatim returns the brackets too.
    return ChunkId(raw.strip().strip("[]").strip())


def _as_outcome(raw: object) -> Outcome:
    # Falls back to refused rather than trusting an unchecked value, since that's the
    # reading that cannot invent an answer.
    return cast(Outcome, raw) if raw in _OUTCOMES else "refused"


def _reply_from_payload(payload: dict[str, Any], usage: Usage) -> ProviderReply:
    """One conversion for both providers, so neither develops its own reading of the
    contract."""
    cited = payload.get("cited") or ()
    return ProviderReply(
        text=str(payload.get("text", "")),
        cited=tuple(_chunk_name(str(name)) for name in cited),
        outcome=_as_outcome(payload.get("outcome")),
        reason=str(payload.get("reason", "")),
        usage=usage,
    )


@dataclass(frozen=True, slots=True)
class JsonReply:
    """A provider's structured reply, before any consumer reads meaning into it.
    `payload` is None when the provider's own safety classifier declined; the answering
    stage turns that into a refusal, while the judge treats it as an infrastructure
    failure rather than a bad-verdict decision."""

    payload: dict[str, Any] | None
    usage: Usage


class StructuredClient(Protocol):
    """One request against one JSON schema. The seam the provider difference lives
    behind."""

    @property
    def model(self) -> str: ...

    def complete_json(self, prompt: str) -> JsonReply: ...


class OpenAIEmbedder:
    """Real embeddings. `dimensions` is requested rather than assumed, so the matrix
    width matches EmbeddingSettings.size by construction instead of by luck."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        name = settings.model_name
        if not name.startswith("text-embedding-"):
            raise ValueError(
                f"vectors.model_name is {name!r}, which is not an OpenAI embeddings "
                f"model; expected something like {DEFAULT_EMBEDDING_MODEL!r}"
            )
        self.model_name = require_priced(name)
        self.model_version = settings.model_version
        self.size = settings.size
        self.normalised = settings.normalise
        self._input_tokens = 0
        self._client = _openai_client()

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.size), dtype=np.float32)
        matrix = np.zeros((len(texts), self.size), dtype=np.float32)
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start:start + _EMBED_BATCH_SIZE]
            matrix[start:start + len(batch)] = self._encode_batch(batch)
        return matrix

    def _encode_batch(self, batch: Sequence[str]) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self.model_name, input=list(batch), dimensions=self.size,
        )
        self._input_tokens += response.usage.prompt_tokens
        # Ordered by the endpoint's own index, not trusted reply order, so a row can
        # never acquire another chunk's vector.
        ordered = sorted(response.data, key=lambda item: item.index)
        matrix = np.array([item.embedding for item in ordered], dtype=np.float32)
        if matrix.shape[1] != self.size:
            raise ValueError(f"got width {matrix.shape[1]}, configured size is {self.size}")
        if len(matrix) != len(batch):
            raise ValueError(f"asked for {len(batch)} vectors, got {len(matrix)}")
        if self.normalised:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix


class OpenAIStructuredClient:
    """One OpenAI request constrained to whichever schema it was built with, so the
    answering stage and the judge share this code instead of each keeping a copy."""

    def __init__(self, model: str, schema: dict[str, Any], name: str, randomness: float) -> None:
        self.model = require_priced(model)
        self._schema = schema
        self._name = name
        self._randomness = randomness
        # False once the temperature retry below fires, meaning this model rejects
        # sampling temperature and every reply since came back at the provider's default.
        # `evals.calibration.noise_floor` relies on this: if the retry fires, its two
        # columns become the same request twice.
        self.randomness_applied: bool | None = None
        self._client = _openai_client()

    def _request(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": self._name, "schema": self._schema, "strict": True},
            },
        }

    def complete_json(self, prompt: str) -> JsonReply:
        from openai import BadRequestError

        messages: list[ChatCompletionUserMessageParam] = [{"role": "user", "content": prompt}]
        try:
            response = self._client.chat.completions.create(
                **self._request(), messages=messages, temperature=self._randomness,
            )
            self.randomness_applied = True
        except BadRequestError as error:
            # Several current models reject the temperature parameter outright.
            if "temperature" not in str(error):
                raise
            self.randomness_applied = False
            response = self._client.chat.completions.create(**self._request(), messages=messages)

        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        return JsonReply(
            payload=json.loads(response.choices[0].message.content or "{}"),
            usage=Usage(tokens_in, tokens_out, cost_usd(self.model, tokens_in, tokens_out)),
        )


class AnthropicStructuredClient:
    """The same request against Claude. Sends no `temperature`: the models this targets
    reject it outright, so the depth knob is `effort` instead. A classifier decline is
    surfaced as an empty payload, never read as content."""

    def __init__(self, model: str, schema: dict[str, Any], effort: Effort = "medium") -> None:
        import anthropic

        if not model.startswith("claude"):
            raise ValueError(
                f"MODEL_PROVIDER=anthropic cannot serve {model!r}. Name a claude model in "
                "answer.model / judge.model, or set MODEL_PROVIDER=openai."
            )
        self.model = require_priced(model)
        self._schema = schema
        self._effort = effort
        self._client = anthropic.Anthropic(
            timeout=_REQUEST_TIMEOUT_SECONDS, max_retries=_MAX_RETRIES,
        )

    def complete_json(self, prompt: str) -> JsonReply:
        output_config: OutputConfigParam = {
            "effort": self._effort,
            "format": {"type": "json_schema", "schema": self._schema},
        }
        response = self._client.messages.create(
            model=self.model, max_tokens=4096, output_config=output_config,
            messages=[{"role": "user", "content": prompt}],
        )
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        usage = Usage(tokens_in, tokens_out, cost_usd(self.model, tokens_in, tokens_out))

        # A classifier decline returns HTTP 200 with content that is empty or partial;
        # check stop_reason before indexing into it.
        if response.stop_reason == "refusal":
            return JsonReply(payload=None, usage=usage)

        text = next((b.text for b in response.content if b.type == "text"), "{}")
        return JsonReply(payload=json.loads(text), usage=usage)


class StructuredAnswerClient:
    """Satisfies `generate.ModelClient` on top of whichever provider answered."""

    def __init__(self, inner: StructuredClient) -> None:
        self._inner = inner

    def complete(self, prompt: str) -> ProviderReply:
        reply = self._inner.complete_json(prompt)
        if reply.payload is None:
            return ProviderReply(
                text="", cited=(), outcome="refused",
                reason="the provider's safety classifier declined this request",
                usage=reply.usage,
            )
        return _reply_from_payload(reply.payload, reply.usage)


def _refuse_unusable_randomness(randomness: float, field: str) -> None:
    """Anthropic's current models reject `temperature`. Refuses rather than silently
    dropping it, since `randomness` feeds `run_id` and a dropped value would file
    requests as different experiments that actually issued identical calls."""
    if randomness != 0.0:
        raise ValueError(
            f"MODEL_PROVIDER=anthropic cannot honour {field}={randomness}: the models this "
            "client targets reject `temperature`, so the value would be dropped and the run "
            "would be filed under a setting that never reached a model. Set it to 0.0, or "
            "set MODEL_PROVIDER=openai for anything that varies it -- which includes "
            "evals.calibration.noise_floor, whose two columns are exactly this difference."
        )


def _structured_client(
    model: str, schema: dict[str, Any], name: str, randomness: float, field: str,
    provider: Provider | None = None,
) -> StructuredClient:
    """The seam, written once: the provider branch and the randomness refusal live here
    rather than in both callers. `provider` overrides `selected_provider()` when given."""
    chosen = provider if provider is not None else selected_provider()
    if chosen == "anthropic":
        _refuse_unusable_randomness(randomness, field)
        return AnthropicStructuredClient(model, schema)
    return OpenAIStructuredClient(model, schema, name, randomness)


def get_answer_client(settings: AnswerSettings) -> ModelClient:
    """Generation runs on whichever provider MODEL_PROVIDER names."""
    return StructuredAnswerClient(
        _structured_client(
            settings.model, ANSWER_SCHEMA, "atlas_answer",
            settings.randomness, "answer.randomness",
        )
    )


def get_judge_client(settings: JudgeSettings, randomness: float = 0.0) -> StructuredClient:
    """The judge's seam, deliberately the same one the answering stage uses.
    `randomness` is an argument rather than a `JudgeSettings` field because
    `evals.calibration.noise_floor` reruns the judge at the configured setting and at
    zero within one process, comparing the two; a setting would split that into two
    different run_ids."""
    return _structured_client(
        settings.model, JUDGE_SCHEMA, "atlas_verdict", randomness, "judge randomness",
        provider=selected_judge_provider(),
    )
