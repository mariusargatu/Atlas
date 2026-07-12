"""`OpenAiEmbeddingClient`: the `EmbeddingClient` port's OpenAI adapter (the second embedder axis,
SP9 task 3: `text-embedding-3-small`, pinned in `models.lock`'s API model-id shape).

The `openai` package lives ONLY in the `record` dependency group (`pyproject.toml`); the hermetic PR
lane installs neither it nor any provider SDK, so this module imports `openai` NOWHERE at module
scope, only lazily inside `_live_client`, which runs exclusively when no `client` was injected. Every
hermetic test injects a stub client (matching the shape below), so `_live_client` is never reached in
the PR lane; `test_embedding_port.py`'s own
`test_openai_embedding_client_never_imports_openai_at_module_scope` guards this directly.

NO record/replay mode (see `atlas.ports.embedding`'s module docstring for the decision).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class OpenAiEmbeddingClient:
    """`EmbeddingClient` over the OpenAI SDK. `client` is any object exposing
    `embeddings.create(model=..., input=[...]) -> object with .data[i].embedding`, the real
    `openai.OpenAI` client's own shape; hermetic tests inject a minimal stub built to that same
    shape, never a real client, so no network call and no API key are ever needed to test this."""

    def __init__(self, model_id: str, client: Any | None = None) -> None:
        self._model_id = model_id
        self._client = client

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._client if self._client is not None else self._live_client()
        response = client.embeddings.create(model=self._model_id, input=list(texts))
        return [item.embedding for item in response.data]

    def _live_client(self) -> Any:  # pragma: no cover - live only, needs OPENAI_API_KEY + network
        """Constructed only when no `client` was injected (never reached by a hermetic test): reads
        `OPENAI_API_KEY` from the environment, the same way every other live provider path in this
        repo does (`replay.providers.build_chat_model`'s own lazy per provider import)."""
        import openai

        return openai.OpenAI()


__all__ = ["OpenAiEmbeddingClient"]
