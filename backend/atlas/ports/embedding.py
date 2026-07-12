"""The embedding client port. Pure, no client, no framework.

`EmbeddingClient` is the narrow seam D19 names as one of the five record/replay ports this system
should eventually have (`testing.harness.rag_tools.ingest.embed_texts` is, until SP9 task 3, one
hardcoded direct `httpx` call against a TEI base URL, with no `Protocol` behind it at all, unlike
`Retriever`/`KnowledgeGraph`/`Reranker`). SP9 task 3 is the first sub project that actually needs a
second embedder (OpenAI `text-embedding-3-small`) to exist, so this port exists to let the matrix
runner call either embedder through the same shape, never a second bespoke function per provider.

DECISION (SP9 task 3, documented rather than assumed): this port carries NO record/replay mode.
Building the full D19 seam now (REPLAY/RECORD/LIVE modes mirroring `replay.gateway.GatewayChatModel`,
a cassette format for embedding calls) is more machinery than one new embedder axis strictly needs,
and the property record/replay exists to buy -- "an unchanged call is paid once" -- already holds at
a coarser, cheaper grain: `EmbeddingFingerprint`/`index_build_id` (`rag_tools.fingerprint`) make an
index build content addressed, so a rerun against an unchanged corpus/model/params never embeds
again at all, it reuses the cached build. Embedding calls are also individually cheap (pennies per batch,
unlike a generation call), so the spend gate's "paid once" property matters far less for this axis
than for the generator axis it protects in SP9 task 5. The full D19 embedding record/replay seam
(a real cassette per embedding call, mirroring the gateway) stays SP12's territory, not this one's.
"""
from __future__ import annotations

from typing import Protocol


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, one vector per text, returned in the SAME order as `texts` (no
        adapter may reorder or drop an item; a batch failure fails the whole call loud instead)."""
        ...
