from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from atlas.config import RerankSettings
from atlas.contracts import Chunk, Hit, Reranker, RerankResult
from atlas.retrieval.ranking import ranked_hits


class PassthroughReranker:
    """Truncates to depth_in then depth_out without reordering or scoring; the baseline
    every reranking backend is judged against."""

    name = "passthrough"

    def __init__(self, depth_in: int, depth_out: int) -> None:
        self.depth_in = depth_in
        self.depth_out = depth_out

    def __call__(self, query: str, chunks: Sequence[Chunk]) -> RerankResult:
        prefix = tuple(chunks)[: self.depth_in]
        hits = tuple(
            Hit(chunk=f.name, score=0.0, rank=rank)
            for rank, f in enumerate(prefix[: self.depth_out], start=1)
        )
        return RerankResult(query=query, hits=hits, input_order=tuple(f.name for f in prefix))


class OnnxReranker:
    """Cross encoder scoring a query against each chunk's text as a sequence pair; the
    model's raw logit is the score, used only for ordering."""

    name = "onnx"

    # The cross encoder's positional embedding table size. A pair longer than this fails
    # at runtime inside the model rather than merely running slow.
    MAX_SEQUENCE_TOKENS = 512

    def __init__(self, model_name: str, depth_in: int, depth_out: int) -> None:
        self.depth_in = depth_in
        self.depth_out = depth_out
        model_path = hf_hub_download(repo_id=model_name, filename="onnx/model_quantized.onnx")
        tokenizer_path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        # "longest_first" truncates whichever side is currently longer; on this corpus
        # that's almost always the question, so one query ends up scored at several
        # different lengths depending on the chunk it's paired with. See docs/the-reranker.md.
        self._tokenizer.enable_truncation(
            max_length=self.MAX_SEQUENCE_TOKENS, strategy="longest_first")
        self._session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def _score(self, query: str, text: str) -> float:
        encoding = self._tokenizer.encode(query, text)
        (logits,) = self._session.run(
            ["logits"],
            {
                "input_ids": np.array([encoding.ids], dtype=np.int64),
                "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
                "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
            },
        )
        return float(logits[0][0])

    def __call__(self, query: str, chunks: Sequence[Chunk]) -> RerankResult:
        """Takes the first depth_in candidates, scores each, returns the best depth_out.
        `input_order` names the whole prefix, evidence the reranker only reordered what
        search had already found."""
        prefix = tuple(chunks)[: self.depth_in]
        scores = [self._score(query, f.text) for f in prefix]
        return RerankResult(
            query=query,
            hits=ranked_hits(zip((f.name for f in prefix), scores, strict=True), self.depth_out),
            input_order=tuple(f.name for f in prefix),
        )


def get_reranker(settings: RerankSettings) -> Reranker:
    if settings.backend == "passthrough":
        return PassthroughReranker(depth_in=settings.depth_in, depth_out=settings.depth_out)
    if settings.backend == "onnx":
        return OnnxReranker(
            model_name=settings.model_name, depth_in=settings.depth_in, depth_out=settings.depth_out
        )
    raise NotImplementedError(f"the {settings.backend!r} reranker backend is not wired in yet")
