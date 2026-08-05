"""Cutting a document into chunks.

Three strategies, and two of them are the same algorithm at different granularity: split
the text into leaf units, subdivide any leaf that is over the token limit on its own,
then pack whole leaves up to the limit. The ladder of subdivision is paragraph ->
sentence -> token window, and each rung is the fallback of the one above it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import tiktoken

from atlas.config import ChunkSettings
from atlas.contracts import Chunk, Chunker, ChunkId, Document, Range

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")

# Character spans, relative to the text handed in. Every splitter below has this shape,
# which is what lets one be the fallback of another.
Span = tuple[int, int]
Splitter = Callable[[str, tiktoken.Encoding, int], list[Span]]


def _char_bounds(
    text: str, encoding: tiktoken.Encoding, tokens: list[int], tok_start: int, tok_end: int
) -> Span:
    """The character span a token window covers, decoding the prefix before each edge.
    Both ends are special-cased: the final window must reach the end of the text exactly,
    which round-tripping through the encoder does not guarantee for every input."""
    start = len(encoding.decode(tokens[:tok_start])) if tok_start > 0 else 0
    end = len(text) if tok_end >= len(tokens) else len(encoding.decode(tokens[:tok_end]))
    return start, end


def _spans_at_boundary(text: str, boundary: re.Pattern[str]) -> list[Span]:
    """Contiguous, non-overlapping spans cut at a boundary pattern; concatenating every
    span reconstructs the text exactly."""
    spans: list[Span] = []
    start = 0
    for match in boundary.finditer(text):
        spans.append((start, match.end()))
        start = match.end()
    if start < len(text) or not spans:
        spans.append((start, len(text)))
    return spans


def _token_spans(text: str, encoding: tiktoken.Encoding, max_tokens: int) -> list[Span]:
    """Bottom of the ladder: fixed token windows, so nothing is emitted oversized."""
    tokens = encoding.encode(text)
    if not tokens:
        return [(0, len(text))]
    return [
        _char_bounds(text, encoding, tokens, start, min(start + max_tokens, len(tokens)))
        for start in range(0, len(tokens), max_tokens)
    ]


def _split_at(boundary: re.Pattern[str], finer: Splitter) -> Splitter:
    """One rung of the ladder: cut at `boundary`, hand any unit still over the limit to
    `finer`. Returned spans are absolute."""

    def split(text: str, encoding: tiktoken.Encoding, max_tokens: int) -> list[Span]:
        leaves: list[Span] = []
        for start, end in _spans_at_boundary(text, boundary):
            unit = text[start:end]
            if len(encoding.encode(unit)) > max_tokens:
                leaves.extend((start + s, start + e) for s, e in finer(unit, encoding, max_tokens))
            else:
                leaves.append((start, end))
        return leaves

    return split


_sentence_spans = _split_at(_SENTENCE_BOUNDARY, _token_spans)
_paragraph_spans = _split_at(_PARAGRAPH_BOUNDARY, _sentence_spans)


class FixedChunker:
    """Fixed token windows over the whole document, advancing by max_tokens less the
    configured overlap. The only strategy that overlaps, and the only one that will cut a
    sentence in half."""

    name = "fixed"

    def __init__(self, settings: ChunkSettings) -> None:
        self._settings = settings

    def __call__(self, doc: Document) -> tuple[Chunk, ...]:
        settings = self._settings
        encoding = tiktoken.get_encoding(settings.encoding)
        tokens = encoding.encode(doc.text)
        stride = settings.max_tokens - settings.overlap_tokens

        spans: list[Span] = []
        start_tok = 0
        while True:
            end_tok = min(start_tok + settings.max_tokens, len(tokens))
            spans.append(_char_bounds(doc.text, encoding, tokens, start_tok, end_tok))
            if end_tok >= len(tokens):
                break
            start_tok += stride
        return _chunks(doc, spans)


class PackingChunker:
    """Splits a document into leaf units and packs whole leaves up to the token limit,
    never splitting a leaf that already fits. A leaf that arrives already at the limit,
    from having been subdivided upstream, becomes its own chunk."""

    def __init__(self, name: str, settings: ChunkSettings, spans: Splitter) -> None:
        self.name = name
        self._settings = settings
        self._spans = spans

    def __call__(self, doc: Document) -> tuple[Chunk, ...]:
        max_tokens = self._settings.max_tokens
        encoding = tiktoken.get_encoding(self._settings.encoding)
        leaves = self._spans(doc.text, encoding, max_tokens)

        packed: list[Span] = []
        start: int | None = None
        running = 0
        for leaf_start, leaf_end in leaves:
            leaf_tokens = len(encoding.encode(doc.text[leaf_start:leaf_end]))
            if start is None:
                start, running = leaf_start, leaf_tokens
            elif running + leaf_tokens > max_tokens:
                packed.append((start, leaf_start))
                start, running = leaf_start, leaf_tokens
            else:
                running += leaf_tokens
        if start is not None:
            packed.append((start, len(doc.text)))
        return _chunks(doc, packed)


def _chunks(doc: Document, spans: list[Span]) -> tuple[Chunk, ...]:
    """Numbers spans from zero and names each for its document and ordinal."""
    return tuple(
        Chunk(
            name=ChunkId(f"{doc.name}#{ordinal:04d}"), document=doc.name, ordinal=ordinal,
            span=Range(start, end), text=doc.text[start:end],
        )
        for ordinal, (start, end) in enumerate(spans)
    )


def get_chunker(settings: ChunkSettings) -> Chunker:
    if settings.strategy == "fixed":
        return FixedChunker(settings)
    if settings.strategy == "sentence":
        return PackingChunker("sentence", settings, _sentence_spans)
    if settings.strategy == "recursive":
        return PackingChunker("recursive", settings, _paragraph_spans)
    raise ValueError(f"no chunker registered for strategy {settings.strategy!r}")


def cut(documents: Sequence[Document], settings: ChunkSettings) -> tuple[Chunk, ...]:
    """Every document cut into chunks, by one chunker built for all of them."""
    chunker = get_chunker(settings)
    return tuple(chunk for doc in documents for chunk in chunker(doc))
