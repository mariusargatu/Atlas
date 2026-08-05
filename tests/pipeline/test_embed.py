from __future__ import annotations

from dataclasses import replace

import numpy
import pytest

from atlas.config import EmbeddingSettings
from atlas.models.embed import CachedEmbedder, cache_key, get_embedder
from atlas.models.providers import OpenAIEmbedder

BASE = EmbeddingSettings(model_name="model.a", model_version="one", size=256)


class CountingEmbedder:
    model_name = "counting"

    def __init__(self, size: int = 8, model_version: str = "one") -> None:
        self.size = size
        self.model_version = model_version
        self.normalised = False
        self.calls = 0
        self.input_tokens = 0

    def encode(self, texts: list[str]) -> numpy.ndarray:
        self.calls += 1
        seed = hash(self.model_version) % 1000
        return numpy.array(
            [[float(seed + i) for i in range(self.size)] for _ in texts], dtype=numpy.float32
        )


def key(settings: EmbeddingSettings, text: str) -> str:
    """The key a CachedEmbedder built from these settings would file `text` under."""
    return cache_key(
        model_name=settings.model_name, model_version=settings.model_version,
        size=settings.size, normalised=settings.normalise, text=text,
    )


def test_the_cache_key_changes_with_every_component_and_not_only_the_text():
    """Keyed on text alone, the cache hands back vectors made by a different model after
    somebody swaps the model: no error, every number downstream wrong, every test green."""
    assert key(BASE, "hello") != key(replace(BASE, model_name="model.b"), "hello")
    assert key(BASE, "hello") != key(replace(BASE, model_version="two"), "hello")
    assert key(BASE, "hello") != key(replace(BASE, size=128), "hello")
    assert key(BASE, "hello") != key(replace(BASE, normalise=False), "hello")
    assert key(BASE, "hello") != key(BASE, "goodbye")
    assert key(BASE, "hello") == key(BASE, "hello")


def test_two_different_settings_cannot_run_together_into_the_same_key():
    # cache_key prefixes each component by its length so one value cannot impersonate
    # another. The pair has to be one that really collides without the prefixes: these
    # two concatenate to "text-embedding-3-small-2024-01" either way.
    left = replace(BASE, model_name="text-embedding-3", model_version="-small-2024-01")
    right = replace(BASE, model_name="text-embedding-3-", model_version="small-2024-01")
    assert key(left, "hello") != key(right, "hello")


def test_a_second_encode_of_the_same_text_is_served_from_disk(tmp_path):
    inner = CountingEmbedder(size=8)
    cached = CachedEmbedder(inner, tmp_path)
    first, second = cached.encode(["x"]), cached.encode(["x"])
    assert numpy.array_equal(first, second) and inner.calls == 1


def test_a_changed_model_does_not_read_the_previous_model_vectors(tmp_path):
    # The tests above still pass for a cache that computes the composite key and then
    # files everything under the text. This one connects the key to the cache.
    one = CountingEmbedder(size=8, model_version="one")
    two = CountingEmbedder(size=8, model_version="two")
    first = CachedEmbedder(one, tmp_path).encode(["x"])
    second = CachedEmbedder(two, tmp_path).encode(["x"])
    assert two.calls == 1
    assert not numpy.array_equal(first, second)


@pytest.mark.parametrize("build", [OpenAIEmbedder, get_embedder])
def test_a_model_name_that_is_not_an_embeddings_model_is_refused_by_either_route(build):
    # Both routes, because production code calls the factory and only tests call the
    # constructor. The message is matched, because a bare ValueError check also passes
    # when the constructor rejects the argument for some unrelated reason.
    with pytest.raises(ValueError, match="not an OpenAI embeddings model"):
        build(replace(BASE, model_name="minishlab/potion-base-8M"))


def test_a_cache_write_lands_as_the_real_file_and_never_a_stray_temp_one(tmp_path):
    # CachedEmbedder writes through a temp file plus os.replace, so a process killed
    # mid-write cannot leave a truncated .npy file for a later run to load.
    inner = CountingEmbedder(size=8)
    cached = CachedEmbedder(inner, tmp_path)
    cached.encode(["x"])
    written = list(tmp_path.iterdir())
    assert len(written) == 1 and written[0].suffix == ".npy"
    assert numpy.array_equal(numpy.load(written[0]), cached.encode(["x"])[0])


def test_a_purchase_larger_than_one_batch_is_cached_incrementally(tmp_path):
    # One inner call over the whole miss list means a batch failing partway through
    # discards every vector already bought in that call. Chunking the purchase and
    # caching each chunk as it lands bounds that loss to one batch.
    inner = CountingEmbedder(size=4)
    cached = CachedEmbedder(inner, tmp_path)
    texts = [f"text-{i}" for i in range(250)]  # over two batches at the 96-text batch size
    cached.encode(texts)
    assert inner.calls > 1, "the whole miss list went through the inner embedder in one call"
    assert len(list(tmp_path.iterdir())) == len(texts)


def test_the_configured_model_name_is_the_one_that_would_be_sent():
    # The old code fell back to text-embedding-3-small for any unrecognised name, so a
    # run could report one model in run_id and bill another.
    embedder = OpenAIEmbedder(replace(BASE, model_name="text-embedding-3-large"))
    assert embedder.model_name == "text-embedding-3-large"
