"""models.lock: schema shape, real (non alias) 40 hex commit sha revisions (D26/D18 discipline).

The repo root `models.lock` pins every embedding/reranker/generator model this system can serve.
Two entry shapes exist, named explicitly (SP9 task 3), not conflated:

- `local-tei` entries pin a real 40 hex Hugging Face commit sha, the only identity a self-hosted
  model has; `fingerprint.from_models_lock` is the runtime reader for the `embedding` list.
- Every other provider (`openai`, `anthropic`, `ollama`: an API model with no git sha to pin to)
  pins the model id string itself as its own identity: `revision == model_id` exactly. A revision
  of "latest"/"main" would let the served model drift silently underneath a passing test suite
  either way; only which string plays the pinned-identity role differs between the two shapes.

`test_reranker_entries_carry_every_required_field`/`test_generator_entries_carry_every_required_field`
assert directly against the JSON (not through `fingerprint.py`, which only ever reads `embedding`):
the reranker and generator lists have always been validated this way, unchanged by this task.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LOCK_PATH = Path("models.lock")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_ALIASES = {"latest", "main"}


def _load() -> dict:
    return json.loads(LOCK_PATH.read_text())


def _all_entries(data: dict) -> list[dict]:
    return [*data["embedding"], *data["reranker"], *data["generator"]]


def _tei_entries(data: dict) -> list[dict]:
    return [e for e in _all_entries(data) if e["provider"] == "local-tei"]


def _api_entries(data: dict) -> list[dict]:
    return [e for e in _all_entries(data) if e["provider"] != "local-tei"]


def test_models_lock_is_valid_json_with_the_three_top_level_keys() -> None:
    data = _load()
    assert set(data.keys()) == {"embedding", "reranker", "generator"}


def test_generator_list_carries_the_three_axis_entries() -> None:
    data = _load()
    assert {(e["provider"], e["model_id"]) for e in data["generator"]} == {
        ("anthropic", "claude-sonnet-5"),
        ("openai", "gpt-5.6-sol"),
        ("ollama", "qwen2.5:7b"),
    }


def test_embedding_entries_carry_every_required_field() -> None:
    data = _load()
    assert len(data["embedding"]) >= 1
    required = {"provider", "model_id", "revision", "dim", "normalize", "query_prefix", "document_prefix"}
    for entry in data["embedding"]:
        assert set(entry.keys()) == required


def test_reranker_entries_carry_every_required_field() -> None:
    data = _load()
    assert len(data["reranker"]) >= 1
    required = {"provider", "model_id", "revision"}
    for entry in data["reranker"]:
        assert set(entry.keys()) == required


def test_generator_entries_carry_every_required_field() -> None:
    data = _load()
    assert len(data["generator"]) >= 1
    required = {"provider", "model_id", "revision"}
    for entry in data["generator"]:
        assert set(entry.keys()) == required


def test_every_local_tei_entry_carries_a_40_hex_revision() -> None:
    data = _load()
    tei_entries = _tei_entries(data)
    assert len(tei_entries) >= 1  # the TEI shape must actually be exercised, not vacuously true
    for entry in tei_entries:
        revision = entry["revision"]
        assert _HEX40.match(revision), f"{entry['model_id']}: revision {revision!r} is not 40 lowercase hex chars"


def test_every_api_entry_pins_revision_to_its_own_model_id() -> None:
    """The API model-id shape (SP9 task 3): a provider other than `local-tei` has no git sha to pin
    to, so the pinned identity is the model id string itself, `revision == model_id` exactly."""
    data = _load()
    api_entries = _api_entries(data)
    assert len(api_entries) >= 1  # the API shape must actually be exercised, not vacuously true
    for entry in api_entries:
        assert entry["revision"] == entry["model_id"], (
            f"{entry['model_id']} ({entry['provider']}): API provider entries pin revision == model_id"
        )


def test_no_entry_uses_an_alias_revision() -> None:
    data = _load()
    for entry in _all_entries(data):
        assert entry["revision"] not in _ALIASES, f"{entry['model_id']}: alias revision is not allowed"


def test_no_entry_uses_an_alias_model_id() -> None:
    # The API shape self-pins revision to model_id, so a model_id of "latest"/"main" would sneak an
    # alias in under the other check's radar; guarded independently.
    data = _load()
    for entry in _all_entries(data):
        assert entry["model_id"] not in _ALIASES, f"{entry['provider']}: alias model_id is not allowed"


def test_bge_m3_embedding_entry_matches_the_discovered_pin() -> None:
    data = _load()
    entry = next(e for e in data["embedding"] if e["model_id"] == "BAAI/bge-m3")
    assert entry["provider"] == "local-tei"
    assert entry["dim"] == 1024
    assert entry["normalize"] is True
    assert entry["query_prefix"] == ""
    assert entry["document_prefix"] == ""
    assert entry["revision"] == "5617a9f61b028005a4858fdac845db406aefb181"


def test_openai_text_embedding_3_small_entry_matches_the_documented_api_shape() -> None:
    data = _load()
    entry = next(e for e in data["embedding"] if e["model_id"] == "text-embedding-3-small")
    assert entry["provider"] == "openai"
    assert entry["dim"] == 1536
    assert entry["normalize"] is True  # OpenAI's text-embedding-3 family already returns unit vectors
    assert entry["query_prefix"] == ""
    assert entry["document_prefix"] == ""
    assert entry["revision"] == "text-embedding-3-small"


def test_bge_reranker_entry_matches_the_discovered_pin() -> None:
    data = _load()
    entry = next(e for e in data["reranker"] if e["model_id"] == "BAAI/bge-reranker-v2-m3")
    assert entry["provider"] == "local-tei"
    assert entry["revision"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def test_anthropic_generator_entry_matches_the_configured_default() -> None:
    # Matches `replay.providers.DEFAULT_MODEL_IDS["anthropic"]`, the live default this repo already
    # configures; models.lock pins the same identity as its own axis entry, not a second guess.
    data = _load()
    entry = next(e for e in data["generator"] if e["provider"] == "anthropic")
    assert entry["model_id"] == "claude-sonnet-5"
    assert entry["revision"] == "claude-sonnet-5"


def test_openai_generator_entry_matches_the_configured_default() -> None:
    data = _load()
    entry = next(e for e in data["generator"] if e["provider"] == "openai")
    assert entry["model_id"] == "gpt-5.6-sol"
    assert entry["revision"] == "gpt-5.6-sol"


def test_ollama_generator_entry_matches_the_configured_default() -> None:
    data = _load()
    entry = next(e for e in data["generator"] if e["provider"] == "ollama")
    assert entry["model_id"] == "qwen2.5:7b"
    assert entry["revision"] == "qwen2.5:7b"


def test_generator_entries_match_replay_providers_default_model_ids() -> None:
    # Closes the loop against the one other place these three identities are configured (`replay.
    # providers.DEFAULT_MODEL_IDS`, the gateway's own live-provider factory): models.lock pins the
    # SAME model ids the runtime would actually call, never a second, silently-diverging guess.
    from replay.providers import DEFAULT_MODEL_IDS

    data = _load()
    pinned = {e["provider"]: e["model_id"] for e in data["generator"]}
    assert pinned == DEFAULT_MODEL_IDS
