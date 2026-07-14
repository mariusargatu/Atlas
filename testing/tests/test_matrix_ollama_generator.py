"""`matrix.ollama_generator`, hermetic (SP9 task 7): the Ollama qwen2.5:7b LOCAL generator cell,
made real. Every test here is keyless and networkless: `inner` is always an injected stub
`BaseChatModel` (never a real Ollama daemon), and the module scope import guard proves the wiring
never NEEDS the `ollama` dependency group (`langchain_ollama`) installed to be collected or run,
matching `task install`'s bare `uv sync` in the hermetic PR lane.
"""
from __future__ import annotations

import ast
import inspect
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from replay.gateway import GatewayMode

from matrix.generators import GeneratorComponent
from matrix.ollama_generator import PROVIDER, build_ollama_generator_component


class _StubProvider(BaseChatModel):
    """The smallest real `BaseChatModel` a gateway will accept, mirroring `test_gateway.py`'s and
    `test_matrix_spend_gate.py`'s own `_StubProvider` shape -- never actually a live Ollama call."""

    reply: str = "stub qwen reply"

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _models_lock(tmp_path, *, generator_entries):
    path = tmp_path / "models.lock"
    path.write_text(json.dumps({"embedding": [], "reranker": [], "generator": generator_entries}))
    return path


# ---- the component builds with the real models.lock pinned identity ----------------------------


def test_build_ollama_generator_component_reads_the_real_models_lock_entry(tmp_path):
    component = build_ollama_generator_component(cassette_dir=tmp_path, inner=_StubProvider())
    assert isinstance(component, GeneratorComponent)
    assert component.component_id == "ollama-qwen2.5:7b"
    assert component.model_snapshot == {
        "provider": "ollama", "model_id": "qwen2.5:7b", "revision": "qwen2.5:7b",
    }
    assert component.estimated_usd == 0.0


def test_component_model_snapshot_matches_the_manifest_contracts_required_shape(tmp_path):
    # contracts/manifest/schema.json's model_snapshot: exactly {provider, model_id, revision},
    # additionalProperties false -- this must never drift silently.
    component = build_ollama_generator_component(cassette_dir=tmp_path, inner=_StubProvider())
    assert set(component.model_snapshot) == {"provider", "model_id", "revision"}


def test_gateway_is_always_record_mode(tmp_path):
    component = build_ollama_generator_component(cassette_dir=tmp_path, inner=_StubProvider())
    assert component.gateway.mode is GatewayMode.RECORD


def test_gateway_model_id_is_tagged_with_the_provider(tmp_path):
    component = build_ollama_generator_component(cassette_dir=tmp_path, inner=_StubProvider())
    assert component.gateway.model_id == "ollama:qwen2.5:7b"


# ---- the gateway actually round trips through RECORD, persisting a cassette for later REPLAY ----


def test_invoking_the_gateway_calls_the_stub_and_persists_a_cassette(tmp_path):
    component = build_ollama_generator_component(cassette_dir=tmp_path, inner=_StubProvider())
    result = component.gateway.invoke([HumanMessage("What is plan A's price?")])
    assert result.content == "stub qwen reply"

    cassettes = list(tmp_path.glob("*.json"))
    assert cassettes, "RECORD mode must persist a cassette the hermetic REPLAY lane can later serve"


def test_a_recorded_cassette_replays_identically_in_replay_mode(tmp_path):
    from replay.gateway import GatewayChatModel

    component = build_ollama_generator_component(cassette_dir=tmp_path, inner=_StubProvider())
    live_result = component.gateway.invoke([HumanMessage("Is my plan contract free?")])

    replay_gw = GatewayChatModel(model_id="ollama:qwen2.5:7b", cassette_dir=tmp_path, mode="replay")
    replayed = replay_gw.invoke([HumanMessage("Is my plan contract free?")])
    assert replayed.content == live_result.content


# ---- fail closed on a models.lock missing the ollama axis ---------------------------------------


def test_missing_ollama_entry_in_models_lock_fails_closed(tmp_path):
    lock_path = _models_lock(tmp_path, generator_entries=[
        {"provider": "anthropic", "model_id": "claude-sonnet-5", "revision": "claude-sonnet-5"},
    ])
    with pytest.raises(ValueError, match="no generator entry for provider='ollama'"):
        build_ollama_generator_component(
            cassette_dir=tmp_path, inner=_StubProvider(), models_lock_path=lock_path,
        )


def test_a_custom_models_lock_pin_is_honored_over_the_real_one(tmp_path):
    lock_path = _models_lock(tmp_path, generator_entries=[
        {"provider": "ollama", "model_id": "qwen2.5:14b", "revision": "qwen2.5:14b"},
    ])
    component = build_ollama_generator_component(
        cassette_dir=tmp_path, inner=_StubProvider(), models_lock_path=lock_path,
    )
    assert component.model_snapshot["model_id"] == "qwen2.5:14b"
    assert component.component_id == "ollama-qwen2.5:14b"


# ---- keyless: the live Ollama/langchain_ollama seam is never touched when `inner` is injected ---


def test_module_never_imports_replay_providers_at_module_scope():
    # `replay.providers` is itself lazy on `langchain_ollama` (not installed in the hermetic PR
    # lane's bare `uv sync`); this module must only ever reach it from inside a function body
    # (`_live_ollama_model`), never at module scope, or merely importing `matrix.ollama_generator`
    # would require the `ollama` dependency group to be installed.
    import matrix.ollama_generator as mod

    tree = ast.parse(inspect.getsource(mod))
    top_level_imports = [n for n in ast.iter_child_nodes(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = set()
    for node in top_level_imports:
        if isinstance(node, ast.ImportFrom):
            names.add(node.module)
        for alias in node.names:
            names.add(alias.name)
    assert "replay.providers" not in names
    assert "langchain_ollama" not in names


def test_provider_constant_is_ollama():
    assert PROVIDER == "ollama"


# ---- wired into the matrix: run_matrix accepts the real component end to end --------------------


def test_ollama_component_wires_into_run_matrix_end_to_end(tmp_path):
    """Proves the local generator cell is not just buildable in isolation but actually usable where
    the matrix runner (SP9 task 4) expects a `GeneratorComponent`: a full `run_matrix` call, one
    embedder, one reranker, ONE case, the Ollama component as the sole generator, a stub judge
    panel of one -- every piece keyless, none of it a live daemon call."""
    from atlas.adapters.cassette_reranker import CassetteReranker
    from atlas.ports.knowledge import Chunk

    from replay.gateway import GatewayChatModel

    from matrix.cache import MatrixCache
    from matrix.cases import MatrixCase
    from matrix.embedders import EmbedderComponent
    from matrix.rerankers import RerankerComponent
    from matrix.runner import MatrixRunConfig, run_matrix

    case = MatrixCase("case-a", "how much is plan a", frozenset({"d1"}), ({"fact_id": "a:price", "value": "10"},))
    chunk = Chunk(chunk_id="d1", doc_id="d1", text="plan a costs 10")

    def _search(_case: MatrixCase):
        return [chunk]

    embedder = EmbedderComponent(
        "bge-m3-local", _search, embedding_model={"id": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"},
    )
    reranker = RerankerComponent("bge-reranker-v2-m3", CassetteReranker({}))

    ollama_component = build_ollama_generator_component(
        cassette_dir=tmp_path / "cassettes", inner=_StubProvider(reply="Plan a costs 10."),
    )
    judge_gateway = GatewayChatModel(
        model_id="judge-stub", mode="record", cassette_dir=tmp_path / "cassettes", inner=_StubProvider(reply="PASS"),
    )

    config = MatrixRunConfig(
        run_id="run-ollama-wiring-test", git_sha="b" * 40, corpus_version="corpus-test-0.0.1",
        dataset_version="dataset-test-0.0.1", chunker_config_hash="chk-test", k_retrieval=1,
        seed=20260721, n_top_configs=1, reranker_depths=(20,),
    )
    manifest = run_matrix(
        (case,), embedders=[embedder], rerankers=[reranker], generators=[ollama_component],
        judges=[judge_gateway], judge_ids=("judge-stub",), cache=MatrixCache(tmp_path / "cache"),
        config=config, output_dir=tmp_path / "run",
    )

    generator_rows = manifest["stages"]["generators"]
    assert len(generator_rows) == 1
    row = generator_rows[0]
    assert row["generator_component_id"] == "ollama-qwen2.5:7b"
    assert row["lineage"]["model_snapshot"] == {
        "provider": "ollama", "model_id": "qwen2.5:7b", "revision": "qwen2.5:7b",
    }
    assert manifest["dropped_cells"] == []  # never rationed: ALWAYS_RUNS, even with no spend_gate passed
