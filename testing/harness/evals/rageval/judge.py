"""The judge seam. Every DeepEval metric takes an injectable judge via ``model=``, so Option A is a
one-line choice and a future door.

- ``build_ollama_judge`` pins a LOCAL judge (Ollama, temperature 0): no provider key, no egress, the
  operator lane's default. This is what makes DeepEval's RAG metrics runnable in this repo's spirit.
- ``build_replay_judge`` sketches the future: a ``DeepEvalBaseLLM`` whose ``generate`` is a cassette
  lookup, so calibrated judged metrics could replay DETERMINISTICALLY inside the PR gate, record in
  this lane, replay in the gate, Atlas's ``GatewayMode RECORD/REPLAY`` one level up. Not wired by
  default: DeepEval builds its own internal prompts, so the cassette keys must be proven stable across
  deepeval patch versions before this can gate. It is here to document the seam, not to switch on.

deepeval is imported lazily so importing this module never requires the ``rageval`` group.
"""
from __future__ import annotations

DEFAULT_JUDGE_MODEL = "llama3.1"  # a local Ollama model; pin the exact tag/quant in provenance


def build_ollama_judge(model_id: str = DEFAULT_JUDGE_MODEL, temperature: float = 0.0):
    """The pinned local judge for the operator lane. temperature 0 reduces (never eliminates) judge
    variance; the judge-calibration lane quantifies what remains before its verdicts are trusted."""
    from deepeval.models import OllamaModel  # lazy: rageval group only

    return OllamaModel(model=model_id, temperature=temperature)


def build_replay_judge(cassette_dir: str):
    """A cassette-backed judge (the future deterministic path). Returns a ``DeepEvalBaseLLM`` whose
    ``generate`` replays a recorded verdict for a prompt. Defined lazily so deepeval is only needed
    when an operator actually opts in. RECORD these verdicts in this lane; REPLAY them in the gate."""
    import json
    from pathlib import Path

    from determinism.canonical import digest  # the repo's canonical cassette-key digest (ADR-007)
    from deepeval.models.base_model import DeepEvalBaseLLM  # lazy: rageval group only

    class ReplayJudge(DeepEvalBaseLLM):
        def __init__(self, directory: str) -> None:
            self._dir = Path(directory)

        def load_model(self):  # no model to load; the "model" is the cassette store
            return self

        def _key(self, prompt: str) -> str:
            # Key via the repo's canonical digest (ADR-007) — the one content-addressing function the
            # cassette store and run digest already use — rather than a bespoke sha256, so this seam
            # shares the same key contract if/when it is switched on.
            return digest(prompt)

        def generate(self, prompt: str, schema=None):
            path = self._dir / f"{self._key(prompt)}.json"
            if not path.exists():
                raise KeyError(
                    "no recorded judge verdict for this prompt; record in the operator lane first "
                    "(a replayed judge cannot invent a verdict it never saw)"
                )
            payload = json.loads(path.read_text())
            return schema(**payload) if schema is not None else payload["text"]

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return "replay-judge"

    return ReplayJudge(cassette_dir)
