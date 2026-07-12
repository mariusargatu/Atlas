"""The judge seam. Every DeepEval metric takes an injectable judge via ``model=``, so Option A is a
one-line choice and a future door.

- ``build_ollama_judge`` pins a local judge (Ollama, temperature 0): no provider key, no egress, the
  operator lane's default.
- ``build_replay_judge`` sketches the future: a ``DeepEvalBaseLLM`` whose ``generate`` is a cassette
  lookup, so calibrated judged metrics could replay deterministically inside the PR gate, the same
  record/replay shape ``GatewayMode`` uses one level up. Not wired by default: DeepEval builds its own
  internal prompts, so the cassette keys must be proven stable across deepeval patch versions first.
  It documents the seam, not a switched-on feature.

deepeval is imported lazily so importing this module never requires the ``rageval`` group.
"""
from __future__ import annotations

DEFAULT_JUDGE_MODEL = "llama3.1"  # a local Ollama model; pin the exact tag/quant in provenance


def build_ollama_judge(model_id: str = DEFAULT_JUDGE_MODEL, temperature: float = 0.0):
    """The pinned local judge for the operator lane. temperature 0 reduces (never eliminates) judge
    variance; the judge-calibration lane quantifies what remains before its verdicts are trusted."""
    from deepeval.models import OllamaModel  # lazy: rageval group only

    return OllamaModel(model=model_id, temperature=temperature)


# The judge calibration study's prior winner (`task judge-live`, see
# testing/harness/judge/artifacts/live_provisional/): the cheapest cross family tier that licensed
# under the retired account facts rubric. Kept for reference; this key cannot use it for DeepEval's
# structured output judge calls, so it is not the default below.
CALIBRATION_JUDGE_MODEL = "gpt-5.6-luna"
# The default DeepEval judge: cross-family (ADR-004), pinned to temperature 0, and the one model
# this key can run DeepEval's structured-output judge calls with.
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.4-nano"


def build_openai_judge(model_id: str = DEFAULT_OPENAI_JUDGE_MODEL, temperature: float = 0.0):
    """A pinned OpenAI judge for the DeepEval lanes (needs OPENAI_API_KEY, egress). Returned as a model
    object (not a bare name) so it can be wrapped for recording.

    Default ``gpt-5.4-nano`` at temperature 0, the model this key can run DeepEval's structured-output
    judge calls with; ``CALIBRATION_JUDGE_MODEL`` cannot. Re-run ``task judge-live`` before trusting
    either judge; both drift."""
    from deepeval.models import GPTModel  # lazy: rageval group only

    return GPTModel(model=model_id, temperature=temperature)


def build_default_judge():
    """The operator lane's default judge, as ``(judge, label)``. Uses the cross-family OpenAI judge
    (``gpt-5.4-nano``, the tier this key can run DeepEval's structured-output calls with) when
    ``OPENAI_API_KEY`` is set; falls back to the local Ollama judge (egress-free) when it is not. The
    calibration winner ``gpt-5.6-luna`` is not used here: it 401s on DeepEval's structured-output calls."""
    import os

    if os.environ.get("OPENAI_API_KEY"):
        return build_openai_judge(), f"openai:{DEFAULT_OPENAI_JUDGE_MODEL}"
    return build_ollama_judge(), f"ollama:{DEFAULT_JUDGE_MODEL}"


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
            # Key via the repo's canonical digest (ADR-007), the one content-addressing function the
            # cassette store and run digest already use, rather than a bespoke sha256, so this seam
            # shares the same key contract if/when it is switched on.
            return digest(prompt)

        def generate(self, prompt: str, schema=None):
            path = self._dir / f"{self._key(prompt)}.json"
            if not path.exists():
                raise KeyError(
                    "no recorded judge verdict for this prompt; record in the operator lane first "
                    "(a replayed judge cannot invent a verdict it never saw)"
                )
            data = json.loads(path.read_text())
            payload = data.get("payload", data)  # new cassettes wrap the verdict under `payload` with a
            return schema(**payload) if schema is not None else payload["text"]  # `provenance` sibling

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return "replay-judge"

    return ReplayJudge(cassette_dir)


def build_recording_judge(inner, cassette_dir: str, provenance: str = ""):
    """The RECORD half of ``build_replay_judge``'s REPLAY: wrap a LIVE judge so every verdict it
    returns is also frozen to a cassette (keyed by the same canonical digest of the prompt). Run once
    against a live model to capture a session; ``build_replay_judge`` replays it forever after with no
    egress. A built-in provider (e.g. ``GPTModel``) returns a ``(result, cost)`` tuple where a custom
    judge returns the bare result, so both shapes are unwrapped before persisting.

    Each cassette records a ``provenance`` block (which judge / model / deepeval version produced it)
    beside the ``payload``, so a replayed verdict is auditable back to its source rather than an
    unattributed magic number."""
    import json
    from importlib.metadata import PackageNotFoundError, version
    from pathlib import Path

    from determinism.canonical import digest  # the same key function ReplayJudge reads with
    from deepeval.models.base_model import DeepEvalBaseLLM  # lazy: rageval group only

    directory = Path(cassette_dir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        _deepeval_version = version("deepeval")
    except PackageNotFoundError:
        _deepeval_version = "unknown"
    _prov = {"judge": provenance or inner.get_model_name(), "model": inner.get_model_name(),
             "deepeval": _deepeval_version}

    def _save(prompt: str, result) -> None:
        payload = result.model_dump() if hasattr(result, "model_dump") else {"text": str(result)}
        data = {"provenance": _prov, "payload": payload}
        (directory / f"{digest(prompt)}.json").write_text(json.dumps(data, indent=2, sort_keys=True))

    class RecordingJudge(DeepEvalBaseLLM):
        def load_model(self):
            return self

        def get_model_name(self) -> str:
            return f"recording:{inner.get_model_name()}"

        def generate(self, prompt: str, schema=None):
            out = inner.generate(prompt, schema=schema)
            result = out[0] if isinstance(out, tuple) else out  # unwrap (result, cost); deepeval's
            _save(prompt, result)                               # extractor expects the bare result from
            return result                                       # a custom judge, not the tuple

        async def a_generate(self, prompt: str, schema=None):
            out = await inner.a_generate(prompt, schema=schema)
            result = out[0] if isinstance(out, tuple) else out
            _save(prompt, result)
            return result

    return RecordingJudge()
