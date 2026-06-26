"""SP10 task 4: the Simulator lane's own driver (`.github/workflows/simulator.yml`, manual
`workflow_dispatch`, pre burst stage). WIRES existing mechanics; builds only the genuinely new
piece the HLD names for this lane (a live, generative, cross model persona loop), never a second
copy of anything an earlier sub project already shipped:

  - the persona roster: `evals.simulation.personas.PERSONAS` (name/disposition/goal), reused
    unchanged -- the SAME roster `task simulation`'s own deterministic demo already prints.
  - grading: `judge.llm_judge.judge_label`, reused unchanged, against TWO new rubrics
    (`judge.rubric.RUBRIC_PERSONA_ADHERENCE`, `judge.rubric.RUBRIC_TASK_SUCCESS`) -- the SAME
    generic `prompt()`/PASS/FAIL parsing shape `RUBRIC_GROUNDEDNESS` already uses, just re
    purposed (see rubric.py's own comment on why two rubrics, not one).
  - the executed action audit trail: `atlas.domain.actions.ActionsBackend.applied`, read the SAME
    way `evals.simulation.driver.drive_conversation` already reads it for the scripted mind-changer
    fixture -- what the agent actually DID, never the prose it ended on.

THE ONE GENUINELY NEW PIECE: `evals.simulation.driver.drive_conversation` replays a pre SCRIPTED
`Conversation` (a fixture authored once, frozen, ADR-019); it has no generative loop, because it was
never asked to have one. A live Simulator lane needs the persona's NEXT customer utterance produced
live, by a model, in character, turn by turn, against the real agent's real replies -- that loop
(`drive_persona_episode`/`next_persona_turn` below) is what this module adds; nothing upstream
already provides it, and this module does not touch `evals.simulation.driver` to avoid duplicating
its own, deliberately different, scripted replay contract.

THE CROSS MODEL BOUNDARY (SP8's own: never let the same model grade its own persona conversation,
generalised here from "the agent never judges its own answer" to "the persona player never judges
its own persona performance"): `select_driver_and_evaluator_tiers` requires BOTH `OPENAI_API_KEY`
AND `ANTHROPIC_API_KEY` (not "either one", unlike `judge.live_pr_lane.select_judge_tier` /
`judge.full_sweep.select_frontier_judge_tier`, since a genuine cross model boundary needs two
distinct model families to exist at all, never one key read twice for two roles).

PASS^K (k=4, HLD 7.3's "pass to the k power"), NEVER GATES (D18, and the HLD's own lane table:
Simulator "Never" gates): every episode's PERSONA ADHERENCE is graded first; an episode whose
persona player broke character is marked INVALID and EXCLUDED from the pass^k denominator (a broken
simulator is evidence about the SIMULATOR, never about the agent under test), not silently counted
as a task failure. `PersonaPassKReport` reports the reading TWO ways, together, never one standing
in for the other (the same "report every honest reading" discipline `judge.calibration
.CalibrationReport` already holds itself to for kappa, raw agreement, AC1, prevalence): the literal
empirical readout (did every valid trial in this k sized batch pass) and the textbook reliability
estimate (the measured per trial pass rate raised to the k power, the probability of k independent
successes in a row, the SAME sample the rate itself came from, not a hypothetical).

INFRASTRUCTURE, NAMED (the SP10 digest's own 3g, NOT infrastructure free): the live loop drives the
REAL `atlas_graph` through retrieval exactly like `labeling.generate_label_set` already does, so it
needs a reachable TEI endpoint (`docker compose up postgres tei-embed tei-rerank rag-init`, keyless
retrieval) PLUS the driver/evaluator provider keys above. "Manual, pre burst stage" names WHEN this
lane runs, never that it runs with no infrastructure at all.

DEPENDENCY INJECTION for hermetic testability (the SAME discipline `judge.live_pr_lane` /
`judge.full_sweep` already hold to): `run_persona`/`drive_persona_episode` take an already built SUT
graph factory, an already built persona player gateway, and an already built evaluator gateway, so
the WHOLE multi turn loop plus grading here is proven end to end in
`testing/tests/test_judge_simulator_lane.py` with REPLAY gateways on every one of the three roles --
zero keys, zero egress. Only `main()` and the `build_live_*` helpers reach for a real live provider
or a real Postgres/TEI retriever, and only when this file is run directly (`task simulation:live`,
manual, never the PR lane).
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from evals.artifacts import write_artifacts
from evals.simulation.personas import PERSONAS, Persona

from judge.llm_judge import judge_label
from judge.live_pr_lane import build_live_judge_gateway
from judge.rubric import RUBRIC_PERSONA_ADHERENCE, RUBRIC_TASK_SUCCESS

from labeling.generate_label_set import _GENERATION_CUSTOMER

from matrix.live_driver import MissingEnvVarError

from quality.stats import wilson_interval_from_rate

from replay.gateway import GatewayChatModel

_ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "simulator_lane"
# LIVE mode never reads this (GatewayChatModel._check_wiring only requires a cassette store for
# REPLAY/RECORD); named for symmetry with `judge.live_pr_lane._LIVE_CASSETTE_DIR`.
_LIVE_CASSETTE_DIR = Path("var") / "simulator_lane" / "cassettes"

K = 4  # HLD 7.3's own "pass to the k power (k=4)" -- SP10's plan text pins this number explicitly.

# A bound on the persona driven loop: the persona player is asked to end its own turn with
# `_END_TOKEN` once its goal is reasonably addressed, but a live model can fail to follow that
# instruction, and this loop must never spin unboundedly against a real, billed provider. Four
# turns is enough for every persona in the roster to reach or abandon its stated goal (the
# mind-changer fixture itself, `evals.datasets.simulation_golden.MIND_CHANGER`, settles in three).
MAX_TURNS = 4

_END_TOKEN = "[END]"

# The two provider tiers this lane picks between. Cheapest per family, matching the tie break
# `judge.live_pr_lane._JUDGE_TIERS` already establishes, kept as an independent, smaller pair here
# (a shared abstraction over three call sites this small would be a premature one -- hoist to a
# shared module if a fourth caller ever needs it, matching `judge.live_pr_lane`'s own reasoning for
# not importing `judge.live_provisional`'s private constant).
_OPENAI_TIER = ("openai", "gpt-5.4-nano")
_ANTHROPIC_TIER = ("anthropic", "claude-haiku-4-5-20251001")


def select_driver_and_evaluator_tiers() -> tuple[tuple[str, str], tuple[str, str]]:
    """Returns ``((driver_provider, driver_model_id), (evaluator_provider, evaluator_model_id))``,
    ALWAYS two different provider families. Needs BOTH `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`
    configured -- see this module's own docstring for why "either one" (the Live PR / Full sweep
    lanes' own rule) is not enough here. OpenAI plays the persona (run many times per episode/turn,
    so the cheap tier matters most there); Anthropic's Haiku tier evaluates -- an arbitrary but
    deterministic assignment, never re rolled per run. Raises `MissingEnvVarError` -- the SAME fail
    closed exception type every other live entrypoint in this repo raises -- before any live call if
    either key is missing.
    """
    have_openai = bool(os.environ.get("OPENAI_API_KEY"))
    have_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not (have_openai and have_anthropic):
        missing = [
            name for name, present in (("OPENAI_API_KEY", have_openai), ("ANTHROPIC_API_KEY", have_anthropic))
            if not present
        ]
        raise MissingEnvVarError(
            "the Simulator lane needs BOTH OPENAI_API_KEY and ANTHROPIC_API_KEY set (missing: "
            f"{', '.join(missing)}): the persona driver and the calibrated evaluator must be two "
            "different model families (SP8's own cross model boundary -- never the same model "
            "grading its own persona conversation), never one key read twice for both roles. "
            "Refusing to attempt any live call with either missing."
        )
    return _OPENAI_TIER, _ANTHROPIC_TIER


# ---- the persona driven turn loop -------------------------------------------------------------------


def persona_system_prompt(persona: Persona) -> str:
    return (
        f"You are role playing a broadband customer support caller. Disposition: {persona.disposition}. "
        f"Goal: {persona.goal}. Write ONLY the next thing you (the customer) would say, one turn at a "
        "time, in character; never break character, never mention you are an AI or a simulation, never "
        "narrate stage directions. When your goal has been reasonably addressed (met, correctly "
        f"declined, or you have genuinely given up), end your final message with the exact token "
        f"{_END_TOKEN} and nothing after it."
    )


def _persona_messages(persona: Persona, transcript: tuple[tuple[str, str], ...]) -> list[BaseMessage]:
    """The persona player's own view of the conversation so far: the agent's replies arrive as
    `HumanMessage` (what the persona player reacts to), and the persona's OWN prior turns replay as
    `AIMessage` (its own past output) -- the standard role inversion for "this model plays the OTHER
    seat," which `evals.simulation.driver` itself never needed (it replays pre scripted turns, never
    generative ones)."""
    messages: list[BaseMessage] = [SystemMessage(persona_system_prompt(persona))]
    for speaker, text in transcript:
        messages.append(AIMessage(text) if speaker == "customer" else HumanMessage(text))
    return messages


def next_persona_turn(persona_gateway, persona: Persona, transcript: tuple[tuple[str, str], ...]) -> tuple[str, bool]:
    """Ask the persona player model for the next customer utterance. Returns ``(utterance, done)``:
    `done` is True iff the reply carried `_END_TOKEN` (stripped out of the returned utterance
    either way, so a caller never has to re parse it)."""
    reply = persona_gateway.invoke(_persona_messages(persona, transcript))
    content = (getattr(reply, "content", "") or "").strip()
    done = _END_TOKEN in content
    utterance = content.replace(_END_TOKEN, "").strip()
    return utterance, done


async def drive_persona_episode(
    persona: Persona,
    *,
    sut_graph,
    backend,
    persona_gateway,
    thread_id: str,
    customer_id: str = _GENERATION_CUSTOMER,
    max_turns: int = MAX_TURNS,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, dict], ...]]:
    """Drive one live episode: the persona player proposes each customer turn, the real `sut_graph`
    answers it, on one thread, up to `max_turns` rounds or until the persona player signals it is
    done. Mirrors `evals.simulation.driver.drive_conversation`'s own confirmation interrupt handling
    (a write action pauses the graph; resuming with a literal `CONFIRM` is exactly what a customer
    saying yes does) and its own audit log reading of what actually executed, reused the same way,
    just against a GENERATED transcript instead of a pre scripted one. Returns
    ``(transcript, actions)``: `transcript` is an ordered ``(speaker, text)`` tuple (`speaker` is
    `"customer"` or `"agent"`); `actions` is the SAME `(tool, args)` audit shape
    `evals.simulation.model.ConversationOutcome.actions` already uses.
    """
    from langgraph.types import Command  # lazy: pure grade tests never build a graph

    from atlas.orchestration.atlas_graph import thread_config

    transcript: tuple[tuple[str, str], ...] = ()
    config = thread_config(thread_id)
    for _ in range(max_turns):
        utterance, done = next_persona_turn(persona_gateway, persona, transcript)
        if not utterance:
            break
        transcript = transcript + (("customer", utterance),)
        out = await sut_graph.ainvoke(
            {"messages": [HumanMessage(utterance)], "session": {"customer_id": customer_id}}, config
        )
        if "__interrupt__" in out:  # a write action paused at the confirmation gate
            out = await sut_graph.ainvoke(Command(resume="CONFIRM"), config)
        transcript = transcript + (("agent", out.get("final_response") or ""),)
        if done:
            break
    applied = backend.applied(customer_id)
    actions = tuple((a.tool, dict(a.args)) for a in applied)
    return transcript, actions


def _speaker_text(transcript: tuple[tuple[str, str], ...], speaker: str) -> str:
    return "\n".join(text for who, text in transcript if who == speaker)


def _full_transcript_text(transcript: tuple[tuple[str, str], ...]) -> str:
    return "\n".join(f"{who}: {text}" for who, text in transcript)


def persona_adherence_question(persona: Persona) -> str:
    """The persona adherence rubric's own `question` text (extracted, not inlined, so the hermetic
    test that seeds a REPLAY cassette builds the exact same request `grade_persona_adherence` will
    send, the SAME "reuse the caller's own builder" idiom `judge.live_pr_lane.judge_context` already
    sets for its own callers/tests)."""
    return (
        f"Persona: {persona.name} -- {persona.disposition}. Goal: {persona.goal}. Does every CUSTOMER "
        "line below stay in character for this persona?"
    )


def task_success_question(persona: Persona, actions: tuple[tuple[str, dict], ...]) -> str:
    """The task success rubric's own `question` text (extracted for the same reason
    `persona_adherence_question` is)."""
    return (
        f"Customer goal: {persona.goal}. Actions the agent actually executed: {list(actions)!r}. Given "
        "the whole conversation below, did the agent reasonably serve this goal, or safely and "
        "correctly decline/hand off when it could not?"
    )


def grade_persona_adherence(evaluator_gateway, persona: Persona, transcript: tuple[tuple[str, str], ...]) -> int:
    """Judged, never deterministic: whether every CUSTOMER turn stayed in character. Reuses
    `judge.llm_judge.judge_label` unchanged, re purposing `prompt()`'s question/answer/context shape
    (see `judge.rubric`'s own comment for why -- `question` carries the persona spec, `answer`
    carries the customer's own turns, `context` stays empty)."""
    question = persona_adherence_question(persona)
    return judge_label(evaluator_gateway, RUBRIC_PERSONA_ADHERENCE, question, _speaker_text(transcript, "customer"), "")


def grade_task_success(
    evaluator_gateway, persona: Persona, transcript: tuple[tuple[str, str], ...], actions: tuple[tuple[str, dict], ...]
) -> int:
    """Judged: whether the AGENT's handling of the whole conversation reasonably served the
    persona's goal. `actions` (the executed audit trail) is folded into the question text, not just
    the transcript, because a write action's own correctness is what the account state changed to,
    not necessarily a sentence the final reply states verbatim."""
    question = task_success_question(persona, actions)
    return judge_label(evaluator_gateway, RUBRIC_TASK_SUCCESS, question, _full_transcript_text(transcript), "")


# ---- pass^k aggregation, per persona ------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeResult:
    """One trial: the generated transcript, the executed action audit trail, and both judged
    verdicts. `task_pass` is `False` (never graded) when `adherent` is `False` -- a non adherent
    episode is excluded from the pass^k denominator regardless of this value (see
    `PersonaPassKReport.valid_episodes`), and skipping the second live judge call on an already
    invalid episode is one fewer paid call for a reading nothing downstream will read anyway."""

    persona: str
    trial: int
    transcript: tuple[tuple[str, str], ...]
    actions: tuple[tuple[str, dict], ...]
    adherent: bool
    task_pass: bool

    def as_dict(self) -> dict:
        return {
            "persona": self.persona,
            "trial": self.trial,
            "adherent": self.adherent,
            "task_pass": self.task_pass,
            "actions": [[tool, args] for tool, args in self.actions],
            "transcript": [[who, text] for who, text in self.transcript],
        }


@dataclass(frozen=True)
class PersonaPassKReport:
    """One persona's k trial batch. `valid_episodes` (adherent only) is the denominator for every
    rate below; `invalid_count` is reported alongside, never silently dropped."""

    persona: str
    k: int
    episodes: tuple[EpisodeResult, ...]

    @property
    def valid_episodes(self) -> tuple[EpisodeResult, ...]:
        return tuple(e for e in self.episodes if e.adherent)

    @property
    def invalid_count(self) -> int:
        return len(self.episodes) - len(self.valid_episodes)

    @property
    def valid_pass_rate(self) -> float | None:
        """`None` when every trial in the batch was invalidated (persona adherence failed every
        time -- a simulator quality problem, nothing to read the agent's own reliability from)."""
        valid = self.valid_episodes
        return None if not valid else sum(1 for e in valid if e.task_pass) / len(valid)

    @property
    def valid_pass_rate_ci95(self) -> tuple[float, float]:
        """The Wilson interval over the valid trial count (never a bare rate, matching this repo's
        own discipline elsewhere); `(0.0, 1.0)` -- maximally uninformative -- when nothing survived
        adherence filtering."""
        rate = self.valid_pass_rate
        if rate is None:
            return (0.0, 1.0)
        return wilson_interval_from_rate(rate, len(self.valid_episodes))

    @property
    def pass_k_estimate(self) -> float | None:
        """The textbook reading of "pass to the k power": the measured per trial pass RATE raised
        to the k power, the probability of k independent successes in a row, assuming the trials are
        iid -- this lane's own k trials are the SAME sample the rate is measured from, so this is
        the estimate that sample actually licenses, not a hypothetical one. `None` when
        `valid_pass_rate` is (nothing survived adherence filtering)."""
        rate = self.valid_pass_rate
        return None if rate is None else rate**self.k

    @property
    def all_valid_trials_passed(self) -> bool | None:
        """The literal empirical readout for this batch: True iff EVERY valid episode in the
        k trial batch passed AND at least one trial survived adherence filtering. `None` when every
        trial was invalidated."""
        valid = self.valid_episodes
        return None if not valid else all(e.task_pass for e in valid)

    def render(self) -> str:
        rate = self.valid_pass_rate
        est = self.pass_k_estimate
        lo, hi = self.valid_pass_rate_ci95
        rate_text = "n/a (every trial invalidated)" if rate is None else f"{rate:.3f} 95% CI [{lo:.3f}, {hi:.3f}]"
        est_text = "n/a" if est is None else f"{est:.4f}"
        all_text = "n/a" if self.all_valid_trials_passed is None else str(self.all_valid_trials_passed)
        return (
            f"persona={self.persona:<20} k={self.k}  valid={len(self.valid_episodes)}  "
            f"invalid={self.invalid_count}  per trial pass rate={rate_text}  "
            f"pass^{self.k} estimate={est_text}  all {self.k} valid trials passed={all_text}"
        )

    def as_dict(self) -> dict:
        return {
            "persona": self.persona,
            "k": self.k,
            "valid_count": len(self.valid_episodes),
            "invalid_count": self.invalid_count,
            "valid_pass_rate": self.valid_pass_rate,
            "valid_pass_rate_ci95": list(self.valid_pass_rate_ci95),
            "pass_k_estimate": self.pass_k_estimate,
            "all_valid_trials_passed": self.all_valid_trials_passed,
            "episodes": [e.as_dict() for e in self.episodes],
        }


def run_persona(
    persona: Persona,
    *,
    k: int,
    sut_graph_factory,
    persona_gateway,
    evaluator_gateway,
    customer_id: str = _GENERATION_CUSTOMER,
) -> PersonaPassKReport:
    """Run `k` independent trials of one persona. `sut_graph_factory()` must return a FRESH
    `(graph, backend)` pair per call (a fresh checkpointer and a fresh `ActionsBackend`), so one
    trial's executed actions can never leak into another trial's own audit read -- the SAME "fresh
    backend per run" discipline `labeling.generate_label_set.build_generation_graph` already holds
    per script invocation, applied here per TRIAL instead, since a persona runs k of them."""
    episodes = []
    for trial in range(k):
        sut_graph, backend = sut_graph_factory()
        transcript, actions = asyncio.run(
            drive_persona_episode(
                persona, sut_graph=sut_graph, backend=backend, persona_gateway=persona_gateway,
                thread_id=f"simulator::{persona.name}::{trial}", customer_id=customer_id,
            )
        )
        adherent = bool(grade_persona_adherence(evaluator_gateway, persona, transcript))
        task_pass = bool(grade_task_success(evaluator_gateway, persona, transcript, actions)) if adherent else False
        episodes.append(EpisodeResult(persona.name, trial, transcript, actions, adherent, task_pass))
    return PersonaPassKReport(persona.name, k, tuple(episodes))


def run_simulator_lane(
    personas: list[Persona], *, k: int, sut_graph_factory, persona_gateway, evaluator_gateway,
    customer_id: str = _GENERATION_CUSTOMER,
) -> tuple[PersonaPassKReport, ...]:
    return tuple(
        run_persona(
            p, k=k, sut_graph_factory=sut_graph_factory, persona_gateway=persona_gateway,
            evaluator_gateway=evaluator_gateway, customer_id=customer_id,
        )
        for p in personas
    )


def render_report(
    reports: tuple[PersonaPassKReport, ...], *, driver_provider: str, driver_model_id: str,
    evaluator_provider: str, evaluator_model_id: str,
) -> str:
    lines = [
        "# Simulator lane: cross model persona driver + separate calibrated evaluator, pass^k (k=4)",
        f"persona driver: {driver_provider}:{driver_model_id}",
        f"evaluator (SEPARATE model family, SP8's cross model boundary): {evaluator_provider}:{evaluator_model_id}",
        "",
        "## Per persona (report only -- NEVER GATES, D18 / HLD 7.3's Simulator row 'Never')",
    ]
    for report in reports:
        lines.append(report.render())
    lines.append(
        "\n(persona adherence honesty: an episode whose SIMULATOR broke character is EXCLUDED from "
        "every rate above, not counted as a task failure -- see judge/simulator_lane.py's own module "
        "docstring. pass^k here is BOTH the literal all k trials passed empirical readout and the "
        "per trial rate raised to the k reliability estimate, reported together, never one standing "
        "in for the other.)"
    )
    return "\n".join(lines)


# ---- live wiring (reached only when this file is run directly; never imported by task test) --------


def build_live_persona_gateway(provider: str, model_id: str) -> GatewayChatModel:
    """The persona player's own live gateway. Deliberately NOT `judge.live_pr_lane
    .build_live_judge_gateway` under a borrowed name: that function's `:judge` model_id tag would
    misdescribe this role (the persona player never returns a PASS/FAIL verdict, it improvises a
    customer's next line), even though the two are otherwise the same one line "wrap a live chat
    model for the gateway" construction. `build_live_judge_gateway` itself is reused UNCHANGED for
    the evaluator role below, since that role genuinely is a judge."""
    from replay.providers import build_chat_model

    return GatewayChatModel(
        model_id=f"{provider}:{model_id}:persona", mode="live", inner=build_chat_model(provider, model_id)
    )


def build_live_sut_graph_factory():
    """A zero argument factory building one FRESH live SUT graph + backend per call (real
    generation via `MODEL_PROVIDER`/`MODEL_ID` env, real retrieval via `PgvectorRetriever`) --
    deliberately NOT `judge.live_pr_lane.build_live_agent` (which returns `(graph, tracer)`, never
    exposing the `ActionsBackend` this lane's own task success grading needs to read). A minimal,
    independent graph assembly, the SAME shape `labeling.generate_label_set.build_generation_graph`
    already uses internally, kept separate rather than changing that already reviewed function's
    return shape under its own two existing callers."""
    from determinism.checkpointer import new_checkpointer
    from determinism.sources import IdFactory
    from tracing import InMemoryTracer

    from atlas.adapters.pgvector_retriever import PgvectorRetriever
    from atlas.domain.actions import ActionsBackend
    from atlas.orchestration.atlas_graph import build_atlas_graph

    from replay.providers import build_chat_model, provider_tag

    def factory():
        gw = GatewayChatModel(
            model_id=provider_tag(), cassette_dir=_LIVE_CASSETTE_DIR, mode="live", inner=build_chat_model()
        )
        backend = ActionsBackend(IdFactory("ref"))
        graph = build_atlas_graph(
            gw, IdFactory("idem"), backend, new_checkpointer(), retriever=PgvectorRetriever(), tracer=InMemoryTracer()
        )
        return graph, backend

    return factory


def main() -> None:
    (driver_provider, driver_model_id), (evaluator_provider, evaluator_model_id) = (
        select_driver_and_evaluator_tiers()
    )  # fail closed before any live call, key or retriever
    persona_gateway = build_live_persona_gateway(driver_provider, driver_model_id)
    evaluator_gateway = build_live_judge_gateway(evaluator_provider, evaluator_model_id)
    sut_graph_factory = build_live_sut_graph_factory()

    reports = run_simulator_lane(
        PERSONAS, k=K, sut_graph_factory=sut_graph_factory, persona_gateway=persona_gateway,
        evaluator_gateway=evaluator_gateway,
    )

    rendered = render_report(
        reports, driver_provider=driver_provider, driver_model_id=driver_model_id,
        evaluator_provider=evaluator_provider, evaluator_model_id=evaluator_model_id,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    write_artifacts(
        [(_ARTIFACT_DIR / "latest.md", rendered), (_ARTIFACT_DIR / f"{stamp}.md", rendered)], echo=rendered
    )
    json_path = _ARTIFACT_DIR / f"{stamp}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([r.as_dict() for r in reports], indent=2, sort_keys=True))
    print("\n(this lane never gates -- D18, HLD 7.3's Simulator row 'Never' -- the process exit code "
          "below reflects a real script error only, never a pass^k reading.)")


if __name__ == "__main__":
    main()


__all__ = [
    "K",
    "MAX_TURNS",
    "EpisodeResult",
    "PersonaPassKReport",
    "build_live_persona_gateway",
    "build_live_sut_graph_factory",
    "drive_persona_episode",
    "grade_persona_adherence",
    "grade_task_success",
    "next_persona_turn",
    "persona_adherence_question",
    "persona_system_prompt",
    "render_report",
    "run_persona",
    "run_simulator_lane",
    "select_driver_and_evaluator_tiers",
    "task_success_question",
]
