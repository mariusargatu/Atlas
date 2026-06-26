# Judge rubrics (drafts, not yet wired)

Four LLM judge rubrics authored as YAML, kept beside `judge/rubric.py`, the module that owns the
`Rubric` concept and is their only plausible consumer.

Nothing loads these yet. `judge/rubric.py` still declares its rubrics as Python literals
(`RUBRIC_GROUNDEDNESS`, and the persona/task-success pair). These YAML files are the newer,
config driven shape; wiring a loader is not done.

| file | `version` |
|---|---|
| `groundedness.yaml` | `groundedness-v1` |
| `task_success_direct_answer.yaml` | `task-success-direct-answer-v1` |
| `tone_empathy.yaml` | `tone-empathy-v1` |
| `exposition_clarity.yaml` | `exposition-clarity-v1` |

## Unresolved: `groundedness-v1` names two different rubrics

`groundedness.yaml` declares `version: groundedness-v1`, and so does the shipping
`judge.rubric.RUBRIC_GROUNDEDNESS`. They are not the same check:

- the **shipping** one is reference FREE. It asks whether every claim is entailed by the context
  the agent actually cited this turn.
- the **draft** here is reference BASED. It asks whether every claim matches the real, current
  facts of the case (account state, active policy). Its own description says it exists to catch
  what a reference free check "cannot see by construction".

That version string is not free to reuse. It is frozen into `contracts/trace/freeze_evidence.json`
(the v1.0.0 trace contract) and pinned by eight test modules, so a recorded verdict carrying
`groundedness-v1` is currently unambiguous about which rubric produced it. Shipping this draft
under the same string would end that.

Resolve before wiring a loader: give the draft its own version (`groundedness-v2`, or a distinct
id such as `groundedness-vs-truth-v1` if both are meant to run), rather than letting two rubrics
answer to one name. The other three versions do not collide.
