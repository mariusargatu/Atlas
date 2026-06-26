"""The SP8 judge: a versioned, fingerprinted instrument that grades one answer for groundedness
against its cited retrieved context, the calibration report and kappa deployment gate that check it,
and the trace boundary that carries its verdict out.

Absorbed from the pre rewrite `evals/judge/` tree, file by file:

- `contract`      `JudgeContract`/`fingerprint()`, kept verbatim (D15's own identity rule: model
  snapshot id, prompt hash, rubric version; any change voids a prior calibration). Task 1.
- `rubric`        the `Rubric`/`template_hash`/`prompt`/`compare_prompt` scaffolding, kept; the
  CONTENT is fresh (a binary groundedness rubric, D15), never the pre rewrite `RUBRIC_V1`/`V2`
  helpfulness/account truth pair. Task 1.
- `llm_judge`     `judge_label`/`order_swap`/`_parse_label`'s parsing mechanics, kept; the old
  `LlmJudgeGrader` (the retired `evalkit` `Composite`/`GradeContext` wiring) is not. `_parse_label`
  reads the first STANDALONE token (hardened in task 2: a naive prefix match misparsed "PASSABLE" as
  PASS). Task 1, hardened task 2.
- `emission`      the one place a computed verdict crosses the trace boundary, opening a
  `kind="judge"` span (`atlas.judge.*`, span_kind `EVALUATOR`). Task 1.
- `calibration`   `AgreementRow`/`CalibrationReport`/`calibrate()`, kept, with the digest's own named
  fix: the kappa lower bound gate routes through `quality.gate.gate_on_lower_bound`, never a hand
  rolled `kappa_ci[1] >= bar` copy. The report also carries AC1 and prevalence alongside kappa (D15).
  Task 2.
- `panel`         `PanelVote`/`panel_vote`, kept verbatim (D15's jury, ties fail closed). SP9's
  benchmark matrix runner is the caller that invokes it in a headline benchmark context; this module
  is the mechanism only. Task 2.
- `provisional`    the manufactured failure generator (registry contradictions, ground truth by
  construction), registry truth agreement, and judge vs judge kappa: two PROVISIONAL signals,
  neither one ever licensing a deployment (KAPPA HONESTY, this module's own binding discipline).
  Task 3.
- `live_provisional`  the live sweep entrypoint (`task judge-live`), rewritten against
  `provisional`'s manufactured cases and the groundedness rubric; retires
  `evals/judge/live_calibration.py`'s RUBRIC_V4/account facts content. Task 3.
- `promotion`      the taxonomy gated promotion loop (D34): reads judge fail spans and end user
  thumbs down feedback, joins each against a turn's own question/answer content, and promotes only
  the ones carrying a known `contracts/dataset/taxonomy.yaml` code into `origin: promoted` dataset
  cases. No pre rewrite equivalent (the old tree had no failure taxonomy at all). Task 5.

The pre rewrite `testing/harness/evals/judge/` tree is now fully retired: `contract.py`/
`rubric.py`/`llm_judge.py` (absorbed task 1), `calibration.py`/`panel.py`/`__main__.py` (absorbed
or retired task 2), and `live_calibration.py` plus its own `artifacts/live_calibration/` snapshots
(retired task 3, its sweep and settle SHAPE absorbed into `live_provisional`, its RUBRIC_V4/account
facts content discarded for `provisional`'s registry truth and judge vs judge design) are all
deleted, not left as a second, stale definition of the same names. `evals/datasets/
judge_calibration.py`'s `CALIBRATION` fixture (naive/corrected labels for the discarded
RUBRIC_V1/V2) is deleted alongside it, superseded by `provisional.manufactured_cases`.
"""
from __future__ import annotations
