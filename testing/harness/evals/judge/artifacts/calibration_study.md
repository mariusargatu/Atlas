# Judge calibration study (before / after one documented correction)

The same cross-family judge over the same human-labelled set, under two rubrics.
The only change is the rubric: V1 scores helpfulness (truth-blind), V2 scores truth
against the account. Every verdict is served from a committed cassette in REPLAY.

## Before — naive helpfulness rubric (the lying judge)
```
judge contract: gpt-judge / naive-helpfulness-v1 / tmpl:f85088a8 (fp:79b52847)
n=28  raw agreement=61%  Cohen's kappa=0.21 95% CI [-0.15, 0.58]  bar=0.60  -> NOT licensed, keep manual / fix the rubric
  ok  cap-correct-current          human=1 judge=1
  ok  bill-correct-current         human=1 judge=1
  ok  usage-over-cap-correct       human=1 judge=1
  ok  plan-explained-correct       human=1 judge=1
  ok  addon-correct-current        human=1 judge=1
  ok  garbled-false-cap            human=0 judge=0
  ok  evasive-nonanswer            human=0 judge=0
  ok  rambling-wrong-fee           human=0 judge=0
  ok  confused-wrong-usage         human=0 judge=0
  MISS cold-open-contract-free      human=0 judge=1
  MISS cap-uncapped-legacy          human=0 judge=1
  MISS correct-handoff              human=1 judge=0
  MISS correct-scope-refusal        human=1 judge=0
  MISS terse-true-cap               human=1 judge=0
  ok  price-correct-legacy         human=1 judge=1
  ok  term-correct-current         human=1 judge=1
  ok  bill-due-correct-legacy      human=1 judge=1
  ok  addon-none-correct-legacy    human=1 judge=1
  ok  garbled-wrong-price          human=0 judge=0
  ok  vague-wrong-bill             human=0 judge=0
  ok  garbled-wrong-addon          human=0 judge=0
  MISS legacy-no-fee-false          human=0 judge=1
  MISS legacy-uncapped-false        human=0 judge=1
  MISS current-overcap-false        human=0 judge=1
  MISS terse-true-price             human=1 judge=0
  MISS scope-refusal-bill           human=1 judge=0
  MISS handoff-contracted-cancel    human=1 judge=0
  ok  rambling-wrong-cap           human=0 judge=0
```

## After — account-truth rubric (the documented correction)
```
judge contract: gpt-judge / account-truth-v2 / tmpl:09b13f9a (fp:1537df6f)
n=28  raw agreement=93%  Cohen's kappa=0.85 95% CI [0.66, 1.00]  bar=0.60  -> LICENSED to automate
  ok  cap-correct-current          human=1 judge=1
  ok  bill-correct-current         human=1 judge=1
  ok  usage-over-cap-correct       human=1 judge=1
  ok  plan-explained-correct       human=1 judge=1
  ok  addon-correct-current        human=1 judge=1
  ok  garbled-false-cap            human=0 judge=0
  ok  evasive-nonanswer            human=0 judge=0
  MISS rambling-wrong-fee           human=0 judge=1
  ok  confused-wrong-usage         human=0 judge=0
  ok  cold-open-contract-free      human=0 judge=0
  ok  cap-uncapped-legacy          human=0 judge=0
  ok  correct-handoff              human=1 judge=1
  ok  correct-scope-refusal        human=1 judge=1
  ok  terse-true-cap               human=1 judge=1
  ok  price-correct-legacy         human=1 judge=1
  ok  term-correct-current         human=1 judge=1
  ok  bill-due-correct-legacy      human=1 judge=1
  ok  addon-none-correct-legacy    human=1 judge=1
  ok  garbled-wrong-price          human=0 judge=0
  ok  vague-wrong-bill             human=0 judge=0
  ok  garbled-wrong-addon          human=0 judge=0
  ok  legacy-no-fee-false          human=0 judge=0
  ok  legacy-uncapped-false        human=0 judge=0
  ok  current-overcap-false        human=0 judge=0
  ok  terse-true-price             human=1 judge=1
  ok  scope-refusal-bill           human=1 judge=1
  ok  handoff-contracted-cancel    human=1 judge=1
  MISS rambling-wrong-cap           human=0 judge=1
```

## Position-bias probe (order-swap)

flip rate = 50% (one of two pairs flipped when the order was swapped); a flipped verdict is recorded as a tie and a flag, not a preference.

## The headline

- before: Cohen's κ = **0.21** (95% CI [-0.15, 0.58]) → NOT licensed (bar 0.60)
- after:  Cohen's κ = **0.85** (95% CI [0.66, 1.00]) → LICENSED (bar 0.60)

The naive judge's raw agreement already looked respectable at 61%; chance-corrected agreement is what exposed it (κ 0.21, barely above chance). Kappa is the honest measure, the one a judge cannot fool by passing everything.

But even the corrected judge is licensed on the FLOOR of its interval, not its point. At n=14 its κ=0.85 carried a 95% floor of ~0.59, below the 0.6 bar: a high point over an unconvinced floor, the exact optimism a release gated on its point estimate would ship. So the set is sized past that point (n=28), and the floor (0.66), not just the point, clears the bar.
A judge you have not checked against a known reference is a vibe with a decimal point.
