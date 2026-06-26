# Judge calibration study (before / after one documented correction)

The same cross-family judge over the same human-labelled set, under two rubrics.
The only change is the rubric: V1 scores helpfulness (truth-blind), V2 scores truth
against the account. Every verdict is served from a committed cassette in REPLAY.

## Before — naive helpfulness rubric (the lying judge)
```
judge contract: gpt-judge / naive-helpfulness-v1 / tmpl:f85088a8 (fp:79b52847)
n=14  raw agreement=64%  Cohen's kappa=0.29  bar=0.60  -> NOT licensed, keep manual / fix the rubric
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
```

## After — account-truth rubric (the documented correction)
```
judge contract: gpt-judge / account-truth-v2 / tmpl:09b13f9a (fp:1537df6f)
n=14  raw agreement=93%  Cohen's kappa=0.85  bar=0.60  -> LICENSED to automate
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
```

## Position-bias probe (order-swap)

flip rate = 50% (one of two pairs flipped when the order was swapped); a flipped verdict is recorded as a tie and a flag, not a preference.

## The headline

- before: Cohen's κ = **0.29** → NOT licensed (bar 0.60)
- after:  Cohen's κ = **0.85** → LICENSED (bar 0.60)

The naive judge's raw agreement already looked respectable at 64%; chance-corrected agreement is what exposed it (κ 0.29, barely above chance). After the fix both rise, but kappa is the honest measure, the one a judge cannot fool by passing everything.
A judge you have not checked against a known reference is a vibe with a decimal point.
