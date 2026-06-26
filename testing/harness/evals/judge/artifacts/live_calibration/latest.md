# Live judge-calibration probe (real models, real account facts, real cost)
Cheapest tier first per provider; settling on the first that clears the 0.6 bar.

# openai

## openai:gpt-5.6-luna  ($1.00 / $6.00 per 1M tokens, in/out)
judge contract: openai:gpt-5.6-luna / account-truth-v3-live-facts / tmpl:8e5602bd (fp:8907d4b7)
n=14  raw agreement=86%  Cohen's kappa=0.72  bar=0.60  -> LICENSED to automate
  ok  cap-correct-current          human=1 judge=1
  MISS bill-correct-current         human=1 judge=0
  ok  usage-over-cap-correct       human=1 judge=1
  ok  plan-explained-correct       human=1 judge=1
  MISS addon-correct-current        human=1 judge=0
  ok  garbled-false-cap            human=0 judge=0
  ok  evasive-nonanswer            human=0 judge=0
  ok  rambling-wrong-fee           human=0 judge=0
  ok  confused-wrong-usage         human=0 judge=0
  ok  cold-open-contract-free      human=0 judge=0
  ok  cap-uncapped-legacy          human=0 judge=0
  ok  correct-handoff              human=1 judge=1
  ok  correct-scope-refusal        human=1 judge=1
  ok  terse-true-cap               human=1 judge=1

## openai:gpt-5.6-terra  ($2.50 / $15.00 per 1M tokens, in/out)
judge contract: openai:gpt-5.6-terra / account-truth-v3-live-facts / tmpl:8e5602bd (fp:a0b6a277)
n=14  raw agreement=71%  Cohen's kappa=0.44  bar=0.60  -> NOT licensed, keep manual / fix the rubric
  ok  cap-correct-current          human=1 judge=1
  MISS bill-correct-current         human=1 judge=0
  ok  usage-over-cap-correct       human=1 judge=1
  ok  plan-explained-correct       human=1 judge=1
  MISS addon-correct-current        human=1 judge=0
  ok  garbled-false-cap            human=0 judge=0
  ok  evasive-nonanswer            human=0 judge=0
  ok  rambling-wrong-fee           human=0 judge=0
  MISS confused-wrong-usage         human=0 judge=1
  ok  cold-open-contract-free      human=0 judge=0
  ok  cap-uncapped-legacy          human=0 judge=0
  MISS correct-handoff              human=1 judge=0
  ok  correct-scope-refusal        human=1 judge=1
  ok  terse-true-cap               human=1 judge=1

## openai:gpt-5.6-sol  ($5.00 / $30.00 per 1M tokens, in/out)
judge contract: openai:gpt-5.6-sol / account-truth-v3-live-facts / tmpl:8e5602bd (fp:c8a4b76d)
n=14  raw agreement=79%  Cohen's kappa=0.57  bar=0.60  -> NOT licensed, keep manual / fix the rubric
  ok  cap-correct-current          human=1 judge=1
  MISS bill-correct-current         human=1 judge=0
  ok  usage-over-cap-correct       human=1 judge=1
  ok  plan-explained-correct       human=1 judge=1
  MISS addon-correct-current        human=1 judge=0
  ok  garbled-false-cap            human=0 judge=0
  ok  evasive-nonanswer            human=0 judge=0
  ok  rambling-wrong-fee           human=0 judge=0
  MISS confused-wrong-usage         human=0 judge=1
  ok  cold-open-contract-free      human=0 judge=0
  ok  cap-uncapped-legacy          human=0 judge=0
  ok  correct-handoff              human=1 judge=1
  ok  correct-scope-refusal        human=1 judge=1
  ok  terse-true-cap               human=1 judge=1

# anthropic

## anthropic:claude-haiku-4-5-20251001  ($1.00 / $5.00 per 1M tokens, in/out)
judge contract: anthropic:claude-haiku-4-5-20251001 / account-truth-v3-live-facts / tmpl:8e5602bd (fp:3bba201a)
n=14  raw agreement=93%  Cohen's kappa=0.86  bar=0.60  -> LICENSED to automate
  ok  cap-correct-current          human=1 judge=1
  ok  bill-correct-current         human=1 judge=1
  ok  usage-over-cap-correct       human=1 judge=1
  ok  plan-explained-correct       human=1 judge=1
  MISS addon-correct-current        human=1 judge=0
  ok  garbled-false-cap            human=0 judge=0
  ok  evasive-nonanswer            human=0 judge=0
  ok  rambling-wrong-fee           human=0 judge=0
  ok  confused-wrong-usage         human=0 judge=0
  ok  cold-open-contract-free      human=0 judge=0
  ok  cap-uncapped-legacy          human=0 judge=0
  ok  correct-handoff              human=1 judge=1
  ok  correct-scope-refusal        human=1 judge=1
  ok  terse-true-cap               human=1 judge=1

## anthropic:claude-sonnet-5  ($2.00 / $10.00 per 1M tokens, in/out)
judge contract: anthropic:claude-sonnet-5 / account-truth-v3-live-facts / tmpl:8e5602bd (fp:4761b0b7)
n=14  raw agreement=79%  Cohen's kappa=0.57  bar=0.60  -> NOT licensed, keep manual / fix the rubric
  ok  cap-correct-current          human=1 judge=1
  MISS bill-correct-current         human=1 judge=0
  ok  usage-over-cap-correct       human=1 judge=1
  ok  plan-explained-correct       human=1 judge=1
  MISS addon-correct-current        human=1 judge=0
  ok  garbled-false-cap            human=0 judge=0
  MISS evasive-nonanswer            human=0 judge=1
  ok  rambling-wrong-fee           human=0 judge=0
  ok  confused-wrong-usage         human=0 judge=0
  ok  cold-open-contract-free      human=0 judge=0
  ok  cap-uncapped-legacy          human=0 judge=0
  ok  correct-handoff              human=1 judge=1
  ok  correct-scope-refusal        human=1 judge=1
  ok  terse-true-cap               human=1 judge=1

## anthropic:claude-opus-4-8  ($5.00 / $25.00 per 1M tokens, in/out)
judge contract: anthropic:claude-opus-4-8 / account-truth-v3-live-facts / tmpl:8e5602bd (fp:f47cc677)
n=14  raw agreement=86%  Cohen's kappa=0.72  bar=0.60  -> LICENSED to automate
  ok  cap-correct-current          human=1 judge=1
  MISS bill-correct-current         human=1 judge=0
  ok  usage-over-cap-correct       human=1 judge=1
  ok  plan-explained-correct       human=1 judge=1
  MISS addon-correct-current        human=1 judge=0
  ok  garbled-false-cap            human=0 judge=0
  ok  evasive-nonanswer            human=0 judge=0
  ok  rambling-wrong-fee           human=0 judge=0
  ok  confused-wrong-usage         human=0 judge=0
  ok  cold-open-contract-free      human=0 judge=0
  ok  cap-uncapped-legacy          human=0 judge=0
  ok  correct-handoff              human=1 judge=1
  ok  correct-scope-refusal        human=1 judge=1
  ok  terse-true-cap               human=1 judge=1

RECOMMENDATION (cross-family, safe to deploy as the judge): openai:gpt-5.6-luna is the cheapest tier that clears the automation bar live (kappa=0.72 >= 0.60).
FOR COMPARISON ONLY (same-family as the agent, reintroduces self-enhancement bias risk per ADR-004): anthropic:claude-haiku-4-5-20251001 clears the bar too (kappa=0.86).

# Panel: openai:gpt-5.6-luna  +  anthropic:claude-haiku-4-5-20251001

1/14 cases split between the two judges.
  SPLIT bill-correct-current         votes=(0, 1) -> majority=0 (tie fails closed to 0)