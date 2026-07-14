# Runbook: promote a production failure to the golden set

**Trigger.** A session lands in the review queue (`task monitor` shape: flagged first, seeded
random fill) and a human confirms it is a real failure worth preventing forever.

**The invariant.** A real customer's data never enters a set engineers read and CI replays. The
failure's shape survives; the customer does not. `promote` refuses structured PII and any name you
supply — it cannot refuse a name you forgot, so the names list is part of the checklist, not an
optional argument.

## Steps

1. **Triage** (human): write down what correct behaviour would have been (`expected`), the risk,
   and the oracle that decides it. If you cannot state the oracle, the case is not ready.
2. **Record provenance** on the session before scrubbing: `captured_at` (date), `model_id` (the
   snapshot that failed), `trace_ref` (link into the trace store). A regression case that cannot
   answer "which model failed and where is the trace" is folklore.
3. **Scrub**: `scrub(session, as_customer=<seeded account>, names=[every name in the transcript])`.
   - Pick the seeded account whose shape matches the failure (a legacy-term customer for a
     legacy-term failure) so the promoted case still exercises the same oracle.
   - The names list is yours to compile by reading the transcript. The regexes cover structured
     identifiers only (UK-flavoured email/card/phone/postcode/account); free-text PII
     ("I live next to the church on Elm Street") is caught by you or not at all.
4. **Promote**: `promote(scrubbed, names=<the same list>, graders=...)` → a silver
   `GoldenCase`, `source="production"`. Silver is provisional by design.
5. **Dedup** before committing: search the golden set for the same failure shape (same category +
   risk + oracle). A recurring production failure should deepen one case, not clone N near-copies.
6. **Ratify silver → gold**: a second human reads the case, the evaluator run agrees the graders
   fire, and the case lands through code review like any other test. Golden-set edits that relax
   an expectation need their own justification in the PR.
7. **Close the loop**: if the same failure keeps arriving, stop promoting and design the risk away
   (deterministic catalog logic, tighter binding, typed confirmation) — then delete the cases the
   removed risk made necessary.
