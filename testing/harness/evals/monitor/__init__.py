"""Production monitoring: the deterministic slice that gates the hermetic lane.

``budget`` checks the call budget against recorded numbers. ``sampling`` builds the human-review
queue, flagged sessions first, then a seeded random fill. ``feedback`` scrubs a flagged production
failure and promotes it to a silver GoldenCase; ``promote`` refuses structured PII and any
caller-supplied name, not free-text DLP.

The online half (trace capture, sampled judge scoring, triage) is deliberately deferred; see
README, Scope & status.
"""
