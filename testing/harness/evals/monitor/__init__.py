"""Production monitoring (doc 11's territory), split by determinism like everything else.

Most of it is a distinct future PROD lane: online LLM-as-judge on SAMPLED live traffic, watched as a
trend against a baseline, never a red build; OTel ``gen_ai.*`` spans as the trace schema; Langfuse as
the backend; Patronus Lynx-8B (via Ollama) as the cheap local "distilled judge"; and the feedback
loop where PII-scrubbed production failures become golden rows the gate replays. That lane needs the
deferred vector adapter and real infra, so it is sketched (``__main__``), not wired.

Two seams ARE deterministic and cross back into the gate: cost/latency/token budgets and the call
budget (``budget``), exact recorded numbers a hermetic test can assert. "Trend not gate" made
concrete: CI asserts correctness on a frozen world; the monitor asks whether the world moved.
"""
