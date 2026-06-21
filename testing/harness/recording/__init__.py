"""Recording: capture new cassettes against a live model (operator-run, not the PR lane).

These scripts call a real provider, so they need API keys (or a local Ollama) and a
network. They are how the committed ../cassettes/ data is (re)generated; the hermetic
PR lane only ever REPLAYS what they produced. Run them deliberately, never in CI.

  record_turn.py             Prove record -> replay end to end against a live provider.
  record_atlas_cassettes.py  Record the real Atlas cold-open cassette (replayed by
                             test_recorded_turns.py).
  seed_e2e_cassettes.py      Regenerate the cassettes the Playwright E2E lane replays.
"""
