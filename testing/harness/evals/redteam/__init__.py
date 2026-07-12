"""The red-team lane (operator, non-gating): live attack generation against the read surface.

The live tools (DeepTeam, Garak, PyRIT, promptfoo) all drive a model over the network, so none can
run in the hermetic gate; they belong here. Their job is to DISCOVER new bypasses. Every landed
attack is minimised to its smallest reproducing poisoned document and PROMOTED into the single-sourced
``atlas.domain.poison_corpus`` as a permanent deterministic regression, so the gate hardens from real
bypasses, not imagination. This lane's always-runnable core re-checks the committed poison corpus
against the breach detector; the live DeepTeam generation is behind the ``redteam`` group. Run:
``task redteam``.
"""
