"""The red-team lane (operator, non-gating): live attack generation against the read surface.

The live tools (DeepTeam, Garak, PyRIT, promptfoo) all drive a model over the network, so none can
run in the hermetic gate; they belong here to discover new bypasses. Every landed attack is minimised
to its smallest reproducing poisoned document and promoted into the single-sourced
``atlas.domain.poison_corpus`` as a permanent deterministic regression. This lane's always-runnable
core re-checks the committed poison corpus against the breach detector; the live DeepTeam generation
is behind the ``redteam`` group. Run: ``task redteam``.
"""
