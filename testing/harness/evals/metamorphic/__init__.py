"""Metamorphic augmentation (operator lane): grow the golden set by generating meaning-preserving
paraphrases of a seed case, whose label is a RELATION checked deterministically, not a value an LLM
had to get right. That is why metamorphic data dodges the oracle problem that plagues plain synthetic
generation. Generation is non-deterministic, so it lives here; the FROZEN family
(``evals.datasets.metamorphic_golden``) replays in the hermetic gate (``test_metamorphic``). Run:
``task metamorphic``.
"""
