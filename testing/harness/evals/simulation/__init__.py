"""Agent simulation: the failures that live between turns, not inside one.

A static golden case is a snapshot of one turn, and the failures that matter build across a
conversation: a customer who changes her mind and an agent that loses track of which plan it was
proposing, a polite request that curdles into a frustrated one and an agent that agrees to the wrong
thing four turns later.

Three agents, each in its own role: a persona simulator drives the traffic, the system under test
(Atlas on its graph) responds, and a separate calibrated evaluator scores, because an agent grading
its own conversation grades itself generously. The simulation is nondeterministic, so it is recorded
once and replayed as a deterministic fixture in the PR lane (ADR-019); the live loop, and the
multi-trial variance runs, are the operator lane.

- ``personas``  the roster (disposition + goal), the generators the simulator plays.
- ``driver``    replays a recorded ``Conversation`` through the real ``atlas_graph`` and collects the
  actions it actually took across all turns.
- ``grade``     the multi-turn assertions: exactly one action, on the settled intent, the one the
  customer landed on and not one she walked back.
- ``model``     the shapes ``driver`` produces and ``grade`` reads.
"""
