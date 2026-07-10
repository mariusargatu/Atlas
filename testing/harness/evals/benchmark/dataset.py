"""The committed paired outcome set for the worked example: v_a vs v_b on the answer golden.

Two model versions scored on the same 100 items, so the comparison is paired. Rather than
assert the headline numbers on a slide, the outcomes are pinned here as a 2x2 contingency of
the paired result, then expanded into per item 0/1 vectors the real stats functions consume.
This is a committed fixture, not a live recording of two versions: standing up two recorded
model versions over a golden set of 100 items is significant operational effort, and the point
here is the *reasoning*, which the fixture exercises end to end. The marginals match the
cold open exactly, v_a 84/100 and v_b 81/100, and the discordant split is what the paired
test actually reads.

  +-----------------+-------------+-------------+
  |                 |  v_b pass   |  v_b fail   |
  +-----------------+-------------+-------------+
  |  v_a pass       |    77       |     7       |   -> v_a passes 84
  |  v_a fail       |     4       |    12       |   -> v_a fails  16
  +-----------------+-------------+-------------+
       v_b passes 81      v_b fails 19      n = 100

The concordant cells (77 both pass, 12 both fail) carry no signal about which version is
better. The 7 + 4 = 11 discordant pairs are the whole basis for the verdict, and they split
7 to 4, a net of three items out of a hundred. That is the gap the cold open mistook for a
regression.
"""
from __future__ import annotations

# The paired 2x2: (v_a result, v_b result) -> item count.
BOTH_PASS = 77
A_PASS_B_FAIL = 7  # the "b" discordant cell McNemar reads
A_FAIL_B_PASS = 4  # the "c" discordant cell McNemar reads
BOTH_FAIL = 12

N = BOTH_PASS + A_PASS_B_FAIL + A_FAIL_B_PASS + BOTH_FAIL  # 100

# The bootstrap/permutation seed, stamped into the artifact's provenance so the interval
# recomputes identically in CI. An interval whose seed is not recorded is weather, not a number.
SEED = 0xBEAC04


def paired_vectors() -> tuple[list[int], list[int]]:
    """Expand the contingency into two aligned per item 0/1 vectors (v_a, v_b), fixed order.

    Index i of each vector is the same golden item, so the pairing is positional and the
    paired tests read item by item differences directly off the two lists.
    """
    a = [1] * BOTH_PASS + [1] * A_PASS_B_FAIL + [0] * A_FAIL_B_PASS + [0] * BOTH_FAIL
    b = [1] * BOTH_PASS + [0] * A_PASS_B_FAIL + [1] * A_FAIL_B_PASS + [0] * BOTH_FAIL
    return a, b


__all__ = [
    "A_FAIL_B_PASS",
    "A_PASS_B_FAIL",
    "BOTH_FAIL",
    "BOTH_PASS",
    "N",
    "SEED",
    "paired_vectors",
]
