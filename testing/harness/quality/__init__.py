"""The shared quality module: statistics, gate, deterministic IR metrics, and graph-RAG metrics.

Absorbed by relocation (SP7 task 2) from the pre rewrite `evals` tree, where `stats.py` and
`retrieval/ir_metrics.py` already lived pure stdlib and property tested, and `gate.py` sat beside
them as the release discipline both draw on. `graph_metrics.py` joined the same way (SP9 task 2),
relocated from `evals/retrieval/graph_metrics.py`. Content and public API are unchanged by either
move; only the import path is. Pure stdlib throughout, no numpy, no scipy, no network, so it runs
under the same hermetic `task test` as everything else.

- `stats`         confidence intervals (Wilson, mean, bootstrap, BCa, cluster), Cohen's kappa, paired
  significance tests (bootstrap diff, permutation, McNemar), multiple comparison correction (Holm
  Bonferroni), and power sizing (`required_n` / `detectable_effect`).
- `ir_metrics`    deterministic information retrieval metrics (precision/recall@k, hit rate, MRR, MAP,
  NDCG) over a ranked list of chunk ids and a set of relevant ids, the trec_eval/BEIR convention.
- `gate`          release gating on the honest lower bound of an interval, never the point, with a
  variance budget quarantine.
- `graph_metrics` graph-RAG failure modes as set arithmetic on ids: entity resolution F1
  (pairwise + B-Cubed), relationship/triple F1, and multi-hop path recall. SP9's own gold-vs-
  extracted graph comparison (`testing/harness/evals/graphrag/__main__.py`) scores through here.

The legacy `evals/` tree (evalkit, benchmark, judge, mutation, graphrag, simulation, and friends)
imports this package rather than owning its own copy; that is the whole point of the relocation.
No judge, no rubric, nothing reading `atlas.judge.*`: that boundary belongs to a later sub project.
"""
from __future__ import annotations
