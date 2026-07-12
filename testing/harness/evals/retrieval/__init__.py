"""Retrieval evaluation math and helpers (the RAG testing layer of the series, doc 07).

Pure, hermetic, judge free. This subpackage is to retrieval what ``evals.stats`` is to
significance: closed form grading over labelled ground truth, computed in plain Python so it runs
under the same ``task test`` as everything else, with no model call, no network, no keys.

- ``ir_metrics``   deterministic information retrieval metrics (Precision/Recall@k, Hit Rate, MRR,
  MAP, NDCG) over a ranked list of chunk ids and a set of relevant ids.
- ``graph_metrics``  the graph RAG failure modes as set arithmetic on ids (entity resolution F1,
  relationship/triple F1, multi hop path recall, per hop Hit@k).
- ``fusion``       Reciprocal Rank Fusion (RRF), the deterministic hybrid used in the
  with/without and BM25+dense comparisons.
- ``injection``    the read surface breach detector (canary / forbidden tool / exfil absence),
  Garak's ``TriggerListDetector`` mechanism reused without a model.

Everything the LLM judge grades (faithfulness, answer relevancy, correctness vs the oracle) lives
in the OPERATOR eval lane instead, because it is not deterministic. The split is the whole point:
this package is the contract the hermetic gate can enforce.
"""
