"""Retrieval evaluation math and helpers (the RAG testing layer).

Pure, hermetic, judge free. This subpackage is to retrieval what ``quality.stats`` is to
significance: closed form grading over labelled ground truth, computed in plain Python so it runs
under the same ``task test`` as everything else, with no model call, no network, no keys.

The deterministic IR metrics themselves (Precision/Recall@k, Hit Rate, MRR, MAP, NDCG) moved to
``quality.ir_metrics`` (SP7 task 2, the statistics toolbox relocation); ``graph_metrics`` (the graph
RAG failure modes as set arithmetic on ids: entity resolution F1, relationship/triple F1, multi hop
path recall) moved the same way to ``quality.graph_metrics`` (SP9 task 2). This subpackage keeps the
retrieval specific graders that stay beside the legacy eval lane:

- ``graph_contract``  the one shared behavioural contract every `KnowledgeGraph` adapter (in-memory,
  Postgres, Neo4j) must honour, asserted against whichever implementation is passed in.
- ``injection``    the read surface breach detector (canary / forbidden tool / exfil absence),
  Garak's ``TriggerListDetector`` mechanism reused without a model.

Reciprocal Rank Fusion (RRF) itself lives in ``atlas.domain.retrieval.rrf_fuse`` (pure domain math,
not a subpackage member here); this subpackage no longer carries its own ``fusion`` module.

Everything the LLM judge grades (faithfulness, answer relevancy, correctness vs the oracle) lives
in the OPERATOR eval lane instead, because it is not deterministic. The split is the whole point:
this package is the contract the hermetic gate can enforce.
"""
