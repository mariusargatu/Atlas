"""The labelled retrieval slice: the dataset article's golden set, sliced for retrieval.

Each case pairs a question with the chunk ids a correct answer must draw on, so retrieval quality
is measured before any answer exists (doc 07, "retrieval first"). The corpus is a small broadband
support KB with STABLE ids, which is what lets the gate use exact id membership as the relevance
primitive: id based recall is *exactly* deterministic, not approximately (RAGAS IDBasedContextRecall
pattern), so no embedding, no string-similarity threshold, no float drift enters the hermetic lane.

It is deliberately its own corpus, passed to ``InMemoryRetriever(corpus=...)``, so growing or
relabelling the retrieval slice never perturbs the cold-open corpus in ``atlas.domain.corpus`` that
the knowledge/security tests pin. Two of the queries carry more than one relevant chunk, so recall,
MAP and NDCG are exercised below a perfect score and not only at the ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass

from atlas.ports.knowledge import Chunk

# A broadband-support KB. Ids are stable and are the ground truth every metric is computed against.
RETRIEVAL_CORPUS: list[Chunk] = [
    Chunk("plan-current", "The current Fast plan is unlimited with no data cap and no contract."),
    Chunk("plan-legacy", "The legacy Saver plan has a monthly data cap and a twelve month contract term."),
    Chunk("throttling-terms", "When a capped plan exceeds its data cap the connection is throttled to a lower speed."),
    Chunk("outage-latefee", "During a confirmed network outage late fees are waived for affected customers."),
    Chunk("router-reset", "If the router light blinks orange restart it by holding the power button for ten seconds."),
    Chunk("coverage-regional", "Regional coverage exceptions can modify plan terms in specific postcode areas."),
    Chunk("billing-cycle", "Bills are issued monthly on the account anniversary date and list each line item."),
]


@dataclass(frozen=True)
class RetrievalCase:
    """A query paired with the ids a correct retrieval must surface. Frozen and hashable."""

    query: str
    relevant_ids: frozenset[str]


RETRIEVAL_GOLDEN: list[RetrievalCase] = [
    RetrievalCase("is the current plan unlimited", frozenset({"plan-current"})),
    RetrievalCase("does the legacy plan have a data cap", frozenset({"plan-legacy"})),
    RetrievalCase("how do I restart my router", frozenset({"router-reset"})),
    RetrievalCase("are late fees waived during an outage", frozenset({"outage-latefee"})),
    RetrievalCase("when are bills issued", frozenset({"billing-cycle"})),
    # multi-hop-ish: the relational cold-open shape, two relevant chunks (cap term + its throttling)
    RetrievalCase("what happens when I exceed my data cap", frozenset({"throttling-terms", "plan-legacy"})),
    RetrievalCase("capped plan throttled data", frozenset({"throttling-terms", "plan-legacy"})),
]
