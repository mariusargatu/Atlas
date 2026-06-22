"""The gold knowledge graph: the cold open as a multi-hop relational question wearing a simple one's
clothes. A legacy customer, her plan, its data-cap term, the throttling that enforces it, and the
regional exception that also reaches the throttling clause, exactly the chain a single vector lookup
is structurally prone to fumble and a graph is good at.

Modelled the way 2WikiMultiHopQA models evidence: typed triples plus explicit reasoning paths, so
entity-resolution F1, triple F1, and multi-hop path recall are pure assertions on stable ids. Two
reasoning paths reach the throttling clause (through the cap term, and through the regional
exception), so a hop-short traversal reads below the ceiling instead of at it.
"""
from __future__ import annotations

from dataclasses import dataclass

from atlas.ports.knowledge import Chunk
from atlas.ports.knowledge_graph import Edge, Node

GRAPH_NODES: list[Node] = [
    Node("cust:legacy", "Customer", "Ada", aliases=("the customer", "account holder")),
    Node("plan:saver", "Plan", "Saver"),
    Node("term:cap", "Term", "data cap"),
    Node("term:throttle", "Term", "throttling"),
    Node("region:north", "Region", "North"),
]

GRAPH_EDGES: list[Edge] = [
    Edge("cust:legacy", "ON_PLAN", "plan:saver"),
    Edge("plan:saver", "HAS_TERM", "term:cap"),
    Edge("term:cap", "ENFORCED_BY", "term:throttle"),
    Edge("plan:saver", "MODIFIED_IN", "region:north"),
    Edge("region:north", "ADJUSTS", "term:throttle"),
]

# Entity-resolution ground truth: the surface forms that must collapse to one node.
GOLD_ENTITY_CLUSTERS: list[set[str]] = [
    {"Ada", "the customer", "account holder"},
    {"Saver"},
]

# Relationship-resolution ground truth: the exact (src, rel, dst) edge set.
GOLD_TRIPLES: frozenset[tuple[str, str, str]] = frozenset(
    (e.src, e.rel, e.dst) for e in GRAPH_EDGES
)


@dataclass(frozen=True)
class GraphCase:
    query: str
    start: str
    goal: str
    max_hops: int
    gold_paths: frozenset[tuple[str, ...]]


COLD_OPEN_CASE = GraphCase(
    query="is the customer's speed throttled",
    start="cust:legacy",
    goal="term:throttle",
    max_hops=3,
    gold_paths=frozenset({
        ("cust:legacy", "plan:saver", "term:cap", "term:throttle"),
        ("cust:legacy", "plan:saver", "region:north", "term:throttle"),
    }),
)

# An injected node the retriever can surface as data. It hangs off the throttling clause but sits on
# no legitimate reasoning path, so a traversal reads it without obeying it (the graph-poison analogue
# of the read-surface injection corpus).
POISON_NODE = Node("cmd:reset", "Command", "reset equipment", aliases=("SYSTEM",))
POISON_EDGE = Edge("term:throttle", "MENTIONS", "cmd:reset")


# --- the vector view of the same cold-open: why the graph earns its cost on the relational slice ---
#
# The answer to the cold-open needs a chain: the account holder -> her legacy Saver plan -> the
# throttling its cap is enforced by. The customer->plan link is an account relation that lives in a
# graph edge (cust:legacy ON_PLAN plan:saver), so no document co-mentions the customer and her plan.
# A keyword retriever therefore reaches the throttling chunk from a customer-worded query but can never
# reach the plan chunk, at any k, not just a small one. That missing edge, not a stingy retrieval
# budget, is exactly what a graph traversal supplies. This is its own overlap-controlled corpus (not
# the reranking slice) so the graph-vs-vector study can compare the two strategies on one question
# without a shared-fixture coupling, and the vector's miss survives the deployed k=3.
RELATIONAL_QUERY = "will account holder ada have her speed throttled"
RELATIONAL_CORPUS: list[Chunk] = [
    Chunk(chunk_id="throttle-clause", doc_id="throttle-clause", text="A capped connection is throttled to a lower speed once its allowance is spent."),
    Chunk(chunk_id="saver-plan", doc_id="saver-plan", text="The legacy Saver plan carries a monthly allowance and a minimum contract term."),
    Chunk(chunk_id="router-help", doc_id="router-help", text="Restart the router by holding its power button for ten seconds."),
    Chunk(chunk_id="billing-help", doc_id="billing-help", text="Invoices are raised monthly on the anniversary date."),
    Chunk(chunk_id="outage-help", doc_id="outage-help", text="Late fees are waived during a confirmed network outage."),
]
# Both chunks are needed to answer; only 'throttle-clause' shares vocabulary with the customer-worded
# query, so vector recall tops out at 0.5 (misses 'saver-plan') while graph traversal reaches every hop.
RELATIONAL_RELEVANT: frozenset[str] = frozenset({"throttle-clause", "saver-plan"})
