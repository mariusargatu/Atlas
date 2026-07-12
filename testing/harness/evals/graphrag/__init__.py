"""The graph-RAG operator lane: make the graph earn its 35-45x cost (doc 07).

Graph RAG beats plain vector retrieval only on relational / multi-hop questions and loses on simple
fact lookups, and many published win-rates evaporate under bias-corrected evaluation. So Atlas does
not adopt it on faith: this lane runs both strategies over the flat and the relational slices and
reports where the graph actually wins. The comparison is deterministic over the in-memory adapters
(runnable with no database); the real Neo4j path (``Neo4jKnowledgeGraph``) plugs into the same study
in dev/prod. Run: ``task graph``.
"""
