"""Finding chunks for a query: two search arms, reciprocal rank fusion, and reranking.

Every module here ends by turning scores into ranks through `ranking.ranked_hits`, so one
rule for breaking ties governs all of them. Nothing here knows where a chunk came from or
what a correct answer is.
"""
