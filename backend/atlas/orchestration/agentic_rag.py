"""The agentic RAG variant (D6): a NEW, narrow LangGraph subgraph beside `atlas_graph.py`, never a
branch inside it. Query in, chunks plus an answer out (D6's "identical signatures, selected by
config"): this module carries no session/account/action surface at all, unlike the full production
graph, because HLD 4.4 is explicit that Atlas invests in this variant AS AN EVAL SUBJECT
(reproducing the "a tuned fixed pipeline matches open agentic loops at a third of the token cost"
result on this corpus), not as the recommended serving default.

Nodes, each a named conditional edge D14's deterministic trajectory checks can grade directly off
`tools_called` (this module's own analogue of `atlas_graph.AtlasState.tools_called`, the SAME "one
tally, both the runtime and the grader read it" idea, just tracking node names instead of MCP tool
names since this subgraph has no MCP tool loop to speak of):

  route_query        -- `atlas.domain.binding.classify_intent`, reused verbatim, never a second
                         heuristic. Recorded for parity with D14's intent confusion matrix column;
                         this narrow variant has no other tool surface to bind against it.
  retrieve            -- `Retriever.search_chunks` (the exact port the naive path's
                         `_knowledge_call` already calls), widened (`rerank_enabled=False`) so the
                         candidate pool the rerank step gets to work with is genuinely wide.
  rerank              -- `Reranker.rerank` (`atlas.ports.reranker`): built and tested ahead of any
                         wiring (its own module docstring says so explicitly), this is its first
                         real caller. Reusing it here, rather than deriving a rerank step inside
                         this file again, is the same "helpers the naive path already has, verbatim" reuse
                         research 01 asks for -- the naive pipeline's own two stage shape (fuse,
                         then optionally rerank) decomposed into two nodes over the SAME two ports.
  grade_documents     -- CRAG style per chunk relevance, entity_id overlap. `agentic_rag.py` cannot
                         import `testing.harness.quality.agent_metrics`
                         (`testing/tests/test_import_lint.py`'s product/harness boundary is one
                         way: harness may import backend, never the reverse), so the SAME
                         precision/recall overlap arithmetic `citation_precision_recall` already
                         applies to a whole response's citations is restated here
                         (`_entity_overlap_precision_recall`), applied per retrieval instead, and
                         the hermetic test cross checks the two on a shared fixture so they cannot
                         silently drift. `query_entity_ids` is an OPTIONAL grading input (empty by
                         default): a live caller with no golden entity linking for this query has
                         nothing to grade against, so grading is vacuously ok rather than forcing an
                         unwinnable rewrite loop on every ordinary call; the benchmark matrix (SP9
                         task 4) is the caller that actually supplies it, from the registry derived
                         golden set.
  rewrite_query       -- bounded to EXACTLY ONE retry via `atlas.domain.budget.Budget`/
                         `check_budget` (a narrower `_AGENTIC_BUDGET` instance, never a new ad hoc
                         counter): `route_after_grade` simulates the retrieve this retry would cost
                         BEFORE taking the edge, so a permanently failing grade (the tamper case)
                         still hands off to `generate` once the budget is spent, rather than
                         spinning. The rewrite itself is a fixed deterministic transform (strip a
                         leading interrogative/politeness prefix, widen `k_fused`), never a second
                         model call: CRAG's own bounded design, kept hermetic.
  generate            -- one direct `model._agenerate([...])` call (this subgraph does no tool
                         binding at all, unlike `atlas_graph.agent`, so `_generate_message`'s
                         tool binding machinery does not apply here), prompted from `query` + `chunks`
                         via `build_generate_prompt`. Reused for the regenerate pass too (the plan's
                         own node list never names a separate "regenerate" node): a prior `generate`
                         already in `tools_called` switches the prompt to the corrective wording.
  check_faithfulness  -- reference based (grounded against the retrieved chunks, this variant's own
                         reference set, RAGAS's sense of "faithfulness"), deterministic word overlap
                         proxy, never a judge. Bounded to exactly one regenerate BY CONSTRUCTION
                         (`tools_called.count("check_faithfulness")`, no separate flag needed): a
                         still unfaithful second answer ships with an appended disclosure rather
                         than a silent pass or a third attempt -- the same "never silent" doctrine
                         the degradation ladder and the contract narrowing rules already apply
                         elsewhere in this repo.

Touches NO backend/atlas/domain except reusing `atlas.domain.budget` (`Budget`/`check_budget`) and
`atlas.domain.binding.classify_intent` (both imported, neither modified) and
`atlas.domain.retrieval.RetrievalConfig` (the existing retrieval knob shape, likewise unmodified).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from atlas.adapters.cassette_reranker import CassetteReranker
from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.binding import classify_intent
from atlas.domain.budget import Budget, check_budget
from atlas.domain.retrieval import K_FINAL, K_FUSED, RetrievalConfig
from atlas.ports.knowledge import Chunk, Retriever
from atlas.ports.reranker import Reranker

# The retrieval widths come from `domain.retrieval` (K_FUSED/K_FINAL), the one declaration
# `knowledge_server.DEPLOYED_K` and the two sibling variants also read. This module used to declare
# its own copy under a comment asserting it matched them.

# The narrower Budget instance the digest calls for ("a narrower Budget instance, not a new ad hoc
# counter"): `max_retrieval_rounds` is the initial retrieve plus EXACTLY one rewrite triggered
# retry. `route_after_grade` passes `check_budget` only the RETRIEVAL subsequence of
# `tools_called`, so `max_tool_calls` and `max_retrieval_rounds` police the identical quantity and
# both come from `_MAX_RETRIEVAL_ROUNDS`. That replaces a hand counted `max_tool_calls=12` (the
# longest node sequence this graph could ever take, recounted by a human on every node added) which
# was structurally incapable of binding: the two reachable sequences are 5 and 9 nodes long, and
# only `max_retrieval_rounds` ever decided anything.
_MAX_RETRIEVAL_ROUNDS = 2
_AGENTIC_BUDGET = Budget(max_tool_calls=_MAX_RETRIEVAL_ROUNDS, max_retrieval_rounds=_MAX_RETRIEVAL_ROUNDS)
_RETRIEVAL_TOOLS = frozenset({"retrieve"})

# A cue word list for the deterministic rewrite (never a second model call): stripping a leading
# interrogative/politeness prefix is the one lever CRAG's own bounded retry gets here.
_LEADING_WORDS = frozenset(
    {"what", "why", "how", "can", "could", "would", "please", "is", "are", "do", "does", "tell"}
)

_STRIP_CHARS = ".,!?;:\"'()"
# A word must be at least this long to count as "substantive" for the faithfulness proxy (drops
# "it", "by", "the", ... which would otherwise trivially "match" any reference text).
_SIGNIFICANT_WORD_MIN_LEN = 4
# The fraction of the answer's substantive words that must appear in the retrieved chunks' own text
# for the answer to be graded faithful. A proxy, not a judge: exact wording reuse passes, a
# hallucinated fact (introducing words absent from every retrieved passage) fails.
_FAITHFULNESS_MIN_COVERAGE = 0.6

_DISCLOSURE_SUFFIX = " I could not fully verify this answer against the retrieved sources."

_INITIAL_INSTRUCTION = "Answer using ONLY the passages below; do not add facts the passages do not contain."
_CORRECTIVE_INSTRUCTION = (
    "Your previous answer was not grounded in the passages below. Answer again using ONLY facts "
    "the passages below actually contain."
)


class AgenticRagState(TypedDict):
    query: str
    query_entity_ids: Optional[tuple[str, ...]]  # OPTIONAL grading ground truth; () = ungraded live query
    k_fused: Optional[int]
    chunks: Optional[list[Chunk]]
    intent: Optional[str]
    graded_ok: Optional[bool]
    answer: Optional[str]
    faithful: Optional[bool]
    disclosed: Optional[bool]
    tools_called: Optional[tuple[str, ...]]  # every node visited, in order: the D14 trajectory tally


def _rewrite_query(query: str) -> str:
    """Deterministic query rewrite, never a second model call: drop a leading
    interrogative/politeness prefix and collapse whitespace, so a retry issues a leaner keyword
    string against the SAME retrieve/rerank helpers. Bounded to exactly one attempt by the graph's
    own edges (`route_after_grade`), never by this function."""
    words = query.strip().split()
    while words and words[0].strip(_STRIP_CHARS).lower() in _LEADING_WORDS:
        words = words[1:]
    rewritten = " ".join(words)
    return rewritten if rewritten else query


def _entity_overlap_precision_recall(cited: frozenset[str], expected: frozenset[str]) -> tuple[float, float]:
    """The exact arithmetic `testing.harness.quality.agent_metrics.citation_precision_recall`
    already applies to a whole response's citations (hits = cited & expected; precision =
    hits/len(cited); recall = hits/len(expected)), restated here because this module cannot import
    that harness module (see the module docstring). The hermetic test cross checks the two directly
    on a shared fixture so the formulas can never silently drift apart."""
    hits = len(cited & expected)
    precision = hits / len(cited) if cited else 0.0
    recall = hits / len(expected) if expected else 0.0
    return precision, recall


def build_generate_prompt(query: str, chunks: Sequence[Chunk], *, corrective: bool) -> str:
    """The one prompt building function both the initial `generate` call and the regenerate pass
    use (the corrective wording is the only difference), so a hermetic test can seed a REPLAY
    cassette keyed on the exact same string the graph will construct."""
    context = "\n".join(f"- {c.text}" for c in chunks)
    instruction = _CORRECTIVE_INSTRUCTION if corrective else _INITIAL_INSTRUCTION
    return f"{instruction}\n\nQuestion: {query}\n\nPassages:\n{context}"


def _is_faithful(answer: str, chunks: Sequence[Chunk]) -> bool:
    """Reference based (grounded against the retrieved chunks), deterministic, no judge: every
    substantive word (`_SIGNIFICANT_WORD_MIN_LEN`+ chars) the answer asserts must appear somewhere
    in the retrieved chunks' own text, at or above `_FAITHFULNESS_MIN_COVERAGE`. A paraphrase that
    reuses the source's own vocabulary passes; a hallucinated fact that introduces vocabulary absent
    from every retrieved passage fails. An answer with no substantive words is vacuously faithful
    (nothing asserted, nothing to contradict the reference)."""
    reference_words = {w.strip(_STRIP_CHARS) for c in chunks for w in c.text.lower().split()}
    answer_words = {w.strip(_STRIP_CHARS) for w in answer.lower().split()}
    significant = {w for w in answer_words if len(w) >= _SIGNIFICANT_WORD_MIN_LEN}
    if not significant:
        return True
    grounded = sum(1 for w in significant if w in reference_words)
    return (grounded / len(significant)) >= _FAITHFULNESS_MIN_COVERAGE


def build_agentic_rag_graph(
    model: BaseChatModel,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    checkpointer=None,
):
    """`retriever` defaults to `InMemoryRetriever()` (the hermetic CI adapter, mirroring
    `build_atlas_graph`'s own `retriever or InMemoryRetriever()` fallback). `reranker` defaults to
    `CassetteReranker({})` (an empty score table: a stable reorder with no effect, deterministic and
    keyless) -- the FIRST wiring of the previously unwired `Reranker` port anywhere in this repo.
    `checkpointer` is optional: this subgraph is one query in, one answer out, with no cross turn
    memory and no `interrupt()`, so a caller that omits it gets an uncheckpointed compiled graph,
    fine for a single `ainvoke`."""
    retriever = retriever or InMemoryRetriever()
    reranker = reranker or CassetteReranker({})

    def route_query(state: AgenticRagState) -> dict:
        intent = classify_intent(state["query"])
        return {"intent": intent, "tools_called": (state.get("tools_called") or ()) + ("route_query",)}

    def retrieve(state: AgenticRagState) -> dict:
        k_fused = state.get("k_fused") or K_FUSED
        chunks = retriever.search_chunks(
            state["query"], k=k_fused, config=RetrievalConfig(rerank_enabled=False, k_fused=k_fused, k_final=k_fused)
        )
        return {
            "chunks": chunks, "k_fused": k_fused,
            "tools_called": (state.get("tools_called") or ()) + ("retrieve",),
        }

    def rerank(state: AgenticRagState) -> dict:
        reranked = reranker.rerank(state["query"], state["chunks"])[:K_FINAL]
        return {"chunks": reranked, "tools_called": (state.get("tools_called") or ()) + ("rerank",)}

    def grade_documents(state: AgenticRagState) -> dict:
        query_entity_ids = frozenset(state.get("query_entity_ids") or ())
        if not query_entity_ids:
            graded_ok = True  # no ground truth to grade against: pass through, never a forced loop
        else:
            cited = frozenset(eid for c in state["chunks"] for eid in c.entity_ids)
            _, recall = _entity_overlap_precision_recall(cited, query_entity_ids)
            graded_ok = recall > 0.0
        return {
            "graded_ok": graded_ok,
            "tools_called": (state.get("tools_called") or ()) + ("grade_documents",),
        }

    def route_after_grade(state: AgenticRagState) -> str:
        if state.get("graded_ok"):
            return "generate"
        # simulate the retry's own retrieve BEFORE taking it: a permanently failing grade (the
        # tamper case) must still hand off to generate once the budget is spent, never spin. Only
        # the retrieval subsequence is handed to `check_budget`, so both of `_AGENTIC_BUDGET`'s
        # limits measure retrieve rounds and neither needs a hand counted node total.
        retrieved_so_far = tuple(t for t in (state.get("tools_called") or ()) if t in _RETRIEVAL_TOOLS)
        simulated = retrieved_so_far + ("retrieve",)
        report = check_budget(simulated, _AGENTIC_BUDGET, retrieval_tools=_RETRIEVAL_TOOLS)
        return "rewrite_query" if report.ok else "generate"

    def rewrite_query(state: AgenticRagState) -> dict:
        rewritten = _rewrite_query(state["query"])
        return {
            "query": rewritten,
            "k_fused": (state.get("k_fused") or K_FUSED) * 2,
            "tools_called": (state.get("tools_called") or ()) + ("rewrite_query",),
        }

    async def generate(state: AgenticRagState) -> dict:
        corrective = (state.get("tools_called") or ()).count("generate") >= 1
        prompt = build_generate_prompt(state["query"], state["chunks"], corrective=corrective)
        result = await model._agenerate([HumanMessage(prompt)])
        message = result.generations[0].message
        text = message.content if isinstance(message.content, str) else str(message.content)
        return {"answer": text, "tools_called": (state.get("tools_called") or ()) + ("generate",)}

    def check_faithfulness(state: AgenticRagState) -> dict:
        faithful = _is_faithful(state["answer"], state["chunks"])
        tools_called = (state.get("tools_called") or ()) + ("check_faithfulness",)
        attempts = tools_called.count("check_faithfulness")
        update: dict = {"faithful": faithful, "tools_called": tools_called}
        if not faithful and attempts >= 2:
            # the final attempt: never a silent ship, never a third generate -- disclose instead.
            update["answer"] = state["answer"] + _DISCLOSURE_SUFFIX
            update["disclosed"] = True
        return update

    def route_after_faithfulness(state: AgenticRagState) -> str:
        if state.get("faithful"):
            return "end"
        attempts = (state.get("tools_called") or ()).count("check_faithfulness")
        return "generate" if attempts < 2 else "end"

    g = StateGraph(AgenticRagState)
    g.add_node("route_query", route_query)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("check_faithfulness", check_faithfulness)
    g.add_edge(START, "route_query")
    g.add_edge("route_query", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade_documents")
    g.add_conditional_edges(
        "grade_documents", route_after_grade, {"rewrite_query": "rewrite_query", "generate": "generate"}
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "check_faithfulness")
    g.add_conditional_edges("check_faithfulness", route_after_faithfulness, {"generate": "generate", "end": END})
    return g.compile(checkpointer=checkpointer)
