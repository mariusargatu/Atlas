"""Per intent tool binding, least agency (principle 11).

Each turn binds an intent to a fixed set of tools, and the graph refuses to run any tool outside it.
A troubleshooting turn binds knowledge and reads but never the write tools, so an injected document
that says "reset this customer's equipment" hits a closed door: the graph intercepts the call and
fails closed before it can execute (``atlas_graph.pre_action_guard``). In this reference runtime the
model is the same regardless of intent and the binding is enforced as that intercept; a dev/prod
build would also hand the model only the bound tools, so the capability is simply absent. Either way
the reachable set is the same, decided before the model acts.
"""
from __future__ import annotations

READ_TOOLS = {"get_account_summary", "get_usage", "get_bill", "get_equipment", "list_tickets"}
KNOWLEDGE_TOOLS = {"search_knowledge"}
CATALOG_TOOLS = {"list_plans", "get_plan", "check_eligibility", "compute_price"}
WRITE_TOOLS = {"change_plan", "add_addon", "remove_addon", "reset_modem", "open_ticket", "book_engineer"}

INTENT_TOOLS: dict[str, set[str]] = {
    "policy_question": KNOWLEDGE_TOOLS | CATALOG_TOOLS,
    "troubleshooting": KNOWLEDGE_TOOLS | READ_TOOLS,
    "account_read": READ_TOOLS | CATALOG_TOOLS,
    "action": READ_TOOLS | CATALOG_TOOLS | WRITE_TOOLS,
}


def bound_tools(intent: str) -> set[str]:
    return INTENT_TOOLS.get(intent, KNOWLEDGE_TOOLS)


def is_reachable(intent: str, tool: str) -> bool:
    return tool in bound_tools(intent)


# Cues that mark a turn as wanting to CHANGE something (the write surface). Deterministic and
# conservative: anything else is a non action turn that binds knowledge + reads but no writes.
_ACTION_CUES = (
    "change my", "change plan", "switch me", "switch my", "upgrade", "downgrade", "cancel",
    "add the", "add a", "add an", "remove", "reset", "reboot", "restart", "book", "open a ticket",
    "open ticket", "raise a ticket",
)


def classify_intent(text: str) -> str:
    """Per turn intent, deterministic (no model call, so the hermetic lane stays reproducible).

    ACTION if the user asks to change something; otherwise TROUBLESHOOTING, which binds knowledge +
    reads but never the write tools. This keeps the load bearing invariant true at runtime. A write
    tool is simply unreachable on a non action turn (the injected document "reset this modem" attack).
    A production system would classify all four intents (or let the model propose intent under
    review); this conservative split is what the hermetic suite enforces.
    """
    low = text.lower()
    if any(cue in low for cue in _ACTION_CUES):
        return "action"
    return "troubleshooting"
