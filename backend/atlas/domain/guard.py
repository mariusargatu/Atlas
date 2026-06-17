"""The guard, fail closed policy checks, pure domain.

The LangGraph guard nodes call these. The default is always "no". Scope keeps identity from
the session (an id the model produced is rejected). The single write rule fails closed on a
multi or mixed tool batch. Value bounds reject an argument nudged out of the catalog (the
"move me to the internal £0 plan" attack).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from atlas.domain.accounts import HARDSHIP_REASON_CATEGORIES, SEED
from atlas.domain.catalog import ADDONS, CATALOG
from atlas.domain.oracle import truth_for

# A booking slot must look like an ISO minute stamp (YYYY-MM-DDThh:mm), not free text the model invented.
_SLOT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_MAX_SUBJECT = 200

WRITE_TOOLS = {"change_plan", "add_addon", "remove_addon", "reset_modem", "open_ticket", "book_engineer", "cancel_service"}

# Cues a fluent but false "you can leave free" answer uses. The render check turns each into a
# question for the account oracle, never a verdict on the words alone.
_NO_CONTRACT_CUES = ("contract-free", "no fee", "cancel any time", "cancel anytime", "no contract")

# Executable markup the Lena failure (LLM05) smuggled through a chat reply: a tag, an event
# handler, or a javascript: URL. The model's output is untrusted. This is the escape before render
# rule from web security, enforced at the door rather than trusted to the model.
_UNSAFE_MARKUP = re.compile(r"<\s*/?\s*(script|iframe|img|svg|object|embed)\b|on\w+\s*=|javascript:", re.IGNORECASE)

# System prompt leakage (LLM07) and credential markers that must never reach a browser.
_SECRET_CUES = ("-----begin", "sk-", "bearer ", "system prompt", "you are atlas")


@dataclass
class GuardVerdict:
    ok: bool
    reason: str = ""


def check_scope(call_customer_id: str, session_customer_id: str) -> GuardVerdict:
    if call_customer_id != session_customer_id:
        return GuardVerdict(False, "customer-scope: id does not match the session")
    return GuardVerdict(True)


def check_single_write(tool_calls: list[str]) -> GuardVerdict:
    writes = [t for t in tool_calls if t in WRITE_TOOLS]
    if len(writes) > 1:
        return GuardVerdict(False, "more than one write in a turn")
    if writes and len(tool_calls) > len(writes):  # a write alongside any non write tool
        return GuardVerdict(False, "mixed read+write batch")
    return GuardVerdict(True)


def check_value_bounds(tool: str, args: dict) -> GuardVerdict:
    """Every write argument stays inside the offered catalog / a sane shape, the "move me to the
    internal £0 plan" attack and its siblings (a bogus add on, an invented engineer slot) caught
    before anything is proposed. ``reset_modem`` takes no argument, so it is bounded by construction.
    """
    if tool == "change_plan" and args.get("plan_id") not in CATALOG:
        return GuardVerdict(False, f"plan {args.get('plan_id')!r} is not a real, offered plan")
    if tool in ("add_addon", "remove_addon") and args.get("addon_id") not in ADDONS:
        return GuardVerdict(False, f"add-on {args.get('addon_id')!r} is not a real, offered add-on")
    if tool == "open_ticket":
        subject = (args.get("subject") or "").strip()
        if not subject:
            return GuardVerdict(False, "a ticket needs a subject")
        if len(subject) > _MAX_SUBJECT:
            return GuardVerdict(False, "ticket subject is too long")
    if tool == "book_engineer" and not _SLOT.match(args.get("slot") or ""):
        return GuardVerdict(False, f"slot {args.get('slot')!r} is not a valid booking time")
    if tool == "cancel_service":
        reason = args.get("reason_category", "none")
        if reason not in HARDSHIP_REASON_CATEGORIES | {"none"}:
            return GuardVerdict(False, f"cancel_service reason_category {reason!r} is not a recognised value")
        return GuardVerdict(True)
    return GuardVerdict(True)


def check_render_safe(text: str) -> GuardVerdict:
    """Output handling (LLM05/LLM07): the reply carries no executable markup and no leaked secret
    or system prompt. Treat the model's output as untrusted and refuse anything a browser would run.
    """
    if _UNSAFE_MARKUP.search(text):
        return GuardVerdict(False, "reply contains unsafe markup")
    low = text.lower()
    if any(cue in low for cue in _SECRET_CUES):
        return GuardVerdict(False, "reply contains a secret or system-prompt leak")
    return GuardVerdict(True)


def check_no_other_customer(text: str, customer_id: str) -> GuardVerdict:
    """Confidentiality (LLM02): the reply names no other customer's identity, the Asana
    cross tenant disclosure caught at the door. The session customer's own name/id is allowed.
    Matches on word boundaries so a name never trips on a substring inside an ordinary word.
    """
    for other_id, account in SEED.items():
        if other_id == customer_id:
            continue
        for marker in (other_id, account.name):
            if re.search(rf"\b{re.escape(marker)}\b", text):
                return GuardVerdict(False, "reply contains another customer's data")
    return GuardVerdict(True)


def check_render_truth(text: str, customer_id: str) -> GuardVerdict:
    """Content check (pre render): a 'no contract / no fee' claim served to a customer who actually
    has a term contradicts the account oracle, the cold open's last line render catch.

    Scope, stated honestly: this is a cue-based heuristic. A cue from `_NO_CONTRACT_CUES` must be
    present *first*, and only then does the account oracle decide the contradiction. So it catches the
    demonstrated affirmative phrasings, not an arbitrary paraphrase ("you're free to leave") or a wrong
    fee *amount* (grading a numeric claim needs structured-claim extraction, deliberately not built
    here). It is the single home of the contradiction *rule*, kept deliberately narrow and
    reproducible; the eval grader (`metric_graders`) reuses it so the eval can never grade more
    leniently than the runtime.
    """
    claims_no_contract = any(cue in text.lower() for cue in _NO_CONTRACT_CUES)
    if claims_no_contract and truth_for(customer_id).has_contract:
        return GuardVerdict(False, "that answer contradicts your account (term/fee)")
    return GuardVerdict(True)
