"""The accounts, this customer's reality. The single source of account state, write through.

`SEED` is the pristine fixture. `_STATE` is the live store the application reads and writes. Reads
return `_STATE`, writes replace a customer's record with an immutably updated `Account`
(`dataclasses.replace`, never field mutation), so a read after a confirmed write reflects the
change. `_STATE` is the one mutable boundary, the in memory "database". `reset_state()` restores it
from `SEED` so each test starts from the same world (CI calls it via an autouse fixture).

The seed is a test artifact: `cust_legacy_term` (Daniel) carries the cold open bug by construction,
`cust_current` (Sarah) is the happy path, `cust_neighbor` is the second customer scope and
cache isolation need. Money is `Decimal`. Bills equal the catalog price for the plan on file, so the
oracle stays internally consistent, and a `change_plan` re prices the bill to keep it that way.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from atlas.domain import catalog


@dataclass(frozen=True)
class Usage:
    period: str                  # billing month, e.g. "2026-06"
    gigabytes_used: Decimal
    data_cap_gb: int | None      # mirrors the plan, None == uncapped


@dataclass(frozen=True)
class Bill:
    period: str
    amount: Decimal
    due_date: str
    paid: bool


@dataclass(frozen=True)
class Equipment:
    kind: str                    # "router" | "modem" | ...
    model: str
    serial: str


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    subject: str
    status: str                  # "open" | "closed"


@dataclass(frozen=True)
class Account:
    customer_id: str
    name: str
    plan_id: str
    usage: Usage
    bill: Bill
    equipment: tuple[Equipment, ...]
    tickets: tuple[Ticket, ...]
    addons: tuple[str, ...] = ()
    bookings: tuple[str, ...] = ()


SEED: dict[str, Account] = {
    "cust_current": Account(
        "cust_current", "Sarah", "plan_current_fast",
        usage=Usage("2026-06", Decimal("240.5"), None),
        bill=Bill("2026-06", Decimal("35.00"), "2026-06-28", paid=False),
        equipment=(Equipment("router", "Fast Hub 3000", "RH-0001"),),
        tickets=(Ticket("tk-1001", "Wi-Fi drops in the evening", "closed"),),
        addons=("static_ip",),
    ),
    "cust_legacy_term": Account(
        "cust_legacy_term", "Daniel", "plan_legacy_value",
        usage=Usage("2026-06", Decimal("512.0"), 500),   # over the legacy cap, a real read to grade
        bill=Bill("2026-06", Decimal("39.00"), "2026-06-28", paid=False),
        equipment=(Equipment("modem", "Value Box 1000", "VB-7777"),),
        tickets=(Ticket("tk-2002", "Charged an early-termination fee", "open"),),
    ),
    "cust_neighbor": Account(
        "cust_neighbor", "Emma Clarke", "plan_current_fast",
        usage=Usage("2026-06", Decimal("88.0"), None),
        bill=Bill("2026-06", Decimal("35.00"), "2026-06-28", paid=True),
        equipment=(Equipment("router", "Fast Hub 3000", "RH-0002"),),
        tickets=(),
    ),
}

# The live store. Seeded from SEED, mutated only by the write helpers below, never in place.
_STATE: dict[str, Account] = dict(SEED)


def reset_state() -> None:
    """Restore the live store to the pristine seed, the per test world reset (CI autouse fixture)."""
    global _STATE
    _STATE = dict(SEED)


# ---- reads (current state) ----
def get_account(customer_id: str) -> Account:
    return _STATE[customer_id]


def get_usage(customer_id: str) -> Usage:
    return _STATE[customer_id].usage


def get_bill(customer_id: str) -> Bill:
    return _STATE[customer_id].bill


def get_equipment(customer_id: str) -> tuple[Equipment, ...]:
    return _STATE[customer_id].equipment


def list_tickets(customer_id: str) -> tuple[Ticket, ...]:
    """The customer's OPEN tickets, what the agent surfaces as 'your open tickets'."""
    return tuple(t for t in _STATE[customer_id].tickets if t.status == "open")


# ---- writes (write through, each returns nothing, replaces the stored record immutably) ----
def _commit(customer_id: str, account: Account) -> None:
    _STATE[customer_id] = account


def _next_ticket_id(account: Account) -> str:
    return f"tk-{account.customer_id}-{len(account.tickets) + 1}"


def apply_change_plan(customer_id: str, plan_id: str) -> None:
    acct = _STATE[customer_id]
    plan = catalog.get_plan(plan_id)  # KeyError on an unreal plan, value bounds guard catches it first
    new_bill = replace(acct.bill, amount=catalog.compute_price(plan_id))
    new_usage = replace(acct.usage, data_cap_gb=plan.data_cap_gb)
    _commit(customer_id, replace(acct, plan_id=plan_id, bill=new_bill, usage=new_usage))


def apply_add_addon(customer_id: str, addon_id: str) -> None:
    acct = _STATE[customer_id]
    if addon_id not in acct.addons:
        _commit(customer_id, replace(acct, addons=acct.addons + (addon_id,)))


def apply_remove_addon(customer_id: str, addon_id: str) -> None:
    acct = _STATE[customer_id]
    _commit(customer_id, replace(acct, addons=tuple(a for a in acct.addons if a != addon_id)))


def apply_reset_modem(customer_id: str) -> None:
    """An operational action with no persistent account field, the audit log is its only record."""
    return None


def apply_open_ticket(customer_id: str, subject: str) -> None:
    acct = _STATE[customer_id]
    ticket = Ticket(_next_ticket_id(acct), subject, "open")
    _commit(customer_id, replace(acct, tickets=acct.tickets + (ticket,)))


def apply_book_engineer(customer_id: str, slot: str) -> None:
    acct = _STATE[customer_id]
    _commit(customer_id, replace(acct, bookings=acct.bookings + (slot,)))


# The write surface, generic over the tool, the executor's write through hook (principle 11).
_WRITES = {
    "change_plan": lambda cid, a: apply_change_plan(cid, a["plan_id"]),
    "add_addon": lambda cid, a: apply_add_addon(cid, a["addon_id"]),
    "remove_addon": lambda cid, a: apply_remove_addon(cid, a["addon_id"]),
    "reset_modem": lambda cid, a: apply_reset_modem(cid),
    "open_ticket": lambda cid, a: apply_open_ticket(cid, a["subject"]),
    "book_engineer": lambda cid, a: apply_book_engineer(cid, a["slot"]),
}


def apply_write(tool: str, customer_id: str, args: dict) -> None:
    """Mutate the account store for a confirmed write. Called once per action by the executor.
    The idempotency key guarantees a retry never re runs this."""
    _WRITES[tool](customer_id, args)
