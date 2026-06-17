# Atlas Rebuild — Product Requirements

Owned by product/business, not QA — the given input everything else (invariants, rubrics, tests)
derives from, never the other way around. This is a FIRST DRAFT proposed by the assistant, not
authored by an actual product owner: correct anything below based on your own judgment. Nothing
here is load-bearing until you've reviewed it.

## Purpose

A customer support chatbot for a broadband ISP, answering existing customers' questions about
their plan and taking limited, safe actions on their behalf. Not a sales/acquisition bot (no new
customer signup), not a general-purpose assistant.

## Catalog

**Regions:** north, central, south

**Plans** (available in all regions, flat national pricing — region is currently a data field
only, does not affect price or availability):

| Plan | Speed | Price/month |
|---|---|---|
| Essential | 100 Mbps | £25 |
| Plus | 500 Mbps | £40 |
| Ultra | 1000 Mbps | £55 |

## Policy data

**Contract term:** all plans carry a fixed 12-month minimum term.

**Early-exit fee:** £75 flat if service is cancelled before the 12-month term ends. Flat
regardless of how many months remain — same "start simple, flat rather than prorated" pattern as
the catalog's flat national pricing.

**Hardship waiver:** the early-exit fee is fully waived for bereavement, job loss, or serious
illness, subject to proof. The chatbot's job is to recognise a qualifying reason and mark the fee
as waived-pending-verification in the cancellation proposal — it does not itself verify the
proof; that's a human/back-office step. (This is the exact case the bereavement example named:
the agent must surface this waiver, not just quote the standard £75 fee.)

**Billing disputes:** the chatbot can look up and explain the customer's own billing history
against policy. It never proposes or issues a refund, confirmed error or not — every dispute
hands off to a human. Unlike #4/#5/#7, there is no propose-an-action step here at all.

**Installation fee:** flat £50 for any new installation or address change, regardless of region.

## Capabilities (in scope)

1. **Plan lookup** — answer questions about available plans (catalog data: name, speed, price).
2. **Account lookup** — answer questions about the customer's OWN current plan and price (reads
   their account, never another customer's).
3. **Policy questions** — answer using the knowledge base (promotions, contract terms,
   cancellation policy), grounded in what's actually retrieved plus the customer's own account
   state where relevant (e.g. whether a promo is still active for THEM specifically).
4. **Propose a plan change** — when a customer asks to switch plans, produce a structured
   proposal (current plan, target plan, new price, effective date). Never executes it directly;
   an explicit UI action outside the chatbot's own turn is what actually applies it.
5. **Propose cancelling service** — customer requests to cancel; agent must surface every
   applicable exception (the hardship waiver vs. the standard early-exit fee — see Policy data)
   before producing a structured cancellation proposal. Never executes directly, same action
   policy as #4.
6. **Billing disputes and refunds** — customer disputes a charge or requests a refund; agent
   explains the account's billing history against policy (see Policy data), then hands off to a
   human. No refund-proposal capability, unlike #4/#5/#7.
7. **Address changes / new installation** — customer requests to move service or install at a
   new address; agent quotes the installation fee (see Policy data) and proposes the change.
   Same action policy as #4.

## Explicitly out of scope (declines / hands off to a human)

- Technical support / outage troubleshooting
- Anything not about the customer's own account or the plan catalog (e.g. general knowledge,
  other companies' products, unrelated topics)

## Action policy

The chatbot never directly mutates account state under any capability above. Any change is a
proposal only; a human-triggered UI action, outside the chatbot's turn, is the only thing that
executes it.
