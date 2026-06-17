# Cancelling service, with hardship waiver

## What it does

A customer can ask Atlas to cancel their service. As with every account change, Atlas proposes the cancellation first. It only carries it out once the customer confirms. Atlas also tells the customer if the plan's early exit fee applies to them or if it is being waived. This depends on the reason the customer gives for cancelling.

## Why we built a waiver

Not every plan has an exit fee. Our current plan does not. The older plan some customers are still on does. Charging that fee no matter what a customer's situation is would not be right. We decided the fee should be waived in three situations: a bereavement, a job loss or a serious illness. A support assistant that gets someone's account details right but still charges a grieving customer an exit fee has failed them. This is true even if every number it gave was correct.

## The three qualifying reasons

1. Bereavement.
2. Job loss.
3. Serious illness.

We chose a short fixed list rather than leaving this open. This matches how we have scoped every other request Atlas can act on. A customer explains their situation in their own words. Atlas is responsible for recognising when it matches one of these three. The customer should not need to name a category themselves.

## How the fee is decided

1. If the customer's plan has no exit fee, the reason they give does not matter. There is nothing to waive.
2. If the plan has a fee and the customer's reason matches one of the three situations, the fee is waived pending verification. Atlas does not verify the customer's claim itself. That is a step for a person to handle afterward, the same way we would want a human agent to check a claim like this too rather than take it purely at face value.
3. If the plan has a fee and no qualifying reason was given, the standard fee applies.

## What we tell the customer

The customer needs to know if their fee is being waived before they confirm the cancellation. They should not find this out afterward. The confirmation message needs to say this too. It should not just give a reference number and leave the fee outcome unsaid. Getting this right matters as much as getting the fee calculation right. A correct decision the customer is never told about is not a good customer experience.

## What happens to the account

Confirming a cancellation does not change the customer's account status right now. We record that the request happened. We have not yet built the part where an account is actually marked cancelled. This is a choice we made on purpose, not an oversight. We wanted the waiver decision and the customer message right first. Marking accounts as cancelled is planned before this goes to real customers.

## Out of scope for this phase

1. Billing disputes and refunds. We have decided Atlas should explain a customer's billing history. It should never decide a refund itself. Every dispute should go to a person. This has not been built yet.
2. Address changes and new installations. Also not built yet. We would want them to follow the same propose and confirm pattern as everything else.
3. Verifying hardship claims. Entirely a job for a person. Atlas's role stops at recognising that a customer's stated reason qualifies for a waiver.

## How we will know this is working

Getting the fee outcome right is necessary but not enough on its own. We also care if Atlas actually answers what the customer asked instead of avoiding it. We care if its responses are clear rather than full of filler. We care if its tone fits what the customer just told it. It should not sound cold and transactional. It should not sound falsely upbeat when someone has just described a bereavement. We have defined what good looks like for each of these. We have not yet built the process to measure them on an ongoing basis. That is the next piece of work before we would trust this at scale.
