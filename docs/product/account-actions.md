# Account changes

Atlas can make a small set of changes to a customer's account. Every one works the same way. Atlas proposes the change. It shows the change to the customer. It only applies the change once the customer confirms it. Atlas never makes a change without that confirmation.

## What a customer can ask Atlas to do

1. Switch plans. Once confirmed the account moves to the new plan. The bill and data cap update to match it. The customer is never left with the old plan's price or data cap by mistake.
2. Add an addon. Once confirmed the addon is added. Asking to add one that is already there does nothing. It is not treated as an error.
3. Remove an addon. Once confirmed the addon is removed. Asking to remove one that is not there also does nothing.
4. Reset a modem. This is logged as a request. By design it does not change anything on the account. It is an operational action, not an account change.
5. Open a support ticket. A new open ticket is created with the subject the customer gave.
6. Book an engineer visit. The requested time slot is added to the account.
7. Cancel service. This one has real rules behind it. See the cancellation policy.

## Two things we should fix before this goes further

1. Switching plans does not check if a customer is actually eligible for the plan they asked for. Right now only a lower level safety check stops an invalid plan change. There is no proper eligibility check at the point the request is made. We should add a real eligibility check here before this becomes a bigger part of the product.
2. Adding or removing an addon does not check that it is a real addon we offer. A customer could in theory ask for an addon that does not exist. Atlas would accept the request. We meant to restrict this to our real addon list and never finished the work. This is low risk today since nothing bad happens with a made up addon, but it is worth closing before addons matter more.

## Why cancelling service works differently

Every request above is handled the same simple way. Take the request. Apply it if confirmed. Cancelling service is the first one where that was not good enough. The outcome depends on more than the request itself. See the cancellation policy for why.
