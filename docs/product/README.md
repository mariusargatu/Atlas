# Atlas: product overview

Atlas is our support assistant for broadband customers. It answers questions about plans and policies. It looks up a customer's own account. It can also make some account changes for a customer.

## What Atlas handles today

1. Atlas answers questions about our plans and policies. A customer can ask what our cancellation policy is or if we offer a static IP.
2. Atlas answers questions about a customer's own account. A customer can ask what plan they are on, how much data they have used, when their bill is due or what the status of a ticket is.
3. Atlas can make some changes for a customer. It can switch a customer to a new plan, add or remove an addon, reset a modem, open a support ticket, book an engineer visit or cancel service.

Every change is shown to the customer first. Atlas never makes a change until the customer confirms it.

## The risk we care most about

An answer can sound confident and still be wrong for the customer asking it. Our help content describes our current plan. That plan has no contract. Some customers are still on an older plan with a 12 month contract. If Atlas answers a question about cancelling using only the general help content it can give an answer that sounds right and is not right. The fact that matters lives on the customer's account, not in the help article. This is the biggest risk in this product. Every part of it should be read with this in mind.

## A current limitation worth tracking

Account questions and general help questions are handled by the same flow today, not two separate ones. This has not caused a wrong answer so far because both flows have the tools they need. But it means we cannot yet treat a billing question differently from a troubleshooting question. Worth fixing before we build something that depends on telling these two apart.

## Areas of functionality

1. [Knowledge and policy answers](knowledge-and-policy.md)
2. [Account and plan information](account-access.md)
3. [Account changes](account-actions.md)
4. [Cancelling service with hardship waiver](cancel-service.md)

## Requirements draft

[Product requirements](requirements.md) is a separate artifact from this overview. This README
describes what Atlas does today. That document states what a rebuild should require, and is a first
draft awaiting a product owner's review, so treat it as a proposal rather than a decision.

## Principles we hold to

1. Nothing a customer does can affect another customer's account.
2. Every change is shown to the customer and confirmed before it happens.
3. A customer's identity is set once at sign in and never changes during the conversation.

Every doc below assumes point 3 without saying it again.
