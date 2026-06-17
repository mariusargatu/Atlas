# Knowledge and policy answers

Customers can ask Atlas general questions. Atlas answers using our help content. This is the same content a customer could search on our help site.

## What is covered today

1. Our current plan and its terms. A customer can ask if there is a contract or if they can cancel any time and get a real answer.
2. Basic router troubleshooting. For example what to do when the connection light is the wrong color.

We are running this small test set of help content right now while we build the full library. That full library is built and ready for the next phase. It covers our older plans, fee schedules, device manuals, contract terms and current promotions. It is not live for customers yet. Do not assume a question about an older or discontinued plan will get a correct answer until that library goes live.

## A safety property we tested for

Help content is written by us. But it still passes through the same handling as anything else Atlas reads. We ran a test where a piece of content had a hidden instruction. The instruction tried to get Atlas to take an action just by being read. Atlas ignored it and kept answering the customer's real question. We treat this as a baseline requirement, not a nice to have. Help content could be edited by more people than our own team over time.

## What happens when nothing is found

If a search does not find a good answer Atlas says so. It does not guess. It offers to hand the customer to a person. We would rather have a customer wait a bit longer for a correct answer than get a confident wrong one.

## What this does not cover

Whether an answer is true for a specific customer is a separate question from whether an answer was found at all. See the risk we care most about in the product overview.
