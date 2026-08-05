---
version: 1.0.0
---
# Scoring guide

Score the answer against the passages it was shown, not against outside
knowledge. Pass only when every one of the following holds:

1. Every fact stated in the answer is supported by at least one shown passage.
2. Every citation names a passage that was actually shown.
3. The answer states unknown when the question is about an entity that
   appears nowhere in the shown passages, rather than guessing.
4. The answer states not applicable when the question asks about a field
   absent from an otherwise relevant record, rather than guessing.
5. A refusal is only a pass when the shown passages genuinely do not contain
   the answer.

Fail on any fabricated fact, any citation of a passage that was not shown, or
any confident answer where the correct outcome was unknown, not applicable,
or a refusal.
