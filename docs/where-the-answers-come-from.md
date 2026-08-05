# Where the answers come from

Every score here compares what the system retrieved against what should have been retrieved. This page is about the second half of that: where the documents come from, where the correct answers come from, and why we work out which chunks are correct while we measure.

## The documents and the answer key are somebody else's

The corpus is not ours. It is the banking knowledge domain of Sierra Research's tau2-bench, copied into `data/tau2/banking_knowledge/` under its MIT licence. `NOTICE` names the source and the terms.

We get 698 documents and 97 customer service tasks. Each task carries a `required_documents` list: the documents an agent has to consult to serve that customer. That list is our answer key. People often call it the gold set, which just means the record of what correct looks like, fixed before the system runs.

We read those files and hand back the documents and one question per task. Nothing is edited and nothing is generated. So we did not write the documents, we did not write the questions, and we did not choose which documents answer which question.

## Why we take an answer key instead of writing one

You could build the corpus yourself: write the documents and the questions, and record the correct answer as you go. That has a real upside: you know exactly how good your answer key is, because you built it. We still do not, for two reasons. A corpus you built cannot surprise you: the documents are as long as you made them, worded the way you word things, and hard in the ways you thought to make them hard. And if you write the documents, the answer key and the metrics, every number is your own homework marked by yourself. Nothing stops you from slowly reshaping the corpus, in good faith, into the shape your metric scores well on. An answer key from outside stops that, instead of leaving it to your own care.

## We never store a list of correct chunks

We record which **documents** answer each question. We never record which **chunks** answer it. A chunk does not exist until something has cut a document into chunks, and comparing ways of cutting documents is one of the things this project is for. Change the chunk settings and the chunks change: at 256 tokens the recursive strategy gives 1010 chunks, the sentence strategy 1011, and the fixed strategy 977. A saved list of chunk names would be right for exactly one of those and wrong for the others, including the ones it was being used to judge. The wrongness would be invisible too: stale chunk names still point at real chunks, so a score still comes back. It is a plausible number that means nothing, which is the failure this repository exists to teach you to spot.

So the chunk level answer is worked out at measuring time, against whichever chunks are under test. Resolving all 97 questions against the 1010 chunks takes well under a tenth of a second, and the answer is right for every setting, because it is read off the chunks in front of it. If a required document was cut into no chunks at all, we raise rather than return an empty correct set, because recall over an empty set has no meaning and the easy way to write it hands back a perfect score.

One place does write chunk names down, and it is not an answer key: the per question ledger under `.cache/report/`, which is gitignored and named after the run id, so it is a resume cache for one configuration. The Langfuse dataset carries the answer key too, and there an item's expected output is `required_documents`. Documents again, never chunks.

## The one thing we do decide, and how

One caveat to "we did not choose which documents answer which question", said here rather than left for you to find. A tau2 task requires several documents, but only some of them carry the answer. The task asking which credit card pays the most cash back requires four card documents and is answered by one; the rest are there to be ruled out. A ranking that puts the answer last has done a worse job than one that puts it first, and flat recall cannot see it.

So we work out which of a task's required documents carry the answer. The input is tau2's own success criteria, the tool calls a correct agent would make, whose arguments name real things like a card type or an account class. The matching rule, though, was written here: a list of argument keys to ignore because they describe the call rather than the bank, a floor of six characters so short values do not match by accident, and a test for whether the value appears inside a document id, once capitals and punctuation are stripped from both. It finds answer documents for 22 of the 97 tasks and returns nothing for the rest. Only the graded nDCG measurement scores against it, and every recorded run marks that row `questions: 22`, so a figure averaged over 22 questions is not mistaken for one averaged over 97. So read the required list as tau2's, and this smaller set as our reading of tau2's criteria.

## What this costs us

**We cannot check the answer key.** We know our numbers agree with tau2's annotators, not whether the annotators were right, and nothing here can find out. A cheap check we have not yet done: read fifty tasks against their documents and report how often we disagree.

**The answer key names documents, not places inside them.** We can measure what a cutting strategy does to retrieval, but not whether a chunk boundary sliced a value in half. Two things were built before that cost was taken seriously, and both are now deleted. A fact record type held a fact name, a document and a character range, but every record named its own document and spanned the whole of it, so resolving one did arithmetic on character ranges to reach an answer a plain equality check reaches. A question also used to carry the value a correct answer states, the one thing that could catch an answer giving a wrong value rather than no value. No collection source ever set it, so the check reading it could never fire. Carrying the shape of a measurement without the data behind it is how an eval comes to look like it is running.

**The questions were not written as search queries.** tau2 is a multi turn benchmark, so a question here is a whole customer scenario: about 3,250 characters at the median and about 1,160 at the shortest. We search over the whole scenario because that is what the task gives us. Some of the difficulty in our numbers is real, and some is that mismatch, and nothing here can separate the two.

## The numbers this answer key produces

From the recorded run `1a737014a9eb` at k=10. Print it with `just report --render --run-id 1a737014a9eb`, which reads the run store and needs no key. Plain `just report --render` prints the run recorded most recently, usually another one.

| system | recall@10 | nDCG@10 |
|---|---|---|
| vector | 0.184 | 0.355 |
| keyword | 0.192 | 0.348 |
| fused | 0.226 | 0.422 |
| the best list that ignores the question | 0.107 | 0.256 |
| a random ranking | 0.012 | 0.027 |

Recall@10 is the share of a question's correct chunks that turn up in the top ten. nDCG@10 asks how well the list is ordered, counting a correct chunk near the top for more than one near the bottom, with 1.0 as a perfect order. The reranked row is missing on purpose: it scores exactly what fused scores, because the reranker we ship is a passthrough that reorders nothing. Recall@10 cannot reach 1.0 here, either. The average question requires about twenty one chunks and 73 of the 97 require more than ten, so ten slots cannot hold them all and the average maximum is 0.604. That ceiling is published beside every recall figure, because 0.226 read against an imagined 1.0 looks like a broken retriever, and read against 0.604 it is 37 percent of what was available.

The fourth row is the one to sit with. A fixed list of ten chunks that never reads the question reaches recall 0.107, and that is not search working. It is the answer key being generous: with about twenty one correct chunks per question out of 1010, guessing well is worth something, and a change claiming to improve retrieval has to clear that row first.
