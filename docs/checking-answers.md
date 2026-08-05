# Checking answers without a model

Retrieval hands the answering model a short list of chunks, the small pieces of text a document was cut into. The model writes a reply. Deciding whether that reply is *good* needs a judge model or a person. Both cost money and both can be wrong.

But two things about a reply can be settled by comparing two lists of ids. No reference answer, no judge, no money. So we do not record them as scores to read later. We test for them, and the build fails if either one breaks.

## What the writer returns

The answering model is not asked for prose. It has to fill in a small set of fields:

- `text`, the answer a customer would read
- `cited`, the chunk ids it used
- `outcome`, one of `answered`, `refused`, `unknown`, `not_applicable`
- `reason`, why, when it refuses

The alternative was reading prose and guessing. Looking for "I don't know" in free text misses a model that words it differently, and it fires wrongly when the answer only talks about being uncertain. A field with four values is just a field to read.

One rough edge: `reason` is required of the model and then thrown away. Nothing copies it onto the answer, and the answer has no field to hold it. We pay for those tokens and drop them.

## The two checks

**Every cited chunk must be one that was shown.** A model citing a chunk that was never put in front of it has invented its evidence. The comparison only means something because the list of shown chunks is filled in by the writer, from the chunks it was handed, in the order it handed them over. It is never read back out of the model's reply. If it were, the check would compare a list against itself and pass forever.

**An answer that says it answered must cite something.** That is a fault only when `citation_required` is on. The setting defaults to true and is part of the run id, so a run with it off is a different run and its numbers stay separate.

There used to be a third check: anything other than `answered` had to cite nothing, because a refusal that cites a chunk was said to contradict itself. It does not. A refusal that names the passages it consulted before concluding they do not hold the answer is showing its work, and it is better than one that refuses with nothing to point at. The prompt never asked for silence on refusal either, so the rule was stricter than the instruction it was checking. It fired on 39 of 97 answers in the first run after the answer prompt was fixed, every one of them correct behaviour, and a check that fires mostly on correct behaviour teaches whoever inherits it to stop reading the output.

## Recorded, never raised

Both faults land on the answer as a tuple of strings. Nothing raises an error.

That is on purpose. We build deliberately broken systems so we can check that our measurements actually notice a broken system, and two of them exist to produce these very faults. One says it answered and cites nothing. The other writes a confident sentence and cites one fixed chunk from the corpus instead of anything it was shown. If a violation raised an error, both would die on their first question, and the detector they exist to trip could never run against them.

The cost is that a bad answer travels. Asking a question prints the violations and carries on. Collecting a run copies them onto the row it writes. So if you read the output of a run, remember that an answer can carry a fault and still be a normal, complete answer.

## Where they block

Most of the assertions sit in the contract tests, with one more in the relations tests, and both of those jobs are on the blocking list. Another test asserts that the list and the workflow agree, so nobody can quietly mark a job `continue-on-error` and let it go green while failing.

Be clear about what that buys. Every test that asserts a clean answer drives the writer with a fake client that returns a reply the test states in advance. What is under test is the writer's bookkeeping: that the shown list really is what went into the prompt, and that the writer records the fault itself rather than leaving it to the caller. Nothing asserts that a real model produced no violations. The decision record this page replaces said such a step existed. It did not.

## Why not ask a model instead

The scoring suite used to carry one more scorer, which asked a language model whether every cited chunk was among the chunks shown. That question is `set(cited) <= set(shown)`. Paying a model to approximate a comparison of two lists is slower, costs money, and can be wrong, so we removed it. The four scorers left all ask questions a set comparison cannot answer.

## Two things we got wrong

**Brackets.** The prompt lists each chunk as `[chunk id] text`, so a model that copies the label copies the brackets too. Left alone, every citation failed the first check and the numbers read as total fabrication when nothing had been fabricated. We now strip the brackets while parsing the reply. A free check can be wrong for a reason that has nothing to do with the model's honesty, and this one was.

**We shipped a check that was stricter than the prompt.** This page used to argue that the two checks were different in kind: the first about invented evidence, the second mostly about whether the model followed the format. That was the diagnosis, and the remedy was to delete the second one rather than keep explaining it. The lesson worth keeping is the shape of the mistake: a rule about model output has to be checked against what the prompt actually asks for, or it measures the gap between two things you wrote rather than anything the model did.

## What we cannot tell you

How often either check fires. The run store carries no violations field at any level, so `just report --render` will not print you a count. The count does exist one level down, per question, but those files are written to `.cache/report/`, which git ignores. We do send a count per answer to Langfuse as a score called `citation_violations`, which helps if you run the trace stack and not at all if you do not.

That is a gap rather than a decision. Putting a violation count into the run store is the obvious next change, and until somebody makes it, this page cannot give you a number you can check for yourself.

It matters more than it sounds, because writing answers is optional. `just report` does retrieval only and is free against a warm cache. `--generate` is what pays a model to write something. Four of the eighteen rows in the run store are generated runs: two under `f45992f36859`, recorded before either prompt was given a new version, one under `1a737014a9eb` after the first move, and one under `6818412cb6ec` after the answer prompt went to 4.0.0. On every other recorded run there was no answer at all, so these two checks had nothing to look at.
