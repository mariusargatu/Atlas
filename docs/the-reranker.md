# The reranker, and why we ship it switched off

A reranker takes the list search came back with and puts it in a better order. Ours is a cross encoder: a small model that reads the question and one chunk together and returns a single number for how well that chunk answers it. Search compares a question against every chunk quickly and roughly, while a cross encoder compares it against a few candidates slowly and carefully. Usually that is a good trade. Here it is not. On this corpus it left the ranking worse than doing nothing, so the reranker backend defaults to `"passthrough"` and chunks come out in the order search put them in.

## What we measured

There are two backends and they work the same way. Both take the first 50 candidates from the blended list and return the best ten of them. One keeps the order search gave it. The other scores all 50 pairs and sorts by that score. So the only difference between the two runs below is who put the ten chunks in order.

| | search only | after the cross encoder |
|---|---|---|
| recall@10, how much of what should be found was found | 0.226 | 0.110 |
| precision@10, how much of the top ten belonged there | 0.364 | 0.187 |
| MRR@10, how high the first right chunk came | 0.639 | 0.357 |
| nDCG@10, which rewards right chunks near the top | 0.422 | 0.200 |
| success@10, at least one right chunk in the ten | 0.907 | 0.814 |

Both cover the same 97 questions, scored on the top ten against the documents tau2 says each one needs. Both sit in the run store, which `just report --render --all` prints. Run `45a977139549` is `data/settings/reranked.json`, which sets one field, the reranker backend, to `"onnx"`. The search-only column is run `1a737014a9eb`, and this page used to call that "the defaults", which it is not any more.

The reason is worth writing down, because it is the naming scheme doing its job in a way that looks at first like a bug. `run_id` hashes all eight settings groups, and `answer.prompt_version` is one of the values inside them. Editing the prose of the answer prompt from 3.0.0 to 4.0.0 therefore renamed every configuration at once, and the defaults now hash to `6818412cb6ec`. Not one retrieval number moved: `6818412cb6ec` and `1a737014a9eb` carry byte-identical figures for both arms, the fusion and the reranked row, because retrieval never reads either prompt. So the store holds 18 rows under 15 ids but only **7 distinct retrieval configurations**, and the ids in this table are the ones that were current when the measurement was taken. Check what the defaults hash to today rather than trusting the string above:

```
uv run python3 -c "from atlas.config import Settings; print(Settings().run_id)"
``` Recall cannot pass 0.604 here whatever we do, because a question's documents are cut into more chunks than ten slots can hold. And 0.200 nDCG is under the 0.256 the same run gets by handing every question one fixed list of ten chunks. Reranking put the ranking below one that never reads the question at all.

## Is that a result, or is it noise?

`just compare` answers that. It scores two settings files on the same questions and resamples the gaps question by question, so whatever makes a question hard cancels instead of counting twice. Give it one settings file and it compares against the defaults. It is free once `.cache/vectors` is warm.

    just compare data/settings/reranked.json
    just compare data/settings/reranked.json --metric recall@k

nDCG moves by -0.222, with a 95% interval of [-0.263, -0.181]. Recall moves by -0.115, interval [-0.147, -0.087]. Neither interval touches zero, so both are real.

Recall is the interesting one. Our blunt worst case bar at 97 questions reads 0.199, and 0.115 sits under it. It resolves anyway, because that bar assumes two unrelated groups of questions while this comparison scores both settings on the same ones. Sitting under the blunt bar does not make a gap noise. It only means the cheapest argument has not yet shown it to be real.

Only two of the five rows in the table above carry an interval. Precision, MRR and success@10 do not: nobody has run `just compare` for them and pasted the result here, the way this page does for recall and nDCG. Success@10 is the one most worth doing that for before reading its 0.907-versus-0.814 gap as settled: `evals/adversaries.py`'s own comment on `_SUCCESS_THRESHOLD` describes this metric's usable range on this corpus as unusually narrow, roughly 0.73 to 1.0 between a list that ignores the question and a perfect one, which is exactly the situation where a nine-hundredths point gap most needs an interval and least often gets one before somebody quotes it.

## It is not just the one model

We ran a second cross encoder over the identical candidate lists, same 50 in and same ten out, so the scorer is the only thing that changed.

| scorer | recall@10 | nDCG@10 |
|---|---|---|
| no reranking | 0.226 | 0.422 |
| ms-marco-MiniLM-L-6-v2 (the one we ship) | 0.110 | 0.200 |
| ms-marco-MiniLM-L-12-v2 | 0.075 | 0.144 |

So it is not model size. The bigger MiniLM is the worse of the two, and both land far below the 0.422 you get by leaving the list alone. One limit worth knowing: our reranker asks Hugging Face for `onnx/model_quantized.onnx` by name, so a model without that exact file cannot be tried by pointing a setting at it. We once compared six cross encoders from four families and every one lost to no reranking, but that ran on eight questions, too few to settle anything, and we have not repeated it at 97. The two rows above are what we can reproduce today, so they are the only ones we quote.

## The cause is what we are calling a question

Our questions are not questions. They are tau2 tasks, and a tau2 task is the brief handed to the model that plays the customer in a support conversation. `task_001` opens:

> You are playing the role of a customer contacting a customer service representative agent. Your character is a management consultant named Sarah Bosch who earns $100,000 annually. You travel frequently for work...

and ends, 1,575 characters later, with "Never respond as a customer service representative/assistant. You are playing the role of the customer." Over all 97 tasks the mean is 3,470 characters, the median 3,246, the shortest 1,163 and the longest 7,368.

A cross encoder expects a short question and a longer chunk. Here it is the other way round. With the model's own tokenizer the median task encodes to 835 tokens and the median chunk to 194, and the pair is cut to the model's 512 token limit by taking tokens from whichever side is longer. So it is nearly always the question that loses them. Over the 4,850 pairs one run scores, the question is truncated in 4,689 and the chunk in 701, and the question loses 2,789,264 tokens against the chunk's 8,475. A question is also cut to a different length beside each chunk, because truncation stops the moment a pair fits.

We had this backwards for a while. A comment in the code used to say the chunk was truncated first, so that a long question kept its wording. That is what would happen on a corpus of short questions and long chunks, and it is not what happens here.

Nothing is wrong with the model. Hand the same cross encoder a real question and it is good at its job. Ask "how do I dispute a card transaction?" with the reranker set to `"onnx"` and the top three chunks come from "How to Dispute a Credit Card Transaction" at 7.781, "Filing a Credit Card Transaction Dispute (Internal)" at 7.691 and "Internal: Filing a Debit Card Transaction Dispute" at 6.154. (`just ask` will not show you this. It has no backend flag, and under `passthrough` every score is zero, so it prints no score column.) The task text does not hurt the search arms nearly as much either: on those same tasks the vector arm reaches nDCG 0.355, the keyword arm 0.348 and the blend 0.422, against the cross encoder's 0.200.

## Why the stage is still in the code

Because the finding is about this corpus, not about reranking, and a repository that deleted the stage could not show you what it learned. One field turns it back on, free, in about two minutes. It costs us three things. The pipeline has seven stages and one of them, by default, only cuts the list rather than reordering it, which is awkward to explain. With the cross encoder on, that stage takes about 1.17 seconds a question, and the whole 97 question retrieval run goes from about 2 seconds to about 116. And comparing one candidate depth against another, which is the reason those depth settings exist, would now only measure how much damage each depth does.

## What this does not show

Not "rerankers do not help". What these numbers show is that a component measured on the wrong input looks broken, and that you cannot tell a broken component from a badly fed one without measuring. Reranking is a normal and useful stage. It does not earn its place on questions that are not questions.

If somebody derives a short question from each task, all of this has to be measured again before the default stands. We have not done it, on purpose. The only tools for it are a language model or a rule we write by hand, and both mean this repository writing its own questions against an answer key we chose because we did not write it. That trade may be worth making, but deliberately and with a measurement beside it.
