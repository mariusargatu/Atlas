"""The load lane (SP9 task 6): a k6 + xk6-sse script against the real `/chat/stream` SSE endpoint
(`k6/chat_sse_load.js`), BURST TIER ONLY (D3/7.1's own rule: local numbers are never quoted for a
saturation knee). This package is the pure Python half the k6 script leans on and the load lane's
own post run reporting reads:

  - `thresholds.py`: parses `thresholds.json` ("thresholds as code"), the SAME file the k6 script
    reads via `open()` for its own `options.thresholds`, so the two never silently disagree.
  - `prompt_corpus.py`: loads the fixed, controlled token count golden prompt corpus
    (`prompt_corpus.json`) every concurrency step cycles through.
  - `phoenix_join.py`: the post run join between a k6 iteration capture and a Phoenix span export,
    over `atlas.turn.seq` (SP6's fix wave repaired join key; verified still present on this
    branch's HEAD by `test_load_phoenix_join.py`'s own cross check against a real `OtelTracer`).

Fixed replicas for the burst run; KEDA autoscaling on `te_queue_size` (the TEI embed/rerank queue
depth) is the plan's own named, DEFERRED seam -- documented here and in `k6/chat_sse_load.js`'s own
header comment, never built by this task.

Every module here is pure (no network, no k6 binary, no live Phoenix call): `task test` gates them.
The actual k6 run and the live Phoenix export it joins against are live/burst, deferred, same as the
rest of SP9's DEFERRED LIVE BACKLOG.
"""
from __future__ import annotations
