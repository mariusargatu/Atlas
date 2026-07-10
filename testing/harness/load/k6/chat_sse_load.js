// SP9 task 6 (D31): k6 + xk6-sse load lane against the real /chat/stream SSE endpoint.
//
// BURST TIER ONLY (D3/7.1: local numbers are never quoted for a saturation knee) -- point
// ATLAS_BASE_URL at the recreated Hetzner burst tier's own ingress, never localhost. Needs a k6
// binary built with the xk6-sse extension (github.com/grafana/xk6-sse), NOT the stock k6 binary
// (`k6/x/sse` does not exist there): `xk6 build --with github.com/grafana/xk6-sse@latest`.
//
// Custom Trend/Rate metrics (ttft_ms, tokens_per_sec, e2e_ms, goodput) plus thresholds AS CODE,
// loaded from ./thresholds.json -- the SAME file testing/harness/load/thresholds.py parses
// hermetically, one source of truth so this script and the Python side can never silently drift
// onto different numbers (task test gates thresholds.py; there is no k6 binary in that lane, so
// this file itself is never executed there).
//
// Stepped concurrency, 1 to 32 VUs: one k6 scenario per named step, run back to back (never
// overlapping), each tagged with its own `concurrency` value so every Trend/Rate sample carries
// which step produced it -- the shape the report run after the fact
// (phoenix_join.summarize_by_concurrency) groups by.
//
// Trace context: every iteration mints its own W3C traceparent header (correct practice for a dual
// plane), though this reference system's own tracer does not yet extract an inbound traceparent
// into its span tree (a documented, separate gap, not this task's to close). The join this task
// actually relies on is the backend's OWN message_start.trace_id (atlas.turn.seq, SP6's fix wave)
// -- captured per iteration below as LOAD_ITER console lines, joined after the run against a real
// Phoenix span export by testing/harness/load/phoenix_join.py (its two part join mechanism IS
// hermetically proven, over a fixture span set, by task test; the live Phoenix export itself is
// live, burst tier only, and deferred).
//
// Fixed replicas for the whole run. KEDA autoscaling on te_queue_size (the TEI embed/rerank queue
// depth) is the plan's own named, DEFERRED autoscaling seam -- documented here, never built.

import exec from 'k6/execution';
import http from 'k6/http';
import sse from 'k6/x/sse';
import { Rate, Trend } from 'k6/metrics';
import { check } from 'k6';

const BASE_URL = __ENV.ATLAS_BASE_URL || 'http://localhost:8000';
const CUSTOMER_ID = __ENV.ATLAS_LOAD_CUSTOMER_ID || 'cust_current';
const STEP_DURATION_S = parseInt(__ENV.ATLAS_LOAD_STEP_SECONDS || '60', 10);
const STEP_GAP_S = 10; // headroom between steps so two scenarios never overlap in time

const CONCURRENCY_STEPS = [1, 2, 4, 8, 16, 32];
const PROMPTS = JSON.parse(open('../prompt_corpus.json'));
const THRESHOLD_SPEC = JSON.parse(open('../thresholds.json'));
const TTFT_CEILING_MS = THRESHOLD_SPEC.ttft_ms.value;

const ttftTrend = new Trend('ttft_ms', true);
const tokensPerSecTrend = new Trend('tokens_per_sec', true);
const e2eTrend = new Trend('e2e_ms', true);
const goodputRate = new Rate('goodput');

function buildScenarios() {
  const scenarios = {};
  let cursor = 0;
  for (const vus of CONCURRENCY_STEPS) {
    scenarios[`vus_${vus}`] = {
      executor: 'constant-vus',
      vus: vus,
      duration: `${STEP_DURATION_S}s`,
      startTime: `${cursor}s`,
      exec: 'chatTurn',
    };
    cursor += STEP_DURATION_S + STEP_GAP_S;
  }
  return scenarios;
}

function renderThresholds(spec) {
  const out = {};
  for (const metric of Object.keys(spec)) {
    const { stat, op, value } = spec[metric];
    out[metric] = [`${stat}${op}${value}`];
  }
  return out;
}

export const options = {
  scenarios: buildScenarios(),
  thresholds: renderThresholds(THRESHOLD_SPEC),
};

function randomHex(byteLength) {
  let s = '';
  for (let i = 0; i < byteLength; i++) {
    s += Math.floor(Math.random() * 256).toString(16).padStart(2, '0');
  }
  return s;
}

// setup() runs once, not per VU: one bearer token, reused by every iteration (the demo login has
// no password; a real target on the burst tier still needs this exact call to obtain a scoped
// token).
export function setup() {
  const res = http.post(
    `${BASE_URL}/auth/login`,
    JSON.stringify({ customer_id: CUSTOMER_ID }),
    { headers: { 'Content-Type': 'application/json' } },
  );
  check(res, { 'login succeeded': (r) => r.status === 200 });
  return { token: res.json('access_token') };
}

export function chatTurn(data) {
  const concurrency = parseInt(exec.scenario.name.split('_')[1], 10);
  const prompt = PROMPTS[exec.scenario.iterationInTest % PROMPTS.length];

  // W3C traceparent (00-<32 hex trace id>-<16 hex parent id>-01): correct practice for a dual
  // plane, see this file's own header comment for why this is not the actual join key run after
  // the fact.
  const traceparent = `00-${randomHex(16)}-${randomHex(8)}-01`;

  const turnStart = Date.now();
  let ttftMs = null;
  let tokenCount = 0;
  let backendTraceId = null;
  let finishReason = null;

  const response = sse.open(
    `${BASE_URL}/chat/stream`,
    {
      method: 'POST',
      body: JSON.stringify({ message: prompt.text, thread_id: `load-${__VU}-${__ITER}` }),
      headers: {
        Authorization: `Bearer ${data.token}`,
        'Content-Type': 'application/json',
        traceparent: traceparent,
      },
      tags: { concurrency: String(concurrency) },
    },
    function (client) {
      client.on('event', function (event) {
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch (e) {
          return; // not JSON: a keepalive/comment frame, not one of the six typed events
        }
        if (payload.event === 'message_start') {
          backendTraceId = String(payload.trace_id);
        } else if (payload.event === 'token') {
          if (ttftMs === null) {
            ttftMs = Date.now() - turnStart;
          }
          tokenCount += 1;
        } else if (payload.event === 'message_end') {
          finishReason = payload.finish_reason;
          client.close();
        } else if (payload.event === 'error') {
          finishReason = 'error';
          client.close();
        }
      });
      client.on('error', function () {
        finishReason = finishReason || 'transport_error';
      });
    },
  );

  const e2eMs = Date.now() - turnStart;
  const tokensPerSec = ttftMs !== null && tokenCount > 0 ? tokenCount / (e2eMs / 1000) : 0;
  const good = finishReason === 'complete' && ttftMs !== null && ttftMs < TTFT_CEILING_MS;

  const tags = { concurrency: String(concurrency) };
  // NOTE (SP9 final review, M3): a turn that never streams a token (ttftMs stays null) folds its
  // full e2eMs into the ttft Trend here, and tokensPerSec below only ever divides by the full turn
  // wall time, never a post-first-token window -- both are metric-definition choices to recalibrate
  // against the first real burst pass, not before (the live sweep itself is deferred).
  ttftTrend.add(ttftMs !== null ? ttftMs : e2eMs, tags);
  tokensPerSecTrend.add(tokensPerSec, tags);
  e2eTrend.add(e2eMs, tags);
  goodputRate.add(good, tags);

  // One structured line per iteration: testing/harness/load/phoenix_join.py's
  // load_iteration_records() reads exactly this shape (JSON with a LOAD_ITER prefix), skipping
  // every other k6 console line as ordinary noise. An operator captures it via
  // `k6 run chat_sse_load.js | tee run.log`, then greps the LOAD_ITER lines out to join them
  // against a Phoenix export once the run is over.
  console.log('LOAD_ITER ' + JSON.stringify({
    trace_id: backendTraceId,
    concurrency: concurrency,
    ttft_ms: ttftMs,
    tokens_per_sec: tokensPerSec,
    e2e_ms: e2eMs,
    goodput: good,
    prompt_id: prompt.prompt_id,
  }));

  check(response, { 'sse handshake ok': (r) => r && r.status === 200 });
}
