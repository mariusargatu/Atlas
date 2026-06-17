import type { components } from "@/api/generated/types";
import { http, HttpResponse } from "msw";

type ChatOut = components["schemas"]["ChatOut"];

/**
 * The DETERMINISTIC FAKE ADAPTER. MSW handlers built from the same OpenAPI contract, mirroring the
 * backend's in memory MCP fakes. These script FRONTEND behavior (rendering, the confirm flow). The
 * backend's own logic is proven in pytest. Identity is irrelevant here by design. These fakes test
 * the UI, not the oracle.
 */
const base = "*/api"; // wildcard origin + the client's base path, matches any test/dev origin

export const handlers = [
  http.post(`${base}/auth/login`, async ({ request }) => {
    const body = (await request.json()) as { customer_id: string };
    return HttpResponse.json({
      access_token: "test-access",
      customer_id: body.customer_id,
      name: "Test User",
    });
  }),
  http.post(`${base}/auth/refresh`, () =>
    HttpResponse.json({
      access_token: "test-access-refreshed",
      customer_id: "cust_current",
      name: "Test User",
    }),
  ),
  http.post(`${base}/chat`, async ({ request }) => {
    const { message, thread_id = "t" } = (await request.json()) as {
      message: string;
      thread_id?: string;
    };
    if (/contract|fee|cancel/i.test(message)) {
      const out: ChatOut = {
        type: "final",
        thread_id,
        final_response:
          "[safe handoff] that answer contradicts your account (term/fee); let me get a person.",
      };
      return HttpResponse.json(out);
    }
    if (/switch|change|plan/i.test(message)) {
      const out: ChatOut = {
        type: "interrupt",
        thread_id,
        pending: { tool: "change_plan", args: { plan_id: "plan_current_fast" } },
      };
      return HttpResponse.json(out);
    }
    const out: ChatOut = { type: "final", thread_id, final_response: "You are on the Value plan." };
    return HttpResponse.json(out);
  }),
  http.post(`${base}/chat/resume`, async ({ request }) => {
    const { thread_id, confirmation } = (await request.json()) as {
      thread_id: string;
      confirmation: string;
    };
    const final_response =
      confirmation === "CONFIRM"
        ? "Done. Your reference is ref-000001."
        : "[safe handoff] needs a typed confirmation.";
    const out: ChatOut = {
      type: "final",
      thread_id,
      final_response,
      result: confirmation === "CONFIRM" ? { reference: "ref-000001", applied: true } : null,
    };
    return HttpResponse.json(out);
  }),
];
