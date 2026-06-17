import { env } from "@/env";
import createClient from "openapi-fetch";
import type { components, paths } from "./generated/types";

export type PendingAction = components["schemas"]["PendingOut"];

/**
 * THE PORT. The UI depends on this typed client, never on raw `fetch`. The live adapter is
 * openapi-fetch → FastAPI; the deterministic fake adapter is MSW (src/test/msw) built from the
 * same OpenAPI contract — mirroring the backend's in-memory MCP fakes.
 *
 * Identity invariant: `customer_id` rides ONLY in the bearer token, never in a request body or
 * message. The access token is held in memory by the session layer and injected per request.
 */
let accessToken: string | null = null;
export function setAccessToken(token: string | null): void {
  accessToken = token;
}

export const api = createClient<paths>({
  baseUrl: env.VITE_API_BASE,
  credentials: "include", // send the httpOnly refresh cookie on /auth/refresh
  // Late-bind fetch: resolve globalThis.fetch at CALL time, not module-load time, so MSW (patched
  // after import) and any other interceptor are honored. Harmless in the browser.
  fetch: (input: Request) => globalThis.fetch(input),
});

// Auth middleware: attach the in-memory access token; on 401, try one silent refresh + retry.
api.use({
  async onRequest({ request }) {
    if (accessToken) request.headers.set("authorization", `Bearer ${accessToken}`);
    return request;
  },
  async onResponse({ request, response }) {
    if (response.status !== 401) return response;
    // never recurse on the refresh endpoint itself
    if (new URL(request.url).pathname.endsWith("/auth/refresh")) return response;
    if (!(await refreshAccessToken())) return response;
    const retry = new Request(request);
    retry.headers.set("authorization", `Bearer ${accessToken}`);
    return globalThis.fetch(retry);
  },
});

/** Refresh the access token from the httpOnly cookie. Returns false if the session is gone. */
async function refreshAccessToken(): Promise<boolean> {
  const { data } = await api.POST("/auth/refresh");
  if (data?.access_token) {
    setAccessToken(data.access_token);
    return true;
  }
  return false;
}
