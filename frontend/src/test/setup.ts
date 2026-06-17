import "@testing-library/jest-dom/vitest";
import { setAccessToken } from "@/api/client";
import { afterAll, afterEach, beforeAll } from "vitest";
import { server } from "./msw/server";

// jsdom lacks scrollIntoView/scrollTo + ResizeObserver; assistant-ui's Viewport/Composer use them.
Element.prototype.scrollIntoView = () => {};
Element.prototype.scrollTo = () => {};
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => {
  server.resetHandlers();
  setAccessToken(null);
});
afterAll(() => server.close());
