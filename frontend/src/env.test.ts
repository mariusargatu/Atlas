import { describe, expect, it } from "vitest";
import { env } from "./env";

describe("env", () => {
  it("parses with a sane default API base", () => {
    expect(env.VITE_API_BASE).toBeTypeOf("string");
    expect(env.VITE_API_BASE.length).toBeGreaterThan(0);
  });
});
