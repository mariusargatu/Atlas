import { describe, expect, it } from "vitest";
import { isTypedConfirmation } from "./confirmation";

describe("isTypedConfirmation", () => {
  it("accepts the exact typed token", () => {
    expect(isTypedConfirmation("CONFIRM")).toBe(true);
    expect(isTypedConfirmation("  CONFIRM  ")).toBe(true);
  });
  it("rejects a bare yes or anything else", () => {
    expect(isTypedConfirmation("yes")).toBe(false);
    expect(isTypedConfirmation("confirm")).toBe(false);
    expect(isTypedConfirmation("")).toBe(false);
  });
});
