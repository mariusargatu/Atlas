import { describe, expect, it } from "vitest";
import { colors, spacing, typography } from "./index";

/** The TS tokens are the typed mirror of the CSS vars in styles.css. Assert the contract holds. */
describe("design tokens", () => {
  it("every color maps to a CSS custom property", () => {
    for (const value of Object.values(colors)) {
      expect(value).toMatch(/^var\(--color-[a-z]+\)$/);
    }
  });

  it("exposes a spacing scale and type scale", () => {
    expect(Object.keys(spacing)).toContain("md");
    expect(typography.fontSans).toMatch(/system-ui/);
  });
});
