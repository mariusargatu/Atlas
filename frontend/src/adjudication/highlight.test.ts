import { describe, expect, it } from "vitest";
import { highlightFacts } from "./highlight";

describe("highlightFacts", () => {
  it("returns the whole text unhighlighted when no fact values are given", () => {
    expect(highlightFacts("Fiber 500 costs 39.99 per month.", [])).toEqual([
      { text: "Fiber 500 costs 39.99 per month.", highlighted: false, start: 0 },
    ]);
  });

  it("marks a single matching fact value", () => {
    const segments = highlightFacts("Fiber 500 costs 39.99 per month.", ["39.99"]);
    expect(segments).toEqual([
      { text: "Fiber 500 costs ", highlighted: false, start: 0 },
      { text: "39.99", highlighted: true, start: 16 },
      { text: " per month.", highlighted: false, start: 21 },
    ]);
  });

  it("marks every occurrence of a value and multiple distinct values", () => {
    const segments = highlightFacts("Fiber 500 is Fiber 500, our fastest plan.", [
      "Fiber 500",
      "fastest",
    ]);
    const highlighted = segments.filter((s) => s.highlighted).map((s) => s.text);
    expect(highlighted).toEqual(["Fiber 500", "Fiber 500", "fastest"]);
  });

  it("gives every segment a unique, stable start offset usable as a React key", () => {
    const segments = highlightFacts("Fiber 500 is Fiber 500, our fastest plan.", [
      "Fiber 500",
      "fastest",
    ]);
    const starts = segments.map((s) => s.start);
    expect(new Set(starts).size).toBe(starts.length);
  });

  it("is case insensitive", () => {
    const segments = highlightFacts("contract-free plan", ["Contract-Free"]);
    expect(segments.some((s) => s.highlighted && s.text === "contract-free")).toBe(true);
  });

  it("prefers the longest match so a short value never splits a longer one", () => {
    const segments = highlightFacts("The fee is 39.99 total.", ["9", "39.99"]);
    const highlighted = segments.filter((s) => s.highlighted).map((s) => s.text);
    expect(highlighted).toEqual(["39.99"]);
  });

  it("ignores blank and whitespace only fact values", () => {
    const segments = highlightFacts("plain text", ["", "   "]);
    expect(segments).toEqual([{ text: "plain text", highlighted: false, start: 0 }]);
  });

  it("handles a value with regex special characters safely", () => {
    const segments = highlightFacts("cost: $39.99 (approx)", ["$39.99"]);
    expect(segments.some((s) => s.highlighted && s.text === "$39.99")).toBe(true);
  });

  it("returns the text unchanged when nothing matches", () => {
    const segments = highlightFacts("no mention here", ["Fiber 500"]);
    expect(segments).toEqual([{ text: "no mention here", highlighted: false, start: 0 }]);
  });
});
