import { server } from "@/test/msw/server";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { Adjudicate } from "./Adjudicate";

describe("Adjudicate (HITL page)", () => {
  it("renders the question, answer, and retrieved chunk text for the first item", async () => {
    render(<Adjudicate />);
    expect(await screen.findByText("Is my plan contract-free?")).toBeInTheDocument();
    expect(screen.getByText("Yes.")).toBeInTheDocument();
    expect(screen.getByText("d1").closest("li")).toHaveTextContent("contract-free plan");
  });

  it("highlights a registry fact value inside the retrieved chunk text", async () => {
    render(<Adjudicate />);
    await screen.findByText("Is my plan contract-free?");
    const mark = document.querySelector("mark");
    expect(mark?.textContent).toBe("contract-free");
  });

  it("shows the initial progress counter from the server", async () => {
    render(<Adjudicate />);
    expect(await screen.findByLabelText("Progress")).toHaveTextContent("0 / 2");
  });

  it("requires a critique before Pass or Fail can submit", async () => {
    const user = userEvent.setup();
    render(<Adjudicate />);
    await screen.findByText("Is my plan contract-free?");

    await user.click(screen.getByRole("button", { name: "Pass (P)" }));

    expect(await screen.findByText(/critique is required/i)).toBeInTheDocument();
    // still on the same item -- nothing was submitted
    expect(screen.getByText("Is my plan contract-free?")).toBeInTheDocument();
  });

  it("Pass button submits the critique and advances to the next item, progress incremented", async () => {
    const user = userEvent.setup();
    render(<Adjudicate />);
    await screen.findByText("Is my plan contract-free?");

    await user.type(screen.getByLabelText(/Critique/i), "Fully grounded in the cited page.");
    await user.click(screen.getByRole("button", { name: "Pass (P)" }));

    expect(await screen.findByText("Data cap?")).toBeInTheDocument();
    expect(screen.getByLabelText("Progress")).toHaveTextContent("1 / 2");
  });

  it("the P keyboard shortcut submits pass once a critique is typed, outside the textarea", async () => {
    const user = userEvent.setup();
    render(<Adjudicate />);
    await screen.findByText("Is my plan contract-free?");

    await user.type(screen.getByLabelText(/Critique/i), "Fully grounded in the cited page.");
    await user.tab(); // leave the textarea so the shortcut is not swallowed by typing
    await user.keyboard("p");

    expect(await screen.findByText("Data cap?")).toBeInTheDocument();
  });

  it("the F keyboard shortcut submits fail", async () => {
    server.use(
      http.post("*/api/labels", async ({ request }) => {
        const body = (await request.json()) as { verdict: string };
        expect(body.verdict).toBe("fail");
        return HttpResponse.json({ progress: { labeled: 1, total: 2 } });
      }),
    );
    const user = userEvent.setup();
    render(<Adjudicate />);
    await screen.findByText("Is my plan contract-free?");

    await user.type(screen.getByLabelText(/Critique/i), "Unsupported claim about the fee.");
    await user.tab();
    await user.keyboard("f");

    expect(await screen.findByText("Data cap?")).toBeInTheDocument();
  });

  it("typing the letter p inside the critique textarea does not trigger the shortcut", async () => {
    const user = userEvent.setup();
    render(<Adjudicate />);
    await screen.findByText("Is my plan contract-free?");

    await user.type(screen.getByLabelText(/Critique/i), "possibly grounded");

    // still on the same item: the keystrokes went into the textarea, not a submit
    expect(screen.getByText("Is my plan contract-free?")).toBeInTheDocument();
    expect(screen.getByLabelText(/Critique/i)).toHaveValue("possibly grounded");
  });

  it("shows an 'all caught up' state once every item is labeled", async () => {
    server.use(
      http.get("*/api/labels/items", () =>
        HttpResponse.json({ items: [], progress: { labeled: 0, total: 0 } }),
      ),
    );
    render(<Adjudicate />);
    expect(await screen.findByText(/all caught up/i)).toBeInTheDocument();
  });
});
