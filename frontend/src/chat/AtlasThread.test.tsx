import { setAccessToken } from "@/api/client";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { AtlasThread } from "./AtlasThread";

describe("AtlasThread (assistant-ui runtime)", () => {
  beforeEach(() => setAccessToken("test-access")); // token in memory; skip the login screen

  async function ask(text: string) {
    const user = userEvent.setup();
    render(<AtlasThread />);
    await user.type(screen.getByLabelText("Message Atlas"), text);
    await user.click(screen.getByRole("button", { name: "Send" }));
    return user;
  }

  it("renders the held safe-handoff for the cold-open, not the false answer", async () => {
    await ask("Is my plan contract-free?");
    expect(await screen.findByText(/safe handoff/i)).toBeInTheDocument();
    expect(screen.queryByText(/no fee|cancel any time/i)).not.toBeInTheDocument();
  });

  it("requires a typed CONFIRM for a write, then shows the reference", async () => {
    const user = await ask("Switch me to the fast plan");

    const card = await screen.findByRole("region", { name: "Action confirmation" });
    const confirmBtn = screen.getByRole("button", { name: "Confirm" });
    expect(confirmBtn).toBeDisabled();

    const input = screen.getByLabelText(/Type/i);
    await user.type(input, "yes");
    expect(confirmBtn).toBeDisabled();

    await user.clear(input);
    await user.type(input, "CONFIRM");
    expect(confirmBtn).toBeEnabled();
    await user.click(confirmBtn);

    expect(await screen.findByText(/Done\. Your reference is/i)).toBeInTheDocument();
    await waitFor(() => expect(card).not.toBeInTheDocument());
  });
});
