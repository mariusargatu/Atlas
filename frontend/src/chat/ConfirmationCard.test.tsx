import { pendingActionFactory } from "@/test/factories";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ConfirmationCard } from "./ConfirmationCard";

describe("ConfirmationCard", () => {
  it("only enables confirm for the exact typed token, and emits it", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    const pending = pendingActionFactory.build({
      tool: "change_plan",
      args: { plan_id: "plan_current_fast" },
    });
    render(<ConfirmationCard pending={pending} busy={false} onConfirm={onConfirm} />);

    const confirm = screen.getByRole("button", { name: "Confirm" });
    const input = screen.getByLabelText(/Type/i);

    await user.type(input, "yes");
    expect(confirm).toBeDisabled();

    await user.clear(input);
    await user.type(input, "CONFIRM");
    expect(confirm).toBeEnabled();
    await user.click(confirm);
    expect(onConfirm).toHaveBeenCalledWith("CONFIRM");
  });

  it("shows the proposed tool and its args", () => {
    const pending = pendingActionFactory.build();
    render(<ConfirmationCard pending={pending} busy={false} onConfirm={() => {}} />);
    expect(screen.getByText("change_plan")).toBeInTheDocument();
    expect(screen.getByText(/plan_current_fast/)).toBeInTheDocument();
  });
});
