import type { PendingAction } from "@/api/client";
import { Button } from "@/ui/components/Button";
import { Input } from "@/ui/components/Input";
import { useState } from "react";
import { REQUIRED_CONFIRMATION, isTypedConfirmation } from "./confirmation";

/**
 * The propose → typed CONFIRM → execute gate, surfaced as a card. The agent has paused on an
 * irreversible write (LangGraph interrupt). The human must type CONFIRM. A bare "yes" is refused
 * on the client for UX. The server refuses it for real.
 */
export function ConfirmationCard({
  pending,
  busy,
  onConfirm,
}: {
  pending: PendingAction;
  busy: boolean;
  onConfirm: (confirmation: string) => void;
}) {
  const [typed, setTyped] = useState("");
  const valid = isTypedConfirmation(typed);

  return (
    <section
      aria-label="Action confirmation"
      className="rounded-[var(--radius-md)] border border-[var(--color-warn)] bg-[var(--color-surface)] p-4"
    >
      <h2 className="m-0 text-sm font-semibold text-[var(--color-warn)]">Confirm this change</h2>
      <p className="mt-1 text-sm text-[var(--color-muted)]">
        Atlas wants to run <code className="text-[var(--color-text)]">{pending.tool}</code> with:
      </p>
      <pre className="mt-2 overflow-x-auto rounded bg-[var(--color-bg)] p-2 text-xs text-[var(--color-text)]">
        {JSON.stringify(pending.args, null, 2)}
      </pre>
      <label htmlFor="confirm-input" className="mt-3 block text-xs text-[var(--color-muted)]">
        Type <strong>{REQUIRED_CONFIRMATION}</strong> to proceed
      </label>
      <div className="mt-1 flex gap-2">
        <Input
          id="confirm-input"
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder={REQUIRED_CONFIRMATION}
          aria-invalid={typed.length > 0 && !valid}
        />
        <Button disabled={!valid || busy} onClick={() => onConfirm(typed)}>
          Confirm
        </Button>
      </div>
    </section>
  );
}
