import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import { ConfirmationCard } from "./ConfirmationCard";
import { useAtlasRuntime } from "./runtime";
import { useChatStore } from "./useChatStore";

/**
 * The chat surface, rendered with assistant-ui primitives over our custom runtime (ADR-028).
 * A held `[safe handoff]` reply renders like any other assistant message, the cold open made
 * visible. A write proposal hides the composer and surfaces the typed CONFIRM gate instead.
 */
function Message() {
  return (
    <MessagePrimitive.Root className="my-1 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm aria-[roledescription=user]:self-end">
      <MessagePrimitive.Content />
    </MessagePrimitive.Root>
  );
}

export function AtlasThread() {
  const store = useChatStore();
  const runtime = useAtlasRuntime(store);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="mx-auto flex h-full max-w-2xl flex-col gap-3 p-4">
        <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col">
          <ThreadPrimitive.Viewport className="flex flex-1 flex-col gap-1 overflow-y-auto">
            <ThreadPrimitive.Messages components={{ Message }} />
          </ThreadPrimitive.Viewport>
        </ThreadPrimitive.Root>

        {store.pending ? (
          <ConfirmationCard pending={store.pending} busy={store.busy} onConfirm={store.confirm} />
        ) : (
          <ComposerPrimitive.Root className="flex gap-2">
            <ComposerPrimitive.Input
              aria-label="Message Atlas"
              placeholder="Ask about your plan, bill, or make a change…"
              className="flex-1 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-primary)]"
            />
            <ComposerPrimitive.Send className="rounded-[var(--radius-md)] bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-white transition disabled:opacity-40">
              Send
            </ComposerPrimitive.Send>
          </ComposerPrimitive.Root>
        )}
      </div>
    </AssistantRuntimeProvider>
  );
}
