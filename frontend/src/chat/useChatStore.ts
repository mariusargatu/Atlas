import { type PendingAction, api } from "@/api/client";
import { useCallback, useState } from "react";

/**
 * The chat external store — the integration logic the UI binds to (assistant-ui's
 * ExternalStoreRuntime, via src/chat/runtime.ts). One turn = one POST /chat; a write proposal pauses
 * on `pending` until /chat/resume. `customer_id` is NEVER sent — it rides in the bearer token.
 */
export type ChatMessage = { id: string; role: "user" | "assistant"; text: string };

export type ChatStore = {
  messages: ChatMessage[];
  pending: PendingAction | null;
  busy: boolean;
  send: (message: string) => Promise<void>;
  confirm: (confirmation: string) => Promise<void>;
};

let counter = 0;
const nextId = () => `m${counter++}`;
const newThreadId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `t-${nextId()}`;

export function useChatStore(): ChatStore {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState<PendingAction | null>(null);
  const [busy, setBusy] = useState(false);
  const [threadId] = useState(newThreadId);

  const append = useCallback((role: ChatMessage["role"], text: string) => {
    setMessages((prev) => [...prev, { id: nextId(), role, text }]);
  }, []);

  const send = useCallback(
    async (message: string) => {
      append("user", message);
      setBusy(true);
      try {
        const { data, error } = await api.POST("/chat", { body: { message, thread_id: threadId } });
        if (error || !data) {
          append("assistant", "Sorry — something went wrong. Please try again.");
          return;
        }
        if (data.type === "interrupt") {
          setPending(data.pending ?? null);
        } else {
          append("assistant", data.final_response ?? "");
        }
      } finally {
        setBusy(false);
      }
    },
    [append, threadId],
  );

  const confirm = useCallback(
    async (confirmation: string) => {
      setBusy(true);
      try {
        const { data } = await api.POST("/chat/resume", {
          body: { thread_id: threadId, confirmation },
        });
        setPending(null);
        if (data) append("assistant", data.final_response ?? "");
      } finally {
        setBusy(false);
      }
    },
    [append, threadId],
  );

  return { messages, pending, busy, send, confirm };
}
