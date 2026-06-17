import { type AppendMessage, useExternalStoreRuntime } from "@assistant-ui/react";
import type { ChatMessage, ChatStore } from "./useChatStore";

/**
 * The custom assistant-ui runtime (ADR-028): an ExternalStoreRuntime over our chat store. assistant-ui
 * owns the message/composer primitives; the store owns the integration logic (POST /chat). This is the
 * seam — swap the store's transport and the UI is unchanged.
 */
function textOf(content: AppendMessage["content"]): string {
  return content
    .filter((p): p is { type: "text"; text: string } => p.type === "text")
    .map((p) => p.text)
    .join("");
}

export function useAtlasRuntime(store: ChatStore) {
  return useExternalStoreRuntime<ChatMessage>({
    isRunning: store.busy,
    messages: store.messages,
    convertMessage: (m) => ({ role: m.role, content: [{ type: "text", text: m.text }], id: m.id }),
    onNew: async (message) => {
      await store.send(textOf(message.content));
    },
  });
}
