import type { components } from "@/api/generated/types";
import { Factory } from "fishery";

type PendingAction = components["schemas"]["PendingOut"];

/** Deterministic test data builders (learnings KEEP: Fishery factories). */
export const pendingActionFactory = Factory.define<PendingAction>(() => ({
  tool: "change_plan",
  args: { plan_id: "plan_current_fast" },
}));
