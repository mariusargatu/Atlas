/**
 * The confirmation rule, mirrored from the backend (`domain/confirmation.py`): an irreversible
 * action needs a TYPED confirmation, not a bare "yes". Kept pure so it unit-tests without a DOM.
 * The server re-checks this — the client gate is UX, never the security boundary.
 */
export const REQUIRED_CONFIRMATION = "CONFIRM";

export function isTypedConfirmation(input: string): boolean {
  return input.trim() === REQUIRED_CONFIRMATION;
}
