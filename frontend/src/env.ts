import { z } from "zod";

/**
 * Runtime env validation at boot. Fail fast on misconfig (learnings ADJUST).
 * `VITE_API_BASE` is where the FastAPI edge lives. In dev the Vite proxy maps /api → :8000.
 */
const schema = z.object({
  VITE_API_BASE: z.string().default("/api"),
});

export const env = schema.parse({
  VITE_API_BASE: import.meta.env.VITE_API_BASE,
});
