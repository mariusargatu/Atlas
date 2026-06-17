import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/** className composition (learnings KEEP): clsx for conditionals + tailwind-merge for conflicts. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
