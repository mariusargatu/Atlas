import { cn } from "@/ui/lib/cn";
import type { InputHTMLAttributes } from "react";

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)]",
        "px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-primary)]",
        className,
      )}
      {...props}
    />
  );
}
