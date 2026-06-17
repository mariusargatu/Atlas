import { cn } from "@/ui/lib/cn";
import { type TextareaHTMLAttributes, forwardRef } from "react";

export const Textarea = forwardRef<
  HTMLTextAreaElement,
  TextareaHTMLAttributes<HTMLTextAreaElement>
>(function Textarea({ className, ...props }, ref) {
  return (
    <textarea
      ref={ref}
      className={cn(
        "rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)]",
        "px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-primary)]",
        className,
      )}
      {...props}
    />
  );
});
