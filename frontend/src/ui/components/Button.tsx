import { cn } from "@/ui/lib/cn";
import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "ghost" | "danger";

const styles: Record<Variant, string> = {
  primary: "bg-[var(--color-primary)] text-white hover:opacity-90",
  ghost: "bg-transparent text-[var(--color-muted)] hover:text-[var(--color-text)]",
  danger: "bg-[var(--color-danger)] text-white hover:opacity-90",
};

export function Button({
  variant = "primary",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      type="button"
      className={cn(
        "rounded-[var(--radius-md)] px-4 py-2 text-sm font-medium transition disabled:opacity-40",
        styles[variant],
        className,
      )}
      {...props}
    />
  );
}
