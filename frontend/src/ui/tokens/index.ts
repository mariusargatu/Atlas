/**
 * Tokens as typed TS objects (learnings KEEP) — the source the Tailwind v4 `@theme` CSS vars in
 * styles.css mirror. Programmatic access + autocomplete on one side, CSS-var output on the other.
 */
export const colors = {
  bg: "var(--color-bg)",
  surface: "var(--color-surface)",
  border: "var(--color-border)",
  text: "var(--color-text)",
  muted: "var(--color-muted)",
  primary: "var(--color-primary)",
  danger: "var(--color-danger)",
  warn: "var(--color-warn)",
} as const;

export const spacing = {
  xs: "0.25rem",
  sm: "0.5rem",
  md: "1rem",
  lg: "1.5rem",
  xl: "2rem",
} as const;

export const typography = {
  fontSans: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  sizeSm: "0.875rem",
  sizeMd: "1rem",
  sizeLg: "1.25rem",
} as const;
