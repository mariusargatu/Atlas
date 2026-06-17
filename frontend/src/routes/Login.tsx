import { useSession } from "@/auth/session";
import { Button } from "@/ui/components/Button";
import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";

/** Demo sign-in AS a seeded customer (no password — real IdP is out of scope, 00-overview). */
const SEEDED = [
  { id: "cust_current", label: "Sarah — current plan (term-free)" },
  { id: "cust_legacy_term", label: "Daniel — legacy plan (has a term)" },
];

export function Login() {
  const { signIn } = useSession();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  async function pick(id: string) {
    setError(null);
    const ok = await signIn(id);
    if (ok) navigate({ to: "/chat" });
    else setError("Sign-in failed.");
  }

  return (
    <div className="mx-auto flex max-w-md flex-col gap-4 p-8">
      <h1 className="text-lg font-semibold">Sign in</h1>
      <p className="text-sm text-[var(--color-muted)]">
        Pick a seeded customer. Identity comes from the session — never from anything you type into
        chat.
      </p>
      {SEEDED.map((c) => (
        <Button
          key={c.id}
          variant="ghost"
          className="justify-start border border-[var(--color-border)]"
          onClick={() => pick(c.id)}
        >
          {c.label}
        </Button>
      ))}
      {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}
    </div>
  );
}
