import { useSession } from "@/auth/session";
import { Button } from "@/ui/components/Button";
import { Link, Outlet } from "@tanstack/react-router";

export function Root() {
  const { customerId, name, signOut } = useSession();
  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-3">
        <Link to="/" className="text-sm font-semibold text-[var(--color-text)] no-underline">
          Atlas <span className="text-[var(--color-muted)]">Support</span>
        </Link>
        {customerId && (
          <div className="flex items-center gap-3 text-xs text-[var(--color-muted)]">
            <span>Signed in: {name ?? customerId}</span>
            <Button variant="ghost" onClick={signOut}>
              Sign out
            </Button>
          </div>
        )}
      </header>
      <main className="min-h-0 flex-1">
        <Outlet />
      </main>
    </div>
  );
}
