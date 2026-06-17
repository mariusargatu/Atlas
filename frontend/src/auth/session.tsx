import { api, setAccessToken } from "@/api/client";
import { type ReactNode, createContext, useCallback, useContext, useMemo, useState } from "react";

/**
 * Session = the identity seam (principle 1). The access token lives in MEMORY (never localStorage —
 * XSS), the refresh token is an httpOnly cookie the browser sends automatically. `customer_id` is
 * read back from the login response for display only; every API call derives identity server-side
 * from the token, never from anything the client asserts.
 */
type SessionState = {
  customerId: string | null;
  name: string | null; // display name, shown in the UI instead of the raw id
  signIn: (customerId: string) => Promise<boolean>;
  signOut: () => void;
};

const SessionContext = createContext<SessionState | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [customerId, setCustomerId] = useState<string | null>(null);
  const [name, setName] = useState<string | null>(null);

  const signIn = useCallback(async (id: string) => {
    const { data } = await api.POST("/auth/login", { body: { customer_id: id } });
    if (!data?.access_token) return false;
    setAccessToken(data.access_token);
    setCustomerId(data.customer_id);
    setName(data.name);
    return true;
  }, []);

  const signOut = useCallback(() => {
    setAccessToken(null);
    setCustomerId(null);
    setName(null);
  }, []);

  const value = useMemo(
    () => ({ customerId, name, signIn, signOut }),
    [customerId, name, signIn, signOut],
  );
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionState {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used within a SessionProvider");
  return ctx;
}
