import { useSession } from "@/auth/session";
import { AtlasThread } from "@/chat/AtlasThread";
import { useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";

export function Chat() {
  const { customerId } = useSession();
  const navigate = useNavigate();

  useEffect(() => {
    if (!customerId) navigate({ to: "/" });
  }, [customerId, navigate]);

  if (!customerId) return null;
  return <AtlasThread />;
}
