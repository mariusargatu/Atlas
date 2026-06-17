import { api } from "@/api/client";
import type { components } from "@/api/generated/types";
import { useCallback, useEffect, useState } from "react";

type LabelItem = components["schemas"]["LabelItemOut"];
type Progress = components["schemas"]["ProgressOut"];
type Verdict = "pass" | "fail";

type AdjudicationStore = {
  items: LabelItem[];
  current: LabelItem | null;
  index: number;
  progress: Progress | null;
  loading: boolean;
  busy: boolean;
  done: boolean;
  error: string | null;
  submit: (verdict: Verdict, critique: string) => Promise<boolean>;
};

/**
 * The HITL page's own store: loads the item set ONCE, in the FIXED SEED ORDER the backend already
 * returns it in (D30: no managed queue -- this hook never reorders, shuffles, or skips), then walks
 * it forward one item per successful submit. `progress` always mirrors the server's own count
 * (`atlas.label_routes`'s `ProgressOut`), never a client side guess, so a reload or a second tab
 * stays honest about how much of the set is actually labeled.
 */
export function useAdjudicationStore(): AdjudicationStore {
  const [items, setItems] = useState<LabelItem[]>([]);
  const [index, setIndex] = useState(0);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const { data, error: fetchError } = await api.GET("/labels/items");
      if (cancelled) return;
      if (fetchError || !data) {
        setError("Could not load the label item set.");
      } else {
        setItems(data.items);
        setProgress(data.progress);
      }
      setLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const submit = useCallback(
    async (verdict: Verdict, critique: string): Promise<boolean> => {
      const item = items[index];
      if (!item) return false;
      setBusy(true);
      setError(null);
      try {
        const { data, error: postError } = await api.POST("/labels", {
          body: { trace_id: item.trace_id, verdict, critique, role: "adjudicator" },
        });
        if (postError || !data) {
          setError("Could not save that label. Check the critique and try again.");
          return false;
        }
        setProgress(data.progress);
        setIndex((i) => i + 1);
        return true;
      } finally {
        setBusy(false);
      }
    },
    [items, index],
  );

  return {
    items,
    current: items[index] ?? null,
    index,
    progress,
    loading,
    busy,
    done: !loading && items.length > 0 && index >= items.length,
    error,
    submit,
  };
}
