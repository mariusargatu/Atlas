import { highlightFacts } from "@/adjudication/highlight";
import { useAdjudicationStore } from "@/adjudication/useAdjudicationStore";
import { Button } from "@/ui/components/Button";
import { Textarea } from "@/ui/components/Textarea";
import { useEffect, useRef, useState } from "react";

/**
 * The HITL adjudication page (SP8 Task 4, label collection half, pulled early): the collection
 * tool for the ~200 item human vs judge calibration set. Question, answer, retrieved chunks with
 * registry facts highlighted, pass/fail keyboard shortcuts, a required one sentence critique, a
 * progress counter, fixed seed order (D30: no managed queue -- items are walked in the order the
 * backend returns them, never reordered by this page).
 *
 * Not gated behind customer sign in: this is an internal tool for the person running the labeling
 * session, not the customer facing chat surface.
 */
export function Adjudicate() {
  const store = useAdjudicationStore();
  const [critique, setCritique] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  async function decide(verdict: "pass" | "fail") {
    if (!critique.trim()) {
      setValidationError("A one sentence critique is required before you can submit a verdict.");
      return;
    }
    setValidationError(null);
    const ok = await store.submit(verdict, critique.trim());
    // A successful submit moves `store.current` to the next item; clear the critique box for it.
    // A failed submit (a rejected label) leaves the critique text in place so nothing typed is lost.
    if (ok) setCritique("");
  }

  // `decideRef` always holds the LATEST `decide` closure (reassigned every render, a plain read
  // during render, not an effect) so the keydown listener below can call the current one without
  // needing `decide` (which is recreated every render) in its own dependency array.
  const decideRef = useRef(decide);
  decideRef.current = decide;

  // Global pass/fail shortcuts (P/F), ignored while the critique textarea has focus so ordinary
  // typing is never hijacked. Registered once, resubscribed only when `store.busy`/`store.current`
  // actually change value (both are read directly in the handler below).
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (document.activeElement === textareaRef.current || store.busy || !store.current) return;
      if (event.key === "p" || event.key === "P") {
        event.preventDefault();
        void decideRef.current("pass");
      } else if (event.key === "f" || event.key === "F") {
        event.preventDefault();
        void decideRef.current("fail");
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [store.busy, store.current]);

  if (store.loading) {
    return <p className="p-8 text-sm text-[var(--color-muted)]">Loading the label set...</p>;
  }
  if (store.error && !store.current) {
    return <p className="p-8 text-sm text-[var(--color-danger)]">{store.error}</p>;
  }
  if (store.done || !store.current) {
    return (
      <div className="mx-auto max-w-2xl p-8">
        <h1 className="text-lg font-semibold text-[var(--color-text)]">All caught up</h1>
        <p className="mt-2 text-sm text-[var(--color-muted)]">
          Every item in this set has an adjudicator label.
          {store.progress && ` ${store.progress.labeled} / ${store.progress.total} labeled.`}
        </p>
      </div>
    );
  }

  const item = store.current;
  const factValues = item.registry_facts.map((f) => f.value);

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-4 p-8">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold text-[var(--color-text)]">Adjudicate</h1>
        {store.progress && (
          <span aria-label="Progress" className="text-sm text-[var(--color-muted)]">
            {store.progress.labeled} / {store.progress.total}
          </span>
        )}
      </div>

      <section aria-label="Question">
        <h2 className="text-xs font-semibold uppercase text-[var(--color-muted)]">Question</h2>
        <p className="mt-1 text-sm text-[var(--color-text)]">{item.question}</p>
      </section>

      <section aria-label="Answer">
        <h2 className="text-xs font-semibold uppercase text-[var(--color-muted)]">Answer</h2>
        <p className="mt-1 text-sm text-[var(--color-text)]">{item.answer}</p>
      </section>

      <section aria-label="Retrieved chunks">
        <h2 className="text-xs font-semibold uppercase text-[var(--color-muted)]">
          Retrieved chunks
        </h2>
        <ul className="mt-1 flex flex-col gap-2">
          {item.retrieved_chunks.map((chunk) => (
            <li
              key={`${chunk.doc_id}:${chunk.chunk_id}`}
              className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3 text-sm"
            >
              <p className="mb-1 text-xs text-[var(--color-muted)]">{chunk.doc_id}</p>
              <p>
                {highlightFacts(chunk.text, factValues).map((segment) =>
                  segment.highlighted ? (
                    <mark key={segment.start} className="rounded bg-[var(--color-warn)]/40 px-0.5">
                      {segment.text}
                    </mark>
                  ) : (
                    <span key={segment.start}>{segment.text}</span>
                  ),
                )}
              </p>
            </li>
          ))}
        </ul>
      </section>

      <label
        htmlFor="critique"
        className="text-xs font-semibold uppercase text-[var(--color-muted)]"
      >
        Critique (one sentence, required)
      </label>
      <Textarea
        id="critique"
        ref={textareaRef}
        value={critique}
        onChange={(e) => setCritique(e.target.value)}
        rows={2}
        placeholder="Why does this answer pass or fail groundedness?"
      />
      {validationError && <p className="text-sm text-[var(--color-danger)]">{validationError}</p>}
      {store.error && <p className="text-sm text-[var(--color-danger)]">{store.error}</p>}

      <div className="flex gap-2">
        <Button onClick={() => decide("pass")} disabled={store.busy}>
          Pass (P)
        </Button>
        <Button variant="danger" onClick={() => decide("fail")} disabled={store.busy}>
          Fail (F)
        </Button>
      </div>
    </div>
  );
}
