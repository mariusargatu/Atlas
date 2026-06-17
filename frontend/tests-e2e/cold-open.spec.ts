import { expect, test } from "@playwright/test";

/**
 * The cold open, end to end, against the REPLAYED backend (real render guard, real gateway,
 * committed cassette from `testing/harness/recording/seed_e2e_cassettes.py`). The model gives the
 * SAME grounded-but-false answer to every customer ("your plan is contract-free, no fee"); only the
 * account decides whether that is true. Daniel (`cust_legacy_term`) has a term and a fee, so the
 * render guard must hold the answer behind a safe handoff. `AtlasThread.test.tsx` proves this same
 * assertion against an MSW mock; this is the same story through the real graph and a real HTTP call.
 */
test("the render guard holds the grounded but false cold-open answer for the legacy customer", async ({
  page,
}) => {
  await page.goto("/");

  await page.getByRole("button", { name: /Daniel, legacy plan/ }).click();
  await expect(page).toHaveURL(/\/chat$/);

  await page.getByLabel("Message Atlas").fill("Is my plan contract-free?");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText(/safe handoff/i)).toBeVisible();
  await expect(page.getByText(/no fee|cancel any time/i)).not.toBeVisible();
});
