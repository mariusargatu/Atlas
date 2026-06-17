import { expect, test } from "@playwright/test";

/**
 * The propose -> interrupt -> confirm -> execute write protocol, end to end against the REPLAYED
 * backend: a real LangGraph interrupt, a real typed-CONFIRM gate, and (since the step-up fix) a
 * real `/auth/step-up` call before `/chat/resume`, which requires "write" scope that login alone
 * never grants (ADR-027, least agency). `AtlasThread.test.tsx` proves the same UI contract against
 * an MSW mock; this proves the real client actually clears the real server's scope gate.
 */
test("a typed CONFIRM clears the write gate and lands the change; a bare 'yes' does not", async ({
  page,
}) => {
  await page.goto("/");

  await page.getByRole("button", { name: /Sarah, current plan/ }).click();
  await expect(page).toHaveURL(/\/chat$/);

  await page.getByLabel("Message Atlas").fill("Switch me to the fast plan");
  await page.getByRole("button", { name: "Send" }).click();

  const card = page.getByRole("region", { name: "Action confirmation" });
  await expect(card).toBeVisible();
  const confirmBtn = page.getByRole("button", { name: "Confirm" });
  await expect(confirmBtn).toBeDisabled();

  const input = page.getByLabel(/Type/i);
  await input.fill("yes");
  await expect(confirmBtn).toBeDisabled();

  await input.fill("CONFIRM");
  await expect(confirmBtn).toBeEnabled();
  await confirmBtn.click();

  await expect(page.getByText(/Done\. Your reference is/i)).toBeVisible();
  await expect(card).not.toBeVisible();
});
