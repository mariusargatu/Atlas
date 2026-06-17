import { defineConfig, devices } from "@playwright/test";

/**
 * E2E runs in CI for real (learnings ADJUST, not commented out). The webServer block boots the Vite
 * dev server, and the FastAPI edge is expected on :8000 (CI starts it with the replay gateway, hermetic).
 */
export default defineConfig({
  testDir: "./tests-e2e",
  fullyParallel: false,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: "http://localhost:5173", trace: "on-first-retry" },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: "pnpm dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
  },
});
