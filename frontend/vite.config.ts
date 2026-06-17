import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react-swc";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    // Dev proxy: the SPA calls /api/* → FastAPI edge, so the bearer cookie stays same origin.
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  test: {
    // jsdom (not happy-dom): happy-dom 15's Request locks the body stream under MSW request.json().
    // The learnings rule (happy-dom by default, jsdom when an API is missing) lands here.
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    // undici (node fetch) can't resolve a relative base, so give tests an absolute one. MSW's relative
    // handler patterns still match it. In the browser/dev the default "/api" + Vite proxy is used.
    env: { VITE_API_BASE: "http://localhost/api" },
    css: false,
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
