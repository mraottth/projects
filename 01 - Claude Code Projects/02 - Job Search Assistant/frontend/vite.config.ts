import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages answers a path it doesn't have with 404.html, so the demo's copy of index.html there
// lets a deep link (/projects/jsa-demo/tracker) or a reload load the app at that route.
const spaFallback: Plugin = {
  name: "demo-404",
  enforce: "post",
  generateBundle(_options, bundle) {
    const index = bundle["index.html"];
    if (index && index.type === "asset") {
      this.emitFile({ type: "asset", fileName: "404.html", source: index.source });
    }
  },
};

// `npm run dev` proxies API calls to `make serve` (uvicorn on :8000).
// `npm run build:demo` (mode "demo", .env.demo) builds the public, read-only demo (src/demo.ts),
// served from a sub-path (override with --base), with demo/record.py's recording (demo-build/)
// copied in as static files.
export default defineConfig(({ mode }) => {
  const demo = mode === "demo";
  return {
    plugins: [react(), ...(demo ? [spaFallback] : [])],
    base: demo ? "/projects/jsa-demo/" : "/",
    publicDir: demo ? "demo-build" : "public",
    build: { outDir: demo ? "dist-demo" : "dist" },
    server: { proxy: { "/api": "http://localhost:8000" } },
  };
});
