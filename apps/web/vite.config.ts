import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // recharts pulls react in through CJS deps (redux); dedupe so the app and the
  // charts share one React instance (otherwise: "Invalid hook call").
  resolve: {
    dedupe: ["react", "react-dom"],
  },
  optimizeDeps: {
    include: ["recharts", "react", "react-dom"],
  },
  server: {
    port: 5173,
    strictPort: false,
  },
});
