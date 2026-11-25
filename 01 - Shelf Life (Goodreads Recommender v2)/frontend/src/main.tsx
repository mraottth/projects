import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { WarmupGate } from "./components/Warmup";
import { ShelfProvider } from "./store";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ShelfProvider>
      <WarmupGate>
        <App />
      </WarmupGate>
    </ShelfProvider>
  </StrictMode>,
);
