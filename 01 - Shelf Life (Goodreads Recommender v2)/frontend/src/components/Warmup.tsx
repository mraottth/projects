import { Fragment, useEffect, useState, type ReactNode } from "react";
import logo from "../assets/logo-header.png";

type Phase = "checking" | "warming" | "slow" | "error" | "ready";
const LONG_POLL_S = 15;     // the server holds /api/health up to this long while it loads (keeps Cloud Run's CPU on)
const SLOW_AFTER_MS = 25_000;

/**
 * Cold starts: the server answers within a second or two but loads the models in the background, answering /api/*
 * with 503 "warming" until they're in (main.py `Startup`). The app renders straight away; if the server says it's
 * warming, a loading screen covers it, and once the server is ready the app is re-mounted so it fetches afresh.
 * A warm server costs one /api/health call and shows nothing.
 */
export function WarmupGate({ children }: { children: ReactNode }) {
  const [phase, setPhase] = useState<Phase>("checking");
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    let stop = false, waited = false;
    const started = Date.now();
    const slow = setTimeout(() => setPhase((p) => (p === "warming" ? "slow" : p)), SLOW_AFTER_MS);
    (async () => {
      while (!stop) {
        try {
          const r = await fetch(`/api/health${waited ? `?wait=${LONG_POLL_S}` : ""}`);
          const h = await r.json();
          if (stop) return;
          if (h.ok || h.status === "ready") {
            if (waited) setEpoch((e) => e + 1);
            setPhase("ready");
            return;
          }
          if (h.status === "error") { setPhase("error"); return; }
        } catch {
          // Not up yet (or a proxy error page): try again shortly.
          if (stop) return;
          await new Promise((res) => setTimeout(res, 1500));
        }
        waited = true;
        setPhase((p) => (p === "checking" ? (Date.now() - started > SLOW_AFTER_MS ? "slow" : "warming") : p));
      }
    })();
    return () => { stop = true; clearTimeout(slow); };
  }, []);

  return (
    <>
      <Fragment key={epoch}>{children}</Fragment>
      {phase !== "checking" && phase !== "ready" && <WarmupScreen phase={phase} />}
    </>
  );
}

function WarmupScreen({ phase }: { phase: "warming" | "slow" | "error" }) {
  return (
    <div className="warmup" role="status" aria-live="polite">
      <div className="warmup-card">
        <img src={logo} alt="Shelf Life" width={125} height={48} />
        {phase === "error" ? (
          <>
            <h1>Shelf Life couldn&apos;t start</h1>
            <p className="muted">Something went wrong loading the recommender. Please try again in a minute.</p>
            <p><button type="button" className="primary" onClick={() => window.location.reload()}>Reload</button></p>
          </>
        ) : (
          <>
            <h1>Waking up Shelf Life…</h1>
            <p className="muted">
              The site naps when nobody&apos;s using it. Loading the book catalog and the recommendation models takes a few
              seconds.
            </p>
            <div className="warmup-bar" aria-hidden="true"><span /></div>
            {phase === "slow" && <p className="muted small">This is taking longer than usual. Almost there.</p>}
          </>
        )}
      </div>
    </div>
  );
}
