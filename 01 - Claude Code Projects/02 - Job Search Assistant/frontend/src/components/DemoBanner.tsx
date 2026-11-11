import { useEffect, useState } from "react";
import { DEMO_READONLY_EVENT, DEMO_READONLY_MESSAGE } from "../demo";
import { TOUR_EVENT } from "./Tour";

// Where "Request access" goes: the real app, set at build time (the GitHub workflow passes the repo
// variable JSA_APP_URL). Without it the banner just says what the demo is.
const APP_URL = import.meta.env.VITE_DEMO_APP_URL as string | undefined;

/** The public demo's strip under the top bar, plus a toast whenever a change is attempted (demo.ts). */
export function DemoBanner() {
  const [toast, setToast] = useState(false);
  useEffect(() => {
    let timer: number | undefined;
    const show = () => {
      setToast(true);
      window.clearTimeout(timer);
      timer = window.setTimeout(() => setToast(false), 3500);
    };
    window.addEventListener(DEMO_READONLY_EVENT, show);
    return () => {
      window.removeEventListener(DEMO_READONLY_EVENT, show);
      window.clearTimeout(timer);
    };
  }, []);
  return (
    <>
      <div className="demo-banner" role="note">
        {/* The welcome page published next to the demo (demo/landing/, D-033). */}
        <a className="demo-banner-home" href={`${import.meta.env.BASE_URL}../job-search/`}>
          ← Back<span className="demo-banner-long"> to the welcome page</span>
        </a>
        <strong>Demo</strong>
        <span className="demo-banner-text" title="Everything here is fictional: Jordan Rivera's search, with made-up organizations and jobs. Look around freely; nothing can be changed.">
          A fictional job seeker's search. Look around freely; nothing can be changed.
        </span>
        <span className="demo-banner-links">
          <button className="demo-banner-tour" onClick={() => window.dispatchEvent(new Event(TOUR_EVENT))}>
            Take the tour
          </button>
          {APP_URL && (
            <a className="demo-banner-cta" href={APP_URL}>
              Request access →
            </a>
          )}
        </span>
      </div>
      {toast && (
        <div className="demo-toast" role="status">
          {DEMO_READONLY_MESSAGE}
        </div>
      )}
    </>
  );
}
