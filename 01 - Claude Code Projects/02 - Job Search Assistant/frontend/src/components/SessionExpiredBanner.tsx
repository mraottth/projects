import { useEffect, useState } from "react";
import { api, SESSION_EXPIRED_EVENT, SESSION_EXPIRED_MESSAGE } from "../api";

// Back on the tab after this long: check the sign-in straight away, so the banner can show before
// the person clicks into a page that would come up empty.
const AWAY_MS = 5 * 60_000;

/**
 * Shown when Google's sign-in (IAP) session has run out after time away (asked 2026-11-10): the
 * page's data requests fail until a reload, which makes pages look empty. Says the data is safe
 * and offers the reload. api.ts's apiFetch raises SESSION_EXPIRED_EVENT on any 401.
 */
export function SessionExpiredBanner() {
  const [expired, setExpired] = useState(false);

  useEffect(() => {
    const onExpired = () => setExpired(true);
    window.addEventListener(SESSION_EXPIRED_EVENT, onExpired);
    let hiddenAt: number | null = null;
    const onVisibility = () => {
      if (document.hidden) {
        hiddenAt = Date.now();
      } else if (hiddenAt !== null && Date.now() - hiddenAt > AWAY_MS) {
        hiddenAt = null;
        api.getMe().catch(() => {}); // a 401 raises the event
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener(SESSION_EXPIRED_EVENT, onExpired);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  if (!expired) return null;
  return (
    <div className="session-banner" role="alert">
      <span>{SESSION_EXPIRED_MESSAGE}</span>
      <button className="primary" onClick={() => window.location.reload()}>
        Reload
      </button>
    </div>
  );
}
