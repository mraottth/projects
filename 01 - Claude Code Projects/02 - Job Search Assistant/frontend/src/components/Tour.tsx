import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../api";
import { DEMO } from "../demo";

/**
 * The guided tour (asked 2026-11-11): one step per tab, each opening that page with a short note
 * under its tab about what it's for. It never blocks the page: clicking a tab mid-tour jumps to that
 * tab's step. "Take the tour" (the demo bar, the About page) restarts it via TOUR_EVENT.
 *
 * When it starts on its own differs by build:
 * - the demo: a first visit to its home page, remembered in localStorage;
 * - the app: once after onboarding. Finishing or skipping onboarding sets the person's `tour_pending`
 *   setting, and ending the tour clears it, so it doesn't come back on another device. People set up
 *   before the tour existed never have the flag.
 */

export const TOUR_EVENT = "jobhunt:tour";
const DONE_KEY = "jsa-demo-tour-done";

interface Step {
  path: string;
  /** Finds the tab the note points at. */
  selector: string;
  title: string;
  body: string;
}

const DEMO_STEPS: Step[] = [
  {
    path: "/",
    selector: '.topbar nav a[data-path="/"]',
    title: "New & Noteworthy",
    body: "Each day's best new matches from the organizations and boards you follow, rated for how much you'd want the job and how strong a candidate you'd be, with the reasons highlighted.",
  },
  {
    path: "/all-jobs",
    selector: '.topbar nav a[data-path="/all-jobs"]',
    title: "All Jobs",
    body: "Everything your sources turned up, ranked best first. Filter and sort it any way you like; your location and salary dealbreakers hide what's clearly not for you.",
  },
  {
    path: "/tracker",
    selector: '.topbar nav a[data-path="/tracker"]',
    title: "My Jobs",
    body: "Track every application from draft to offer, see what's getting interviews, and tailor your resume and cover letter for any job with one click.",
  },
  {
    path: "/jobby",
    selector: '.topbar nav a[data-path="/jobby"]',
    title: "Jobby",
    body: "An assistant that knows your background and the jobs you like: mock interviews, likely questions, strategy. Open one of Jordan's chats on the left.",
  },
  {
    path: "/sources",
    selector: '.topbar nav a[data-path="/sources"]',
    title: "Manage Sources",
    body: "Pick the organizations and job boards to follow, or ask for suggestions that suit you. Every one is read for new postings each day.",
  },
  {
    path: "/profile",
    selector: '.topbar nav a[data-path="/profile"]',
    title: "Profile",
    body: "Your resume, what you're looking for, and the profile built from them. It's what every rating and tailored document is based on.",
  },
  {
    path: "/settings",
    selector: ".topbar .settings-gear",
    title: "Settings",
    body: "Your daily email, what the app has spent on your behalf, and your data: download it, start over, or delete your account.",
  },
];

// The same tabs for a real person, about their own data.
const APP_STEPS: Step[] = DEMO_STEPS.map((step) =>
  step.path === "/jobby"
    ? {
        ...step,
        body: "An assistant that knows your background and the jobs you like. Start a chat about your search, or prep for a job on My Jobs: mock interviews, likely questions, strategy.",
      }
    : step,
);
const STEPS = DEMO ? DEMO_STEPS : APP_STEPS;

function readDone(): boolean {
  try {
    return window.localStorage.getItem(DONE_KEY) === "1";
  } catch {
    return false;
  }
}

function markDone() {
  if (!DEMO) {
    api.putSettings({ tour_pending: false }).catch(() => undefined);
    return;
  }
  try {
    window.localStorage.setItem(DONE_KEY, "1");
  } catch {
    // Private browsing etc.: the tour just offers itself again next time.
  }
}

interface Place {
  top: number;
  left: number;
  arrow: number;
  ring: { top: number; left: number; width: number; height: number };
}

const POPOVER_WIDTH = 340;
const PHONE = 760;

export function Tour() {
  const location = useLocation();
  const navigate = useNavigate();
  // The step showing, or null when the tour isn't running.
  const [step, setStep] = useState<number | null>(() =>
    DEMO && location.pathname === "/" && !readDone() ? 0 : null,
  );
  const [place, setPlace] = useState<Place | null>(null);
  const [phone, setPhone] = useState(() => window.innerWidth < PHONE);

  const go = useCallback(
    (n: number) => {
      setStep(n);
      navigate(STEPS[n].path);
    },
    [navigate],
  );
  const end = useCallback(() => {
    markDone();
    setStep(null);
  }, []);

  // The app: start once if onboarding just ended (the person's tour_pending setting), but only on New
  // & Noteworthy. Onboarding's last step marks setup done before its first job search finishes, and
  // the tour mustn't pull them off that progress screen.
  const [pending, setPending] = useState(false);
  useEffect(() => {
    if (DEMO) return;
    let cancelled = false;
    api
      .getSettings()
      .then((s) => {
        if (!cancelled && s.tour_pending) setPending(true);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);
  useEffect(() => {
    if (pending && location.pathname === "/") {
      setPending(false);
      setStep(0);
    }
  }, [pending, location.pathname]);

  // "Take the tour" (the demo bar, the About page).
  useEffect(() => {
    const start = () => go(0);
    window.addEventListener(TOUR_EVENT, start);
    return () => window.removeEventListener(TOUR_EVENT, start);
  }, [go]);

  // A tab clicked mid-tour: follow it to that tab's step rather than fighting the visitor.
  useEffect(() => {
    if (step === null) return;
    const at = STEPS.findIndex((s) => s.path === location.pathname);
    if (at >= 0 && at !== step) setStep(at);
  }, [location.pathname, step]);

  useEffect(() => {
    if (step === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") end();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, end]);

  // Keep the note under its tab as the window resizes or the page scrolls.
  useLayoutEffect(() => {
    if (step === null) return;
    const measure = () => {
      setPhone(window.innerWidth < PHONE);
      const tab = document.querySelector<HTMLElement>(STEPS[step].selector);
      if (!tab) return setPlace(null);
      const r = tab.getBoundingClientRect();
      const center = r.left + r.width / 2;
      const left = Math.min(Math.max(12, center - POPOVER_WIDTH / 2), window.innerWidth - POPOVER_WIDTH - 12);
      setPlace({
        top: r.bottom + 12,
        left,
        arrow: Math.min(Math.max(18, center - left), POPOVER_WIDTH - 18),
        ring: { top: r.top - 3, left: r.left - 3, width: r.width + 6, height: r.height + 6 },
      });
    };
    measure();
    const id = window.setTimeout(measure, 150); // after the page has rendered under the bar
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, { passive: true });
    return () => {
      window.clearTimeout(id);
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure);
    };
  }, [step, location.pathname]);

  if (step === null) return null;
  const s = STEPS[step];
  const last = step === STEPS.length - 1;
  const card = (
    <div
      className={`demo-tour${phone ? " sheet" : ""}`}
      role="dialog"
      aria-label={`Tour: ${s.title}`}
      style={!phone && place ? { top: place.top, left: place.left, width: POPOVER_WIDTH } : undefined}
    >
      {!phone && place && <span className="demo-tour-arrow" style={{ left: place.arrow }} />}
      <div className="demo-tour-count">
        Step {step + 1} of {STEPS.length}
      </div>
      <h3>{s.title}</h3>
      <p>{s.body}</p>
      {last && (
        <p className="demo-tour-end">
          {DEMO ? (
            <>
              That's the tour. Look around as much as you like, or{" "}
              <a href={`${import.meta.env.BASE_URL}../job-search/`}>go back to the welcome page</a>.
            </>
          ) : (
            "That's the tour. You can take it again any time from the About page."
          )}
        </p>
      )}
      <div className="demo-tour-actions">
        <button className="link small" onClick={end}>
          End tour
        </button>
        <span className="demo-tour-nav">
          {step > 0 && (
            <button className="ghost small" onClick={() => go(step - 1)}>
              Back
            </button>
          )}
          <button className="primary small" onClick={() => (last ? end() : go(step + 1))} autoFocus>
            {last ? "Finish" : "Next →"}
          </button>
        </span>
      </div>
    </div>
  );
  return (
    <>
      {!phone && place && <div className="demo-tour-ring" style={place.ring} aria-hidden="true" />}
      {card}
    </>
  );
}
