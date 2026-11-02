import { lazy, Suspense, useEffect, useState } from "react";
import { Navigate, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { api, type Me } from "./api";
import { JobbyLogo } from "./components/JobbyLogo";
import { Spinner } from "./components/Spinner";
import DigestPage from "./pages/DigestPage";
import JobDetailPage from "./pages/JobDetailPage";
import JobbyPage from "./pages/JobbyPage";
import AllJobsPage from "./pages/AllJobsPage";
import MyJobsPage from "./pages/MyJobsPage";
import OnboardingPage from "./pages/OnboardingPage";
import ManageSourcesPage from "./pages/ManageSourcesPage";
import ProfilePage from "./pages/ProfilePage";
import SettingsPage from "./pages/SettingsPage";
import AccessPage from "./pages/AccessPage";
import AdminPage from "./pages/AdminPage";
import { MeContext } from "./me";

// Carries ~220 KB of prompts and commits, so it's only loaded when visited.
const ChangelogPage = lazy(() => import("./pages/ChangelogPage"));

const NAV_LINKS = [
  { to: "/", label: "New & Noteworthy", end: true },
  { to: "/all-jobs", label: "All Jobs" },
  { to: "/tracker", label: "My Jobs" },
  { to: "/jobby", label: "Jobby" },
  { to: "/sources", label: "Manage Sources" },
  { to: "/profile", label: "Profile" },
];

function GearIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

// These pages need the extra room (a filter sidebar, a wide table, Jobby's job picker beside
// the chat, Manage Sources' two columns, or the Timeline's milestone strip) that the default
// prose-friendly max-width doesn't give them.
const WIDE_ROUTES = ["/", "/all-jobs", "/tracker", "/jobby", "/sources", "/changelog", "/admin"];

function AppMain({
  needsOnboarding,
  onOnboardingDone,
  isAdmin,
}: {
  needsOnboarding: boolean;
  onOnboardingDone: () => void;
  isAdmin: boolean;
}) {
  const location = useLocation();
  const wide = WIDE_ROUTES.includes(location.pathname);
  if (needsOnboarding && location.pathname !== "/onboarding") {
    return <Navigate to="/onboarding" replace />;
  }
  return (
    // My Jobs' table gets a little more room still, so most job titles fit on one line.
    <main className={`app-main${wide ? " wide" : ""}${location.pathname === "/tracker" ? " extra-wide" : ""}`}>
      <Routes>
        <Route path="/onboarding" element={<OnboardingPage onDone={onOnboardingDone} />} />
        <Route path="/" element={<DigestPage />} />
        <Route path="/jobs/:id" element={<JobDetailPage />} />
        <Route path="/all-jobs" element={<AllJobsPage />} />
        <Route path="/tracker" element={<MyJobsPage />} />
        <Route path="/jobby" element={<JobbyPage />} />
        <Route path="/sources" element={<ManageSourcesPage />} />
        <Route path="/profile" element={<ProfilePage />} />
        <Route path="/settings" element={<SettingsPage />} />
        {isAdmin && <Route path="/admin" element={<AdminPage />} />}
        {isAdmin && (
          <Route
            path="/changelog"
            element={
              <Suspense fallback={<Spinner label="Loading…" />}>
                <ChangelogPage />
              </Suspense>
            }
          />
        )}
      </Routes>
    </main>
  );
}

export default function App() {
  const location = useLocation();
  // Who's signed in comes first (D-021): anyone not yet approved gets the request-access page and
  // nothing else. null while loading.
  const [me, setMe] = useState<Me | null>(null);
  const [meError, setMeError] = useState<string | null>(null);
  // null while loading: hold off rendering any page so a first-run user doesn't see a flash of
  // the empty digest before being redirected. A failed status check just skips onboarding.
  const [needsOnboarding, setNeedsOnboarding] = useState<boolean | null>(null);
  const approved = me?.status === "approved";

  useEffect(() => {
    api
      .getMe()
      .then(setMe)
      .catch((e: Error) => setMeError(e.message));
  }, []);

  useEffect(() => {
    if (!approved) return;
    api
      .getOnboardingStatus()
      .then((s) => setNeedsOnboarding(s.needs_onboarding))
      .catch(() => setNeedsOnboarding(false));
  }, [approved]);

  if (meError) {
    return (
      <div className="app-shell">
        <header className="topbar">
          <span className="brand">Job Search Assistant</span>
        </header>
        <main className="app-main">
          <p className="error">Couldn't sign you in: {meError}</p>
        </main>
      </div>
    );
  }
  if (me && !approved) {
    return (
      <div className="app-shell">
        <header className="topbar">
          <span className="brand">Job Search Assistant</span>
        </header>
        <AccessPage me={me} onChange={(status) => setMe({ ...me, status })} />
      </div>
    );
  }

  // The wizard hides the nav so it reads as one focused flow; "Skip setup" is the way out.
  const inOnboarding = location.pathname === "/onboarding";

  return (
    <MeContext.Provider value={me}>
    <div className="app-shell">
      <header className="topbar">
        <span className="brand">Job Search Assistant</span>
        {!inOnboarding && (
          <nav>
            {NAV_LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.end}
                className={({ isActive }) => (isActive ? "active" : "")}
              >
                {link.to === "/jobby" && <JobbyLogo size={22} title="" />}
                {link.label}
              </NavLink>
            ))}
          </nav>
        )}
        {!inOnboarding && me?.is_admin && (
          <NavLink to="/admin" className={({ isActive }) => `admin-link${isActive ? " active" : ""}`}>
            Admin
          </NavLink>
        )}
        {!inOnboarding && (
          <NavLink
            to="/settings"
            className={({ isActive }) => `settings-gear${isActive ? " active" : ""}`}
            aria-label="Settings"
            title="Settings"
          >
            <GearIcon />
          </NavLink>
        )}
      </header>
      {needsOnboarding !== null && (
        <AppMain
          needsOnboarding={needsOnboarding}
          onOnboardingDone={() => setNeedsOnboarding(false)}
          isAdmin={!!me?.is_admin}
        />
      )}
    </div>
    </MeContext.Provider>
  );
}
