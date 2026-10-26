import { NavLink, Route, Routes } from "react-router-dom";
import DigestPage from "./pages/DigestPage";
import JobDetailPage from "./pages/JobDetailPage";
import ApplicationTrackerPage from "./pages/ApplicationTrackerPage";
import SourcesPage from "./pages/SourcesPage";
import ResumesPage from "./pages/ResumesPage";
import SettingsPage from "./pages/SettingsPage";

const NAV_LINKS = [
  { to: "/", label: "Digest", end: true },
  { to: "/tracker", label: "Tracker" },
  { to: "/sources", label: "Sources" },
  { to: "/resumes", label: "Resumes" },
  { to: "/settings", label: "Settings" },
];

export default function App() {
  return (
    <div className="app-shell">
      <nav className="app-nav">
        <span className="app-nav-title">Job Search Assistant</span>
        <div className="app-nav-links">
          {NAV_LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              {link.label}
            </NavLink>
          ))}
        </div>
      </nav>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<DigestPage />} />
          <Route path="/jobs/:id" element={<JobDetailPage />} />
          <Route path="/tracker" element={<ApplicationTrackerPage />} />
          <Route path="/sources" element={<SourcesPage />} />
          <Route path="/resumes" element={<ResumesPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}
