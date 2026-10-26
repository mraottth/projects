import { NavLink, Route, Routes } from "react-router-dom";
import DigestPage from "./pages/DigestPage";
import JobDetailPage from "./pages/JobDetailPage";
import MyJobsPage from "./pages/MyJobsPage";
import SourcesPage from "./pages/SourcesPage";
import ProfilePage from "./pages/ProfilePage";
import SettingsPage from "./pages/SettingsPage";

const NAV_LINKS = [
  { to: "/", label: "Digest", end: true },
  { to: "/sources", label: "My Sources" },
  { to: "/tracker", label: "My Jobs" },
  { to: "/profile", label: "Profile" },
  { to: "/settings", label: "Settings" },
];

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <span className="brand">Job Search Assistant</span>
        <nav>
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
        </nav>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<DigestPage />} />
          <Route path="/jobs/:id" element={<JobDetailPage />} />
          <Route path="/tracker" element={<MyJobsPage />} />
          <Route path="/sources" element={<SourcesPage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}
