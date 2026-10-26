import { NavLink, Route, Routes } from "react-router-dom";
import DigestPage from "./pages/DigestPage";
import JobDetailPage from "./pages/JobDetailPage";
import AllJobsPage from "./pages/AllJobsPage";
import MyJobsPage from "./pages/MyJobsPage";
import ManageSourcesPage from "./pages/ManageSourcesPage";
import ProfilePage from "./pages/ProfilePage";
import SettingsPage from "./pages/SettingsPage";

const NAV_LINKS = [
  { to: "/", label: "New & Noteworthy", end: true },
  { to: "/all-jobs", label: "All Jobs" },
  { to: "/tracker", label: "My Jobs" },
  { to: "/sources", label: "Manage Sources" },
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
          <Route path="/all-jobs" element={<AllJobsPage />} />
          <Route path="/tracker" element={<MyJobsPage />} />
          <Route path="/sources" element={<ManageSourcesPage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}
