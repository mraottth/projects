import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, Application, ApplicationStatus } from "../api";

const STATUSES: ApplicationStatus[] = [
  "saved",
  "applied",
  "phone_screen",
  "interview",
  "offer",
  "rejected",
  "withdrawn",
];

function StatusSelect({
  app,
  onChanged,
}: {
  app: Application;
  onChanged: (app: Application) => void;
}) {
  const [busy, setBusy] = useState(false);

  const change = async (status: ApplicationStatus) => {
    setBusy(true);
    try {
      const updated = await api.updateApplication(app.id, { status });
      onChanged(updated);
    } finally {
      setBusy(false);
    }
  };

  return (
    <select value={app.status} disabled={busy} onChange={(e) => change(e.target.value as ApplicationStatus)}>
      {STATUSES.map((s) => (
        <option key={s} value={s}>
          {s.replace("_", " ")}
        </option>
      ))}
    </select>
  );
}

export default function ApplicationTrackerPage() {
  const [apps, setApps] = useState<Application[]>([]);

  useEffect(() => {
    api.listApplications().then(setApps);
  }, []);

  const updateInPlace = (updated: Application) =>
    setApps((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));

  return (
    <div>
      <h1>Application tracker</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        Every job you've decided to pursue, in one place.
      </p>

      {apps.length === 0 && (
        <p className="empty-state">No applications tracked yet — use "Track application" on a digest card.</p>
      )}

      {apps.map((app) => (
        <div className="card" key={app.id}>
          <div className="card-header">
            <div>
              <div className="card-title">
                <Link to={`/jobs/${app.job_id}`}>
                  {app.title} — {app.company}
                </Link>
              </div>
              <div className="card-meta">{app.location ?? "location unknown"}</div>
            </div>
            <span className={`pill pill-${app.status}`}>{app.status.replace("_", " ")}</span>
          </div>

          <div className="form-row">
            <div className="field" style={{ marginBottom: 0 }}>
              <span>Status</span>
              <StatusSelect app={app} onChanged={updateInPlace} />
            </div>
          </div>

          <div className="card-section small muted">
            <strong>History: </strong>
            {app.history.map((h) => h.status.replace("_", " ")).join(" → ")}
          </div>
        </div>
      ))}
    </div>
  );
}
