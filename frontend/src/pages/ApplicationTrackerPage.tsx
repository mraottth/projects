import { useEffect, useState } from "react";
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
      {apps.length === 0 && (
        <p className="muted">
          No applications tracked yet — use "Track application" on a digest card.
        </p>
      )}
      {apps.map((app) => (
        <div className="card" key={app.id}>
          <h3>
            {app.title} — {app.company}
          </h3>
          <p className="muted">{app.location ?? "location unknown"}</p>
          <StatusSelect app={app} onChanged={updateInPlace} />
          <div style={{ marginTop: "0.5rem" }}>
            <strong>History: </strong>
            {app.history.map((h) => h.status.replace("_", " ")).join(" → ")}
          </div>
        </div>
      ))}
    </div>
  );
}
