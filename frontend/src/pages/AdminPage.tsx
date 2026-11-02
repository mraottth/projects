import { useEffect, useState } from "react";
import { api, type AdminUser } from "../api";
import { Spinner } from "../components/Spinner";

/**
 * Admin only (D-021): approve or decline access requests, and see each person's activity, today's
 * scoring against their daily cap, and this month's estimated Claude spend against their budget.
 */
export default function AdminPage() {
  const [users, setUsers] = useState<AdminUser[] | null>(null);
  const [shared, setShared] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  async function load() {
    try {
      const data = await api.listUsers();
      setUsers(data.users);
      setShared(data.shared_spend_this_month);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function update(user: AdminUser, patch: Parameters<typeof api.updateUser>[1]) {
    setBusyId(user.id);
    setError(null);
    try {
      await api.updateUser(user.id, patch);
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyId(null);
    }
  }

  if (error && !users) return <p className="error">{error}</p>;
  if (!users) return <Spinner label="Loading…" />;

  const pending = users.filter((u) => u.status === "pending");
  const others = users.filter((u) => u.status !== "pending");
  const totalSpend = users.reduce((sum, u) => sum + u.spend_this_month, 0) + shared;

  return (
    <div className="admin-page">
      <h1>Admin</h1>
      {error && <p className="error">{error}</p>}

      <section className="card">
        <h2>Access requests</h2>
        {pending.length === 0 ? (
          <p className="muted">No one is waiting.</p>
        ) : (
          <ul className="admin-requests">
            {pending.map((u) => (
              <li key={u.id}>
                <div>
                  <strong>{u.name ?? u.email}</strong> <span className="muted">{u.email}</span>
                  <div className="muted small">Asked {when(u.created_at)}</div>
                  {u.note && <p className="admin-note">“{u.note}”</p>}
                </div>
                <div className="admin-actions">
                  <button className="primary" disabled={busyId === u.id} onClick={() => update(u, { status: "approved" })}>
                    Approve
                  </button>
                  <button className="ghost" disabled={busyId === u.id} onClick={() => update(u, { status: "declined" })}>
                    Decline
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card">
        <h2>People</h2>
        <p className="muted small">
          Claude spend this month: ${totalSpend.toFixed(2)} in all, including ${shared.toFixed(2)} for reading
          shared job boards. Spend is estimated from the usage log.
        </p>
        <table className="admin-table">
          <thead>
            <tr>
              <th>Person</th>
              <th>Status</th>
              <th>Last visit</th>
              <th>Scored today</th>
              <th>Spend this month</th>
              <th>Sources</th>
              <th>Jobs tracked</th>
              <th aria-label="Actions" />
            </tr>
          </thead>
          <tbody>
            {others.map((u) => (
              <tr key={u.id}>
                <td>
                  <strong>{u.name ?? u.email}</strong>
                  {!!u.is_admin && <span className="pill admin-pill">admin</span>}
                  {u.name && <div className="muted small">{u.email}</div>}
                </td>
                <td>
                  {u.is_admin ? (
                    "approved"
                  ) : (
                    <select
                      value={u.status}
                      disabled={busyId === u.id}
                      onChange={(e) => update(u, { status: e.target.value as AdminUser["status"] })}
                    >
                      <option value="approved">approved</option>
                      <option value="blocked">blocked</option>
                      <option value="declined">declined</option>
                    </select>
                  )}
                </td>
                <td>{u.last_seen_at ? when(u.last_seen_at) : "never"}</td>
                <td>
                  {u.scored_today} of{" "}
                  <NumberCell
                    value={u.daily_score_cap}
                    min={0}
                    step={1}
                    onSave={(v) => update(u, { daily_score_cap: v })}
                  />
                </td>
                <td>
                  ${u.spend_this_month.toFixed(2)} of $
                  <NumberCell
                    value={u.monthly_budget_usd}
                    min={0}
                    step={1}
                    onSave={(v) => update(u, { monthly_budget_usd: v })}
                  />
                </td>
                <td>{u.sources_followed}</td>
                <td>{u.applications}</td>
                <td>{!u.is_admin && <DeleteUser user={u} onDeleted={load} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">
          Scoring only runs for people who opened the app in the last 7 days. Anyone who reaches their
          monthly budget has paid features paused until next month; you get an email at 80% and at 100%.
          Your own account has no limit.
        </p>
      </section>
    </div>
  );
}

function NumberCell({
  value,
  min,
  step,
  onSave,
}: {
  value: number;
  min: number;
  step: number;
  onSave: (v: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  return (
    <input
      className="admin-number"
      type="number"
      min={min}
      step={step}
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => {
        const v = Number(draft);
        if (draft !== "" && Number.isFinite(v) && v !== value) onSave(v);
        else setDraft(String(value));
      }}
    />
  );
}

function when(ts: string): string {
  const d = new Date(ts.includes("T") ? ts : ts.replace(" ", "T") + "Z");
  const days = Math.floor((Date.now() - d.getTime()) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  return d.toLocaleDateString();
}

/** "Delete…" then type DELETE: removes the account and everything in it, and emails them. */
function DeleteUser({ user, onDeleted }: { user: AdminUser; onDeleted: () => void }) {
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!open) {
    return (
      <button className="link small danger-link" onClick={() => setOpen(true)}>
        Delete…
      </button>
    );
  }
  return (
    <div className="admin-delete">
      <input
        aria-label={`Type DELETE to delete ${user.email}`}
        placeholder="Type DELETE"
        value={typed}
        onChange={(e) => setTyped(e.target.value)}
        disabled={busy}
      />
      <button
        className="danger-solid"
        disabled={typed !== "DELETE" || busy}
        onClick={async () => {
          setBusy(true);
          setError(null);
          try {
            await api.deleteUser(user.id, typed);
            onDeleted();
          } catch (e) {
            setError((e as Error).message);
            setBusy(false);
          }
        }}
      >
        {busy ? <Spinner label="Deleting…" /> : "Delete account"}
      </button>
      <button className="link small" onClick={() => setOpen(false)} disabled={busy}>
        Cancel
      </button>
      {error && <p className="error small">{error}</p>}
    </div>
  );
}

