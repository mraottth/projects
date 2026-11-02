import { useState } from "react";
import { api, type Me } from "../api";
import { Spinner } from "../components/Spinner";

/**
 * What someone who isn't approved yet sees (D-021): a request form for a first visit, then
 * "waiting for approval", or a plain no-access message. Nothing else in the app is reachable
 * until the admin approves them.
 */
export default function AccessPage({ me, onChange }: { me: Me; onChange: (status: Me["status"]) => void }) {
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const { status } = await api.requestAccess(name.trim(), note.trim());
      onChange(status);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-main access-page">
      <div className="card">
        {me.status === "none" && (
          <>
            <h1>Request access</h1>
            <p>
              Job Search Assistant finds job postings for you, rates how well each one fits, helps tailor your
              resume and cover letter, and tracks your applications. It's invite-only: send a request and
              you'll get an email at <strong>{me.email}</strong> once it's approved.
            </p>
            <form onSubmit={submit}>
              <div className="fsection">
                <label className="flabel" htmlFor="access-name">Your name</label>
                <input id="access-name" value={name} onChange={(e) => setName(e.target.value)} required maxLength={100} />
              </div>
              <div className="fsection">
                <label className="flabel" htmlFor="access-note">Anything the admin should know? (optional)</label>
                <textarea id="access-note" value={note} onChange={(e) => setNote(e.target.value)} rows={3} maxLength={1000} />
              </div>
              <button className="primary" type="submit" disabled={busy || !name.trim()}>
                {busy ? <Spinner label="Sending…" /> : "Request access"}
              </button>
              {error && <p className="error">{error}</p>}
            </form>
            <PrivacyNote />
          </>
        )}
        {me.status === "pending" && (
          <>
            <h1>Waiting for approval</h1>
            <p>
              Your request is in. You'll get an email at <strong>{me.email}</strong> once it's approved, and
              then this page will open the app.
            </p>
          </>
        )}
        {(me.status === "declined" || me.status === "blocked") && (
          <>
            <h1>No access</h1>
            <p>This Google account ({me.email}) doesn't have access to Job Search Assistant.</p>
          </>
        )}
      </div>
    </main>
  );
}

export function PrivacyNote() {
  return (
    <details className="privacy-note">
      <summary>What happens to your information</summary>
      <ul>
        <li>
          The app keeps what you give it: your resumes and cover letters, your profile and preferences, the jobs
          you track and your notes. Only you can see them in the app.
        </li>
        <li>
          To rate jobs, write tailored documents and chat with Jobby, it sends the relevant parts (your resume
          text, profile and the job posting) to Anthropic's Claude API, which by default doesn't use API data to train its models.
        </li>
        <li>
          The app's owner runs it and can reach the stored data, including backups. Settings' "Start over"
          clears everything you've added at any time.
        </li>
      </ul>
    </details>
  );
}
