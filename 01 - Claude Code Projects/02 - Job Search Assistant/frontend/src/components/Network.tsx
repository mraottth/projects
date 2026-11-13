import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, NetworkOverview, NetworkTier, PersonAtJob } from "../api";
import { Spinner } from "./Spinner";

/**
 * Your LinkedIn network (D-036, asked 2026-11-13). LinkedIn doesn't let a personal app read your
 * connections, so you upload LinkedIn's own export (Connections.csv). Manage Sources shows who to
 * reach out to in three tiers, each with a Jobby draft, and organizations you know people at but
 * don't follow, each with "+ Add". The job card lists the people you know at its organization.
 */

const TIERS: NetworkTier[] = ["High", "Medium", "Low"];
type Tab = NetworkTier | "Not listed";
const TIER_HINT: Record<Tab, string> = {
  High: "A job you'd want there and they're on the hiring side, in that line of work or a leader; or they lead an organization in your sources.",
  Medium:
    "A job you'd want there in another role, or the right role where the openings aren't a strong match.",
  Low: "Worth a hello: an organization that suits you, or someone in your line of work, with no openings yet.",
  "Not listed":
    "Everyone else, and why. If someone belongs higher, their organization may be missing from your sources.",
};

function since(date: string | null): string {
  if (!date) return "";
  const d = new Date(`${date}T00:00:00`);
  return Number.isNaN(d.getTime())
    ? date
    : d.toLocaleDateString(undefined, { month: "short", year: "numeric" });
}

/** A Jobby link with a ready-to-send first message asking for an outreach email (it's sent only
 * when you press Send, so nothing is paid for until then). */
export function outreachLink(p: {
  name: string;
  position: string | null;
  company: string | null;
  connected_on: string | null;
  why?: string;
  job_id?: number | null;
  job_title?: string | null;
}): string {
  const who = [p.name, p.position, p.company && `at ${p.company}`]
    .filter(Boolean)
    .join(", ");
  const parts = [
    `Help me write a short outreach email to ${who}.`,
    p.connected_on &&
      `We've been connected on LinkedIn since ${since(p.connected_on)}.`,
    p.job_title && `I'm interested in their ${p.job_title} opening.`,
    p.why && `Why them: ${p.why}`,
    "Keep it brief and specific to them, suggest a subject line, and give me a version for a LinkedIn message too.",
  ].filter(Boolean);
  const prompt = encodeURIComponent(parts.join(" "));
  return p.job_id
    ? `/jobby?job=${p.job_id}&prompt=${prompt}`
    : `/jobby?new=1&prompt=${prompt}`;
}

export function NetworkSection({
  onSourcesChanged,
}: {
  onSourcesChanged: () => void;
}) {
  const [info, setInfo] = useState<NetworkOverview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tier, setTier] = useState<Tab>("High");
  const [search, setSearch] = useState("");
  const [adding, setAdding] = useState<
    Record<string, "busy" | "added" | string>
  >({});
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .getNetwork()
      .then((n) => {
        setInfo(n);
        const first = TIERS.find((t) => n.tiers[t].length > 0);
        if (first) setTier(first);
      })
      .catch(() => setInfo(null));
  }, []);

  const upload = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const n = await api.uploadConnections(file);
      setInfo(n);
      setTier(TIERS.find((t) => n.tiers[t].length > 0) ?? "High");
      setAdding({});
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const addSource = async (name: string) => {
    setAdding((a) => ({ ...a, [name]: "busy" }));
    try {
      await api.resolveSource(name);
      setAdding((a) => ({ ...a, [name]: "added" }));
      onSourcesChanged();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setAdding((a) => ({
        ...a,
        [name]: `Couldn't find its job board (${msg}). Add it by link under Organizations.`,
      }));
    }
  };

  const remove = async () => {
    setInfo(await api.deleteConnections());
  };

  const has = !!info && info.count > 0;
  const all = info
    ? tier === "Not listed"
      ? info.not_listed
      : info.tiers[tier]
    : [];
  const q = search.trim().toLowerCase();
  const rows = q
    ? all.filter((p) =>
        [p.name, p.company, p.position].some((v) =>
          (v ?? "").toLowerCase().includes(q),
        ),
      )
    : all;

  return (
    <div className="card network">
      <h2>Your LinkedIn network</h2>
      <p className="muted small">
        See which of your connections to reach out to about the jobs you want,
        and organizations you know people at. LinkedIn doesn't let apps read
        your connections directly, so this uses the file LinkedIn gives you.
      </p>

      <details className="network-steps" open={!has}>
        <summary>How to get your Connections.csv from LinkedIn</summary>
        <ol className="small">
          <li>
            On LinkedIn, open <strong>Settings &amp; Privacy</strong> →{" "}
            <strong>Data privacy</strong> →{" "}
            <strong>Get a copy of your data</strong>.
          </li>
          <li>
            Choose <strong>"Want something in particular?"</strong>, tick only{" "}
            <strong>Connections</strong>, and press{" "}
            <strong>Request archive</strong>.
          </li>
          <li>
            LinkedIn emails you a link, usually within 10 minutes. Download the
            file and unzip it if it's zipped.
          </li>
          <li>
            Upload <strong>Connections.csv</strong> below. Upload a fresh one
            now and then: it's a snapshot.
          </li>
        </ol>
        <p className="muted small">
          Your connections stay in your account only; nobody else using the app
          sees them. To find organizations that suit you, their organizations'
          names (never people's names) go to Claude once per upload, for about a
          cent. A person's details go to Claude only when you ask Jobby to write
          to them.
        </p>
      </details>

      <div className="form-actions network-upload">
        <input
          ref={fileRef}
          type="file"
          accept=".csv,text/csv"
          style={{ display: "none" }}
          onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
        />
        <button
          className={has ? "ghost" : "primary"}
          onClick={() => fileRef.current?.click()}
          disabled={busy}
        >
          {busy ? (
            <Spinner label="Reading your connections..." />
          ) : has ? (
            "Upload a newer file"
          ) : (
            "Upload Connections.csv"
          )}
        </button>
        {has && (
          <span className="muted small">
            {info!.count.toLocaleString()} connections
            {info!.uploaded_at &&
              `, uploaded ${new Date(info!.uploaded_at).toLocaleDateString()}`}{" "}
            ·{" "}
            <button className="link small" onClick={remove}>
              Remove them
            </button>
          </span>
        )}
      </div>
      {has && <CurrentOrg info={info!} onSaved={setInfo} />}
      {error && <p className="error-text small">{error}</p>}
      {info?.picks_error && (
        <p className="muted small">
          Couldn't check which other organizations suit you ({info.picks_error}
          ), so the Low tier only lists people at organizations with jobs.
          Uploading again retries.
        </p>
      )}

      {has && (
        <>
          <h3 className="network-heading">Who to reach out to</h3>
          <div className="seg network-tiers" role="tablist">
            {TIERS.map((t) => (
              <button
                key={t}
                className={t === tier ? "on" : ""}
                onClick={() => setTier(t)}
              >
                {t} ({info!.tiers[t].length})
              </button>
            ))}
            <button
              className={tier === "Not listed" ? "on" : ""}
              onClick={() => setTier("Not listed")}
            >
              Not listed ({info!.not_listed.length})
            </button>
          </div>
          <div className="network-filter">
            <p className="muted small">{TIER_HINT[tier]}</p>
            <input
              type="search"
              placeholder="Search name, organization, position..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          {rows.length === 0 ? (
            <p className="empty-state small">
              {q ? "Nobody here matches that." : "Nobody in this tier."}
            </p>
          ) : (
            <div className="network-table-wrap">
              <table className="jobs-table network-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Position</th>
                    <th>Organization</th>
                    <th>Connected</th>
                    <th>Why</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {rows.map((p) => (
                    <tr key={p.id}>
                      <td>
                        {p.url ? (
                          <a href={p.url} target="_blank" rel="noreferrer">
                            {p.name}
                          </a>
                        ) : (
                          p.name
                        )}
                      </td>
                      <td>{p.position}</td>
                      <td>{p.company}</td>
                      <td className="nowrap">{since(p.connected_on)}</td>
                      <td className="small">{p.why}</td>
                      <td>
                        {tier !== "Not listed" && (
                          <Link
                            className="ghost small button-link nowrap"
                            to={outreachLink(p)}
                          >
                            Write outreach with Jobby
                          </Link>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {info!.suggested_sources.length > 0 && (
            <>
              <h3 className="network-heading">
                Organizations to consider adding
              </h3>
              <p className="muted small">
                You know people here but don't follow their jobs yet. Adding one
                checks it every day like your other sources.
              </p>
              {/* Grouped by tier (an organization's is its best person's), in a box that scrolls. */}
              <div className="network-sources-wrap">
                {TIERS.filter((t) =>
                  info!.suggested_sources.some((s) => s.tier === t),
                ).map((t) => (
                  <div key={t}>
                    <h4 className="network-org-tier">
                      <span className={`network-tier tier-${t.toLowerCase()}`}>
                        {t}
                      </span>
                    </h4>
                    <ul className="network-sources">
                      {info!.suggested_sources
                        .filter((s) => s.tier === t)
                        .map((s) => {
                          const state = adding[s.name];
                          return (
                            <li key={s.name}>
                              <div>
                                <strong>{s.name}</strong>{" "}
                                <span className="muted small">
                                  · {s.connections} connection
                                  {s.connections === 1 ? "" : "s"}
                                  {s.leader && " · you know a leader there"}
                                </span>
                                {s.why && (
                                  <div className="muted small">{s.why}</div>
                                )}
                                {state &&
                                  state !== "busy" &&
                                  state !== "added" && (
                                    <div className="error-text small">
                                      {state}
                                    </div>
                                  )}
                              </div>
                              {state === "added" ? (
                                <span className="small added-mark">
                                  ✓ Added
                                </span>
                              ) : (
                                <button
                                  className="primary small"
                                  onClick={() => addSource(s.name)}
                                  disabled={state === "busy"}
                                >
                                  {state === "busy" ? (
                                    <Spinner label="Adding..." />
                                  ) : (
                                    "+ Add"
                                  )}
                                </button>
                              )}
                            </li>
                          );
                        })}
                    </ul>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}

/** The expanded job card's "People you know here" (nothing when there's nobody). */
export function PeopleYouKnow({
  jobId,
  company,
  jobTitle,
}: {
  jobId: number;
  company: string;
  jobTitle: string;
}) {
  const [people, setPeople] = useState<PersonAtJob[]>([]);
  useEffect(() => {
    api
      .peopleAtJob(jobId)
      .then(setPeople)
      .catch(() => setPeople([]));
  }, [jobId]);
  if (people.length === 0) return null;
  return (
    <div className="card">
      <h2>People you know here</h2>
      <p className="muted small">
        From the LinkedIn connections you uploaded on Manage Sources.
      </p>
      <ul className="network-people">
        {people.map((p) => (
          <li key={p.id}>
            <div>
              {p.url ? (
                <a href={p.url} target="_blank" rel="noreferrer">
                  <strong>{p.name}</strong>
                </a>
              ) : (
                <strong>{p.name}</strong>
              )}
              {p.tier && (
                <span className={`network-tier tier-${p.tier.toLowerCase()}`}>
                  {p.tier}
                </span>
              )}
              <div className="muted small">
                {[
                  p.position,
                  p.connected_on && `connected since ${since(p.connected_on)}`,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </div>
            </div>
            <Link
              className="ghost small button-link nowrap"
              to={outreachLink({
                ...p,
                company,
                job_id: jobId,
                job_title: jobTitle,
              })}
            >
              Write outreach with Jobby
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** "Leaving out people at your current organization: DCCC (guessed from your resume) · Change". */
function CurrentOrg({
  info,
  onSaved,
}: {
  info: NetworkOverview;
  onSaved: (n: NetworkOverview) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(info.current_org ?? "");
  const [saving, setSaving] = useState(false);

  const save = async (name: string) => {
    setSaving(true);
    try {
      await api.putSettings({
        network_current_org: name.trim(),
        network_current_org_guessed: false,
      });
      onSaved(await api.getNetwork());
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };

  if (editing) {
    return (
      <div className="form-row network-current">
        <span className="small">Your current organization:</span>
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="e.g. DCCC"
        />
        <button
          className="primary small"
          onClick={() => save(value)}
          disabled={saving}
        >
          Save
        </button>
        <button className="link small" onClick={() => setEditing(false)}>
          Cancel
        </button>
      </div>
    );
  }
  return (
    <p className="small network-current">
      {info.current_org ? (
        <>
          Leaving out {info.colleagues}{" "}
          {info.colleagues === 1 ? "person" : "people"} at your current
          organization, <strong>{info.current_org}</strong>
          {info.current_org_guessed && (
            <span className="muted"> (guessed from your resume)</span>
          )}{" "}
          ·{" "}
          <button className="link small" onClick={() => setEditing(true)}>
            Change
          </button>{" "}
          ·{" "}
          <button
            className="link small"
            onClick={() => save("")}
            disabled={saving}
          >
            Don't leave anyone out
          </button>
        </>
      ) : (
        <>
          <span className="muted">
            Your coworkers aren't leads, so their names can be left out.
          </span>{" "}
          <button className="link small" onClick={() => setEditing(true)}>
            Set your current organization
          </button>
        </>
      )}
    </p>
  );
}
