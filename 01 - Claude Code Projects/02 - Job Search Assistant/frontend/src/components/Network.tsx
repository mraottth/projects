import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  api,
  Application,
  NetworkOverview,
  NetworkTier,
  PersonAtJob,
} from "../api";
import { Modal } from "./Modal";
import { Spinner } from "./Spinner";

/**
 * Your LinkedIn network (D-036, asked 2026-11-13). LinkedIn doesn't let a personal app read your
 * connections, so you upload LinkedIn's own export (Connections.csv). Manage Sources shows who to
 * reach out to in three tiers, each with a Jobby draft, and organizations you know people at but
 * don't follow, each with "+ Add". The job card lists the people you know at its organization.
 */

const TIERS: NetworkTier[] = ["High", "Medium", "Low"];
type Tab = NetworkTier | "Not listed";

/** A High / Med / Low rating with a short note (asked 2026-11-16), colored like the job cards'. */
function Alignment({
  level,
  note,
  tracked,
}: {
  level: "High" | "Med" | "Low";
  note: string;
  tracked?: boolean;
}) {
  const cls =
    level === "High"
      ? "badge-high"
      : level === "Med"
        ? "badge-med"
        : "badge-low";
  return (
    <div className="alignment">
      <span className={`badge ${cls}`}>{level}</span>
      <span className="muted small">
        {note}
        {tracked && (
          <span className={`tracked-tag${note ? "" : " alone"}`}>
            Tracked source
          </span>
        )}
      </span>
    </div>
  );
}

function since(date: string | null): string {
  if (!date) return "";
  const d = new Date(`${date}T00:00:00`);
  return Number.isNaN(d.getTime())
    ? date
    : d.toLocaleDateString(undefined, { month: "short", year: "numeric" });
}

export interface OutreachPerson {
  name: string;
  position: string | null;
  company: string | null;
  connected_on: string | null;
  why?: string;
  job_id?: number | null;
  job_title?: string | null;
}

export interface OutreachAnswers {
  closeness: string;
  how: string;
  lastTouch: string;
  asks: string[];
  stage: string;
  format: string;
  notes: string;
  // Which role it's about (asked 2026-11-16): a job on My Jobs (or the one the person came up for),
  // something written in ("other"), or none.
  role: "none" | "job" | "other";
  roleJob: { id: number; title: string; company: string | null } | null;
  roleOther: string;
}

/** A Jobby link with a ready-to-send first message asking for outreach, built from the person and
 * the guiding questions' answers (asked 2026-11-16: a draft to a close friend read like a cold
 * email). It's sent only when you press Send, so nothing is paid for until then. */
export function outreachLink(p: OutreachPerson, a?: OutreachAnswers): string {
  const who = [p.name, p.position, p.company && `at ${p.company}`]
    .filter(Boolean)
    .join(", ");
  const format =
    a?.format === "email"
      ? "an email"
      : a?.format === "linkedin"
        ? "a LinkedIn message"
        : "an email and a shorter LinkedIn message";
  const lines = [
    `Help me write ${format} to ${who}.`,
    a?.closeness && `How well I know them: ${a.closeness}.`,
    a?.how.trim() && `How we know each other: ${a.how.trim()}.`,
    a?.lastTouch && `Last in touch: ${a.lastTouch}.`,
    roleLine(p, a),
    a?.asks.length
      ? `What I'd like from them: ${a.asks.join("; ").toLowerCase()}.`
      : null,
    a?.notes.trim() && `Also: ${a.notes.trim()}`,
    p.why && `Background on why they came up: ${p.why}`,
    p.connected_on &&
      `(LinkedIn says we connected in ${since(p.connected_on)}; that's for you, not for the message.)`,
    "Write it the way I'd actually talk to this person, with one clear ask. If something important is still unclear, ask me a question or two first.",
  ].filter(Boolean);
  const prompt = encodeURIComponent(lines.join("\n"));
  // A job on My Jobs opens that job's chat, so Jobby has the posting; anything else a general chat.
  const jobId = a ? (a.role === "job" ? a.roleJob?.id : null) : p.job_id;
  return jobId
    ? `/jobby?job=${jobId}&prompt=${prompt}`
    : `/jobby?new=1&prompt=${prompt}`;
}

function roleLine(p: OutreachPerson, a?: OutreachAnswers): string | null {
  const stage = a?.stage ? ` (${a.stage.toLowerCase()})` : "";
  if (!a)
    return p.job_title ? `It's about their ${p.job_title} opening.` : null;
  if (a.role === "job" && a.roleJob)
    return `It's about the ${a.roleJob.title} role${a.roleJob.company ? ` at ${a.roleJob.company}` : ""}${stage}.`;
  if (a.role === "other" && a.roleOther.trim())
    return `It's about: ${a.roleOther.trim()}${stage}.`;
  return null;
}

const CLOSENESS = [
  "Close: a friend or someone I worked closely with",
  "Well, professionally",
  "A bit",
  "Barely: we've met once or twice",
];
const LAST_TOUCH = [
  "Recently",
  "Within the past year",
  "A few years ago",
  "Not sure",
];
const ASKS = [
  "A referral or a word with the hiring manager",
  "Their honest take on the role or team",
  "An introduction to someone else",
  "Advice on my search",
  "Just reconnecting",
];
const STAGES = ["Haven't applied yet", "Already applied", "Interviewing"];
const FORMATS: [string, string][] = [
  ["both", "Email and LinkedIn message"],
  ["email", "Email"],
  ["linkedin", "LinkedIn message"],
];

/** "Write outreach with Jobby": a few quick questions about the relationship and the ask, then
 * Jobby with the answers in its message box (asked 2026-11-16). */
export function OutreachButton({ person }: { person: OutreachPerson }) {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [a, setA] = useState<OutreachAnswers>({
    closeness: "",
    how: "",
    lastTouch: "",
    asks: [],
    stage: "",
    format: "both",
    notes: "",
    role: person.job_id ? "job" : "none",
    roleJob: person.job_id
      ? {
          id: person.job_id,
          title: person.job_title ?? "this role",
          company: person.company,
        }
      : null,
    roleOther: "",
  });
  // My Jobs, for "Which role is this about?", loaded the first time the dialog opens.
  const [apps, setApps] = useState<Application[] | null>(null);
  useEffect(() => {
    if (open && apps === null)
      api
        .listApplications()
        .then(setApps)
        .catch(() => setApps([]));
  }, [open, apps]);
  const jobOptions = [
    ...(person.job_id && !(apps ?? []).some((x) => x.job_id === person.job_id)
      ? [
          {
            id: person.job_id,
            title: person.job_title ?? "Their opening",
            company: person.company,
          },
        ]
      : []),
    ...(apps ?? []).map((x) => ({
      id: x.job_id,
      title: x.title,
      company: x.company,
    })),
  ];
  const roleValue =
    a.role === "job" && a.roleJob ? `job:${a.roleJob.id}` : a.role;
  const chooseRole = (value: string) => {
    if (value.startsWith("job:")) {
      const job = jobOptions.find((j) => `job:${j.id}` === value) ?? null;
      set({ role: "job", roleJob: job });
    } else set({ role: value as "none" | "other", roleJob: null });
  };
  const set = (patch: Partial<OutreachAnswers>) =>
    setA((prev) => ({ ...prev, ...patch }));
  const toggleAsk = (ask: string) =>
    set({
      asks: a.asks.includes(ask)
        ? a.asks.filter((x) => x !== ask)
        : [...a.asks, ask],
    });

  return (
    <>
      <button className="ghost small nowrap" onClick={() => setOpen(true)}>
        Write outreach with Jobby
      </button>
      <Modal
        open={open}
        onClose={() => setOpen(false)}
        title={`Reaching out to ${person.name}`}
      >
        <div className="outreach-form">
          <p className="muted small">
            A few quick answers help Jobby write it the way you'd actually say
            it. Skip any you like.
          </p>
          <div className="fsection">
            <span className="flabel">How well do you know them?</span>
            <div className="chip-toggle-group">
              {CLOSENESS.map((c) => (
                <button
                  key={c}
                  className={`chip-toggle${a.closeness === c ? " on" : ""}`}
                  onClick={() => set({ closeness: a.closeness === c ? "" : c })}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>
          <label className="fsection">
            <span className="flabel">How do you know each other?</span>
            <input
              value={a.how}
              onChange={(e) => set({ how: e.target.value })}
              placeholder="e.g. we worked together on the 2020 campaign; grad school friends"
            />
          </label>
          <div className="fsection">
            <span className="flabel">When were you last in touch?</span>
            <div className="chip-toggle-group">
              {LAST_TOUCH.map((c) => (
                <button
                  key={c}
                  className={`chip-toggle${a.lastTouch === c ? " on" : ""}`}
                  onClick={() => set({ lastTouch: a.lastTouch === c ? "" : c })}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>
          <div className="fsection">
            <span className="flabel">What would help most? (any)</span>
            <div className="chip-toggle-group">
              {ASKS.map((c) => (
                <button
                  key={c}
                  className={`chip-toggle${a.asks.includes(c) ? " on" : ""}`}
                  onClick={() => toggleAsk(c)}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>
          <div className="fsection">
            <span className="flabel">Which role is this about?</span>
            <div className="select-wrap">
              <select
                value={roleValue}
                onChange={(e) => chooseRole(e.target.value)}
              >
                <option value="none">No specific job</option>
                {jobOptions.length > 0 && (
                  <optgroup label="Jobs (My Jobs, and the opening they came up for)">
                    {jobOptions.map((j) => (
                      <option key={j.id} value={`job:${j.id}`}>
                        {j.title}
                        {j.company ? ` — ${j.company}` : ""}
                      </option>
                    ))}
                  </optgroup>
                )}
                <option value="other">Other (describe it)</option>
              </select>
            </div>
            {a.role === "other" && (
              <input
                value={a.roleOther}
                onChange={(e) => set({ roleOther: e.target.value })}
                placeholder="e.g. a data director role on their team that isn't posted yet"
              />
            )}
          </div>
          {a.role !== "none" && (
            <div className="fsection">
              <span className="flabel">Where are you with it?</span>
              <div className="chip-toggle-group">
                {STAGES.map((c) => (
                  <button
                    key={c}
                    className={`chip-toggle${a.stage === c ? " on" : ""}`}
                    onClick={() => set({ stage: a.stage === c ? "" : c })}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>
          )}
          <label className="fsection">
            <span className="flabel">Anything else Jobby should know?</span>
            <textarea
              rows={3}
              value={a.notes}
              onChange={(e) => set({ notes: e.target.value })}
              placeholder="Shared history, what's going on with them, what you'd say in person, anything to avoid..."
            />
          </label>
          <div className="fsection">
            <span className="flabel">Write</span>
            <div className="chip-toggle-group">
              {FORMATS.map(([value, label]) => (
                <button
                  key={value}
                  className={`chip-toggle${a.format === value ? " on" : ""}`}
                  onClick={() => set({ format: value })}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="form-actions">
            <button
              className="primary"
              onClick={() => navigate(outreachLink(person, a))}
            >
              Open in Jobby →
            </button>
            <button className="link small" onClick={() => setOpen(false)}>
              Cancel
            </button>
          </div>
          <p className="muted small">
            Nothing is sent until you press Send in Jobby, and you can edit the
            message there first.
          </p>
        </div>
      </Modal>
    </>
  );
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

  const rerate = async () => {
    setBusy(true);
    setError(null);
    try {
      setInfo(await api.rateNetworkOrganizations());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
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
        {/* Matches LinkedIn's page as of 2026-11-16 (the owner's screenshot): Connections isn't in
            the "Want something in particular?" list any more, so it's the full archive. */}
        <ol className="small">
          <li>
            On LinkedIn, open <strong>Settings &amp; Privacy</strong> →{" "}
            <strong>Data privacy</strong> → <strong>Download my data</strong>.
          </li>
          <li>
            Keep the first option selected,{" "}
            <strong>
              "Download larger data archive, including connections…"
            </strong>{" "}
            (Connections isn't in the "Want something in particular?" list), and
            press <strong>Request archive</strong>.
          </li>
          <li>
            LinkedIn takes a while to prepare it. Wait for LinkedIn's email
            saying it's ready, then download the archive and unzip it.
          </li>
          <li>
            Upload <strong>Connections.csv</strong> from it below. The archive
            holds more than your connections; only that file is uploaded here.
            Upload a fresh one now and then: it's a snapshot.
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
      {has && info!.ratings_outdated && (
        <p className="small network-current">
          <span className="muted">
            High is now for organizations that suit your profile, so your
            connections' organizations need rating (their names only, about a
            cent or two).
          </span>{" "}
          <button className="ghost small" onClick={rerate} disabled={busy}>
            {busy ? <Spinner label="Rating..." /> : "Re-rate organizations"}
          </button>
        </p>
      )}
      {error && <p className="error-text small">{error}</p>}
      {info?.picks_error && (
        <p className="muted small">
          Couldn't rate your connections' organizations ({info.picks_error}), so
          org alignment only counts your jobs and sources for now. Uploading
          again, or Re-rate organizations, retries.
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
          {/* Search sits on the left under the tabs (asked 2026-11-16); the tier hints are gone. */}
          <div className="network-filter">
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
                    <th>Org alignment</th>
                    <th>Title alignment</th>
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
                      <td className="align-cell">
                        <Alignment
                          level={p.org_align}
                          note={p.org_note}
                          tracked={p.org_tracked}
                        />
                      </td>
                      <td className="align-cell">
                        <Alignment level={p.title_align} note={p.title_note} />
                      </td>
                      <td>
                        {tier !== "Not listed" && <OutreachButton person={p} />}
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
            <OutreachButton
              person={{ ...p, company, job_id: jobId, job_title: jobTitle }}
            />
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
