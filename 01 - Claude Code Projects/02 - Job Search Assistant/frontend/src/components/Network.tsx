import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  api,
  Application,
  NetworkOverview,
  NetworkTier,
  PersonAtJob,
} from "../api";
import { JobbyLogo } from "./JobbyLogo";
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

/** "+ Add a contact" (asked 2026-11-20): one person by hand, without a new Connections.csv. */
function AddContact({ onAdded }: { onAdded: (n: NetworkOverview & { added_id: number }) => void }) {
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ name: "", url: "", position: "", company: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm({ ...form, [k]: e.target.value });
  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      onAdded(await api.addContact(form));
      setForm({ name: "", url: "", position: "", company: "" });
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <button className="primary small" onClick={() => setOpen(true)}>
        + Add a contact
      </button>
      <Modal open={open} onClose={() => setOpen(false)} title="Add a contact">
        <div className="outreach-form">
          <p className="muted small">
            Someone you know who isn't in your last Connections.csv. They're kept when you upload a
            newer file.
          </p>
          <label className="fsection">
            <span className="flabel">Name</span>
            <input value={form.name} onChange={set("name")} placeholder="Jordan Rivera" />
          </label>
          <label className="fsection">
            <span className="flabel">LinkedIn link</span>
            <input
              type="url"
              value={form.url}
              onChange={set("url")}
              placeholder="https://www.linkedin.com/in/..."
            />
          </label>
          <label className="fsection">
            <span className="flabel">Title</span>
            <input value={form.position} onChange={set("position")} placeholder="Director of Analytics" />
          </label>
          <label className="fsection">
            <span className="flabel">Organization</span>
            <input value={form.company} onChange={set("company")} placeholder="Civic Signal" />
          </label>
          <div className="form-actions">
            <button
              className="primary"
              disabled={busy || !form.name.trim() || !form.url.trim()}
              onClick={save}
            >
              {busy ? <Spinner label="Adding..." /> : "Add"}
            </button>
            <button className="ghost" disabled={busy} onClick={() => setOpen(false)}>
              Cancel
            </button>
            {error && <span className="error-text small">{error}</span>}
          </div>
        </div>
      </Modal>
    </>
  );
}

/** Your own priority for a person (asked 2026-11-20); "Auto" is our tier. */
function PrioritySelect({
  person,
  onChange,
}: {
  person: { name: string; tier: NetworkTier | null; auto_tier: NetworkTier | null; priority: string | null };
  onChange: (priority: NetworkTier | "Not listed" | null) => void;
}) {
  const auto = person.auto_tier ?? "Not listed";
  return (
    <select
      className={`priority-select${person.priority ? " yours" : ""}`}
      value={person.priority ?? ""}
      aria-label={`Priority for ${person.name}`}
      title={person.priority ? `You set this (we'd say ${auto})` : "Our tier; pick one to set your own"}
      onChange={(e) => onChange((e.target.value || null) as NetworkTier | "Not listed" | null)}
    >
      <option value="">Auto: {auto}</option>
      {[...TIERS, "Not listed"].map((t) => (
        <option key={t} value={t}>
          {t}
        </option>
      ))}
    </select>
  );
}

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
export function OutreachButton({
  person,
  compact = false,
}: {
  person: OutreachPerson;
  /** My Network's table: just Jobby's face, under a "Write outreach" heading (asked 2026-11-17). */
  compact?: boolean;
}) {
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
      {compact ? (
        <button
          className="outreach-icon"
          onClick={() => setOpen(true)}
          title="Write outreach with Jobby"
          aria-label={`Write outreach to ${person.name} with Jobby`}
        >
          <JobbyLogo size={32} title="" />
        </button>
      ) : (
        <button className="ghost small nowrap" onClick={() => setOpen(true)}>
          Write outreach with Jobby
        </button>
      )}
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
  // "Reached out" filter (asked 2026-11-17).
  const [reach, setReach] = useState<"any" | "yes" | "no">("any");
  const [adding, setAdding] = useState<
    Record<string, "busy" | "added" | string>
  >({});
  const fileRef = useRef<HTMLInputElement>(null);
  // Nothing that depends on whether there's an upload shows until we know: the how-to flashed
  // up while the page loaded (asked 2026-11-17).
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    api
      .getNetwork()
      .then((n) => {
        setInfo(n);
        const first = TIERS.find((t) => n.tiers[t].length > 0);
        if (first) setTier(first);
      })
      .catch(() => setInfo(null))
      .finally(() => setLoaded(true));
  }, []);

  const [showSteps, setShowSteps] = useState(false);
  const upload = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const n = await api.uploadConnections(file);
      setInfo(n);
      setShowSteps(false);
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
  const q = search.trim().toLowerCase();
  // A search looks in every tier, Not listed included (asked 2026-11-20); otherwise the tab's tier.
  const all = info
    ? q
      ? [...TIERS.flatMap((t) => info.tiers[t]), ...info.not_listed]
      : tier === "Not listed"
        ? info.not_listed
        : info.tiers[tier]
    : [];
  const searched = q
    ? all.filter((p) =>
        [p.name, p.company, p.position].some((v) =>
          (v ?? "").toLowerCase().includes(q),
        ),
      )
    : all;
  const reachedCount = searched.filter((p) => p.reached_out).length;
  const rows = searched.filter((p) =>
    reach === "any" ? true : reach === "yes" ? !!p.reached_out : !p.reached_out,
  );

  // Saved at once; the box flips back with an error if the save fails.
  const markReached = async (id: number, reached: boolean) => {
    const apply = (value: string | null) =>
      setInfo((n) => {
        if (!n) return n;
        const fix = (list: typeof n.not_listed) =>
          list.map((p) => (p.id === id ? { ...p, reached_out: value } : p));
        return {
          ...n,
          not_listed: fix(n.not_listed),
          tiers: {
            High: fix(n.tiers.High),
            Medium: fix(n.tiers.Medium),
            Low: fix(n.tiers.Low),
          },
        };
      });
    apply(reached ? new Date().toISOString().slice(0, 10) : null);
    try {
      apply((await api.setReachedOut(id, reached)).reached_out);
    } catch (e) {
      apply(reached ? null : new Date().toISOString().slice(0, 10));
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const [notice, setNotice] = useState<string | null>(null);
  const changePriority = async (id: number, priority: NetworkTier | "Not listed" | null) => {
    setError(null);
    try {
      setInfo(await api.setNetworkPriority(id, priority));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  const removeContact = async (id: number) => {
    setError(null);
    try {
      setInfo(await api.removeContact(id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  const added = (n: NetworkOverview & { added_id: number }) => {
    setInfo(n);
    const person = [...TIERS.flatMap((t) => n.tiers[t]), ...n.not_listed].find((p) => p.id === n.added_id);
    if (person) {
      setTier(person.tier ?? "Not listed");
      setNotice(`Added ${person.name} (${person.tier ?? "not listed"}).`);
    }
  };

  return (
    <div className="network-page">
      {/* Top row (asked 2026-11-16): the upload on the left, organizations to add on the right. */}
      <div
        className={`network-top${has && info!.suggested_sources.length > 0 ? "" : " single"}`}
      >
        <div className="card network">
          <h2>Your LinkedIn network</h2>
          <p className="muted small">
            See which of your connections to reach out to about the jobs you
            want, and organizations you know people at. LinkedIn doesn't let
            apps read your connections directly, so this uses the file LinkedIn
            gives you.
          </p>

          {/* The how-to shows until a file is uploaded, and again when they go to upload a newer
              one or remove theirs (asked 2026-11-16). */}
          {!loaded && <p className="muted small">Loading…</p>}
          {loaded && (!has || showSteps) && (
            <details className="network-steps" open>
              <summary>How to get your Connections.csv from LinkedIn</summary>
              {/* Matches LinkedIn's page as of 2026-11-16 (the owner's screenshot): Connections isn't in
            the "Want something in particular?" list any more, so it's the full archive. */}
              <ol className="small">
                <li>
                  On LinkedIn, open <strong>Settings &amp; Privacy</strong> →{" "}
                  <strong>Data privacy</strong> →{" "}
                  <strong>Download my data</strong>.
                </li>
                <li>
                  Keep the first option selected,{" "}
                  <strong>
                    "Download larger data archive, including connections…"
                  </strong>{" "}
                  (Connections isn't in the "Want something in particular?"
                  list), and press <strong>Request archive</strong>.
                </li>
                <li>
                  LinkedIn takes a while to prepare it. Wait for LinkedIn's
                  email saying it's ready, then download the archive and unzip
                  it.
                </li>
                <li>
                  Upload <strong>Connections.csv</strong> from it below. The
                  archive holds more than your connections; only that file is
                  uploaded here. Upload a fresh one now and then: it's a
                  snapshot.
                </li>
              </ol>
              <p className="muted small">
                Your connections stay in your account only; nobody else using
                the app sees them. To find organizations that suit you, their
                organizations' names (never people's names) go to Claude once
                per upload, for about a cent. A person's details go to Claude
                only when you ask Jobby to write to them.
              </p>
            </details>
          )}

          <div className="form-actions network-upload" hidden={!loaded}>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,text/csv"
              style={{ display: "none" }}
              onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
            />
            <button
              className={has ? "ghost" : "primary"}
              onClick={() => {
                if (has) setShowSteps(true);
                fileRef.current?.click();
              }}
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
          {has && <NetworkStats info={info!} />}
          {has && <CurrentOrg info={info!} onSaved={setInfo} />}
          {has && info!.ratings_outdated && (
            <p className="small network-current">
              <span className="muted">
                High is now for organizations that suit your profile, so your
                connections' organizations need rating (their names only, about
                a cent or two).
              </span>{" "}
              <button className="ghost small" onClick={rerate} disabled={busy}>
                {busy ? <Spinner label="Rating..." /> : "Re-rate organizations"}
              </button>
            </p>
          )}
          {error && <p className="error-text small">{error}</p>}
          {info?.picks_error && (
            <p className="muted small">
              Couldn't rate your connections' organizations ({info.picks_error}
              ), so org alignment only counts your jobs and sources for now.
              Uploading again, or Re-rate organizations, retries.
            </p>
          )}
        </div>
        {has && info!.suggested_sources.length > 0 && (
          // The right card takes the left card's height and its list scrolls (asked 2026-11-16).
          <div className="org-suggestions-holder">
            <OrgSuggestions
              suggestions={info!.suggested_sources}
              adding={adding}
              onAdd={addSource}
            />
          </div>
        )}
      </div>

      {/* Shown once loaded even before an upload, so someone can be added by hand (asked 2026-11-20). */}
      {loaded && info && (
        <div className="card network">
          <div className="network-reach-head">
            <h3 className="network-heading first">Who to reach out to</h3>
            <AddContact onAdded={added} />
          </div>
          {notice && <p className="muted small">{notice}</p>}
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
            <div className="seg small" role="group" aria-label="Reached out">
              <button
                className={reach === "any" ? "on" : ""}
                onClick={() => setReach("any")}
              >
                Anyone
              </button>
              <button
                className={reach === "yes" ? "on" : ""}
                onClick={() => setReach("yes")}
              >
                Reached out ({reachedCount})
              </button>
              <button
                className={reach === "no" ? "on" : ""}
                onClick={() => setReach("no")}
              >
                Not yet
              </button>
            </div>
          </div>
          {q && <p className="muted small">Searching every tier.</p>}
          {rows.length === 0 ? (
            <p className="empty-state small">
              {q || reach !== "any"
                ? "Nobody here matches that."
                : has
                  ? "Nobody in this tier."
                  : "Nobody yet: upload your Connections.csv above, or add someone with + Add a contact."}
            </p>
          ) : (
            <div className="network-table-wrap">
              <table className="jobs-table network-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Position</th>
                    <th>Organization</th>
                    <th>Org alignment</th>
                    <th>Title alignment</th>
                    <th>Priority</th>
                    <th className="center">Reached out</th>
                    <th className="center">Write outreach</th>
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
                        {p.manual && (
                          <button
                            className="link small remove-contact"
                            title="Remove this contact (added by you)"
                            aria-label={`Remove ${p.name}`}
                            onClick={() => removeContact(p.id)}
                          >
                            ×
                          </button>
                        )}
                      </td>
                      <td>{p.position}</td>
                      <td>{p.company}</td>
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
                        <PrioritySelect person={p} onChange={(v) => changePriority(p.id, v)} />
                      </td>
                      <td className="center">
                        <input
                          type="checkbox"
                          checked={!!p.reached_out}
                          onChange={(e) => markReached(p.id, e.target.checked)}
                          title={
                            p.reached_out
                              ? `Reached out ${p.reached_out}`
                              : "Mark as reached out"
                          }
                          aria-label={`Reached out to ${p.name}`}
                        />
                      </td>
                      <td className="center">
                        {p.tier && <OutreachButton person={p} compact />}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** A few numbers about the uploaded network (asked 2026-11-16), from what's already loaded. No
 * "at aligned orgs": every Med/High org contact is High or Medium, so it repeated "worth reaching
 * out to". */
function NetworkStats({ info }: { info: NetworkOverview }) {
  const everyone = [...TIERS.flatMap((t) => info.tiers[t]), ...info.not_listed];
  const goodFit = everyone.filter((p) => p.org_align === "High").length;
  const roles = everyone.filter((p) => p.title_align !== "Low").length;
  const worth = info.tiers.High.length + info.tiers.Medium.length;
  const stats: [number, string][] = [
    [info.count, "connections"],
    [goodFit, "at good-fit orgs"],
    [roles, "in relevant roles"],
    [worth, "worth reaching out to"],
  ];
  return (
    <div className="screen-stats network-stats">
      {stats.map(([n, label]) => (
        <div key={label}>
          <strong>{n.toLocaleString()}</strong>
          <span className="muted small">{label}</span>
        </div>
      ))}
    </div>
  );
}

const FITS: { tier: NetworkTier; label: string }[] = [
  { tier: "High", label: "High" },
  { tier: "Medium", label: "Med" },
  { tier: "Low", label: "Low" },
];

/** Organizations you know people at but don't follow, in a small card that scrolls, filtered by
 * fit (the tier of the best person you know there; asked 2026-11-16). */
function OrgSuggestions({
  suggestions,
  adding,
  onAdd,
}: {
  suggestions: NetworkOverview["suggested_sources"];
  adding: Record<string, string>;
  onAdd: (name: string) => void;
}) {
  const [fits, setFits] = useState<NetworkTier[]>(["High", "Medium", "Low"]);
  const toggle = (t: NetworkTier) =>
    setFits((f) => (f.includes(t) ? f.filter((x) => x !== t) : [...f, t]));
  const shown = suggestions.filter((s) => fits.includes(s.tier));
  return (
    <div className="card org-suggestions">
      <h3 className="network-heading first">
        Organizations to consider tracking
      </h3>
      <div className="org-fit-toggles">
        <span className="small">Fit:</span>
        <div className="chip-toggle-group">
          {FITS.map(({ tier, label }) => (
            <button
              key={tier}
              className={`chip-toggle${fits.includes(tier) ? " on" : ""}`}
              onClick={() => toggle(tier)}
            >
              {label} ({suggestions.filter((s) => s.tier === tier).length})
            </button>
          ))}
        </div>
      </div>
      <ul className="network-sources org-suggestions-list">
        {shown.length === 0 && (
          <li className="muted small">None with this fit.</li>
        )}
        {shown.map((s) => {
          const state = adding[s.name];
          return (
            <li key={s.name}>
              <div>
                <strong>{s.name}</strong>{" "}
                <span className={`network-tier tier-${s.tier.toLowerCase()}`}>
                  {s.tier === "Medium" ? "Med" : s.tier}
                </span>
                <div className="muted small">
                  {s.connections} connection{s.connections === 1 ? "" : "s"}
                  {s.leader && " · you know a leader there"}
                </div>
                {s.why && <div className="muted small">{s.why}</div>}
                {state && state !== "busy" && state !== "added" && (
                  <div className="error-text small">{state}</div>
                )}
              </div>
              {state === "added" ? (
                <span className="small added-mark">✓ Added</span>
              ) : (
                <button
                  className="primary small"
                  onClick={() => onAdd(s.name)}
                  disabled={state === "busy"}
                >
                  {state === "busy" ? <Spinner label="Adding..." /> : "+ Add"}
                </button>
              )}
            </li>
          );
        })}
      </ul>
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
