import { useEffect, useState } from "react";
import { api, type LocationCheck, type SalaryCheck, Settings } from "../api";
import { TagInput } from "./TagInput";
import { DEMO } from "../demo";

// The "What are you looking for?" card -- shared by the Profile page and onboarding's second step.
// Every field saves itself on change/blur, so there's no separate submit.

export function DesiredRoleForm() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checkVersion, setCheckVersion] = useState(0);
  // Sector ideas from their resume, shown as chips in the hint; clicking one adds it. Never
  // pre-filled (asked 2026-11-12: a tester didn't know what "sector" meant here).
  const [ideas, setIdeas] = useState<string[]>([]);

  useEffect(() => {
    api.getSettings().then(setSettings);
    if (!DEMO) {
      // A POST (it may make a small paid call), which the read-only demo would refuse with a notice.
      api
        .getSectorSuggestions()
        .then((r) => setIdeas(r.sectors))
        .catch(() => setIdeas([]));
    }
  }, []);

  if (!settings) return <p className="muted">Loading...</p>;

  // Optimistic: the field updates immediately (a tag chip shouldn't wait on a server round trip,
  // which on a cold Cloud Run instance can take seconds), then saves in the background. The
  // server's response deliberately isn't written back -- with two quick edits in flight, the first
  // response would briefly clobber the second edit. A failed save reverts just the fields it touched.
  const save = async (patch: Partial<Settings>) => {
    const previous = Object.fromEntries(
      Object.keys(patch).map((k) => [k, settings[k as keyof Settings]]),
    ) as Partial<Settings>;
    setSettings((s) => (s ? { ...s, ...patch } : s));
    setBusy(true);
    setSaved(false);
    setError(null);
    try {
      await api.putSettings(patch);
      setSaved(true);
    } catch (e) {
      setSettings((s) => (s ? { ...s, ...previous } : s));
      setError(`Couldn't save: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>What are you looking for?</h2>
      <p className="muted small">
        This drives both scoring (jobs that don't match get down-weighted) and company/job-board
        discovery on the Manage Sources page.
      </p>

      <div className="fsection">
        <span className="flabel">Desired job titles</span>
        <p className="muted small" style={{ marginTop: 0 }}>
          Add as many as you like — type one and press Enter (or a comma) after each.
        </p>
        <TagInput
          value={settings.prefilter_titles_include}
          onChange={(tags) => save({ prefilter_titles_include: tags })}
          placeholder="e.g. Senior Backend Engineer, Staff Engineer"
        />
      </div>

      <div className="fsection">
        <span className="flabel">Desired sectors</span>
        <p className="muted small" style={{ marginTop: 0 }}>
          The kind of work or organization: for example civic tech, political data and campaigns,
          public health, education, climate, fintech, nonprofit or government.
          {ideas.filter((i) => !settings.prefilter_sectors.some((s) => s.toLowerCase() === i.toLowerCase()))
            .length > 0 && (
            <>
              {" "}
              From your resume:{" "}
              {ideas
                .filter((i) => !settings.prefilter_sectors.some((s) => s.toLowerCase() === i.toLowerCase()))
                .map((idea) => (
                  <button
                    key={idea}
                    type="button"
                    className="sector-idea"
                    title={`Add "${idea}"`}
                    onClick={() => save({ prefilter_sectors: [...settings.prefilter_sectors, idea] })}
                  >
                    + {idea}
                  </button>
                ))}
            </>
          )}
        </p>
        <TagInput
          value={settings.prefilter_sectors}
          onChange={(tags) => save({ prefilter_sectors: tags })}
          placeholder="e.g. Climate tech, Healthcare"
        />
      </div>

      <div className="fsection">
        <span className="flabel">Desired locations</span>
        <TagInput
          value={settings.prefilter_locations}
          onChange={(tags) => save({ prefilter_locations: tags }).then(() => setCheckVersion((v) => v + 1))}
          placeholder="e.g. DC, New York, Remote"
        />
        {settings.prefilter_locations.length > 0 && (
          <>
            <label className="field inline" style={{ margin: "10px 0 0" }}>
              <input
                type="checkbox"
                checked={!!settings.location_dealbreaker}
                onChange={(e) =>
                  save({ location_dealbreaker: e.target.checked }).then(() => setCheckVersion((v) => v + 1))
                }
              />
              <span style={{ margin: 0, fontWeight: 400, color: "var(--ink)" }}>
                Dealbreaker: only show jobs in these locations
              </span>
            </label>
            <p className="muted small" style={{ margin: "4px 0 0" }}>
              Jobs clearly somewhere else aren't scored or shown (My Jobs is never filtered). Remote
              jobs, jobs listing several locations, and jobs whose location can't be told always
              stay.
            </p>
            <LocationReadout version={checkVersion} enabled={!!settings.location_dealbreaker} />
          </>
        )}
      </div>

      <div className="fsection">
        <label className="field inline" style={{ marginBottom: 0 }}>
          <input
            type="checkbox"
            defaultChecked={settings.prefilter_remote_only}
            onChange={(e) => save({ prefilter_remote_only: e.target.checked })}
          />
          <span style={{ margin: 0, fontWeight: 400, color: "var(--ink)" }}>Remote only</span>
        </label>
      </div>

      <div className="fsection">
        <span className="flabel">Minimum salary</span>
        <input
          type="number"
          defaultValue={settings.salary_floor ?? ""}
          onBlur={(e) =>
            save({ salary_floor: e.target.value ? Number(e.target.value) : null }).then(() =>
              setCheckVersion((v) => v + 1),
            )
          }
          style={{ width: "200px" }}
        />
        {!!settings.salary_floor && (
          <>
            <label className="field inline" style={{ margin: "10px 0 0" }}>
              <input
                type="checkbox"
                checked={!!settings.salary_dealbreaker}
                onChange={(e) =>
                  save({ salary_dealbreaker: e.target.checked }).then(() => setCheckVersion((v) => v + 1))
                }
              />
              <span style={{ margin: 0, fontWeight: 400, color: "var(--ink)" }}>
                Dealbreaker: hide jobs that pay less
              </span>
            </label>
            <p className="muted small" style={{ margin: "4px 0 0" }}>
              Compares the low end of a posted range, per year (hourly pay × 2,080). Jobs with no
              pay listed, or in another currency, always stay.
            </p>
            <SalaryReadout version={checkVersion} enabled={!!settings.salary_dealbreaker} />
          </>
        )}
      </div>

      <div className="fsection" style={{ marginBottom: 0 }}>
        <span className="flabel">In your own words, what are you looking for?</span>
        <p className="muted small" style={{ marginTop: 0 }}>
          A few sentences about what you want — type of work, company stage, what you're trying to
          avoid, anything that doesn't fit neatly into the fields above. Used for both company/job-
          board discovery and job scoring, weighted heavily since it's your own description, not a
          structured guess.
        </p>
        <textarea
          defaultValue={settings.looking_for_notes}
          onBlur={(e) => save({ looking_for_notes: e.target.value })}
          rows={4}
          placeholder="e.g. I want to stay in early-stage startups doing hands-on technical work, ideally with a mission I care about. Not interested in pure management roles or big tech."
          style={{ width: "100%" }}
        />
      </div>

      {busy && <p className="muted small">Saving...</p>}
      {saved && !busy && <p className="muted small">Saved.</p>}
      {error && <p className="error-text small">{error}</p>}
    </div>
  );
}

/** What the minimum-salary dealbreaker hides (or would hide) among All Jobs right now. */
function SalaryReadout({ version, enabled }: { version: number; enabled: boolean }) {
  const [check, setCheck] = useState<SalaryCheck | null>(null);

  useEffect(() => {
    api
      .getSalaryCheck()
      .then(setCheck)
      .catch(() => setCheck(null));
  }, [version]);

  if (!check || !check.floor || check.preview.total === 0) return null;
  const { preview } = check;
  const money = (n: number) => `$${Math.round(n / 1000)}k`;
  return (
    <div className="location-readout small">
      <p className="muted" style={{ margin: 0 }}>
        {preview.with_pay} of your {preview.total} jobs on All Jobs list pay.{" "}
        {preview.hidden === 0
          ? `None of them start below ${money(check.floor)}.`
          : `${enabled ? "Hiding" : "Would hide"} ${preview.hidden} that start below ${money(check.floor)}`}
        {preview.examples.length > 0 &&
          `, e.g. ${preview.examples
            .map((e) => `${e.title} at ${e.company} (from ${money(e.annual_low)}/yr)`)
            .join(" · ")}.`}
      </p>
    </div>
  );
}

/** How each desired location is read, and what the dealbreaker hides (or would hide) right now. */
function LocationReadout({ version, enabled }: { version: number; enabled: boolean }) {
  const [check, setCheck] = useState<LocationCheck | null>(null);
  const [open, setOpen] = useState(false);
  const [unsureOpen, setUnsureOpen] = useState(false);

  useEffect(() => {
    api
      .getLocationCheck()
      .then(setCheck)
      .catch(() => setCheck(null));
  }, [version]);

  if (!check) return null;
  const { preview } = check;
  const shown = open ? preview.hidden_locations : preview.hidden_locations.slice(0, 6);
  return (
    <div className="location-readout small">
      <ul>
        {check.entries.map((e) => (
          <li key={e.entry}>
            <strong>{e.entry}</strong> → {e.matches.join("; ")}
            {!e.recognized && (
              <span className="muted"> (not a place we recognize, so matched as written)</span>
            )}
          </li>
        ))}
      </ul>
      {preview.total > 0 && (
        <p className="muted" style={{ margin: "6px 0 0" }}>
          {preview.hidden === 0
            ? `None of your ${preview.total} jobs on All Jobs are outside these locations.`
            : `${enabled ? "Hiding" : "Would hide"} ${preview.hidden} of your ${preview.total} jobs on All Jobs: `}
          {shown.map((h, i) => (
            <span key={h.location}>
              {i > 0 && " · "}
              {h.location} ({h.count})
            </span>
          ))}
          {preview.hidden_locations.length > 6 && (
            <button className="link small" onClick={() => setOpen(!open)}>
              {open ? " show fewer" : ` +${preview.hidden_locations.length - 6} more`}
            </button>
          )}
        </p>
      )}
      {preview.unsure_locations.length > 0 && (
        <p className="muted" style={{ margin: "6px 0 0" }}>
          Kept because we can't tell where they are ({preview.unsure_locations.reduce((n, u) => n + u.count, 0)}{" "}
          jobs):{" "}
          {(unsureOpen ? preview.unsure_locations : preview.unsure_locations.slice(0, 6)).map((u, i) => (
            <span key={u.location}>
              {i > 0 && " · "}
              {u.location} ({u.count})
            </span>
          ))}
          {preview.unsure_locations.length > 6 && (
            <button className="link small" onClick={() => setUnsureOpen(!unsureOpen)}>
              {unsureOpen ? " show fewer" : ` +${preview.unsure_locations.length - 6} more`}
            </button>
          )}
        </p>
      )}
    </div>
  );
}
