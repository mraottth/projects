import { useEffect, useState } from "react";
import { api, Settings } from "../api";
import { TagInput } from "./TagInput";

// The "What are you looking for?" card -- shared by the Profile page and onboarding's second step.
// Every field saves itself on change/blur, so there's no separate submit.

export function DesiredRoleForm() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getSettings().then(setSettings);
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
          onChange={(tags) => save({ prefilter_locations: tags })}
          placeholder="e.g. New York, Remote"
        />
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
          onBlur={(e) => save({ salary_floor: e.target.value ? Number(e.target.value) : null })}
          style={{ width: "200px" }}
        />
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
