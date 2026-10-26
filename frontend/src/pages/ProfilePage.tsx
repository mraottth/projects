import { useEffect, useRef, useState } from "react";
import { api, ApiError, CompanySuggestion, ResumeSummary, Settings } from "../api";

function toList(s: string): string[] {
  return s
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

function ResumesSection() {
  const [resumes, setResumes] = useState<ResumeSummary[]>([]);
  const [profile, setProfile] = useState<string | null>(null);
  const [kind, setKind] = useState<"resume" | "cover_letter">("resume");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [libraryPath, setLibraryPath] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = () => {
    api.listResumes().then(setResumes);
    api
      .getCandidateProfile()
      .then((r) => setProfile(r.text))
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) setProfile(null);
      });
    api
      .getSettings()
      .then((s) => setLibraryPath((s as unknown as Record<string, string>).resume_library_path ?? null));
  };

  useEffect(refresh, []);

  const handleUpload = async () => {
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      await api.uploadResume(file, kind);
      if (fileInput.current) fileInput.current.value = "";
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleRebuild = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.rebuildCandidateProfile();
      setProfile(r.text);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="card">
        <h2>Resumes &amp; cover letters</h2>
        <input ref={fileInput} type="file" accept=".pdf,.docx,.md,.txt" />
        <select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
          <option value="resume">Resume</option>
          <option value="cover_letter">Cover letter</option>
        </select>
        <button onClick={handleUpload} disabled={busy}>
          Upload
        </button>
        {error && <p style={{ color: "crimson" }}>{error}</p>}
        {libraryPath && (
          <p className="muted">
            Bulk-imported from <code>{libraryPath}</code> — new tailored docs save to{" "}
            <code>{libraryPath}/AI Generated</code>.
          </p>
        )}

        {(() => {
          const current = resumes.filter((r) => !r.is_archived);
          const archived = resumes.filter((r) => r.is_archived);
          const renderRow = (r: ResumeSummary) => (
            <li key={r.id}>
              [{r.kind}] {r.is_template ? "★ " : ""}
              {r.file_name} — {r.text_length} chars —{" "}
              <button onClick={() => api.deleteResume(r.id).then(refresh)}>delete</button>
            </li>
          );
          return (
            <>
              <ul>{current.map(renderRow)}</ul>
              {current.length === 0 && resumes.length === 0 && <p className="muted">None yet.</p>}
              {archived.length > 0 && (
                <>
                  <button onClick={() => setShowArchived((s) => !s)}>
                    {showArchived ? "Hide" : "Show"} {archived.length} archived document
                    {archived.length === 1 ? "" : "s"}
                  </button>
                  {showArchived && <ul>{archived.map(renderRow)}</ul>}
                </>
              )}
            </>
          );
        })()}
      </div>

      <div className="card">
        <h2>Candidate profile</h2>
        <p className="muted">Synthesized from everything uploaded above.</p>
        <button onClick={handleRebuild} disabled={busy || resumes.length === 0}>
          Rebuild from uploaded documents
        </button>
        {profile ? (
          <p style={{ whiteSpace: "pre-wrap" }}>{profile}</p>
        ) : (
          <p className="muted">No profile yet — upload at least one document and rebuild.</p>
        )}
      </div>
    </>
  );
}

function DesiredRoleSection() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.getSettings().then(setSettings);
  }, []);

  if (!settings) return <p className="muted">Loading...</p>;

  const save = async (patch: Partial<Settings>) => {
    setBusy(true);
    setSaved(false);
    try {
      const updated = await api.putSettings(patch);
      setSettings(updated);
      setSaved(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>What are you looking for?</h2>
      <p className="muted">
        This drives both scoring (jobs that don't match get down-weighted) and the "discover
        companies" search below.
      </p>

      <label style={{ display: "block", marginBottom: "0.5rem" }}>
        Desired job titles
        <input
          defaultValue={settings.prefilter_titles_include.join(", ")}
          onBlur={(e) => save({ prefilter_titles_include: toList(e.target.value) })}
          placeholder="e.g. Senior Backend Engineer, Staff Engineer"
          style={{ display: "block", width: "100%" }}
        />
      </label>

      <label style={{ display: "block", marginBottom: "0.5rem" }}>
        Desired locations
        <input
          defaultValue={settings.prefilter_locations.join(", ")}
          onBlur={(e) => save({ prefilter_locations: toList(e.target.value) })}
          placeholder="e.g. New York, Remote"
          style={{ display: "block", width: "100%" }}
        />
      </label>

      <label style={{ display: "block", marginBottom: "0.5rem" }}>
        <input
          type="checkbox"
          defaultChecked={settings.prefilter_remote_only}
          onChange={(e) => save({ prefilter_remote_only: e.target.checked })}
        />{" "}
        Remote only
      </label>

      <label style={{ display: "block" }}>
        Minimum salary
        <input
          type="number"
          defaultValue={settings.salary_floor ?? ""}
          onBlur={(e) => save({ salary_floor: e.target.value ? Number(e.target.value) : null })}
          style={{ display: "block", width: "100%" }}
        />
      </label>

      {busy && <p className="muted">Saving...</p>}
      {saved && !busy && <p className="muted">Saved.</p>}
    </div>
  );
}

function DiscoverCompaniesSection() {
  const [suggestions, setSuggestions] = useState<CompanySuggestion[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<Record<string, string>>({});

  const runDiscovery = async () => {
    setBusy(true);
    setError(null);
    try {
      setSuggestions(await api.suggestCompanies());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const approve = async (companyName: string, favorite: boolean) => {
    setStatus((s) => ({ ...s, [companyName]: "working" }));
    try {
      if (favorite) {
        const result = await api.addFavorite(companyName);
        setStatus((s) => ({
          ...s,
          [companyName]: result.resolution_status === "resolved" ? "added" : "not found",
        }));
      } else {
        await api.resolveSource(companyName);
        setStatus((s) => ({ ...s, [companyName]: "added" }));
      }
    } catch (e) {
      setStatus((s) => ({ ...s, [companyName]: e instanceof Error ? e.message : String(e) }));
    }
  };

  return (
    <div className="card">
      <h2>Discover companies</h2>
      <p className="muted">
        Uses your candidate profile and desired role above, plus a web search, to suggest
        companies worth monitoring. Nothing is added until you approve it.
      </p>
      <button onClick={runDiscovery} disabled={busy}>
        {busy ? "Searching..." : "Suggest companies"}
      </button>
      {error && <p style={{ color: "crimson" }}>{error}</p>}

      {suggestions.map((s) => (
        <div key={s.company_name} className="card" style={{ marginTop: "0.5rem" }}>
          <strong>{s.company_name}</strong>
          <p className="muted">{s.reason}</p>
          {status[s.company_name] ? (
            <p className="muted">{status[s.company_name]}</p>
          ) : (
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button onClick={() => approve(s.company_name, false)}>Monitor</button>
              <button onClick={() => approve(s.company_name, true)}>Monitor &amp; favorite</button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function ProfilePage() {
  return (
    <div>
      <h1>Profile</h1>
      <ResumesSection />
      <DesiredRoleSection />
      <DiscoverCompaniesSection />
    </div>
  );
}
