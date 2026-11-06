import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";
import { Markdown } from "./Markdown";
import { Spinner } from "./Spinner";

// The candidate-profile card -- shared by the Profile page and onboarding's "Your profile" step.
// Changes come in three ways: a full rebuild from uploaded documents, a plain-language change
// request Claude applies ("Ask Claude to change it"), or a direct edit of the text.
export function CandidateProfileCard({
  hasDocuments,
  autoRebuild = false,
  title = "Candidate profile",
  description = "Synthesized from everything uploaded above.",
  onProfileChange,
}: {
  hasDocuments: boolean;
  // Onboarding builds the profile as soon as the step opens instead of waiting for a click.
  autoRebuild?: boolean;
  title?: string;
  description?: string;
  onProfileChange?: (text: string | null) => void;
}) {
  const [profile, setProfileState] = useState<string | null>(null);
  const setProfile = (text: string | null) => {
    setProfileState(text);
    onProfileChange?.(text);
  };
  const [busy, setBusy] = useState<null | "rebuild" | "revise" | "save">(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [instructions, setInstructions] = useState("");

  const run = async (kind: "rebuild" | "revise" | "save", action: () => Promise<{ text: string }>) => {
    setBusy(kind);
    setError(null);
    try {
      const r = await action();
      setProfile(r.text);
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setBusy(null);
    }
  };

  const handleRebuild = () => run("rebuild", api.rebuildCandidateProfile);

  // The ref keeps StrictMode's dev-only double mount from firing two (paid) rebuilds.
  const autoRebuilt = useRef(false);
  useEffect(() => {
    if (autoRebuild && !autoRebuilt.current) {
      autoRebuilt.current = true;
      handleRebuild();
      return;
    }
    api
      .getCandidateProfile()
      .then((r) => setProfile(r.text))
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) setProfile(null);
      });
  }, []);

  const handleRevise = async () => {
    if (await run("revise", () => api.reviseCandidateProfile(instructions))) setInstructions("");
  };

  const handleSaveEdit = async () => {
    if (await run("save", () => api.updateCandidateProfile(draft))) setEditing(false);
  };

  return (
    <div className="card">
      <h2>{title}</h2>
      <p className="muted small">{description}</p>
      {!editing && (
        <div className="form-actions" style={{ marginTop: 0 }}>
          <button className="ghost" onClick={handleRebuild} disabled={!!busy || !hasDocuments}>
            {busy === "rebuild" ? (
              <Spinner label={autoRebuild ? "Reading your documents..." : "Rebuilding..."} />
            ) : (
              "Rebuild from uploaded documents"
            )}
          </button>
          {profile && (
            <button
              className="link small"
              onClick={() => {
                setDraft(profile);
                setEditing(true);
              }}
              disabled={!!busy}
            >
              Edit directly
            </button>
          )}
        </div>
      )}
      {error && <p className="error-text small">{error}</p>}

      {editing ? (
        <div className="card-section">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={14}
            style={{ width: "100%", fontFamily: "inherit" }}
          />
          <div className="form-actions">
            <button className="primary" onClick={handleSaveEdit} disabled={!!busy || !draft.trim()}>
              {busy === "save" ? <Spinner label="Saving..." /> : "Save"}
            </button>
            <button className="ghost" onClick={() => setEditing(false)} disabled={!!busy}>
              Cancel
            </button>
          </div>
        </div>
      ) : profile ? (
        <>
          <div className="card-section">
            <Markdown text={profile} />
          </div>
          <div className="card-section">
            <span className="flabel">Ask Claude to change it</span>
            <textarea
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              rows={3}
              placeholder="e.g. Put more weight on my policy work, and drop the part about teaching — I'm not looking to go back to that."
              style={{ width: "100%" }}
            />
            <div className="form-actions">
              <button
                className="ghost"
                onClick={handleRevise}
                disabled={!!busy || !instructions.trim()}
              >
                {busy === "revise" ? <Spinner label="Revising..." /> : "Revise profile"}
              </button>
            </div>
          </div>
        </>
      ) : (
        busy !== "rebuild" && (
          <p className="empty-state">No profile yet — upload at least one document and rebuild.</p>
        )
      )}
    </div>
  );
}
