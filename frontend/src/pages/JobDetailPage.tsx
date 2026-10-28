import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, Application, APPLICATION_STATUS_LABELS, GeneratedDoc, Job } from "../api";
import { Markdown } from "../components/Markdown";
import { Spinner } from "../components/Spinner";

interface JobWithScores extends Job {
  scores: {
    interest_fit: string | null;
    candidate_fit: string | null;
    summary_text: string | null;
    rationale_text: string | null;
    scored_at: string;
  }[];
}

function FitBadge({ label, value }: { label: string; value: string | null }) {
  if (!value) return null;
  const cls = value === "High" ? "badge-high" : value === "Med" ? "badge-med" : "badge-low";
  return (
    <span className={`badge ${cls}`}>
      {label}: {value}
    </span>
  );
}

export default function JobDetailPage() {
  const { id } = useParams();
  const jobId = Number(id);

  const [job, setJob] = useState<JobWithScores | null>(null);
  const [application, setApplication] = useState<Application | null>(null);
  const [docs, setDocs] = useState<GeneratedDoc[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    const jobData = (await api.getJob(jobId)) as JobWithScores;
    setJob(jobData);
    const apps = await api.listApplications();
    setApplication(apps.find((a) => a.job_id === jobId) ?? null);
    setDocs(await api.getGenerationHistory(jobId));
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  if (!job) return <p className="muted">Loading...</p>;
  const latestScore = job.scores[0];

  const track = async () => {
    setBusy("track");
    setError(null);
    try {
      await api.createApplication(jobId);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const generate = async (kind: "resume" | "cover-letter") => {
    setBusy(kind);
    setError(null);
    try {
      if (kind === "resume") {
        await api.generateResume(jobId, application?.id);
      } else {
        await api.generateCoverLetter(jobId, application?.id);
      }
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div>
      <h1>
        {job.title} — {job.company}
      </h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        {job.location ?? "location unknown"}
        {job.url && (
          <>
            {" · "}
            <a href={job.url} target="_blank" rel="noreferrer">
              original posting
            </a>
          </>
        )}
      </p>

      {latestScore && (
        <div className="card">
          <h2>Scoring</h2>
          <div className="badge-row">
            <FitBadge label="Interest" value={latestScore.interest_fit} />
            <FitBadge label="Fit" value={latestScore.candidate_fit} />
          </div>
          {latestScore.summary_text && <p>{latestScore.summary_text}</p>}
          {latestScore.rationale_text && (
            <div className="muted small">
              <Markdown text={latestScore.rationale_text} />
            </div>
          )}
        </div>
      )}

      <div className="card">
        <h2>Description</h2>
        {job.description_text ? (
          <Markdown text={job.description_text} />
        ) : (
          <p className="empty-state">(no description available)</p>
        )}
      </div>

      <div className="card">
        <h2>Pursuing this one?</h2>
        {!application && (
          <button className="ghost" onClick={track} disabled={busy === "track"}>
            Track application
          </button>
        )}
        {application && (
          <p className="muted">
            Tracked — status:{" "}
            <span className={`pill pill-${application.status}`}>
              {APPLICATION_STATUS_LABELS[application.status]}
            </span>
          </p>
        )}

        <div className="form-actions">
          <button className="primary" onClick={() => generate("resume")} disabled={busy === "resume"}>
            {busy === "resume" ? <Spinner label="Tailoring..." /> : "Tailor my resume"}
          </button>
          <button className="primary" onClick={() => generate("cover-letter")} disabled={busy === "cover-letter"}>
            {busy === "cover-letter" ? <Spinner label="Tailoring..." /> : "Tailor my cover letter"}
          </button>
        </div>
        {error && <p className="error-text small">{error}</p>}
      </div>

      {docs.map((doc) => (
        <div className="card" key={doc.id}>
          <h3>{doc.kind === "resume" ? "Tailored resume" : "Cover letter"}</h3>
          <p className="muted small">{doc.created_at}</p>
          {doc.file_path ? (
            <p className="muted small">
              Saved to: <code className="path">{doc.file_path}</code>
            </p>
          ) : (
            <p className="muted small">
              Not saved as a formatted file (no usable .docx template found — generated as plain
              text instead).
            </p>
          )}
          <div className="card-section">
            <Markdown text={doc.content_text} />
          </div>
        </div>
      ))}
    </div>
  );
}
