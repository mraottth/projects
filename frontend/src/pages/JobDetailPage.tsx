import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, Application, GeneratedDoc, Job } from "../api";

interface JobWithScores extends Job {
  scores: {
    interest_fit: string | null;
    candidate_fit: string | null;
    summary_text: string | null;
    rationale_text: string | null;
    scored_at: string;
  }[];
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
      <p className="muted">
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
          <p>
            <strong>Interest fit:</strong> {latestScore.interest_fit} ·{" "}
            <strong>Candidate fit:</strong> {latestScore.candidate_fit}
          </p>
          <p>{latestScore.summary_text}</p>
          <p className="muted">{latestScore.rationale_text}</p>
        </div>
      )}

      <div className="card">
        <h2>Description</h2>
        <p style={{ whiteSpace: "pre-wrap" }}>{job.description_text ?? "(no description available)"}</p>
      </div>

      <div className="card">
        <h2>Pursuing this one?</h2>
        {!application && (
          <button onClick={track} disabled={busy === "track"}>
            Track application
          </button>
        )}
        {application && <p className="muted">Tracked — status: {application.status}</p>}

        <div style={{ marginTop: "0.5rem", display: "flex", gap: "0.5rem" }}>
          <button onClick={() => generate("resume")} disabled={busy === "resume"}>
            {busy === "resume" ? "Generating..." : "Generate tailored resume"}
          </button>
          <button onClick={() => generate("cover-letter")} disabled={busy === "cover-letter"}>
            {busy === "cover-letter" ? "Generating..." : "Generate cover letter"}
          </button>
        </div>
        {error && <p style={{ color: "crimson" }}>{error}</p>}
      </div>

      {docs.map((doc) => (
        <div className="card" key={doc.id}>
          <h3>{doc.kind === "resume" ? "Tailored resume" : "Cover letter"}</h3>
          <p className="muted">{doc.created_at}</p>
          {doc.file_path ? (
            <p className="muted">Saved to: {doc.file_path}</p>
          ) : (
            <p className="muted">
              Not saved as a formatted file (no usable .docx template found — generated as plain
              text instead).
            </p>
          )}
          <p style={{ whiteSpace: "pre-wrap" }}>{doc.content_text}</p>
        </div>
      ))}
    </div>
  );
}
