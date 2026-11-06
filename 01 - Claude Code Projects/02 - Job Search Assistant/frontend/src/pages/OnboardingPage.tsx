import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, OnboardingStep, ResumeSummary, RunProgress, runProgressText } from "../api";
import { CandidateProfileCard } from "../components/CandidateProfileCard";
import { DesiredRoleForm } from "../components/DesiredRoleForm";
import { DiscoverCompaniesSection, DiscoverJobBoardsSection } from "../components/Discovery";
import { Markdown } from "../components/Markdown";
import { Spinner } from "../components/Spinner";

const STEPS: { key: OnboardingStep; label: string }[] = [
  { key: "resume", label: "Resume" },
  { key: "looking_for", label: "What you want" },
  { key: "cover_letter", label: "Cover letter" },
  { key: "old_docs", label: "Older docs" },
  { key: "profile", label: "Profile" },
  { key: "sources", label: "Sources" },
  { key: "finish", label: "First jobs" },
];

const ACCEPT = ".pdf,.docx,.md,.txt";

// The first run scores only the most promising jobs so the user isn't kept waiting; the rest stay
// queued (highest priority first) for the next daily run.
const FIRST_RUN_CAP = 20;

function errorText(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

// Same filename rule as cli/ingest_resumes.py's kind classification.
function guessKind(fileName: string): "resume" | "cover_letter" {
  return fileName.toLowerCase().includes("cover") ? "cover_letter" : "resume";
}

// The upload control for a step's "current" document. Once one is on file, the confirmation is
// the prominent thing and replacing it shrinks to a small link -- before this, a tiny check and
// chip sat above a full-size file input and button, so a successful upload didn't read as done.
function DocUploader({
  kind,
  current,
  onUploaded,
}: {
  kind: "resume" | "cover_letter";
  current?: ResumeSummary;
  onUploaded: () => void;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [replacing, setReplacing] = useState(false);

  const upload = async () => {
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      await api.uploadResume(file, kind);
      if (fileInput.current) fileInput.current.value = "";
      setReplacing(false);
      onUploaded();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const picker = (small: boolean) => (
    <div className="form-row" style={{ marginBottom: 0 }}>
      <input ref={fileInput} type="file" accept={ACCEPT} />
      <button className={small ? "ghost small" : "primary"} onClick={upload} disabled={busy}>
        {busy ? <Spinner label="Uploading..." /> : "Upload"}
      </button>
    </div>
  );

  return (
    <>
      {current ? (
        <>
          <div className="upload-success">
            <span className="upload-success-check" aria-hidden="true">
              ✓
            </span>
            <div>
              <span className="upload-success-file">{current.file_name}</span>
              <div className="muted small">
                Uploaded — {current.text_length.toLocaleString()} characters read
              </div>
            </div>
          </div>
          {replacing ? (
            picker(true)
          ) : (
            <button className="link small" onClick={() => setReplacing(true)}>
              Replace with a different file
            </button>
          )}
        </>
      ) : (
        picker(false)
      )}
      {error && <p className="error-text small">{error}</p>}
    </>
  );
}

function ResumeStep({ current, onUploaded }: { current?: ResumeSummary; onUploaded: () => void }) {
  return (
    <div className="card">
      <h2>Upload your current resume</h2>
      <p className="muted small">
        Everything else builds on this: job scoring, company suggestions, and tailored resumes. A
        Word file (<code>.docx</code>) works best. Tailored resumes are written as tracked changes
        to your own <code>.docx</code>, so with a PDF you'd only get plain text back.
      </p>
      <DocUploader kind="resume" current={current} onUploaded={onUploaded} />
    </div>
  );
}

function CoverLetterStep({ current, onChanged }: { current?: ResumeSummary; onChanged: () => void }) {
  const [mode, setMode] = useState<"upload" | "generate">("upload");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generated, setGenerated] = useState<{ id: number; content_text: string } | null>(null);

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await api.generateStarterCoverLetter();
      // A regenerated draft replaces the previous one outright rather than piling up in Old --
      // a discarded AI draft isn't a real past cover letter, and shouldn't feed the profile.
      if (generated) await api.deleteResume(generated.id).catch(() => undefined);
      setGenerated(result);
      onChanged();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>Add a recent cover letter</h2>
      <p className="muted small">
        Tailored cover letters reuse its structure and formatting. Don't have a recent one? Claude
        can write a general-purpose one from your resume and what you're looking for.
      </p>
      <div style={{ marginBottom: "12px" }}>
        <div className="seg">
          <button className={mode === "upload" ? "on" : ""} onClick={() => setMode("upload")}>
            Upload my own
          </button>
          <button className={mode === "generate" ? "on" : ""} onClick={() => setMode("generate")}>
            Write one for me
          </button>
        </div>
      </div>

      {mode === "upload" ? (
        <DocUploader
          kind="cover_letter"
          current={generated ? undefined : current}
          onUploaded={() => {
            setGenerated(null);
            onChanged();
          }}
        />
      ) : (
        <>
          <button className={generated ? "ghost" : "primary"} onClick={generate} disabled={busy}>
            {busy ? (
              <Spinner label="Writing your cover letter..." />
            ) : generated ? (
              "Regenerate"
            ) : (
              "Write my cover letter"
            )}
          </button>
          {generated && (
            <div className="card-section">
              <p className="muted small">
                Saved as your current cover letter (a Word file you can download and polish from the
                Profile page later).
              </p>
              <Markdown text={generated.content_text} />
            </div>
          )}
        </>
      )}
      {error && <p className="error-text small">{error}</p>}
    </div>
  );
}

type PendingFile = { file: File; kind: "resume" | "cover_letter"; status?: string };

function OldDocsStep({ existing, onUploaded }: { existing: ResumeSummary[]; onUploaded: () => void }) {
  const [files, setFiles] = useState<PendingFile[]>([]);
  const [busy, setBusy] = useState(false);

  const pick = (list: FileList | null) => {
    if (!list) return;
    setFiles(Array.from(list).map((file) => ({ file, kind: guessKind(file.name) })));
  };

  const uploadAll = async () => {
    setBusy(true);
    // Sequential, so each row's status updates as it goes and one bad file doesn't sink the rest.
    for (let i = 0; i < files.length; i++) {
      if (files[i].status === "✓ uploaded") continue;
      const setStatus = (status: string) =>
        setFiles((prev) => prev.map((f, j) => (j === i ? { ...f, status } : f)));
      setStatus("uploading...");
      try {
        await api.uploadResume(files[i].file, files[i].kind, { isArchived: true });
        setStatus("✓ uploaded");
      } catch (e) {
        setStatus(errorText(e));
      }
    }
    setBusy(false);
    onUploaded();
  };

  return (
    <div className="card">
      <h2>Add older resumes and cover letters</h2>
      <p className="muted small">
        Optional, but it helps: older documents give Claude your career progression and the skills
        that keep showing up. They're weighted below your current ones, so a 2019 cover letter won't
        override what you want today. Select as many files as you like.
      </p>
      {existing.length > 0 && (
        <p className="onboarding-done">
          ✓ {existing.length} older document{existing.length === 1 ? "" : "s"} on file
        </p>
      )}
      <input type="file" accept={ACCEPT} multiple onChange={(e) => pick(e.target.files)} />
      {files.length > 0 && (
        <>
          <table className="onboarding-files">
            <tbody>
              {files.map((f, i) => (
                <tr key={f.file.name + i}>
                  <td>{f.file.name}</td>
                  <td>
                    <select
                      value={f.kind}
                      disabled={busy || f.status === "✓ uploaded"}
                      onChange={(e) =>
                        setFiles((prev) =>
                          prev.map((p, j) =>
                            j === i ? { ...p, kind: e.target.value as PendingFile["kind"] } : p,
                          ),
                        )
                      }
                    >
                      <option value="resume">Resume</option>
                      <option value="cover_letter">Cover letter</option>
                    </select>
                  </td>
                  <td className="muted small">{f.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <button
            className="primary"
            onClick={uploadAll}
            disabled={busy || files.every((f) => f.status === "✓ uploaded")}
          >
            {busy ? <Spinner label="Uploading..." /> : `Upload ${files.length} file${files.length === 1 ? "" : "s"}`}
          </button>
        </>
      )}
    </div>
  );
}

function FinishStep({ onDone }: { onDone: () => void }) {
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<RunProgress | null>(null);
  const started = useRef(false);

  useEffect(() => {
    // The ref keeps StrictMode's dev-only double mount from starting two daily runs.
    if (started.current) return;
    started.current = true;
    (async () => {
      try {
        await api.putSettings({ onboarding_completed: true, onboarding_step: "finish" });
        onDone();
      } catch (e) {
        setError(errorText(e));
        return;
      }
      // The run streams its progress back as it goes.
      try {
        await api.runDailyNow(FIRST_RUN_CAP, setProgress);
        navigate("/", { replace: true });
      } catch (e) {
        setError(errorText(e));
      }
    })();
  }, []);

  return (
    <div className="card onboarding-finish">
      {error ? (
        <>
          <h2>Setup's done, but the first job search hit a problem</h2>
          <p className="error-text small">{error}</p>
          <p className="muted small">
            You can retry any time from Manage Sources ("Run now"). Otherwise it runs on its daily
            schedule.
          </p>
          <button className="primary" onClick={() => navigate("/", { replace: true })}>
            Go to New &amp; Noteworthy
          </button>
        </>
      ) : (
        <>
          <h2>Finding and scoring your first jobs</h2>
          <p className="muted small">
            Checking your sources, then having Claude score your {FIRST_RUN_CAP} most promising
            matches against your profile — the rest are scored in tomorrow's daily run. You'll land
            on New &amp; Noteworthy when it's done.
          </p>
          <Spinner label={runProgressText(progress, "Done — loading your matches...")} />
        </>
      )}
    </div>
  );
}

export default function OnboardingPage({ onDone }: { onDone: () => void }) {
  const navigate = useNavigate();
  const [step, setStep] = useState<OnboardingStep | null>(null);
  const [docs, setDocs] = useState<ResumeSummary[]>([]);
  const [hasProfile, setHasProfile] = useState(false);
  // Rebuild the profile automatically on arriving at that step only when there's none yet or a
  // document was added since it was built -- never just because the user went back to it, which
  // would spend a Claude call and throw away any revisions they made.
  const [rebuildOnProfileStep, setRebuildOnProfileStep] = useState(false);
  const [docsChanged, setDocsChanged] = useState(false);

  const refreshDocs = () => api.listResumes().then(setDocs);
  const docsUpdated = () => {
    setDocsChanged(true);
    refreshDocs();
  };

  useEffect(() => {
    refreshDocs();
    api
      .getOnboardingStatus()
      .then((s) => {
        setHasProfile(s.has_profile);
        setRebuildOnProfileStep(!s.has_profile);
        setStep(s.step === "finish" ? "sources" : s.step);
      })
      .catch(() => setStep("resume"));
  }, []);

  if (step === null) return <Spinner label="Loading..." />;

  const index = STEPS.findIndex((s) => s.key === step);
  const goTo = (next: OnboardingStep) => {
    if (next === "profile") setRebuildOnProfileStep(docsChanged || !hasProfile);
    setStep(next);
    window.scrollTo(0, 0);
    if (next !== "finish") api.putSettings({ onboarding_step: next }).catch(() => undefined);
  };
  const next = () => goTo(STEPS[index + 1].key);
  const back = () => goTo(STEPS[index - 1].key);

  const skipSetup = async () => {
    await api.putSettings({ onboarding_completed: true }).catch(() => undefined);
    onDone();
    navigate("/", { replace: true });
  };

  const currentResume = docs.find((d) => d.kind === "resume" && d.is_template);
  const currentCoverLetter = docs.find((d) => d.kind === "cover_letter" && d.is_template);
  const olderDocs = docs.filter((d) => !d.is_template);

  // Whether the main action of a step is done -- Next is the primary button once it is; until
  // then, only the resume step blocks moving on (everything after it is skippable).
  const stepDone: Record<OnboardingStep, boolean> = {
    resume: !!currentResume,
    looking_for: true,
    cover_letter: !!currentCoverLetter,
    old_docs: olderDocs.length > 0,
    profile: hasProfile,
    sources: true,
    finish: true,
  };

  const nextLabel =
    step === "profile" ? "Looks good" : step === "sources" ? "Finish & find my first jobs" : "Next";

  return (
    <div className="onboarding">
      <div className="onboarding-header">
        <div>
          <h1>Let's get you set up</h1>
          <p className="muted">
            A few steps so the assistant knows who you are and what you're looking for.
          </p>
        </div>
        {step !== "finish" && (
          <button className="link small" onClick={skipSetup}>
            Skip setup
          </button>
        )}
      </div>

      <ol className="stepper">
        {STEPS.map((s, i) => (
          <li key={s.key} className={i < index ? "done" : i === index ? "current" : ""}>
            {/* Earlier steps are links back; forward still goes through Next, which is where the
                required resume is enforced. Locked while the first job search runs. */}
            {i < index && step !== "finish" ? (
              <button
                className="stepper-step"
                onClick={() => goTo(s.key)}
                aria-label={`Go back to ${s.label}`}
              >
                <span className="stepper-dot">✓</span>
                <span className="stepper-label">{s.label}</span>
              </button>
            ) : (
              <>
                <span className="stepper-dot">{i < index ? "✓" : i + 1}</span>
                <span className="stepper-label">{s.label}</span>
              </>
            )}
          </li>
        ))}
      </ol>

      {step === "resume" && <ResumeStep current={currentResume} onUploaded={docsUpdated} />}
      {step === "looking_for" && <DesiredRoleForm />}
      {step === "cover_letter" && <CoverLetterStep current={currentCoverLetter} onChanged={docsUpdated} />}
      {step === "old_docs" && <OldDocsStep existing={olderDocs} onUploaded={docsUpdated} />}
      {step === "profile" && (
        <CandidateProfileCard
          hasDocuments={docs.length > 0}
          autoRebuild={rebuildOnProfileStep}
          title="Your candidate profile"
          description="A summary of your background built from the documents you added. Jobs and company suggestions are matched against it, so make sure it sounds like you. Ask Claude for changes, or edit it directly."
          onProfileChange={(text) => {
            setHasProfile(!!text);
            if (text) setDocsChanged(false);
          }}
        />
      )}
      {step === "sources" && (
        <>
          <p className="muted small">
            Claude is searching the web for organizations and job boards that match your profile.
            Click <strong>Monitor</strong> on any you want checked for new postings every day. You
            can add more later from Manage Sources.
          </p>
          <DiscoverCompaniesSection autoRun />
          <DiscoverJobBoardsSection autoRun />
        </>
      )}
      {step === "finish" && <FinishStep onDone={onDone} />}

      {step !== "finish" && (
        <div className="onboarding-nav">
          {index > 0 ? (
            <button className="ghost" onClick={back}>
              Back
            </button>
          ) : (
            <span />
          )}
          <div className="form-actions" style={{ marginTop: 0 }}>
            {index > 0 && !stepDone[step] && (
              <button className="link small" onClick={next}>
                Skip this step
              </button>
            )}
            <button
              className={stepDone[step] ? "primary" : "ghost"}
              onClick={next}
              disabled={step === "resume" && !stepDone.resume}
            >
              {nextLabel}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
