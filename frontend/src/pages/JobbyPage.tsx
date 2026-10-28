import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, Application, APPLICATION_STATUS_LABELS, JobbyMessage, JobbyThread } from "../api";
import { JobbyLogo } from "../components/JobbyLogo";
import { Markdown } from "../components/Markdown";
import { Spinner } from "../components/Spinner";

// Jobby, the interview-prep assistant. The job being prepped for lives in the URL (?job=42, or
// none for general help) so My Jobs and the job page can link straight to a job's conversation.
// Each job's conversation is saved server-side (core/jobby.py), so it survives reloads.

const STARTERS: { label: string; blurb: string; text: string }[] = [
  {
    label: "Mock interview",
    blurb: "Practice with Jobby as your interviewer",
    text: "Let's do a mock interview for this role. You be the interviewer: ask me one question at a time, wait for my answer, and give me quick feedback before the next one.",
  },
  {
    label: "Likely questions",
    blurb: "What they'll probably ask, and how to answer",
    text: "What questions is the interviewer most likely to ask me for this role? Group them, flag the ones I should prepare most carefully given my background, and then offer to work through answers with me.",
  },
  {
    label: "Frame my background",
    blurb: "Your pitch and strongest stories for this role",
    text: "Help me frame my background for this role: draft a 60–90 second \"tell me about yourself\", map my strongest stories to what the posting asks for, and suggest honest ways to handle any gaps.",
  },
  {
    label: "Questions to ask them",
    blurb: "Thoughtful questions for each interviewer",
    text: "Suggest thoughtful questions I could ask my interviewers for this role — for the hiring manager, future teammates, and leadership — informed by what you can find about the company right now.",
  },
  {
    label: "Prep & research plan",
    blurb: "A checklist of what to do beforehand",
    text: "Give me a prep checklist for this interview: what to research about the company (look up anything recent and relevant), skills or examples to brush up on, and logistics to sort out beforehand.",
  },
];

// Applications still in play come first in the picker; closed-out ones last.
const CLOSED = new Set(["rejected", "withdrawn"]);

function JobPicker({
  apps,
  selected,
  onSelect,
  disabled,
}: {
  apps: Application[];
  selected: number | null;
  onSelect: (jobId: number | null) => void;
  disabled: boolean;
}) {
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  const shown = apps
    .filter((a) => !q || `${a.title} ${a.company}`.toLowerCase().includes(q))
    .sort((a, b) => Number(CLOSED.has(a.status)) - Number(CLOSED.has(b.status)));

  return (
    <div className="card jobby-picker">
      <span className="flabel">Which job are you prepping for?</span>
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search My Jobs..."
        style={{ width: "100%", marginBottom: "8px" }}
      />
      <ul>
        <li>
          <button
            className={`jobby-pick${selected === null ? " on" : ""}`}
            onClick={() => onSelect(null)}
            disabled={disabled}
          >
            <span className="jobby-pick-title">No specific job</span>
            <span className="muted small">General interview help</span>
          </button>
        </li>
        {shown.map((a) => (
          <li key={a.job_id}>
            <button
              className={`jobby-pick${selected === a.job_id ? " on" : ""}`}
              onClick={() => onSelect(a.job_id)}
              disabled={disabled}
            >
              <span className="jobby-pick-title">{a.title}</span>
              <span className="muted small">
                {a.company} ·{" "}
                <span className={`pill pill-${a.status}`}>{APPLICATION_STATUS_LABELS[a.status]}</span>
              </span>
            </button>
          </li>
        ))}
        {apps.length === 0 && (
          <li className="muted small" style={{ padding: "6px 4px" }}>
            Jobs you track in <Link to="/tracker">My Jobs</Link> show up here.
          </li>
        )}
        {apps.length > 0 && shown.length === 0 && (
          <li className="muted small" style={{ padding: "6px 4px" }}>
            No jobs match "{query}".
          </li>
        )}
      </ul>
    </div>
  );
}

function Bubble({ message }: { message: Pick<JobbyMessage, "role" | "content"> }) {
  if (message.role === "user") {
    return <div className="jobby-msg user">{message.content}</div>;
  }
  return (
    <div className="jobby-msg assistant">
      <JobbyLogo size={28} />
      <div className="jobby-msg-body">
        <Markdown text={message.content} />
      </div>
    </div>
  );
}

export default function JobbyPage() {
  const [params, setParams] = useSearchParams();
  const jobParam = params.get("job");
  const jobId = jobParam && /^\d+$/.test(jobParam) ? Number(jobParam) : null;

  const [apps, setApps] = useState<Application[]>([]);
  const [thread, setThread] = useState<JobbyThread | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState<{ user: string; reply: string; status: string | null } | null>(
    null,
  );
  const [sendError, setSendError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.listApplications().then(setApps).catch(() => undefined);
  }, []);

  useEffect(() => {
    setThread(null);
    setLoadError(null);
    setSendError(null);
    api
      .getJobbyThread(jobId)
      .then(setThread)
      .catch((e) => setLoadError(e instanceof Error ? e.message : String(e)));
  }, [jobId]);

  // Keep the newest message in view as it streams in.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [thread?.messages.length, pending?.reply, pending?.status]);

  const selectJob = (id: number | null) => setParams(id === null ? {} : { job: String(id) });

  const send = async (text: string) => {
    const content = text.trim();
    if (!content || pending) return;
    setDraft("");
    setSendError(null);
    setPending({ user: content, reply: "", status: null });
    try {
      await api.streamJobbyMessage(jobId, content, (event) => {
        if (event.type === "delta") {
          setPending((p) => (p ? { ...p, reply: p.reply + event.text, status: null } : p));
        } else if (event.type === "status") {
          setPending((p) => (p ? { ...p, status: event.text } : p));
        } else if (event.type === "done") {
          setThread((t) => (t ? { ...t, messages: [...t.messages, event.user, event.assistant] } : t));
          setPending(null);
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      });
    } catch (e) {
      // Nothing from a failed turn is saved, so put their message back to retry.
      setSendError(e instanceof Error ? e.message : String(e));
      setDraft(content);
    } finally {
      setPending(null);
    }
  };

  const startOver = async () => {
    await api.clearJobbyThread(jobId);
    setThread((t) => (t ? { ...t, messages: [] } : t));
  };

  const messages = thread?.messages ?? [];
  const empty = messages.length === 0 && !pending;
  const job = thread?.job;

  return (
    <div>
      <div className="jobby-header">
        <JobbyLogo size={56} />
        <div>
          <h1>Jobby</h1>
          <p className="muted" style={{ margin: 0 }}>
            Your interview-prep sidekick. Jobby knows the job, your profile, and your documents.
          </p>
        </div>
      </div>

      <div className="jobby-layout">
        <aside>
          <JobPicker apps={apps} selected={jobId} onSelect={selectJob} disabled={!!pending} />
        </aside>

        <section className="card jobby-chat">
          <div className="jobby-chat-head">
            <div>
              {job ? (
                <>
                  <div className="card-title">
                    <Link to={`/jobs/${job.id}`}>{job.title}</Link>
                  </div>
                  <div className="muted small">{job.company}</div>
                </>
              ) : (
                <>
                  <div className="card-title">General interview help</div>
                  <div className="muted small">Pick a job on the left for prep tailored to it.</div>
                </>
              )}
            </div>
            {messages.length > 0 && (
              <button className="ghost small" onClick={startOver} disabled={!!pending}>
                Start over
              </button>
            )}
          </div>

          <div className="jobby-messages">
            {loadError && <p className="error-text small">{loadError}</p>}
            {!thread && !loadError && <Spinner label="Loading..." />}
            {thread && empty && (
              <div className="jobby-welcome">
                <JobbyLogo variant="full" size={180} title="Jobby at a laptop" />
                <Bubble
                  message={{
                    role: "assistant",
                    content: job
                      ? `Hi! I'm **Jobby**. Let's get you ready for **${job.title}** at **${job.company}**. Pick a place to start, or ask me anything.`
                      : "Hi! I'm **Jobby**, your interview-prep sidekick. Pick a job on the left for prep tailored to it, or start with something general.",
                  }}
                />
                <div className="jobby-starters">
                  {STARTERS.map((s) => (
                    <button key={s.label} className="jobby-starter" onClick={() => send(s.text)}>
                      <span className="jobby-starter-label">{s.label}</span>
                      <span className="muted small">{s.blurb}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m) => (
              <Bubble key={m.id} message={m} />
            ))}
            {pending && (
              <>
                <Bubble message={{ role: "user", content: pending.user }} />
                <div className="jobby-msg assistant">
                  <JobbyLogo size={28} />
                  <div className="jobby-msg-body">
                    {pending.reply && <Markdown text={pending.reply} />}
                    {(pending.status || !pending.reply) && (
                      <Spinner label={pending.status ?? "Jobby is thinking..."} />
                    )}
                  </div>
                </div>
              </>
            )}
            <div ref={endRef} />
          </div>

          {!empty && !pending && (
            <div className="jobby-try">
              <span className="muted small">Try:</span>
              {STARTERS.map((s) => (
                <button key={s.label} className="chip-toggle" onClick={() => send(s.text)}>
                  {s.label}
                </button>
              ))}
            </div>
          )}

          {sendError && <p className="error-text small">{sendError}</p>}
          <div className="jobby-input">
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(draft);
                }
              }}
              rows={2}
              placeholder={pending ? "Jobby is replying..." : "Ask Jobby anything — Enter to send, Shift+Enter for a new line"}
              disabled={!!pending || !thread}
            />
            <button className="primary" onClick={() => send(draft)} disabled={!!pending || !draft.trim()}>
              Send
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
