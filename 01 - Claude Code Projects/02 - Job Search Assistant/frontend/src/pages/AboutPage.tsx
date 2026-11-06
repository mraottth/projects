import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { request } from "../api";
import { type Milestone, MilestoneTimeline } from "../components/MilestoneTimeline";
import { Spinner } from "../components/Spinner";
import { useMe } from "../me";
import { PrivacyNote } from "./AccessPage";

/**
 * About (asked 2026-11-04): what this project is, how it works, a short usage guide, and the
 * Product Timeline's milestones (moved here from Settings). The full Timeline, with every prompt,
 * commit and decision, stays at /changelog; a milestone's prompt or decision opens it there.
 */

interface TimelineData {
  milestones: Milestone[];
  categories: Record<string, string>;
  prompts: { id: string; kind: string; text: string; reply: string | null }[];
  commits: unknown[];
  decisions: unknown[];
}

const GUIDE: { title: string; to: string; link: string; body: string }[] = [
  {
    title: "Tell it what you're looking for",
    to: "/profile",
    link: "Profile",
    body:
      "Upload your current resume (a .docx works best, so tailored versions keep your formatting), plus a recent " +
      "cover letter and any older ones. Then fill in “What are you looking for?”: titles, sectors, locations, a " +
      "minimum salary and a few sentences in your own words. Those words count the most. Location and salary " +
      "can each be a dealbreaker that hides jobs clearly outside them.",
  },
  {
    title: "Choose where to look",
    to: "/sources",
    link: "Manage Sources",
    body:
      "Add organizations you'd like to work for and job boards in your field, by link or from suggestions " +
      "Claude finds for you. Every source is checked once a day.",
  },
  {
    title: "Check what's new",
    to: "/",
    link: "New & Noteworthy",
    body:
      "Each day's best new jobs, scored for how interested you'd likely be (Interest) and how strong a " +
      "candidate you'd be (Fit), with the reasons highlighted. Mark one Interested to move it to My Jobs, or " +
      "Not interested to stop seeing it; either way it leaves this list, and your marks teach it what you like. " +
      "Refresh checks your sources again now.",
  },
  {
    title: "Browse everything",
    to: "/all-jobs",
    link: "All Jobs",
    body:
      "Every job from your sources, scored or not, with filters and sorting. Click an organization or location " +
      "chip on a job to filter by it.",
  },
  {
    title: "Track your applications",
    to: "/tracker",
    link: "My Jobs",
    body:
      "Your tracker: status from drafting through offer, the date you applied, and tailored resumes and cover " +
      "letters for each job (downloaded as Word files with the changes tracked, so you review every edit). " +
      "“Ask Jobby what's working” looks at which kinds of jobs are getting you interviews.",
  },
  {
    title: "Prepare with Jobby",
    to: "/jobby",
    link: "Jobby",
    body:
      "An interview-prep chat that knows the posting, your documents and your profile: mock interviews, likely " +
      "questions, how to frame your background, and research on the organization.",
  },
  {
    title: "Settings",
    to: "/settings",
    link: "Settings",
    body:
      "Your daily email, this month's usage, rescoring your lists after you change your profile, backups of " +
      "your data, and starting over or deleting your account.",
  },
];

export default function AboutPage() {
  const me = useMe();
  const usage = me?.usage;

  return (
    <div className="about">
      <h1>About</h1>
      <p className="lead-left">
        Job Search Assistant finds job postings from the organizations and boards you choose, has Claude rate
        each one for you, tailors your resume and cover letter to the ones you pursue, and tracks your
        applications. It's a small personal project, shared with a few people its owner invites.
      </p>

      <div className="card">
        <h2>How it works</h2>
        <ol className="about-steps">
          <li>
            <strong>Every day at 2pm Eastern</strong>, it reads every source you follow and saves any new postings.
          </li>
          <li>
            <strong>A quick free pass</strong> puts the most promising first (matching titles and sectors) and
            skips “join our talent community” listings and anything your dealbreakers rule out.
          </li>
          <li>
            <strong>Claude scores the best of them</strong>
            {usage ? `, up to ${usage.daily_score_cap} a day,` : ""} against your profile, what you're looking for
            and what you've marked before: Interest and Fit, High, Med or Low, with the reasons.
          </li>
          <li>
            <strong>They land in New &amp; Noteworthy</strong>, and the good ones (High or Med interest with a
            High or Med fit) go out in a daily email.
          </li>
          <li>
            <strong>Your marks and outcomes feed back in</strong>: once enough build up, it rebuilds what it knows
            about your preferences, and an interview counts for more than a click.
          </li>
        </ol>
      </div>

      <div className="card">
        <h2>Getting started</h2>
        <ol className="about-guide">
          {GUIDE.map((step) => (
            <li key={step.title}>
              <strong>{step.title}</strong> <Link to={step.to}>({step.link})</Link>
              <p className="muted small">{step.body}</p>
            </li>
          ))}
        </ol>
      </div>

      <div className="card">
        <h2>Costs and limits</h2>
        <ul className="about-list">
          <li>
            Claude costs money for every score, tailored document and Jobby reply, so each person has a daily
            scoring limit{usage ? ` (yours is ${usage.daily_score_cap} jobs)` : ""} and a monthly budget
            {usage && !usage.exempt ? ` (yours is $${usage.monthly_budget_usd.toFixed(2)})` : ""}. A score costs
            about a cent.
          </li>
          <li>
            If you don't open the app for a week, your daily scoring pauses (you'll get one email saying so) and
            picks up again when you come back.
          </li>
          <li>Settings shows what you've used this month.</li>
        </ul>
        <PrivacyNote />
      </div>

      <div className="card">
        <h2>Built with</h2>
        <p className="muted small" style={{ margin: 0 }}>
          Built in conversation with Claude Code. Claude (Sonnet) does the scoring, writing and chat; the app is
          Python (FastAPI) with a SQLite database kept safe by Litestream, a React front end, and Google Cloud
          Run behind Google sign-in. The timeline below shows how it came together.
        </p>
      </div>

      <TimelineSection />
    </div>
  );
}

/** The Product Timeline's milestones (moved here from Settings), with a link to the full Timeline. */
function TimelineSection() {
  const [data, setData] = useState<TimelineData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    request<TimelineData>("/api/changelog")
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="card about-timeline">
      <div className="about-timeline-head">
        <h2>Product Timeline</h2>
        <Link className="ghost small" to="/changelog">
          Open the full Timeline
        </Link>
      </div>
      <p className="muted small">
        How this app was built: the big milestones below, and on the full Timeline every prompt that shaped it,
        Claude's replies, the commits they produced and the decisions along the way.
        {data && ` ${data.prompts.length} prompts, ${data.commits.length} commits and ${data.decisions.length} decisions so far.`}
      </p>
      {error && <p className="muted small">The Timeline isn't available right now ({error}).</p>}
      {!data && !error && <Spinner label="Loading the Timeline…" />}
      {data && data.milestones.length > 0 && (
        <MilestoneTimeline
          milestones={data.milestones}
          prompts={new Map(data.prompts.map((p) => [p.id, p]))}
          categories={data.categories}
          onShowPrompt={(id) => navigate(`/changelog?prompt=${encodeURIComponent(id)}`)}
          onShowDecision={(id) => navigate(`/changelog?decision=${encodeURIComponent(id)}`)}
        />
      )}
    </div>
  );
}
