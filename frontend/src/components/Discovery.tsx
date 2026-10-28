import { useEffect, useRef, useState } from "react";
import { api, CompanySuggestion, DiscoveryRequest, JobBoardSuggestion } from "../api";
import { detectSourceFromUrl } from "../sourceUrl";
import { Spinner } from "./Spinner";

// Shared by the Manage Sources page (run on demand via the button) and onboarding's "Sources to
// follow" step (autoRun: searching starts as soon as the step opens).
//
// Each search can be steered two ways: free text in the "Steer this search" box ("mission-driven
// tech startups"), and "More like this" -- on a suggestion here, or (via `seed`) on a source the
// user already tracks on Manage Sources. Both go to the backend as DiscoveryRequest.

const SEARCHING_LABEL = "Searching the web — usually 1–2 minutes...";
const NO_RESULTS =
  "No new suggestions this time — try again, steer the search differently, or add more detail under \"What are you looking for?\"";

// Set by the parent to start a "find more like this" search. `nonce` changes on every request,
// so asking twice for the same source still re-runs.
export interface DiscoverySeed {
  like: string;
  label: string;
  nonce: number;
}

function errorText(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

// Runs a discovery call, on mount too when autoRun is set. `results` stays null until a search
// finishes, so "searched and found nothing" can be told apart from "hasn't searched yet".
function useDiscovery<T>(fetch: (req: DiscoveryRequest) => Promise<T[]>, autoRun: boolean) {
  const [results, setResults] = useState<T[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [guidance, setGuidance] = useState("");
  // What the current results are "like", shown as a removable chip; null for an unseeded search.
  const [like, setLike] = useState<{ like: string; label: string } | null>(null);

  const run = async (seed: { like: string; label: string } | null = null) => {
    setLike(seed);
    setBusy(true);
    setError(null);
    try {
      setResults(
        await fetch({ guidance: guidance.trim() || undefined, like: seed?.like || undefined }),
      );
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  // The ref keeps StrictMode's dev-only double mount from firing two (paid) Claude calls.
  const autoRan = useRef(false);
  useEffect(() => {
    if (autoRun && !autoRan.current) {
      autoRan.current = true;
      run();
    }
  }, []);

  return { results, busy, error, run, guidance, setGuidance, like, setLike };
}

// A company suggestion that came with its careers URL is added directly, like a board; only one
// without falls back to resolving the name to a job board (slower, and it can come up empty).
function monitorCompany(s: CompanySuggestion) {
  const detected = s.careers_url ? detectSourceFromUrl(s.careers_url) : null;
  if (detected) {
    return api.createSource({
      kind: detected.kind,
      name: s.company_name,
      company_name: s.company_name,
      config: detected.config,
    });
  }
  return api.resolveSource(s.company_name).catch((e) => {
    throw new Error(
      `${errorText(e)} — you can add it by URL under Manage Sources → "Add a source".`,
    );
  });
}

// Per-suggestion "Monitor" state, keyed by name/URL: working, added, or an error message.
function useApprovals(onAdded?: () => void) {
  const [status, setStatus] = useState<Record<string, string>>({});
  const approve = async (key: string, action: () => Promise<unknown>) => {
    setStatus((s) => ({ ...s, [key]: "Adding..." }));
    try {
      await action();
      setStatus((s) => ({ ...s, [key]: "✓ Monitoring" }));
      onAdded?.();
    } catch (e) {
      setStatus((s) => ({ ...s, [key]: errorText(e) }));
    }
  };
  return { status, approve };
}

function SuggestionActions({
  status,
  onMonitor,
  onMoreLikeThis,
  busy,
}: {
  status?: string;
  onMonitor: () => void;
  onMoreLikeThis: () => void;
  busy: boolean;
}) {
  return (
    <div className="form-actions" style={{ marginTop: 0 }}>
      {status ? (
        <span className="muted small">{status}</span>
      ) : (
        <button className="ghost small" onClick={onMonitor}>
          Monitor
        </button>
      )}
      <button className="link small" onClick={onMoreLikeThis} disabled={busy}>
        More like this
      </button>
    </div>
  );
}

type DiscoveryState = ReturnType<typeof useDiscovery<unknown>>;

// The steering box, the search button, and the "like X" chip -- shared by both sections.
function DiscoveryControls({
  noun,
  placeholder,
  d,
}: {
  noun: string;
  placeholder: string;
  d: DiscoveryState;
}) {
  return (
    <>
      <div className="fsection" style={{ marginBottom: "10px" }}>
        <span className="flabel">Steer this search (optional)</span>
        <input
          type="text"
          value={d.guidance}
          onChange={(e) => d.setGuidance(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !d.busy) d.run(d.like);
          }}
          placeholder={placeholder}
          maxLength={500}
          style={{ width: "100%" }}
        />
      </div>
      {d.like && (
        <p className="small discovery-like">
          Finding {noun} like <strong>{d.like.label}</strong>{" "}
          <button
            className="link small"
            onClick={() => d.setLike(null)}
            disabled={d.busy}
            aria-label="Stop finding similar ones"
          >
            × clear
          </button>
        </p>
      )}
      <button className="ghost" onClick={() => d.run(d.like)} disabled={d.busy}>
        {d.busy ? <Spinner label={SEARCHING_LABEL} /> : `Suggest ${noun}`}
      </button>
      {d.error && <p className="error-text small">{d.error}</p>}
      {!d.busy && d.results?.length === 0 && <p className="empty-state small">{NO_RESULTS}</p>}
    </>
  );
}

// Starts a seeded search whenever the parent sends a new seed.
function useSeed(seed: DiscoverySeed | null | undefined, run: (s: { like: string; label: string }) => void) {
  const lastNonce = useRef<number | null>(null);
  useEffect(() => {
    if (seed && seed.nonce !== lastNonce.current) {
      lastNonce.current = seed.nonce;
      run({ like: seed.like, label: seed.label });
    }
  }, [seed]);
}

export function DiscoverCompaniesSection({
  autoRun = false,
  seed,
  onAdded,
}: {
  autoRun?: boolean;
  seed?: DiscoverySeed | null;
  // Called after a suggestion is Monitored, so a parent listing sources can refresh.
  onAdded?: () => void;
}) {
  const d = useDiscovery<CompanySuggestion>(api.suggestCompanies, autoRun);
  const { status, approve } = useApprovals(onAdded);
  useSeed(seed, d.run);

  return (
    <div className="card" id="discover-organizations">
      <h2>Discover organizations</h2>
      <p className="muted small">
        Uses your candidate profile and what you're looking for, plus a web search, to suggest
        organizations worth monitoring. Monitoring one checks its job board for new postings every
        day. Nothing is added until you click Monitor.
      </p>
      <DiscoveryControls
        noun="organizations"
        placeholder="e.g. mission-driven tech startups, or public-health nonprofits in New York"
        d={d}
      />

      {d.results?.map((s) => (
        <div key={s.company_name} className="card-section">
          <div className="card-title" style={{ fontSize: "0.95rem" }}>
            {s.company_name}
          </div>
          <p className="muted small">{s.reason}</p>
          <SuggestionActions
            status={status[s.company_name]}
            onMonitor={() => approve(s.company_name, () => monitorCompany(s))}
            onMoreLikeThis={() =>
              d.run({
                like: s.careers_url ? `${s.company_name} (${s.careers_url})` : s.company_name,
                label: s.company_name,
              })
            }
            busy={d.busy}
          />
        </div>
      ))}
    </div>
  );
}

export function DiscoverJobBoardsSection({
  autoRun = false,
  seed,
  onAdded,
}: {
  autoRun?: boolean;
  seed?: DiscoverySeed | null;
  // Called after a suggestion is Monitored, so a parent listing sources can refresh.
  onAdded?: () => void;
}) {
  const d = useDiscovery<JobBoardSuggestion>(api.suggestJobBoards, autoRun);
  const { status, approve } = useApprovals(onAdded);
  useSeed(seed, d.run);

  return (
    <div className="card" id="discover-job-boards">
      <h2>Discover job boards</h2>
      <p className="muted small">
        Suggests multi-employer boards worth monitoring — broader reach than tracking one
        organization at a time. Monitoring one checks it for new postings every day. Nothing is
        added until you click Monitor.
      </p>
      <DiscoveryControls
        noun="job boards"
        placeholder="e.g. boards for remote climate-tech jobs"
        d={d}
      />

      {d.results?.map((s) => (
        <div key={s.url} className="card-section">
          <div className="card-title" style={{ fontSize: "0.95rem" }}>
            {s.name}
          </div>
          <p className="muted small">
            <code className="path">{s.url}</code>
          </p>
          <p className="muted small">{s.reason}</p>
          <SuggestionActions
            status={status[s.url]}
            onMonitor={() =>
              approve(s.url, () =>
                api.createSource({ kind: "custom_board", name: s.name, config: { url: s.url } }),
              )
            }
            onMoreLikeThis={() => d.run({ like: `${s.name} (${s.url})`, label: s.name })}
            busy={d.busy}
          />
        </div>
      ))}
    </div>
  );
}
