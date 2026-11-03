import { useEffect, useState } from "react";
import { api, type RescoreScope, type RescoreStatus } from "../api";
import { Spinner } from "./Spinner";

const LABELS: Record<RescoreScope, string> = { my_jobs: "My Jobs", all_jobs: "All Jobs" };

function money(v: number): string {
  return v < 0.995 ? `${Math.max(1, Math.round(v * 100))}¢` : `$${v.toFixed(2)}`;
}

function waitText(iso: string): string {
  const hours = (new Date(iso).getTime() - Date.now()) / 3_600_000;
  if (hours >= 1) {
    const n = Math.round(hours);
    return `${n} hour${n === 1 ? "" : "s"}`;
  }
  const n = Math.max(1, Math.round(hours * 60));
  return `${n} minute${n === 1 ? "" : "s"}`;
}

/**
 * "Rescore": scores every job on this list again with your current profile, after showing about
 * what it costs (core/rescore.py). Once a day for everyone but the admin.
 */
export function RescoreButton({ scope, onDone }: { scope: RescoreScope; onDone?: () => void }) {
  const [status, setStatus] = useState<RescoreStatus | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [progress, setProgress] = useState<[number, number] | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () =>
    api
      .getRescoreStatus(scope)
      .then((s) => {
        setStatus(s);
        if (s.running) run(); // follow a rescore already going
      })
      .catch(() => setStatus(null));

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope]);

  async function run() {
    setConfirming(false);
    setError(null);
    setMessage(null);
    setProgress([0, 0]);
    try {
      const result = await api.rescore(scope, (done, total) => setProgress([done, total]));
      const parts = [`Rescored ${result.scored} job${result.scored === 1 ? "" : "s"}.`];
      if (result.failed) parts.push(`${result.failed} couldn't be scored.`);
      if (result.stopped) parts.push(`Stopped early: ${result.stopped}`);
      setMessage(parts.join(" "));
      onDone?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setProgress(null);
      api.getRescoreStatus(scope).then(setStatus).catch(() => undefined);
    }
  }

  if (!status) return null;
  if (status.count === 0) return <p className="muted small rescore">No jobs on {LABELS[scope]} yet.</p>;
  const overBudget = status.budget_remaining_usd !== null && status.estimate_usd > status.budget_remaining_usd;

  return (
    <div className="rescore">
      {progress ? (
        <Spinner label={progress[1] ? `Rescoring ${progress[0]} of ${progress[1]} jobs…` : "Starting…"} />
      ) : confirming ? (
        <div className="rescore-confirm">
          <span className="small">
            Score all {status.count} jobs on {LABELS[scope]} with your current profile, for about{" "}
            <strong>{money(status.estimate_usd)}</strong>?
            {overBudget && (
              <>
                {" "}
                You have {money(status.budget_remaining_usd!)} of this month's budget left, so it will stop when that
                runs out.
              </>
            )}
            {status.budget_remaining_usd !== null && " You can do this once a day."}
          </span>
          <button className="primary small" onClick={run}>
            Rescore
          </button>
          <button className="link small" onClick={() => setConfirming(false)}>
            Cancel
          </button>
        </div>
      ) : (
        <button
          className="ghost small"
          disabled={!!status.available_at}
          title={
            status.available_at
              ? `You can rescore ${LABELS[scope]} again in ${waitText(status.available_at)}.`
              : "Score every job on this list again with your current profile"
          }
          onClick={() => setConfirming(true)}
        >
          Rescore {LABELS[scope]}: {status.count} jobs (about {money(status.estimate_usd)})
        </button>
      )}
      {!progress && status.available_at && !confirming && (
        <span className="muted small"> Available again in {waitText(status.available_at)}.</span>
      )}
      {message && <span className="muted small"> {message}</span>}
      {error && <span className="error-text small"> {error}</span>}
    </div>
  );
}
