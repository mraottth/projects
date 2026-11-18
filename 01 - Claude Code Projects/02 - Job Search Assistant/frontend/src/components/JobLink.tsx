import { MouseEvent, ReactNode, useCallback } from "react";
import { Link, Location, useLocation, useNavigate } from "react-router-dom";

/**
 * A link to a job that opens it in a card over the current page (asked 2026-11-10), not a new
 * page: the address still becomes /jobs/<id>, carrying the page it was opened from as
 * `background`, which App keeps showing underneath (filters, sort and scroll intact). Back, Esc,
 * × or a click outside closes it. Opened directly (an email link, a reload), /jobs/<id> is the
 * full page.
 */
export function JobLink({ jobId, children, className }: { jobId: number; children: ReactNode; className?: string }) {
  const location = useLocation();
  const state = location.state as { background?: Location } | null;
  return (
    <Link to={`/jobs/${jobId}`} state={{ background: state?.background ?? location }} className={className}>
      {children}
    </Link>
  );
}

/**
 * A click handler that opens a job's overlay from anywhere on its card (asked 2026-11-16), the same
 * as its JobLink. Clicks on the card's own controls (links, buttons, chips, inputs) and finishing a
 * text selection are left alone.
 */
export function useOpenJobOnClick(jobId: number) {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as { background?: Location } | null;
  return useCallback(
    (e: MouseEvent<HTMLElement>) => {
      const target = e.target as HTMLElement;
      // A dialog opened from the card (e.g. Not Interested's "why not") sits inside it in React's tree.
      if (target.closest("a, button, input, select, textarea, label, summary, details, dialog")) return;
      if (window.getSelection()?.toString()) return;
      navigate(`/jobs/${jobId}`, { state: { background: state?.background ?? location } });
    },
    [jobId, location, navigate, state],
  );
}
