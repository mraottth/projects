import { ReactNode } from "react";
import { Link, Location, useLocation } from "react-router-dom";

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
