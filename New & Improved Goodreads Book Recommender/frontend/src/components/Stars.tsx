import { useState } from "react";

interface Props {
  value: number;            // 0 = unrated
  onChange?: (v: number) => void;
  size?: "sm" | "md";
  label?: string;
}

/** Interactive 1-5 star rating; click the current value again to clear. */
export function Stars({ value, onChange, size = "md", label = "Rate" }: Props) {
  const [hover, setHover] = useState(0);
  const shown = hover || value;
  return (
    <div className={`stars stars-${size}`} role="radiogroup" aria-label={label} onMouseLeave={() => setHover(0)}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button key={n} type="button" role="radio" aria-checked={value === n} aria-label={`${n} star${n > 1 ? "s" : ""}`}
                className={n <= shown ? "on" : ""} disabled={!onChange}
                onMouseEnter={() => onChange && setHover(n)}
                onClick={(e) => { e.stopPropagation(); onChange?.(value === n ? 0 : n); }}>
          ★
        </button>
      ))}
    </div>
  );
}

export function AvgRating({ value, count }: { value: number; count?: number }) {
  return (
    <span className="avg">
      <span className="avg-star">★</span> {value.toFixed(2)}
      {count != null && <span className="muted"> · {formatCount(count)}</span>}
    </span>
  );
}

/** Read-only fractional star display (e.g. 4.3 = four full stars and 30% of the fifth). */
export function StarBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, (value / 5) * 100));
  return (
    <span className="star-bar" role="img" aria-label={`${value.toFixed(1)} out of 5 stars`}>
      <span className="star-bar-bg">★★★★★</span>
      <span className="star-bar-fg" style={{ width: `${pct}%` }}>★★★★★</span>
    </span>
  );
}

export function formatCount(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return String(n);
}
