export function Spinner({ label }: { label?: string }) {
  return (
    <span className="spinner-row">
      <span className="spinner" aria-hidden="true" />
      {label && <span className="muted small">{label}</span>}
    </span>
  );
}
