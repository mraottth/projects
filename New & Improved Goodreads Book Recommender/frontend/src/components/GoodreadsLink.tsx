/** Small "g" badge linking to the book's Goodreads page (opens in a new tab). */
export function GoodreadsLink({ url, title }: { url: string; title: string }) {
  return (
    <a className="gr-link" href={url} target="_blank" rel="noreferrer"
       aria-label={`${title} on Goodreads (opens in a new tab)`} title="View on Goodreads">
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
        <circle cx="12" cy="12" r="12" fill="#f4f1ea" />
        <text x="12" y="16.5" textAnchor="middle" fontFamily="Georgia, 'Times New Roman', serif" fontSize="15"
              fontWeight="700" fill="#382110">g</text>
      </svg>
    </a>
  );
}
