import { useState } from "react";
import { genreColor } from "../genres";

interface Props {
  book: { title: string; author: string; cover_url: string | null; cover_url_small?: string | null; isbn?: string | null; genre?: string | null };
  size?: "sm" | "md" | "lg";
  onClick?: () => void;
}

/** Cover with a fallback chain: Goodreads large -> Goodreads small -> Open Library by ISBN -> typographic placeholder. */
export function Cover({ book, size = "md", onClick }: Props) {
  const sources = [
    book.cover_url,
    book.cover_url_small,
    book.isbn ? `https://covers.openlibrary.org/b/isbn/${book.isbn}-M.jpg?default=false` : null,
  ].filter((s, i, a): s is string => !!s && a.indexOf(s) === i);
  const [attempt, setAttempt] = useState(0);
  const src = sources[attempt];
  const cls = `cover cover-${size}${onClick ? " clickable" : ""}`;

  if (!src) {
    const c = genreColor(book.genre);
    return (
      <div className={`${cls} cover-placeholder`} onClick={onClick}
           style={{ background: `linear-gradient(160deg, ${c}33 0%, ${c}99 100%)`, borderColor: `${c}66` }}>
        <span className="ph-title">{book.title}</span>
        <span className="ph-author">{book.author}</span>
      </div>
    );
  }
  return (
    <img className={cls} src={src} alt={`${book.title} by ${book.author}`} loading="lazy" onClick={onClick}
         onError={() => setAttempt((a) => a + 1)}
         // Goodreads serves a 1x1 or tiny "nophoto" image for some dead covers; treat as missing.
         onLoad={(e) => { if ((e.target as HTMLImageElement).naturalWidth < 20) setAttempt((a) => a + 1); }} />
  );
}
