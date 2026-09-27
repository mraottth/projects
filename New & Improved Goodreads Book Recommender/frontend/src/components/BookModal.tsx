import { useEffect, useState } from "react";
import { api, type BookDetail } from "../api";
import { useShelf } from "../store";
import { Cover } from "./Cover";
import { GenreChip, Tag } from "./GenreChip";
import { AvgRating, Stars } from "./Stars";

export function BookModal({ id, onClose, onOpen }: { id: number; onClose: () => void; onOpen: (id: number) => void }) {
  const [book, setBook] = useState<BookDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const shelf = useShelf();

  useEffect(() => {
    setBook(null);
    setError(null);
    api.book(id).then(setBook).catch((e) => setError(e.message));
  }, [id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close">×</button>
        {error && <p className="error">{error}</p>}
        {!book && !error && <div className="spinner" />}
        {book && (
          <>
            <div className="modal-head">
              <Cover book={book} size="lg" />
              <div className="modal-info">
                <h2>{book.title}</h2>
                <div className="card-author">{book.author}{book.year ? ` · ${book.year}` : ""}</div>
                {book.series && <div className="muted">{book.series}{book.series_pos ? ` #${book.series_pos}` : ""}</div>}
                <AvgRating value={book.avg_rating} count={book.ratings_count} />
                <div className="card-tags">
                  {book.genre && <GenreChip genre={book.genre} />}
                  {book.tags.filter((t) => t !== book.genre).map((t) => <Tag key={t}>{t}</Tag>)}
                </div>
                <div className="modal-rate">
                  <span className="muted small">Your rating</span>
                  <Stars value={shelf.ratings[book.id]?.rating ?? 0}
                         onChange={(v) => (v ? shelf.rate(book, v) : shelf.unrate(book.id))} />
                </div>
                {book.description && <p className="description">{book.description}{book.description.length >= 1200 ? "…" : ""}</p>}
                <a href={book.url} target="_blank" rel="noreferrer">View on Goodreads ↗</a>
              </div>
            </div>
            {book.similar.length > 0 && (
              <>
                <h3>Readers who liked this also liked</h3>
                <div className="cover-strip">
                  {book.similar.map((s) => (
                    <div key={s.id} className="strip-item" title={`${s.title} — ${s.author}`}>
                      <Cover book={s} size="sm" onClick={() => onOpen(s.id)} />
                      <span className="strip-title">{s.title}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
