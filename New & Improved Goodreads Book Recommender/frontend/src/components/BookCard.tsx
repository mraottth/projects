import type { Book, ReaderBook, RecBook } from "../api";
import { useShelf } from "../store";
import { Cover } from "./Cover";
import { GenreChip, Tag } from "./GenreChip";
import { AvgRating, Stars } from "./Stars";

type AnyBook = Book & Partial<RecBook> & Partial<ReaderBook>;

interface Props {
  book: AnyBook;
  onOpen: (id: number) => void;
  rank?: number;
}

export function BookCard({ book, onOpen, rank }: Props) {
  const shelf = useShelf();
  const myRating = shelf.ratings[book.id]?.rating ?? 0;
  const seriesLabel = book.series ? `${book.series}${book.series_pos ? ` #${book.series_pos}` : ""}` : null;

  return (
    <article className="card">
      <div className="card-cover">
        <Cover book={book} onClick={() => onOpen(book.id)} />
        {rank != null && <span className="rank">{rank}</span>}
        {book.next_in_series && <span className="badge badge-series">Next in series</span>}
      </div>
      <div className="card-body">
        <button type="button" className="card-title" onClick={() => onOpen(book.id)}>{book.title}</button>
        <div className="card-author">
          {book.author}{book.year ? <span className="muted"> · {book.year}</span> : null}
        </div>
        {seriesLabel && <div className="muted small">{seriesLabel}</div>}
        <div className="card-stats">
          <AvgRating value={book.avg_rating} count={book.ratings_count} />
          {book.pct_read != null && <span className="stat-pill">{book.pct_read}% of similar readers read it</span>}
          {book.neighbor_avg != null && (
            <span className="stat-pill">similar readers ★ {book.neighbor_avg.toFixed(2)} ({book.neighbor_raters})</span>
          )}
        </div>
        <div className="card-tags">
          {book.genre && <GenreChip genre={book.genre} />}
          {book.tags.filter((t) => t !== book.genre).slice(0, 3).map((t) => <Tag key={t}>{t}</Tag>)}
        </div>
        {book.because && book.because.length > 0 && (
          <div className="because">
            Because you liked{" "}
            {book.because.map((b, i) => (
              <span key={b.id}>
                {i > 0 && (i === book.because!.length - 1 ? " and " : ", ")}
                <button type="button" className="link" onClick={() => onOpen(b.id)}>{b.title}</button>
              </span>
            ))}
          </div>
        )}
        <div className="card-actions">
          <Stars value={myRating} size="sm" label={`Rate ${book.title}`}
                 onChange={(v) => (v ? shelf.rate(book, v) : shelf.unrate(book.id))} />
          <button type="button" className="ghost small" onClick={() => shelf.dismiss(book.id)}
                  title="Not interested — hide this book">Not interested</button>
        </div>
      </div>
    </article>
  );
}
