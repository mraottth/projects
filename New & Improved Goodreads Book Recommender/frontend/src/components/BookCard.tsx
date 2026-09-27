import type { Book, ReaderBook, RecBook } from "../api";
import { useShelf } from "../store";
import { Cover } from "./Cover";
import { Description } from "./Description";
import { GoodreadsLink } from "./GoodreadsLink";
import { GenreChip, Tag } from "./GenreChip";
import { ScorePanel } from "./ScorePanel";
import { AvgRating, Stars } from "./Stars";

type AnyBook = Book & Partial<RecBook> & Partial<ReaderBook>;

export type FilterClick = { kind: "author"; id: number; name: string } | { kind: "genre" | "tag"; value: string };

interface Props {
  book: AnyBook;
  onOpen: (id: number) => void;
  rank?: number;
  onFilter?: (f: FilterClick) => void;   // clicking author / genre / tag filters the page
}

/** Full-width ranked row: rank | cover | details | predicted rating. */
export function BookCard({ book, onOpen, rank, onFilter }: Props) {
  const shelf = useShelf();
  const myRating = shelf.ratings[book.id]?.rating ?? 0;
  const seriesLabel = book.series ? `${book.series}${book.series_pos ? ` #${book.series_pos}` : ""}` : null;

  return (
    <article className="rank-card">
      {rank != null && <div className={`rank-num${rank <= 3 ? " top" : ""}`} aria-label={`Rank ${rank}`}>{rank}</div>}

      <div className="rank-cover">
        <Cover book={book} onClick={() => onOpen(book.id)} />
        {book.next_in_series && <span className="badge badge-series">Next in series</span>}
      </div>

      <div className="rank-body">
        <div className="title-row">
          <button type="button" className="card-title" onClick={() => onOpen(book.id)}>{book.title}</button>
          <GoodreadsLink url={book.url} title={book.title} />
        </div>
        <div className="card-author">
          {onFilter && book.author_id != null
            ? <button type="button" className="filter-link" title={`Show only books by ${book.author}`}
                      onClick={() => onFilter({ kind: "author", id: book.author_id!, name: book.author })}>{book.author}</button>
            : book.author}
          {book.year ? <span className="muted"> · {book.year}</span> : null}
          {seriesLabel && <span className="muted"> · {seriesLabel}</span>}
        </div>
        <div className="card-stats">
          <AvgRating value={book.avg_rating} count={book.ratings_count} />
          {book.pct_read != null && <span className="stat-pill">{book.pct_read}% of similar readers read it</span>}
        </div>
        <div className="card-tags">
          {book.genre && <GenreChip genre={book.genre} onClick={onFilter && (() => onFilter({ kind: "genre", value: book.genre! }))} />}
          {book.tags.filter((t) => t !== book.genre).slice(0, 4).map((t) => (
            <Tag key={t} onClick={onFilter && (() => onFilter({ kind: "tag", value: t }))}>{t}</Tag>
          ))}
        </div>
        {book.description && <Description text={book.description} url={book.url} />}
        {book.because && book.because.length > 0 && (
          <div className="because">
            Because you liked{" "}
            {book.because.map((b, i) => (
              <span key={b.id}>
                {i > 0 && (i === book.because!.length - 1 ? " and " : ", ")}
                <a href="#" className="because-link"
                   onClick={(e) => { e.preventDefault(); onOpen(b.id); }}>{b.title}</a>
              </span>
            ))}
          </div>
        )}
        <div className="card-actions">
          <span className="muted small">Read it?</span>
          <Stars value={myRating} size="sm" label={`Rate ${book.title}`}
                 onChange={(v) => (v ? shelf.rate(book, v) : shelf.unrate(book.id))} />
          <button type="button" className="ghost small" onClick={() => shelf.dismiss(book.id)}
                  title="Not interested — hide this book">Not interested</button>
        </div>
      </div>

      {book.predicted_rating != null && (
        <ScorePanel predicted={book.predicted_rating} avgRating={book.avg_rating} ratingsCount={book.ratings_count}
                    readersAvg={book.readers_avg} readersN={book.readers_n} />
      )}
    </article>
  );
}
