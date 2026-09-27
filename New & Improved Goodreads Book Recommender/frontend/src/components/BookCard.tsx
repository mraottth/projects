import type { Book, ReaderBook, RecBook } from "../api";
import { useShelf } from "../store";
import { Cover } from "./Cover";
import { Description } from "./Description";
import { GoodreadsLink } from "./GoodreadsLink";
import { GenreChip, Tag } from "./GenreChip";
import { AvgRating, formatCount, StarBar, Stars } from "./Stars";

type AnyBook = Book & Partial<RecBook> & Partial<ReaderBook>;

interface Props {
  book: AnyBook;
  onOpen: (id: number) => void;
  rank?: number;
}

/** Full-width ranked row: rank | cover | details | predicted rating. */
export function BookCard({ book, onOpen, rank }: Props) {
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
          {book.author}{book.year ? <span className="muted"> · {book.year}</span> : null}
          {seriesLabel && <span className="muted"> · {seriesLabel}</span>}
        </div>
        <div className="card-stats">
          <AvgRating value={book.avg_rating} count={book.ratings_count} />
          {book.pct_read != null && <span className="stat-pill">{book.pct_read}% of similar readers read it</span>}
        </div>
        <div className="card-tags">
          {book.genre && <GenreChip genre={book.genre} />}
          {book.tags.filter((t) => t !== book.genre).slice(0, 4).map((t) => <Tag key={t}>{t}</Tag>)}
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
        <div className="predicted" title="The star rating we expect you'd give this book">
          <span className="predicted-label">Predicted for you</span>
          <span className="predicted-value">{book.predicted_rating.toFixed(1)}</span>
          <StarBar value={book.predicted_rating} />
          <dl className="compare">
            <div title={`Average of ${book.ratings_count.toLocaleString()} Goodreads ratings`}>
              <dt>Goodreads avg</dt>
              <dd>★ {book.avg_rating.toFixed(2)}<span className="compare-n"> ({formatCount(book.ratings_count)})</span></dd>
            </div>
            {book.readers_n != null && (
              <div title={book.readers_n
                ? `Average of ${book.readers_n} rating${book.readers_n === 1 ? "" : "s"} from the readers most like you`
                : "None of the readers most like you have rated this book"}>
                <dt>Readers like you</dt>
                <dd>
                  {book.readers_avg != null ? <>★ {book.readers_avg.toFixed(1)}</> : "—"}
                  <span className={`compare-n${book.readers_n < 3 ? " thin" : ""}`}>
                    {book.readers_n ? ` (${book.readers_n})` : ""}
                  </span>
                </dd>
              </div>
            )}
          </dl>
        </div>
      )}
    </article>
  );
}
