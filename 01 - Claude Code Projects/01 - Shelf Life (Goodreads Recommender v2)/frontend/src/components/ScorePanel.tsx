import { formatCount, StarBar } from "./Stars";

/** Readers-like-you average is shown only when at least this many of them rated the book. */
export const MIN_READER_RATINGS = 3;

interface Props {
  predicted: number;
  avgRating: number;
  ratingsCount: number;
  readersAvg?: number | null;
  readersN?: number | null;
}

/** The blue score box: predicted rating (big), then Goodreads average and readers-like-you average. */
export function ScorePanel({ predicted, avgRating, ratingsCount, readersAvg, readersN }: Props) {
  const showReaders = readersAvg != null && readersN != null && readersN >= MIN_READER_RATINGS;
  return (
    <div className="predicted" title="The star rating we expect you'd give this book">
      <span className="predicted-label">Predicted for you</span>
      <span className="predicted-value">{predicted.toFixed(2)}</span>
      <StarBar value={predicted} />
      <dl className="compare">
        <div title={`Average of ${ratingsCount.toLocaleString()} Goodreads ratings`}>
          <dt>Goodreads avg</dt>
          <dd>★ {avgRating.toFixed(2)}<span className="compare-n"> ({formatCount(ratingsCount)})</span></dd>
        </div>
        {showReaders && (
          <div title={`Average of ${readersN} ratings from the readers most like you`}>
            <dt>Readers like you</dt>
            <dd>★ {readersAvg!.toFixed(1)}<span className="compare-n"> ({readersN})</span></dd>
          </div>
        )}
      </dl>
    </div>
  );
}
