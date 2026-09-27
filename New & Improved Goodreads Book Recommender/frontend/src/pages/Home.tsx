import { useEffect, useState } from "react";
import { api, type Book } from "../api";
import { Cover } from "../components/Cover";
import { useShelf } from "../store";

export function Home({ go, onOpen }: { go: (v: "rate" | "import" | "recs") => void; onOpen: (id: number) => void }) {
  const [books, setBooks] = useState<Book[]>([]);
  const shelf = useShelf();
  useEffect(() => { api.homeWall().then(setBooks).catch(() => {}); }, []);

  return (
    <div className="home">
      <div className="cover-wall" aria-hidden="true">
        {books.slice(0, 48).map((b, i) => (
          <div key={b.id} className="wall-item" style={{ animationDelay: `${(i % 12) * 60}ms` }}>
            <Cover book={b} size="sm" onClick={() => onOpen(b.id)} />
          </div>
        ))}
      </div>
      <section className="hero">
        <h1>Find your next favorite book</h1>
        <div className="entry-cards">
          <button type="button" className="entry" onClick={() => go("import")}>
            <span className="entry-icon">⇪</span>
            <strong>Upload Your Ratings from Goodreads</strong>
          </button>
          <button type="button" className="entry" onClick={() => go("rate")}>
            <span className="entry-icon">★</span>
            <strong>Rate a few books</strong>
          </button>
        </div>
        {shelf.count > 0 && (
          <p><button type="button" className="primary" onClick={() => go("recs")}>
            See recommendations for your {shelf.count} rated books →
          </button></p>
        )}
        <p className="muted small">Book data covers titles published through 2017 (UCSD Goodreads dataset).</p>
      </section>
    </div>
  );
}
