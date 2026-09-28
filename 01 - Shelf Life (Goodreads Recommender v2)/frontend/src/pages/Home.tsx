import { useState } from "react";
import { api, type Book } from "../api";
import { Cover } from "../components/Cover";
import { importBody, topCovers, type FlowBook } from "../components/CoverFlow";
import { useShelf } from "../store";

export function Home({ go, onOpen, toRecs, wall }: {
  go: (v: "rate" | "import" | "recs") => void; onOpen: (id: number) => void; toRecs: (covers: FlowBook[], body: object, opts?: { demo?: boolean }) => void; wall: Book[];
}) {
  const [demoState, setDemoState] = useState<"idle" | "loading" | "error">("idle");
  const [ownOpen, setOwnOpen] = useState(false);   // "Use my own ratings" reveals the upload / rate cards
  const shelf = useShelf();

  // Load a real Goodreads library (the site author's) so visitors without one can see the app in action.
  const seeDemo = async () => {
    if (shelf.count > 0 && !shelf.demo && !confirm("Replace the books you've rated here with the demo library?")) return;
    setDemoState("loading");
    try {
      const res = await api.demo();
      shelf.applyImport(res, "replace", true);
      setDemoState("idle");
      toRecs(topCovers(res), importBody(res), { demo: true });
    } catch {
      setDemoState("error");
    }
  };

  return (
    <div className="home">
      <div className="cover-wall" aria-hidden="true">
        {wall.slice(0, 48).map((b, i) => (
          <div key={b.id} className="wall-item" style={{ animationDelay: `${(i % 12) * 60}ms` }}>
            <Cover book={b} size="sm" onClick={() => onOpen(b.id)} />
          </div>
        ))}
      </div>
      <section className="hero">
        <h1>Find your next favorite book</h1>
        <div className="demo-cta">
          <div className="home-choices">
            <button type="button" className="primary demo-button" onClick={seeDemo} disabled={demoState === "loading"}>
              {demoState === "loading" ? "Loading demo…" : "See a demo"}
            </button>
            <button type="button" className={`own-button${ownOpen ? " on" : ""}`} aria-expanded={ownOpen}
                    aria-controls="own-ratings" onClick={() => setOwnOpen((o) => !o)}>
              Use my own ratings <span className="own-chevron" aria-hidden="true">▾</span>
            </button>
          </div>
          <span className="muted small">
            {demoState === "error" ? "The demo couldn't load. Please try again." : "The demo uses a real Goodreads library of 770 books"}
          </span>
        </div>
        <div id="own-ratings" className={`entry-reveal${ownOpen ? " open" : ""}`} inert={!ownOpen}>
        <div className="entry-reveal-inner">
        <div className="entry-cards">
          <button type="button" className="entry" onClick={() => go("import")}>
            <span className="entry-icon">⇪</span>
            <strong>Upload your ratings from Goodreads</strong>
          </button>
          <span className="entry-or">or</span>
          <button type="button" className="entry" onClick={() => go("rate")}>
            <span className="entry-icon">★</span>
            <strong>Rate books here to get recommendations</strong>
          </button>
        </div>
        </div>
        </div>
        {shelf.count > 0 && (
          <p><button type="button" className="outline-button" onClick={() => go("recs")}>
            {shelf.demo ? "See the demo recommendations →" : `See recommendations for your ${shelf.count} rated books →`}
          </button></p>
        )}
        <p className="muted small">Book data covers titles published through 2017 (UCSD Goodreads dataset).</p>
      </section>
    </div>
  );
}
