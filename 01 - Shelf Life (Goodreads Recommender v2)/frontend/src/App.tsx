import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { api, type Book } from "./api";
import { BookModal } from "./components/BookModal";
import { CoverFlow, fillCovers, FLOW_RECS, type FlowBook } from "./components/CoverFlow";
import { Home } from "./pages/Home";
import { ExplorePage } from "./pages/ExplorePage";
import { ImportPage } from "./pages/ImportPage";
import { RatePage } from "./pages/RatePage";
import { RecsPage } from "./pages/RecsPage";
import { YourBooksPage } from "./pages/YourBooksPage";
import { useShelf } from "./store";
import { useUrlState, type UrlState } from "./useUrlState";

// About (with KaTeX) is loaded on demand so the math library isn't in the main bundle.
const AboutPage = lazy(() => import("./pages/AboutPage").then((m) => ({ default: m.AboutPage })));
const ChatPage = lazy(() => import("./pages/ChatPage").then((m) => ({ default: m.ChatPage })));

export function App() {
  const [url, update] = useUrlState();
  const [openId, setOpenId] = useState<number | null>(null);
  const [chatSeed, setChatSeed] = useState<string | null>(null);   // "Chat about this book" from a pop-up
  // import / Rate books -> Recommendations transition (components/CoverFlow.tsx)
  const [flow, setFlow] = useState<{ books: FlowBook[]; recs: FlowBook[] | null; caption: string; duration: number; lead: number } | null>(null);
  const [wall, setWall] = useState<Book[]>([]);                  // homepage cover wall; also tops up the transition
  useEffect(() => { api.homeWall().then(setWall).catch(() => {}); }, []);
  const shelf = useShelf();
  const go = useCallback((view: UrlState["view"]) => update({ view }, true), [update]);
  const onOpen = useCallback((id: number) => setOpenId(id), []);
  /** Go to Recommendations after an import, with the user's top covers streaming across the screen. */
  const toRecsWithCovers = useCallback((own: FlowBook[], body: object, opts: { demo?: boolean } = {}) => {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) { go("recs"); return; }
    setFlow({
      books: fillCovers(own, wall), recs: null, duration: opts.demo ? 5000 : 4000,
      lead: opts.demo ? 1000 : 0,   // demo: the caption shows alone for a second first
      caption: opts.demo ? "Generating recommendations from a real Goodreads user's ratings library…" : "Finding your next favorite book…",
    });
    // Top recommendations join as a second wave (this request also warms the server cache for the Recs page).
    api.recommend({ ...body, limit: FLOW_RECS })
      .then((r) => setFlow((f) => f && { ...f, recs: r.for_you }))
      .catch(() => {});
  }, [go, wall]);
  const shelfSize = new Set([...Object.keys(shelf.ratings).map(Number), ...shelf.read]).size;

  return (
    <>
      <header className="topbar">
        <button type="button" className="brand" onClick={() => go("home")}>📚 Shelf Life</button>
        <nav>
          <button type="button" className={url.view === "recs" ? "on" : ""} aria-current={url.view === "recs" ? "page" : undefined} onClick={() => go("recs")}>
            Recommendations
          </button>
          <button type="button" className={url.view === "explore" ? "on" : ""} aria-current={url.view === "explore" ? "page" : undefined} onClick={() => go("explore")}>Explore</button>
          <button type="button" className={url.view === "yours" ? "on" : ""} aria-current={url.view === "yours" ? "page" : undefined} onClick={() => go("yours")}>
            Your books{shelfSize ? <span className="count" aria-label={`${shelfSize} books`}>{shelfSize}</span> : null}
          </button>
          <button type="button" className={url.view === "chat" ? "on" : ""} aria-current={url.view === "chat" ? "page" : undefined} onClick={() => go("chat")}>Assistant 🤖</button>
          <button type="button" className={url.view === "rate" ? "on" : ""} aria-current={url.view === "rate" ? "page" : undefined} onClick={() => go("rate")}>Rate books</button>
          <button type="button" className={url.view === "import" ? "on" : ""} aria-current={url.view === "import" ? "page" : undefined} onClick={() => go("import")}>Import</button>
          <button type="button" className={url.view === "about" ? "on" : ""} aria-current={url.view === "about" ? "page" : undefined} onClick={() => go("about")}>About</button>
        </nav>
      </header>

      <main>
        {url.view === "home" && <Home go={go} onOpen={onOpen} toRecs={toRecsWithCovers} wall={wall} />}
        {url.view === "rate" && <RatePage onOpen={onOpen} toRecs={toRecsWithCovers} />}
        {url.view === "import" && <ImportPage toRecs={toRecsWithCovers} />}
        {url.view === "recs" && (
          <RecsPage tab={url.tab} sort={url.sort} layout={url.layout} filters={url.filters} setTab={(tab) => update({ tab })}
                    setSort={(sort) => update({ sort })} setLayout={(layout) => update({ layout })}
                    setFilters={(filters) => update({ filters })} go={go} onOpen={onOpen} />
        )}
        {url.view === "yours" && <YourBooksPage go={go} onOpen={onOpen} />}
        {url.view === "chat" && (
          <Suspense fallback={<div className="spinner" />}>
            <ChatPage go={go} onOpen={onOpen} seed={chatSeed} clearSeed={() => setChatSeed(null)} />
          </Suspense>
        )}
        {url.view === "about" && <Suspense fallback={<div className="spinner" />}><AboutPage go={go} /></Suspense>}
        {url.view === "explore" && (
          <ExplorePage filters={url.explore} setFilters={(explore) => update({ explore })}
                       sort={url.browseSort} setSort={(browseSort) => update({ browseSort })} onOpen={onOpen} />
        )}
      </main>

      <footer className="footer muted small">
        Data: UCSD Book Graph — M. Wan &amp; J. McAuley, "Item Recommendation on Monotonic Behavior Chains" (RecSys 2018)
        and M. Wan et al., "Fine-Grained Spoiler Detection from Large-Scale Review Corpora" (ACL 2019). Non-commercial use.
        {shelf.count > 0 && (
          <> · <button type="button" className="link" onClick={() => confirm("Clear all your ratings from this browser?") && shelf.clear()}>Clear my shelf</button></>
        )}
      </footer>

      {flow && <CoverFlow {...flow} onSwitch={() => go("recs")} onDone={() => setFlow(null)} />}
      {openId != null && <BookModal id={openId} onClose={() => setOpenId(null)} onOpen={onOpen}
                                       onAsk={(title) => { setOpenId(null); setChatSeed(`Let's talk about ${title}.`); go("chat"); }} />}
    </>
  );
}
