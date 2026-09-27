import { lazy, Suspense, useCallback, useState } from "react";
import { BookModal } from "./components/BookModal";
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
  const shelf = useShelf();
  const go = useCallback((view: UrlState["view"]) => update({ view }, true), [update]);
  const onOpen = useCallback((id: number) => setOpenId(id), []);
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
        {url.view === "home" && <Home go={go} onOpen={onOpen} />}
        {url.view === "rate" && <RatePage go={go} onOpen={onOpen} />}
        {url.view === "import" && <ImportPage go={go} />}
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

      {openId != null && <BookModal id={openId} onClose={() => setOpenId(null)} onOpen={onOpen}
                                       onAsk={(title) => { setOpenId(null); setChatSeed(`Let's talk about ${title}.`); go("chat"); }} />}
    </>
  );
}
