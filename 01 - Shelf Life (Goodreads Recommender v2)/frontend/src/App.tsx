import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { api, type Book } from "./api";
import logo from "./assets/logo-header.png";
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
// Changelog carries ~230 KB of prompts and commits, so it's only loaded when visited.
const ChangelogPage = lazy(() => import("./pages/ChangelogPage").then((m) => ({ default: m.ChangelogPage })));

// `tip`: shown on hover (desktop) and under each item in the phone menu.
const NAV: { view: UrlState["view"]; label: string; tip: string }[] = [
  { view: "recs", label: "Recommendations", tip: "See your personal picks" },
  { view: "explore", label: "Explore", tip: "Browse and filter the whole library" },
  { view: "yours", label: "Your books", tip: "Your shelf and reading stats" },
  { view: "chat", label: "Assistant 🤖", tip: "Chat with AI about books and your recommendations" },
  { view: "rate", label: "Rate books", tip: "Rate books for better recommendations" },
  { view: "import", label: "Import", tip: "Bring in your Goodreads library" },
  { view: "about", label: "About", tip: "How the recommendations work" },
];

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
      lead: opts.demo ? 750 : 0,    // demo: the caption shows alone for 0.75 s first
      caption: opts.demo ? "Generating recommendations from a real Goodreads user's ratings library…" : "Finding your next favorite book…",
    });
    // Top recommendations join as a second wave (this request also warms the server cache for the Recs page).
    api.recommend({ ...body, limit: FLOW_RECS })
      .then((r) => setFlow((f) => f && { ...f, recs: r.for_you }))
      .catch(() => {});
  }, [go, wall]);
  const [menuOpen, setMenuOpen] = useState(false);            // phone nav drop-down
  const toggleRef = useRef<HTMLButtonElement>(null);
  const navRef = useRef<HTMLElement>(null);
  useEffect(() => setMenuOpen(false), [url.view]);
  useEffect(() => {
    if (!menuOpen) return;
    document.body.classList.add("menu-open");
    navRef.current?.querySelector<HTMLButtonElement>("button.on, button")?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { setMenuOpen(false); toggleRef.current?.focus(); } };
    window.addEventListener("keydown", onKey);
    return () => { document.body.classList.remove("menu-open"); window.removeEventListener("keydown", onKey); };
  }, [menuOpen]);
  const current = NAV.find((n) => n.view === url.view);
  const shelfSize = new Set([...Object.keys(shelf.ratings).map(Number), ...shelf.read]).size;

  return (
    <>
      <header className={menuOpen ? "topbar menu-open" : "topbar"}>
        <button type="button" className="brand" onClick={() => go("home")} aria-label="Shelf Life, home">
          <img src={logo} alt="" width={125} height={48} />
        </button>
        {current && <span className="current-view" aria-hidden="true">{current.label}</span>}
        {/* Phones: the nav collapses into a drop-down behind this button (styles.css, "mobile menu"). */}
        <button type="button" className="menu-toggle" aria-expanded={menuOpen} aria-controls="site-nav" ref={toggleRef}
                aria-label={menuOpen ? "Close menu" : "Open menu"} onClick={() => setMenuOpen((o) => !o)}>
          <span className="menu-icon" aria-hidden="true" />
          {shelfSize > 0 && !menuOpen && <span className="menu-dot" aria-hidden="true" />}
        </button>
        <nav id="site-nav" ref={navRef}>
          {NAV.map((n) => (
            <button key={n.view} type="button" className={url.view === n.view ? "on" : ""} aria-describedby={`tip-${n.view}`}
                    aria-current={url.view === n.view ? "page" : undefined} onClick={() => { setMenuOpen(false); go(n.view); }}>
              {n.label}
              {n.view === "yours" && shelfSize ? <span className="count" aria-label={`${shelfSize} books`}>{shelfSize}</span> : null}
              <span className="nav-tip" id={`tip-${n.view}`} role="tooltip">{n.tip}</span>
            </button>
          ))}
        </nav>
      </header>
      {menuOpen && <div className="menu-backdrop" onClick={() => setMenuOpen(false)} />}

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
        {url.view === "changelog" && <Suspense fallback={<div className="spinner" />}><ChangelogPage /></Suspense>}
        {url.view === "explore" && (
          <ExplorePage filters={url.explore} setFilters={(explore) => update({ explore })}
                       sort={url.browseSort} setSort={(browseSort) => update({ browseSort })} onOpen={onOpen} />
        )}
      </main>

      <footer className="footer muted small">
        Data: UCSD Book Graph — M. Wan &amp; J. McAuley, "Item Recommendation on Monotonic Behavior Chains" (RecSys 2018)
        and M. Wan et al., "Fine-Grained Spoiler Detection from Large-Scale Review Corpora" (ACL 2019). Non-commercial use.
        {" "}· <button type="button" className="link" onClick={() => go("changelog")}>Changelog: how this was built</button>
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
