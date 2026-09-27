import { useCallback, useState } from "react";
import { BookModal } from "./components/BookModal";
import { Home } from "./pages/Home";
import { ImportPage } from "./pages/ImportPage";
import { RatePage } from "./pages/RatePage";
import { RecsPage } from "./pages/RecsPage";
import { useShelf } from "./store";
import { useUrlState, type UrlState } from "./useUrlState";

export function App() {
  const [url, update] = useUrlState();
  const [openId, setOpenId] = useState<number | null>(null);
  const shelf = useShelf();
  const go = useCallback((view: UrlState["view"]) => update({ view }, true), [update]);
  const onOpen = useCallback((id: number) => setOpenId(id), []);

  return (
    <>
      <header className="topbar">
        <button type="button" className="brand" onClick={() => go("home")}>📚 Shelf Life</button>
        <nav>
          <button type="button" className={url.view === "rate" ? "on" : ""} onClick={() => go("rate")}>Rate books</button>
          <button type="button" className={url.view === "import" ? "on" : ""} onClick={() => go("import")}>Import</button>
          <button type="button" className={url.view === "recs" ? "on" : ""} onClick={() => go("recs")}>
            Recommendations{shelf.count ? <span className="count">{shelf.count}</span> : null}
          </button>
        </nav>
      </header>

      <main>
        {url.view === "home" && <Home go={go} onOpen={onOpen} />}
        {url.view === "rate" && <RatePage go={go} onOpen={onOpen} />}
        {url.view === "import" && <ImportPage go={go} />}
        {url.view === "recs" && (
          <RecsPage tab={url.tab} filters={url.filters} setTab={(tab) => update({ tab })}
                    setFilters={(filters) => update({ filters })} go={go} onOpen={onOpen} />
        )}
      </main>

      <footer className="footer muted small">
        Data: UCSD Book Graph — M. Wan &amp; J. McAuley, "Item Recommendation on Monotonic Behavior Chains" (RecSys 2018)
        and M. Wan et al., "Fine-Grained Spoiler Detection from Large-Scale Review Corpora" (ACL 2019). Non-commercial use.
        {shelf.count > 0 && (
          <> · <button type="button" className="link" onClick={() => confirm("Clear all your ratings from this browser?") && shelf.clear()}>Clear my shelf</button></>
        )}
      </footer>

      {openId != null && <BookModal id={openId} onClose={() => setOpenId(null)} onOpen={onOpen} />}
    </>
  );
}
