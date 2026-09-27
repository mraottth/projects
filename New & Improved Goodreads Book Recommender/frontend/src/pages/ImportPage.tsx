import { useState } from "react";
import { api, type ImportResult } from "../api";
import { Cover } from "../components/Cover";
import { useShelf } from "../store";

export function ImportPage({ go }: { go: (v: "recs") => void }) {
  const shelf = useShelf();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<ImportResult | null>(null);
  const [drag, setDrag] = useState(false);
  const [kept, setKept] = useState(0);              // books rated on this site that survived the re-import
  const [replaced, setReplaced] = useState(false);

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true); setError(null); setRes(null);
    try {
      const r = await api.importCsv(file);
      setRes(r);
      // Re-importing replaces the previous import (clearing any stale matches) but keeps books rated here.
      setKept(Object.values(shelf.ratings).filter((v) => v.source === "manual" && !r.rated.some((x) => x.id === v.book.id)).length);
      setReplaced(false);
      shelf.applyImport(r, "update");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="import-page">
      <h1>Import your Goodreads library</h1>
      <ol className="steps">
        <li>Open <a href="https://www.goodreads.com/review/import" target="_blank" rel="noreferrer">goodreads.com/review/import</a> and click <strong>Export Library</strong>.</li>
        <li>Download the <code>goodreads_library_export.csv</code> file when it's ready.</li>
        <li>Drop it below. It's processed on the server and not stored; your shelf stays in this browser.</li>
      </ol>

      <label className={`dropzone${drag ? " drag" : ""}`}
             onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
             onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files[0]); }}>
        <input type="file" accept=".csv,text/csv" onChange={(e) => upload(e.target.files?.[0])} hidden />
        {busy ? <div className="spinner" /> : <><strong>Drop your CSV here</strong><span className="muted">or click to choose a file</span></>}
      </label>
      {error && <p className="error">{error}</p>}

      {res && (
        <section className="import-summary">
          <h2>Matched {res.stats.matched} of {res.stats.rows} books</h2>
          <p>
            <strong>{res.rated.length}</strong> rated, <strong>{res.read_unrated.length}</strong> read without a rating,{" "}
            <strong>{res.to_read.length}</strong> on your to-read shelf.
            {" "}Anything from a previous import was replaced.
            {kept > 0 && !replaced && (
              <> {kept} book{kept === 1 ? "" : "s"} you rated on this site {kept === 1 ? "was" : "were"} kept.{" "}
                <button type="button" className="link" onClick={() => { shelf.applyImport(res, "replace"); setReplaced(true); }}>
                  Replace my whole shelf instead
                </button>
              </>
            )}
            {replaced && " Your shelf now contains only this import."}
          </p>
          <div className="cover-strip">
            {res.rated.slice(0, 24).map((b) => (
              <div key={b.id} className="strip-item" title={`${b.title} — ${"★".repeat(b.rating)}`}>
                <Cover book={b} size="sm" />
              </div>
            ))}
          </div>
          <p><button type="button" className="primary" onClick={() => go("recs")}>See my recommendations →</button></p>
          {res.unmatched.length > 0 && (
            <details>
              <summary>{res.unmatched.length} rated or read books couldn't be matched</summary>
              <p className="muted small">
                The catalog covers books published through 2017 with at least 20 ratings in the dataset, so newer and
                very niche titles are missing.
              </p>
              <ul className="unmatched">
                {res.unmatched.map((u, i) => <li key={i}>{u.title} <span className="muted">— {u.author}{u.year ? ` (${u.year})` : ""}</span></li>)}
              </ul>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
