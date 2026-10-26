import { useEffect, useState } from "react";
import { api, DigestEntry } from "../api";
import { ScoredJobCard } from "../components/ScoredJobCard";

type FitFilter = "all" | "High" | "Med" | "Low";

export default function DigestPage() {
  const [entries, setEntries] = useState<DigestEntry[]>([]);
  const [interestFilter, setInterestFilter] = useState<FitFilter>("all");
  const [candidateFilter, setCandidateFilter] = useState<FitFilter>("all");
  const [favoritesOnly, setFavoritesOnly] = useState(false);

  useEffect(() => {
    api.getDigestToday().then((r) => setEntries(r.entries));
  }, []);

  const filtered = entries.filter((entry) => {
    if (interestFilter !== "all" && entry.interest_fit !== interestFilter) return false;
    if (candidateFilter !== "all" && entry.candidate_fit !== candidateFilter) return false;
    if (favoritesOnly && !entry.is_favorite) return false;
    return true;
  });

  return (
    <div>
      <h1>Today's digest</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        New and noteworthy — the best of what showed up today. For everything your tracked
        sources have turned up, see My Sources.
      </p>

      {entries.length === 0 && <p className="empty-state">Nothing in today's digest yet.</p>}

      {entries.length > 0 && (
        <div className="card">
          <div className="form-row" style={{ marginBottom: 0 }}>
            <div className="field" style={{ marginBottom: 0 }}>
              <span>Interest fit</span>
              <select value={interestFilter} onChange={(e) => setInterestFilter(e.target.value as FitFilter)}>
                <option value="all">All</option>
                <option value="High">High</option>
                <option value="Med">Med</option>
                <option value="Low">Low</option>
              </select>
            </div>
            <div className="field" style={{ marginBottom: 0 }}>
              <span>Candidate fit</span>
              <select value={candidateFilter} onChange={(e) => setCandidateFilter(e.target.value as FitFilter)}>
                <option value="all">All</option>
                <option value="High">High</option>
                <option value="Med">Med</option>
                <option value="Low">Low</option>
              </select>
            </div>
            <label className="field inline" style={{ marginBottom: 0 }}>
              <input
                type="checkbox"
                checked={favoritesOnly}
                onChange={(e) => setFavoritesOnly(e.target.checked)}
              />
              <span style={{ margin: 0, fontWeight: 400, color: "var(--ink)" }}>Favorites only</span>
            </label>
          </div>
        </div>
      )}

      {entries.length > 0 && filtered.length === 0 && (
        <p className="empty-state">No jobs match these filters.</p>
      )}

      {filtered.map((entry) => (
        <ScoredJobCard key={entry.id} job={entry} />
      ))}
    </div>
  );
}
