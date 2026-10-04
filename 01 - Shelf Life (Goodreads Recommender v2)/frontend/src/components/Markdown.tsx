import { Fragment, type CSSProperties, type ReactNode } from "react";
import type { ChatBook } from "../api";
import { Cover } from "./Cover";

/**
 * Minimal, safe Markdown: builds React elements only (no HTML injection). Handles paragraphs, headings, lists,
 * block quotes, fenced code, tables, **bold**, *italic*, `code` and links. Used by the Assistant (which also
 * turns [[book:<work_id>]] markers into book cards via `books` / `onOpen`) and the Changelog page.
 */
type Books = Record<number, ChatBook>;
interface Ctx { books?: Books; onOpen?: (id: number) => void }

const INLINE = /(\[\[book:\d+\]\]|`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\[[^\]]+\]\([^)\s]+\))/g;

function inline(text: string, ctx: Ctx): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    let m: RegExpMatchArray | null;
    if ((m = part.match(/^\[\[book:(\d+)\]\]$/))) {
      const b = ctx.books?.[Number(m[1])];
      return b
        ? <button type="button" key={i} className="book-ref" onClick={() => ctx.onOpen?.(b.id)}><em>{b.title}</em> by {b.author}</button>
        : <span key={i} className="book-ref pending">…</span>;
    }
    if ((m = part.match(/^`([^`]+)`$/))) return <code key={i}>{m[1]}</code>;
    if ((m = part.match(/^\*\*([^*]+)\*\*$/))) return <strong key={i}>{inline(m[1], ctx)}</strong>;
    if ((m = part.match(/^\*([^*]+)\*$/))) return <em key={i}>{m[1]}</em>;
    if ((m = part.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/))) {
      // Only absolute http(s) links become anchors; repo-relative paths show as their label.
      return /^https?:\/\//.test(m[2])
        ? <a key={i} href={m[2]} target="_blank" rel="noreferrer">{m[1]}</a>
        : <span key={i} className="md-path">{m[1]}</span>;
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

/** A paragraph or list item that starts with a book marker becomes a card: cover, title/author, then the reason. */
/** Leading number of a table cell ("**0.0874**", "+0.0104 [..]", "−3.1%", "34.2%"); null if it isn't numeric. */
function cellNumber(c: string): number | null {
  const m = c.replace(/\*\*/g, "").trim().match(/^([+\u2212-]?)\s*(\d+(?:\.\d+)?)\s*%?(?:\s|\[|$)/);
  return m ? (m[1] === "-" || m[1] === "\u2212" ? -1 : 1) * Number(m[2]) : null;
}

/** Subtle conditional formatting for numeric table columns (Evaluation report cards): signed columns green or
 * red by sign (reversed for error differences, where negative is better), "wins" / "losses" green / red by
 * value, other numeric columns a light blue scale. User counts, ties, bias (signed but neither good nor bad) and
 * text cells are left plain. */
function heatStyles(head: string[], rows: string[][]): (CSSProperties | undefined)[][] {
  const styles = rows.map((r) => r.map(() => undefined as CSSProperties | undefined));
  head.forEach((h, i) => {
    if (i === 0 || /users|tie|^bias|^ratings$/i.test(h)) return;
    const lowerBetter = /^MAE diff|^relative$/i.test(h);
    const vals = rows.map((r) => cellNumber(r[i] ?? ""));
    if (vals.filter((v) => v !== null).length < 2) return;
    const signed = lowerBetter || rows.some((r) => /^\s*[+\u2212]/.test((r[i] ?? "").replace(/\*\*/g, "")));
    const nums = vals.filter((v): v is number => v !== null);
    const lo = Math.min(...nums), hi = Math.max(...nums);
    const rgb = signed ? null : /loss/i.test(h) ? "208, 59, 59" : /win/i.test(h) ? "29, 122, 60" : "42, 120, 214";
    rows.forEach((_, j) => {
      const v = vals[j];
      if (v === null) return;
      if (signed) {
        if (Math.abs(v) < 1e-12) return;
        styles[j][i] = { background: (v > 0) !== lowerBetter ? "rgba(29, 122, 60, 0.11)" : "rgba(208, 59, 59, 0.11)" };
      } else if (hi > lo) {
        styles[j][i] = { background: `rgba(${rgb}, ${(0.03 + 0.17 * (v - lo) / (hi - lo)).toFixed(3)})` };
      }
    });
  });
  return styles;
}

/** Column group of a report-table header, for drawing boundaries between groups (@10 | @20, wins/ties/losses |
 * differences, coverage | popularity). Columns with the same key sit together; "" never groups. */
function colGroup(h: string): string {
  const t = h.replace(/\*\*/g, "").trim().toLowerCase();
  if (/@10$/.test(t)) return "k10";
  if (/@20$/.test(t)) return "k20";
  if (/^(wins|ties|losses)$/.test(t)) return "wlt";
  if (/^(mean diff|mae diff|lift|relative|median gain)/.test(t)) return "gain";
  if (/^(mae|rmse|mae\/σ)$/.test(t)) return "err";
  if (/^(pearson|spearman)$/.test(t)) return "corr";
  if (/^±/.test(t)) return "within";
  if (/^(bias|mae|mae\/σ|spearman) (narrow|typical|wide)$/.test(t)) return `style-${t.split(" ")[0]}`;
  if (/^(bias|mae) \d★$/.test(t)) return `star-${t.split(" ")[0]}`;
  if (/^coverage/.test(t)) return "coverage";
  if (/^popularity/.test(t)) return "popularity";
  if (/^authors /.test(t)) return "authors";
  if (/^one author /.test(t)) return "oneauthor";
  if (/^(all|n=\d+)$/.test(t)) return "n";
  return t;
}

/** Spanning header for a column group, and the column's own label under it (heat tables only). */
const GROUP_LABEL: Record<string, string> = {
  k10: "Top 10", k20: "Top 20", coverage: "Coverage", popularity: "Popularity",
  authors: "Distinct authors", oneauthor: "Most from one author",
  err: "Error (lower is better)", corr: "Correlation", within: "Within",
  "style-mae": "MAE", "style-mae/σ": "Scale-adjusted MAE", "style-spearman": "Spearman", "style-bias": "Bias",
  "star-bias": "Bias", "star-mae": "MAE",
};
function subLabel(h: string, g: string): string {
  const t = h.replace(/\*\*/g, "").trim();
  if (g === "k10" || g === "k20") return t.replace(/@(10|20)$/, "");
  if (g === "coverage" || g === "popularity") return t.replace(/^(coverage|popularity)\s+/i, "");
  if (g === "authors" || g === "oneauthor") return t.replace(/^(authors|one author)\s+/i, "");
  if (/^(style-|star-)/.test(g)) return t.replace(/^\S+\s+/, "");
  return t;
}

/** Row group of a report-table row: the model under test (bold), the previous best, baselines, ablations. */
function rowGroup(first: string): number {
  const t = first.replace(/^vs\. /, "").trim();
  if (/^\*\*.*\*\*$/.test(t)) return 0;
  if (/^Previous best/.test(t)) return 1;
  if (/^Best match as displayed/.test(t)) return 1;
  if (/^(item-kNN only|ALS only|Shelf Life, no |Shelf Life, uncalibrated|grid: )/.test(t)) return 3;
  return 2;
}

function Line({ text, ctx }: { text: string; ctx: Ctx }) {
  const m = text.match(/^\s*(?:\*\*)?\[\[book:(\d+)\]\](?:\*\*)?\s*[—–:-]*\s*/);
  const b = m ? ctx.books?.[Number(m[1])] : undefined;
  if (!m || !b) return <>{inline(text, ctx)}</>;
  return (
    <span className="chat-book">
      <Cover book={b} size="sm" onClick={() => ctx.onOpen?.(b.id)} />
      <span>
        <button type="button" className="chat-book-title" onClick={() => ctx.onOpen?.(b.id)}>{b.title}</button>
        <span className="muted small"> {b.author}{b.year ? ` · ${b.year}` : ""}</span>
        <span className="chat-book-why">{inline(text.slice(m[0].length), ctx)}</span>
      </span>
    </span>
  );
}

const cells = (row: string) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

export function Markdown({ text, books, onOpen, heat = false }: {
  text: string; books?: Books; onOpen?: (id: number) => void; heat?: boolean;   // heat: shade numeric table columns
}) {
  const ctx: Ctx = { books, onOpen };
  const out: ReactNode[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let quote: string[] = [];
  const flush = () => {
    if (para.length) {
      const lines = para;
      out.push(<p key={out.length}>{lines.map((l, i) => <Fragment key={i}>{i > 0 && <br />}<Line text={l} ctx={ctx} /></Fragment>)}</p>);
    }
    if (list) {
      const { ordered, items } = list;
      const Tag = ordered ? "ol" : "ul";
      out.push(<Tag key={out.length}>{items.map((it, i) => <li key={i}><Line text={it} ctx={ctx} /></li>)}</Tag>);
    }
    if (quote.length) out.push(<blockquote key={out.length}>{inline(quote.join(" "), ctx)}</blockquote>);
    para = [];
    list = null;
    quote = [];
  };
  const lines = text.split("\n");
  for (let k = 0; k < lines.length; k++) {
    const raw = lines[k];
    const line = raw.trimEnd();
    if (/^\s*```/.test(line)) {                                    // fenced code block
      flush();
      const code: string[] = [];
      while (++k < lines.length && !/^\s*```/.test(lines[k])) code.push(lines[k]);
      out.push(<pre key={out.length}><code>{code.join("\n")}</code></pre>);
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && k + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[k + 1])) {   // table
      flush();
      const head = cells(line);
      const rows: string[][] = [];
      k += 1;
      while (k + 1 < lines.length && /^\s*\|.*\|\s*$/.test(lines[k + 1])) rows.push(cells(lines[++k]));
      const st = heat ? heatStyles(head, rows) : null;
      // Report tables (heat): boundaries between column groups and row groups, numbers right-aligned.
      const groups = head.map(colGroup);
      const colCls = head.map((_, i) => {
        if (!heat || i === 0) return "";
        const numeric = rows.filter((r) => cellNumber(r[i] ?? "") !== null).length >= Math.max(1, rows.length / 2);
        return [i === 1 || groups[i] !== groups[i - 1] ? "gb" : "", numeric ? "num" : ""].filter(Boolean).join(" ");
      });
      const rowCls = rows.map((r, j) => (heat && j > 0 && rowGroup(r[0] ?? "") !== rowGroup(rows[j - 1][0] ?? "") ? "rb" : ""));
      // Two header rows when columns form named groups (@10 / @20, coverage / popularity): a spanning group label
      // over short column labels.
      const spans: { label: string; start: number; len: number }[] = [];
      groups.forEach((g, i) => {
        const last = spans[spans.length - 1];
        if (i > 0 && GROUP_LABEL[g] && last && last.label === GROUP_LABEL[g] && last.start + last.len === i) last.len += 1;
        else if (i > 0 && GROUP_LABEL[g]) spans.push({ label: GROUP_LABEL[g], start: i, len: 1 });
      });
      const twoRow = heat && spans.length >= 2;
      out.push(
        <div key={out.length} className={`md-table${heat ? " heat" : ""}`}><table>
          <thead>
            {twoRow ? (
              <>
                <tr>
                  {head.map((c, i) => {
                    const sp = spans.find((x) => x.start === i);
                    if (sp) return <th key={i} colSpan={sp.len} className="gb grp">{sp.label}</th>;
                    if (spans.some((x) => i > x.start && i < x.start + x.len)) return null;
                    return <th key={i} rowSpan={2} className={colCls[i] || undefined}>{inline(c, ctx)}</th>;
                  })}
                </tr>
                <tr>
                  {head.map((c, i) => (spans.some((x) => i >= x.start && i < x.start + x.len)
                    ? <th key={i} className={colCls[i] || undefined}>{subLabel(c, groups[i])}</th> : null))}
                </tr>
              </>
            ) : (
              <tr>{head.map((c, i) => <th key={i} className={colCls[i] || undefined}>{inline(c, ctx)}</th>)}</tr>
            )}
          </thead>
          <tbody>{rows.map((r, j) => (
            <tr key={j} className={rowCls[j] || undefined}>
              {r.map((c, i) => <td key={i} className={colCls[i] || undefined} style={st?.[j]?.[i]}>{inline(c, ctx)}</td>)}
            </tr>
          ))}</tbody>
        </table></div>,
      );
      continue;
    }
    const li = line.match(/^\s*([-*•]|\d+[.)])\s+(.*)$/);
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    const q = line.match(/^\s*>\s?(.*)$/);
    if (!line.trim()) flush();
    else if (h) { flush(); out.push(<h4 key={out.length} className={`md-h${h[1].length}`}>{inline(h[2], ctx)}</h4>); }
    else if (q) { if (para.length || list) flush(); quote.push(q[1]); }
    else if (li) {
      const ordered = /\d/.test(li[1]);
      if (para.length || quote.length || (list && list.ordered !== ordered)) flush();
      list = list ?? { ordered, items: [] };
      list.items.push(li[2]);
    } else if (list && /^\s{2,}/.test(raw)) list.items[list.items.length - 1] += " " + line.trim();   // wrapped list item
    else { if (list || quote.length) flush(); para.push(line); }
  }
  flush();
  return <div className="md">{out}</div>;
}
