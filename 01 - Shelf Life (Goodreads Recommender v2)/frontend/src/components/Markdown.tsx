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
 * red by sign, "wins" / "losses" green / red by value, other numeric columns a light blue scale. User counts,
 * ties and text cells are left plain. */
function heatStyles(head: string[], rows: string[][]): (CSSProperties | undefined)[][] {
  const styles = rows.map((r) => r.map(() => undefined as CSSProperties | undefined));
  head.forEach((h, i) => {
    if (i === 0 || /users|tie/i.test(h)) return;
    const vals = rows.map((r) => cellNumber(r[i] ?? ""));
    if (vals.filter((v) => v !== null).length < 2) return;
    const signed = rows.some((r) => /^\s*[+\u2212]/.test((r[i] ?? "").replace(/\*\*/g, "")));
    const nums = vals.filter((v): v is number => v !== null);
    const lo = Math.min(...nums), hi = Math.max(...nums);
    const rgb = signed ? null : /loss/i.test(h) ? "208, 59, 59" : /win/i.test(h) ? "29, 122, 60" : "42, 120, 214";
    rows.forEach((_, j) => {
      const v = vals[j];
      if (v === null) return;
      if (signed) {
        if (Math.abs(v) < 1e-12) return;
        styles[j][i] = { background: v > 0 ? "rgba(29, 122, 60, 0.11)" : "rgba(208, 59, 59, 0.11)" };
      } else if (hi > lo) {
        styles[j][i] = { background: `rgba(${rgb}, ${(0.03 + 0.17 * (v - lo) / (hi - lo)).toFixed(3)})` };
      }
    });
  });
  return styles;
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
      out.push(
        <div key={out.length} className={`md-table${heat ? " heat" : ""}`}><table>
          <thead><tr>{head.map((c, i) => <th key={i}>{inline(c, ctx)}</th>)}</tr></thead>
          <tbody>{rows.map((r, j) => <tr key={j}>{r.map((c, i) => <td key={i} style={st?.[j]?.[i]}>{inline(c, ctx)}</td>)}</tr>)}</tbody>
        </table></div>,
      );
      continue;
    }
    const li = line.match(/^\s*([-*•]|\d+[.)])\s+(.*)$/);
    const h = line.match(/^#{1,4}\s+(.*)$/);
    const q = line.match(/^\s*>\s?(.*)$/);
    if (!line.trim()) flush();
    else if (h) { flush(); out.push(<h4 key={out.length}>{inline(h[1], ctx)}</h4>); }
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
