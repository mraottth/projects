import { Fragment, type ReactNode } from "react";

/**
 * Minimal, safe Markdown: builds React elements only (no HTML injection). Handles paragraphs,
 * headings, lists, block quotes, fenced code, tables, **bold**, *italic*, `code` and links.
 * Adapted from Shelf Life's components/Markdown.tsx, trimmed of its book-card/heat-table
 * specifics (not relevant here) -- used anywhere Claude's prose output (candidate profile,
 * preference profile, job rationale, generated resume/cover-letter text) is displayed, so it
 * doesn't render literal "**bold**" asterisks.
 */
const INLINE = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|==[^=]+==|\[[^\]]+\]\([^)\s]+\))/g;

function inline(text: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    let m: RegExpMatchArray | null;
    if ((m = part.match(/^`([^`]+)`$/))) return <code key={i}>{m[1]}</code>;
    if ((m = part.match(/^\*\*([^*]+)\*\*$/))) return <strong key={i}>{inline(m[1])}</strong>;
    if ((m = part.match(/^\*([^*]+)\*$/))) return <em key={i}>{m[1]}</em>;
    // "==i:...==" / "==f:...==": a scoring-rationale phrase tagged with the rating it explains
    // (interest / fit), colored to match that rating by the surrounding .rationale (index.css).
    if ((m = part.match(/^==(?:([if]):)?([^=]+)==$/))) {
      return (
        <mark key={i} className={m[1] ? `mark-${m[1]}` : undefined}>
          {inline(m[2])}
        </mark>
      );
    }
    if ((m = part.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/))) {
      return /^https?:\/\//.test(m[2]) ? (
        <a key={i} href={m[2]} target="_blank" rel="noreferrer">
          {m[1]}
        </a>
      ) : (
        <span key={i}>{m[1]}</span>
      );
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

const cells = (row: string) =>
  row
    .trim()
    .replace(/^\||\|$/g, "")
    .split("|")
    .map((c) => c.trim());

export function Markdown({ text }: { text: string }) {
  const out: ReactNode[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let quote: string[] = [];

  const flush = () => {
    if (para.length) {
      out.push(
        <p key={out.length}>
          {para.map((l, i) => (
            <Fragment key={i}>
              {i > 0 && <br />}
              {inline(l)}
            </Fragment>
          ))}
        </p>,
      );
    }
    if (list) {
      const { ordered, items } = list;
      const Tag = ordered ? "ol" : "ul";
      out.push(
        <Tag key={out.length}>
          {items.map((it, i) => (
            <li key={i}>{inline(it)}</li>
          ))}
        </Tag>,
      );
    }
    if (quote.length) out.push(<blockquote key={out.length}>{inline(quote.join(" "))}</blockquote>);
    para = [];
    list = null;
    quote = [];
  };

  const lines = text.split("\n");
  for (let k = 0; k < lines.length; k++) {
    const raw = lines[k];
    const line = raw.trimEnd();

    if (/^\s*```/.test(line)) {
      flush();
      const code: string[] = [];
      while (++k < lines.length && !/^\s*```/.test(lines[k])) code.push(lines[k]);
      out.push(
        <pre key={out.length}>
          <code>{code.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    if (/^\s*\|.*\|\s*$/.test(line) && k + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[k + 1])) {
      flush();
      const head = cells(line);
      const rows: string[][] = [];
      k += 1;
      while (k + 1 < lines.length && /^\s*\|.*\|\s*$/.test(lines[k + 1])) rows.push(cells(lines[++k]));
      out.push(
        <div key={out.length} className="md-table">
          <table>
            <thead>
              <tr>
                {head.map((c, i) => (
                  <th key={i}>{inline(c)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, j) => (
                <tr key={j}>
                  {r.map((c, i) => (
                    <td key={i}>{inline(c)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    const li = line.match(/^\s*([-*•]|\d+[.)])\s+(.*)$/);
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    const q = line.match(/^\s*>\s?(.*)$/);

    if (!line.trim()) flush();
    else if (h) {
      flush();
      out.push(
        <h4 key={out.length} className={`md-h${h[1].length}`}>
          {inline(h[2])}
        </h4>,
      );
    } else if (q) {
      if (para.length || list) flush();
      quote.push(q[1]);
    } else if (li) {
      const ordered = /\d/.test(li[1]);
      if (para.length || quote.length || (list && list.ordered !== ordered)) flush();
      list = list ?? { ordered, items: [] };
      list.items.push(li[2]);
    } else if (list && /^\s{2,}/.test(raw)) {
      list.items[list.items.length - 1] += " " + line.trim();
    } else {
      if (list || quote.length) flush();
      para.push(line);
    }
  }
  flush();

  return <div className="md">{out}</div>;
}
