import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";
import { api, type ChatBook, type ChatStatus, type ChatTier } from "../api";
import { Cover } from "../components/Cover";
import { useShelf } from "../store";
import type { UrlState } from "../useUrlState";

/** The "Assistant" tab: a reading assistant (Claude + tools over the recommender; see src/goodrec/api/chat.py). */

interface Msg { role: "user" | "assistant"; content: string; tier?: ChatTier; label?: string }
interface Saved { messages: Msg[]; books: Record<number, ChatBook> }

const KEY = "goodrec.chat.v1";
const load = (): Saved => {
  try {
    return { messages: [], books: {}, ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return { messages: [], books: {} };
  }
};

// Buttons that send right away carry a fixed model tier; prefilled ones are routed once the user finishes the sentence.
const STARTERS: { label: string; prompt: string; prefill?: boolean; tier?: ChatTier }[] = [
  { label: "What should I read next?", prompt: "What should I read next? Give me a few strong picks with a reason for each.", tier: "simple" },
  { label: "I'm in the mood for...", prompt: "I'm in the mood for ", prefill: true },
  { label: "Help me find newer books", prompt: "Based on my taste, what books published since 2017 would you recommend for me?", tier: "complex" },
  { label: "What does my reading say about me?",
    prompt: "What does my reading history say about my taste? Point out patterns, blind spots, and anything surprising.", tier: "complex" },
  { label: "Chat about a book", prompt: "Let's talk about ", prefill: true },
  { label: "More like a book I loved", prompt: "Find me books that feel like ", prefill: true },
  { label: "Take me outside my comfort zone", prompt: "Recommend a few books outside my usual genres that I'd still probably love.", tier: "complex" },
];

export function ChatPage({ go, onOpen, seed, clearSeed }: {
  go: (v: UrlState["view"]) => void; onOpen: (id: number) => void; seed: string | null; clearSeed: () => void;
}) {
  const shelf = useShelf();
  const [saved, setSaved] = useState<Saved>(load);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cfg, setCfg] = useState<ChatStatus | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => { api.chatStatus().then(setCfg).catch(() => setCfg(null)); }, []);
  useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch { /* storage full: keep in memory */ }
  }, [saved]);
  useEffect(() => { endRef.current?.scrollIntoView({ block: "end" }); }, [saved.messages.length, busy]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const send = async (text: string, tier?: ChatTier) => {
    text = text.trim();
    if (!text || busy) return;
    setError(null);
    setInput("");
    const history: Msg[] = [...saved.messages, { role: "user", content: text }];
    setSaved((s) => ({ ...s, messages: [...history, { role: "assistant", content: "" }] }));
    setBusy(true);
    const ctl = new AbortController();
    abortRef.current = ctl;
    let got = "";
    const fail = (msg: string) => {
      setError(msg);
      if (!got.trim()) {   // nothing came back: drop the exchange and give the question back
        setSaved((s) => ({ ...s, messages: s.messages.slice(0, -2) }));
        setInput(text);
      }
    };
    try {
      const setLast = (patch: Partial<Msg>) => setSaved((s) => {
        const msgs = s.messages.slice();
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], ...patch };
        return { ...s, messages: msgs };
      });
      await api.chatStream({ messages: history.map(({ role, content, tier: t }) => ({ role, content, tier: t })), tier,
                             shelf: shelf.chatShelf() }, (e) => {
        if (e.type === "text") {
          got += e.data;
          setStatus(null);
          setLast({ content: got });
        } else if (e.type === "model") setLast({ tier: e.data.tier, label: e.data.label });
        else if (e.type === "status") setStatus(e.data);
        else if (e.type === "books") setSaved((s) => ({ ...s, books: { ...s.books, ...Object.fromEntries(e.data.map((b) => [b.id, b])) } }));
        else if (e.type === "error") fail(`Something went wrong: ${e.data}`);
      }, ctl.signal);
    } catch (err) {
      if ((err as Error).name !== "AbortError") fail((err as Error).message);
    } finally {
      setBusy(false);
      setStatus(null);
      abortRef.current = null;
    }
  };

  const pick = (s: (typeof STARTERS)[number]) => {
    if (s.prefill) {
      setInput(s.prompt);
      requestAnimationFrame(() => { inputRef.current?.focus(); inputRef.current?.setSelectionRange(s.prompt.length, s.prompt.length); });
    } else send(s.prompt, s.tier);
  };

  // "Chat about this book" from a book pop-up.
  const seeded = useRef(false);
  useEffect(() => {
    if (seed && !seeded.current && cfg?.enabled) {
      seeded.current = true;
      clearSeed();
      send(seed);
    }
  }); // eslint-disable-line react-hooks/exhaustive-deps

  const newChat = () => {
    abortRef.current?.abort();
    setSaved({ messages: [], books: {} });
    setError(null);
    inputRef.current?.focus();
  };

  const empty = saved.messages.length === 0;
  if (cfg && !cfg.enabled) {
    return (
      <div className="chat-page">
        <h1>Assistant 🤖</h1>
        <p className="muted">The reading assistant isn't configured on this server.</p>
      </div>
    );
  }

  return (
    <div className="chat-page">
      <div className="chat-head">
        <h1>Assistant 🤖</h1>
        {!empty && <button type="button" className="ghost" onClick={newChat}>New chat</button>}
      </div>

      {empty && (
        <div className="chat-intro">
          <p className="lead-sm">An AI assistant that knows your reading tastes</p>
          {shelf.count === 0 && (
            <p className="note">
              You haven't rated anything yet, so answers won't be personalized.{" "}
              <button type="button" className="link" onClick={() => go("import")}>Import from Goodreads</button> or{" "}
              <button type="button" className="link" onClick={() => go("rate")}>rate a few books</button> first.
            </p>
          )}
          <div className="starters">
            {STARTERS.map((s) => (
              <button type="button" key={s.label} className="starter" onClick={() => pick(s)} disabled={busy}>
                {s.label}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="chat-log" aria-live="polite">
        {saved.messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            {m.role === "user" ? <p>{m.content}</p> : (
              <Markdown text={busy && i === saved.messages.length - 1 ? m.content.replace(/\[\[[^\]]*$/, "") : m.content}
                        books={saved.books} onOpen={onOpen} />
            )}
            {m.role === "assistant" && m.label && m.content && !(busy && i === saved.messages.length - 1) && (
              <div className="chat-model muted small">{m.label}</div>
            )}
            {m.role === "assistant" && busy && i === saved.messages.length - 1 && (
              <div className="chat-status muted small"><span className="dot-pulse" />{status ?? (m.content ? "" : "Thinking…")}</div>
            )}
          </div>
        ))}
        {error && <p className="error">{error}</p>}
        <div ref={endRef} />
      </div>

      <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <textarea ref={inputRef} value={input} rows={2} placeholder="Ask about books, your taste, or what to read next…"
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }} />
        {busy
          ? <button type="button" className="ghost" onClick={() => abortRef.current?.abort()}>Stop</button>
          : <button type="submit" className="primary" disabled={!input.trim()}>Send</button>}
      </form>
      <p className="muted small chat-privacy">
        Answers come from Claude (Anthropic). Your ratings, reviews and shelves are sent with each message; nothing is stored on
        our server. Books marked with a cover are in our catalog; others come from the assistant's own knowledge or the web.
      </p>
    </div>
  );
}

// ------------------------------------------------------------------ minimal markdown (React elements only, no HTML injection)

const INLINE = /(\[\[book:\d+\]\]|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g;

function inline(text: string, books: Record<number, ChatBook>, onOpen: (id: number) => void): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    let m: RegExpMatchArray | null;
    if ((m = part.match(/^\[\[book:(\d+)\]\]$/))) {
      const b = books[Number(m[1])];
      return b
        ? <button type="button" key={i} className="book-ref" onClick={() => onOpen(b.id)}><em>{b.title}</em> by {b.author}</button>
        : <span key={i} className="book-ref pending">…</span>;
    }
    if ((m = part.match(/^\*\*([^*]+)\*\*$/))) return <strong key={i}>{inline(m[1], books, onOpen)}</strong>;
    if ((m = part.match(/^\*([^*]+)\*$/))) return <em key={i}>{m[1]}</em>;
    if ((m = part.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/))) return <a key={i} href={m[2]} target="_blank" rel="noreferrer">{m[1]}</a>;
    return <Fragment key={i}>{part}</Fragment>;
  });
}

/** A paragraph or list item that starts with a book marker becomes a card: cover, title/author, then the reason. */
function Line({ text, books, onOpen }: { text: string; books: Record<number, ChatBook>; onOpen: (id: number) => void }) {
  const m = text.match(/^\s*(?:\*\*)?\[\[book:(\d+)\]\](?:\*\*)?\s*[—–:-]*\s*/);
  const b = m ? books[Number(m[1])] : undefined;
  if (!m || !b) return <>{inline(text, books, onOpen)}</>;
  return (
    <span className="chat-book">
      <Cover book={b} size="sm" onClick={() => onOpen(b.id)} />
      <span>
        <button type="button" className="chat-book-title" onClick={() => onOpen(b.id)}>{b.title}</button>
        <span className="muted small"> {b.author}{b.year ? ` · ${b.year}` : ""}</span>
        <span className="chat-book-why">{inline(text.slice(m[0].length), books, onOpen)}</span>
      </span>
    </span>
  );
}

function Markdown({ text, books, onOpen }: { text: string; books: Record<number, ChatBook>; onOpen: (id: number) => void }) {
  const out: ReactNode[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  const flush = () => {
    if (para.length) {
      const lines = para;
      out.push(<p key={out.length}>{lines.map((l, i) => <Fragment key={i}>{i > 0 && <br />}<Line text={l} books={books} onOpen={onOpen} /></Fragment>)}</p>);
    }
    if (list) {
      const { ordered, items } = list;
      const Tag = ordered ? "ol" : "ul";
      out.push(<Tag key={out.length}>{items.map((it, i) => <li key={i}><Line text={it} books={books} onOpen={onOpen} /></li>)}</Tag>);
    }
    para = [];
    list = null;
  };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const li = line.match(/^\s*([-*•]|\d+[.)])\s+(.*)$/);
    const h = line.match(/^#{1,4}\s+(.*)$/);
    if (!line.trim()) flush();
    else if (h) { flush(); out.push(<h4 key={out.length}>{inline(h[1], books, onOpen)}</h4>); }
    else if (li) {
      const ordered = /\d/.test(li[1]);
      if (para.length || (list && list.ordered !== ordered)) flush();
      list = list ?? { ordered, items: [] };
      list.items.push(li[2]);
    } else if (list && /^\s{2,}/.test(raw)) list.items[list.items.length - 1] += " " + line.trim();   // wrapped list item
    else { if (list) flush(); para.push(line); }
  }
  flush();
  return <div className="md">{out}</div>;
}
