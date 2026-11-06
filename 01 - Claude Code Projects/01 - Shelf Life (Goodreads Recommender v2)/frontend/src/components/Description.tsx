import { useLayoutEffect, useRef, useState } from "react";

/** Description clamped to 3 lines with a More/Less toggle (shown only when the text overflows). */
export function Description({ text, url }: { text: string; url?: string }) {
  const [expanded, setExpanded] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const ref = useRef<HTMLParagraphElement>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || expanded) return;
    const check = () => setOverflows(el.scrollHeight > el.clientHeight + 1);
    check();
    const ro = new ResizeObserver(check);
    ro.observe(el);
    return () => ro.disconnect();
  }, [text, expanded]);

  const truncatedAtSource = text.endsWith("…");
  return (
    <div className="description-block">
      <p ref={ref} className={`blurb${expanded ? " expanded" : ""}`}>{text}</p>
      {(overflows || expanded) && (
        <button type="button" className="link small" aria-expanded={expanded} onClick={() => setExpanded((e) => !e)}>
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
      {expanded && truncatedAtSource && url && (
        <> · <a className="small" href={url} target="_blank" rel="noreferrer">Full description on Goodreads</a></>
      )}
    </div>
  );
}
