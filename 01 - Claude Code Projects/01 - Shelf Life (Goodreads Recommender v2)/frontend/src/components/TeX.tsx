import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

/** Render a LaTeX string with KaTeX (static, author-written formulas only). */
export function TeX({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: block, throwOnError: false, strict: false }), [children, block]);
  return block
    ? <div className="tex-block" dangerouslySetInnerHTML={{ __html: html }} />
    : <span className="tex-inline" dangerouslySetInnerHTML={{ __html: html }} />;
}
