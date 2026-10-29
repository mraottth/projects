// Job descriptions saved as plain text (Adzuna's snippets, pages that only gave plain text, and
// jobs saved before descriptions kept their formatting) are one line per bit of text: no
// headings, no bullets, and a bolded phrase split onto a line of its own. This rebuilds a readable
// layout for Markdown.tsx without any server call. Already-formatted descriptions (the backend's
// html_to_markdown output, which has blank lines between blocks) pass through untouched.

const BULLET = /^\s*[•·▪◦‣*–-]\s+(.*)$/;
// A line ending on one of these is mid-sentence ("We're hiring a" / "Data Scientist ...").
const DANGLING = /\b(a|an|the|of|to|and|or|with|for|in|on|at|by|our|your|their|is|are)$/i;

function isHeading(line: string): boolean {
  if (line.length > 60 || /[.!?]$/.test(line)) return false;
  if (line.endsWith(":")) return true;
  const letters = line.replace(/[^A-Za-z]/g, "");
  return letters.length >= 4 && letters === letters.toUpperCase();
}

export function tidyDescription(text: string): string {
  if (/\n\s*\n/.test(text) || /^#{2,4} /m.test(text)) return text;

  const blocks: string[] = [];
  let paragraph = "";
  const flush = () => {
    if (paragraph) blocks.push(paragraph);
    paragraph = "";
  };

  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const bullet = line.match(BULLET);
    if (bullet) {
      flush();
      // Consecutive bullets stay one list: append to the previous block when it's a list too.
      const last = blocks[blocks.length - 1];
      if (last?.startsWith("- ")) blocks[blocks.length - 1] = `${last}\n- ${bullet[1]}`;
      else blocks.push(`- ${bullet[1]}`);
    } else if (isHeading(line)) {
      flush();
      blocks.push(`### ${line.replace(/:$/, "")}`);
    } else if (paragraph && (/^[a-z,.;:)]/.test(line) || DANGLING.test(paragraph))) {
      // A mid-sentence break (the old one-line-per-text-node layout, which split out every bolded
      // phrase): rejoin it. Otherwise each line keeps its own paragraph, since an unpunctuated line
      // may well be a heading or a list item that lost its bullet.
      paragraph += /^[,.;:)]/.test(line) ? line : ` ${line}`;
    } else {
      flush();
      paragraph = line;
    }
  }
  flush();
  return blocks.join("\n\n");
}
