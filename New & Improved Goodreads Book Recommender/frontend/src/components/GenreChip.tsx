import { genreColor } from "../genres";

/** Genre name in text ink with a family-color dot; the text, not the color, carries identity. */
export function GenreChip({ genre, active, onClick }: { genre: string; active?: boolean; onClick?: () => void }) {
  const Tag = onClick ? "button" : "span";
  return (
    <Tag type={onClick ? "button" : undefined} className={`chip${active ? " active" : ""}`} onClick={onClick}
         aria-pressed={onClick ? !!active : undefined}>
      <span className="dot" style={{ background: genreColor(genre) }} />
      {genre}
    </Tag>
  );
}

export function Tag({ children }: { children: string }) {
  return <span className="tag">{children}</span>;
}
