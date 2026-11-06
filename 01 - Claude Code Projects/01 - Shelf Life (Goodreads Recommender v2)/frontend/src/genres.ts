/**
 * 34 parent genres grouped into 8 families, one validated categorical hue per family
 * (dataviz reference palette, light mode; passes CVD/normal-vision checks as an 8-slot set).
 * Family hues are decorative identity cues only: every use pairs them with the genre name as
 * text (chips) or the title/author (cover placeholders), since three hues sit below 3:1 contrast.
 */
export const FAMILY_COLORS = {
  speculative: "#2a78d6", // blue
  suspense: "#eb6834",    // orange
  literary: "#1baf7a",    // aqua
  past: "#eda100",        // yellow
  romance: "#e87ba4",     // magenta
  life: "#008300",        // green
  ideas: "#4a3aa7",       // violet
  young: "#e34948",       // red
} as const;

type Family = keyof typeof FAMILY_COLORS;

const FAMILY_OF: Record<string, Family> = {
  "Science Fiction": "speculative", "Fantasy": "speculative", "Paranormal & Urban Fantasy": "speculative",
  "Mystery & Crime": "suspense", "Thriller & Suspense": "suspense", "Horror": "suspense",
  "True Crime": "suspense", "Adventure & Action": "suspense",
  "Literary Fiction": "literary", "Contemporary Fiction": "literary", "Classics": "literary",
  "Short Stories & Anthologies": "literary", "Poetry": "literary", "Plays & Drama": "literary",
  "Historical Fiction": "past", "History": "past", "Biography & Memoir": "past",
  "Romance": "romance", "Erotica": "romance", "Women's Fiction": "romance",
  "Science & Nature": "life", "Travel": "life", "Food, Health & Lifestyle": "life", "Sports": "life",
  "Self-Help & Psychology": "life", "Business & Economics": "life",
  "Philosophy & Religion": "ideas", "Christian & Inspirational": "ideas", "Politics & Society": "ideas",
  "Arts & Culture": "ideas", "Humor": "ideas",
  "Young Adult": "young", "Children's & Middle Grade": "young", "Graphic Novels & Comics": "young",
};

export function genreColor(genre: string | null | undefined): string {
  return (genre && FAMILY_COLORS[FAMILY_OF[genre]]) || "#898781";
}
