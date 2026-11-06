import { SourceKind } from "./api";

// Turns a job-board URL into a source kind + config: a Greenhouse/Lever/Ashby board URL becomes
// that ATS with its token/slug (polled through its API), anything else a custom_board scraped by
// URL. Shared by Manage Sources' "Add a source" form and discovery's Monitor button.

const URL_PATTERNS: { kind: SourceKind; configKey: string; pattern: RegExp }[] = [
  { kind: "ats_greenhouse", configKey: "board_token", pattern: /(?:boards|job-boards)\.greenhouse\.io\/([a-z0-9_-]+)/i },
  { kind: "ats_lever", configKey: "company_slug", pattern: /jobs\.lever\.co\/([a-z0-9_-]+)/i },
  { kind: "ats_ashby", configKey: "board_name", pattern: /jobs\.ashbyhq\.com\/([a-z0-9_-]+)/i },
];

export function detectSourceFromUrl(input: string): { kind: SourceKind; config: Record<string, string> } | null {
  const trimmed = input.trim();
  if (!/^https?:\/\//i.test(trimmed)) return null;
  for (const { kind, configKey, pattern } of URL_PATTERNS) {
    const match = trimmed.match(pattern);
    if (match) return { kind, config: { [configKey]: match[1] } };
  }
  return { kind: "custom_board", config: { url: trimmed } };
}
