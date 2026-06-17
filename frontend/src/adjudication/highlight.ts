/**
 * Highlights registry facts inside a retrieved chunk's text (the HITL page's own throughput aid:
 * a labeler should not have to hunt for the fact the answer claims). Pure so it unit tests without
 * a DOM. Longest fact values match first, so a short value never splits a longer one it is a
 * substring of (e.g. "5" inside "39.99" would otherwise fragment the match).
 */
type Segment = { text: string; highlighted: boolean; start: number };

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** `start` (the character offset the segment begins at) rides along so a caller mapping segments
 * to React elements has a stable, unique key for free -- segments never reorder within a render,
 * and no two segments of the same text can share a start offset, so this needs no array index. */
export function highlightFacts(text: string, factValues: string[]): Segment[] {
  const values = [...new Set(factValues.map((v) => v.trim()).filter((v) => v.length > 0))].sort(
    (a, b) => b.length - a.length,
  );
  if (!text || values.length === 0) return [{ text, highlighted: false, start: 0 }];

  const pattern = new RegExp(values.map(escapeRegExp).join("|"), "gi");
  const segments: Segment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null = pattern.exec(text);
  while (match !== null) {
    if (match.index > lastIndex) {
      segments.push({
        text: text.slice(lastIndex, match.index),
        highlighted: false,
        start: lastIndex,
      });
    }
    segments.push({ text: match[0], highlighted: true, start: match.index });
    lastIndex = match.index + match[0].length;
    match = pattern.exec(text);
  }
  if (lastIndex < text.length)
    segments.push({ text: text.slice(lastIndex), highlighted: false, start: lastIndex });
  return segments;
}
