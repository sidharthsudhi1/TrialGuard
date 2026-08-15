/**
 * Highlight a verbatim quote inside eligibility_raw.
 * If the quote is not a substring, show grounding_failure — never fuzzy-match.
 */
export function EligibilityHighlight({
  source,
  quotes,
}: {
  source: string;
  quotes: { quote?: string; grounded?: boolean; grounding_failure?: boolean }[];
}) {
  const grounded = quotes.filter((q) => q.quote && q.grounded && !q.grounding_failure);
  const failures = quotes.filter((q) => q.grounding_failure);

  if (!source) {
    return <p className="muted">No eligibility text on file.</p>;
  }

  const segments = buildSegments(
    source,
    grounded.map((q) => (q.quote || "").trim()).filter(Boolean)
  );

  return (
    <div className="eligibility">
      {failures.length > 0 && (
        <p className="quote-fail">
          {failures.length} quote{failures.length === 1 ? "" : "s"} failed verbatim
          grounding and were not highlighted.
        </p>
      )}
      <pre className="eligibility-raw">
        {segments.map((seg, i) =>
          seg.hit ? (
            <mark key={i} className="quote-mark">
              {seg.text}
            </mark>
          ) : (
            <span key={i}>{seg.text}</span>
          )
        )}
      </pre>
    </div>
  );
}

function buildSegments(
  source: string,
  quotes: string[]
): { text: string; hit: boolean }[] {
  // Prefer longer quotes first so nested spans do not fragment incorrectly.
  const unique = [...new Set(quotes)].sort((a, b) => b.length - a.length);
  type Span = { start: number; end: number };
  const spans: Span[] = [];

  for (const q of unique) {
    let from = 0;
    while (from < source.length) {
      const idx = source.indexOf(q, from);
      if (idx < 0) break;
      const end = idx + q.length;
      const overlaps = spans.some((s) => !(end <= s.start || idx >= s.end));
      if (!overlaps) spans.push({ start: idx, end });
      from = end;
    }
  }

  spans.sort((a, b) => a.start - b.start);
  const out: { text: string; hit: boolean }[] = [];
  let cursor = 0;
  for (const s of spans) {
    if (cursor < s.start) out.push({ text: source.slice(cursor, s.start), hit: false });
    out.push({ text: source.slice(s.start, s.end), hit: true });
    cursor = s.end;
  }
  if (cursor < source.length) out.push({ text: source.slice(cursor), hit: false });
  if (out.length === 0) out.push({ text: source, hit: false });
  return out;
}
