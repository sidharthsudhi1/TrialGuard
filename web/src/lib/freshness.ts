/** How old a trial record is, per CT.gov's own lastUpdatePostDate.
 *
 * The corpus is reconciled on a schedule, so a record can be current, or it can
 * be one CT.gov has since revised. Showing the date rather than implying
 * currency is the point: a reader deciding whether to trust an eligibility quote
 * should be able to see how old the text it came from is.
 */

const DAY = 24 * 60 * 60 * 1000;

export interface Freshness {
  label: string;
  /** True past a year, where the record is old enough to say so outright. */
  stale: boolean;
}

export function freshness(lastUpdated: string | null | undefined): Freshness | null {
  if (!lastUpdated) return null;
  const then = Date.parse(lastUpdated);
  if (Number.isNaN(then)) return null;

  const days = Math.floor((Date.now() - then) / DAY);
  if (days < 0) return { label: `updated ${lastUpdated}`, stale: false };
  if (days === 0) return { label: "updated today", stale: false };
  if (days === 1) return { label: "updated yesterday", stale: false };
  if (days < 60) return { label: `updated ${days} days ago`, stale: false };

  const months = Math.floor(days / 30);
  if (days < 365) return { label: `updated ${months} months ago`, stale: false };

  const years = Math.floor(days / 365);
  return {
    label: years === 1 ? "updated over a year ago" : `updated over ${years} years ago`,
    stale: true,
  };
}
