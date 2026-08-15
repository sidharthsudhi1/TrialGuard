import type { CriterionVerdict, TrialVerdict } from "../lib/types";

const TRIAL: Record<TrialVerdict, string> = {
  eligible: "Eligible",
  excluded: "Excluded",
  cannot_determine: "Cannot determine",
};

const CRIT: Record<CriterionVerdict, string> = {
  met: "met",
  not_met: "not met",
  cannot_determine: "cannot determine",
  unverifiable: "unverifiable",
};

export function TrialVerdictBadge({ verdict }: { verdict: TrialVerdict }) {
  return <span className={`badge badge-${verdict}`}>{TRIAL[verdict]}</span>;
}

export function CriterionVerdictBadge({ verdict }: { verdict: CriterionVerdict }) {
  return <span className={`badge badge-crit badge-${verdict}`}>{CRIT[verdict]}</span>;
}
