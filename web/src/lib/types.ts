/** Shared types matching the Stage A FastAPI JSON contract. */

export type TrialVerdict = "eligible" | "excluded" | "cannot_determine";
export type CriterionVerdict =
  | "met"
  | "not_met"
  | "cannot_determine"
  | "unverifiable";

export interface SearchTrial {
  nct_id: string;
  title: string | null;
  status: string | null;
  phase: string | null;
  conditions: string[];
  /** CT.gov lastUpdatePostDate, YYYY-MM-DD. Null on records ingested before it was stored. */
  last_updated: string | null;
  score: number;
}

export interface SearchResponse {
  trials: SearchTrial[];
  top_k: number;
  latency_ms: Record<string, number>;
  notice: string;
}

export interface Assessment {
  criterion: string;
  verdict: CriterionVerdict;
  kind?: "inclusion" | "exclusion";
  quote?: string;
  grounded?: boolean;
  grounding_failure?: boolean;
  rationale?: string;
}

export interface TrialEvent {
  type: "trial";
  nct_id: string;
  title?: string | null;
  status?: string | null;
  trial_verdict: TrialVerdict;
  criteria_truncated?: boolean;
  assessments: Assessment[];
  error?: string;
}

/** Provisional progress event: one criterion as the analyst emits it.
 *  Pre-grounding, and a retry can supersede it — the TrialEvent for the same
 *  nct_id is authoritative and replaces anything shown from these. */
export interface CriterionEvent {
  type: "criterion";
  nct_id: string;
  provisional: true;
  criterion: string;
  verdict: CriterionVerdict;
}

export interface TrialDetail {
  nct_id: string;
  title: string | null;
  status: string | null;
  phase: string | null;
  conditions: string[];
  eligibility_raw: string;
  inclusion_criteria: string[];
  exclusion_criteria: string[];
  last_updated: string | null;
}

export interface BudgetInfo {
  usd_spent: number;
  usd_cap: number;
  remaining_usd: number;
  exhausted: boolean;
  calls: number;
  date: string;
}

/** Caps and measured per-trial rates, so the UI can quote before spending. */
export interface Limits {
  max_assess_trials: number;
  max_assess_trials_deep: number;
  assess_workers: number;
  usd_per_trial: number;
  seconds_per_trial: number;
}
