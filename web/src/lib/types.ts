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

export interface TrialDetail {
  nct_id: string;
  title: string | null;
  status: string | null;
  phase: string | null;
  conditions: string[];
  eligibility_raw: string;
  inclusion_criteria: string[];
  exclusion_criteria: string[];
}

export interface BudgetInfo {
  usd_spent: number;
  usd_cap: number;
  remaining_usd: number;
  exhausted: boolean;
  calls: number;
  date: string;
}
