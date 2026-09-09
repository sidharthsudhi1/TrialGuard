import type { BudgetInfo, Limits, SearchResponse, TrialDetail } from "./types";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(
  /\/$/,
  ""
);

/** An API failure that kept the status, so the UI can tell "wait" from "broken". */
export class ApiError extends Error {
  status: number;
  retryAfter: number | null;

  constructor(message: string, status: number, retryAfter: number | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }

  get isRateLimited(): boolean {
    return this.status === 429;
  }
}

async function parseError(res: Response): Promise<ApiError> {
  const header = res.headers.get("retry-after");
  const retryAfter = header && /^\d+$/.test(header) ? Number(header) : null;
  let message: string;
  try {
    const body = await res.json();
    const detail = body.detail;
    if (typeof detail === "string") {
      message = detail;
    } else if (detail && typeof detail === "object" && detail.message) {
      message = `${detail.error || "Error"}: ${detail.message}`;
    } else {
      message = JSON.stringify(detail ?? body);
    }
  } catch {
    message = res.statusText || `HTTP ${res.status}`;
  }
  return new ApiError(message, res.status, retryAfter);
}

export function apiBase(): string {
  return API_URL;
}

export async function searchTrials(note: string, topK = 5): Promise<SearchResponse> {
  const res = await fetch(`${API_URL}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note, top_k: topK }),
  });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function startAssess(
  note: string,
  nctIds: string[],
  deep = false
): Promise<{ job_id: string }> {
  const res = await fetch(`${API_URL}/api/assess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note, nct_ids: nctIds, deep }),
  });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function fetchTrial(nctId: string): Promise<TrialDetail> {
  const res = await fetch(`${API_URL}/api/trials/${encodeURIComponent(nctId)}`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function fetchBudget(): Promise<BudgetInfo> {
  const res = await fetch(`${API_URL}/api/budget`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export function assessStreamUrl(jobId: string): string {
  return `${API_URL}/api/assess/${encodeURIComponent(jobId)}`;
}

export async function fetchLimits(): Promise<Limits> {
  const res = await fetch(`${API_URL}/api/limits`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}
