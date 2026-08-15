import type { BudgetInfo, SearchResponse, TrialDetail } from "./types";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(
  /\/$/,
  ""
);

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && detail.message) {
      return `${detail.error || "Error"}: ${detail.message}`;
    }
    return JSON.stringify(detail ?? body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
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
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function startAssess(
  note: string,
  nctIds: string[]
): Promise<{ job_id: string }> {
  const res = await fetch(`${API_URL}/api/assess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note, nct_ids: nctIds }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchTrial(nctId: string): Promise<TrialDetail> {
  const res = await fetch(`${API_URL}/api/trials/${encodeURIComponent(nctId)}`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchBudget(): Promise<BudgetInfo> {
  const res = await fetch(`${API_URL}/api/budget`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export function assessStreamUrl(jobId: string): string {
  return `${API_URL}/api/assess/${encodeURIComponent(jobId)}`;
}
