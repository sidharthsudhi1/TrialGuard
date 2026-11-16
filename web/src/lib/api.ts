import type { BudgetInfo, Limits, SearchResponse, SearchTrial, TrialDetail } from "./types";

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

export async function searchTrials(
  note: string,
  topK = 5,
  deep = false
): Promise<SearchResponse> {
  const res = await fetch(`${API_URL}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note, top_k: topK, deep }),
  });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function searchTrialsStream(
  note: string,
  topK: number,
  deep: boolean,
  onEvent: (ev: { type: string; trials?: SearchTrial[]; latency_ms?: Record<string, number> }) => void
): Promise<void> {
  const res = await fetch(`${API_URL}/api/search/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note, top_k: topK, deep }),
  });
  if (!res.ok || !res.body) throw await parseError(res);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let cut = buf.indexOf("\n\n");
    while (cut >= 0) {
      const block = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      const line = block.split("\n").find((l) => l.startsWith("data:"));
      if (line) {
        const ev = JSON.parse(line.slice(5).trim());
        if (ev.type === "error") {
          const detail = ev.detail;
          const message =
            typeof detail === "string"
              ? detail
              : detail?.message || "Search failed";
          throw new ApiError(message, ev.status || 500, null);
        }
        onEvent(ev);
      }
      cut = buf.indexOf("\n\n");
    }
  }
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
