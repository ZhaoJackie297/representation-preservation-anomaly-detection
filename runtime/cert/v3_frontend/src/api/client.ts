import type {
  ApiAlertDetail,
  ApiAlertPage,
  ApiBlockSummary,
  ApiEvidence,
  ApiHealth,
  ApiJob,
  ApiMetric,
  ApiModelInfo,
  ApiOverview,
  ApiRunSummary,
  ApiSystem
} from "./types";

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ?? "/api/v1"
).replace(/\/$/, "");

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers
    }
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as {
        detail?: string;
      };
      detail = body.detail ?? detail;
    } catch {
      // Keep the HTTP status when the body is not JSON.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export interface AlertQuery {
  block: string;
  page: number;
  pageSize: number;
  band?: string;
  source?: string;
  split?: string;
  q?: string;
  startDate?: string;
  endDate?: string;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

export const api = {
  health: () => request<ApiHealth>("/health"),
  model: () => request<ApiModelInfo>("/model"),
  runs: () => request<ApiRunSummary[]>("/runs"),
  run: (runId: string) =>
    request<ApiRunSummary>(`/runs/${encodeURIComponent(runId)}`),
  blocks: (runId: string) =>
    request<ApiBlockSummary[]>(
      `/runs/${encodeURIComponent(runId)}/blocks`
    ),
  overview: (runId: string, block: string) =>
    request<ApiOverview>(
      `/runs/${encodeURIComponent(runId)}/overview?block=${encodeURIComponent(block)}`
    ),
  alerts: (runId: string, query: AlertQuery) => {
    const params = new URLSearchParams({
      block: query.block,
      page: String(query.page),
      page_size: String(query.pageSize),
      sort_by: query.sortBy ?? "date",
      sort_order: query.sortOrder ?? "desc"
    });
    if (query.band) params.set("band", query.band);
    if (query.source) params.set("source", query.source);
    if (query.split) params.set("split", query.split);
    if (query.q) params.set("q", query.q);
    if (query.startDate) params.set("start_date", query.startDate);
    if (query.endDate) params.set("end_date", query.endDate);
    return request<ApiAlertPage>(
      `/runs/${encodeURIComponent(runId)}/alerts?${params}`
    );
  },
  alert: (runId: string, alertId: string) =>
    request<ApiAlertDetail>(
      `/runs/${encodeURIComponent(runId)}/alerts/${encodeURIComponent(alertId)}`
    ),
  evidence: (runId: string, block: string) =>
    request<ApiEvidence>(
      `/runs/${encodeURIComponent(runId)}/evidence?block=${encodeURIComponent(block)}`
    ),
  metrics: (runId: string, block?: string) =>
    request<ApiMetric[]>(
      `/runs/${encodeURIComponent(runId)}/metrics${
        block ? `?block=${encodeURIComponent(block)}` : ""
      }`
    ),
  system: (runId: string) =>
    request<ApiSystem>(
      `/runs/${encodeURIComponent(runId)}/system`
    ),
  createJob: (rebuildS1 = false, adminToken?: string) =>
    request<ApiJob>("/jobs", {
      method: "POST",
      body: JSON.stringify({ rebuild_s1: rebuildS1 }),
      headers: adminToken
        ? { "X-V3-Admin-Token": adminToken }
        : undefined
    }),
  jobs: () => request<ApiJob[]>("/jobs"),
  job: (jobId: string) =>
    request<ApiJob>(`/jobs/${encodeURIComponent(jobId)}`)
};

export function apiDocsUrl(): string {
  const prefix = API_BASE.endsWith("/api/v1")
    ? API_BASE.slice(0, -"/api/v1".length)
    : "";
  return `${prefix}/docs`;
}
