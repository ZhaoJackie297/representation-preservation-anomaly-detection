export interface ApiHealth {
  status: "ok";
  model_version: string;
  runs_root: string;
}

export interface ApiModelInfo {
  model_version: string;
  name: string;
  primary_k: number;
  task: string;
  interfaces: string[];
  limitations: string[];
}

export interface ApiRunSummary {
  run_id: string;
  model_version: string;
  model_title: string;
  status: string;
  audit_pass: boolean;
  created_at: string | null;
  completed_at: string | null;
  primary_k: number;
  model_card_available: boolean;
  dashboard_available: boolean;
}

export interface ApiBlockSummary {
  id: string;
  label: string;
  alerts: number;
  days: number;
  users: number;
  date_start: string;
  date_end: string;
  bands: Record<string, number>;
  sources: Record<string, number>;
  splits: Record<string, number>;
  channels: Record<string, number>;
  ssl_views: Record<string, number>;
  ssl_agreement_rate: number;
  mean_severity: number;
  k10: Array<{
    split: string;
    balanced: number;
    incident_hits: number;
    incident_total: number;
    event_recall_3d: number;
    precision_at_k: number;
  }>;
}

export interface ApiAlert {
  id: string;
  block: string;
  split: string;
  date: string;
  rank: number;
  user_id: string;
  source: string;
  reason_code: string;
  band: string;
  severity: number;
  risk_percentile: number;
  triple_quantile: number;
  memory: number;
  frequency: number;
  top_channel_1: string;
  top_channel_1_score: number;
  top_channel_2: string;
  top_channel_2_score: number;
  dominant_ssl_view: string;
  ssl_high_view_count: number;
  ssl_view_agreement: boolean;
  ssl_nearest_normal_distance: number;
  ssl_velocity: number;
  ssl_curvature: number;
  explanation: string;
  ssl_explanation: string;
  raw_observed_event_count: number | null;
  raw_after_hours_logon_count: number | null;
  raw_file_copy_count: number | null;
  raw_external_recipient_count: number | null;
  raw_external_http_count: number | null;
}

export interface ApiPagination {
  page: number;
  page_size: number;
  total: number;
  pages: number;
}

export interface ApiAlertPage {
  items: ApiAlert[];
  pagination: ApiPagination;
}

export interface ApiTimeline {
  block: string;
  user_id: string;
  alert_count: number;
  first_alert_date: string;
  last_alert_date: string;
  span_days: number;
  mean_gap_days: number;
  max_severity: number;
  mean_severity: number;
  last_severity: number;
  trend: string;
  first_band: string;
  last_band: string;
  max_band: string;
  dominant_top_channel: string;
}

export interface ApiAlertDetail {
  alert: ApiAlert;
  timeline: ApiTimeline | null;
}

export interface ApiOverview {
  run: ApiRunSummary;
  block: ApiBlockSummary;
  weekly_severity: Array<{
    week: string;
    severity: number;
    alerts: number;
  }>;
  risk_distribution: Array<{
    band: string;
    alerts: number;
  }>;
  top_users: Array<{
    user_id: string;
    alerts: number;
    max_severity: number;
  }>;
  latest_queue: ApiAlert[];
}

export interface ApiEvidence {
  block: ApiBlockSummary;
  high_view_distribution: Array<{
    count: number;
    alerts: number;
  }>;
  representative_alert: ApiAlert | null;
}

export interface ApiMetric {
  block: string;
  split: string;
  method: string;
  variant: string;
  k: number;
  q: number;
  balanced: number;
  incident_hits: number;
  incident_total: number;
  event_recall_3d: number;
  precision_at_k: number;
  false_alerts_per_day: number;
  scenario_macro_recall_3d: number;
}

export interface ApiSystem {
  run: ApiRunSummary;
  stages: Array<{
    key: string;
    title: string;
    status: string;
    duration_seconds: number;
    started_at: string;
    completed_at: string;
  }>;
  audit: Record<string, unknown>;
  optimizations: Record<string, boolean>;
}

export interface ApiJob {
  job_id: string;
  status: string;
  rebuild_s1: boolean;
  run_id: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}
