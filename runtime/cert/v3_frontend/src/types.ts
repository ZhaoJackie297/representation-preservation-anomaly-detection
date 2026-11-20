export type RiskBand = "critical" | "elevated" | "watch";
export type AlertSource = "ts_main" | "iforest_rescue" | "ts_fallback";
export type SslView =
  | "channel_counterfactual"
  | "personal_manifold"
  | "masked_reconstruction";

export interface K10Metric {
  split: string;
  balanced: number;
  incidentHits: number;
  incidentTotal: number;
  eventRecall3d: number;
  precisionAtK: number;
}

export interface BlockSummary {
  id: string;
  label: string;
  alerts: number;
  days: number;
  users: number;
  dateRange: {
    start: string;
    end: string;
  };
  bands: Record<string, number>;
  sources: Record<string, number>;
  splits: Record<string, number>;
  channels: Record<string, number>;
  sslViews: Record<string, number>;
  sslAgreementRate: number;
  meanSeverity: number;
  k10: K10Metric[];
}

export interface AlertRecord {
  id: string;
  block: string;
  split: string;
  date: string;
  rank: number;
  userId: string;
  source: AlertSource;
  reasonCode: string;
  band: RiskBand;
  severity: number;
  riskPercentile: number;
  tripleQuantile: number;
  memory: number;
  frequency: number;
  topChannel1: string;
  topChannel1Score: number;
  topChannel2: string;
  topChannel2Score: number;
  dominantSslView: SslView;
  sslHighViewCount: number;
  sslViewAgreement: boolean;
  sslNearestNormalDistance: number;
  sslVelocity: number;
  sslCurvature: number;
  explanation: string;
  sslExplanation: string;
  rawObservedEventCount: number | null;
  rawAfterHoursLogonCount: number | null;
  rawFileCopyCount: number | null;
  rawExternalRecipientCount: number | null;
  rawExternalHttpCount: number | null;
}

export interface TimelineSummary {
  block: string;
  userId: string;
  alertCount: number;
  firstAlertDate: string;
  lastAlertDate: string;
  spanDays: number;
  meanGapDays: number;
  maxSeverity: number;
  meanSeverity: number;
  lastSeverity: number;
  trend: "rising" | "falling" | "stable";
  firstBand: RiskBand;
  lastBand: RiskBand;
  maxBand: RiskBand;
  dominantTopChannel: string;
}

export interface MetricRow {
  block: string;
  split: string;
  method: string;
  variant: string;
  k: number;
  q: number;
  balanced: number;
  incidentHits: number;
  incidentTotal: number;
  eventRecall3d: number;
  precisionAtK: number;
  falseAlertsPerDay: number;
  scenarioMacroRecall3d: number;
}

export interface StageRecord {
  key: string;
  title: string;
  status: "completed" | "failed" | "running";
  durationSeconds: number;
  startedAt: string;
  completedAt: string;
}

export interface AuditResult {
  model_version: string;
  audit_pass: boolean;
  rows: number;
  blocks: number;
  timelines: number;
  duplicate_queue_keys: number;
  rank_problems: number;
  illegal_sources: string[];
  label_columns_in_model_queue: string[];
  missing_required_fields: string[];
  missing_cells: Record<string, number>;
  s5_card_integrity_pass: boolean;
  s5_ssl_integrity_pass: boolean;
  timeline_card_consistency: boolean;
  labels_written_to_model_outputs: boolean;
  metrics_blocks_at_k10: number;
  limitations: string[];
}

export interface RunInfo {
  id: string;
  modelVersion: string;
  modelTitle: string;
  status: string;
  primaryK: number;
  createdAt: string;
  completedAt: string;
  parameters: Record<string, unknown>;
  optimizations: Record<string, boolean>;
}

export interface V3Data {
  schemaVersion: string;
  generatedAt: string;
  run: RunInfo;
  blocks: BlockSummary[];
  alerts: AlertRecord[];
  timelines: TimelineSummary[];
  metrics: MetricRow[];
  stages: StageRecord[];
  audit: AuditResult;
}
