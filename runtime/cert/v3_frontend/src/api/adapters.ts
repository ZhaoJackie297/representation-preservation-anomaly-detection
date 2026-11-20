import type {
  AlertRecord,
  AlertSource,
  AuditResult,
  BlockSummary,
  MetricRow,
  RiskBand,
  RunInfo,
  SslView,
  StageRecord,
  TimelineSummary
} from "../types";
import type {
  ApiAlert,
  ApiBlockSummary,
  ApiMetric,
  ApiRunSummary,
  ApiSystem,
  ApiTimeline
} from "./types";

export function adaptBlock(block: ApiBlockSummary): BlockSummary {
  return {
    id: block.id,
    label: block.label,
    alerts: block.alerts,
    days: block.days,
    users: block.users,
    dateRange: {
      start: block.date_start,
      end: block.date_end
    },
    bands: block.bands,
    sources: block.sources,
    splits: block.splits,
    channels: block.channels,
    sslViews: block.ssl_views,
    sslAgreementRate: block.ssl_agreement_rate,
    meanSeverity: block.mean_severity,
    k10: block.k10.map((item) => ({
      split: item.split,
      balanced: item.balanced,
      incidentHits: item.incident_hits,
      incidentTotal: item.incident_total,
      eventRecall3d: item.event_recall_3d,
      precisionAtK: item.precision_at_k
    }))
  };
}

export function adaptAlert(alert: ApiAlert): AlertRecord {
  return {
    id: alert.id,
    block: alert.block,
    split: alert.split,
    date: alert.date,
    rank: alert.rank,
    userId: alert.user_id,
    source: alert.source as AlertSource,
    reasonCode: alert.reason_code,
    band: alert.band as RiskBand,
    severity: alert.severity,
    riskPercentile: alert.risk_percentile,
    tripleQuantile: alert.triple_quantile,
    memory: alert.memory,
    frequency: alert.frequency,
    topChannel1: alert.top_channel_1,
    topChannel1Score: alert.top_channel_1_score,
    topChannel2: alert.top_channel_2,
    topChannel2Score: alert.top_channel_2_score,
    dominantSslView: alert.dominant_ssl_view as SslView,
    sslHighViewCount: alert.ssl_high_view_count,
    sslViewAgreement: alert.ssl_view_agreement,
    sslNearestNormalDistance: alert.ssl_nearest_normal_distance,
    sslVelocity: alert.ssl_velocity,
    sslCurvature: alert.ssl_curvature,
    explanation: alert.explanation,
    sslExplanation: alert.ssl_explanation,
    rawObservedEventCount: alert.raw_observed_event_count,
    rawAfterHoursLogonCount: alert.raw_after_hours_logon_count,
    rawFileCopyCount: alert.raw_file_copy_count,
    rawExternalRecipientCount: alert.raw_external_recipient_count,
    rawExternalHttpCount: alert.raw_external_http_count
  };
}

export function adaptTimeline(
  timeline: ApiTimeline
): TimelineSummary {
  return {
    block: timeline.block,
    userId: timeline.user_id,
    alertCount: timeline.alert_count,
    firstAlertDate: timeline.first_alert_date,
    lastAlertDate: timeline.last_alert_date,
    spanDays: timeline.span_days,
    meanGapDays: timeline.mean_gap_days,
    maxSeverity: timeline.max_severity,
    meanSeverity: timeline.mean_severity,
    lastSeverity: timeline.last_severity,
    trend: timeline.trend as TimelineSummary["trend"],
    firstBand: timeline.first_band as RiskBand,
    lastBand: timeline.last_band as RiskBand,
    maxBand: timeline.max_band as RiskBand,
    dominantTopChannel: timeline.dominant_top_channel
  };
}

export function adaptMetric(metric: ApiMetric): MetricRow {
  return {
    block: metric.block,
    split: metric.split,
    method: metric.method,
    variant: metric.variant,
    k: metric.k,
    q: metric.q,
    balanced: metric.balanced,
    incidentHits: metric.incident_hits,
    incidentTotal: metric.incident_total,
    eventRecall3d: metric.event_recall_3d,
    precisionAtK: metric.precision_at_k,
    falseAlertsPerDay: metric.false_alerts_per_day,
    scenarioMacroRecall3d: metric.scenario_macro_recall_3d
  };
}

export function adaptRun(
  run: ApiRunSummary,
  optimizations: Record<string, boolean> = {}
): RunInfo {
  return {
    id: run.run_id,
    modelVersion: run.model_version,
    modelTitle: run.model_title,
    status: run.status,
    primaryK: run.primary_k,
    createdAt: run.created_at ?? "",
    completedAt: run.completed_at ?? "",
    parameters: {},
    optimizations
  };
}

export function adaptStages(
  system: ApiSystem
): StageRecord[] {
  return system.stages.map((stage) => ({
    key: stage.key,
    title: stage.title,
    status: stage.status as StageRecord["status"],
    durationSeconds: stage.duration_seconds,
    startedAt: stage.started_at,
    completedAt: stage.completed_at
  }));
}

export function adaptAudit(system: ApiSystem): AuditResult {
  const audit = system.audit as unknown as AuditResult;
  return audit;
}
