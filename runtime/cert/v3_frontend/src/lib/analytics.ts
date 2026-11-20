import type {
  AlertRecord,
  BlockSummary,
  TimelineSummary
} from "../types";
import { channelLabels, sslViewLabels } from "./format";

export interface WeeklySeverityPoint {
  week: string;
  severity: number;
  critical: number;
  alerts: number;
}

export interface DailyAlertPoint {
  date: string;
  critical: number;
  elevated: number;
  watch: number;
  total: number;
}

export function weeklySeverity(
  alerts: AlertRecord[]
): WeeklySeverityPoint[] {
  const groups = new Map<
    string,
    { severity: number[]; critical: number; alerts: number }
  >();
  for (const alert of alerts) {
    const day = new Date(`${alert.date}T00:00:00Z`);
    const offset = (day.getUTCDay() + 6) % 7;
    day.setUTCDate(day.getUTCDate() - offset);
    const week = day.toISOString().slice(0, 10);
    const current = groups.get(week) ?? {
      severity: [],
      critical: 0,
      alerts: 0
    };
    current.severity.push(alert.severity);
    current.critical += alert.band === "critical" ? 1 : 0;
    current.alerts += 1;
    groups.set(week, current);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([week, value]) => ({
      week,
      severity:
        value.severity.reduce((sum, item) => sum + item, 0) /
        value.severity.length,
      critical: value.critical,
      alerts: value.alerts
    }));
}

export function dailyBandCounts(
  alerts: AlertRecord[]
): DailyAlertPoint[] {
  const groups = new Map<
    string,
    { critical: number; elevated: number; watch: number }
  >();
  for (const alert of alerts) {
    const current = groups.get(alert.date) ?? {
      critical: 0,
      elevated: 0,
      watch: 0
    };
    current[alert.band] += 1;
    groups.set(alert.date, current);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, value]) => ({
      date,
      ...value,
      total: value.critical + value.elevated + value.watch
    }));
}

export function topUsers(
  alerts: AlertRecord[],
  limit = 8
): Array<{ userId: string; alerts: number; maxSeverity: number }> {
  const groups = new Map<
    string,
    { alerts: number; maxSeverity: number }
  >();
  for (const alert of alerts) {
    const current = groups.get(alert.userId) ?? {
      alerts: 0,
      maxSeverity: 0
    };
    current.alerts += 1;
    current.maxSeverity = Math.max(
      current.maxSeverity,
      alert.severity
    );
    groups.set(alert.userId, current);
  }
  return [...groups.entries()]
    .map(([userId, value]) => ({ userId, ...value }))
    .sort(
      (a, b) =>
        b.alerts - a.alerts ||
        b.maxSeverity - a.maxSeverity
    )
    .slice(0, limit);
}

export function blockTrend(
  block: BlockSummary
): "稳定" | "上升" | "下降" {
  if (block.k10.length < 2) return "稳定";
  const values = block.k10.map((item) => item.balanced);
  const first = values[0];
  const last = values[values.length - 1];
  if (last - first > 0.01) return "上升";
  if (first - last > 0.01) return "下降";
  return "稳定";
}

export function compactChannelCounts(
  values: Record<string, number>
): Array<{ key: string; label: string; value: number }> {
  return Object.entries(values)
    .map(([key, value]) => ({
      key,
      label: channelLabels[key] ?? key,
      value
    }))
    .sort((a, b) => b.value - a.value);
}

export function sslViewCounts(
  values: Record<string, number>
): Array<{ key: string; label: string; value: number }> {
  return Object.entries(values)
    .map(([key, value]) => ({
      key,
      label:
        sslViewLabels[
          key as keyof typeof sslViewLabels
        ] ?? key,
      value
    }))
    .sort((a, b) => b.value - a.value);
}

export function userTimeline(
  timelines: TimelineSummary[],
  block: string,
  userId: string
): TimelineSummary | undefined {
  return timelines.find(
    (item) => item.block === block && item.userId === userId
  );
}
