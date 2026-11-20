import {
  ArrowDownRight,
  ArrowUpRight,
  LoaderCircle,
  Minus
} from "lucide-react";
import type { ReactNode } from "react";
import type {
  AlertSource,
  RiskBand
} from "../types";
import {
  bandLabels,
  formatNumber,
  formatPercent,
  sourceLabels
} from "../lib/format";

interface MetricCardProps {
  label: string;
  value: string;
  note: string;
  trend?: "up" | "down" | "neutral";
  accent?: "green" | "red" | "amber" | "blue";
}

export function MetricCard({
  label,
  value,
  note,
  trend = "neutral",
  accent = "green"
}: MetricCardProps) {
  const TrendIcon =
    trend === "up"
      ? ArrowUpRight
      : trend === "down"
        ? ArrowDownRight
        : Minus;
  return (
    <article className={`metric-card accent-${accent}`}>
      <div className="metric-card-head">
        <span>{label}</span>
        <span className={`metric-trend ${trend}`}>
          <TrendIcon size={14} />
        </span>
      </div>
      <div className="metric-card-value">{value}</div>
      <div className="metric-card-note">{note}</div>
    </article>
  );
}

export function RiskBadge({ band }: { band: RiskBand }) {
  return (
    <span className={`risk-badge ${band}`}>
      {bandLabels[band]}
    </span>
  );
}

export function SourceBadge({
  source
}: {
  source: AlertSource;
}) {
  return (
    <span className={`source-badge ${source}`}>
      {sourceLabels[source]}
    </span>
  );
}

export function PageHeading({
  eyebrow,
  title,
  description,
  actions
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </div>
  );
}

export function ProgressBar({
  value,
  max,
  className = ""
}: {
  value: number;
  max: number;
  className?: string;
}) {
  return (
    <div
      className={`progress ${className}`}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={value}
    >
      <span
        style={{
          width: `${Math.max(2, (value / Math.max(max, 1)) * 100)}%`
        }}
      />
    </div>
  );
}

export function SplitMetric({
  label,
  value,
  detail
}: {
  label: string;
  value: number;
  detail: string;
}) {
  return (
    <div className="split-metric">
      <div className="split-metric-head">
        <span>{label}</span>
        <strong>{formatPercent(value)}</strong>
      </div>
      <ProgressBar value={value} max={1} />
      <div className="split-metric-detail">{detail}</div>
    </div>
  );
}

export function EmptyState({
  title,
  description
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="empty-state">
      <div className="empty-orbit" aria-hidden="true">
        <span />
        <span />
      </div>
      <strong>{title}</strong>
      <p>{description}</p>
    </div>
  );
}

export function SummaryStat({
  label,
  value
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="summary-stat">
      <span>{label}</span>
      <strong>
        {typeof value === "number"
          ? formatNumber(value)
          : value}
      </strong>
    </div>
  );
}

export function DataLoading({
  label = "正在读取模型接口"
}: {
  label?: string;
}) {
  return (
    <div className="data-state" role="status">
      <LoaderCircle className="spin" size={20} />
      <strong>{label}</strong>
      <span>结果按需从 V3 API 获取，不在前端保存。</span>
    </div>
  );
}

export function DataError({
  message,
  onRetry
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="data-state data-error" role="alert">
      <strong>接口请求失败</strong>
      <span>{message}</span>
      <button className="button secondary" type="button" onClick={onRetry}>
        重试
      </button>
    </div>
  );
}
