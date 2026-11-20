import type {
  AlertSource,
  RiskBand,
  SslView
} from "../types";

const numberFormatter = new Intl.NumberFormat("zh-CN");
const compactFormatter = new Intl.NumberFormat("zh-CN", {
  notation: "compact",
  maximumFractionDigits: 1
});
const dateFormatter = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit"
});
const dateTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit"
});

export const channelLabels: Record<string, string> = {
  logon: "登录",
  device: "设备",
  file: "文件",
  email: "邮件",
  http: "网页"
};

export const sourceLabels: Record<AlertSource, string> = {
  ts_main: "TS 主路",
  iforest_rescue: "IForest 补位",
  ts_fallback: "TS 回退"
};

export const bandLabels: Record<RiskBand, string> = {
  critical: "Critical",
  elevated: "Elevated",
  watch: "Watch"
};

export const sslViewLabels: Record<SslView, string> = {
  channel_counterfactual: "通道反事实",
  personal_manifold: "个人正常流形",
  masked_reconstruction: "遮罩重建"
};

export const splitLabels: Record<string, string> = {
  validation: "验证",
  test: "测试",
  external_test: "外部测试"
};

export const blockLabels: Record<string, string> = {
  r42_original: "r4.2 原模型",
  r42_retrained: "r4.2 重训",
  r52_external: "r5.2 外部"
};

export function formatNumber(value: number): string {
  return numberFormatter.format(value);
}

export function formatCompact(value: number): string {
  return compactFormatter.format(value);
}

export function formatPercent(
  value: number,
  digits = 1
): string {
  return new Intl.NumberFormat("zh-CN", {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  }).format(value);
}

export function formatDecimal(
  value: number,
  digits = 3
): string {
  return new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  }).format(value);
}

export function formatDate(value: string): string {
  return dateFormatter.format(new Date(`${value}T00:00:00`));
}

export function formatDateTime(value: string): string {
  return dateTimeFormatter.format(new Date(value));
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)} 秒`;
  }
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes} 分 ${rest} 秒`;
}

export function titleCase(value: string): string {
  return value.replaceAll("_", " ");
}
