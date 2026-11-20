import {
  ChevronLeft,
  ChevronRight,
  Download,
  LoaderCircle,
  Search,
  SlidersHorizontal,
  X
} from "lucide-react";
import {
  useEffect,
  useState,
  type ReactNode
} from "react";
import { useSearchParams } from "react-router-dom";
import { adaptAlert, adaptTimeline } from "../api/adapters";
import { api, type AlertQuery } from "../api/client";
import {
  DataError,
  DataLoading,
  EmptyState,
  PageHeading,
  RiskBadge,
  SourceBadge
} from "../components/Common";
import { useWorkspace } from "../context/WorkspaceContext";
import { useApiResource } from "../hooks/useApiResource";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import {
  bandLabels,
  channelLabels,
  formatDate,
  formatDecimal,
  formatNumber,
  formatPercent,
  sourceLabels,
  splitLabels,
  sslViewLabels
} from "../lib/format";
import type {
  AlertRecord,
  RiskBand,
  TimelineSummary
} from "../types";

const PAGE_SIZES = [15, 25, 50];

export function AlertsPage() {
  const {
    block,
    blockSummary,
    run
  } = useWorkspace();
  const [params, setParams] = useSearchParams();
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 320);
  const [band, setBand] = useState<"all" | RiskBand>("all");
  const [source, setSource] = useState("all");
  const [split, setSplit] = useState("all");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(15);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const selectedId = params.get("alert");

  const request: AlertQuery = {
    block,
    page,
    pageSize,
    band: band === "all" ? undefined : band,
    source: source === "all" ? undefined : source,
    split: split === "all" ? undefined : split,
    q: debouncedQuery.trim() || undefined,
    startDate: startDate || undefined,
    endDate: endDate || undefined
  };

  const resource = useApiResource(
    () => api.alerts(run.run_id, request),
    [
      run.run_id,
      block,
      page,
      pageSize,
      band,
      source,
      split,
      debouncedQuery,
      startDate,
      endDate
    ]
  );
  const detailResource = useApiResource(
    () =>
      selectedId
        ? api.alert(run.run_id, selectedId)
        : Promise.resolve(null),
    [run.run_id, selectedId]
  );

  useEffect(() => {
    setPage(1);
  }, [
    block,
    band,
    source,
    split,
    debouncedQuery,
    startDate,
    endDate,
    pageSize,
    run.run_id
  ]);

  const alertPage = resource.data;
  const alerts = alertPage?.items.map(adaptAlert) ?? [];
  const pagination = alertPage?.pagination;

  function setSelected(id: string | null) {
    const next = new URLSearchParams(params);
    if (id) next.set("alert", id);
    else next.delete("alert");
    setParams(next, { replace: true });
  }

  function resetFilters() {
    setQuery("");
    setBand("all");
    setSource("all");
    setSplit("all");
    setStartDate("");
    setEndDate("");
  }

  async function exportCsv() {
    setExporting(true);
    setExportError(null);
    try {
      const first = await api.alerts(run.run_id, {
        ...request,
        page: 1,
        pageSize: 100
      });
      const rows = [...first.items];
      for (let current = 2; current <= first.pagination.pages; current += 1) {
        const next = await api.alerts(run.run_id, {
          ...request,
          page: current,
          pageSize: 100
        });
        rows.push(...next.items);
      }
      const header = [
        "date",
        "rank",
        "user_id",
        "source",
        "risk_band",
        "severity",
        "risk_percentile",
        "top_channel_1",
        "top_channel_2",
        "ssl_dominant_view"
      ];
      const records = rows.map((alert) => [
        alert.date,
        alert.rank,
        alert.user_id,
        alert.source,
        alert.band,
        alert.severity,
        alert.risk_percentile,
        alert.top_channel_1,
        alert.top_channel_2,
        alert.dominant_ssl_view
      ]);
      const csv = [header, ...records]
        .map((row) =>
          row
            .map(
              (cell) =>
                `"${String(cell).replaceAll('"', '""')}"`
            )
            .join(",")
        )
        .join("\n");
      const blob = new Blob([`\uFEFF${csv}`], {
        type: "text/csv;charset=utf-8"
      });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `v3-alerts-${block}-${new Date()
        .toISOString()
        .slice(0, 10)}.csv`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setExportError(
        error instanceof Error ? error.message : "导出失败"
      );
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="风险运营 / 队列"
        title="每日告警队列"
        description={`${blockSummary.label} · 固定预算 K=${run.primary_k} · 完整告警 ${formatNumber(
          blockSummary.alerts
        )} 条 · 分页由 API 执行`}
        actions={
          <button
            className="button secondary"
            type="button"
            onClick={() => void exportCsv()}
            disabled={exporting}
          >
            {exporting ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Download size={16} />
            )}
            {exporting ? "正在导出" : "导出筛选结果"}
          </button>
        }
      />

      <section className="filter-panel" aria-label="告警筛选">
        <div className="filter-primary">
          <label className="search-field">
            <Search size={17} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索用户 ID 或告警 ID"
              aria-label="搜索用户或告警"
            />
          </label>
          <div className="filter-summary">
            <SlidersHorizontal size={15} />
            共 {formatNumber(pagination?.total ?? 0)} 条匹配
          </div>
        </div>
        <div className="filter-grid">
          <label className="field">
            <span>风险带</span>
            <select
              value={band}
              onChange={(event) =>
                setBand(event.target.value as "all" | RiskBand)
              }
            >
              <option value="all">全部风险带</option>
              {Object.entries(bandLabels).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>队列来源</span>
            <select
              value={source}
              onChange={(event) => setSource(event.target.value)}
            >
              <option value="all">全部来源</option>
              {Object.entries(sourceLabels).map(
                ([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                )
              )}
            </select>
          </label>
          <label className="field">
            <span>数据划分</span>
            <select
              value={split}
              onChange={(event) => setSplit(event.target.value)}
            >
              <option value="all">全部划分</option>
              {Object.keys(blockSummary.splits).map((value) => (
                <option key={value} value={value}>
                  {splitLabels[value] ?? value}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>开始日期</span>
            <input
              type="date"
              value={startDate}
              min={blockSummary.dateRange.start}
              max={endDate || blockSummary.dateRange.end}
              onChange={(event) => setStartDate(event.target.value)}
            />
          </label>
          <label className="field">
            <span>结束日期</span>
            <input
              type="date"
              value={endDate}
              min={startDate || blockSummary.dateRange.start}
              max={blockSummary.dateRange.end}
              onChange={(event) => setEndDate(event.target.value)}
            />
          </label>
          <button
            className="button ghost filter-reset"
            type="button"
            onClick={resetFilters}
          >
            重置筛选
          </button>
        </div>
        {exportError ? (
          <div className="inline-error" role="alert">
            {exportError}
          </div>
        ) : null}
      </section>

      {resource.loading && !resource.data ? (
        <DataLoading label="正在读取告警接口" />
      ) : resource.error ? (
        <DataError
          message={resource.error}
          onRetry={resource.reload}
        />
      ) : (
        <section className="panel table-panel">
          <div className="panel-header table-header">
            <div>
              <h2>告警列表</h2>
              <p>
                服务端分页与筛选 · 点击行读取单条证据接口
              </p>
            </div>
            <label className="inline-select">
              每页
              <select
                value={pageSize}
                onChange={(event) =>
                  setPageSize(Number(event.target.value))
                }
              >
                {PAGE_SIZES.map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {alerts.length ? (
            <>
              <div className="table-scroll">
                <table className="alert-table">
                  <thead>
                    <tr>
                      <th>日期 / 排名</th>
                      <th>用户</th>
                      <th>来源</th>
                      <th>风险带</th>
                      <th>Severity</th>
                      <th>风险百分位</th>
                      <th>最强通道</th>
                      <th>SSL 证据</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alerts.map((alert) => (
                      <tr
                        key={alert.id}
                        className={
                          selectedId === alert.id ? "selected" : ""
                        }
                        tabIndex={0}
                        onClick={() => setSelected(alert.id)}
                        onKeyDown={(event) => {
                          if (
                            event.key === "Enter" ||
                            event.key === " "
                          ) {
                            event.preventDefault();
                            setSelected(alert.id);
                          }
                        }}
                      >
                        <td>
                          <div className="date-cell">
                            <strong>{formatDate(alert.date)}</strong>
                            <span>
                              #{alert.rank} ·{" "}
                              {splitLabels[alert.split] ?? alert.split}
                            </span>
                          </div>
                        </td>
                        <td>
                          <span className="user-link">
                            {alert.userId}
                          </span>
                        </td>
                        <td>
                          <SourceBadge source={alert.source} />
                        </td>
                        <td>
                          <RiskBadge band={alert.band} />
                        </td>
                        <td>
                          <strong className="mono">
                            {formatDecimal(alert.severity)}
                          </strong>
                        </td>
                        <td className="mono">
                          {formatPercent(alert.riskPercentile)}
                        </td>
                        <td>
                          <div className="channel-pair">
                            <span>
                              {channelLabels[alert.topChannel1] ??
                                alert.topChannel1}
                            </span>
                            <span>
                              {channelLabels[alert.topChannel2] ??
                                alert.topChannel2}
                            </span>
                          </div>
                        </td>
                        <td>
                          <div className="evidence-cell">
                            <span>
                              {sslViewLabels[alert.dominantSslView]}
                            </span>
                            <small>
                              {alert.sslHighViewCount}/3 视图偏高
                            </small>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mobile-alert-list">
                {alerts.map((alert) => (
                  <button
                    className="mobile-alert-card"
                    type="button"
                    key={alert.id}
                    onClick={() => setSelected(alert.id)}
                  >
                    <div className="mobile-alert-top">
                      <span>
                        {formatDate(alert.date)} · #{alert.rank}
                      </span>
                      <RiskBadge band={alert.band} />
                    </div>
                    <div className="mobile-alert-body">
                      <strong>{alert.userId}</strong>
                      <span>{formatDecimal(alert.severity)}</span>
                    </div>
                    <div className="mobile-alert-meta">
                      <SourceBadge source={alert.source} />
                      <span>
                        {channelLabels[alert.topChannel1] ??
                          alert.topChannel1}{" "}
                        /{" "}
                        {channelLabels[alert.topChannel2] ??
                          alert.topChannel2}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </>
          ) : (
            <EmptyState
              title="没有匹配的告警"
              description="调整筛选条件后重试。"
            />
          )}

          <div className="pagination">
            <span>
              第 {pagination?.page ?? 1} /{" "}
              {pagination?.pages ?? 1} 页
            </span>
            <div>
              <button
                className="icon-button bordered"
                type="button"
                aria-label="上一页"
                disabled={(pagination?.page ?? 1) <= 1}
                onClick={() => setPage((value) => value - 1)}
              >
                <ChevronLeft size={17} />
              </button>
              <button
                className="icon-button bordered"
                type="button"
                aria-label="下一页"
                disabled={
                  (pagination?.page ?? 1) >=
                  (pagination?.pages ?? 1)
                }
                onClick={() => setPage((value) => value + 1)}
              >
                <ChevronRight size={17} />
              </button>
            </div>
          </div>
        </section>
      )}

      {selectedId ? (
        detailResource.loading ? (
          <DrawerShell onClose={() => setSelected(null)}>
            <DataLoading label="正在读取告警详情" />
          </DrawerShell>
        ) : detailResource.error || !detailResource.data ? (
          <DrawerShell onClose={() => setSelected(null)}>
            <DataError
              message={detailResource.error ?? "告警为空"}
              onRetry={detailResource.reload}
            />
          </DrawerShell>
        ) : (
          <AlertDrawer
            alert={adaptAlert(detailResource.data.alert)}
            timeline={
              detailResource.data.timeline
                ? adaptTimeline(detailResource.data.timeline)
                : undefined
            }
            onClose={() => setSelected(null)}
          />
        )
      ) : null}
    </div>
  );
}

function DrawerShell({
  onClose,
  children
}: {
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <>
      <button
        className="drawer-scrim"
        type="button"
        aria-label="关闭告警详情"
        onClick={onClose}
      />
      <aside
        className="alert-drawer"
        role="dialog"
        aria-modal="true"
      >
        <div className="drawer-header">
          <div>
            <div className="eyebrow">告警详情</div>
            <h2>加载中</h2>
          </div>
          <button
            className="icon-button bordered"
            type="button"
            aria-label="关闭"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>
        {children}
      </aside>
    </>
  );
}

function AlertDrawer({
  alert,
  timeline,
  onClose
}: {
  alert: AlertRecord;
  timeline: TimelineSummary | undefined;
  onClose: () => void;
}) {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <>
      <button
        className="drawer-scrim"
        type="button"
        aria-label="关闭告警详情"
        onClick={onClose}
      />
      <aside
        className="alert-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="alert-drawer-title"
      >
        <div className="drawer-header">
          <div>
            <div className="eyebrow">告警详情</div>
            <h2 id="alert-drawer-title">{alert.userId}</h2>
            <p>
              {formatDate(alert.date)} · 队列第 {alert.rank} 名
            </p>
          </div>
          <button
            className="icon-button bordered"
            type="button"
            aria-label="关闭"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>

        <div className="drawer-hero">
          <div>
            <span>severity</span>
            <strong>{formatDecimal(alert.severity)}</strong>
          </div>
          <RiskBadge band={alert.band} />
          <div>
            <span>风险百分位</span>
            <strong>{formatPercent(alert.riskPercentile)}</strong>
          </div>
        </div>

        <div className="drawer-section">
          <h3>队列判定</h3>
          <div className="detail-matrix">
            <DetailItem
              label="来源"
              value={sourceLabels[alert.source]}
            />
            <DetailItem label="原因码" value={alert.reasonCode} />
            <DetailItem
              label="最强通道"
              value={`${
                channelLabels[alert.topChannel1] ??
                alert.topChannel1
              } / ${
                channelLabels[alert.topChannel2] ??
                alert.topChannel2
              }`}
            />
            <DetailItem
              label="主导 SSL 视图"
              value={sslViewLabels[alert.dominantSslView]}
            />
          </div>
          <p className="explanation-box">{alert.explanation}</p>
        </div>

        <div className="drawer-section">
          <h3>自监督证据</h3>
          <div className="ssl-evidence">
            <div className="ssl-orb">
              <span>{alert.sslHighViewCount}</span>
              <small>/ 3</small>
            </div>
            <div>
              <strong>
                {alert.sslViewAgreement
                  ? "多视图一致"
                  : "部分视图偏高"}
              </strong>
              <p>{alert.sslExplanation}</p>
            </div>
          </div>
          <div className="evidence-bars">
            <EvidenceBar
              label="最近正常距离"
              value={alert.sslNearestNormalDistance}
              max={Math.max(alert.sslNearestNormalDistance, 0.25)}
            />
            <EvidenceBar
              label="表示速度"
              value={alert.sslVelocity}
              max={Math.max(alert.sslVelocity, 0.1)}
            />
            <EvidenceBar
              label="表示曲率"
              value={alert.sslCurvature}
              max={Math.max(alert.sslCurvature, 0.1)}
            />
          </div>
        </div>

        <div className="drawer-section">
          <h3>用户风险时间线</h3>
          {timeline ? (
            <div className="timeline-summary">
              <div>
                <span>累计进入队列</span>
                <strong>{timeline.alertCount} 次</strong>
              </div>
              <div>
                <span>历史最大 severity</span>
                <strong>{formatDecimal(timeline.maxSeverity)}</strong>
              </div>
              <div>
                <span>首次 / 最近</span>
                <strong>
                  {formatDate(timeline.firstAlertDate)} →{" "}
                  {formatDate(timeline.lastAlertDate)}
                </strong>
              </div>
              <div>
                <span>趋势</span>
                <strong>
                  {timeline.trend === "rising"
                    ? "上升"
                    : timeline.trend === "falling"
                      ? "下降"
                      : "稳定"}
                </strong>
              </div>
            </div>
          ) : (
            <p className="muted-copy">暂无用户时间线摘要。</p>
          )}
        </div>

        <div className="drawer-section">
          <h3>当日原始行为摘要</h3>
          <div className="detail-matrix">
            <DetailItem
              label="观测事件"
              value={numberOrDash(alert.rawObservedEventCount)}
            />
            <DetailItem
              label="非工作时间登录"
              value={numberOrDash(alert.rawAfterHoursLogonCount)}
            />
            <DetailItem
              label="文件复制"
              value={numberOrDash(alert.rawFileCopyCount)}
            />
            <DetailItem
              label="外部邮件收件"
              value={numberOrDash(
                alert.rawExternalRecipientCount
              )}
            />
          </div>
        </div>
      </aside>
    </>
  );
}

function DetailItem({
  label,
  value
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="detail-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EvidenceBar({
  label,
  value,
  max
}: {
  label: string;
  value: number;
  max: number;
}) {
  return (
    <div className="evidence-bar-row">
      <div>
        <span>{label}</span>
        <strong>{formatDecimal(value, 3)}</strong>
      </div>
      <div className="progress">
        <span
          style={{
            width: `${Math.max(
              2,
              Math.min(100, (value / max) * 100)
            )}%`
          }}
        />
      </div>
    </div>
  );
}

function numberOrDash(value: number | null): string {
  return value == null ? "—" : formatNumber(value);
}
