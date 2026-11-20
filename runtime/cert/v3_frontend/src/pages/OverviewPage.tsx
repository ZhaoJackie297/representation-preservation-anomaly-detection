import {
  Activity,
  ArrowRight,
  Eye,
  ShieldAlert,
  Users
} from "lucide-react";
import { Link } from "react-router-dom";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  DataError,
  DataLoading,
  MetricCard,
  PageHeading,
  ProgressBar,
  RiskBadge
} from "../components/Common";
import { useWorkspace } from "../context/WorkspaceContext";
import { adaptAlert, adaptBlock } from "../api/adapters";
import { api } from "../api/client";
import { useApiResource } from "../hooks/useApiResource";
import { compactChannelCounts } from "../lib/analytics";
import { chartTheme } from "../lib/chartTheme";
import {
  bandLabels,
  formatDate,
  formatDecimal,
  formatNumber,
  formatPercent,
  splitLabels
} from "../lib/format";

interface TooltipPayload {
  name?: string;
  value?: number | string;
  color?: string;
  dataKey?: string;
  payload?: Record<string, unknown>;
}

function ChartTooltip({
  active,
  payload,
  label
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="tooltip-title">{label}</div>
      {payload.map((item) => (
        <div key={item.dataKey ?? item.name} className="tooltip-row">
          <span>
            <i style={{ background: item.color }} />
            {item.name}
          </span>
          <strong>
            {typeof item.value === "number"
              ? Number.isInteger(item.value)
                ? formatNumber(item.value)
                : formatDecimal(item.value, 3)
              : item.value}
          </strong>
        </div>
      ))}
    </div>
  );
}

export function OverviewPage() {
  const {
    block,
    run,
    theme
  } = useWorkspace();
  const resource = useApiResource(
    () => api.overview(run.run_id, block),
    [run.run_id, block]
  );
  if (resource.loading && !resource.data) {
    return <DataLoading label="正在读取总览接口" />;
  }
  if (resource.error || !resource.data) {
    return (
      <DataError
        message={resource.error ?? "总览数据为空"}
        onRetry={resource.reload}
      />
    );
  }
  const overview = resource.data;
  const blockSummary = adaptBlock(overview.block);
  const weekly = overview.weekly_severity.map((item) => ({
    week: item.week,
    severity: item.severity,
    critical: 0,
    alerts: item.alerts
  }));
  const channels = compactChannelCounts(blockSummary.channels);
  const users = overview.top_users.map((item) => ({
    userId: item.user_id,
    alerts: item.alerts,
    maxSeverity: item.max_severity
  }));
  const latestAlerts = overview.latest_queue.map(adaptAlert);
  const latestDate = latestAlerts[0]?.date ?? "";
  const totalHits = blockSummary.k10.reduce(
    (sum, item) => sum + item.incidentHits,
    0
  );
  const totalIncidents = blockSummary.k10.reduce(
    (sum, item) => sum + item.incidentTotal,
    0
  );
  const meanBalanced =
    blockSummary.k10.reduce(
      (sum, item) => sum + item.balanced,
      0
    ) / Math.max(blockSummary.k10.length, 1);
  const bandOrder = { critical: 0, elevated: 1, watch: 2 };
  const riskPie = overview.risk_distribution
    .map((item) => ({
      key: item.band,
      name:
        bandLabels[item.band as keyof typeof bandLabels] ??
        item.band,
      value: item.alerts
    }))
    .sort(
      (a, b) =>
        (bandOrder[a.key as keyof typeof bandOrder] ?? 9) -
        (bandOrder[b.key as keyof typeof bandOrder] ?? 9)
    );
  const colors = chartTheme(theme);
  const pieColors = [colors.red, colors.amber, colors.blue];
  const weeklyValues = weekly.map((item) => item.severity);
  const weeklyMin = Math.min(...weeklyValues) - 0.08;
  const weeklyMax = Math.max(...weeklyValues) + 0.08;

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="风险运营 / 每日审查"
        title="异常交互风险总览"
        description={`${blockSummary.label} · ${formatDate(
          blockSummary.dateRange.start
        )} 至 ${formatDate(blockSummary.dateRange.end)}`}
        actions={
          <Link className="button primary" to="/alerts">
            查看今日队列
            <ArrowRight size={16} />
          </Link>
        }
      />

      <section className="metric-grid" aria-label="核心指标">
        <MetricCard
          label="累计进入队列"
          value={formatNumber(blockSummary.alerts)}
          note={`${formatNumber(blockSummary.days)} 个告警日`}
          trend="up"
        />
        <MetricCard
          label="Critical 优先级"
          value={formatNumber(blockSummary.bands.critical ?? 0)}
          note="每日排名前 3"
          trend="up"
          accent="red"
        />
        <MetricCard
          label="多视图一致率"
          value={formatPercent(blockSummary.sslAgreementRate)}
          note="至少两项 SSL 证据偏高"
          trend="neutral"
          accent="blue"
        />
        <MetricCard
          label="K=10 均衡效用"
          value={formatPercent(meanBalanced)}
          note={`事件命中 ${totalHits}/${totalIncidents}`}
          trend={meanBalanced >= 0.47 ? "up" : "down"}
          accent="amber"
        />
      </section>

      <section className="overview-main-grid">
        <article className="panel chart-panel">
          <div className="panel-header">
            <div>
              <h2>周均风险强度</h2>
              <p>进入队列用户的 severity 周均值</p>
            </div>
            <span className="panel-chip">
              最新 {formatDecimal(weekly.at(-1)?.severity ?? 0)}
            </span>
          </div>
          <div
            className="chart-shell large"
            role="img"
            aria-label="周均风险强度趋势图"
          >
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={weekly}
                margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
              >
                <defs>
                  <linearGradient
                    id="severityFill"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor={colors.green}
                      stopOpacity={0.24}
                    />
                    <stop
                      offset="100%"
                      stopColor={colors.green}
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  stroke={colors.grid}
                  strokeDasharray="0"
                  vertical={false}
                />
                <XAxis
                  dataKey="week"
                  tickFormatter={(value: string) =>
                    value.slice(0, 7)
                  }
                  tick={{ fill: colors.text, fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  minTickGap={54}
                />
                <YAxis
                  domain={[weeklyMin, weeklyMax]}
                  tickFormatter={(value: number) =>
                    formatDecimal(value, 2)
                  }
                  tick={{ fill: colors.text, fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={48}
                />
                <Tooltip
                  content={<ChartTooltip />}
                  cursor={{
                    stroke: colors.grid,
                    strokeWidth: 1
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="severity"
                  name="平均 severity"
                  stroke={colors.green}
                  strokeWidth={2}
                  fill="url(#severityFill)"
                  activeDot={{
                    r: 4,
                    fill: colors.green,
                    stroke: colors.surface
                  }}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>风险队列构成</h2>
              <p>固定预算内的优先级分布</p>
            </div>
          </div>
          <div
            className="donut-layout"
            role="img"
            aria-label="风险队列优先级构成"
          >
            <div className="donut-chart">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskPie}
                    dataKey="value"
                    nameKey="name"
                    innerRadius="66%"
                    outerRadius="88%"
                    paddingAngle={3}
                    stroke="none"
                    isAnimationActive={false}
                  >
                    {riskPie.map((item, index) => (
                      <Cell
                        key={item.key}
                        fill={pieColors[index]}
                      />
                    ))}
                  </Pie>
                  <Tooltip content={<ChartTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="donut-center">
                <strong>{formatNumber(blockSummary.alerts)}</strong>
                <span>告警总数</span>
              </div>
            </div>
            <div className="legend-list">
              {riskPie.map((item, index) => (
                <div className="legend-row" key={item.key}>
                  <span>
                    <i style={{ background: pieColors[index] }} />
                    {item.name}
                  </span>
                  <strong>{formatNumber(item.value)}</strong>
                </div>
              ))}
            </div>
          </div>
        </article>
      </section>

      <section className="overview-secondary-grid">
        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>主要行为通道</h2>
              <p>最强通道证据的告警数量</p>
            </div>
          </div>
          <div
            className="chart-shell compact"
            role="img"
            aria-label="行为通道分布图"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={channels}
                layout="vertical"
                margin={{ top: 0, right: 12, left: 0, bottom: 0 }}
              >
                <CartesianGrid
                  stroke={colors.grid}
                  horizontal={false}
                />
                <XAxis
                  type="number"
                  hide
                  domain={[0, "dataMax"]}
                />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={42}
                  tick={{ fill: colors.text, fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  content={<ChartTooltip />}
                  cursor={{ fill: "rgba(100,120,105,.06)" }}
                />
                <Bar
                  dataKey="value"
                  name="告警数"
                  fill={colors.green}
                  radius={[0, 5, 5, 0]}
                  barSize={12}
                  isAnimationActive={false}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>高频审查用户</h2>
              <p>按进入队列次数排序</p>
            </div>
            <Users size={17} className="panel-icon" />
          </div>
          <div className="rank-list">
            {users.map((user) => (
              <div className="rank-row" key={user.userId}>
                <div className="rank-avatar">
                  {user.userId.slice(0, 2)}
                </div>
                <div className="rank-content">
                  <div className="rank-top">
                    <strong>{user.userId}</strong>
                    <span>{user.alerts} 次</span>
                  </div>
                  <ProgressBar
                    value={user.alerts}
                    max={users[0]?.alerts ?? 1}
                  />
                </div>
                <span className="rank-severity">
                  {formatDecimal(user.maxSeverity, 2)}
                </span>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>最近完整队列</h2>
            <p>
              {formatDate(latestDate)} · Top-
              {Math.min(10, latestAlerts.length)}
            </p>
          </div>
          <Link className="text-link" to="/alerts">
            全部告警
            <ArrowRight size={15} />
          </Link>
        </div>
        <div className="recent-alerts">
          {latestAlerts.slice(0, 5).map((alert) => (
            <Link
              className="recent-alert"
              key={alert.id}
              to={`/alerts?alert=${encodeURIComponent(alert.id)}`}
            >
              <span className="recent-rank">#{alert.rank}</span>
              <div>
                <strong>{alert.userId}</strong>
                <span>
                  {splitLabels[alert.split] ?? alert.split}
                </span>
              </div>
              <RiskBadge band={alert.band} />
              <span className="recent-source">
                <Eye size={14} />
                {alert.topChannel1}
              </span>
              <strong className="recent-score">
                {formatDecimal(alert.severity)}
              </strong>
            </Link>
          ))}
        </div>
      </section>

      <section className="insight-strip">
        <div className="insight-icon">
          <ShieldAlert size={18} />
        </div>
        <div>
          <strong>审查提示</strong>
          <p>
            Critical 是有限预算内的复核优先级，不是恶意概率。
            自监督主导视图用于解释偏离，不构成因果结论。
          </p>
        </div>
        <Activity size={20} className="insight-spark" />
      </section>
    </div>
  );
}
