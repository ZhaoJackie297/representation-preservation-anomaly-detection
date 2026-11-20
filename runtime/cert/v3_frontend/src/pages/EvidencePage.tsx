import { Eye, Fingerprint, Layers3 } from "lucide-react";
import {
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
  PageHeading,
  ProgressBar
} from "../components/Common";
import { useWorkspace } from "../context/WorkspaceContext";
import { adaptAlert, adaptBlock } from "../api/adapters";
import { api } from "../api/client";
import { useApiResource } from "../hooks/useApiResource";
import {
  compactChannelCounts,
  sslViewCounts
} from "../lib/analytics";
import { chartTheme } from "../lib/chartTheme";
import {
  channelLabels,
  formatDecimal,
  formatNumber,
  formatPercent,
  sslViewLabels
} from "../lib/format";

export function EvidencePage() {
  const { block, run, theme } = useWorkspace();
  const resource = useApiResource(
    () => api.evidence(run.run_id, block),
    [run.run_id, block]
  );
  if (resource.loading && !resource.data) {
    return <DataLoading label="正在读取证据接口" />;
  }
  if (resource.error || !resource.data) {
    return (
      <DataError
        message={resource.error ?? "证据数据为空"}
        onRetry={resource.reload}
      />
    );
  }
  const evidence = resource.data;
  const blockSummary = adaptBlock(evidence.block);
  const sslViews = sslViewCounts(blockSummary.sslViews);
  const channels = compactChannelCounts(blockSummary.channels);
  const highViewDistribution = evidence.high_view_distribution.map(
    (item) => ({
      count: item.count,
      label: `${item.count} 项`,
      alerts: item.alerts
    })
  );
  const feature = evidence.representative_alert
    ? adaptAlert(evidence.representative_alert)
    : null;
  const colors = chartTheme(theme);
  const pieColors = [
    colors.purple,
    colors.green,
    colors.blue
  ];
  const highViews = highViewDistribution
    .filter((item) => item.count >= 2)
    .reduce((sum, item) => sum + item.alerts, 0);

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="模型解释 / 证据"
        title="自监督证据分析"
        description={`${blockSummary.label} · 从遮罩、个人流形和通道反事实三个角度解释异常排序。`}
      />

      <section className="evidence-summary-grid">
        <article className="panel evidence-summary-card">
          <div className="summary-icon green">
            <Fingerprint size={19} />
          </div>
          <span>多视图一致</span>
          <strong>{formatPercent(blockSummary.sslAgreementRate)}</strong>
          <p>
            {formatNumber(highViews)} 条告警至少两项 SSL
            证据同时偏高
          </p>
        </article>
        <article className="panel evidence-summary-card">
          <div className="summary-icon purple">
            <Layers3 size={19} />
          </div>
          <span>主导证据</span>
          <strong>{sslViews[0]?.label ?? "—"}</strong>
          <p>{formatNumber(sslViews[0]?.value ?? 0)} 条告警</p>
        </article>
        <article className="panel evidence-summary-card">
          <div className="summary-icon blue">
            <Eye size={19} />
          </div>
          <span>最强行为通道</span>
          <strong>{channels[0]?.label ?? "—"}</strong>
          <p>{formatNumber(channels[0]?.value ?? 0)} 条告警</p>
        </article>
      </section>

      <section className="evidence-grid">
        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>自监督视图构成</h2>
              <p>每条告警的主导表示证据</p>
            </div>
          </div>
          <div
            className="evidence-donut-layout"
            role="img"
            aria-label="自监督视图构成图"
          >
            <div className="donut-chart evidence-donut">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={sslViews}
                    dataKey="value"
                    nameKey="label"
                    innerRadius="64%"
                    outerRadius="88%"
                    paddingAngle={3}
                    stroke="none"
                    isAnimationActive={false}
                  >
                    {sslViews.map((item, index) => (
                      <Cell
                        key={item.key}
                        fill={pieColors[index]}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      borderRadius: 10,
                      borderColor: colors.grid,
                      background: colors.surface,
                      color: colors.text
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="donut-center">
                <strong>
                  {formatNumber(blockSummary.alerts)}
                </strong>
                <span>证据卡</span>
              </div>
            </div>
            <div className="legend-list">
              {sslViews.map((item, index) => (
                <div className="legend-row" key={item.key}>
                  <span>
                    <i
                      style={{
                        background: pieColors[index]
                      }}
                    />
                    {item.label}
                  </span>
                  <strong>{formatNumber(item.value)}</strong>
                </div>
              ))}
            </div>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>高视图数量分布</h2>
              <p>多个自监督视图同时偏离的情况</p>
            </div>
          </div>
          <div
            className="chart-shell compact"
            role="img"
            aria-label="高视图数量分布图"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={highViewDistribution}
                margin={{ top: 12, right: 8, left: -18, bottom: 0 }}
              >
                <CartesianGrid
                  stroke={colors.grid}
                  vertical={false}
                />
                <XAxis
                  dataKey="label"
                  tick={{ fill: colors.text, fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: colors.text, fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: 10,
                    borderColor: colors.grid,
                    background: colors.surface,
                    color: colors.text
                  }}
                />
                <Bar
                  dataKey="alerts"
                  name="告警数"
                  fill={colors.purple}
                  radius={[5, 5, 0, 0]}
                  barSize={36}
                  isAnimationActive={false}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="evidence-grid">
        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>通道反事实贡献</h2>
              <p>移除单一行为通道后的表示变化排序</p>
            </div>
          </div>
          <div className="channel-evidence-list">
            {channels.map((item, index) => (
              <div className="channel-evidence-row" key={item.key}>
                <span className="channel-index">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div>
                  <div className="rank-top">
                    <strong>{item.label}</strong>
                    <span>{formatNumber(item.value)} 条</span>
                  </div>
                  <ProgressBar
                    value={item.value}
                    max={channels[0]?.value ?? 1}
                  />
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="panel feature-evidence">
          <div className="panel-header">
            <div>
              <h2>代表性告警证据</h2>
              <p>当前数据块中 severity 最高的告警</p>
            </div>
            <span className="panel-chip">{feature?.userId}</span>
          </div>
          {feature ? (
            <>
              <div className="feature-evidence-score">
                <div>
                  <span>severity</span>
                  <strong>
                    {formatDecimal(feature.severity)}
                  </strong>
                </div>
                <div>
                  <span>风险百分位</span>
                  <strong>
                    {formatPercent(feature.riskPercentile)}
                  </strong>
                </div>
                <div>
                  <span>SSL 视图</span>
                  <strong>
                    {
                      sslViewLabels[
                        feature.dominantSslView
                      ]
                    }
                  </strong>
                </div>
              </div>
              <div className="feature-channel-pair">
                <span>
                  {channelLabels[feature.topChannel1] ??
                    feature.topChannel1}
                </span>
                <span>+</span>
                <span>
                  {channelLabels[feature.topChannel2] ??
                    feature.topChannel2}
                </span>
              </div>
              <p className="explanation-box">
                {feature.sslExplanation}
              </p>
            </>
          ) : null}
        </article>
      </section>

      <section className="explanation-note">
        自监督证据描述模型看到了什么，不表示某个行为导致了异常。
        主导视图和通道排序用于人工审查，不进入因果判定。
      </section>
    </div>
  );
}
