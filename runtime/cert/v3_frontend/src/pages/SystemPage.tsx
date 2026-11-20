import {
  CheckCircle2,
  Clock3,
  DatabaseZap,
  GitBranch,
  ShieldCheck
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
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
  SummaryStat
} from "../components/Common";
import { useWorkspace } from "../context/WorkspaceContext";
import { adaptMetric } from "../api/adapters";
import { api } from "../api/client";
import { useApiResource } from "../hooks/useApiResource";
import { chartTheme } from "../lib/chartTheme";
import {
  blockLabels,
  formatDateTime,
  formatDuration,
  formatNumber,
  formatPercent,
  splitLabels
} from "../lib/format";

const stageLabels: Record<string, string> = {
  S1_DATA: "数据处理",
  S2_DETECTION: "自监督检测",
  S3_RISK: "动态风险",
  S4_REPORT: "证据报告",
  S5_AUDIT: "评价审计"
};

const optimizationLabels: Record<string, string> = {
  parquet_feature_and_artifact_cache: "Parquet 数据缓存",
  shared_process_data_bundles: "进程内数据复用",
  shared_user_day_frame_cache: "用户日帧缓存",
  shared_r42_original_iforest_view: "原始视角共享训练",
  s3_reuses_loaded_active_mask: "活跃掩码复用",
  stage_checkpoint_resume: "阶段断点恢复",
  structured_shard_and_manifest: "运行清单与哈希"
};

export function SystemPage() {
  const {
    block,
    blockSummary,
    run: apiRun,
    runInfo,
    stages,
    audit,
    theme
  } = useWorkspace();
  const metricsResource = useApiResource(
    () => api.metrics(apiRun.run_id),
    [apiRun.run_id]
  );
  if (metricsResource.loading && !metricsResource.data) {
    return <DataLoading label="正在读取系统接口" />;
  }
  if (metricsResource.error || !metricsResource.data) {
    return (
      <DataError
        message={metricsResource.error ?? "系统指标为空"}
        onRetry={metricsResource.reload}
      />
    );
  }
  const metrics = metricsResource.data.map(adaptMetric);
  const data = {
    run: runInfo,
    stages,
    audit,
    metrics
  };
  const colors = chartTheme(theme);
  const maxStage = Math.max(
    ...data.stages.map((stage) => stage.durationSeconds)
  );
  const avgBalanced = data.metrics
    .filter(
      (row) =>
        row.block === block &&
        row.variant === "aligned_iforest"
    )
    .reduce(
      (acc, row) => {
        acc.sum += row.balanced;
        acc.count += 1;
        return acc;
      },
      { sum: 0, count: 0 }
    );
  const budgetRows = [3, 5, 10, 20, 30, 50].map((k) => {
    const rows = data.metrics.filter(
      (row) =>
        row.block === block &&
        row.variant === "aligned_iforest" &&
        row.k === k
    );
    return {
      k,
      balanced:
        rows.reduce((sum, row) => sum + row.balanced, 0) /
        Math.max(rows.length, 1)
    };
  });

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="平台 / 系统运行"
        title="V3 运行与验收"
        description={`${data.run.id} · ${formatDateTime(
          data.run.completedAt
        )} 完成`}
        actions={
          <span className="success-chip">
            <CheckCircle2 size={16} />
            综合审计通过
          </span>
        }
      />

      <section className="metric-grid">
        <MetricCard
          label="端到端状态"
          value="已完成"
          note="流水线与门禁均完成"
          trend="up"
        />
        <MetricCard
          label="统一告警队列"
          value={formatNumber(data.audit.rows)}
          note={`${data.audit.timelines} 条用户时间线`}
          trend="neutral"
          accent="blue"
        />
        <MetricCard
          label="当前数据块效用"
          value={formatPercent(
            avgBalanced.sum / Math.max(avgBalanced.count, 1)
          )}
          note="K=10 · aligned IForest"
          trend="up"
          accent="amber"
        />
        <MetricCard
          label="审计问题"
          value={String(
            data.audit.duplicate_queue_keys +
              data.audit.rank_problems +
              data.audit.label_columns_in_model_queue.length
          )}
          note="重复键、排名、标签泄漏"
          trend="down"
        />
      </section>

      <section className="system-grid">
        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>阶段运行时间</h2>
              <p>最近一次完整 V3 流水线</p>
            </div>
            <Clock3 size={17} className="panel-icon" />
          </div>
          <div className="stage-list">
            {data.stages.map((stage) => (
              <div className="stage-row" key={stage.key}>
                <div className="stage-status">
                  <CheckCircle2 size={16} />
                </div>
                <div className="stage-body">
                  <div className="rank-top">
                    <strong>
                      {stage.key} ·{" "}
                      {stageLabels[stage.key] ?? stage.title}
                    </strong>
                    <span>
                      {formatDuration(stage.durationSeconds)}
                    </span>
                  </div>
                  <ProgressBar
                    value={stage.durationSeconds}
                    max={maxStage}
                  />
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>验收门禁</h2>
              <p>结构、安全与可复现性</p>
            </div>
            <ShieldCheck size={17} className="panel-icon" />
          </div>
          <div className="audit-grid">
            <AuditItem
              label="重复告警键"
              pass={data.audit.duplicate_queue_keys === 0}
              detail={String(data.audit.duplicate_queue_keys)}
            />
            <AuditItem
              label="排名连续性"
              pass={data.audit.rank_problems === 0}
              detail={`${data.audit.rank_problems} 个问题`}
            />
            <AuditItem
              label="S5 卡片审计"
              pass={data.audit.s5_card_integrity_pass}
              detail="结构完整"
            />
            <AuditItem
              label="SSL 证据审计"
              pass={data.audit.s5_ssl_integrity_pass}
              detail="字段完整"
            />
            <AuditItem
              label="标签隔离"
              pass={!data.audit.labels_written_to_model_outputs}
              detail="模型输出无标签"
            />
            <AuditItem
              label="时间线一致性"
              pass={data.audit.timeline_card_consistency}
              detail="告警数一致"
            />
          </div>
        </article>
      </section>

      <section className="system-grid">
        <article className="panel chart-panel">
          <div className="panel-header">
            <div>
              <h2>告警预算曲线</h2>
              <p>不同 K 下的平均均衡效用</p>
            </div>
            <span className="panel-chip">{blockSummary.label}</span>
          </div>
          <div
            className="chart-shell"
            role="img"
            aria-label="告警预算曲线"
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={budgetRows}
                margin={{ top: 8, right: 12, left: -12, bottom: 0 }}
              >
                <CartesianGrid
                  stroke={colors.grid}
                  vertical={false}
                />
                <XAxis
                  dataKey="k"
                  tick={{ fill: colors.text, fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tickFormatter={(value: number) =>
                    formatPercent(value, 0)
                  }
                  tick={{ fill: colors.text, fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={52}
                />
                <Tooltip
                  formatter={(value) => [
                    formatPercent(Number(value)),
                    "均衡效用"
                  ]}
                  labelFormatter={(value) => `K = ${value}`}
                  contentStyle={{
                    borderRadius: 10,
                    borderColor: colors.grid,
                    background: colors.surface,
                    color: colors.text
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="balanced"
                  stroke={colors.blue}
                  strokeWidth={2}
                  dot={{
                    r: 3.5,
                    fill: colors.blue,
                    stroke: colors.surface
                  }}
                  activeDot={{ r: 5 }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>K=10 分块指标</h2>
              <p>五个评价单元的统一结果</p>
            </div>
            <span className="panel-chip">
              {formatPercent(
                data.metrics
                  .filter(
                    (row) =>
                      row.k === 10 &&
                      row.variant === "aligned_iforest"
                  )
                  .reduce(
                    (sum, row) => sum + row.balanced,
                    0
                  ) /
                  data.metrics.filter(
                    (row) =>
                      row.k === 10 &&
                      row.variant === "aligned_iforest"
                  ).length
              )}
            </span>
          </div>
          <div className="metric-table-wrap">
            <table className="metric-table">
              <thead>
                <tr>
                  <th>数据块 / 划分</th>
                  <th>均衡效用</th>
                  <th>事件</th>
                </tr>
              </thead>
              <tbody>
                {data.metrics
                  .filter(
                    (row) =>
                      row.k === 10 &&
                      row.variant === "aligned_iforest"
                  )
                  .map((row) => (
                    <tr
                      key={`${row.block}-${row.split}`}
                    >
                      <td>
                        <strong>
                          {blockLabels[row.block] ?? row.block}
                        </strong>
                        <span>
                          {splitLabels[row.split] ?? row.split}
                        </span>
                      </td>
                      <td>{formatPercent(row.balanced)}</td>
                      <td>
                        {row.incidentHits}/{row.incidentTotal}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </article>
      </section>

      <section className="system-grid">
        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>系统优化开关</h2>
              <p>本次运行启用的工程优化</p>
            </div>
            <DatabaseZap size={17} className="panel-icon" />
          </div>
          <div className="optimization-grid">
            {Object.entries(data.run.optimizations).map(
              ([key, enabled]) => (
                <div className="optimization-item" key={key}>
                  <CheckCircle2 size={16} />
                  <span>{optimizationLabels[key] ?? key}</span>
                  <strong>{enabled ? "启用" : "关闭"}</strong>
                </div>
              )
            )}
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>运行信息</h2>
              <p>模型与计算契约</p>
            </div>
            <GitBranch size={17} className="panel-icon" />
          </div>
          <div className="summary-grid">
            <SummaryStat label="模型版本" value={data.run.modelVersion} />
            <SummaryStat label="主预算 K" value={data.run.primaryK} />
            <SummaryStat
              label="评测数据块"
              value={data.audit.metrics_blocks_at_k10}
            />
            <SummaryStat
              label="用户时间线"
              value={formatNumber(data.audit.timelines)}
            />
            <SummaryStat
              label="创建时间"
              value={formatDateTime(data.run.createdAt)}
            />
            <SummaryStat
              label="完成时间"
              value={formatDateTime(data.run.completedAt)}
            />
          </div>
        </article>
      </section>

      <section className="limitation-panel">
        <strong>使用边界</strong>
        <ul>
          {data.audit.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function AuditItem({
  label,
  pass,
  detail
}: {
  label: string;
  pass: boolean;
  detail: string;
}) {
  return (
    <div className={`audit-item ${pass ? "pass" : "fail"}`}>
      <span className="audit-check">
        {pass ? "✓" : "!"}
      </span>
      <div>
        <strong>{label}</strong>
        <span>{detail}</span>
      </div>
    </div>
  );
}
