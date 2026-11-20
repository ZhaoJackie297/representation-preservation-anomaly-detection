import {
  CheckCircle2,
  Database,
  ExternalLink,
  FileJson2,
  GitFork,
  Layers3,
  ShieldCheck
} from "lucide-react";
import {
  PageHeading,
  ProgressBar,
  SummaryStat
} from "../components/Common";
import { apiDocsUrl } from "../api/client";
import { useWorkspace } from "../context/WorkspaceContext";
import {
  blockLabels,
  formatDate,
  formatNumber,
  formatPercent
} from "../lib/format";

const systemSteps = [
  {
    key: "S1",
    title: "数据处理",
    detail: "用户日特征、序列切分与标签隔离",
    icon: Database
  },
  {
    key: "S2",
    title: "自监督检测",
    detail: "TS2Vec、遮罩差异、个人流形与通道证据",
    icon: Layers3
  },
  {
    key: "S3",
    title: "动态风险",
    detail: "severity 排序、有限预算与冷却约束",
    icon: GitFork
  },
  {
    key: "S4",
    title: "证据报告",
    detail: "风险带、通道证据和用户时间线",
    icon: FileJson2
  },
  {
    key: "S5",
    title: "评价审计",
    detail: "结构、标签隔离和指标可复现性",
    icon: ShieldCheck
  }
];

export function DataPage() {
  const {
    model,
    runInfo,
    runs,
    blocks,
    audit
  } = useWorkspace();
  const data = {
    run: runInfo,
    schemaVersion: "1.0",
    generatedAt: runInfo.completedAt,
    blocks,
    audit
  };
  const maxAlerts = Math.max(
    ...data.blocks.map((block) => block.alerts)
  );

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="平台 / 数据资产"
        title="数据与模型资产"
        description="前端通过版本化 REST API 按需读取；不打包训练结果，完整队列事实保存在模型服务的运行目录。"
        actions={
          <a
            className="button secondary"
            href={apiDocsUrl()}
            target="_blank"
            rel="noreferrer"
          >
            <ExternalLink size={16} />
            打开 API 文档
          </a>
        }
      />

      <section className="asset-hero">
        <div>
          <div className="asset-kicker">当前模型运行</div>
          <h2>{data.run.modelTitle}</h2>
          <p>
            模型 {model.model_version} · 可用运行 {runs.length} ·
            Schema {data.schemaVersion}
          </p>
        </div>
        <div className="asset-health">
          <CheckCircle2 size={18} />
          数据契约完整
        </div>
      </section>

      <section className="data-layout">
        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>三个数据块</h2>
              <p>开发、重训与外部敏感性数据规模</p>
            </div>
            <span className="panel-chip">{data.blocks.length} 块</span>
          </div>
          <div className="dataset-list">
            {data.blocks.map((block) => (
              <div className="dataset-card" key={block.id}>
                <div className="dataset-title">
                  <div>
                    <strong>{block.label}</strong>
                    <span>{block.id}</span>
                  </div>
                  <span>
                    {formatDate(block.dateRange.start)} —{" "}
                    {formatDate(block.dateRange.end)}
                  </span>
                </div>
                <div className="dataset-stats">
                  <SummaryStat
                    label="告警"
                    value={formatNumber(block.alerts)}
                  />
                  <SummaryStat
                    label="用户"
                    value={formatNumber(block.users)}
                  />
                  <SummaryStat
                    label="告警日"
                    value={formatNumber(block.days)}
                  />
                  <SummaryStat
                    label="SSL 一致"
                    value={formatPercent(block.sslAgreementRate)}
                  />
                </div>
                <ProgressBar
                  value={block.alerts}
                  max={maxAlerts}
                />
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>标签隔离与审计</h2>
              <p>模型可见数据与离线评价数据严格分离</p>
            </div>
            <ShieldCheck size={17} className="panel-icon" />
          </div>
          <div className="contract-list">
            <ContractRow
              label="统一告警键唯一"
              value={data.audit.duplicate_queue_keys === 0}
            />
            <ContractRow
              label="每日排名连续"
              value={data.audit.rank_problems === 0}
            />
            <ContractRow
              label="证据卡无标签列"
              value={
                data.audit.label_columns_in_model_queue.length === 0
              }
            />
            <ContractRow
              label="标签未写入模型输出"
              value={
                !data.audit.labels_written_to_model_outputs
              }
            />
            <ContractRow
              label="时间线与卡片一致"
              value={data.audit.timeline_card_consistency}
            />
          </div>
          <div className="audit-summary-box">
            <span>综合审计</span>
            <strong>
              {data.audit.audit_pass ? "通过" : "未通过"}
            </strong>
            <p>
              离线恶意账户日仅参与 S5 审计，前端队列与证据接口不返回标签。
            </p>
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>五体系统一链路</h2>
            <p>从原始行为数据到可审查风险队列</p>
          </div>
        </div>
        <div className="pipeline-flow">
          {systemSteps.map((step, index) => {
            const Icon = step.icon;
            return (
              <div className="pipeline-step-wrap" key={step.key}>
                <div className="pipeline-step">
                  <div className="pipeline-icon">
                    <Icon size={18} />
                  </div>
                  <div>
                    <span>{step.key}</span>
                    <strong>{step.title}</strong>
                    <p>{step.detail}</p>
                  </div>
                </div>
                {index < systemSteps.length - 1 ? (
                  <div className="pipeline-connector" />
                ) : null}
              </div>
            );
          })}
        </div>
      </section>

      <section className="data-layout">
        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>模型计算口径</h2>
              <p>V3 不改变 V2-M-S 的检测公式</p>
            </div>
          </div>
          <pre className="code-block">
            <code>{`severity =
  (masked_quantile
   + manifold_quantile
   + channel_top2_quantile) / 3

S3_risk = severity

q(K) = min(max(1, round(0.4K)), floor(K/2))
K = 10, q = 4, cooldown = 7 days`}</code>
          </pre>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <h2>接口规模</h2>
              <p>当前静态 JSON 契约</p>
            </div>
            <FileJson2 size={17} className="panel-icon" />
          </div>
          <div className="summary-grid">
            <SummaryStat
              label="告警记录"
              value={formatNumber(data.audit.rows)}
            />
            <SummaryStat
              label="用户时间线"
              value={formatNumber(data.audit.timelines)}
            />
            <SummaryStat
              label="模型接口"
              value={model.interfaces.length}
            />
            <SummaryStat
              label="可用运行"
              value={formatNumber(runs.length)}
            />
            <SummaryStat
              label="数据块"
              value={data.blocks
                .map((item) => blockLabels[item.id])
                .join(" / ")}
            />
            <SummaryStat
              label="Schema"
              value={data.schemaVersion}
            />
          </div>
        </article>
      </section>
    </div>
  );
}

function ContractRow({
  label,
  value
}: {
  label: string;
  value: boolean;
}) {
  return (
    <div className={`contract-row ${value ? "pass" : "fail"}`}>
      <span>{value ? "✓" : "!"}</span>
      <strong>{label}</strong>
      <small>{value ? "通过" : "失败"}</small>
    </div>
  );
}
