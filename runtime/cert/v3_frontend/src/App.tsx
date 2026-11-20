import {
  AlertCircle,
  ArrowRight,
  LoaderCircle,
  Play,
  Server
} from "lucide-react";
import {
  lazy,
  Suspense,
  useEffect,
  useState
} from "react";
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes
} from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { adaptBlock } from "./api/adapters";
import { WorkspaceProvider } from "./context/WorkspaceContext";
import { api, apiDocsUrl } from "./api/client";
import type {
  ApiJob,
  ApiModelInfo
} from "./api/types";
import { useBootstrap } from "./hooks/useBootstrap";

const OverviewPage = lazy(() =>
  import("./pages/OverviewPage").then((module) => ({
    default: module.OverviewPage
  }))
);
const AlertsPage = lazy(() =>
  import("./pages/AlertsPage").then((module) => ({
    default: module.AlertsPage
  }))
);
const EvidencePage = lazy(() =>
  import("./pages/EvidencePage").then((module) => ({
    default: module.EvidencePage
  }))
);
const SystemPage = lazy(() =>
  import("./pages/SystemPage").then((module) => ({
    default: module.SystemPage
  }))
);
const DataPage = lazy(() =>
  import("./pages/DataPage").then((module) => ({
    default: module.DataPage
  }))
);

export default function App() {
  const {
    model,
    runs,
    run,
    blocks,
    system,
    loading,
    error,
    setRunId,
    refreshRuns
  } = useBootstrap();

  if (loading) {
    return (
      <div className="boot-screen" role="status">
        <div className="boot-mark">V3</div>
        <LoaderCircle className="spin" size={22} />
        <strong>正在加载 V3 风险数据</strong>
        <span>队列、证据与运行清单</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="boot-screen error-screen" role="alert">
        <AlertCircle size={28} />
        <strong>无法连接 V3 模型服务</strong>
        <span>{error}</span>
        <code>python -m v3.api</code>
        <button
          className="button primary"
          type="button"
          onClick={() => void refreshRuns()}
        >
          重新加载
        </button>
      </div>
    );
  }

  if (!model || loading) {
    return (
      <div className="boot-screen" role="status">
        <div className="boot-mark">V3</div>
        <LoaderCircle className="spin" size={22} />
        <strong>正在连接模型接口</strong>
        <span>检查运行状态与数据集清单</span>
      </div>
    );
  }

  if (!runs.length) {
    return (
      <NoRunsScreen
        model={model}
        onCompleted={refreshRuns}
      />
    );
  }

  if (!run || !blocks.length || !system) {
    return (
      <div className="boot-screen error-screen" role="alert">
        <AlertCircle size={28} />
        <strong>运行数据不完整</strong>
        <span>请检查 API 运行状态或重新执行流水线。</span>
      </div>
    );
  }

  return (
    <WorkspaceProvider
      model={model}
      runs={runs}
      run={run}
      blocks={blocks.map(adaptBlock)}
      system={system}
      setRunId={(runId) => void setRunId(runId)}
    >
      <BrowserRouter>
        <Suspense fallback={<RouteLoading />}>
          <Routes>
            <Route element={<AppShell />}>
              <Route index element={<OverviewPage />} />
              <Route path="alerts" element={<AlertsPage />} />
              <Route path="evidence" element={<EvidencePage />} />
              <Route path="system" element={<SystemPage />} />
              <Route path="data" element={<DataPage />} />
              <Route
                path="*"
                element={<Navigate to="/" replace />}
              />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
    </WorkspaceProvider>
  );
}

function RouteLoading() {
  return (
    <div className="route-loading" role="status">
      <LoaderCircle className="spin" size={18} />
      正在加载页面
    </div>
  );
}

function NoRunsScreen({
  model,
  onCompleted
}: {
  model: ApiModelInfo;
  onCompleted: () => Promise<void>;
}) {
  const [rebuildS1, setRebuildS1] = useState(false);
  const [job, setJob] = useState<ApiJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) {
      return;
    }
    const timer = window.setInterval(() => {
      void api.job(job.job_id).then(async (next) => {
        setJob(next);
        if (next.status === "completed") {
          window.clearInterval(timer);
          await onCompleted();
        }
      });
    }, 1800);
    return () => window.clearInterval(timer);
  }, [job, onCompleted]);

  async function start() {
    setError(null);
    try {
      setJob(await api.createJob(rebuildS1));
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "启动运行失败"
      );
    }
  }

  return (
    <div className="connection-screen">
      <section className="connection-card">
        <div className="connection-brand">
          <span>V3</span>
          <div>
            <strong>{model.name}</strong>
            <small>{model.task}</small>
          </div>
        </div>
        <div className="connection-status">
          <Server size={18} />
          模型 API 已连接 · 当前没有可用运行
        </div>
        <h1>准备接入第一次模型运行</h1>
        <p>
          前端不保存训练结果。点击后由后端 API 创建 V3
          流水线任务，完成后页面自动加载最新队列。
        </p>
        <label className="job-option">
          <input
            type="checkbox"
            checked={rebuildS1}
            onChange={(event) => setRebuildS1(event.target.checked)}
          />
          <span>
            同时重建 S1 用户日特征和序列数据
            <small>源数据已准备好时可关闭</small>
          </span>
        </label>
        {job ? (
          <div className="job-progress">
            <LoaderCircle className="spin" size={18} />
            <div>
              <strong>{job.status}</strong>
              <span>
                {job.run_id
                  ? `运行 ${job.run_id}`
                  : `任务 ${job.job_id}`}
              </span>
            </div>
          </div>
        ) : (
          <div className="connection-actions">
            <button
              className="button primary"
              type="button"
              onClick={() => void start()}
            >
              <Play size={16} />
              运行 V3 模型
            </button>
            <a
              className="button secondary"
              href={apiDocsUrl()}
              target="_blank"
              rel="noreferrer"
            >
              打开 API 文档
              <ArrowRight size={15} />
            </a>
          </div>
        )}
        {job?.status === "failed" ? (
          <div className="inline-error" role="alert">
            {job.error ?? "模型运行失败"}
          </div>
        ) : null}
        {error ? (
          <div className="inline-error" role="alert">
            {error}
          </div>
        ) : null}
        <div className="connection-endpoints">
          {model.interfaces.slice(0, 4).map((endpoint) => (
            <code key={endpoint}>{endpoint}</code>
          ))}
        </div>
      </section>
    </div>
  );
}
