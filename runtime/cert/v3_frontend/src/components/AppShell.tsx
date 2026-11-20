import {
  Activity,
  Database,
  LayoutDashboard,
  Menu,
  Moon,
  ScanSearch,
  ShieldAlert,
  Sun,
  X
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  NavLink,
  Outlet,
  useLocation
} from "react-router-dom";
import { useWorkspace } from "../context/WorkspaceContext";

const navigation = [
  {
    to: "/",
    label: "总览",
    icon: LayoutDashboard,
    title: "风险总览"
  },
  {
    to: "/alerts",
    label: "告警队列",
    icon: ShieldAlert,
    title: "告警队列"
  },
  {
    to: "/evidence",
    label: "证据分析",
    icon: ScanSearch,
    title: "证据分析"
  },
  {
    to: "/system",
    label: "系统运行",
    icon: Activity,
    title: "系统运行"
  },
  {
    to: "/data",
    label: "数据资产",
    icon: Database,
    title: "数据与模型资产"
  }
];

export function AppShell() {
  const {
    block,
    setBlock,
    blocks,
    blockSummary,
    run,
    runs,
    setRunId,
    theme,
    toggleTheme
  } = useWorkspace();
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  const current = navigation.find(
    (item) =>
      item.to === "/"
        ? location.pathname === "/"
        : location.pathname.startsWith(item.to)
  );

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  return (
    <div className={`app-shell ${menuOpen ? "menu-open" : ""}`}>
      <aside className="sidebar" aria-label="主导航">
        <div className="brand">
          <div className="brand-mark">V3</div>
          <div>
            <div className="brand-name">风险审查台</div>
            <div className="brand-caption">自监督行为异常识别</div>
          </div>
          <button
            className="icon-button sidebar-close"
            type="button"
            aria-label="关闭导航"
            onClick={() => setMenuOpen(false)}
          >
            <X size={18} />
          </button>
        </div>

        <nav className="primary-nav">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  `nav-item ${isActive ? "active" : ""}`
                }
              >
                <Icon size={17} strokeWidth={1.8} />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        <div className="sidebar-meta">
          <div className="meta-label">当前运行</div>
          <div className="meta-value">{run.run_id}</div>
          <div className="meta-status">
            <span className="status-dot" />
            {run.audit_pass ? "V3 审计通过" : "审计未通过"}
          </div>
        </div>
      </aside>

      <button
        className="sidebar-scrim"
        type="button"
        aria-label="关闭导航"
        onClick={() => setMenuOpen(false)}
      />

      <div className="main-column">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-button menu-button"
              type="button"
              aria-label="打开导航"
              onClick={() => setMenuOpen(true)}
            >
              <Menu size={20} />
            </button>
            <div>
              <div className="breadcrumb">
                风险运营 <span>/</span> {current?.title ?? "总览"}
              </div>
              <div className="page-context">
                {blockSummary.label} · 每日 Top-10
              </div>
            </div>
          </div>

          <div className="topbar-actions">
            <label className="block-select run-select">
              <span>运行</span>
              <select
                value={run.run_id}
                onChange={(event) => setRunId(event.target.value)}
                aria-label="切换模型运行"
              >
                {runs.map((item) => (
                  <option key={item.run_id} value={item.run_id}>
                    {item.run_id}
                  </option>
                ))}
              </select>
            </label>
            <label className="block-select">
              <span>数据块</span>
              <select
                value={block}
                onChange={(event) => setBlock(event.target.value)}
                aria-label="切换数据块"
              >
                {blocks.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="audit-pill">
              <span className="status-dot" />
              {run.audit_pass ? "审计通过" : "审计异常"}
            </div>
            <button
              className="icon-button"
              type="button"
              onClick={toggleTheme}
              aria-label={
                theme === "light"
                  ? "切换深色主题"
                  : "切换浅色主题"
              }
            >
              {theme === "light" ? (
                <Moon size={18} />
              ) : (
                <Sun size={18} />
              )}
            </button>
          </div>
        </header>

        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
