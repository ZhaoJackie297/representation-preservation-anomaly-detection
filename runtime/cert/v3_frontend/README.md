# V3 Risk Console Frontend

React、TypeScript、Vite 构建的 V3 风险审查前端。

本目录只包含客户端外壳与 API 适配层。它不含训练数据、模型权重、历史告警
结果或 FastAPI 后端实现，默认调用同源 `/api/v1`。

## 核心原则

- 前端代码不包含训练数据；
- 前端代码不包含历史告警结果；
- 模型服务默认使用独立的 `v3_service_runs`，不会自动暴露研究阶段的历史结果；
- 页面启动后先调用 `/api/v1/runs`；
- 告警筛选和分页由 API 执行；
- 单条证据和用户时间线按需读取；
- 没有运行时，页面引导通过 API 创建模型 job。

## 本地开发

准备一个实现 `/api/v1` 契约的模型服务后，启动前端：

```powershell
Set-Location runtime/cert/v3_frontend
Copy-Item .env.example .env.local
pnpm install --frozen-lockfile
pnpm dev
```

默认地址：

```text
http://127.0.0.1:4173
```

在 `.env.local` 中设置 API 地址：

```text
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

## 构建

```powershell
pnpm typecheck
pnpm build
```

生产环境建议让前端和 API 同源：

```text
/              → dist/
/api/v1/*      → FastAPI
```

并将：

```text
VITE_API_BASE_URL=/api/v1
```

作为构建变量。

## 页面

- `/`：总览；
- `/alerts`：服务端分页、筛选、导出；
- `/evidence`：SSL 和通道证据；
- `/system`：流水线、审计、预算曲线；
- `/data`：模型接口、运行契约和数据隔离。

## 首次运行

当 API 返回空运行列表时，前端不会展示伪造数据，而是显示运行准备页。
管理用户可以通过 `POST /api/v1/jobs` 启动 V3 流水线。

## 安全

- 默认只允许本机 4173 来源访问 API；
- 生产环境应配置明确 CORS 来源；
- 创建 job 可启用 `V3_ADMIN_TOKEN`；
- 前端永远不会返回 Parquet 或服务器运行目录路径。
