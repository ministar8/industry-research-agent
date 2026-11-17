# Industry Research Agent

> **Evidence-driven multi-agent research for industry and competitive intelligence.**

一个面向行业研究与竞品分析的多智能体协作系统。它不只回答“搜索到了什么”，还会持续追踪：

- 研究目标是否被正确拆解；
- 每个子问题和研究对象是否都有证据覆盖；
- 不同来源之间是否存在冲突；
- 报告中的每个引用是否都能回溯到真实来源；
- 继续检索是否仍然值得付出时间与 token。

项目采用 **LangGraph + FastAPI + DeepSeek Search/Fetch** 构建，提供 REST、SSE 和 AG-UI 接口，并配套一个可独立部署的静态展示页。

<p align="center">
  <strong>从模糊问题，到可追溯的研究报告</strong>
</p>

```text
用户长文本
    │
    ▼
Supervisor ── 意图解析 · 任务拆解 · 约束保真 · 路由
    │
    ├── Researcher ── Search / Fetch / 来源归一化
    │        │
    │        ▼
    ├── Analyst ──── 交叉验证 · 证据覆盖 · 冲突与缺口
    │        │
    │        ▼
    └── Writer ───── 结构化报告 · 引用锚点 · 结论边界
             │
             ▼
          finalize ── 组装报告与来源，不再调用模型
```

## 为什么这个项目值得做

普通的 LLM 搜索应用通常只关注“能不能生成一篇文章”。本项目把问题换成了：

> **这篇报告的结论由什么证据支持？哪些地方仍然不确定？**

因此，系统的核心不是增加 Agent 数量，而是建立一条可检查的证据链：

```text
研究需求
  → 研究任务
  → 来源与摘录
  → 任务级 / 对象级证据覆盖
  → 证据强度与信息缺口
  → 带引用的报告
```

这条链路同时服务于三个目标：

| 目标 | 系统实现 |
| --- | --- |
| **可靠** | 来源去重、来源质量评分、交叉验证、冲突识别 |
| **可追溯** | 报告中的 `[n]` 必须解析到已采集来源 |
| **可评估** | 用确定性规则检查约束、格式、引用、覆盖和趋势证据 |

## 核心亮点

### 1. 面向复杂输入的任务拆解

用户可以直接输入一整段自然语言。Supervisor 会剥离寒暄、背景和情绪表达，同时保留真正影响研究结果的约束：

- 行业、公司、产品和技术路线；
- 时间区间、地区和统计口径；
- 明确排除项；
- 报告格式、长度、语言和目标读者。

这些约束会进入共享状态，并传递给 Researcher、Analyst 和 Writer，而不是只停留在入口节点。

### 2. 证据覆盖驱动的协作

Analyst 不只生成一段总结，还会分别记录：

- `evidence_coverage`：每个研究任务是充分、部分、缺失还是冲突；
- `subject_coverage`：每个研究对象缺少哪些证据维度；
- `evidence_statements`：事实、变化、趋势和预测的证据边界。

这样可以避免“某个对象有证据，就替代了另一个对象”的错误，也避免把规划或预测写成已经发生的趋势。

### 3. 可验证的报告引用

Writer 接收编号化来源索引，并保留引用锚点。系统会检查：

```text
报告中的 [n]
    ↓
是否存在第 n 条来源
    ↓
是否来自本轮 Researcher 采集的来源
```

Writer 不会凭空引入未记录的 URL，`finalize` 只组装现有结果，不会在最后一步偷偷生成内容。

### 4. 工程级可靠性

- 结构化输出使用 Pydantic schema；
- 外部 Search / Fetch 的拒绝和错误显式暴露；
- 支持 JSON Schema 不可用时的兼容降级；
- 记录每轮模型、token、检索调用和降级路径；
- 支持 SQLite、PostgreSQL 和 MongoDB checkpoint；
- 支持 JWT 用户认证与旧版 Bearer token 兼容模式；
- 支持 `fresh` / `reuse` 两种研究上下文模式。

## 系统架构

### 节点职责

| 节点 | 负责什么 | 关键输出 |
| --- | --- | --- |
| **Supervisor** | 解析意图、拆解任务、决定下一步 | `intent` `subjects` `tasks` `constraints` |
| **Researcher** | 规划查询、检索页面、抓取正文、归一化来源 | `sources` |
| **Analyst** | 去重、交叉验证、识别冲突和信息缺口 | `analysis` `evidence_coverage` |
| **Writer** | 按证据边界生成报告并保留引用 | `report` |
| **finalize** | 拼接报告、来源和提示信息 | `messages` |

节点之间不直接互相调用，只通过 `ResearchState` 传递增量状态。Supervisor 在每一轮根据状态重新路由：

```text
Supervisor
   ├─ 需要外部信息 ───────────────► Researcher
   ├─ 来源充分但尚未分析 ─────────► Analyst
   ├─ 分析完成但尚未成稿 ─────────► Writer
   └─ 已有最终结果或无研究对象 ───► finalize
```

### 目录结构

```text
src/
├── agents/research_team/   # 研究图、领域模型、检索、分析与写作
├── auth/                    # JWT / Bearer 认证
├── client/                  # 与 HTTP 解耦的同步与异步客户端
├── core/                    # 配置、模型工厂、运行指标
├── evaluation/              # 确定性评估器
├── memory/                  # SQLite / PostgreSQL / MongoDB 后端
├── schema/                  # 服务端、客户端共用的 Pydantic 协议
├── service/                 # FastAPI、SSE、AG-UI、线程管理
├── run_agent.py             # 直接运行研究图
└── run_service.py           # 启动 HTTP 服务
static/                      # 零依赖 HTML / CSS / JavaScript 展示页
evals/                       # 评测用例、运行脚本和历史结果
tests/                       # 单元、服务、端到端和持久化测试
docs/                        # AG-UI 接入与凭据说明
```

## 快速开始

### 环境要求

- Python `>=3.12,<3.15`
- [uv](https://docs.astral.sh/uv/)
- DeepSeek Search/Fetch API key
- 任一可用的 LLM provider key

### 安装与启动

```sh
uv sync
copy .env.example .env
```

编辑 `.env`，至少配置一个 LLM provider。典型配置为：

```dotenv
DASHSCOPE_API_KEY=your-dashscope-key
DEEPSEEK_API_KEY=your-deepseek-key
DATABASE_TYPE=sqlite
```

然后启动服务：

```sh
uv run python src/run_service.py
```

默认地址：<http://127.0.0.1:8080>

如果只想验证链路，不调用真实模型：

```sh
USE_FAKE_MODEL=true uv run python src/run_service.py
```

Fake 模式会跳过检索与分析，适合检查服务启动、请求路由和前后端连接。

### PostgreSQL 模式

默认配置使用 PostgreSQL。开发环境可以直接启动数据库：

```sh
docker compose up -d postgres
uv run python src/run_service.py
```

Windows + PostgreSQL 场景下，项目会使用 Selector event loop 兼容 psycopg 的异步模式；这条路径不提供 uvicorn 热重载。

## API

当前注册的业务 Agent 为 `research-supervisor`。默认 Agent 可以省略 `agent_id`。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `GET` | `/info` | Agent、模型和服务元信息 |
| `POST` | `/{agent_id}/invoke` | 非流式调用 |
| `POST` | `/{agent_id}/stream` | SSE 流式调用 |
| `POST` | `/{agent_id}/history` | 查询会话历史 |
| `GET` | `/{agent_id}/threads` | 查询用户会话列表 |
| `POST` | `/agui/{agent_id}/run` | AG-UI SSE 接口 |
| `POST` | `/auth/register` | 注册用户 |
| `POST` | `/auth/login` | 登录并写入 HttpOnly Cookie |

完整的 AG-UI 接入说明见 [`docs/AGUI.md`](docs/AGUI.md)。

## 配置要点

| 变量 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 百炼兼容端点的 LLM key |
| `DEEPSEEK_API_KEY` | Search / Fetch 必需，也可作为 LLM provider |
| `DEFAULT_MODEL` | 覆盖自动选择的默认模型 |
| `LLM_MAX_TOKENS` | 单次补全上限，默认 `16384` |
| `DATABASE_TYPE` | `sqlite`、`postgres` 或 `mongo` |
| `JWT_SECRET` | 启用用户名登录和 Cookie 鉴权 |
| `AUTH_SECRET` | 未启用 JWT 时的 Bearer token 兼容认证 |
| `USE_FAKE_MODEL` | 使用 Fake 模型进行离线链路验证 |
| `LOG_LEVEL` | `INFO` 查看请求级运行指标，`DEBUG` 查看节点级 token 归因 |

完整变量列表见 [`.env.example`](.env.example)。不要把真实 API key 提交到 Git。

## 评估与可观测性

项目把“报告看起来不错”拆成可以自动检查的指标。评估用例位于 `evals/cases.json`，覆盖：

- 约束是否进入研究流程；
- 排除项是否泄漏到正文；
- 表格、清单等交付格式；
- 引用锚点是否越界；
- 报告章节和句子是否完整；
- 单源、冲突和口径差异是否有提示；
- 趋势表述是否有足够的同口径时间证据；
- 无研究对象时是否正确返回引导，而不是浪费模型调用。

运行评估：

```sh
PYTHONPATH=src uv run python evals/run_eval.py
PYTHONPATH=src uv run python evals/run_eval.py --list
PYTHONPATH=src uv run python evals/run_eval.py catl
```

每次运行会保存到 `evals/runs/<timestamp>/`，包括完整报告和 `summary.md`。改变 prompt、模型或 schema 后，先使用同一批用例对比结果，再查看实际报告确认内容质量。

将日志级别设为 `INFO` 后，可看到类似的请求摘要：

```text
turn thread=metrics-smoke status=ok elapsed=29.4s
llm_calls=8 tokens_in=26196 tokens_out=3866
search=4 fetch=2 search_failed=0 degraded=-
```

指标包括模型调用次数、输入输出 token、Search / Fetch 成功与失败次数、耗时和降级路径，并不会把完整报告写入运行指标表。

## 开发

```sh
uv run ruff format --check src/ tests/ evals/
uv run ruff check src/ tests/ evals/
uv run pyrefly check
uv run pytest -q
```

单独运行研究团队测试：

```sh
uv run pytest tests/agents/test_research_team.py -q
```

启动 LangGraph Studio：

```sh
uv run langgraph dev
```

## 关键设计约束

这些约束是项目可靠性的基础，修改节点或状态时需要保持：

1. `finalize` 只组装结果，不调用模型。
2. `constraints` 必须传递给所有消费研究结果的节点。
3. 新请求默认使用 `context_mode=fresh`；只有明确复用上一轮来源时使用 `reuse`。
4. `sources` 按归一化后的 `url_key` 去重并跨轮累积，不覆盖已有来源。
5. 报告中的每个 `[n]` 都必须能解析到 `sources`。
6. 来源的 `confidence` 是来源层信号，不等同于最终结论的证据强度。
7. 预测、规划和单点事实不能被无证据写成实际趋势。
8. 每次结构化模型调用都需要记录节点指标，避免 token 和失败率不可见。

## 项目定位

这是一个同时面向**毕业设计、工程实践和求职展示**的项目：

- **研究问题**：复杂研究任务下，如何用证据覆盖评估驱动多轮检索；
- **系统实现**：LangGraph 状态图、FastAPI 服务、持久化、认证和流式接口；
- **可靠性机制**：结构化输出、引用校验、错误显式暴露和确定性评估；
- **可扩展方向**：证据缺口优先级、检索收益估计、报告可信度评分和人机协作复核。

推荐的项目表述：

> 设计并实现基于 LangGraph 的证据驱动行业研究多智能体系统，针对复杂自然语言需求自动完成任务拆解、来源检索、证据覆盖评估和可追溯报告生成；构建任务级与对象级证据校验机制，对缺失、冲突和低置信度信息进行显式标注，并通过确定性评估框架验证报告结构、约束遵循和引用完整性。

## License

[MIT](LICENSE)
