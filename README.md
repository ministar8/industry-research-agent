# 行业与竞品调研智能体

面向行业研究与竞品分析的**多智能体协作服务**。一个调度主控加三个专职子智能体，共享同一份调研状态，最终产出带引用锚点、可追溯来源的结构化报告。

技术栈：LangGraph（编排）+ FastAPI（服务）+ 静态展示页（HTML / CSS / 原生 JS）+ DeepSeek V4.1-Flash（百炼托管，关闭思考）+ DeepSeek Search/Fetch（检索）。服务对外提供 REST 与 SSE API；`static/` 是可独立部署的纯静态项目展示页。

## 设计原则

主控**不执行任何具体任务**——不搜索、不分析、不写作。它只做四件事：意图解析、任务拆解、路由调度、结果整合。所有实质工作由子智能体完成，子智能体之间只通过共享状态通信。

```text
                      ┌──────────────┐
                      │  Supervisor  │  意图解析 · 任务拆解 · 路由 · 整合
                      └──────┬───────┘
            ┌──────────┬─────┴─────┬────────────┐
            ▼          ▼           ▼            ▼
      ┌──────────┐ ┌─────────┐ ┌────────┐ ┌──────────┐
      │Researcher│ │ Analyst │ │ Writer │ │ finalize │
      └──────────┘ └─────────┘ └────────┘ └──────────┘
            └──────────┴─────────┘             │
              回到 Supervisor 重新调度           END
```

### 子智能体职责

| 角色 | 职责 | 写入的状态字段 |
|---|---|---|
| **Supervisor** | 意图解析、研究对象与任务拆解、路由、结果整合 | `intent` `subjects` `tasks` `constraints` `plan_notes` `next_action` `rounds` `turn_attempted`（转入子节点时清空旧 `report`） |
| **Researcher** | 多轮检索，记录来源 URL、摘录与置信度 | `sources` `turn_attempted`；检索被拒时置 `abort_reason` |
| **Analyst** | 去重、交叉验证、识别矛盾与信息缺口，建立任务级和对象级证据覆盖判断，区分离散点、变化、趋势与预测 | `analysis` `evidence_coverage` `subject_coverage` `evidence_statements` |
| **Writer** | 只按已归一化的证据类型控制趋势表达，输出摘要 / 对象证据完整度 / 发现 / 分析 / 结论报告，保留引用锚点 | `report` |
| **finalize** | 组装报告与来源清单；无成果时返回能力引导 | `messages` |

关键不变量（修改代码时不可破坏）：

1. `finalize` 只做组装，不做生成——它拼接已有的 `report` 或 `analysis`，或返回提示语，**不调用模型**。因此无调研目标的输入不产生任何模型开销。
2. 没有 chat 子 agent。与调研无关的输入由 Supervisor 剥离后继续调研；确实找不到调研对象时才落到 `finalize` 的提示语。不要再加回对话节点，那会让主控的路由判断变得含糊。
3. `thread_id` 标识整个对话，`turn_id` 标识一次研究请求。新请求默认以 `context_mode=fresh`
   开始，隔离上一轮的任务、来源、分析与报告；需要基于上一轮来源重新分析时，显式传
   `context_mode=reuse`。同一轮内 `sources` 按 URL 去重并合并，不覆盖。
4. 报告中每个引用锚点 `[n]` 必须能解析到 `sources` 中的一条记录，Writer 不得引入 Researcher 未采集的 URL。
5. 每个 `ResearchTask` 都有一条 `evidence_coverage` 记录，标记覆盖为充分、部分、缺失或冲突，并单独标记结论证据强度；它用于校准分析与写作，不会单独触发额外检索。
6. `Source.confidence` 是来源层面的检索评分，不等同于子问题结论的证据强度；后者由 Analyst 针对当前任务和来源判断。
7. `subject_coverage` 按研究对象和证据维度记录完整度；对象缺少关键维度时，系统不会用其他对象的证据替代它。
8. `evidence_statements` 中，单点只能作为离散事实；两个时期最多作为期间变化；只有三个同口径时间点或来源明确声明时才能作为趋势，规划/预测不得冒充实际趋势。
9. 调度决策是结构化对象（`RoutingDecision`），不是自由文本，因此路由逻辑可在无 LLM 的情况下测试。

### 输入形态与抗噪

用户输入是**一整段自然描述**，调研需求与无关内容混在一起，不存在独立的「闲聊」通道。Supervisor 的意图解析必须同时完成两件事：

- **剥离**：寒暄、致谢、个人背景与动机、情绪表达、重复铺垫，不进入任务，也不得影响子问题的措辞。
- **保真**：调研对象、时间区间、地区、统计口径、排除项、交付要求（格式 / 篇幅 / 语言）必须原样保留，写入 `constraints`，由 Researcher、Analyst、Writer 共同遵守。

Researcher 只依据解析后的 `tasks` 与 `constraints` 检索；原始长文本仅以 400 字节选作为兜底参考，并明确标注「不得据此增减子问题」，避免噪声污染查询。

只有当整段输入确实找不到任何可调研对象时，`finalize` 才返回「未能识别调研对象」的提示。

> 已知取舍：一段话里塞入多个互不相关的行业时，仍会合并进同一份报告，由 1–4 条子问题覆盖主要对象。若需要分主题独立成文，应拆成多次会话。

## 快速开始

```sh
# 1. 配置环境变量（至少需要一个 LLM key）
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY（LLM）与 DEEPSEEK_API_KEY（检索）

# 2. 安装依赖
uv sync

# 3. 启动服务
uv run python src/run_service.py
```

服务默认监听 `http://0.0.0.0:8080`，对外提供 `/info`、`/invoke`、`/stream`、`/history`、`/threads` 等端点。项目展示页位于 `static/`，是零依赖的纯静态文件，可直接部署到 Cloudflare Pages 等静态托管，不需要后端参与。

无需真实 API key 也可以验证链路连通性：

```sh
USE_FAKE_MODEL=true uv run python src/run_service.py
```

假模型模式下 Supervisor 直接路由到 `finalize`，不触发检索与分析，只验证链路连通性。

## 配置

完整变量列表见 [`.env.example`](.env.example)。核心项：

| 变量 | 说明 | 默认 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 百炼密钥，托管 `deepseek-v4.1-flash-1017` / `qwen3.8-27b`；存在时优先作为默认模型 | — |
| `DEEPSEEK_API_KEY` | **检索必需**，Search/Fetch 用它鉴权；同时可作 LLM 兜底 | — |
| `DASHSCOPE_BASE_URL` | 百炼兼容端点，工作空间迁移后需改 | 旧版 `dashscope.aliyuncs.com` |
| `DASHSCOPE_ENABLE_THINKING` | 百炼托管模型是否开启思考；DeepSeek V4.1 下必须关闭（开启后会在补全上限内耗尽预算） | `false` |
| `LLM_MAX_TOKENS` | 单次补全上限；开思考时须同时容纳思考与可见正文 | `16384` |
| `DEFAULT_MODEL` | 覆盖默认模型。百炼侧：`deepseek-v4.1-flash` / `deepseek-v4.1-flash-1017` / `qwen3.8-27b`；DeepSeek 直连：`deepseek-flash` / `deepseek-v4-pro`；另有 OpenAI 与 `fake`。必须是已配置 provider 的模型，否则启动即报错并指出缺哪个 key | 按 key 选择：百炼 → `deepseek-v4.1-flash-1017`，DeepSeek → `deepseek-flash` |
| `OPENAI_API_KEY` | OpenAI 直连模型（可选），同时是语音功能的凭据 | 空 |
| `COMPATIBLE_BASE_URL` / `COMPATIBLE_MODEL` | 任意 OpenAI 兼容网关（vLLM / SGLang / 内部网关） | 空 |
| `AUTH_SECRET` | 兼容旧版 Bearer token；未设置 JWT 时生效，`/health` 与 `/auth/*` 不受其保护 | 空 |
| `JWT_SECRET` | 用户登录签名密钥；设置后要求 PostgreSQL，并启用用户名登录与 HttpOnly Cookie | 空 |
| `JWT_ACCESS_TTL_MINUTES` / `JWT_REFRESH_TTL_DAYS` | Access / Refresh Cookie 有效期 | `15` / `30` |
| `AUTH_COOKIE_SECURE` | 是否只通过 HTTPS 发送认证 Cookie | `false` |
| `HOST` / `PORT` / `MODE` | 监听地址、端口；`MODE=dev` 开启热重载 | `0.0.0.0` / `8080` / 空 |
| `FORWARDED_ALLOW_IPS` | 信任的代理 IP（`X-Forwarded-*`）。反向代理（Render / nginx / Docker）后必须设为 `*` 或代理网段，否则请求 scheme 仍是内部 HTTP，Cookie 鉴权的 Origin 校验会拒绝所有 HTTPS POST | `127.0.0.1` |
| `LOG_LEVEL` | 日志级别；指标行与 token 归因分别是 INFO / DEBUG，默认级别下不输出 | `WARNING` |
| `DATABASE_TYPE` | `sqlite` / `postgres` / `mongo` | `sqlite` |
| `USE_FAKE_MODEL` | 使用假模型验证链路 | `false` |
| `VOICE_STT_PROVIDER` / `VOICE_TTS_PROVIDER` | 语音识别 / 合成提供方，当前仅实现 `openai`；留空即关闭 | 空 |

## API

| 端点 | 说明 |
|---|---|
| `GET /info` | 可用 agent、模型列表与服务元信息 |
| `POST /{agent_id}/invoke` | 非流式调用；`/invoke` 为默认 agent 的别名 |
| `POST /{agent_id}/stream` | SSE 流式调用；`/stream` 同上 |
| `POST /{agent_id}/history` | 会话历史；`/history` 同上 |
| `GET /{agent_id}/threads` | 列出当前登录用户的历史会话，最近更新在前 |
| `POST /auth/register` / `POST /auth/login` | 注册或登录；认证状态写入 HttpOnly Cookie |
| `POST /auth/refresh` / `POST /auth/logout` / `GET /auth/me` | 刷新、退出、读取当前用户 |
| `POST /feedback` | 星级反馈（LangSmith） |
| `POST /agui/{agent_id}/run` | AG-UI 协议端点，SSE；`/agui/run` 走默认 agent |
| `GET /health` | 健康检查，唯一不经过 bearer 鉴权的端点 |

`agent_id` 目前只有 `research-supervisor`。设置 `JWT_SECRET` 后，除 `/health` 与 `/auth/*`
外的端点均要求登录 Cookie；调研、历史和线程列表使用服务端解析出的用户 ID，不信任请求体中的
`user_id`。Refresh Token 轮换并可通过退出登录撤销。未设置 JWT 时仍保留 `AUTH_SECRET` 和匿名
开发模式的兼容行为。
AG-UI 的接入细节见
[`docs/AGUI.md`](docs/AGUI.md)。

研究请求体支持 `context_mode`：默认 `fresh` 会隔离上一轮研究状态；`reuse` 会保留上一轮来源，
适用于“基于刚才的来源重新分析”等明确的延续请求。服务端始终生成 `turn_id`，客户端不能通过
`agent_config` 覆盖它。

## 目录结构

```text
src/
├── agents/
│   ├── research_team/        # 唯一的业务 agent
│   │   ├── contracts.py      # 研究任务、来源与证据覆盖等领域模型
│   │   ├── state.py          # LangGraph ResearchState 传输状态
│   │   ├── supervisor.py     # 意图解析、路由、结果整合
│   │   ├── researcher.py     # 检索与来源置信度评分
│   │   ├── analyst.py        # 交叉验证与结构化分析
│   │   ├── writer.py         # 报告生成与锚点校验
│   │   ├── search.py         # DeepSeek Search / Fetch 适配层
│   │   ├── helpers.py        # 去重、结构化输出、置信度归一化等共用逻辑
│   │   └── graph.py          # 图组装
│   ├── agents.py             # agent 注册表
│   ├── lazy_agent.py         # 需要异步初始化的 agent 基类（当前注册的 agent 未使用）
│   ├── tools.py              # 共享工具
│   └── utils.py              # 消息与回调辅助
├── core/                     # settings.py、llm.py（模型工厂）、metrics.py（请求级计数）
├── memory/                   # checkpointer 与 store 后端：sqlite / postgres / mongodb
├── schema/                   # Pydantic 协议模型：models.py（枚举）、schema.py（请求响应）
├── service/                  # service.py（路由与鉴权）、agui.py、threads.py、utils.py
├── client/                   # 通用 AgentClient，同步与异步两套调用
├── evaluation/               # 评估用确定性评分器
├── voice/                    # 可选语音底层：stt.py / tts.py 工厂、providers/
├── run_service.py            # 启动服务
├── run_agent.py              # 命令行直接运行图
└── run_client.py             # 客户端用法示例
```

| 包 | 职责 | 关键约束 |
|---|---|---|
| `agents/` | agent 注册表与调研图（5 个节点） | 只有 `research_team/` 一个业务 agent，其余基础设施不为新增 agent 复制 |
| `service/` | FastAPI 路由、鉴权、SSE、会话列表、AG-UI 适配 | 端点的 `agent_id` 一律经注册表解析，不硬编码 |
| `client/` | 与 HTTP 解耦的调用封装 | 不依赖 `agents/`，只依赖 `schema/` |
| `core/` | 配置、模型工厂、请求指标 | 指标用 contextvar，不写入 `ResearchState` |
| `memory/` | 三套持久化后端 | 切换只改 `DATABASE_TYPE`，业务代码不感知 |
| `schema/` | 对外协议模型 | 服务端、客户端、界面共用同一套定义 |
| `evaluation/` | 评估判据 | 做确定性结构与趋势证据检查，不评判分析质量 |
| `voice/` | STT/TTS 抽象与 provider | 可独立使用，服务端不感知；当前无界面集成 |

`research_team` 内部按依赖方向组织：`graph.py` 只负责节点装配；节点读取并返回
`ResearchState` 的增量更新；`contracts.py` 定义跨节点共享的领域对象；`search.py`
只封装外部 Search / Fetch；`helpers.py` 提供模型结构化输出与来源归一化。数据流为
`Supervisor(tasks, constraints) → Researcher(sources) → Analyst(analysis, coverage) →
Writer(report) → finalize(messages)`，节点之间不直接调用彼此。

仓库其他目录：

```text
static/    # 纯静态项目展示页（HTML / CSS / JS），可独立部署到静态托管
  js/evaluation-data.js  # 只读离线评测快照；不作为线上运行数据
  js/runtime-records.js  # /runtime-metrics 响应归一化与真实运行记录边界
evals/     # 5 个用例、评分脚本、Writer A/B 脚本与 runs/ 历史记录
tests/     # pytest，按包分层，含 service 的端到端与 smoke 持久化用例
scripts/   # ab_bench.py、AG-UI 演示客户端（Node）
docs/      # AG-UI 接入说明、文件式凭据说明
docker/    # 服务端镜像、Mongo 专用 compose
```

## 开发

```sh
uv run ruff format --check
uv run ruff check
uv run pyrefly check
uv run pytest
```

`tests/` 与 `src/` 的包一一对应（agents / client / core / evaluation / schema / service / voice），
另有 `tests/integration` 与 `tests/smoke` 两组需要运行中服务的用例，默认跳过，加 `--run-docker` 才执行
（这两个目录复用 `USE_FAKE_MODEL=true` 与服务端配置，`smoke` 还会校验 checkpoint 确实落到当前
`DATABASE_TYPE` 对应的后端）。

LangGraph Studio：

```sh
uv run langgraph dev
```

`langgraph.json` 指向 `research_supervisor` 图。

## 检索与计费

检索走 DeepSeek 官方的 Search / Fetch 两个独立接口，与 LLM 调用分开计费：

| 接口 | 用途 | 价格 |
|---|---|---|
| `POST /v1/tools/search` | 按 query 发现页面，返回标题、URL、摘要、`authority`、`rerank_score`、发布时间 | ¥5 / 1000 次成功请求 |
| `POST /v1/tools/fetch` | 把指定 URL 抓成 Markdown 正文 | ¥5 / 1000 次成功请求 |

计费只发生在 HTTP 200 且返回内容非空时，失败或空结果不计费，因此可以安全重试。

一次典型调研的实测消耗（2025 年储能电芯问题）：**4 次 Search + 3 次 Fetch = 7 次请求 ≈ ¥0.035**，加上 LLM token，合计在人民币几分钱量级。

两个设计要点：

- **来源评分不由模型猜**：`confidence` 直接由 `authority`（-1~4）与 `rerank_score`（0~3）归一化得出，见 `helpers.confidence_from_authority`。它是来源层面的检索信号，不等同于子问题结论的证据强度。
- **深读有上限**：只对 `authority ≥ 3` 或 `rerank_score ≥ 1.5` 的前 3 条来源调用 Fetch，并按 4000 字截断，避免正文 token 失控。
- **Judge 前硬过滤**：先丢弃非对象、不可抓取 URL 和标题/摘要/snippet 全空的结果，再按 `url_key` 去重、按候选上限截断，最后才交给 Judge；不按低质量评分删除正常来源。每轮指标记录各类过滤数量，并区分空结果、全无效和全重复。

Search 会同时返回同一篇文章的简繁体版本（`/cn/`、`/tw/`）与手机版（`m.` 主机），`url_key` 会按语言段与主机前缀（`www.` / `m.`）归一化后再去重，否则一篇报道会被当成多个独立来源参与交叉验证。

## 数据存储

三个后端都在 `memory/` 下，切换只改 `.env`，不改代码：

| 后端 | 配置 | 说明 |
|---|---|---|
| SQLite | `DATABASE_TYPE=sqlite` | 默认，单文件，适合单机 |
| PostgreSQL | `DATABASE_TYPE=postgres` + `POSTGRES_*` | 本机用 `docker compose up -d postgres` 即可 |
| MongoDB | `DATABASE_TYPE=mongo` + `MONGO_*` | 仅 checkpoint；长期记忆 store 暂回退到进程内存储 |

当前使用 PostgreSQL 16（本机 Docker，`agent_service` 库），checkpoint 落在 `checkpoints` /
`checkpoint_writes` / `checkpoint_blobs`，长期记忆落在 `store`。
SQLite 和 MongoDB 模式的长期记忆目前使用进程内存储，服务重启后不会保留。
数据总览的运行记录落在 `turn_metrics` 表，跟随 `DATABASE_TYPE`：SQLite 与
PostgreSQL 重启后保留，MongoDB 回退到进程内统计（重启清零）。记录包含 UTC
开始/结束时间、模型、问题摘要、报告标题、来源数、耗时、Token、检索调用和错误详情；
问题内容只保存截断摘要，不保存完整报告；API 返回时会隐藏用户 ID，仅用于服务端过滤。

**Windows 注意**：psycopg 的异步模式不能跑在 Proactor 事件循环上，而 uvicorn 0.54 在
Windows 上硬编码 Proactor。`run_service.py` 已在「Windows + PostgreSQL」时改为手动构造
`uvicorn.Server` 并以 Selector 循环运行，代价是这条路径上没有热重载。

## 语音（可选）

`voice/` 提供可独立使用的 STT / TTS 抽象与 OpenAI provider 实现，不依赖界面框架，服务端对它无感知。当前项目没有界面集成，留作后续接入使用。

| 变量 | 作用 | 当前支持 |
|---|---|---|
| `VOICE_STT_PROVIDER` | 语音识别 | 仅 `openai`（`deepgram` 有类型定义，调用会抛 `NotImplementedError`） |
| `VOICE_TTS_PROVIDER` | 语音合成 | 仅 `openai`（`elevenlabs` 同上） |

- 两者留空即关闭。
- `SpeechToText` / `TextToSpeech` 可单独使用，各自通过环境变量选择 provider。
- provider 初始化失败只记日志并降级，不会让服务起不来——语音是能少不能崩的旁路。

## 评估

`evals/cases.json` 固定 5 个用例（4 个调研 + 1 个负样本），判据全部是确定性的：

| 判据 | 检查内容 |
|---|---|
| `constraint:*` | 约束是否落到 `constraints` |
| `exclude:*` | 排除项是否渗入正文（否定句不算违规） |
| `table` / `bullets` | 交付格式是否满足 |
| `anchors` | 引用锚点是否越界到未采集的来源 |
| `complete` | 摘要/分析/结论与发现条目以句读或锚点收尾，拦截半句话截断 |
| `meta` | 正文是否出现元文本（自检清单、「来源编号表/给定来源」等内部输入术语；关键词代理） |
| `sources_min` / `sources_max` | 来源数量 |
| `sections` | 四段式结构是否完整 |
| `caveat` | 是否提示单源/矛盾/口径（关键词代理，非质量评判） |
| `hint` | 负样本是否返回识别失败提示 |
| `trend` | 强趋势措辞是否有三个同口径时间点或来源明确声明，并检查规划/预测不得冒充实际趋势 |

```sh
PYTHONPATH=src uv run python evals/run_eval.py          # 全部
PYTHONPATH=src uv run python evals/run_eval.py catl     # 按 id 过滤
```

每次运行写入 `evals/runs/<时间戳>/`，含每例完整报告与 `summary.md`。改 prompt、换模型后重跑，
与上一次的 `summary.md` 逐行对比即可判断是否退化。按 id 过滤时同样落盘，但 `summary.md` 里只有
被跑的那几例——和全量结果比对前先看清用例数。2026-11-14 加入 `meta` 判据后总判据 32→36；
2026-11-16 加入 `trend` 判据后，全量总判据为 41，
历史 runs 为 32 判据，跨版本对比先核对判据数。成本列按仓库价表估算，只含 LLM token、不含
Search/Fetch 费用；`-1017` 这类快照后缀按基础模型计价，未知模型显示 `—`。

Writer 上下文的对照实验单独跑，结果写到 `evals/ab_writer_<时间戳>.json`，不进 `runs/`：

```sh
PYTHONPATH=src uv run python evals/ab_writer.py
```

基线（2026-11-14T14:04 全量运行，`deepseek-v4.1-flash-1017` 关闭思考，Writer 不接收原文摘录）：
**32/32（100%）**，四个调研用例单轮输入 19k–28k token、耗时 24–46s，负样本 0.8s / 600 token
（不经模型）。`complete` 判据与 Writer 的句读级完整性检查同日加入：此前约束解码曾产出
「合法 JSON 但字段半句话」的报告而旧判据全绿，新判据专门拦截这类截断。
同日加入进度守卫后（14:41）重跑 **32/32**：catl-byd-europe / reducer-substitution 两例触发守卫被救回，
单用例 search 稳定为 4；对比守卫前 14:23 运行（29/32，catl 空转 20 次 search 零产出），
全量 search 32→16、输入 token 115k→86k。
同日 `sub_questions` / `constraints` 必填化后（15:00）：调研用例 tasks 从常态 0 条变为 3–4 条
拆解，守卫与兜底均未触发；catl 三连跑 8/8（其中一次 `length_retry` 被容器化）。全量曾出现一次
约束台账漏报（欧洲），三连跑未复现，属模型概率性少报而非结构回归。
同日加入 `meta` 判据与 Writer 元文本禁令后（16:52）：**36/36**，四份报告均以「现有来源」描述
缺口，内部术语零出现。判据上线前对 81 份历史报告离线回放：精确命中 9 份泄漏（3 份重度 QA
自检块 + 6 份轻度术语），72 份干净报告零误报。成本列同日可用（快照后缀查表修复），该轮全量
LLM ≈¥0.51。

Writer 只拿编号索引、不拿原文摘录，是实测结论而非猜测：同一批来源下，带 12 条摘录的版本
多花约 8k 输入 token，报告反而更短，其中一次分析段被截断（`出货格局呈` 后直接进结论）。
分析薄于 1500 字时会自动回退到带摘录，见 `writer.MIN_ANALYSIS_CHARS_FOR_INDEX`。

## 观测

把 `LOG_LEVEL` 设为 `INFO` 后，每个请求结束时会打一行结构化日志；默认的 `WARNING` 级别下这一行不会出现：

```text
turn thread=metrics-smoke-4 status=ok elapsed=29.4s llm_calls=8 tokens_in=26196
tokens_out=3866 reasoning=0 search=4 fetch=2 search_failed=0 fetch_failed=1
fetch_skipped=0 degraded=-
```

| 字段 | 含义 |
|---|---|
| `status` | `ok` / `error` / `closed`（客户端提前断开） |
| `llm_calls` | 本轮模型调用次数，含结构化输出失败后的重试 |
| `tokens_in` / `tokens_out` / `reasoning` | 输入、输出与思考 token |
| `search` / `fetch` | 成功调用次数；失败单独计数 |
| `fetch_skipped` | 深读目标因 URL 不可 fetch 被跳过的次数；未发起请求，不计入 `fetch_failed` |
| `degraded` | 本轮触发的降级路径，逗号分隔、同值去重；无则 `-` |

计数器存放在 `core/metrics.py` 的 contextvar 中，不写入 `ResearchState`——它描述一次请求，
不是对话历史。9 个降级标记由各节点调用 `record_degradation()` 写入：`query_planning_failed` /
`query_planning_empty` / `judge_fallback` / `search_rejected`（researcher）、`analysis_incomplete`（analyst）、
`report_incomplete`（writer）、`length_retry`（helpers 的结构化输出重试）、
`researcher_forced_analyst` / `sub_questions_missing`（supervisor 的进度守卫与拆解兜底）。
`LOG_LEVEL=DEBUG` 时还会额外打一行按节点拆分的 token 归因
（`supervisor=1x/9000in/600out` 形式），排查「哪一步把预算吃掉了」比总数更有用。

## 已知限制

- **检索稳定性**依赖 DeepSeek 官方搜索服务，单账号限速 Search 100 QPS / Fetch 25 QPS；本项目每次检索最多 4 条查询并发（`researcher.MAX_QUERIES`），多轮累积也远低于上限。
- **调度轮次上限**为 5 轮（`supervisor.MAX_ROUNDS`），超出后强制收敛，有分析时返回阶段性分析。
  **分析前进度守卫**：已有来源而分析为空时，路由 researcher 会被降级为 analyst（`researcher_forced_analyst`），
  避免「分析前反复再检索」空转到上限、把已采集来源全部丢弃。
- **置信度来自检索接口**的 `authority` 与 `rerank_score` 归一化，是来源层级信号，不等同于事实核查结论。
- **约束解码退化**：`json_schema` 约束下 flash 模型偶发「合法 JSON 但字段半句话」（`finish_reason=stop`），
  Writer 的句读级完整性检查会拦截并重试一次（`report_incomplete`）；更罕见的失控生成会烧满该调用的
  token 预算——路由/查询规划/相关性判定/分析已按输出规模分级设上限（2048/2048/4096/8192），Writer 仍为
  `LLM_MAX_TOKENS`，失控因此被限制在较小预算内；`length_retry` 重试仍失败时整轮报错，已消耗的检索与 token 不退回。
