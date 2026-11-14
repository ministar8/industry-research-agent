# 行业与竞品调研智能体

面向行业研究与竞品分析的**多智能体协作服务**。一个调度主控加三个专职子智能体，共享同一份调研状态，最终产出带引用锚点、可追溯来源的结构化报告。

技术栈：LangGraph（编排）+ FastAPI（服务）+ Streamlit（界面）+ DeepSeek V4.1-Flash（百炼托管，关闭思考）+ DeepSeek Search/Fetch（检索）。语音输入输出为可选项，走 OpenAI STT/TTS，仅在 Streamlit 端生效。

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
| **Supervisor** | 意图解析、任务拆解、路由、结果整合 | `intent` `tasks` `constraints` `plan_notes` `next_action` `rounds` `turn_attempted`（转入子节点时清空旧 `report`） |
| **Researcher** | 多轮检索，记录来源 URL、摘录与置信度 | `sources` `turn_attempted`；检索被拒时置 `abort_reason` |
| **Analyst** | 去重、交叉验证、识别矛盾与信息缺口 | `analysis` |
| **Writer** | 输出摘要 / 发现 / 分析 / 结论四段式报告，保留引用锚点 | `report` |
| **finalize** | 组装报告与来源清单；无成果时返回能力引导 | `messages` |

关键不变量（修改代码时不可破坏）：

1. `finalize` 只做组装，不做生成——它拼接已有的 `report` 或 `analysis`，或返回提示语，**不调用模型**。因此无调研目标的输入不产生任何模型开销。
2. 没有 chat 子 agent。与调研无关的输入由 Supervisor 剥离后继续调研；确实找不到调研对象时才落到 `finalize` 的提示语。不要再加回对话节点，那会让主控的路由判断变得含糊。
3. `sources` 跨轮累积，按 URL 去重，不覆盖。
4. 报告中每个引用锚点 `[n]` 必须能解析到 `sources` 中的一条记录，Writer 不得引入 Researcher 未采集的 URL。
5. 调度决策是结构化对象（`RoutingDecision`），不是自由文本，因此路由逻辑可在无 LLM 的情况下测试。

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

# 4. 另开终端启动界面
uv run streamlit run src/streamlit_app.py
```

服务默认监听 `http://0.0.0.0:8080`，界面在 `http://localhost:8501`。界面默认连接本机 `127.0.0.1:8080`，服务不在本机时用 `AGENT_URL` 指定。界面侧栏可切换模型、浏览历史会话，并支持可选的语音输入与朗读（见[语音](#语音可选)）。

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
| `AUTH_SECRET` | Bearer token；**未设置时除 `/health` 外所有端点无鉴权** | 空 |
| `HOST` / `PORT` / `MODE` | 监听地址、端口；`MODE=dev` 开启热重载 | `0.0.0.0` / `8080` / 空 |
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
| `GET /{agent_id}/threads` | 按 `user_id` 列出历史会话，最近更新在前 |
| `POST /feedback` | 星级反馈（LangSmith） |
| `POST /agui/{agent_id}/run` | AG-UI 协议端点，SSE；`/agui/run` 走默认 agent |
| `GET /health` | 健康检查，唯一不经过 bearer 鉴权的端点 |

`agent_id` 目前只有 `research-supervisor`。除 `/health` 外的端点共用 `AUTH_SECRET` bearer 校验；
`user_id` 由调用方自报，不与凭据绑定——面向终端用户前需要自己加一层授权。AG-UI 的接入细节见
[`docs/AGUI.md`](docs/AGUI.md)。

## 目录结构

```text
src/
├── agents/
│   ├── research_team/        # 唯一的业务 agent
│   │   ├── state.py          # ResearchState 与 Source / ResearchTask 模型
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
├── voice/                    # 可选语音：stt.py / tts.py 工厂、manager.py、providers/
├── streamlit_app.py          # 聊天界面
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
| `evaluation/` | 评估判据 | 只做确定性结构检查，不评判分析质量 |
| `voice/` | STT/TTS 抽象与 provider | 客户端能力，服务端不感知 |

仓库其他目录：

```text
evals/     # 5 个用例、评分脚本、Writer A/B 脚本与 runs/ 历史记录
tests/     # pytest，按包分层，含 app / service 的端到端与 smoke 持久化用例
scripts/   # ab_bench.py、AG-UI 演示客户端（Node）
docs/      # AG-UI 接入说明、文件式凭据说明
docker/    # 服务端与界面镜像、Mongo 专用 compose
```

## 开发

```sh
uv run ruff format --check
uv run ruff check
uv run pyrefly check
uv run pytest
```

`tests/` 与 `src/` 的包一一对应（agents / app / client / core / evaluation / schema / service / voice），
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

- **可信度不由模型猜**：`confidence` 直接由 `authority`（-1~4）与 `rerank_score`（0~3）归一化得出，见 `helpers.confidence_from_authority`。模型的职责只剩"是否与子问题相关"。
- **深读有上限**：只对 `authority ≥ 3` 或 `rerank_score ≥ 1.5` 的前 3 条来源调用 Fetch，并按 4000 字截断，避免正文 token 失控。

Search 会同时返回同一篇文章的简繁体版本（`/cn/`、`/tw/`），`url_key` 会按语言段归一化后再去重，否则一篇报道会被当成两个独立来源参与交叉验证。

## 数据存储

三个后端都在 `memory/` 下，切换只改 `.env`，不改代码：

| 后端 | 配置 | 说明 |
|---|---|---|
| SQLite | `DATABASE_TYPE=sqlite` | 默认，单文件，适合单机 |
| PostgreSQL | `DATABASE_TYPE=postgres` + `POSTGRES_*` | 本机用 `docker compose up -d postgres` 即可 |
| MongoDB | `DATABASE_TYPE=mongo` + `MONGO_*` | 仅 checkpoint；长期记忆 store 会回退到 SQLite |

当前使用 PostgreSQL 16（本机 Docker，`agent_service` 库），checkpoint 落在 `checkpoints` /
`checkpoint_writes` / `checkpoint_blobs`，长期记忆落在 `store`。

**Windows 注意**：psycopg 的异步模式不能跑在 Proactor 事件循环上，而 uvicorn 0.54 在
Windows 上硬编码 Proactor。`run_service.py` 已在「Windows + PostgreSQL」时改为手动构造
`uvicorn.Server` 并以 Selector 循环运行，代价是这条路径上没有热重载。

## 语音（可选）

语音是**客户端能力**，只在 Streamlit 界面生效：音频在浏览器端录、由 `voice/` 转成文本后，仍走普通的
`/stream` 调用，服务端对它无感知。

| 变量 | 作用 | 当前支持 |
|---|---|---|
| `VOICE_STT_PROVIDER` | 语音识别 | 仅 `openai`（`deepgram` 有类型定义，调用会抛 `NotImplementedError`） |
| `VOICE_TTS_PROVIDER` | 语音合成 | 仅 `openai`（`elevenlabs` 同上） |

- 两者留空即关闭；`VoiceManager.from_env()` 返回 `None`，界面自动回落到纯文本输入，朗读按钮置灰并提示去配 `.env`。
- `SpeechToText` / `TextToSpeech` 可以脱离 Streamlit 单独使用，`manager.py` 只负责把二者接到
  `st.chat_input(accept_audio=True)` 和消息渲染上。
- 转写结果直接作为用户输入文本进入会话；TTS 音频存放在 `st.session_state` 供重跑时复用，不写磁盘。
- provider 初始化失败只记日志并降级，不会让界面起不来——语音是能少不能崩的旁路。

## 评估

`evals/cases.json` 固定 5 个用例（4 个调研 + 1 个负样本），判据全部是确定性的：

| 判据 | 检查内容 |
|---|---|
| `constraint:*` | 约束是否落到 `constraints` |
| `exclude:*` | 排除项是否渗入正文（否定句不算违规） |
| `table` / `bullets` | 交付格式是否满足 |
| `anchors` | 引用锚点是否越界到未采集的来源 |
| `sources_min` / `sources_max` | 来源数量 |
| `sections` | 四段式结构是否完整 |
| `caveat` | 是否提示单源/矛盾/口径（关键词代理，非质量评判） |
| `hint` | 负样本是否返回识别失败提示 |

```sh
PYTHONPATH=src uv run python evals/run_eval.py          # 全部
PYTHONPATH=src uv run python evals/run_eval.py catl     # 按 id 过滤
```

每次运行写入 `evals/runs/<时间戳>/`，含每例完整报告与 `summary.md`。改 prompt、换模型后重跑，
与上一次的 `summary.md` 逐行对比即可判断是否退化。按 id 过滤时同样落盘，但 `summary.md` 里只有
被跑的那几例——和全量结果比对前先看清用例数。

Writer 上下文的对照实验单独跑，结果写到 `evals/ab_writer_<时间戳>.json`，不进 `runs/`：

```sh
PYTHONPATH=src uv run python evals/ab_writer.py
```

基线（2026-11-13T20:14 全量运行，`deepseek-v4.1-flash-1017` 关闭思考，Writer 不接收原文摘录）：
**28/28（100%）**，四个调研用例单轮输入 18k–29k token、耗时 24–103s，负样本 0.8s / 600 token
（不经模型）。

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
不是对话历史。7 个降级标记由各节点调用 `record_degradation()` 写入：`query_planning_failed` /
`query_planning_empty` / `judge_fallback` / `search_rejected`（researcher）、`analysis_incomplete`（analyst）、
`report_incomplete`（writer）、`length_retry`（helpers 的结构化输出重试）。
`LOG_LEVEL=DEBUG` 时还会额外打一行按节点拆分的 token 归因
（`supervisor=1x/9000in/600out` 形式），排查「哪一步把预算吃掉了」比总数更有用。

## 已知限制

- **检索稳定性**依赖 DeepSeek 官方搜索服务，单账号限速 Search 100 QPS / Fetch 25 QPS；本项目每次检索最多 4 条查询并发（`researcher.MAX_QUERIES`），多轮累积也远低于上限。
- **调度轮次上限**为 5 轮（`supervisor.MAX_ROUNDS`），超出后强制收敛并返回阶段性分析，不会无限循环。
- **置信度来自检索接口**的 `authority` 与 `rerank_score` 归一化，是来源层级信号，不等同于事实核查结论。
