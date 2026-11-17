/* 界面说明文案（数据总览底注、关于页）及其填充；数量从数据计算，不写死 */
import { $ } from './ui.js';
import { EVALUATION_SNAPSHOT, isNegative } from './evaluation-data.js';

const DATA_NOTES = `<div class="data-notes"><span data-icon="circle-help"></span><p>这里显示服务实际完成的调研运行，不是内置评测样例。运行记录持久化保存（SQLite/PostgreSQL），重启后保留；Token 用量来自服务实际记录，不代表费用。</p></div>`;

const aboutSections = (total, research, negative) => `<section class="about-section"><h2>调度与执行分离</h2><dl class="role-list"><div><dt>调度主控</dt><dd>解析意图、保留约束、拆解任务与路由，不代替子智能体检索和写作。</dd></div><div><dt>资料检索</dt><dd>按调研范围检索，累积来源，记录链接与来源信号。</dd></div><div><dt>信息分析</dt><dd>去重与交叉验证，识别口径分歧和信息缺口。</dd></div><div><dt>报告撰写</dt><dd>组织摘要、发现、分析与结论，校验引用编号。</dd></div><div><dt>结果组装</dt><dd>拼接已有报告与来源，不调用模型。</dd></div></dl></section>
            <section class="about-section"><h2>理解数据边界</h2><ul class="plain-list"><li>当前内置 ${total} 条评估记录，含 ${research} 个调研用例和 ${negative} 个负样本。负样本仍有意图识别开销，不等于全程零模型调用。</li><li>来源置信度是检索信号，不是事实核查结论；结构判据通过也不等于分析正确。</li><li>报告一次性完成，不逐字生成。等待期间只呈现服务实际返回的信息，不伪造进度百分比。</li><li>检索受上游服务与限流影响；多行业问题建议拆成不同会话。</li><li>此界面是个人工作台，不提供多租户权限隔离；不要直接暴露到公网。</li><li>报告由模型生成，请结合原始来源核对结论。</li></ul></section>`;

export function renderCopy() {
  const negative = EVALUATION_SNAPSHOT.cases.filter(isNegative).length;
  $('dataNotes').innerHTML = DATA_NOTES;
  $('aboutSections').innerHTML = aboutSections(EVALUATION_SNAPSHOT.cases.length, EVALUATION_SNAPSHOT.cases.length - negative, negative);
  $('aboutRun').textContent = `评估批次：${EVALUATION_SNAPSHOT.run.date} · 模型：${EVALUATION_SNAPSHOT.run.model}。问答需连接后端，查看静态记录不产生费用。`;
}
