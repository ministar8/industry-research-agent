/* 调研档案：档案列表页与报告详情页 */
import { $, esc, icons, download } from './ui.js';
import { EVALUATION_SNAPSHOT, label } from './evaluation-data.js';
import { parseAnswer, markdown, sourceGroups } from './answer.js';

let current = null;

/* 判据名与 src/evaluation/scoring.py 的 run_checks 一一对应 */
const CHECK_LABELS = {
  constraint: (a) => `保留约束「${a}」`,
  exclude: (a) => `正文未断言排除项「${a}」`,
  mention: (a) => `回答提及「${a}」`,
  table: () => '包含数据表格（至少 3 行）',
  bullets: () => '列出要点（至少 5 条）',
  anchors: () => '引用编号都能对应到来源',
  sections: () => '摘要、主要发现、分析、结论齐全',
  complete: () => '各段落完整收尾，无半句截断',
  meta: () => '正文未混入内部说明文字',
  caveat: () => '提示单源、矛盾或口径差异',
  sources_min: (a) => `来源不少于 ${a} 条`,
  sources_max: (a) => (a === '0' ? '未检索任何来源' : `来源不超过 ${a} 条`),
  hint: () => '返回无法识别对象的提示',
};
function checkLabel(name) {
  const [key, ...rest] = name.split(':');
  return CHECK_LABELS[key]?.(rest.join(':')) ?? name;
}

function renderArchive() {
  $('recordsSubtitle').textContent = `${EVALUATION_SNAPSHOT.cases.length} 份 · 评估批次 ${EVALUATION_SNAPSHOT.run.date} · 模型 ${EVALUATION_SNAPSHOT.run.model}`;
  $('archiveList').innerHTML = EVALUATION_SNAPSHOT.cases.map((c, i) => {
    const [title, kind] = label(c);
    return `<li><a class="archive-row" href="#record/${encodeURIComponent(c.id)}">
      <span class="archive-index">${String(i + 1).padStart(2, '0')}</span>
      <span class="archive-main"><span class="archive-title">${esc(title)}</span><span class="archive-kind">${esc(kind)}</span></span>
      <span class="archive-metric"><b>${esc(c.source_count)}</b> 来源</span>
      <span class="archive-time">${esc(c.elapsed)} 秒</span>
      <span class="archive-cost">${esc(c.cost)}</span>
      <span class="archive-go" data-icon="arrow-up-right"></span>
    </a></li>`;
  }).join('');
  icons($('archiveList'));
}

export function renderRecord(record) {
  current = record;
  const { report, sources } = parseAnswer(record.answer);
  $('reportTitle').textContent = label(record)[0];
  $('reportMeta').innerHTML = [`${EVALUATION_SNAPSHOT.run.date} · 评估记录`, `${record.source_count} 条来源`, `${record.elapsed} 秒`, `LLM ${record.cost}`].map((s) => `<span>${esc(s)}</span>`).join('');
  $('reportBody').innerHTML = markdown(report, sources, 'record');
  $('reportSources').innerHTML = sourceGroups(sources, 'record', '此记录未采集来源。');
  $('sourceCount').textContent = sources.length;
  $('reportContext').innerHTML = `<span class="context-label">原始输入</span><p class="context-text">${esc(record.message)}</p><span class="context-label">解析意图</span><p>${esc(record.intent || '未识别到调研对象')}</p><span class="context-label">保留约束</span><div class="chip-row">${record.constraints.length ? record.constraints.map((c) => `<span class="neutral-chip">${esc(c)}</span>`).join('') : '<span class="muted">无约束</span>'}</div><span class="context-label">拆解问题</span><ol>${record.tasks.map((t) => `<li>${esc(t)}</li>`).join('') || '<li>未拆解出子问题</li>'}</ol>`;
  const metrics = [['模型调用', `${record.metrics.llm_calls} 次`], ['输入 / 输出 token', `${record.metrics.tokens_in.toLocaleString()} / ${record.metrics.tokens_out.toLocaleString()}`], ['检索 / 抓取', `${record.metrics.search} / ${record.metrics.fetch}`], ['抓取失败', `${record.metrics.fetch_failed} 次`]];
  $('reportChecks').innerHTML = `<dl class="run-metrics">${metrics.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl><ul class="checks-list">${record.checks.map((c) => `<li><span class="check-chip${c.passed ? '' : ' fail'}">${c.passed ? '通过' : '未通过'}</span><span title="${esc(c.name)}">${esc(checkLabel(c.name))}${c.detail ? `<span class="check-detail">${esc(c.detail)}</span>` : ''}</span></li>`).join('')}</ul>`;
}

export function initReport() {
  renderArchive();
  $('exportReport').addEventListener('click', () => { if (current) download(`${label(current)[0]}.md`, current.answer, 'text/markdown;charset=utf-8'); });
}
