/* 数据总览：指标卡、图表、评估明细表与 CSV 导出 */
import { $, esc, icon, download } from './ui.js';
import { DATA, label, isNegative } from './data.js';
const state = { sort: null, visible: [] };
const money = (value) => /^¥\d+(?:\.\d+)?$/.test(value) ? Number(value.slice(1)) : null;
const isResearch = (c) => !isNegative(c);

function renderDashboard() {
  const query = $('recordSearch').value.trim().toLocaleLowerCase();
  const filter = $('recordFilter').value;
  state.visible = DATA.cases.filter((c) => `${c.title} ${c.message} ${label(c).join(' ')}`.toLocaleLowerCase().includes(query) && (filter === 'all' || (filter === 'research' ? isResearch(c) : !isResearch(c))));
  if (state.sort) state.visible.sort((a, b) => state.sort * (a.elapsed - b.elapsed));
  renderSummary(); renderCharts(); renderRows();
}

function renderSummary() {
  const checks = state.visible.flatMap((c) => c.checks);
  const costs = state.visible.map((c) => money(c.cost));
  const totalCost = costs.every((c) => c !== null) ? `¥${costs.reduce((a, b) => a + b, 0).toFixed(2)}` : '未记录';
  const sourceCount = state.visible.reduce((sum, c) => sum + c.source_count, 0);
  const stats = [
    ['评估记录', state.visible.length, `${state.visible.filter(isResearch).length} 个调研 · ${state.visible.filter((c) => !isResearch(c)).length} 个负样本`],
    ['结构判据通过', `${checks.filter((c) => c.passed).length} / ${checks.length}`, '只检查结构，不评质量'],
    ['累计来源条数', sourceCount, '各用例累加，未跨用例去重'],
    ['LLM 成本', totalCost, '按记录汇总，不含检索费用'],
  ];
  $('summaryMetrics').innerHTML = stats.map(([title, value, note]) => `<div class="summary-item"><p class="summary-label">${esc(title)}</p><p class="summary-value">${esc(value)}</p><p class="summary-note">${esc(note)}</p></div>`).join('');
}

function renderCharts() {
  const maxDuration = Math.max(1, ...state.visible.map((c) => c.elapsed));
  const maxTokens = Math.max(1, ...state.visible.map((c) => c.metrics.tokens_in + c.metrics.tokens_out));
  $('durationChart').innerHTML = state.visible.map((c) => `<div class="bar-row"><span class="bar-label" title="${esc(c.title)}">${esc(label(c)[2])}</span><span class="bar-track" aria-hidden="true"><span class="bar-fill" style="--w:${c.elapsed / maxDuration * 100}%"></span></span><span class="bar-value">${esc(c.elapsed)}</span></div>`).join('') || '<p class="empty-state">没有匹配记录</p>';
  $('tokenChart').innerHTML = state.visible.map((c) => {
    const total = c.metrics.tokens_in + c.metrics.tokens_out;
    return `<div class="bar-row" aria-label="${esc(label(c)[2])}：输入 ${esc(c.metrics.tokens_in)}，输出 ${esc(c.metrics.tokens_out)} token"><span class="bar-label">${esc(label(c)[2])}</span><span class="bar-track" aria-hidden="true" title="输入 ${esc(c.metrics.tokens_in)} / 输出 ${esc(c.metrics.tokens_out)}"><span class="token-input" style="--in:${c.metrics.tokens_in / maxTokens * 100}%"></span><span class="token-output" style="--out:${c.metrics.tokens_out / maxTokens * 100}%"></span></span><span class="bar-value">${esc((total / 1000).toFixed(1))}k</span></div>`;
  }).join('') || '<p class="empty-state">没有匹配记录</p>';
}

function renderRows() {
  $('recordsBody').innerHTML = state.visible.map((c) => {
    const passed = c.checks.filter((x) => x.passed).length;
    return `<tr><td><a class="record-name" href="#record/${encodeURIComponent(c.id)}">${esc(label(c)[0])}</a><span class="record-subtitle">${esc(label(c)[1])} · ${esc(label(c)[3])}</span></td><td><span class="check-chip${passed === c.checks.length ? '' : ' fail'}">${passed} / ${c.checks.length}</span></td><td>${esc(c.elapsed)} 秒</td><td>${esc(c.source_count)}</td><td>${esc(c.metrics.llm_calls)} 次</td><td>${esc(c.cost)}</td><td><a class="icon-button" href="#record/${encodeURIComponent(c.id)}" aria-label="查看${esc(label(c)[0])}">${icon('arrow-up-right')}</a></td></tr>`;
  }).join('') || '<tr><td colspan="7" class="empty-state">没有匹配的记录，请更换关键词或筛选条件。</td></tr>';
  $('tableFooter').textContent = `显示 ${state.visible.length} / ${DATA.cases.length} 条记录 · ${DATA.run.date} 评估批次`;
}

function csvCell(value) {
  let text = String(value ?? '');
  if (/^[=+\-@\t\r\n]/.test(text)) text = `'${text}`;
  return `"${text.replace(/"/g, '""')}"`;
}

export function initDashboard() {
  $('dataSubtitle').textContent = `${DATA.run.date} · ${DATA.run.model} · 静态评估快照`;
  $('recordSearch').addEventListener('input', renderDashboard);
  $('recordFilter').addEventListener('change', renderDashboard);
  $('sortDuration').addEventListener('click', () => {
    state.sort = state.sort === 1 ? -1 : 1;
    $('sortDuration').closest('th').setAttribute('aria-sort', state.sort === 1 ? 'ascending' : 'descending');
    renderDashboard();
  });
  $('exportData').addEventListener('click', () => {
    const rows = [['用例', '判据通过', '判据总数', '耗时（秒）', '来源条数', '模型调用', '输入 token', '输出 token', 'LLM 成本（不含检索）'], ...state.visible.map((c) => [c.title, c.checks.filter((x) => x.passed).length, c.checks.length, c.elapsed, c.source_count, c.metrics.llm_calls, c.metrics.tokens_in, c.metrics.tokens_out, c.cost])];
    download('研序-评估记录.csv', '\uFEFF' + rows.map((r) => r.map(csvCell).join(',')).join('\r\n'), 'text/csv;charset=utf-8');
  });
  renderDashboard();
}
