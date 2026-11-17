/* 数据总览：指标卡、图表、评估明细表与 CSV 导出 */
import { $, esc, icon, download } from './ui.js';
import { fetchRuntimeRecords } from './runtime-records.js';
const state = { sort: null, visible: [], records: [] };
const title = (record) => record.report_title || record.query_summary || record.thread_id || '未命名会话';
const searchText = (record) => [title(record), record.thread_id, record.model].join(' ').toLocaleLowerCase();

function renderDashboard() {
  const query = $('recordSearch').value.trim().toLocaleLowerCase();
  const filter = $('recordFilter').value;
  state.visible = state.records.filter((c) => searchText(c).includes(query) && (filter === 'all' || (filter === 'research' ? c.status === 'ok' : c.status !== 'ok')));
  if (state.sort) state.visible.sort((a, b) => state.sort * (a.elapsed - b.elapsed));
  renderSummary(); renderCharts(); renderRows();
}

function renderSummary() {
  const completed = state.visible.filter((c) => c.status === 'ok').length;
  const llmCalls = state.visible.reduce((sum, c) => sum + c.llm_calls, 0);
  const tokensIn = state.visible.reduce((sum, c) => sum + c.tokens_in, 0);
  const tokensOut = state.visible.reduce((sum, c) => sum + c.tokens_out, 0);
  const stats = [
    ['实际运行', state.visible.length, `${completed} 次完成 · ${state.visible.length - completed} 次异常或中断`],
    ['完成率', state.visible.length ? `${Math.round(completed / state.visible.length * 100)}%` : '—', '来自服务实际返回状态'],
    ['LLM 调用', llmCalls, '来自本进程实际记录'],
    ['Token 用量', `${((tokensIn + tokensOut) / 1000).toFixed(1)}k`, `输入 ${tokensIn} · 输出 ${tokensOut}`],
  ];
  $('summaryMetrics').innerHTML = stats.map(([title, value, note]) => `<div class="summary-item"><p class="summary-label">${esc(title)}</p><p class="summary-value">${esc(value)}</p><p class="summary-note">${esc(note)}</p></div>`).join('');
}

function renderCharts() {
  const maxDuration = Math.max(1, ...state.visible.map((c) => c.elapsed));
  const maxTokens = Math.max(1, ...state.visible.map((c) => c.tokens_in + c.tokens_out));
  $('durationChart').innerHTML = state.visible.map((c) => `<div class="bar-row"><span class="bar-label" title="${esc(title(c))}">${esc(title(c).slice(0, 12))}</span><span class="bar-track" aria-hidden="true"><span class="bar-fill" style="--w:${c.elapsed / maxDuration * 100}%"></span></span><span class="bar-value">${esc(c.elapsed)}</span></div>`).join('') || '<p class="empty-state">连接服务并完成一次调研后显示</p>';
  $('tokenChart').innerHTML = state.visible.map((c) => {
    const total = c.tokens_in + c.tokens_out;
    return `<div class="bar-row" aria-label="${esc(title(c))}：输入 ${esc(c.tokens_in)}，输出 ${esc(c.tokens_out)} token"><span class="bar-label">${esc(title(c).slice(0, 12))}</span><span class="bar-track" aria-hidden="true" title="输入 ${esc(c.tokens_in)} / 输出 ${esc(c.tokens_out)}"><span class="token-input" style="--in:${c.tokens_in / maxTokens * 100}%"></span><span class="token-output" style="--out:${c.tokens_out / maxTokens * 100}%"></span></span><span class="bar-value">${esc((total / 1000).toFixed(1))}k</span></div>`;
  }).join('') || '<p class="empty-state">连接服务并完成一次调研后显示</p>';
}

function renderRows() {
  $('recordsBody').innerHTML = state.visible.map((c) => {
    const success = c.status === 'ok';
    const detail = c.query_summary && c.report_title ? c.query_summary : `${c.model} · ${c.sources_count} 条来源`;
    return `<tr><td><span class="record-name" title="${esc(c.query_summary || '')}">${esc(title(c))}</span><span class="record-subtitle">${esc(detail)}</span></td><td><span class="check-chip${success ? '' : ' fail'}" title="${esc(c.error_detail || '')}">${success ? '完成' : esc(c.status)}</span></td><td>${esc(c.elapsed)} 秒</td><td>${esc(c.sources_count)} 条</td><td>${esc(c.llm_calls)} 次</td><td>${esc(((c.tokens_in + c.tokens_out) / 1000).toFixed(1))}k</td><td>${icon('arrow-up-right')}</td></tr>`;
  }).join('') || '<tr><td colspan="7" class="empty-state">暂无真实运行记录，请先连接服务并发起调研。</td></tr>';
  $('tableFooter').textContent = `显示 ${state.visible.length} / ${state.records.length} 条真实运行记录`;
}

function csvCell(value) {
  let text = String(value ?? '');
  if (/^[=+\-@\t\r\n]/.test(text)) text = `'${text}`;
  return `"${text.replace(/"/g, '""')}"`;
}

export function initDashboard() {
  $('dataSubtitle').textContent = '服务真实运行记录，持久化保存，不使用评测样例';
  $('recordSearch').addEventListener('input', renderDashboard);
  $('recordFilter').addEventListener('change', renderDashboard);
  $('sortDuration').addEventListener('click', () => {
    state.sort = state.sort === 1 ? -1 : 1;
    $('sortDuration').closest('th').setAttribute('aria-sort', state.sort === 1 ? 'ascending' : 'descending');
    renderDashboard();
  });
  $('exportData').addEventListener('click', () => {
    const rows = [['调研', '问题摘要', '线程', '状态', '开始时间', '结束时间', '模型', '耗时（秒）', '来源数', '检索调用', '模型调用', '输入 token', '输出 token', '错误'], ...state.visible.map((c) => [title(c), c.query_summary, c.thread_id, c.status, c.started_at, c.finished_at, c.model, c.elapsed, c.sources_count, c.search_calls + c.fetch_calls, c.llm_calls, c.tokens_in, c.tokens_out, c.error_detail])];
    download('研序-真实运行记录.csv', '\uFEFF' + rows.map((r) => r.map(csvCell).join(',')).join('\r\n'), 'text/csv;charset=utf-8');
  });
  renderDashboard();
}

export async function refreshDashboard() {
  try {
    state.records = await fetchRuntimeRecords();
    renderDashboard();
  } catch {
    state.records = [];
    renderDashboard();
  }
}

export function clearDashboard() {
  state.records = [];
  renderDashboard();
}
