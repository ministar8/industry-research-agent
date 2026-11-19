/* 调研会话：发起调研、读取历史会话、对话页渲染 */
import { $, esc, notify, download } from './ui.js';
import { connection, apiJson, streamResearch } from './api.js';
import { parseAnswer, markdown, sourceGroups } from './answer.js';
import { lockConnect, openConnectionDialog } from './connection.js';
import { clearDashboard, refreshDashboard } from './dashboard.js';

const state = { threadId: null, controller: null, sequence: 0, historyVersion: 0, loadingHistory: false, sources: [], userId: crypto.randomUUID() };
// 非 JWT 模式没有会话身份，用设备级 ID 归组线程；JWT 模式下服务端以登录身份为准，忽略此值。
try { state.userId = localStorage.getItem('research-user-id') || state.userId; localStorage.setItem('research-user-id', state.userId); } catch { /* Storage may be disabled by the browser. */ }

export function isBusy() { return state.controller !== null || state.loadingHistory; }
function busy(value) {
  $('question').readOnly = value; $('sendQuestion').hidden = value; $('stopWaiting').hidden = !value;
  $('newResearch').disabled = value; lockConnect(value);
}

/* ── 对话页渲染 ── */
function addQuestion(text) {
  const node = document.createElement('p'); node.className = 'chat-question'; node.textContent = text;
  $('chatMessages').append(node);
}
/* 成稿报告按 ## 分段：对话流里只展示摘要与主要发现，其余段落折叠；
   非成稿输出（引导提示、阶段性分析、失败说明）没有这两段，保持全文展示。 */
function splitReport(report) {
  const sections = []; let preamble = '';
  for (const line of report.split('\n')) {
    const heading = /^##\s+(.+?)\s*$/.exec(line);
    if (heading) sections.push({ title: heading[1], lines: [line] });
    else if (sections.length) sections.at(-1).lines.push(line);
    else preamble += line + '\n';
  }
  const brief = sections.filter((s) => s.title === '摘要' || s.title === '主要发现');
  if (brief.length < 2) return null;
  const rest = sections.filter((s) => !brief.includes(s));
  if (!rest.length) return null;
  return {
    lead: [preamble.trimEnd(), ...brief.map((s) => s.lines.join('\n'))].join('\n\n'),
    rest: rest.map((s) => s.lines.join('\n')).join('\n\n'),
    restTitles: rest.map((s) => s.title).join('、'),
  };
}
function answerCard(answer, prefix) {
  const { report, sources } = parseAnswer(answer);
  const split = splitReport(report);
  const head = `<p class="section-kicker">调研结果 · 服务返回</p><h2 class="signature">研究结果</h2>`;
  if (!split) return `<article class="chat-answer">${head}<div class="markdown">${markdown(report, sources, prefix)}</div></article>`;
  return `<article class="chat-answer">${head}<div class="markdown">${markdown(split.lead, sources, prefix)}</div><details class="chat-answer-rest"><summary>展开完整报告 · ${esc(split.restTitles)}</summary><div class="markdown">${markdown(split.rest, sources, prefix)}</div></details></article>`;
}
function addAnswer(text) {
  const prefix = `chat-${++state.sequence}`;
  const { sources } = parseAnswer(text);
  state.sources.push(...sources);
  if (state.sources.length === sources.length) $('chatSources').replaceChildren();
  if (sources.length) {
    $('chatSources').insertAdjacentHTML('beforeend', sourceGroups(sources, prefix, '本次回答未附来源。'));
  } else if (!state.sources.length) {
    $('chatSources').innerHTML = sourceGroups([], prefix, '本次回答未附来源。');
  }
  $('chatSourceCount').textContent = state.sources.length ? `${state.sources.length} 条` : '';
  const box = document.createElement('div'); box.innerHTML = answerCard(text, prefix);
  const button = document.createElement('button'); button.className = 'button'; button.textContent = '导出报告';
  button.addEventListener('click', () => download('研序-调研报告.md', text, 'text/markdown;charset=utf-8'));
  box.querySelector('article').append(button); $('chatMessages').append(box);
}
function addProcess(messages) {
  if (!messages.length) return;
  const node = document.createElement('details'); node.className = 'chat-process';
  node.innerHTML = `<summary>查看服务过程消息 · ${messages.length} 条</summary>${messages.map((m) => `<p>${esc(m)}</p>`).join('')}`;
  $('chatMessages').append(node);
}

/* ── 会话 ── */
export function newThread() {
  if (isBusy()) { notify('请先停止等待，再新建调研。'); return false; }
  state.historyVersion++; state.threadId = null; state.sources = []; $('chatSources').innerHTML = '<li class="src-empty">发起一次调研后，引用的来源会按站点归到这里。</li>'; $('chatSourceCount').textContent = ''; $('chatMessages').replaceChildren(); $('question').value = ''; return true;
}
export async function loadHistory(id) {
  if (isBusy()) { notify('当前调研仍在进行，暂不切换会话。'); return; }
  if (!connection.connected) { notify('请先连接服务以读取会话。'); return; }
  if (id === state.threadId) return;
  const version = ++state.historyVersion; state.loadingHistory = true; busy(true); $('stopWaiting').hidden = true;
  state.threadId = null; state.sources = []; $('chatSources').innerHTML = '<li class="src-empty">正在读取会话来源…</li>'; $('chatSourceCount').textContent = ''; $('chatMessages').textContent = '正在读取会话…';
  try {
    const result = await apiJson('history', { thread_id: id, user_id: state.userId });
    if (version !== state.historyVersion || !connection.connected) return;
    state.threadId = id; $('chatMessages').replaceChildren(); $('question').value = '';
    let answers = [];
    const flush = () => { if (answers.length) { addProcess(answers.slice(0, -1)); addAnswer(answers.at(-1)); answers = []; } };
    for (const message of result.messages) {
      if (message.type === 'human') { flush(); addQuestion(message.content); }
      if (message.type === 'ai' && message.content) answers.push(message.content);
    }
    flush(); if (!result.messages.length) $('chatMessages').textContent = '此会话尚无消息。';
  } catch (error) { $('chatMessages').textContent = '无法读取该会话，可从新建调研开始。'; notify(error.message || '读取会话失败。'); }
  finally { state.loadingHistory = false; busy(false); }
}
async function refreshHistory() {
  if (!connection.connected) return;
  const version = state.historyVersion;
  try {
    const result = await apiJson(`threads?user_id=${encodeURIComponent(state.userId)}`);
    if (!connection.connected || version !== state.historyVersion) return;
    $('historySection').hidden = false;
    $('historyNav').innerHTML = result.threads.map((t) => `<a href="#chat/${encodeURIComponent(t.thread_id)}" title="${esc(t.title || '未命名调研')}"><span>${esc((t.title || '未命名调研').slice(0, 32))}</span></a>`).join('') || '<p class="field-help">还没有服务端会话。</p>';
  } catch { $('historySection').hidden = false; $('historyNav').textContent = '会话读取失败，请稍后重试。'; }
}

async function sendQuestion(event, navigate) {
  event.preventDefault();
  const message = $('question').value.trim();
  if (!message) { notify('请先填写调研问题。'); $('question').focus(); return; }
  if (message.length > 20000) { notify('问题不能超过 20,000 字符。'); return; }
  if (isBusy()) return;
  if (!connection.connected) { openConnectionDialog(); return; }
  state.historyVersion++; state.threadId ||= crypto.randomUUID(); state.controller = new AbortController();
  busy(true); navigate('chat'); addQuestion(message); $('question').value = '';
  const status = document.createElement('p'); status.className = 'chat-status'; status.setAttribute('role', 'status');
  $('chatMessages').append(status); const start = Date.now(); const messages = [];
  const update = () => { status.textContent = `调研进行中 · 已等待 ${Math.floor((Date.now() - start) / 1000)} 秒${messages.length ? ` · 收到 ${messages.length} 条服务消息` : ''}`; };
  update(); const timer = setInterval(update, 1000);
  try {
    for await (const text of streamResearch({ message, thread_id: state.threadId, user_id: state.userId }, state.controller.signal)) { if (text.trim()) messages.push(text); update(); }
    if (!messages.length) throw new Error('服务已结束，但未返回研究结果。');
    status.remove(); addProcess(messages.slice(0, -1)); addAnswer(messages.at(-1));
    notify('调研结果已返回，可查看来源或导出报告。');
  } catch (error) {
    status.classList.add('error');
    status.textContent = error.name === 'AbortError' ? '已停止等待。后端可能仍在运行或计费，可稍后从会话记录查看结果。' : error.message || '网络连接失败，未自动重试。';
    if (messages.length) addProcess(messages);
    $('question').value = message;
  } finally { clearInterval(timer); state.controller = null; busy(false); refreshHistory(); }
}

/* ── 连接状态回调（由 connection.js 触发） ── */
export function onConnected() { state.historyVersion++; refreshHistory(); refreshDashboard(); }
export function onDisconnected() { state.controller?.abort(); state.historyVersion++; $('historySection').hidden = true; clearDashboard(); }

export function initChat(navigate) {
  $('composer').addEventListener('submit', (e) => sendQuestion(e, navigate));
  $('question').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !e.isComposing) { e.preventDefault(); $('composer').requestSubmit(); } });
  $('stopWaiting').addEventListener('click', () => state.controller?.abort());
}
