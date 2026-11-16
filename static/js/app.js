/* 入口：路由、侧栏菜单、全局快捷键，以及各模块的装配 */
import { $, icons, notify } from './ui.js';
import { DATA } from './data.js';
import { initSources } from './answer.js';
import { initReport, renderRecord } from './report.js';
import { initDashboard } from './dashboard.js';
import { initChat, newThread, loadHistory, isBusy, onConnected, onDisconnected } from './chat.js';
import { initConnection } from './connection.js';
import { renderCopy } from './copy.js';

const state = { focusQuestion: false };
const titles = { home: '新建调研', data: '数据总览', records: '调研档案', report: '调研档案', chat: '调研会话', about: '关于本项目' };
const mobileBreakpoint = () => parseInt(getComputedStyle(document.documentElement).getPropertyValue('--bp-mobile').trim()) || 900;

/* ── 路由与菜单 ── */
function closeMenu() {
  $('sidebar').classList.remove('is-open'); $('sidebarScrim').hidden = true;
  $('menuToggle').setAttribute('aria-expanded', 'false');
  $('sidebar').inert = matchMedia(`(max-width: ${mobileBreakpoint()}px)`).matches;
  $('workspace').inert = false;
}
function navigate(route) {
  if (location.hash.slice(1) === route) renderRoute(); else location.hash = route;
}
function renderRoute() {
  const [route, id] = location.hash.slice(1).split('/');
  let view = ['home', 'data', 'records', 'about', 'chat'].includes(route) ? route : 'home';
  if (route === 'record') {
    const record = DATA.cases.find((c) => encodeURIComponent(c.id) === id);
    if (record) { view = 'report'; renderRecord(record); }
    else notify('未找到这条记录，已返回首页。');
  }
  if (view === 'chat' && id) {
    try { loadHistory(decodeURIComponent(id)); } catch { notify('会话地址无效。'); }
  }
  for (const key of Object.keys(titles)) $(key + 'View').hidden = key !== view;
  $('pageTitle').textContent = titles[view]; document.title = `${titles[view]} · 研序`;
  const navKey = view === 'report' ? 'records' : view;
  document.querySelectorAll('[data-nav]').forEach((el) => {
    if (el.dataset.nav === navKey) el.setAttribute('aria-current', 'page');
    else el.removeAttribute('aria-current');
  });
  $(view === 'chat' ? 'chatComposer' : 'homeComposer').append($('composer'));
  $('question').placeholder = '描述对象、时间口径和关注的问题…';
  $('mainContent').scrollTop = 0; closeMenu();
  // 挂载输入框会让它失焦，所以等路由渲染完再聚焦
  if (state.focusQuestion) { state.focusQuestion = false; $('question').focus(); }
}

/* ── 初始化与全局事件 ── */
function init() {
  renderCopy(); icons(); closeMenu();
  if (!DATA.cases.length) {
    $('composer').hidden = true;
    $('homeComposer').innerHTML = '<p class="empty-state">评估数据未加载，请检查静态数据文件。</p>';
    window.addEventListener('hashchange', renderRoute);
    // 从窄屏切回桌面后，侧栏需要重新可交互（closeMenu 只在路由变化时执行）
    window.addEventListener('resize', () => {
      if (!matchMedia(`(max-width: ${mobileBreakpoint()}px)`).matches) {
        $('sidebar').inert = false; $('workspace').inert = false;
      }
    });
    renderRoute();
    return;
  }
  initSources(); initReport(); initDashboard(); initChat(navigate);
  initConnection({ isBusy, onConnected, onDisconnected });
  $('menuToggle').addEventListener('click', () => {
    const open = !$('sidebar').classList.contains('is-open');
    $('sidebar').classList.toggle('is-open', open); $('sidebarScrim').hidden = !open;
    $('sidebar').inert = !open; $('workspace').inert = open; $('menuToggle').setAttribute('aria-expanded', String(open));
    if (open) $('newResearch').focus();
  });
  $('sidebarScrim').addEventListener('click', closeMenu);
  $('newResearch').addEventListener('click', () => { if (newThread()) { state.focusQuestion = true; navigate('home'); closeMenu(); } });
  $('composerAdd').addEventListener('click', () => {
    if (isBusy()) { notify('当前调研仍在进行，请先停止等待。'); return; }
    if (newThread()) { closeMenu(); $('question').focus(); }
  });
  $('composerAttach').addEventListener('click', () => notify('暂未开放上传附件，可以把要点直接写在问题里。'));
  document.addEventListener('click', (event) => {
    const prompt = event.target.closest('[data-prompt]');
    if (!prompt) return;
    if (isBusy()) { notify('当前调研仍在进行。'); return; }
    $('question').value = DATA.cases[Number(prompt.dataset.prompt)].message; $('question').focus();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && $('sidebar').classList.contains('is-open')) { closeMenu(); $('menuToggle').focus(); }
    if (event.key === 'Tab' && $('sidebar').classList.contains('is-open')) {
      const items = [...$('sidebar').querySelectorAll('a, button:not(:disabled)')].filter((el) => el.getClientRects().length);
      if (event.shiftKey && document.activeElement === items[0]) { event.preventDefault(); items.at(-1).focus(); }
      else if (!event.shiftKey && document.activeElement === items.at(-1)) { event.preventDefault(); items[0].focus(); }
    }
    const editing = event.target.closest('input, textarea, select, [contenteditable]');
    if (event.key.toLowerCase() === 'n' && !editing && !event.ctrlKey && !event.metaKey && !event.altKey && !$('connectionDialog').open) { event.preventDefault(); $('newResearch').click(); }
  });
  window.addEventListener('hashchange', renderRoute);
  // 从窄屏切回桌面后，侧栏需要重新可交互（closeMenu 只在路由变化时执行）
  window.addEventListener('resize', () => {
    if (!matchMedia(`(max-width: ${mobileBreakpoint()}px)`).matches) {
      $('sidebar').inert = false; $('workspace').inert = false;
    }
  });
  renderRoute();
}
init();
