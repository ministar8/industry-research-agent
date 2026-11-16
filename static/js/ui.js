/* 通用 DOM 工具与图标；不依赖数据或页面 */
export const $ = (id) => document.getElementById(id);
export const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const paths = {
  'book-open': '<path d="M12 7v14m-9-3V3c4-1 7 0 9 2 2-2 5-3 9-2v15c-4-1-7 0-9 2-2-2-5-3-9-2Z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  paperclip: '<path d="m21 11.5-8.5 8.5a5 5 0 0 1-7-7l8.5-8.5a3.5 3.5 0 0 1 5 5L10.5 18a2 2 0 0 1-3-3l7.5-7.5"/>',
  'chart-column': '<path d="M3 3v18h18M9 17V9m4 8V5m4 12v-6"/>',
  'arrow-up-right': '<path d="M7 17 17 7M7 7h10v10"/>',
  'arrow-left': '<path d="M19 12H5m7 7-7-7 7-7"/>',
  'arrow-up': '<path d="M12 19V5m-7 7 7-7 7 7"/>',
  'chevron-right': '<path d="m9 18 6-6-6-6"/>',
  'chevron-down': '<path d="m6 9 6 6 6-6"/>',
  'panel-left': '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/>',
  'circle-help': '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3m.08 4h.01"/>',
  files: '<rect x="8" y="2" width="12" height="16" rx="2"/><path d="M4 6v14a2 2 0 0 0 2 2h10"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4m4-5 5 5 5-5m-5 5V3"/>',
  'arrow-down-up': '<path d="m3 16 4 4 4-4M7 20V4m14 4-4-4-4 4m4-4v16"/>',
  square: '<rect x="4" y="4" width="16" height="16" rx="2"/>',
  x: '<path d="m18 6-12 12M6 6l12 12"/>',
};
export function icon(name) {
  if (!paths[name]) console.warn(`未定义的图标：${name}`);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || ''}</svg>`;
}
export function icons(root = document) {
  root.querySelectorAll('[data-icon]').forEach((el) => { el.innerHTML = icon(el.dataset.icon); });
}
let notifyTimer;
export function notify(text) {
  clearTimeout(notifyTimer); $('toast').textContent = text; $('toast').hidden = false;
  notifyTimer = setTimeout(() => { $('toast').hidden = true; }, 3200);
}
export function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function safeUrl(value) {
  try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; }
  catch { return null; }
}
