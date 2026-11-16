/* 回答渲染（报告页与对话页共用）：解析报告与来源、Markdown 与引用编号、来源分组、引用跳转 */
import { $, esc, icon, safeUrl } from './ui.js';

export function parseAnswer(answer) {
  const parts = String(answer || '').split(/\n#{1,6}\s*参考来源\s*\n/);
  const sources = [];
  for (const line of (parts[1] || '').split('\n')) {
    const match = /^\s*\[(\d+)\]\s+\[(.+?)\]\((https?:\/\/[^\s)]+)\)\s*(.*)$/.exec(line);
    if (match && safeUrl(match[3]) && !sources.some((s) => s.id === match[1])) {
      sources.push({ id: match[1], title: match[2], url: safeUrl(match[3]), meta: match[4].replace(/^[—–-]\s*/, '') });
    }
  }
  return { report: parts[0].replace(/\n---+\s*$/, '').trim(), sources };
}

function inline(text, sources, prefix) {
  const pattern = /\[[^\]\n]+\]\(https?:\/\/[^\s)]+\)|\[\d{1,3}\]|\*\*[^*\n]+\*\*|`[^`\n]+`/g;
  let result = '', cursor = 0;
  for (const match of text.matchAll(pattern)) {
    result += esc(text.slice(cursor, match.index));
    const value = match[0];
    const link = /^\[([^\]]+)\]\((.+)\)$/.exec(value);
    if (link && safeUrl(link[2])) {
      result += `<a href="${esc(safeUrl(link[2]))}" target="_blank" rel="noopener noreferrer">${esc(link[1])}</a>`;
    } else if (/^\[\d+\]$/.test(value)) {
      const id = value.slice(1, -1);
      result += sources.some((source) => source.id === id)
        ? `<button class="citation" data-source="${esc(prefix)}-source-${id}" aria-label="查看来源 ${id}">${id}</button>`
        : `<span class="citation unavailable" title="未提供对应来源">${esc(value)}</span>`;
    } else if (value.startsWith('**')) result += `<strong>${esc(value.slice(2, -2))}</strong>`;
    else if (value.startsWith('`')) result += `<code>${esc(value.slice(1, -1))}</code>`;
    else result += esc(value);
    cursor = match.index + value.length;
  }
  return result + esc(text.slice(cursor));
}

export function markdown(text, sources = [], prefix = 'report') {
  const lines = String(text || '').replace(/\r/g, '').split('\n');
  const out = []; let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const cell = (s) => s.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((v) => v.trim());
  const format = (s) => inline(s, sources, prefix);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) { closeList(); continue; }
    if (line.startsWith('|') && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || '')) {
      closeList(); const columns = cell(line); i++;
      out.push(`<div class="table-scroll" tabindex="0" role="region" aria-label="报告数据表"><table><caption class="sr-only">报告数据表</caption><thead><tr>${columns.map((c) => `<th scope="col">${format(c)}</th>`).join('')}</tr></thead><tbody>`);
      while ((lines[i + 1] || '').trim().startsWith('|')) {
        const row = cell(lines[++i]); out.push(`<tr>${columns.map((_, j) => `<td>${format(row[j] || '')}</td>`).join('')}</tr>`);
      }
      out.push('</tbody></table></div>'); continue;
    }
    const item = /^(?:([-*])\s+|\d+[.)]\s+)(.+)$/.exec(line);
    if (item) {
      const type = item[1] ? 'ul' : 'ol';
      if (list !== type) { closeList(); out.push(`<${type}>`); list = type; }
      out.push(`<li>${format(item[2])}</li>`); continue;
    }
    closeList();
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) { const level = Math.min(4, Math.max(2, heading[1].length)); out.push(`<h${level}>${format(heading[2])}</h${level}>`); }
    else if (/^(---+|\*\*\*+)$/.test(line)) out.push('<hr>');
    else out.push(`<p>${format(line)}</p>`);
  }
  closeList(); return out.join('');
}

export function sourceGroups(sources, prefix, emptyText) {
  if (!sources.length) return `<li class="empty-state">${esc(emptyText)}</li>`;
  const groups = new Map();
  for (const s of sources) {
    const host = new URL(s.url).hostname.replace(/^www\./, '');
    if (!groups.has(host)) groups.set(host, []);
    groups.get(host).push(s);
  }
  return [...groups].map(([host, items], index) => `
    <li class="src-group${index === 0 ? ' open' : ''}">
      <button class="src-group-head" type="button" aria-expanded="${index === 0}">
        ${icon('chevron-right')}<span class="dom">${esc(host)}</span><span class="n">${items.length}</span>
      </button>
      <ul class="src-group-body">
        ${items.map((s) => `<li class="source-item" id="${esc(prefix)}-source-${s.id}" tabindex="-1"><a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a>${s.meta ? `<p class="source-detail">${esc(s.meta)}</p>` : ''}</li>`).join('')}
      </ul>
    </li>`).join('');
}

function focusSource(id) {
  const target = $(id); if (!target) return;
  for (let el = target.parentElement; el; el = el.parentElement) { if (el.tagName === 'DETAILS') el.open = true; }
  target.focus({ preventScroll: true });
  target.scrollIntoView({ block: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
  document.querySelectorAll('.source-active').forEach((el) => el.classList.remove('source-active'));
  target.classList.add('source-active');
}

export function initSources() {
  document.addEventListener('click', (event) => {
    const head = event.target.closest('.src-group-head');
    if (head) {
      const group = head.parentElement;
      group.classList.toggle('open');
      head.setAttribute('aria-expanded', String(group.classList.contains('open')));
      return;
    }
    const citation = event.target.closest('[data-source]');
    if (citation) focusSource(citation.dataset.source);
  });
}
