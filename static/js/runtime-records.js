import { apiJson } from './api.js';

function number(value) {
  return Number.isFinite(Number(value)) ? Number(value) : 0;
}

export function normalizeRuntimeRecord(record) {
  return {
    ...record,
    elapsed: number(record.elapsed),
    sources_count: number(record.sources_count),
    llm_calls: number(record.llm_calls),
    tokens_in: number(record.tokens_in),
    tokens_out: number(record.tokens_out),
    search_calls: number(record.search_calls),
    fetch_calls: number(record.fetch_calls),
    search_failed: number(record.search_failed),
    fetch_failed: number(record.fetch_failed),
    query_summary: record.query_summary || '',
    report_title: record.report_title || '',
    model: record.model || '未知模型',
    error_detail: record.error_detail || '',
  };
}

export async function fetchRuntimeRecords() {
  const result = await apiJson('runtime-metrics');
  return Array.isArray(result.turns) ? result.turns.map(normalizeRuntimeRecord) : [];
}
