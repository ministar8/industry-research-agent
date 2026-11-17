export const connection = { connected: false, token: '', model: '' };

export async function refreshSession() {
  const response = await fetch('/auth/refresh', {
    method: 'POST', credentials: 'include', signal: AbortSignal.timeout(15000), redirect: 'error',
  });
  return response.ok;
}

export async function apiJson(path, body, retry = true) {
  const response = await fetch(`/${path}`, {
    method: body ? 'POST' : 'GET',
    headers: { ...(body ? { 'Content-Type': 'application/json' } : {}), ...(connection.token ? { Authorization: `Bearer ${connection.token}` } : {}) },
    ...(body ? { body: JSON.stringify(body) } : {}),
    credentials: 'include', signal: AbortSignal.timeout(15000), redirect: 'error',
  });
  if (response.status === 401 && retry && !path.startsWith('auth/')) {
    if (await refreshSession()) return apiJson(path, body, false);
  }
  if (!response.ok) throw new Error(errorMessage(response.status));
  if (response.status === 204) return null;
  return response.json();
}

function errorMessage(status) {
  if (status === 401 || status === 403) return '访问未授权，请检查服务口令。';
  if (status === 404 || status === 405) return '未找到调研接口。请启动 FastAPI，并从该服务的根地址打开工作台。';
  if (status === 422) return '服务未接受输入，请检查问题长度与服务配置。';
  if (status === 429) return '服务请求过于频繁，请稍后手动重试。';
  return `服务暂时不可用（HTTP ${status}），未自动重试。`;
}

export async function* streamResearch(body, signal) {
  let response = await fetch('/stream', {
    method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json', ...(connection.token ? { Authorization: `Bearer ${connection.token}` } : {}) },
    body: JSON.stringify({ ...body, stream_tokens: false }), signal, redirect: 'error',
  });
  if (response.status === 401 && await refreshSession()) {
    response = await fetch('/stream', {
      method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json', ...(connection.token ? { Authorization: `Bearer ${connection.token}` } : {}) },
      body: JSON.stringify({ ...body, stream_tokens: false }), signal, redirect: 'error',
    });
  }
  if (!response.ok) throw new Error(errorMessage(response.status));
  if (!response.headers.get('content-type')?.includes('text/event-stream') || !response.body) throw new Error('服务未返回有效的调研事件流。');
  const reader = response.body.getReader();
  const decoder = new TextDecoder(); let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, boundary.index); buffer = buffer.slice(boundary.index + boundary[0].length);
        const data = frame.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n');
        if (!data) continue;
        if (data.trim() === '[DONE]') return;
        let event;
        try { event = JSON.parse(data); } catch { throw new Error('收到无法解析的服务消息，请检查服务日志。'); }
        if (event.type === 'error') throw new Error('调研执行失败。服务可能已产生调用费用，请检查日志后再决定是否重试。');
        if (event.type === 'message' && event.content?.type === 'ai' && typeof event.content.content === 'string') yield event.content.content;
      }
      if (done) throw new Error('连接在完成标记前中断，结果可能不完整；未自动重试。');
    }
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}
