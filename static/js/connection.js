/* 服务连接：自动探活、连接弹窗、连接 / 断开、侧栏连接状态 */
import { $, notify } from './ui.js';
import { connection, apiJson } from './api.js';

function renderStatus() {
  $('openConnection').classList.toggle('is-connected', connection.connected);
  $('connectionLabel').textContent = connection.connected ? '服务已连接' : '连接调研服务';
  $('disconnectService').hidden = !connection.connected;
}

export function openConnectionDialog() { $('connectionDialog').showModal(); }
export function lockConnect(value) { $('connectService').disabled = value; }

async function connect(onConnected) {
  const info = await apiJson('info');
  if (!Array.isArray(info.models) || typeof info.default_model !== 'string') throw new Error('服务信息格式不正确。');
  connection.connected = true; connection.model = info.default_model;
  renderStatus(); onConnected();
}

export function initConnection({ isBusy, onConnected, onDisconnected }) {
  $('serviceOrigin').value = location.origin;
  $('openConnection').addEventListener('click', openConnectionDialog);
  $('closeConnection').addEventListener('click', () => { $('connectionDialog').close(); $('openConnection').focus(); });
  $('connectionForm').addEventListener('submit', async (e) => {
    e.preventDefault(); if (isBusy()) return;
    const oldToken = connection.token;
    connection.token = $('serviceToken').value.trim();
    lockConnect(true);
    $('connectionResult').className = 'connection-result'; $('connectionResult').textContent = '正在读取服务信息…';
    try {
      const username = $('authUsername').value.trim();
      const password = $('authPassword').value;
      if (username || password) {
        const path = $('authMode').value === 'register' ? 'auth/register' : 'auth/login';
        await apiJson(path, { username, password });
      }
      await connect(onConnected);
      $('connectionResult').textContent = `已连接 · ${connection.model}`;
      $('serviceToken').value = ''; $('authPassword').value = '';
      $('connectionDialog').close(); notify('服务已连接，尚未发起调研。'); $('question').focus();
    } catch (error) {
      connection.token = oldToken; $('connectionResult').classList.add('error');
      $('connectionResult').textContent = error.message === 'Failed to fetch' ? '无法连接服务。请从已启动的 FastAPI 根地址访问工作台。' : error.message;
    } finally { lockConnect(false); }
  });
  $('disconnectService').addEventListener('click', async () => {
    await apiJson('auth/logout', {}).catch(() => {});
    onDisconnected(); connection.connected = false; connection.token = ''; connection.model = '';
    $('serviceToken').value = ''; $('authPassword').value = '';
    $('connectionResult').textContent = '已断开连接。'; renderStatus();
  });
  // 同源部署免手动连接：页面加载后自动探活，成功即进入已连接状态；
  // 失败（服务未启动或需要口令）时保留「连接调研服务」手动入口。
  lockConnect(true);
  $('connectionLabel').textContent = '正在连接服务…';
  connect(onConnected).catch(() => {}).finally(() => { lockConnect(false); renderStatus(); });
}
