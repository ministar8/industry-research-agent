/* 服务连接：连接弹窗、连接 / 断开、侧栏连接状态 */
import { $, notify } from './ui.js';
import { connection, apiJson } from './api.js';

function renderStatus() {
  $('openConnection').classList.toggle('is-connected', connection.connected);
  $('connectionLabel').textContent = connection.connected ? '服务已连接' : '连接调研服务';
  $('disconnectService').hidden = !connection.connected;
}

export function openConnectionDialog() { $('connectionDialog').showModal(); }
export function lockConnect(value) { $('connectService').disabled = value; }

export function initConnection({ isBusy, onConnected, onDisconnected }) {
  $('serviceOrigin').value = location.origin;
  $('openConnection').addEventListener('click', openConnectionDialog);
  $('closeConnection').addEventListener('click', () => { $('connectionDialog').close(); $('openConnection').focus(); });
  $('connectionForm').addEventListener('submit', async (e) => {
    e.preventDefault(); if (isBusy()) return;
    const oldToken = connection.token; connection.token = $('serviceToken').value.trim(); lockConnect(true);
    $('connectionResult').className = 'connection-result'; $('connectionResult').textContent = '正在读取服务信息…';
    try {
      const info = await apiJson('info');
      if (!Array.isArray(info.models) || typeof info.default_model !== 'string') throw new Error('服务信息格式不正确。');
      connection.connected = true; connection.model = info.default_model;
      $('connectionResult').textContent = `已连接 · ${connection.model}`; $('serviceToken').value = '';
      renderStatus(); onConnected(); $('connectionDialog').close(); notify('服务已连接，尚未发起调研。'); $('question').focus();
    } catch (error) {
      connection.token = oldToken; $('connectionResult').classList.add('error');
      $('connectionResult').textContent = error.message === 'Failed to fetch' ? '无法连接服务。请从已启动的 FastAPI /ui/ 页面访问。' : error.message;
    } finally { lockConnect(false); }
  });
  $('disconnectService').addEventListener('click', () => {
    onDisconnected(); connection.connected = false; connection.token = ''; connection.model = '';
    $('serviceToken').value = ''; $('connectionResult').textContent = '已断开连接。'; renderStatus();
  });
}
