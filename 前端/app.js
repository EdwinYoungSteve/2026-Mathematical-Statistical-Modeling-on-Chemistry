/* MolSight 前端：零构建依赖，可直接放在静态服务器上运行。 */

const DEMO_FILE = '../数据演示/虚拟筛选结果.csv';
const DEMO_ROWS = [
  { Rank: 1, ChEMBL_ID: 'CHEMBL5172251', SMILES: 'CCc1cc(Nc2ncc(Cl)c(-c3cn(S(=O)(=O)CC)c4ccccc34)n2)c(OC)cc1N1CCC(N2CCN(C)CC2)CC1', MW: 652.26, AlogP: 5.48, Actual_pIC50: 7.46, Pred_pIC50: 7.8658, Actual_Active: 1, Pred_Active: 1, Prob_Active: .9894 },
  { Rank: 2, ChEMBL_ID: 'CHEMBL5395951', SMILES: 'CCS(=O)(=O)N1CCC(c2nc(-c3ccc(F)cc3)c(-c3ccnc(Nc4ccc(N5CCN(C)CC5)cc4)n3)s2)CC1', MW: 621.81, AlogP: 5.43, Actual_pIC50: 7.635, Pred_pIC50: 7.5081, Actual_Active: 1, Pred_Active: 1, Prob_Active: .9884 },
  { Rank: 3, ChEMBL_ID: 'CHEMBL5185922', SMILES: 'CCS(=O)(=O)n1cc(-c2nc(Nc3ccc(N4CCC(N5CCN(C)CC5)CC4)c(Cl)c3)ncc2Cl)c2ccccc21', MW: 628.63, AlogP: 5.56, Actual_pIC50: 8.075, Pred_pIC50: 7.8128, Actual_Active: 1, Pred_Active: 1, Prob_Active: .9867 },
  { Rank: 4, ChEMBL_ID: 'CHEMBL5864620', SMILES: 'CC(C)N(C)C/C=C/C(=O)N1CC[C@@H](n2cc(C(=O)Nc3nc4cc(Cl)ccc4o3)c3c(N)ncnc32)C1', MW: 537.02, AlogP: 3.73, Actual_pIC50: 7.69, Pred_pIC50: 7.5745, Actual_Active: 1, Pred_Active: 1, Prob_Active: .9824 },
  { Rank: 5, ChEMBL_ID: 'CHEMBL4744138', SMILES: 'C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C(C)=O)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C', MW: 527.63, AlogP: 4.63, Actual_pIC50: 8.245, Pred_pIC50: 7.9104, Actual_Active: 1, Pred_Active: 1, Prob_Active: .9822 },
  { Rank: 6, ChEMBL_ID: 'CHEMBL5565103', SMILES: 'COc1ccc(-c2c(NC(=O)/C=C/[C@H]3CCCN3C)ccc3ncnc(Nc4ccc(OCc5ccccn5)c(Cl)c4)c23)cc1OC', MW: 651.17, AlogP: 7.27, Actual_pIC50: 7.24, Pred_pIC50: 7.7980, Actual_Active: 1, Pred_Active: 1, Prob_Active: .9816 },
  { Rank: 7, ChEMBL_ID: 'CHEMBL5631065', SMILES: 'C=CC(=O)Nc1cc(Nc2nccc(-c3cnn4c(C)cccc34)n2)c(OC)cc1N(C)CCN(C)C', MW: 500.61, AlogP: 3.97, Actual_pIC50: 8.315, Pred_pIC50: 7.5894, Actual_Active: 1, Pred_Active: 1, Prob_Active: .9796 },
  { Rank: 8, ChEMBL_ID: 'CHEMBL4210035', SMILES: 'COc1ccc2[nH]c(C(=O)NCCCN3CCOCC3)cc2c1', MW: 326.35, AlogP: 1.12, Actual_pIC50: 6.24, Pred_pIC50: 6.8470, Actual_Active: 1, Pred_Active: 1, Prob_Active: .8162 },
  { Rank: 9, ChEMBL_ID: 'CHEMBL4208121', SMILES: 'CCOc1ccc(NC(=O)c2ccc(C)cc2)cc1', MW: 269.30, AlogP: 2.18, Actual_pIC50: 5.41, Pred_pIC50: 5.7114, Actual_Active: 0, Pred_Active: 0, Prob_Active: .2821 },
  { Rank: 10, ChEMBL_ID: 'CHEMBL3958751', SMILES: 'CC(C)Oc1ccc(C(=O)O)cc1', MW: 194.23, AlogP: 2.03, Actual_pIC50: 4.98, Pred_pIC50: 5.2265, Actual_Active: 0, Pred_Active: 0, Prob_Active: .1794 }
];

const state = {
  page: 'overview',
  rows: [...DEMO_ROWS],
  batchRows: [...DEMO_ROWS],
  threshold: .5,
  batchModel: 'Stacking',
  batchSort: 'probability',
  batchPage: 1,
  batchPageSize: 8,
  selectedRows: new Set(),
  history: [],
  lastResult: null,
  dataSource: 'demo',
  connectionMode: 'demo',
  lastSync: null
};

const JSME_SCRIPT_URL = 'https://cdn.jsdelivr.net/npm/jsme-editor@2024.4.29/jsme.nocache.js';
let moleculeEditor = null;
let jsmeLoadPromise = null;

const MODEL_METRICS = [
  { key: 'Stacking', name: 'Stacking v2', auc: .9336, r2: .6313, color: '#43d5bf' },
  { key: 'Voting', name: 'Voting v2', auc: .9334, r2: .6161, color: '#7db5ff' },
  { key: 'SVM', name: 'SVM (RBF)', auc: .9286, r2: .5882, color: '#6bd5db' },
  { key: 'KNN', name: 'KNN', auc: .9218, r2: .5634, color: '#d7b36a' },
  { key: 'RF', name: 'Random Forest', auc: .9202, r2: .6035, color: '#f7ad69' },
  { key: 'MLP', name: 'MLP 神经网络', auc: .9194, r2: .4589, color: '#849cf0' },
  { key: 'ET', name: 'Extra Trees', auc: .9103, r2: .6382, color: '#b38cff' },
  { key: 'HGBT', name: 'HistGradientBoosting', auc: .8747, r2: .5508, color: '#ef8f8f' }
];

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const number = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
const rowKey = (row) => `${row.ChEMBL_ID || ''}::${row.SMILES || ''}`;

function sourceLabel() {
  return ({ demo: '演示数据', upload: '导入结果', api: '真实模型', local: '本地估算' })[state.dataSource] || '当前数据';
}

function updateSystemStatus(mode = state.connectionMode, detail = '') {
  state.connectionMode = mode;
  const labels = {
    api: ['真实模型', detail || '模型 API 已连接'],
    local: ['本地估算', detail || 'API 不可用，使用演示算法'],
    demo: ['演示模式', detail || '尚未连接模型 API']
  };
  const [label, description] = labels[mode] || labels.demo;
  $('#systemMode').textContent = label;
  $('#systemModeDetail').textContent = description;
  $('#systemStatusDot').className = `status-dot ${mode}`;
}

function updateLastSync(date = new Date()) {
  state.lastSync = date;
  $('#lastSync').textContent = date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove('show'), 2800);
}

function setPage(page, updateLocation = true) {
  if (!['overview', 'single', 'batch', 'evaluation', 'reports'].includes(page)) page = 'overview';
  state.page = page;
  $$('.nav-item').forEach((button) => {
    const active = button.dataset.page === page;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  });
  $$('.page').forEach((element) => element.classList.toggle('active-page', element.id === `page-${page}`));
  const label = ({ overview: '项目总览', single: '单分子预测', batch: '批量虚拟筛选', evaluation: '模型评估', reports: '实验报告' })[page];
  $('#breadcrumbCurrent').textContent = label;
  renderPage(page);
  if (updateLocation && window.location.hash !== `#${page}`) window.history.replaceState(null, '', `#${page}`);
}

function renderPage(page) {
  if (page === 'overview') renderOverview();
  if (page === 'single') renderSingle();
  if (page === 'batch') renderBatch();
  if (page === 'evaluation') renderEvaluation();
  if (page === 'reports') renderReports();
}

function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i += 1; } else quoted = !quoted;
    } else if (char === ',' && !quoted) { row.push(field); field = ''; }
    else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      row.push(field); field = '';
      if (row.some((part) => part.trim() !== '')) rows.push(row);
      row = [];
    } else field += char;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  if (!rows.length) return [];
  const headers = rows.shift().map((header) => header.trim().replace(/^\ufeff/, ''));
  return rows.map((values, index) => {
    const result = {};
    headers.forEach((header, column) => { result[header] = (values[column] ?? '').trim(); });
    return normalizeRow(result, index + 1);
  }).filter((row) => row.SMILES || row.ChEMBL_ID);
}

function getField(row, names, fallback = '') {
  const key = Object.keys(row).find((candidate) => names.some((name) => candidate.toLowerCase() === name.toLowerCase()));
  return key ? row[key] : fallback;
}

function hasField(row, names) {
  return Object.keys(row).some((candidate) => names.some((name) => candidate.toLowerCase() === name.toLowerCase()) && String(row[candidate]).trim() !== '');
}

function toNumber(value, fallback = 0) { const parsed = Number.parseFloat(value); return Number.isFinite(parsed) ? parsed : fallback; }

function normalizeRow(raw, index = 1) {
  const hasPrediction = hasField(raw, ['Pred_pIC50', 'Pred_pChEMBL', 'prediction', 'predicted_pIC50']) && hasField(raw, ['Prob_Active', 'Pred_Prob_Active', 'probability']);
  const probabilityValue = toNumber(getField(raw, ['Prob_Active', 'Pred_Prob_Active', 'probability']), 0);
  const probability = clamp(probabilityValue > 1 && probabilityValue <= 100 ? probabilityValue / 100 : probabilityValue, 0, 1);
  return {
    Rank: toNumber(getField(raw, ['Rank', 'rank']), index),
    ChEMBL_ID: getField(raw, ['ChEMBL_ID', 'Molecule ChEMBL ID', 'id'], `Molecule-${index}`),
    SMILES: getField(raw, ['SMILES', 'Smiles', 'smiles']),
    MW: toNumber(getField(raw, ['MW', 'Molecular Weight', 'molecular_weight'])),
    AlogP: toNumber(getField(raw, ['AlogP', 'alogp'])),
    Actual_pIC50: toNumber(getField(raw, ['Actual_pIC50', 'Actual_pChEMBL', 'pIC50'])),
    Pred_pIC50: toNumber(getField(raw, ['Pred_pIC50', 'Pred_pChEMBL', 'prediction', 'predicted_pIC50'])),
    Actual_Active: toNumber(getField(raw, ['Actual_Active', 'Actual_Label']), 0),
    Pred_Active: hasField(raw, ['Pred_Active', 'Pred_Label']) ? toNumber(getField(raw, ['Pred_Active', 'Pred_Label']), 0) : (probability >= .5 ? 1 : 0),
    Prob_Active: probability,
    _hasPrediction: hasPrediction,
    _predictionMode: hasPrediction ? 'imported' : 'missing'
  };
}

function activeRows(rows = state.rows) { return rows.filter((row) => Number(row.Pred_Active) === 1 || Number(row.Prob_Active) >= .5); }

function renderOverview() {
  const rows = state.rows;
  const active = activeRows(rows);
  const ranked = [...rows].sort((a, b) => b.Prob_Active - a.Prob_Active);
  const top = [...rows].sort((a, b) => b.Pred_pIC50 - a.Pred_pIC50)[0] || DEMO_ROWS[0];
  $('#page-overview').innerHTML = `
    <div class="page-heading"><div><h1>项目总览</h1><p>用机器学习把分子数据变成可行动的候选物。查看当前筛选进度、模型表现和高潜力分子。</p></div><div class="heading-actions"><button class="button" id="overviewRefresh">↻ 刷新数据</button><button class="button primary" data-go="single"><span class="button-icon">＋</span> 新建预测</button></div></div>
    <div class="metrics-grid">
      <div class="metric-card"><span class="metric-label">候选分子</span><strong class="metric-value">${rows.length.toLocaleString()}</strong><span class="metric-foot"><span class="up">${sourceLabel()}</span> 当前数据集</span></div>
      <div class="metric-card"><span class="metric-label">预测活性</span><strong class="metric-value teal">${active.length.toLocaleString()}</strong><span class="metric-foot"><span class="up">${rows.length ? number(active.length / rows.length * 100, 1) : '0.0'}%</span> 通过阈值</span></div>
      <div class="metric-card"><span class="metric-label">最高 pIC50</span><strong class="metric-value">${number(top.Pred_pIC50, 2)}</strong><span class="metric-foot"><span class="up">${escapeHtml(top.ChEMBL_ID)}</span></span></div>
      <div class="metric-card"><span class="metric-label">最佳模型 AUC</span><strong class="metric-value teal">0.934</strong><span class="metric-foot"><span class="up">测试集</span> Stacking v2</span></div>
    </div>
    <div class="content-grid">
      <section class="panel"><div class="panel-header"><div><h2 class="panel-title">预测分布</h2><p class="panel-subtitle">pIC50 预测值与活性概率分布</p></div><span class="chip">${sourceLabel()}</span></div><div class="panel-body"><div class="chart-wrap"><canvas id="distributionChart"></canvas></div><div class="legend"><span>预测 pIC50</span><span>活性阈值 6.0</span></div></div></section>
      <section class="panel"><div class="panel-header"><div><h2 class="panel-title">高潜力分子</h2><p class="panel-subtitle">按预测活性概率排序</p></div><button class="panel-link" data-go="batch">查看全部 →</button></div><div class="panel-body"><div class="ranking-list">${ranked.slice(0, 5).map((row, index) => `<div class="ranking-row"><span class="ranking-rank">${String(index + 1).padStart(2, '0')}</span><span class="ranking-name" title="${escapeHtml(row.SMILES)}">${escapeHtml(row.ChEMBL_ID)}</span><span class="ranking-score">${number(row.Prob_Active * 100, 1)}%</span><div class="progress"><span style="width:${clamp(row.Prob_Active * 100, 2, 100)}%"></span></div></div>`).join('')}</div></div></section>
    </div>
    <section class="panel table-panel"><div class="panel-header"><div><h2 class="panel-title">最近筛选结果</h2><p class="panel-subtitle">${sourceLabel()} · 共 ${rows.length} 条</p></div><button class="panel-link" data-go="batch">打开批量筛选 →</button></div>${renderDataTable(ranked.slice(0, 7), false)}</section>`;
  $$('#page-overview [data-go]').forEach((button) => { button.onclick = () => setPage(button.dataset.go); });
  $('#overviewRefresh').onclick = () => { updateLastSync(); renderOverview(); drawOverviewChart(); showToast('已刷新当前项目数据'); };
  requestAnimationFrame(drawOverviewChart);
}

function renderDataTable(rows, withCheckbox = true) {
  if (!rows.length) return '<div class="empty-state">暂无符合条件的分子</div>';
  return `<div class="table-scroll"><table class="data-table"><thead><tr>${withCheckbox ? '<th><input id="selectAll" type="checkbox" aria-label="选择当前页" /></th>' : ''}<th>#</th><th>分子 / ChEMBL ID</th><th>预测 pIC50</th><th>活性概率</th><th>MW</th><th>AlogP</th><th>结论</th></tr></thead><tbody>${rows.map((row, index) => { const smiles = row.SMILES || ''; const key = rowKey(row); return `<tr>${withCheckbox ? `<td><input class="row-check" type="checkbox" data-key="${escapeHtml(key)}" ${state.selectedRows.has(key) ? 'checked' : ''} aria-label="选择 ${escapeHtml(row.ChEMBL_ID)}" /></td>` : ''}<td class="rank-cell">${String(row.Rank || index + 1).padStart(2, '0')}</td><td class="molecule-cell" title="${escapeHtml(smiles)}"><b>${escapeHtml(row.ChEMBL_ID)}</b><small style="display:block;color:#5f7b86;margin-top:4px;font-family:'Manrope',sans-serif;">${escapeHtml(smiles.slice(0, 42))}${smiles.length > 42 ? '…' : ''}</small></td><td class="${row.Pred_pIC50 >= 6 ? 'score-good' : 'score-warn'}">${number(row.Pred_pIC50, 3)}</td><td class="${row.Prob_Active >= .5 ? 'score-good' : 'score-warn'}">${number(row.Prob_Active * 100, 1)}%</td><td>${number(row.MW, 2)}</td><td>${number(row.AlogP, 2)}</td><td>${row.Pred_Active === 1 || row.Prob_Active >= .5 ? '<span class="active-pill">活性</span>' : '<span class="inactive-pill">低活性</span>'}</td></tr>`; }).join('')}</tbody></table></div>`;
}

function drawOverviewChart() {
  const canvas = $('#distributionChart'); if (!canvas) return;
  const rect = canvas.getBoundingClientRect(); const ratio = window.devicePixelRatio || 1;
  canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  const width = rect.width; const height = rect.height; const pad = { top: 14, right: 8, bottom: 27, left: 29 };
  ctx.clearRect(0, 0, width, height);
  const values = state.rows.slice(0, 45).map((row) => row.Pred_pIC50 || 0); const max = Math.max(10, ...values); const min = Math.min(4, ...values);
  const x = (index) => pad.left + index * ((width - pad.left - pad.right) / Math.max(values.length - 1, 1));
  const y = (value) => pad.top + (max - value) / (max - min) * (height - pad.top - pad.bottom);
  ctx.strokeStyle = 'rgba(146,181,190,.13)'; ctx.lineWidth = 1; ctx.font = '9px DM Mono, monospace'; ctx.fillStyle = '#607b85';
  [4, 6, 8, 10].forEach((tick) => { const py = y(tick); ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(width - pad.right, py); ctx.stroke(); ctx.fillText(tick.toFixed(0), 6, py + 3); });
  const thresholdY = y(6); ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(247,173,105,.72)'; ctx.beginPath(); ctx.moveTo(pad.left, thresholdY); ctx.lineTo(width - pad.right, thresholdY); ctx.stroke(); ctx.setLineDash([]);
  if (values.length > 1) { const gradient = ctx.createLinearGradient(0, pad.top, 0, height); gradient.addColorStop(0, 'rgba(67,213,191,.35)'); gradient.addColorStop(1, 'rgba(67,213,191,0)'); ctx.beginPath(); values.forEach((value, index) => index ? ctx.lineTo(x(index), y(value)) : ctx.moveTo(x(index), y(value))); ctx.lineTo(x(values.length - 1), height - pad.bottom); ctx.lineTo(x(0), height - pad.bottom); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill(); ctx.beginPath(); values.forEach((value, index) => index ? ctx.lineTo(x(index), y(value)) : ctx.moveTo(x(index), y(value))); ctx.strokeStyle = '#43d5bf'; ctx.lineWidth = 2; ctx.stroke(); values.forEach((value, index) => { ctx.beginPath(); ctx.arc(x(index), y(value), 2.3, 0, Math.PI * 2); ctx.fillStyle = value >= 6 ? '#43d5bf' : '#f7ad69'; ctx.fill(); }); }
  ctx.fillStyle = '#607b85'; ctx.fillText('候选分子排序', width - 82, height - 7);
}

function renderSingle() {
  moleculeEditor = null;
  const result = state.lastResult;
  $('#page-single').innerHTML = `
    <div class="page-heading"><div><h1>单分子预测</h1><p>输入标准 SMILES，或在结构画板中直接绘制分子，再快速估计活性与关键属性。</p></div><div class="heading-actions"><button class="button" id="useDemoMolecule">使用示例分子</button><button class="button primary" id="openSketcher"><span class="button-icon">⌬</span> 绘制分子</button></div></div>
    <section class="panel sketch-panel" id="sketchPanel" hidden><div class="sketch-header"><div><h2 class="panel-title">分子结构画板</h2><p class="panel-subtitle">绘制完成后可生成 SMILES 并回填到预测表单</p></div><button class="icon-button sketch-close" id="closeSketcher" type="button" aria-label="关闭结构画板">×</button></div><div class="sketch-canvas" id="jsmeContainer"><div class="sketch-placeholder"><b>准备加载结构编辑器</b><span>首次打开需要联网加载 JSME 组件</span></div></div><div class="sketch-footer"><span class="sketch-status" id="sketchStatus">支持原子、键、环、立体化学和电荷编辑</span><div class="sketch-actions"><button class="button" id="loadSmilesToSketch" type="button">从输入框载入</button><button class="button" id="clearSketch" type="button">清空画板</button><button class="button primary" id="applySketchSmiles" type="button">应用为 SMILES</button></div></div></section>
    <div class="single-grid"><section class="panel form-panel"><form id="singleForm"><label class="form-label" for="smilesInput">SMILES 分子式</label><textarea class="textarea" id="smilesInput" required placeholder="例如：COc1ccc2[nH]c(C(=O)NCCCN3CCOCC3)cc2c1">${escapeHtml(result?.smiles || '')}</textarea><p class="input-hint">可直接输入 SMILES，也可以点击“绘制分子”从结构画板生成。连接后端 API 后将调用训练好的模型。</p><div class="form-row"><div><label class="form-label" for="modelSelect">预测模型</label><select class="select" id="modelSelect"><option value="Stacking">Stacking v2（推荐）</option><option value="Voting">Voting v2</option><option value="RF">Random Forest</option><option value="HGBT">HistGradientBoosting</option></select></div><div><label class="form-label" for="thresholdInput">活性阈值</label><input class="text-input" id="thresholdInput" type="number" min="0" max="12" step="0.1" value="6.0" /></div></div><div class="form-actions"><button type="reset" class="button" id="clearSingle">清空</button><button type="submit" class="button primary" id="predictButton"><span class="button-icon">✦</span> 开始预测</button></div></form></section>
    <section class="panel result-panel">${result ? renderResult(result) : `<div class="empty-state" style="padding:72px 15px;"><div style="font-size:28px;color:#28645f;margin-bottom:12px;">✦</div><b style="display:block;color:#aac6c7;font-size:13px;font-weight:500;">等待输入分子</b><span style="display:block;margin-top:7px;color:#66818b;font-size:10px;">预测结果会显示在这里</span></div>`}</section></div>
    <section class="panel history-panel"><div class="panel-header"><div><h2 class="panel-title">最近预测</h2><p class="panel-subtitle">本次会话中的单分子记录</p></div><span class="chip">${state.history.length} 条记录</span></div><div class="panel-body"><div class="history-list">${state.history.length ? state.history.slice(0, 5).map((item) => `<div class="history-item"><span class="history-dot"></span><div class="smiles" title="${escapeHtml(item.smiles)}">${escapeHtml(item.smiles)}<small>${escapeHtml(item.model)} · ${item.time}</small></div><span class="history-score">${number(item.prediction, 3)}</span></div>`).join('') : '<div class="empty-state" style="padding:20px;">完成第一次预测后，记录会出现在这里</div>'}</div></div></section>`;
  $('#singleForm').onsubmit = handleSingleSubmit;
  $('#clearSingle').onclick = (event) => { event.preventDefault(); state.lastResult = null; renderSingle(); };
  $('#useDemoMolecule').onclick = () => { $('#smilesInput').value = DEMO_ROWS[0].SMILES; showToast('已填入示例分子'); };
  $('#openSketcher').onclick = openMoleculeSketcher;
  $('#closeSketcher').onclick = closeMoleculeSketcher;
  $('#loadSmilesToSketch').onclick = loadSmilesIntoSketcher;
  $('#clearSketch').onclick = clearMoleculeSketcher;
  $('#applySketchSmiles').onclick = applySketcherSmiles;
}

function setSketcherControlsDisabled(disabled) {
  ['#loadSmilesToSketch', '#clearSketch', '#applySketchSmiles'].forEach((selector) => { const button = $(selector); if (button) button.disabled = disabled; });
}

function ensureJsmeLoaded() {
  if (window.JSApplet?.JSME) return Promise.resolve();
  if (jsmeLoadPromise) return jsmeLoadPromise;
  jsmeLoadPromise = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('结构画板加载超时')), 20000);
    window.jsmeOnLoad = () => { clearTimeout(timeout); resolve(); };
    const script = document.createElement('script');
    script.src = JSME_SCRIPT_URL;
    script.async = true;
    script.dataset.moleculeEditor = 'jsme';
    script.onerror = () => { clearTimeout(timeout); reject(new Error('无法下载结构画板组件')); };
    document.head.appendChild(script);
  }).catch((error) => { jsmeLoadPromise = null; throw error; });
  return jsmeLoadPromise;
}

async function openMoleculeSketcher() {
  const panel = $('#sketchPanel');
  panel.hidden = false;
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  if (moleculeEditor) { requestAnimationFrame(resizeMoleculeSketcher); return; }
  const status = $('#sketchStatus');
  const container = $('#jsmeContainer');
  status.textContent = '正在加载结构编辑器…';
  container.innerHTML = '<div class="sketch-placeholder loading"><b>正在加载 JSME</b><span>请稍候，首次加载可能需要几秒</span></div>';
  setSketcherControlsDisabled(true);
  try {
    await ensureJsmeLoaded();
    if (!document.body.contains(container) || panel.hidden) return;
    container.innerHTML = '';
    const editorWidth = `${Math.max(320, Math.floor(container.clientWidth))}px`;
    const editorHeight = window.innerWidth <= 520 ? '340px' : '420px';
    moleculeEditor = new window.JSApplet.JSME('jsmeContainer', editorWidth, editorHeight, { options: 'query,hydrogens', guicolor: '#dce8ea' });
    moleculeEditor.setCallBack('AfterStructureModified', (event) => {
      const smiles = event.src.smiles();
      status.textContent = smiles ? `当前 SMILES：${smiles.slice(0, 88)}${smiles.length > 88 ? '…' : ''}` : '画板为空';
    });
    setSketcherControlsDisabled(false);
    status.textContent = '画板已就绪，可绘制或从输入框载入结构';
    const currentSmiles = $('#smilesInput').value.trim();
    if (currentSmiles) loadSmilesIntoSketcher();
  } catch (error) {
    container.innerHTML = `<div class="sketch-placeholder error"><b>结构画板加载失败</b><span>${escapeHtml(error.message)}，仍可直接输入 SMILES</span><button class="button" id="retrySketcher" type="button">重新加载</button></div>`;
    status.textContent = '未能连接画板资源';
    $('#retrySketcher').onclick = openMoleculeSketcher;
  }
}

function closeMoleculeSketcher() {
  $('#sketchPanel').hidden = true;
  $('#openSketcher').focus();
}

function loadSmilesIntoSketcher() {
  if (!moleculeEditor) { showToast('结构画板仍在加载'); return; }
  const smiles = $('#smilesInput').value.trim();
  if (!smiles) { showToast('输入框中没有可载入的 SMILES'); return; }
  try { moleculeEditor.readGenericMolecularInput(smiles); $('#sketchStatus').textContent = '已从输入框载入结构'; }
  catch (_) { showToast('该 SMILES 无法载入画板，请检查格式'); }
}

function clearMoleculeSketcher() {
  if (!moleculeEditor) return;
  if (typeof moleculeEditor.reset === 'function') moleculeEditor.reset(); else moleculeEditor.readGenericMolecularInput('');
  $('#sketchStatus').textContent = '画板已清空';
}

function applySketcherSmiles() {
  if (!moleculeEditor) { showToast('结构画板仍在加载'); return; }
  const smiles = moleculeEditor.smiles().trim();
  if (!smiles) { showToast('请先在画板中绘制分子'); return; }
  $('#smilesInput').value = smiles;
  closeMoleculeSketcher();
  showToast('已将绘制结构转换为 SMILES');
}

function resizeMoleculeSketcher() {
  const panel = $('#sketchPanel');
  const container = $('#jsmeContainer');
  if (!moleculeEditor || !panel || panel.hidden || !container) return;
  const width = `${Math.max(300, Math.floor(container.clientWidth))}px`;
  const height = window.innerWidth <= 520 ? '340px' : '420px';
  moleculeEditor.setSize(width, height);
}

function renderResult(result) {
  const probability = clamp(result.probability, 0, 1); const isActive = result.prediction >= result.threshold;
  return `<div class="result-head"><div class="result-molecule">${escapeHtml(result.smiles.slice(0, 39))}${result.smiles.length > 39 ? '…' : ''}<small>${escapeHtml(result.model)} · ${result.mode === 'api' ? '真实模型' : '演示估算'}</small></div><span class="confidence-badge">置信度 ${number(result.confidence * 100, 0)}%</span></div><div class="score-display"><div><span class="metric-label">预测 pIC50</span><strong class="big-score">${number(result.prediction, 3)}<span class="score-unit">pIC50</span></strong><span class="metric-foot">阈值 ${number(result.threshold, 1)} · ${isActive ? '<span class="up">预测活性</span>' : '<span class="neutral">低活性</span>'}</span></div><div><div class="prob-ring" style="background:conic-gradient(var(--teal) 0 ${probability * 100}%, #22414a ${probability * 100}% 100%);"><span>${number(probability * 100, 1)}%</span></div><span class="prob-label">活性概率</span></div></div><div class="result-details"><div class="detail-box"><span>估算分子量</span><b>${number(result.mw, 1)} Da</b></div><div class="detail-box"><span>芳香原子</span><b>${result.aromatic}</b></div><div class="detail-box"><span>杂原子</span><b>${result.hetero}</b></div><div class="detail-box"><span>环数量</span><b>${result.rings}</b></div><div class="detail-box"><span>LogP 估计</span><b>${number(result.logp, 2)}</b></div><div class="detail-box"><span>不确定性</span><b>±${number(result.uncertainty, 2)}</b></div></div><div class="interpretation"><b>结果解读：</b>${isActive ? '该分子预测达到活性阈值，可优先进入后续实验验证。' : '该分子预测低于活性阈值，建议结合结构相似性和实验成本综合判断。'} 结果仅供研究筛选参考。</div>`;
}

async function handleSingleSubmit(event) {
  event.preventDefault(); const button = $('#predictButton'); const smiles = $('#smilesInput').value.trim(); const threshold = toNumber($('#thresholdInput').value, 6); const model = $('#modelSelect').value;
  if (!smiles) { showToast('请先输入 SMILES 分子式'); return; }
  button.disabled = true; button.innerHTML = '计算中…';
  try {
    const result = await predict(smiles, model, threshold); state.lastResult = result;
    state.history.unshift({ smiles, model: result.model, prediction: result.prediction, time: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) });
    renderSingle(); showToast(result.mode === 'api' ? '真实模型预测完成' : '演示预测完成（当前未连接 API）');
  } catch (error) { showToast(error.message || '预测失败'); button.disabled = false; button.innerHTML = '<span class="button-icon">✦</span> 开始预测'; }
}

async function predict(smiles, model, threshold) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ smiles, model, threshold }), signal: controller.signal });
    if (response.ok) {
      const payload = await response.json();
      updateSystemStatus('api');
      updateLastSync();
      return normalizeApiResult(payload, smiles, model, threshold);
    }
  } catch (_) { /* 静态演示模式下 API 不存在，继续使用本地估算 */ }
  finally { clearTimeout(timer); }
  updateSystemStatus('local');
  return localPredict(smiles, model, threshold);
}

function normalizeApiResult(payload, smiles, model, threshold) {
  const fallback = localPredict(smiles, model, threshold);
  const prediction = payload.prediction ?? payload.pred_pIC50 ?? payload.pred_pChEMBL;
  const probability = payload.probability ?? payload.prob_active ?? payload.Prob_Active;
  return { ...fallback, ...payload, smiles, model: payload.model || model, prediction: prediction == null ? fallback.prediction : toNumber(prediction, fallback.prediction), probability: probability == null ? fallback.probability : clamp(toNumber(probability, fallback.probability), 0, 1), threshold, mode: 'api' };
}

function hashString(value) { let hash = 2166136261; for (let i = 0; i < value.length; i += 1) { hash ^= value.charCodeAt(i); hash = Math.imul(hash, 16777619); } return (hash >>> 0) / 4294967295; }

function localPredict(smiles, model, threshold) {
  if (smiles.length < 3 || !/[A-Za-z]/.test(smiles)) throw new Error('SMILES 格式看起来不完整，请检查后重试');
  const atoms = smiles.match(/Br|Cl|[A-Z][a-z]?/g) || []; const weights = { C: 12.011, N: 14.007, O: 15.999, S: 32.06, P: 30.974, F: 18.998, Cl: 35.45, Br: 79.904, I: 126.9, H: 1.008 };
  const mw = atoms.reduce((sum, atom) => sum + (weights[atom] || 12), 0) + (smiles.match(/[Hh]/g) || []).length;
  const aromatic = (smiles.match(/[cnops]/g) || []).length; const hetero = atoms.filter((atom) => !['C', 'H'].includes(atom)).length; const rings = (smiles.match(/[0-9]/g) || []).length / 2; const halogens = atoms.filter((atom) => ['F', 'Cl', 'Br', 'I'].includes(atom)).length;
  const logp = clamp(0.54 + atoms.filter((atom) => atom === 'C').length * .075 + aromatic * .035 - hetero * .15 + halogens * .23, -1, 9);
  const noise = (hashString(smiles) - .5) * .5; const adjustments = { Stacking: .18, Voting: .12, RF: .03, HGBT: -.04 }; const prediction = clamp(5.3 + (mw - 250) * .0042 + aromatic * .045 + rings * .12 + hetero * .055 - halogens * .09 + logp * .12 + noise + (adjustments[model] || 0), 3.5, 10.5);
  const probability = 1 / (1 + Math.exp(-(prediction - threshold) * 1.45)); const confidence = clamp(.72 + Math.min(.2, Math.abs(prediction - threshold) * .035), .7, .94);
  return { smiles, model, prediction, probability, threshold, confidence, mw, aromatic, hetero, rings, logp, uncertainty: .18 + Math.abs(prediction - threshold) * .06, mode: 'local' };
}

function renderBatch() {
  const allRows = state.batchRows;
  const sorted = [...allRows].sort((a, b) => state.batchSort === 'pic50' ? b.Pred_pIC50 - a.Pred_pIC50 : b.Prob_Active - a.Prob_Active);
  const filtered = sorted.filter((row) => row.Prob_Active >= state.threshold);
  const selected = filtered.filter((row) => state.selectedRows.has(rowKey(row)));
  const pages = Math.max(1, Math.ceil(filtered.length / state.batchPageSize));
  state.batchPage = clamp(state.batchPage, 1, pages);
  const visible = filtered.slice((state.batchPage - 1) * state.batchPageSize, state.batchPage * state.batchPageSize);
  const fileLabel = state.dataSource === 'demo' ? '虚拟筛选结果.csv（演示）' : `${sourceLabel()} · ${allRows.length} 条`;
  $('#page-batch').innerHTML = `<div class="page-heading"><div><h1>批量虚拟筛选</h1><p>导入候选分子列表；若 CSV 没有预测列，系统会优先调用批量 API，并在不可用时明确回退到本地估算。</p></div><div class="heading-actions"><button class="button" id="downloadTemplate">↓ 下载模板</button><button class="button primary" id="exportResults">${selected.length ? `导出已选 ${selected.length} 条` : `导出结果 ${filtered.length} 条`}</button></div></div><div class="batch-hero"><section class="panel dropzone" id="dropzone"><h3>导入候选分子</h3><p>支持包含 SMILES、ChEMBL_ID 的 CSV；已有预测列将直接读取，缺失预测列会自动计算。</p><label class="dropzone-label" for="csvInput">＋ 选择 CSV 文件</label><input id="csvInput" type="file" accept=".csv,text/csv" /><span id="fileName" class="file-name">当前：${fileLabel}</span></section><section class="panel batch-options"><label class="form-label" for="batchModelSelect">批量预测模型</label><select class="select" id="batchModelSelect"><option value="Stacking" ${state.batchModel === 'Stacking' ? 'selected' : ''}>Stacking v2（推荐）</option><option value="Voting" ${state.batchModel === 'Voting' ? 'selected' : ''}>Voting v2</option><option value="RF" ${state.batchModel === 'RF' ? 'selected' : ''}>Random Forest</option><option value="SVM" ${state.batchModel === 'SVM' ? 'selected' : ''}>SVM (RBF)</option></select><div class="range-line"><span>活性概率不低于</span><b id="thresholdValue">${number(state.threshold * 100, 0)}%</b></div><input class="range" id="probabilityRange" type="range" min="0" max="1" step=".05" value="${state.threshold}" /><div class="range-line"><span>命中分子</span><b>${filtered.length} / ${allRows.length}</b></div><div class="progress" style="margin-top:10px;"><span style="width:${allRows.length ? filtered.length / allRows.length * 100 : 0}%"></span></div></section></div><section class="panel table-panel"><div class="batch-summary"><div><strong>筛选结果</strong><span>${selected.length ? `已选择 ${selected.length} 条 · ` : ''}${state.batchSort === 'pic50' ? '按 pIC50' : '按活性概率'}降序</span></div><div class="table-actions"><button class="button ${state.batchSort === 'pic50' ? 'active-sort' : ''}" id="sortByPic50">按 pIC50</button><button class="button ${state.batchSort === 'probability' ? 'active-sort' : ''}" id="sortByProb">按概率</button></div></div>${renderDataTable(visible, true)}<div class="pagination">${Array.from({ length: pages }, (_, index) => `<button class="page-btn ${index + 1 === state.batchPage ? 'active' : ''}" data-batch-page="${index + 1}">${index + 1}</button>`).join('')}</div></section>`;
  $('#probabilityRange').oninput = (event) => { state.threshold = Number(event.target.value); state.batchPage = 1; renderBatch(); };
  $('#batchModelSelect').onchange = (event) => { state.batchModel = event.target.value; showToast(`批量模型已切换为 ${event.target.options[event.target.selectedIndex].text}`); };
  $('#csvInput').onchange = (event) => { if (event.target.files[0]) loadCsvFile(event.target.files[0]); };
  const dropzone = $('#dropzone'); ['dragenter', 'dragover'].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add('dragging'); })); ['dragleave', 'drop'].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove('dragging'); })); dropzone.addEventListener('drop', (event) => { if (event.dataTransfer.files[0]) loadCsvFile(event.dataTransfer.files[0]); });
  $$('.page-btn').forEach((button) => { button.onclick = () => { state.batchPage = Number(button.dataset.batchPage); renderBatch(); }; });
  $('#sortByPic50').onclick = () => { state.batchSort = 'pic50'; state.batchPage = 1; renderBatch(); };
  $('#sortByProb').onclick = () => { state.batchSort = 'probability'; state.batchPage = 1; renderBatch(); };
  $('#exportResults').onclick = () => exportRows(selected.length ? selected : filtered, selected.length ? 'molsight_selected_results.csv' : 'molsight_screening_results.csv');
  $('#downloadTemplate').onclick = exportTemplate;
  $$('.row-check').forEach((checkbox) => { checkbox.onchange = () => { if (checkbox.checked) state.selectedRows.add(checkbox.dataset.key); else state.selectedRows.delete(checkbox.dataset.key); renderBatch(); }; });
  const selectAll = $('#selectAll');
  if (selectAll) {
    selectAll.checked = visible.length > 0 && visible.every((row) => state.selectedRows.has(rowKey(row)));
    selectAll.onchange = () => { visible.forEach((row) => { const key = rowKey(row); if (selectAll.checked) state.selectedRows.add(key); else state.selectedRows.delete(key); }); renderBatch(); };
  }
}

async function predictBatchRows(rows, model) {
  const missing = rows.filter((row) => !row._hasPrediction && row.SMILES);
  if (!missing.length) return { rows, mode: 'upload' };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch('/api/predict/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ molecules: missing.map((row) => ({ id: row.ChEMBL_ID, smiles: row.SMILES })), model, threshold: 6 }), signal: controller.signal });
    if (response.ok) {
      const payload = await response.json();
      const results = Array.isArray(payload) ? payload : payload.results;
      if (Array.isArray(results) && results.length === missing.length) {
        let resultIndex = 0;
        const predicted = rows.map((row) => row._hasPrediction || !row.SMILES ? row : mergePrediction(row, normalizeApiResult(results[resultIndex++], row.SMILES, model, 6)));
        updateSystemStatus('api', '批量模型 API 已连接');
        updateLastSync();
        return { rows: predicted, mode: 'api' };
      }
    }
  } catch (_) { /* 回退到本地演示估算 */ }
  finally { clearTimeout(timer); }
  updateSystemStatus('local', '批量 API 不可用，使用演示算法');
  return { rows: rows.map((row) => row._hasPrediction || !row.SMILES ? row : mergeLocalPrediction(row, model)), mode: 'local' };
}

function mergePrediction(row, result) {
  return { ...row, MW: row.MW || result.mw, AlogP: row.AlogP || result.logp, Pred_pIC50: result.prediction, Prob_Active: result.probability, Pred_Active: result.prediction >= 6 ? 1 : 0, _hasPrediction: true, _predictionMode: result.mode };
}

function mergeLocalPrediction(row, model) {
  try { return mergePrediction(row, localPredict(row.SMILES, model, 6)); }
  catch (_) { return { ...row, _hasPrediction: false, _predictionMode: 'invalid' }; }
}

function loadCsvFile(file) {
  const reader = new FileReader();
  reader.onload = async () => {
    const rows = parseCsv(reader.result);
    if (!rows.length) { showToast('未识别到有效的 CSV 数据'); return; }
    state.selectedRows.clear();
    state.batchPage = 1;
    state.dataSource = 'upload';
    state.batchRows = rows;
    state.rows = rows;
    renderBatch();
    const missing = rows.filter((row) => !row._hasPrediction && row.SMILES).length;
    if (!missing) { updateLastSync(); showToast(`已导入 ${rows.length} 条已有预测结果`); return; }
    showToast(`正在为 ${missing} 条分子计算预测…`);
    const result = await predictBatchRows(rows, state.batchModel);
    const completed = result.rows.filter((row, index) => !rows[index]._hasPrediction && row._hasPrediction).length;
    state.batchRows = result.rows;
    state.rows = [...result.rows];
    state.dataSource = result.mode;
    renderBatch();
    showToast(result.mode === 'api' ? `真实模型完成 ${completed} 条批量预测` : `本地演示算法完成 ${completed} / ${missing} 条预测`);
  };
  reader.onerror = () => showToast('文件读取失败，请重试');
  reader.readAsText(file, 'UTF-8');
}

function exportRows(rows, filename) {
  if (!rows.length) { showToast('没有可导出的结果'); return; }
  const headers = ['Rank', 'ChEMBL_ID', 'SMILES', 'MW', 'AlogP', 'Pred_pIC50', 'Prob_Active', 'Pred_Active']; const csv = [headers.join(','), ...rows.map((row, index) => [row.Rank || index + 1, row.ChEMBL_ID, `"${String(row.SMILES || '').replace(/"/g, '""')}"`, number(row.MW, 3), number(row.AlogP, 3), number(row.Pred_pIC50, 4), number(row.Prob_Active, 5), row.Pred_Active ?? (row.Prob_Active >= .5 ? 1 : 0)].join(','))].join('\n'); const blob = new Blob([`\ufeff${csv}`], { type: 'text/csv;charset=utf-8' }); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = filename; link.click(); URL.revokeObjectURL(url); showToast(`已导出 ${rows.length} 条记录`);
}

function exportTemplate() {
  const csv = '\ufeffChEMBL_ID,SMILES\nEXAMPLE_001,"CCOc1ccc2[nH]c(C(=O)NCC)cc2c1"';
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a'); link.href = url; link.download = 'molsight_template.csv'; link.click(); URL.revokeObjectURL(url);
  showToast('已下载批量预测模板');
}

function renderEvaluation() {
  $('#page-evaluation').innerHTML = `<div class="page-heading"><div><h1>模型评估</h1><p>对比优化后回归与分类任务表现，指标同步自环节六的最新优化报告。</p></div><div class="heading-actions"><button class="button" id="openReport">查看优化报告</button></div></div><div class="evaluation-grid"><section class="panel"><div class="panel-header"><div><h2 class="panel-title">分类模型对比 · ROC-AUC</h2><p class="panel-subtitle">优化后测试集表现，越高越好</p></div><span class="chip">测试集 n=183</span></div><div class="model-bars">${MODEL_METRICS.map((metric) => `<div class="model-bar"><label>${metric.name}</label><div class="bar-track"><span style="width:${metric.auc * 100}%;background:${metric.color}"></span></div><b>${metric.auc.toFixed(3)}</b></div>`).join('')}</div><div class="model-note"><b>推荐：</b>Stacking v2 的测试集 AUC 为 0.9336，综合表现最佳；Voting v2 达到 0.9334，可作为更轻量的批量方案。</div></section><section class="panel"><div class="panel-header"><div><h2 class="panel-title">回归任务指标</h2><p class="panel-subtitle">Stacking v2 · pIC50 数值预测</p></div></div><div class="metric-large"><div class="detail-box"><span>测试集 R²</span><b class="score-good">0.6313</b><small>Stacking v2</small></div><div class="detail-box"><span>RMSE</span><b>0.7201</b><small>测试集</small></div><div class="detail-box"><span>MAE</span><b>0.5186</b><small>测试集</small></div><div class="detail-box"><span>5-fold CV R²</span><b>0.6562</b><small>± 0.0527</small></div></div></section></div><div class="content-grid section-spaced"><section class="panel"><div class="panel-header"><div><h2 class="panel-title">模型选择建议</h2><p class="panel-subtitle">针对当前 EGFR 活性筛选任务</p></div></div><div class="panel-body"><div class="interpretation" style="margin-top:0;"><b>用于初筛：</b>优先使用 Stacking v2，适合需要较高召回率的候选物排序。<br/><br/><b>用于快速迭代：</b>Voting v2 推理链路更轻量，适合在化合物库扩展阶段批量运行。<br/><br/><b>注意：</b>指标来自当前项目测试集，正式实验前仍应使用独立外部验证集复核。</div></div></section><section class="panel"><div class="panel-header"><div><h2 class="panel-title">现有可视化</h2><p class="panel-subtitle">项目已生成的评估图</p></div></div><div class="panel-body evaluation-links"><a href="../环节六：模型优化/figures/O1_regression_before_after.png" target="_blank" rel="noopener" class="report-card compact-report"><span class="report-icon">◒</span><h3>回归优化前后</h3></a><a href="../环节六：模型优化/figures/O3_roc_all_optimized.png" target="_blank" rel="noopener" class="report-card compact-report"><span class="report-icon">⌁</span><h3>ROC 曲线</h3></a></div></section></div>`;
  $('#openReport').onclick = () => setPage('reports');
}

function renderReports() {
  $('#page-reports').innerHTML = `<div class="page-heading"><div><h1>实验报告</h1><p>快速打开项目各阶段生成的报告、图表与虚拟筛选结果。</p></div><div class="heading-actions"><button class="button primary" data-go="batch">进入筛选结果</button></div></div><div class="report-list"><article class="report-card"><span class="report-icon">◫</span><h3>模型优化报告</h3><p>超参数调优、特征筛选和集成模型结论。</p><a href="../环节六：模型优化/优化报告.txt" target="_blank" rel="noopener">打开报告 →</a></article><article class="report-card"><span class="report-icon">◌</span><h3>虚拟筛选结果</h3><p>按活性概率排序的候选分子清单。</p><a href="../数据演示/虚拟筛选结果.csv" target="_blank" rel="noopener">打开 CSV →</a></article><article class="report-card"><span class="report-icon">▤</span><h3>集成学习评估</h3><p>Stacking / Voting 与基模型性能对比。</p><a href="../环节五：集成学习/集成学习评估报告.txt" target="_blank" rel="noopener">打开报告 →</a></article><article class="report-card"><span class="report-icon">◒</span><h3>模型训练评估</h3><p>分类、回归指标及交叉验证结果。</p><a href="../环节四：模型训练/模型评估报告.txt" target="_blank" rel="noopener">打开报告 →</a></article><article class="report-card"><span class="report-icon">⌁</span><h3>筛选可视化</h3><p>瀑布图、富集曲线和预测误差分析。</p><a href="../数据演示/figures/D5_enrichment_curve.png" target="_blank" rel="noopener">查看图表 →</a></article><article class="report-card"><span class="report-icon">⌘</span><h3>数据清洗报告</h3><p>原始数据清洗、缺失值与异常值处理记录。</p><a href="../环节二：数据清洗与EDA/清洗报告.txt" target="_blank" rel="noopener">打开报告 →</a></article></div><section class="panel section-spaced"><div class="panel-header"><div><h2 class="panel-title">使用说明</h2><p class="panel-subtitle">前端运行与后端接入</p></div></div><div class="panel-body"><div class="interpretation" style="margin-top:0;"><b>直接运行：</b>使用任意静态服务器打开 frontend/index.html。直接双击也可使用内置演示和文件导入功能。<br/><br/><b>单分子 API：</b>POST /api/predict，接收 { smiles, model, threshold }。<br/><br/><b>批量 API：</b>POST /api/predict/batch，接收 { molecules: [{ id, smiles }], model, threshold }，返回 { results: [...] }。接口不可用时会明确标注并回退到本地估算。</div></div></section>`;
  $$('#page-reports [data-go]').forEach((button) => { button.onclick = () => setPage(button.dataset.go); });
}

async function loadDemoData() {
  try {
    const response = await fetch(DEMO_FILE); if (!response.ok) throw new Error('not found');
    const rows = parseCsv(await response.text()); if (rows.length && state.dataSource === 'demo') { state.rows = rows; state.batchRows = [...rows]; updateLastSync(); renderPage(state.page); }
  } catch (_) { /* 直接双击 HTML 时 fetch 会被浏览器阻止，保留内置演示数据 */ }
}

$$('.nav-item').forEach((button) => { button.onclick = () => setPage(button.dataset.page); });
$('#workspaceButton').onclick = () => showToast('当前仓库仅配置了 EGFR 活性项目');
$('#searchButton').onclick = () => { setPage('single'); requestAnimationFrame(() => $('#smilesInput')?.focus()); };
$('#notificationButton').onclick = () => { $('#notificationButton em')?.remove(); showToast('暂无新的模型任务通知'); };
updateSystemStatus();
updateLastSync();
const initialPage = window.location.hash.slice(1) || 'overview';
setPage(initialPage);
loadDemoData();
window.addEventListener('resize', () => { if (state.page === 'overview') drawOverviewChart(); resizeMoleculeSketcher(); });
window.addEventListener('hashchange', () => { const page = window.location.hash.slice(1); if (page && page !== state.page) setPage(page, false); });
window.addEventListener('keydown', (event) => { if (event.key === 'Escape' && $('#sketchPanel') && !$('#sketchPanel').hidden) closeMoleculeSketcher(); });
