/* RinkSpace Solve 2. Native browser UI. All runtime data comes from /api. */
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const escapeHtml = value => String(value ?? '').replace(/[&<>\"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[character]));
const display = value => value === null || value === undefined || value === '' ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value);
const asArray = value => Array.isArray(value) ? value : [];
const numeric = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
const numberText = (value, digits = 0) => numeric(value) === null ? '—' : value.toLocaleString('zh-CN', {maximumFractionDigits:digits});
const pct = value => numeric(value) === null ? '—' : numberText(value, 1) + '%';
const noteText = value => Array.isArray(value) ? value.map(display).join('\n') : value === null || value === undefined ? '' : typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
const dateText = value => { if (!value) return '—'; const date = new Date(value); return Number.isNaN(date.getTime()) ? display(value) : date.toLocaleString('zh-CN', {hour12:false}); };
const shortTime = value => { if (!value) return '—'; const date = new Date(value); return Number.isNaN(date.getTime()) ? display(value) : date.toLocaleTimeString('zh-CN', {hour12:false}); };
const STATUS_LABELS = {
  statistics:'第二阶段：原预算内细化统计拟合', entity_review_required:'当前实体或模型冲突待纠正', numerical_repair_required:'数值停滞，完整状态已保留待纠错', interrupted_recoverable:'本次中断，完整状态可恢复', rebinding:'正在保存并对齐已修改的当前输入',
  source_template_and_object_contract_bound:'源模板与对象拼版契约已绑定', curve_template_bound_object_contract_pending:'曲线已用源模板；对象契约待贯通', component_geometry_available_mapping_pending:'已有图形素材；尚未绑定当前模型', HD_component_missing_or_in_progress:'高清组件缺失或待整理', statistical_level_set:'原统计几何暖启动；暂用保存预算上界', recoverable_failure:'完整状态已保留；正在纠错续算', current_input_update_required:'数值策略已修改；完整状态已保存，正在修复续算', preparing:'正在读取当前自足输入', initializing:'正在求解辅助初值；不计接受步', initialization:'辅助初值更新；物理方程保持不变', preflight_ready:'数值预检通过，尚无求解步', awaiting_geometry_review:'等待主代理核对真实冰线画面', warm_geometry:'S01冰线与围挡初值', warm_artwork:'固定相机细化广告初值', joint_feasibility:'S01全源联合可行性', joint_precision:'S01全源原预算细化', assembling_current_problem:'正在装配当前联合问题', awaiting_full_shot_validation:'待整镜头贴合验收', phase_I:'第一阶段：求全源可行预算', phase_II:'第二阶段：原预算内细化拟合', feasibility:'第一阶段：求全源可行预算', refinement:'第二阶段：原预算内细化拟合', process_exited:'数值进程已退出', stale_model:'前一模型运行；当前结构已更新', stale_inputs:'当前输入已修改，需重建关联', awaiting_visual_validation:'待整镜头真实审阅', preflight:'数值预检中', continue:'状态可续算', inner_incomplete:'内层尚未收敛，状态已保留',
  current_initialization_pose:'当前参数姿态光心', historical_demo_initial_reference_not_pivot:'历史初值光心；安装待估', missing_geometry:'当前包缺失几何', initial_geometry_only:'仅有初始几何', constraint_semantics_unassigned:'约束语义尚未分配', engineering_definition:'工程定义', user_working_choice:'用户工作选择', mathematical:'数学定义', user_tentative:'用户暂定', statistical_prior:'统计先验', unvalidated_initialization_reference:'未验证初值参照', historical_demo_optical_center_at_reference_pts:'历史demo参考PTS光心', physical_assumption:'物理假设', physical_hypothesis:'物理假说', user_confirmed:'用户已确认', user_instruction:'用户明确要求', user_requested:'用户明确要求', user_fact:'用户确认事实', user:'用户', system:'系统', implementation:'实现约定', algorithm:'算法假设', assumption:'假设', inferred:'推断', inference:'推断', unapproved:'未经批准', unconfirmed:'尚未确认', pending:'待确认', proposed:'待确认', rejected:'已否决', approved:'已批准', confirmed:'已确认', not_required:'无需批准', disabled:'已禁用', enabled:'已启用', active:'使用中', inactive:'未启用', design:'设计中', designed:'已设计', planned:'待实现', implemented:'已实现', unimplemented:'未实现', not_implemented:'未实现', pending_implementation:'待实现', pending_verification:'待验证', unverified:'未验证', verified:'已验证', validated:'已验证', ready:'已就绪', missing:'缺失', empty:'暂无数据', not_imported:'未导入', imported:'已导入', failed:'失败', error:'错误', running:'运行中', idle:'空闲', stopped:'已停止', completed:'已完成', paused:'已暂停', waiting:'等待中', blocked:'阻塞', unavailable:'不可用', incomplete:'未完成', not_started:'尚未开始', not_run:'尚未运行', not_evaluated:'尚未评估', no_run:'当前无运行', required:'必须满足', hard:'硬约束', soft:'软约束', fact:'事实', definition:'定义', policy:'项目规则', model:'模型约定', observation:'原片观测', boundary:'边界条件', parameter:'参数', dataset:'数据集'
};
const label = value => STATUS_LABELS[display(value)] || display(value);
const tone = value => {
  const word = String(value ?? '').toLowerCase();
  if (['user_confirmed','user_instruction','user_requested','user_fact','approved','confirmed','verified','validated','ready','implemented','imported','completed','enabled','active'].includes(word)) return 'good';
  if (['unapproved','unconfirmed','pending','proposed','assumption','inferred','inference','unverified','pending_verification','design','planned','pending_implementation','not_implemented','unimplemented','incomplete','not_evaluated','not_started','not_run'].includes(word)) return 'warn';
  if (['rejected','failed','error','blocked','missing'].includes(word)) return 'bad';
  if (['running','observation','fact','definition','policy','user'].includes(word)) return 'info';
  return '';
};
const badge = (value, forcedTone) => '<span class="badge ' + (forcedTone ?? tone(value)) + '">' + escapeHtml(label(value)) + '</span>';
const isApproved = constraint => ['user_confirmed','user_instruction','user_requested','user_fact','approved','confirmed','not_required','engineering_definition','user_working_choice','mathematical'].includes(String(constraint.approval ?? '').toLowerCase());
const effectiveEnabled = constraint => typeof constraint.effective_enabled === 'boolean' ? constraint.effective_enabled : !!constraint.enabled;
const isAssumption = constraint => ['statistical_prior','physical_assumption','physical_hypothesis','assumption','inferred','inference','algorithm'].includes(String(constraint.kind ?? '').toLowerCase()) || ['physical_assumption','physical_hypothesis','assumption','inferred','inference','algorithm'].includes(String(constraint.origin ?? '').toLowerCase());
const moduleName = id => asArray(resources.architecture.data?.modules).find(module => module.id === id)?.name || display(id);
const apiDefinitions = {
  architecture:{url:'/api/architecture', array:'modules'},
  constraints:{url:'/api/constraints', array:'constraints'},
  data:{url:'/api/data', array:'datasets'},
  state:{url:'/api/state'},
  review:{url:'/api/review'},
  model:{url:'/api/model',array:'paths'},
  contracts:{url:'/api/contracts'}
};
const resources = Object.fromEntries(Object.keys(apiDefinitions).map(key => [key, {data:null,error:null,loading:true,at:null}]));
const views = {
  architecture:{eyebrow:'ARCHITECTURE', title:'流程与边界', subtitle:'跟踪每个模块的输入、输出、单位与责任范围。'},
  constraints:{eyebrow:'CONSTRAINTS & ASSUMPTIONS', title:'约束与假设', subtitle:'核对约束的来源与批准状态，并明确控制它是否进入求解。'},
  data:{eyebrow:'DATA CONTRACTS', title:'数据入口', subtitle:'查看当前输入的来源、结构、规模与缺失项。'},
  monitor:{eyebrow:'RUNTIME & RESOURCES', title:'运行监控', subtitle:'仅展示当前 API 报告的运行状态与 CPU / GPU 实际资源读数。'},
  review:{eyebrow:'EVIDENCE REVIEW', title:'证据审阅', subtitle:'从完整镜头与独立证据检查同一个求解模型。'}
};
let currentView = 'architecture';
let constraintFilters = {query:'',approval:'all',origin:'all',enabled:'all'};
let dataQuery = '';
let dataMode = 'datasets';
let contractQuery = '';
let pollPaused = false;
let stateRequestPending = false;
let refreshPending = false;
let drawerState = null;
let drawerOpener = null;
let confirmTarget = null;
let mutationPending = false;
const initialPtsQuery=new URLSearchParams(location.search).get('native_pts');
let modelSelectedPts = initialPtsQuery&&/^[0-9]+$/.test(initialPtsQuery)?Number(initialPtsQuery):null;
let modelRequestSequence = 0;

async function requestJson(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(url, {...options, signal:controller.signal, headers:{'Accept':'application/json', ...(options.body ? {'Content-Type':'application/json'} : {}), ...options.headers}});
    let payload;
    try { payload = await response.json(); } catch { throw new Error('API 未返回可读取的 JSON（HTTP ' + response.status + '）'); }
    if (!response.ok) {
      const error = new Error(display(payload.message || payload.detail || payload.error || 'HTTP ' + response.status));
      error.status = response.status;
      throw error;
    }
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('API 返回结构不符合数据契约');
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('连接超时（12 秒）');
    if (error instanceof TypeError) throw new Error('无法连接本地后端，请确认服务已启动');
    throw error;
  } finally { clearTimeout(timer); }
}

async function loadResource(name, quiet = false) {
  const resource = resources[name];
  const requestSequence=name==='model'?++modelRequestSequence:null;
  if (name === 'state' && stateRequestPending) return;
  if (name === 'state') stateRequestPending = true;
  if (!quiet) { resource.loading = true; renderViewIfRelevant(name); }
  try {
    const requestUrl=apiDefinitions[name].url+(name==='model'&&modelSelectedPts!==null?'?native_pts='+modelSelectedPts:'');
    const payload = await requestJson(requestUrl);
    if(name==='model' && requestSequence!==modelRequestSequence)return;
    const arrayKey = apiDefinitions[name].array;
    if (arrayKey && !Array.isArray(payload[arrayKey])) throw new Error('API 缺少 ' + arrayKey + ' 数组');
    resource.data = payload;
    resource.error = null;
    resource.at = new Date();
  } catch (error) { resource.error = error.message; }
  finally {
    resource.loading = false;
    if (name === 'state') stateRequestPending = false;
    renderHeader();
    renderViewIfRelevant(name);
  }
}

async function refreshAll() {
  if (refreshPending) return;
  refreshPending = true;
  $('#refresh-all').disabled = true;
  await Promise.all(Object.keys(apiDefinitions).map(name => loadResource(name, !!resources[name].data)));
  refreshPending = false;
  $('#refresh-all').disabled = false;
}

function renderHeader() {
  const states = Object.values(resources);
  const failed = states.filter(resource => resource.error);
  const synced = states.filter(resource => resource.data);
  const loading = states.some(resource => resource.loading);
  let connectionText = loading && !synced.length ? '正在连接' : failed.length ? '部分接口未连接' : synced.length === states.length ? '本地后端已连接' : '等待后端连接';
  let connectionTone = failed.length ? (synced.length ? 'warn' : 'bad') : synced.length === states.length ? 'good' : 'neutral';
  $('#connection-label').innerHTML = '<span class="dot ' + connectionTone + '"></span>' + connectionText;
  $('#sidebar-dot').className = 'dot ' + connectionTone;
  $('#sidebar-state').textContent = failed.length ? failed.length + ' 个接口连接异常' : synced.length === states.length ? '本地后端已连接' : '等待后端连接';
  $('#nav-modules').textContent = resources.architecture.data ? asArray(resources.architecture.data.modules).length : '—';
  $('#nav-constraints').textContent = resources.constraints.data ? asArray(resources.constraints.data.constraints).length : '—';
  $('#nav-data').textContent = resources.data.data ? asArray(resources.data.data.datasets).length : '—';
  const relevantKey = currentView === 'monitor' ? 'state' : currentView === 'data' && dataMode === 'contracts' ? 'contracts' : currentView;
  const relevant = resources[relevantKey];
  const revision = relevant.data?.revision ?? (currentView === 'architecture' ? resources.architecture.data?.revision : null);
  $('#revision-label').textContent = '修订 ' + display(revision);
  $('#updated-label').textContent = relevant.at ? (relevant.error ? '上次成功 ' : '同步于 ') + shortTime(relevant.at) : '尚未同步';
  const banner = $('#connection-banner');
  if (failed.length && synced.length) {
    banner.innerHTML = '<div class="banner error"><div><strong>' + failed.length + ' 个接口暂时不可用</strong><p>已保留上次成功返回的数据；异常接口不作为当前运行依据。</p></div><button class="button button-secondary button-small" data-action="retry-all">重新连接</button></div>';
  } else { banner.innerHTML = ''; }
}

function renderViewIfRelevant(resourceName) {
  if (resourceName === 'model') { renderModelDock(); updateProjectionNotice(); return; }
  if (resourceName === 'contracts' && currentView === 'data') {renderData();return;}
  const relevant = currentView === 'monitor' ? 'state' : currentView;
  if (currentView === 'architecture' && resourceName === 'state' && $('.run-summary')) {
    if ($('#architecture-sequence')) $('#architecture-sequence').innerHTML = renderShotSequence(resources.state.data);
    $('.run-summary .summary-value').innerHTML = '<span style="font-size:18px;letter-spacing:0">' + summaryRun() + '</span>';
    $('.run-summary .summary-note').textContent = resources.state.data ? display(resources.state.data.message || label(resources.state.data.phase)) : '等待运行状态接口';
    return;
  }
  if (relevant === resourceName || currentView === 'architecture' && ['constraints','state','data'].includes(resourceName)) renderCurrentView();
}

function switchView(name) {
  if (name === 'data/contracts') {dataMode='contracts';name='data';}
  if (!views[name]) name = 'architecture';
  currentView = name;
  const meta = views[name];
  $('#page-eyebrow').textContent = meta.eyebrow;
  $('#page-title').textContent = meta.title;
  $('#page-subtitle').textContent = meta.subtitle;
  $$('.view').forEach(view => { view.hidden = view.id !== 'view-' + name; });
  $$('.nav-item').forEach(item => {
    const active = item.dataset.view === name;
    item.classList.toggle('active', active);
    if (active) item.setAttribute('aria-current', 'page'); else item.removeAttribute('aria-current');
  });
  const desiredHash=name==='data'&&dataMode==='contracts'?'#data/contracts':'#'+name;
  if (location.hash !== desiredHash) history.replaceState(null, '', desiredHash);
  renderHeader();
  renderCurrentView();
}

function renderCurrentView() {
  ({architecture:renderArchitecture,constraints:renderConstraints,data:renderData,monitor:renderMonitor,review:renderReview}[currentView])();
}

function loadingPanel() {
  return '<div class="panel skeleton-card" aria-busy="true"><p class="compact-note">正在读取当前 API…</p><div class="loading-line"></div><div class="loading-line"></div><div class="loading-line"></div></div>';
}
function emptyState(title, message, icon = '□') {
  return '<div class="empty-state"><div class="empty-symbol" aria-hidden="true">' + icon + '</div><h3>' + escapeHtml(title) + '</h3><p>' + escapeHtml(message) + '</p></div>';
}
function errorPanel(name) {
  const resource = resources[name];
  return '<div class="panel"><div class="error-box"><div class="empty-symbol" aria-hidden="true">⌁</div><h3>尚未连接' + (name === 'state' ? '运行状态' : '数据接口') + '</h3><p>' + escapeHtml(resource.error || '正在等待后端返回') + '。界面不会生成示例数据或模拟运行。</p><code>' + escapeHtml(apiDefinitions[name].url) + '</code><button class="button button-secondary" data-action="retry" data-resource="' + name + '">重新连接</button></div></div>';
}
function resourceNotice(name) {
  const resource = resources[name];
  return resource.error && resource.data ? '<div class="inline-note warn"><strong>当前连接异常，以下为上次成功数据。</strong> ' + escapeHtml(resource.error) + ' · ' + shortTime(resource.at) + '</div>' : '';
}
function readyMarkup(name) {
  const resource = resources[name];
  if (!resource.data) return resource.loading ? loadingPanel() : errorPanel(name);
  return null;
}
function summaryCard(title, value, note, className = '') {
  return '<div class="summary-card ' + className + '"><div class="summary-label">' + escapeHtml(title) + '</div><div class="summary-value">' + value + '</div><div class="summary-note">' + escapeHtml(note) + '</div></div>';
}
function sourcePath(path) {
  if (!path) return '<p class="compact-note">尚未提供路径。</p>';
  const value = typeof path === 'object' ? noteText(path) : String(path);
  return '<div class="path-with-action"><code class="path-box">' + escapeHtml(value) + '</code><button class="button button-secondary button-small" data-action="copy" data-value="' + escapeHtml(value) + '">复制路径</button></div>';
}
function rawDetails(value, title = '查看 API 原始字段') {
  return '<details><summary class="details-toggle">' + escapeHtml(title) + '</summary><pre class="json-block">' + escapeHtml(JSON.stringify(value, null, 2)) + '</pre></details>';
}
function summaryRun() {
  if (!resources.state.data) return '—';
  const run = resources.state.data.run;
  return run ? escapeHtml(label(run.status)) : '当前无运行';
}


function renderShotSequence(state) {
  const sequence = state?.sequence;
  if (!sequence) return '<div class="sequence-panel sequence-unknown"><div><p class="section-label">逐镜头推进</p><h3>等待后端报告当前镜头</h3><p>当前仅允许一个镜头进入求解；前端不提供绕过镜头锁定的选择。</p></div><span class="badge outline">序列尚未同步</span></div>';
  const shots = asArray(sequence.shots);
  return '<div class="sequence-panel"><div class="sequence-heading"><div><p class="section-label">逐镜头推进 · SINGLE ACTIVE SHOT</p><h3>当前镜头 <span class="mono">' + escapeHtml(String(sequence.active_shot || '未指定').toUpperCase()) + '</span></h3><p>当前镜头的整段冰线、围挡、广告与网格全部贴合，并通过同模型连续真实验证，才推进下一镜头。</p></div>' + (sequence.advance_allowed === false ? '<span class="badge warn">后续镜头未解锁</span>' : sequence.advance_allowed === true ? '<span class="badge good">后端允许推进</span>' : '<span class="badge outline">推进状态未提供</span>') + '</div><div class="sequence-shots">' + shots.map(shot => '<div class="shot-chip' + (String(shot.id).toLowerCase() === String(sequence.active_shot).toLowerCase() ? ' active' : '') + (shot.locked ? ' locked' : '') + '" title="' + escapeHtml(shot.reason || '') + '"><span class="shot-id">' + (shot.locked ? '<span aria-hidden="true">▣ </span>' : '') + escapeHtml(String(shot.id || '未指定').toUpperCase()) + '</span><span class="shot-status">' + escapeHtml(shot.locked ? '已锁定' : label(shot.status || '未声明状态')) + '</span>' + (shot.reason ? '<span class="shot-reason">' + escapeHtml(shot.reason) + '</span>' : '') + '</div>').join('') + '</div><p class="sequence-footnote">镜头状态与解锁条件由后端强制执行。历史九镜头 demo 作为只读基线保留。</p></div>';
}

function renderArchitecture() {
  const host = $('#view-architecture');
  const unavailable = readyMarkup('architecture');
  if (unavailable) { host.innerHTML = unavailable; return; }
  const data = resources.architecture.data;
  const modules = asArray(data.modules);
  const edges = asArray(data.edges);
  const constraints = resources.constraints.data ? asArray(resources.constraints.data.constraints) : null;
  const unapproved = constraints ? constraints.filter(item => !isApproved(item)).length : null;
  const enabled = constraints ? constraints.filter(effectiveEnabled).length : null;
  const summaries = '<div class="summary-row">' + summaryCard('独立模块', String(modules.length), edges.length + ' 条实际数据流') + summaryCard('待确认约束', numberText(unapproved), constraints ? '共 ' + constraints.length + ' 项约束' : '等待约束接口', unapproved ? 'warn' : '') + summaryCard('实际参与约束', numberText(enabled), '实际生效与配置开关分别记录') + summaryCard('当前运行', '<span style="font-size:18px;letter-spacing:0">' + summaryRun() + '</span>', resources.state.data ? display(resources.state.data.message || label(resources.state.data.phase)) : '等待运行状态接口', 'run-summary') + '</div>';
  let nodes = '';
  if (!modules.length) nodes = emptyState('尚未登记模块', '后端未提供架构模块，等待实际实现与数据契约。', '↗');
  modules.forEach((module, index) => {
    const outgoing = edges.filter(edge => edge.from === module.id && modules[index + 1]?.id === edge.to);
    nodes += '<button class="flow-node" data-action="module-detail" data-id="' + escapeHtml(module.id) + '" aria-label="查看 ' + escapeHtml(module.name) + ' 的输入输出"><span class="node-number">' + String(index + 1).padStart(2, '0') + '</span><span><span class="node-name">' + escapeHtml(module.name || module.id) + '</span><span class="node-purpose" style="display:block">' + escapeHtml(module.purpose || '尚未说明模块职责') + '</span><span class="node-ports"><span>输入 <strong>' + asArray(module.inputs).length + '</strong></span><span>输出 <strong>' + asArray(module.outputs).length + '</strong></span><span>' + escapeHtml(module.owner || '点击查看责任边界') + '</span></span></span><span class="node-tail">' + badge(module.status || '未声明状态') + '<span class="node-arrow" aria-hidden="true">↗</span></span></button>';
    if (index < modules.length - 1) nodes += outgoing.length ? '<div class="flow-connector" aria-label="数据流向下一个模块"><span>' + escapeHtml(outgoing.map(edge => edge.label).filter(Boolean).join(' / ')) + '</span></div>' : '<div style="height:12px" aria-hidden="true"></div>';
  });
  const nonAdjacentEdges = edges.filter(edge => {
    const index = modules.findIndex(module => module.id === edge.from);
    return index < 0 || modules[index + 1]?.id !== edge.to;
  });
  let crossEdges = nonAdjacentEdges.length ? '<div class="graph-edges"><p class="section-label">其他数据连接</p><div class="edge-list">' + nonAdjacentEdges.map(edge => '<div class="edge-line"><button class="link-button edge-endpoint" data-action="module-detail" data-id="' + escapeHtml(edge.from) + '">' + escapeHtml(moduleName(edge.from)) + '</button><span aria-hidden="true">→</span><button class="link-button edge-endpoint" data-action="module-detail" data-id="' + escapeHtml(edge.to) + '">' + escapeHtml(moduleName(edge.to)) + '</button><span class="edge-label">' + escapeHtml(edge.label || '') + '</span></div>').join('') + '</div></div>' : '';
  host.innerHTML = resourceNotice('architecture') + '<div id="architecture-sequence">' + renderShotSequence(resources.state.data) + '</div>' + summaries + '<div class="flow-layout"><div class="panel"><div class="panel-header"><div><h2>求解数据流</h2><p>点击模块，检查契约与代码位置</p></div><div class="legend"><span class="legend-item"><span class="legend-dot green"></span>实现 / 验证</span><span class="legend-item"><span class="legend-dot amber"></span>尚待完成</span></div></div><div class="flow-canvas">' + nodes + '</div>' + crossEdges + '<div class="panel-footer">箭头对应后端声明的数据连接；模块状态由架构接口提供。</div></div><div class="flow-notes"><div class="note-card"><h3>共享范围与机位事实</h3><p>共享参数、机位归属与逐镜头边界以当前模型和模块契约中的实际登记为准。常驻总览保留全部已导入实体与相机位置，未确认几何单独标注。</p><button class="subtle-link" data-action="navigate" data-view="constraints">检查约束台账 →</button></div><div class="note-card"><h3>来源与批准分开看</h3><p>用户事实、原片观测、实现约定与未经确认的假设各自保留来源。批准不自动意味着启用，未批准项应保持禁用。</p><button class="subtle-link" data-action="navigate" data-view="constraints">检查待确认项 →</button></div><div class="note-card"><h3>可审查的同一条链路</h3><p>当前输入进入独立数据契约，经唯一模型投影、残差装配与数值引擎产出完整状态。叠图与完整镜头验证使用同一个模型。</p><button class="subtle-link" data-action="navigate" data-view="review">查看独立验证 →</button></div></div></div>';
}

function portList(ports, emptyLabel) {
  if (!asArray(ports).length) return '<p class="compact-note">' + emptyLabel + '</p>';
  return '<div class="port-list">' + ports.map(port => {
    if (typeof port !== 'object' || port === null) return '<div class="port-card"><span class="port-name">' + escapeHtml(display(port)) + '</span></div>';
    return '<div class="port-card"><div class="port-top"><span class="port-name">' + escapeHtml(port.name || port.id || '未命名字段') + '</span><span class="port-unit">' + escapeHtml(port.unit || '单位未声明') + '</span></div><p class="port-description">' + escapeHtml(port.description || '尚未说明字段含义') + '</p><p class="port-source">来源 / 去向：' + escapeHtml(display(port.source || port.target)) + '</p></div>';
  }).join('') + '</div>';
}

function openModule(id) {
  const module = asArray(resources.architecture.data?.modules).find(item => item.id === id);
  if (!module) { toast('当前接口未提供这个模块的详情。', true); return; }
  const incoming = asArray(resources.architecture.data?.edges).filter(edge => edge.to === id);
  const outgoing = asArray(resources.architecture.data?.edges).filter(edge => edge.from === id);
  const content = '<div class="badge-row">' + badge(module.status || '未声明状态') + '<span class="badge outline mono">' + escapeHtml(module.id) + '</span></div><p class="drawer-intro">' + escapeHtml(module.purpose || '尚未说明模块职责。') + '</p><dl class="key-values"><dt>责任归属</dt><dd>' + escapeHtml(display(module.owner || module.responsibility)) + '</dd><dt>责任边界</dt><dd>' + escapeHtml(display(module.boundary || module.scope || module.purpose)) + '</dd></dl><section class="drawer-section"><h3>输入 · ' + asArray(module.inputs).length + '</h3>' + portList(module.inputs, '模块尚未声明输入契约。') + '</section><section class="drawer-section"><h3>输出 · ' + asArray(module.outputs).length + '</h3>' + portList(module.outputs, '模块尚未声明输出契约。') + '</section><section class="drawer-section"><h3>上游与下游</h3><dl class="key-values"><dt>数据来自</dt><dd>' + escapeHtml(incoming.map(edge => moduleName(edge.from) + (edge.label ? ' · ' + edge.label : '')).join('\n') || '未声明上游连接') + '</dd><dt>数据流向</dt><dd>' + escapeHtml(outgoing.map(edge => moduleName(edge.to) + (edge.label ? ' · ' + edge.label : '')).join('\n') || '未声明下游连接') + '</dd></dl></section><section class="drawer-section"><h3>代码位置</h3>' + sourcePath(module.source_file) + '</section><section class="drawer-section">' + rawDetails(module) + '</section>';
  const typedLinks=asArray(module.contract_types).length?'<section class="drawer-section"><h3>数据类型与字段定义</h3><div class="contract-type-directory">'+asArray(module.contract_types).map(type=>'<button class="contract-directory-item" data-action="module-type" data-type="'+escapeHtml(type)+'">'+escapeHtml(type)+'</button>').join('')+'</div></section>':'';
  openDrawer('MODULE CONTRACT', module.name || module.id, content+typedLinks, {type:'module',id});
}

function constraintSelectOptions(values, selected, allLabel) {
  return '<option value="all">' + allLabel + '</option>' + [...new Set(values.map(String))].sort().map(value => '<option value="' + escapeHtml(value) + '"' + (selected === value ? ' selected' : '') + '>' + escapeHtml(label(value)) + '</option>').join('');
}
function filteredConstraints() {
  const query = constraintFilters.query.toLocaleLowerCase().trim();
  return asArray(resources.constraints.data?.constraints).filter(constraint => {
    const matchesText = !query || [constraint.id,constraint.title,constraint.expression,constraint.scope,constraint.owner,noteText(constraint.notes),constraint.origin].map(display).join(' ').toLocaleLowerCase().includes(query);
    const matchesApproval = constraintFilters.approval === 'all' || (constraintFilters.approval === 'unapproved' ? !isApproved(constraint) : isApproved(constraint));
    const matchesOrigin = constraintFilters.origin === 'all' || String(constraint.origin) === constraintFilters.origin;
    const matchesEnabled = constraintFilters.enabled === 'all' || (constraintFilters.enabled === 'enabled' ? !!constraint.enabled : !constraint.enabled);
    return matchesText && matchesApproval && matchesOrigin && matchesEnabled;
  });
}
function renderConstraints() {
  const host = $('#view-constraints');
  const unavailable = readyMarkup('constraints');
  if (unavailable) { host.innerHTML = unavailable; return; }
  const constraints = asArray(resources.constraints.data.constraints);
  const pending = constraints.filter(item => !isApproved(item));
  const enabled = constraints.filter(effectiveEnabled);
  const unsafe = pending.filter(item => item.enabled);
  const summaries = '<div class="summary-row">' + summaryCard('约束总数', String(constraints.length), '来源与批准状态逐条记录') + summaryCard('允许使用的来源状态', String(constraints.filter(isApproved).length), '用户确认、工作选择及工程 / 数学定义', 'good') + summaryCard('待用户确认', String(pending.length), '未经批准的假设默认禁用', pending.length ? 'warn' : '') + summaryCard('实际启用', String(enabled.length), unsafe.length ? unsafe.length + ' 项未批准却启用，需纠正' : '根据后端当前台账', unsafe.length ? 'warn' : '') + '</div>';
  host.innerHTML = resourceNotice('constraints') + summaries + (unsafe.length ? '<div class="inline-note warn"><strong>发现 ' + unsafe.length + ' 项未批准却启用的约束。</strong> 请检查后端台账并纠正；本界面不会将其自动标记为已批准。</div>' : '<div class="inline-note"><strong>修改不会自动成为用户事实。</strong> 普通编辑保留原始来源与批准状态；确认假设需要单独点击“用户确认”并明确勾选。引擎未接入时，开关仅保存配置。</div>') + '<div class="toolbar"><label class="search-input"><span aria-hidden="true">⌕</span><input id="constraint-search" type="search" placeholder="搜索名称、表达式、范围或来源" aria-label="搜索约束" value="' + escapeHtml(constraintFilters.query) + '"></label><select class="filter-select" id="filter-approval" aria-label="按批准状态筛选"><option value="all"' + (constraintFilters.approval === 'all' ? ' selected' : '') + '>全部批准状态</option><option value="unapproved"' + (constraintFilters.approval === 'unapproved' ? ' selected' : '') + '>待确认 / 未批准</option><option value="approved"' + (constraintFilters.approval === 'approved' ? ' selected' : '') + '>已确认 / 已批准</option></select><select class="filter-select" id="filter-origin" aria-label="按来源筛选">' + constraintSelectOptions(constraints.map(item => item.origin ?? '未声明'), constraintFilters.origin, '全部来源') + '</select><select class="filter-select" id="filter-enabled" aria-label="按启用状态筛选"><option value="all"' + (constraintFilters.enabled === 'all' ? ' selected' : '') + '>全部开关</option><option value="enabled"' + (constraintFilters.enabled === 'enabled' ? ' selected' : '') + '>已启用</option><option value="disabled"' + (constraintFilters.enabled === 'disabled' ? ' selected' : '') + '>已禁用</option></select></div><div class="panel" id="constraint-table"></div>';
  renderConstraintTable();
}
function renderConstraintTable() {
  const host = $('#constraint-table');
  if (!host) return;
  const constraints = asArray(resources.constraints.data?.constraints);
  const filtered = filteredConstraints();
  if (!constraints.length) { host.innerHTML = emptyState('尚未登记约束', '后端返回了空台账，没有可编辑或批准的条目。', '≡'); return; }
  host.innerHTML = '<div class="table-wrap"><table class="data-table"><thead><tr><th style="width:30%">约束 / 作用范围</th><th>来源 / 类型</th><th>批准状态</th><th>进入求解</th><th style="text-align:right">操作</th></tr></thead><tbody>' + filtered.map(constraint => '<tr><td><button class="subtle-button cell-title" data-action="constraint-detail" data-id="' + escapeHtml(constraint.id) + '">' + escapeHtml(constraint.title || constraint.id) + '</button><div class="cell-id">' + escapeHtml(constraint.id) + '</div><div class="cell-detail">' + escapeHtml(display(constraint.scope)) + '</div></td><td><div class="badge-row">' + badge(constraint.origin || '未声明来源') + '</div><div class="cell-detail">' + escapeHtml(label(constraint.kind || '未声明类型')) + (isAssumption(constraint) ? ' · 假设' : '') + '</div></td><td>' + badge(constraint.approval || 'unconfirmed') + '<div class="cell-detail">' + escapeHtml(label(constraint.status || '未声明状态')) + '</div></td><td><div class="toggle-cell"><button class="switch" role="switch" aria-checked="' + !!constraint.enabled + '" aria-label="' + escapeHtml((constraint.enabled ? '禁用' : '启用') + '：' + (constraint.title || constraint.id)) + '" title="' + (constraint.enabled ? '禁用这项约束' : isApproved(constraint) ? '启用这项约束' : '启用前需要明确用户确认') + '" data-action="toggle-constraint" data-id="' + escapeHtml(constraint.id) + '"' + (mutationPending || resources.constraints.error ? ' disabled' : '') + '></button><span>' + (constraint.enabled ? '启用' : '禁用') + '</span></div>' + (constraint.enabled && !effectiveEnabled(constraint) ? '<div class="cell-detail">后端当前未生效</div>' : constraint.enabled && typeof constraint.effective_enabled === 'boolean' ? '<div class="cell-detail" style="color:var(--green)">后端已生效</div>' : '') + '</td><td><div class="cell-actions"><button class="button button-secondary button-small" data-action="constraint-edit" data-id="' + escapeHtml(constraint.id) + '">编辑</button>' + (!isApproved(constraint) ? '<button class="button button-primary button-small" data-action="constraint-confirm" data-id="' + escapeHtml(constraint.id) + '"' + (resources.constraints.error ? ' disabled' : '') + '>用户确认</button>' : '') + '</div></td></tr>').join('') + (filtered.length ? '' : '<tr><td colspan="5" class="table-message">没有符合当前筛选的约束。</td></tr>') + '</tbody></table></div><div class="list-summary"><span>显示 ' + filtered.length + ' / ' + constraints.length + ' 项</span><span>修订 ' + escapeHtml(display(resources.constraints.data.revision)) + '</span></div>';
}
function getConstraint(id) {
  return asArray(resources.constraints.data?.constraints).find(item => String(item.id) === String(id));
}
function evidenceMarkup(evidence) {
  const items = Array.isArray(evidence) ? evidence : evidence ? [evidence] : [];
  if (!items.length) return '<p class="compact-note">尚未提供来源证据。此项不会因此自动成为已确认事实。</p>';
  return items.map(item => {
    if (typeof item !== 'object' || item === null) return '<div class="source-item"><p class="evidence-quote">' + escapeHtml(display(item)) + '</p></div>';
    return '<div class="source-item"><h4>' + escapeHtml(item.label || item.title || '来源记录') + '</h4>' + (item.quote ? '<blockquote class="evidence-quote">' + escapeHtml(item.quote) + '</blockquote>' : '') + (item.path ? sourcePath(item.path) : '<p class="compact-note">未提供路径</p>') + '</div>';
  }).join('');
}
function openConstraint(id, editing = false) {
  const constraint = getConstraint(id);
  if (!constraint) { toast('当前台账中没有这项约束。', true); return; }
  const revision = resources.constraints.data.revision;
  const meta = '<div class="badge-row">' + badge(constraint.approval || 'unconfirmed') + badge(constraint.origin || '未声明来源') + badge(constraint.enabled ? 'enabled' : 'disabled') + (typeof constraint.effective_enabled === 'boolean' ? '<span class="badge outline">实际' + (effectiveEnabled(constraint) ? '生效' : '未生效') + '</span>' : '') + '</div><p class="drawer-intro">' + escapeHtml(label(constraint.kind || '未声明类型')) + ' · 修订 ' + escapeHtml(display(revision)) + '</p>';
  const evidence = '<section class="drawer-section"><h3>来源证据</h3>' + evidenceMarkup(constraint.evidence) + '</section>';
  let content;
  if (editing) {
    const jsonExpression = typeof constraint.expression === 'object' && constraint.expression !== null;
    const jsonScope = typeof constraint.scope === 'object' && constraint.scope !== null;
    const unapproved = !isApproved(constraint);
    content = meta + '<div class="inline-note' + (unapproved ? ' warn' : '') + '">来源和批准状态在普通编辑中保持原值。' + (unapproved ? '此条尚未批准，普通保存保持禁用。' : '每次保存均由后端记录变更。引擎未接入时，开关仅保存配置。') + '</div><form id="constraint-form" data-id="' + escapeHtml(id) + '"><label class="form-field"><span class="form-label">名称</span><input type="text" name="title" value="' + escapeHtml(constraint.title || '') + '" required></label><label class="form-field"><span class="form-label">表达式' + (jsonExpression ? '（JSON）' : '') + '</span><textarea name="expression" rows="5" class="mono" spellcheck="false">' + escapeHtml(noteText(constraint.expression)) + '</textarea><span class="form-help">描述进入求解的实际约束；单位、参数语义和几何归属应明确。</span></label><label class="form-field"><span class="form-label">作用范围' + (jsonScope ? '（JSON）' : '') + '</span><textarea name="scope" rows="2" spellcheck="false">' + escapeHtml(noteText(constraint.scope)) + '</textarea></label><label class="form-field"><span class="form-label">责任归属</span><input type="text" name="owner" value="' + escapeHtml(display(constraint.owner) === '—' ? '' : display(constraint.owner)) + '"></label><label class="form-field"><span class="form-label">说明 / 待核实事项</span><textarea name="notes" rows="4">' + escapeHtml(noteText(constraint.notes)) + '</textarea></label><label class="form-check"><input type="checkbox" name="enabled"' + (constraint.enabled && !unapproved ? ' checked' : '') + (unapproved ? ' disabled' : '') + '><span>启用此项，允许进入求解' + (unapproved ? '<span class="form-help">需通过单独的“用户确认”操作后才能启用。</span>' : '') + '</span></label><div id="constraint-form-error" hidden></div><div class="form-actions"><span class="form-dirty" id="constraint-dirty">修改后保存到后端台账</span><button class="button button-secondary" type="button" data-action="constraint-detail" data-id="' + escapeHtml(id) + '">查看详情</button><button class="button button-primary" type="submit"' + (resources.constraints.error ? ' disabled' : '') + '>保存修改</button></div></form>' + evidence + '<section class="drawer-section">' + rawDetails(constraint, '查看完整约束记录 / 历史字段') + '</section>';
    openDrawer('EDIT CONSTRAINT', constraint.title || id, content, {type:'constraint-edit',id,revision,original:structuredClone(constraint),jsonExpression,jsonScope});
  } else {
    content = meta + '<dl class="key-values"><dt>标识</dt><dd class="mono">' + escapeHtml(id) + '</dd><dt>原始来源</dt><dd>' + escapeHtml(label(constraint.origin)) + '</dd><dt>批准状态</dt><dd>' + escapeHtml(label(constraint.approval)) + '</dd><dt>作用范围</dt><dd>' + escapeHtml(noteText(constraint.scope) || '—') + '</dd><dt>责任归属</dt><dd>' + escapeHtml(display(constraint.owner)) + '</dd><dt>记录状态</dt><dd>' + escapeHtml(label(constraint.status)) + '</dd><dt>当前运行绑定</dt><dd>' + escapeHtml(constraint.effective_reason || '运行未报告绑定') + '</dd><dt>具体实现</dt><dd>' + escapeHtml(noteText(constraint.implementation) || '未报告') + '</dd></dl><section class="drawer-section"><h3>约束表达式</h3><pre class="json-block">' + escapeHtml(noteText(constraint.expression) || '尚未提供表达式') + '</pre></section><section class="drawer-section"><h3>说明</h3><p class="compact-note" style="white-space:pre-wrap">' + escapeHtml(noteText(constraint.notes) || '尚未补充说明。') + '</p></section>' + evidence + '<div class="form-actions"><button class="button button-secondary" data-action="constraint-edit" data-id="' + escapeHtml(id) + '">编辑此项</button>' + (!isApproved(constraint) ? '<button class="button button-primary" data-action="constraint-confirm" data-id="' + escapeHtml(id) + '"' + (resources.constraints.error ? ' disabled' : '') + '>用户确认</button>' : '') + '</div><section class="drawer-section">' + rawDetails(constraint, '查看完整约束记录 / 历史字段') + '</section>';
    openDrawer('CONSTRAINT RECORD', constraint.title || id, content, {type:'constraint',id,revision});
  }
}
async function saveConstraint(form) {
  if (mutationPending || !drawerState || drawerState.type !== 'constraint-edit') return;
  const snapshot = drawerState;
  const original = snapshot.original;
  const values = new FormData(form);
  const errorBox = $('#constraint-form-error');
  errorBox.hidden = true;
  let expression = values.get('expression');
  let scope = values.get('scope');
  try {
    if (snapshot.jsonExpression) expression = JSON.parse(expression);
    if (snapshot.jsonScope) scope = JSON.parse(scope);
  } catch { showFormError('表达式或作用范围的 JSON 格式不正确，请检查后再保存。'); return; }
  const notes = Array.isArray(original.notes) ? String(values.get('notes')).split(/\r?\n/) : values.get('notes');
  const candidate = {title:String(values.get('title')).trim(),expression,scope,owner:String(values.get('owner')).trim(),notes,enabled:isApproved(original) && values.get('enabled') === 'on'};
  if (!candidate.title) { showFormError('请输入约束名称。'); return; }
  const changes = Object.fromEntries(Object.entries(candidate).filter(([key,value]) => JSON.stringify(value) !== JSON.stringify(original[key] ?? (typeof value === 'boolean' ? false : ''))));
  if (!Object.keys(changes).length) { toast('没有需要保存的修改。'); return; }
  setFormBusy(form, true);
  try {
    await mutateConstraint(original.id, changes, snapshot.revision);
    toast('修改已写入约束台账，批准状态保持原值。');
    openConstraint(original.id, false);
  } catch (error) { showFormError(error.message); }
  finally { if (form.isConnected) setFormBusy(form, false); }
}
function showFormError(message) {
  const box = $('#constraint-form-error');
  if (!box) { toast(message, true); return; }
  box.className = 'form-error'; box.hidden = false; box.textContent = message;
}
function setFormBusy(form, busy) { $$('button[type=submit]', form).forEach(button => { button.disabled = busy; button.textContent = busy ? '正在保存…' : '保存修改'; }); }
async function mutateConstraint(id, changes, revision) {
  if (mutationPending) throw new Error('另一项修改正在保存，请稍后重试。');
  if (revision === undefined || revision === null) throw new Error('后端没有提供修订号，无法进行安全更新。');
  mutationPending = true;
  renderConstraintTable();
  try {
    const payload = await requestJson('/api/constraints/' + encodeURIComponent(id), {method:'PUT',body:JSON.stringify({expected_revision:revision,changes})});
    if (!Array.isArray(payload.constraints) || payload.revision === undefined) throw new Error('保存返回缺少完整约束台账或修订号，请刷新核对。');
    resources.constraints.data = payload;
    resources.constraints.error = null;
    resources.constraints.at = new Date();
    return payload;
  } catch (error) {
    if (error.status === 409) {
      await loadResource('constraints', true);
      throw new Error('台账已被其他操作更新。当前修改未覆盖新修订；请重新打开条目，核对后再保存。');
    }
    throw error;
  } finally { mutationPending = false; renderHeader(); if (currentView === 'constraints') renderConstraints(); else if (currentView === 'architecture') renderArchitecture(); }
}
async function toggleConstraint(id) {
  const constraint = getConstraint(id);
  if (!constraint) return;
  if (!constraint.enabled && !isApproved(constraint)) { openConfirm(id, true); return; }
  const desired = !constraint.enabled;
  try {
    await mutateConstraint(id, {enabled:desired}, resources.constraints.data.revision);
    toast('约束已' + (desired ? '启用' : '禁用') + '，变更已记录。');
    if (drawerState?.id === id && drawerState.type === 'constraint') openConstraint(id);
  } catch (error) { toast(error.message, true); }
}
function openConfirm(id, enableAfter = false) {
  const constraint = getConstraint(id);
  if (!constraint) return;
  if (resources.constraints.error) { toast('约束接口连接异常，请先刷新台账。', true); return; }
  confirmTarget = {id,revision:resources.constraints.data.revision};
  $('#confirm-content').innerHTML = '<p class="confirm-description">此动作将把该项的批准状态写为“用户已确认”，并记录本次明确确认。请先核对表达式、范围和证据。</p><div class="confirm-target"><h3>' + escapeHtml(constraint.title || id) + '</h3><code>' + escapeHtml(noteText(constraint.expression) || '尚未提供表达式') + '</code><small>作用范围：' + escapeHtml(noteText(constraint.scope) || '尚未声明') + '</small><small>原始来源：' + escapeHtml(label(constraint.origin)) + ' · 当前批准：' + escapeHtml(label(constraint.approval)) + '</small></div><label class="form-check"><input type="checkbox" id="confirm-explicit"><span>我已核对该项内容，明确将它确认为我的约束；这不是系统自动确认。</span></label><label class="form-check"><input type="checkbox" id="confirm-enable"' + (enableAfter || constraint.enabled ? ' checked' : '') + '><span>同时启用，允许这项约束进入求解</span></label><div id="confirm-error" hidden></div>';
  $('#confirm-submit').disabled = true;
  $('#confirm-submit').textContent = '确认并保存';
  $('#confirm-dialog').showModal();
}
async function submitConfirmation() {
  if (!confirmTarget || !$('#confirm-explicit')?.checked || mutationPending) return;
  const target = confirmTarget;
  const enabled = !!$('#confirm-enable').checked;
  $('#confirm-submit').disabled = true;
  $('#confirm-submit').textContent = '正在记录…';
  const errorBox = $('#confirm-error');
  try {
    await mutateConstraint(target.id, {approval:'user_confirmed',enabled}, target.revision);
    $('#confirm-dialog').close();
    toast('用户确认已记录；约束' + (enabled ? '已启用。' : '保持禁用。'));
    if (drawerState?.id === target.id) openConstraint(target.id);
  } catch (error) {
    errorBox.hidden = false; errorBox.className = 'form-error'; errorBox.textContent = error.message;
    $('#confirm-submit').textContent = '确认并保存';
    $('#confirm-submit').disabled = !$('#confirm-explicit')?.checked;
  }
}

function datasetColumns(dataset) {
  if (Array.isArray(dataset.columns)) return dataset.columns.map(column => typeof column === 'object' && column !== null ? column : {name:String(column)});
  if (dataset.columns && typeof dataset.columns === 'object') return Object.entries(dataset.columns).map(([name,value]) => typeof value === 'object' && value !== null ? {name,...value} : {name,type:display(value)});
  return [];
}
function datasetRows(dataset) {
  if (Array.isArray(dataset.rows)) return dataset.rows;
  if (Array.isArray(dataset.records)) return dataset.records;
  if (Array.isArray(dataset.preview)) return dataset.preview;
  if (Array.isArray(dataset.sample)) return dataset.sample;
  return null;
}
function datasetCount(dataset) {
  if (numeric(dataset.row_count) !== null) return numberText(dataset.row_count);
  if (numeric(dataset.rows) !== null) return numberText(dataset.rows);
  const rows = datasetRows(dataset);
  return rows ? numberText(rows.length) : '—';
}

function contractTypes(payload) {
  const collection=payload?.types || payload?.contracts || payload?.definitions || payload?.catalog;
  if(Array.isArray(collection))return collection.map((item,index)=>typeof item==='object'&&item!==null?{...item,name:item.name||item.type_name||item.id||item.type||'未命名类型 '+index}:{name:String(item),fields:[]});
  if(collection&&typeof collection==='object')return Object.entries(collection).map(([name,item])=>typeof item==='object'&&item!==null?{name,...item}:{name,description:display(item),fields:[]});
  return [];
}
function contractFields(type) {
  if(Array.isArray(type.fields))return type.fields.map(field=>typeof field==='object'&&field!==null?field:{name:String(field)});
  if(type.fields&&typeof type.fields==='object')return Object.entries(type.fields).map(([name,field])=>typeof field==='object'&&field!==null?{name,...field}:{name,type:display(field)});
  if(type.properties&&typeof type.properties==='object')return Object.entries(type.properties).map(([name,field])=>typeof field==='object'&&field!==null?{name,...field}:{name,type:display(field)});
  return [];
}
function renderData() {
  const modeBar='<div class="data-mode-bar" role="group" aria-label="数据入口内容"><button class="data-mode-button'+(dataMode==='datasets'?' active':'')+'" data-action="data-mode" data-mode="datasets" aria-pressed="'+(dataMode==='datasets')+'">当前数据集</button><button class="data-mode-button'+(dataMode==='contracts'?' active':'')+'" data-action="data-mode" data-mode="contracts" aria-pressed="'+(dataMode==='contracts')+'">数据结构</button><button class="data-mode-button" data-action="observation-browser">原片取点与撤点</button><span>数据结构来自后端当前契约反射</span></div>';
  const host=$('#view-data');
  if(dataMode==='datasets'){renderDatasetContents();host.innerHTML=modeBar+host.innerHTML;return;}
  const unavailable=readyMarkup('contracts');
  if(unavailable){host.innerHTML=modeBar+unavailable;return;}
  const data=resources.contracts.data,types=contractTypes(data);
  host.innerHTML=modeBar+resourceNotice('contracts')+'<div class="inline-note"><strong>当前数据结构。</strong> 类型、字段含义、单位、归属、不可变边界与引用关系均读取 /api/contracts，不维护前端展示副本。</div><div class="toolbar"><label class="search-input"><span aria-hidden="true">⌕</span><input id="contract-search" type="search" placeholder="搜索类型、字段、含义、单位或引用" aria-label="搜索数据结构" value="'+escapeHtml(contractQuery)+'"></label><span class="toolbar-note">'+types.length+' 个实际登记类型 · 修订 '+escapeHtml(display(data.revision || data.schema_version || data.version))+'</span></div><div id="contract-type-directory" class="contract-type-directory"></div><div id="contract-type-cards" class="contract-type-cards"></div><div class="section-gap">'+rawDetails(data,'查看完整契约目录响应')+'</div>';
  renderContractTypes();
}
function renderContractTypes() {
  const data=resources.contracts.data;if(!data)return;
  const host=$('#contract-type-cards'),directory=$('#contract-type-directory');if(!host||!directory)return;
  const query=contractQuery.toLocaleLowerCase().trim(),allTypes=contractTypes(data);
  const types=allTypes.filter(type=>!query||[type.name,type.description,type.meaning,type.docstring,JSON.stringify(type.fields||type.properties)].join(' ').toLocaleLowerCase().includes(query));
  directory.innerHTML=types.map((type,index)=>'<button class="contract-directory-item" data-action="contract-jump" data-index="'+index+'">'+escapeHtml(type.name)+'</button>').join('');
  host.innerHTML=types.length?types.map((type,index)=>{
    const fields=contractFields(type),description=type.description||type.meaning||type.docstring||type.doc;
    return '<details class="contract-card" id="contract-type-'+index+'"'+(index===0||query?' open':'')+'><summary><span><strong>'+escapeHtml(type.name)+'</strong>'+(type.owner?'<small>归属 '+escapeHtml(display(type.owner))+'</small>':'')+'</span><span class="badge outline">'+fields.length+' 字段</span></summary><div class="contract-card-content">'+(description?'<p class="contract-type-description">'+escapeHtml(noteText(description))+'</p>':'<p class="compact-note">API尚未提供类型含义说明。</p>')+(fields.length?'<div class="table-wrap"><table class="contract-fields"><thead><tr><th>字段 / 含义</th><th>类型</th><th>必填</th><th>单位</th><th>归属</th><th>不可变</th><th>引用关系</th></tr></thead><tbody>'+fields.map(field=>'<tr><td><strong class="mono">'+escapeHtml(field.name||field.id||'未命名')+'</strong><p>'+escapeHtml(noteText(field.description||field.meaning||field.doc)||'尚未说明')+'</p></td><td class="mono">'+escapeHtml(noteText(field.type||field.annotation||field.dtype)||'未声明')+'</td><td>'+escapeHtml(typeof field.required==='boolean'?(field.required?'是':'否'):noteText(field.required)||'—')+'</td><td>'+escapeHtml(noteText(field.unit)||'—')+'</td><td>'+escapeHtml(noteText(field.owner)||(type.owner?'类型：'+noteText(type.owner):'—'))+'</td><td>'+escapeHtml(typeof field.immutable==='boolean'?(field.immutable?'是':'否'):noteText(field.immutable)||(typeof type.immutable==='boolean'?'随类型：'+(type.immutable?'是':'否'):'—'))+'</td><td class="mono">'+escapeHtml(noteText(field.refs||field.references||field.ref)||'—')+'</td></tr>').join('')+'</tbody></table></div>':'<p class="compact-note">此类型尚未返回字段定义。</p>')+(type.source_file?'<section class="drawer-section"><h3>定义位置</h3>'+sourcePath(type.source_file)+'</section>':'')+'<div class="section-gap">'+rawDetails(type,'查看类型完整反射记录')+'</div></div></details>';
  }).join(''):emptyState(allTypes.length?'没有匹配的数据结构':'API尚未返回类型目录',allTypes.length?'调整搜索条件查看当前契约。':'已保留原始API响应，待后端登记类型和字段。','▤');
}

function renderDatasetContents() {
  const host = $('#view-data');
  const unavailable = readyMarkup('data');
  if (unavailable) { host.innerHTML = unavailable; return; }
  const datasets = asArray(resources.data.data.datasets);
  host.innerHTML = resourceNotice('data') + '<div class="inline-note"><strong>当前数据契约。</strong> 数据准备交付可独立消费的当前输入；来源、原片 PTS、人工确认与缺失信息应随记录保留。</div><div class="toolbar"><label class="search-input"><span aria-hidden="true">⌕</span><input id="dataset-search" type="search" placeholder="搜索数据集、来源或字段" aria-label="搜索数据集" value="' + escapeHtml(dataQuery) + '"></label><span class="toolbar-note">' + datasets.length + ' 个实际登记的数据集 · 点击查看结构与记录</span></div><div class="panel" id="dataset-table"></div>';
  renderDatasetTable();
}
function renderDatasetTable() {
  const host = $('#dataset-table');
  if (!host) return;
  const datasets = asArray(resources.data.data.datasets);
  const query = dataQuery.toLocaleLowerCase().trim();
  const filtered = datasets.filter(dataset => !query || [dataset.id,dataset.name,display(dataset.source),noteText(dataset.columns),noteText(dataset.notes)].join(' ').toLocaleLowerCase().includes(query));
  if (!datasets.length) { host.innerHTML = emptyState('尚未导入数据', '当前 API 没有登记数据集。原片、观测、人工确认与模型输入将从这里检查。', '▤'); return; }
  host.innerHTML = '<div class="table-wrap"><table class="data-table"><thead><tr><th style="width:25%">数据集</th><th>状态</th><th>来源</th><th>记录数</th><th>字段 / 单位</th><th></th></tr></thead><tbody>' + filtered.map(dataset => '<tr><td><button class="subtle-button cell-title" data-action="dataset-detail" data-id="' + escapeHtml(dataset.id) + '">' + escapeHtml(dataset.name || dataset.id) + '</button><div class="cell-id">' + escapeHtml(dataset.id) + '</div><div class="cell-detail">' + escapeHtml(noteText(dataset.notes)) + '</div></td><td>' + badge(dataset.status || '未声明状态') + '</td><td><div class="cell-detail" style="margin:0;max-width:250px">' + escapeHtml(display(dataset.source)) + '</div></td><td><span class="data-count">' + datasetCount(dataset) + '</span></td><td><div class="column-tags">' + datasetColumns(dataset).map(column => '<span class="column-tag" title="' + escapeHtml(column.description || '') + '">' + escapeHtml(column.name || column.id || '未命名') + (column.unit ? ' · ' + escapeHtml(column.unit) : '') + '</span>').join('') + (datasetColumns(dataset).length ? '' : '<span class="compact-note">字段未声明</span>') + '</div></td><td><button class="button button-secondary button-small" data-action="dataset-detail" data-id="' + escapeHtml(dataset.id) + '">检查</button></td></tr>').join('') + (filtered.length ? '' : '<tr><td colspan="6" class="table-message">没有符合搜索的数据集。</td></tr>') + '</tbody></table></div><div class="list-summary"><span>显示 ' + filtered.length + ' / ' + datasets.length + ' 个数据集</span><span>仅展示当前 API 提供的记录</span></div>';
}
function openDataset(id) {
  const dataset = asArray(resources.data.data?.datasets).find(item => String(item.id) === String(id));
  if (!dataset) return;
  const columns = datasetColumns(dataset);
  const rows = datasetRows(dataset);
  const source = typeof dataset.source === 'string' ? sourcePath(dataset.source) : '<pre class="json-block">' + escapeHtml(noteText(dataset.source) || '未提供来源') + '</pre>';
  let content = '<div class="badge-row">' + badge(dataset.status || '未声明状态') + '<span class="badge outline mono">' + escapeHtml(dataset.id) + '</span><span class="badge outline">' + datasetCount(dataset) + ' 条记录</span></div><p class="drawer-intro">' + escapeHtml(noteText(dataset.notes) || '尚未补充说明。') + '</p><section class="drawer-section"><h3>输入来源</h3>' + source + '</section><section class="drawer-section"><h3>字段契约 · ' + columns.length + '</h3>';
  content += columns.length ? '<div class="table-wrap"><table class="detail-table"><thead><tr><th>字段</th><th>类型</th><th>单位</th><th>说明 / 来源</th></tr></thead><tbody>' + columns.map(column => '<tr><td class="mono">' + escapeHtml(column.name || column.id || '—') + '</td><td>' + escapeHtml(column.type || column.dtype || '—') + '</td><td>' + escapeHtml(column.unit || '未声明') + '</td><td>' + escapeHtml([column.description,column.source].filter(Boolean).join(' · ') || '—') + '</td></tr>').join('') + '</tbody></table></div>' : '<p class="compact-note">后端尚未提供字段与单位定义。</p>';
  content += '</section><section class="drawer-section"><h3>实际记录</h3>';
  if (rows && rows.length) {
    const keys = columns.length ? columns.map(column => column.name || column.id).filter(Boolean) : [...new Set(rows.flatMap(row => row && typeof row === 'object' && !Array.isArray(row) ? Object.keys(row) : []))];
    if (!keys.length) content += '<pre class="json-block">' + escapeHtml(JSON.stringify(rows,null,2)) + '</pre>';
    else content += '<div class="preview-table"><table class="detail-table"><thead><tr>' + keys.map(key => '<th class="mono">' + escapeHtml(key) + '</th>').join('') + '</tr></thead><tbody>' + rows.map(row => '<tr>' + keys.map((key,index) => '<td>' + escapeHtml(display(Array.isArray(row) ? row[index] : row?.[key])) + '</td>').join('') + '</tr>').join('') + '</tbody></table></div><p class="compact-note" style="margin-top:8px">当前 API 提供了 ' + rows.length + ' 条明细；是否为全部记录以数据契约为准。</p>';
  } else content += '<p class="compact-note">' + (rows ? '当前明细为空。' : '当前 API 仅提供数据集元信息，未返回行记录；无法在此核对观测值。') + '</p>';
  content += '</section><section class="drawer-section">' + rawDetails(dataset) + '</section>';
  openDrawer('DATASET CONTRACT', dataset.name || id, content, {type:'dataset',id});
}
function resourceGauge({name,detail,value,valueText,meterValue,bottomLeft,bottomRight}) {
  const safeMeter = numeric(meterValue) === null ? null : Math.max(0, Math.min(100, meterValue));
  return '<div class="hardware-card"><div class="hardware-top"><div><div class="hardware-name">' + escapeHtml(name) + '</div><div class="hardware-detail">' + escapeHtml(detail || '未提供设备详情') + '</div></div><div class="resource-value">' + escapeHtml(valueText || pct(value)) + '</div></div><div class="meter"' + (safeMeter !== null ? ' role="meter" aria-label="' + escapeHtml(name) + ' 使用率" aria-valuemin="0" aria-valuemax="100" aria-valuenow="' + safeMeter + '"' : ' aria-hidden="true"') + '><div class="meter-fill' + (safeMeter >= 85 ? ' high' : '') + '" style="width:' + (safeMeter ?? 0) + '%"></div></div><div class="resource-bottom"><span>' + escapeHtml(bottomLeft || (safeMeter === null ? 'API 未提供使用率' : '实际读数')) + '</span><span>' + escapeHtml(bottomRight || '') + '</span></div></div>';
}
function runFailureMarkup(diagnostics) {
  const failure=diagnostics?.last_domain_failure;
  if(!failure)return '';
  const fields=[['域检查',failure.family],['原生 PTS',failure.native_pts],['具体源点',failure.observation_id],['有限曲线',failure.curve_id],['试步曲线站标',failure.world_station],['试步形式行',failure.formal_row],['试步光线半径',failure.projection_ray_radius],['拒绝试步计数',diagnostics.counters?.domain_rejections]];
  const link=Number.isInteger(failure.native_pts)&&failure.observation_id?'<button class="button button-secondary button-small" data-action="observation-browser" data-native-pts="'+failure.native_pts+'" data-query="'+escapeHtml(failure.observation_id)+'">核对对应原片与实体</button>':'';
  return '<section class="drawer-section"><h3>最近一次未接受试步</h3><p class="compact-note">以下属于被拒绝的候选，不代表当前已接受模型的投影状态。错误不会作为成功步发布。</p><dl class="key-values">'+fields.map(([name,value])=>'<dt>'+name+'</dt><dd class="mono">'+escapeHtml(display(value))+'</dd>').join('')+'</dl>'+link+rawDetails(failure,'查看具体域检查与试步证据')+'</section>';
}
function runDiagnosticsMarkup(state,run) {
  const diagnostics=run.diagnostics || state.diagnostics;
  const identity='<dl class="key-values"><dt>模型结构</dt><dd>'+escapeHtml(display(run.model_id))+' · r'+escapeHtml(display(run.model_revision))+'</dd><dt>参数修订</dt><dd>'+escapeHtml(display(run.parameter_revision))+'</dd><dt>状态摘要</dt><dd><code>'+escapeHtml(display(run.parameter_sha256))+'</code></dd><dt>运行状态更新时间</dt><dd>'+escapeHtml(dateText(run.updated_at))+'</dd></dl>';
  if(!diagnostics)return '<section class="drawer-section"><h3>实际状态身份</h3>'+identity+'<p class="compact-note">本次运行尚未报告约束诊断。</p></section>';
  const precise=value=>numeric(value)===null?'—':value===0?'0':value.toExponential(3);
  const b=diagnostics.budget_squared ?? diagnostics.budget_sq ?? diagnostics.b ?? run.budget_squared;
  const metrics=[['全局预算 b（平方）',b],['当前阶段目标',diagnostics.objective],['既定E预算外源点数',diagnostics.engineering_budget?.violating_at_current_target],['原UV像素差 RMS',diagnostics.pixel_error?.rms],['原UV最大像素差',diagnostics.pixel_error?.maximum],['最坏像素误差 / 既定E',diagnostics.engineering_budget?.maximum_ratio],['规范化可行性 ∞',diagnostics.normalized_feasibility_inf],['投影驻点条件 ∞',diagnostics.normalized_projected_stationarity_inf],['互补条件 ∞',diagnostics.normalized_complementarity_inf],['对偶违反 ∞',diagnostics.normalized_dual_violation_inf]];
  const objectiveStage=diagnostics.stage || run.stage;
  const objectiveNote=objectiveStage==='feasibility'?'当前最小化一个全源共享工程预算 b；原W与独立σ保留到原预算内细化，RS和Query等式仍须满足。':objectiveStage==='statistical_level_set'?'当前临时固定保存的宽松b上界优化原统计；该阶段不保证真实材料贴合，不能代替回原预算收紧。':objectiveStage==='statistics'?'当前b固定为1，在全部原工程预算与RS/Query等式内细化原W及原独立σ统计。':'';
  const families=asArray(diagnostics.families);
  const body=families.map(f=>'<tr><td>'+escapeHtml(display(f.id))+'<small class="compact-note" style="display:block">'+escapeHtml(display(f.sense))+' · '+escapeHtml(display(f.unit))+'</small></td><td class="numeric">'+precise(f.raw_violation_inf)+'</td><td class="numeric">'+precise(f.normalized_violation_inf)+'</td><td class="numeric">'+precise(f.penalty)+'</td></tr>').join('');
  return '<section class="drawer-section"><h3>实际状态身份</h3>'+identity+'</section><section class="drawer-section"><h3>数值阶段与约束</h3><p>'+escapeHtml(label(run.scope || '未报告数据范围'))+' · '+escapeHtml(label(diagnostics.stage || run.stage || '尚未报告阶段'))+'</p><dl class="key-values">'+metrics.map(([name,value])=>'<dt>'+name+'</dt><dd class="mono">'+precise(value)+'</dd>').join('')+'</dl>'+(objectiveNote?'<p class="compact-note">'+escapeHtml(objectiveNote)+'</p>':'')+'<p class="compact-note">b=1对应当前像素预算。可行性、代价下降和正常退出均不能替代整镜头真实贴合验收。</p>'+(families.length?'<div class="table-wrap"><table class="data-table"><thead><tr><th>约束族与单位</th><th>原单位违反 ∞</th><th>规范化违反 ∞</th><th>ρ</th></tr></thead><tbody>'+body+'</tbody></table></div>':'<p class="compact-note">后端尚未报告约束族。</p>')+runFailureMarkup(diagnostics)+rawDetails(diagnostics,'查看原始单位、缩放定义与约束来源')+'</section>';
}

function renderMonitor() {
  const host = $('#view-monitor');
  const openDetails = $$('details',host).map(element => element.open);
  const focusedId = host.contains(document.activeElement) ? document.activeElement.id : null;
  const unavailable = readyMarkup('state');
  if (unavailable) { host.innerHTML = unavailable; return; }
  const state = resources.state.data;
  const run = state.run;
  const hardware = state.hardware || {};
  const gpus = asArray(hardware.gpus);
  const events = asArray(state.events);
  const memoryTotal = numeric(hardware.memory_total_gb);
  const memoryUsed = numeric(hardware.memory_used_gb);
  const memoryPercent = memoryTotal && memoryUsed !== null ? memoryUsed / memoryTotal * 100 : null;
  let runPanel;
  if (run) {
    runPanel = '<div class="panel"><div class="run-header"><div><h2 class="run-title">' + escapeHtml(run.name || '当前运行') + '</h2><p class="run-subtitle">开始于 ' + escapeHtml(dateText(run.started_at)) + '</p></div>' + badge(run.status || state.phase) + '</div><dl class="run-stats"><div class="stat-item"><dt>当前迭代</dt><dd>' + numberText(run.iteration) + '</dd></div><div class="stat-item"><dt>实际代价</dt><dd>' + (numeric(run.cost) === null ? '—' : escapeHtml(run.cost.toPrecision(6))) + '</dd></div><div class="stat-item"><dt>已接受步</dt><dd>' + numberText(run.accepted_steps) + '</dd></div></dl><div class="run-log"><span class="section-label">本次主日志</span>' + sourcePath(run.main_log) + '</div><div class="panel-body">' + runDiagnosticsMarkup(state,run) + '</div><div class="panel-footer">' + escapeHtml(state.message || '后端未提供额外运行说明。') + '</div></div>';
  } else {
    runPanel = '<div class="panel"><div class="panel-header"><h2>当前运行</h2>' + badge(state.phase || 'no_run') + '</div>' + emptyState('当前没有求解运行', state.message || '后端没有报告运行实例。迭代、代价与进度均不作模拟。', '⌁') + '</div>';
  }
  let hardwareMarkup = resourceGauge({name:'CPU',detail:hardware.cpu_count !== null && hardware.cpu_count !== undefined ? display(hardware.cpu_count) + ' 个逻辑处理器' : '处理器数量未提供',value:hardware.cpu_utilization,meterValue:hardware.cpu_utilization,bottomLeft:'CPU 实际占用',bottomRight:numeric(hardware.cpu_utilization) === null ? '读数不可用' : '系统采样'});
  hardwareMarkup += resourceGauge({name:'系统内存',detail:memoryTotal !== null ? '总计 ' + numberText(memoryTotal,1) + ' GB' : '总内存未提供',valueText:memoryUsed !== null ? numberText(memoryUsed,1) + ' GB' : '—',meterValue:memoryPercent,bottomLeft:memoryUsed !== null && memoryTotal !== null ? numberText(memoryUsed,1) + ' / ' + numberText(memoryTotal,1) + ' GB' : 'API 未提供完整内存读数',bottomRight:memoryPercent !== null ? pct(memoryPercent) : ''});
  gpus.forEach((gpu,index) => {
    const gpuTotal = numeric(gpu.memory_total_gb), gpuUsed = numeric(gpu.memory_used_gb);
    hardwareMarkup += resourceGauge({name:gpu.name || 'GPU ' + index,detail:'GPU ' + index + (numeric(gpu.temperature) !== null ? ' · ' + numberText(gpu.temperature,1) + ' °C' : ''),value:gpu.utilization,meterValue:gpu.utilization,bottomLeft:gpuTotal !== null && gpuUsed !== null ? '显存 ' + numberText(gpuUsed,2) + ' / ' + numberText(gpuTotal,2) + ' GB' : '显存读数未提供',bottomRight:numeric(gpu.utilization) === null ? '占用读数不可用' : '设备实际占用'});
  });
  if (!gpus.length) hardwareMarkup += '<div class="hardware-card"><div class="hardware-name">GPU</div><p class="compact-note" style="margin-top:7px">当前 API 没有返回 GPU 设备或实际读数。</p></div>';
  host.innerHTML = resourceNotice('state') + renderShotSequence(state) + '<div class="toolbar"><span class="badge ' + (resources.state.error ? 'warn' : 'good') + '">' + (resources.state.error ? '保留上次数据' : '实际 API 读数') + '</span><span class="time-pill">采样于 ' + shortTime(resources.state.at) + '</span><label class="toolbar-check"><input id="pause-polling" type="checkbox"' + (pollPaused ? ' checked' : '') + '>暂停自动刷新</label><span class="toolbar-note">每 3 秒读取 /api/state；界面不启动求解</span></div><div class="monitor-layout"><div class="monitor-stack">' + runPanel + '<div class="panel"><div class="panel-header"><div><h2>运行事件</h2><p>后端报告的事件；不补写成功或失败记录</p></div><span class="badge outline">' + events.length + ' 条</span></div><div class="panel-body">' + (events.length ? '<div class="event-list">' + events.map(event => {
    if (typeof event !== 'object' || event === null) return '<div class="event-item"><span class="event-time">—</span><div class="event-title">' + escapeHtml(display(event)) + '</div></div>';
    return '<div class="event-item"><span class="event-time">' + escapeHtml(shortTime(event.timestamp || event.time || event.at)) + '</span><div><div class="event-title">' + escapeHtml(event.message || event.title || event.reason || event.event || '未命名事件') + '</div>' + (event.details || event.detail ? '<div class="event-detail">' + escapeHtml(noteText(event.details || event.detail)) + '</div>' : '') + '</div></div>';
  }).join('') + '</div>' : '<p class="compact-note">当前没有事件记录。</p>') + '</div></div></div><div class="monitor-stack"><div class="panel"><div class="panel-header"><div><h2>硬件实际占用</h2><p>设备占用不等于求解器独占使用</p></div></div><div class="panel-body"><div class="hardware-grid">' + hardwareMarkup + '</div></div><div class="panel-footer">缺失读数显示为“—”；可据此核对实际 CPU / GPU 使用。</div></div><div class="panel"><div class="panel-header"><h2>后端状态字段</h2></div><div class="panel-body">' + rawDetails(state) + '</div></div></div></div>';
  $$('details',host).forEach((element,index) => { element.open = !!openDetails[index]; });
  if (focusedId) document.getElementById(focusedId)?.focus({preventScroll:true});
}

function reviewEvidenceCard(item) {
  if (typeof item !== 'object' || item === null) return '<div class="evidence-card"><div class="evidence-description">' + escapeHtml(display(item)) + '</div></div>';
  const path = item.path || item.local_path || item.video_path || item.file;
  const time = item.source_time || item.source_timestamp || item.pts || item.time_range;
  const isVideo = item.kind === 'video' || item.type === 'video' || typeof path === 'string' && /\.(mp4|mkv|mov|avi|webm)$/i.test(path);
  return '<div class="evidence-card"><div class="evidence-title"><span>' + escapeHtml(item.label || item.title || item.name || item.id || '证据记录') + '</span>' + (item.status ? badge(item.status) : '') + '</div>' + (item.description || item.notes || item.conclusion ? '<p class="evidence-description">' + escapeHtml(noteText(item.description || item.notes || item.conclusion)) + '</p>' : '') + (time ? '<p class="source-time" style="margin-top:8px">源时间 / PTS：' + escapeHtml(display(time)) + '</p>' : '') + (path ? sourcePath(path) : '<p class="compact-note" style="margin-top:8px">未提供本地证据路径</p>') + (isVideo ? '<p class="compact-note" style="margin-top:8px">本地视频请用 PotPlayer 按源时间审阅。</p>' : '') + (item.quote ? '<blockquote class="evidence-quote">' + escapeHtml(item.quote) + '</blockquote>' : '') + '</div>';
}

function renderBaselinePanel(review) {
  const baselines = asArray(review.baselines);
  const cards = baselines.map(baseline => {
    if (typeof baseline !== 'object' || baseline === null) return '<div class="baseline-card"><p class="compact-note">' + escapeHtml(display(baseline)) + '</p></div>';
    const proof = baseline.proof || baseline.projection_proof || baseline.validation;
    const proofStatus = baseline.projection_status || baseline.equivalence_status || (typeof proof === 'object' && proof !== null ? proof.status : null);
    const path = baseline.path || baseline.source_path || baseline.model_path;
    return '<article class="baseline-card"><div class="evidence-title"><span>' + escapeHtml(baseline.name || baseline.title || baseline.id || '未命名基线') + '</span>' + (baseline.status ? badge(baseline.status) : badge('未声明状态')) + '</div><p class="cell-id">' + escapeHtml(display(baseline.id)) + '</p>' + (baseline.description || baseline.notes ? '<p class="evidence-description">' + escapeHtml(noteText(baseline.description || baseline.notes)) + '</p>' : '') + '<dl class="key-values baseline-values"><dt>原片时钟</dt><dd>' + escapeHtml(noteText(baseline.source_time || baseline.clock || baseline.source_clock) || '尚未登记') + '</dd><dt>投影对应验证</dt><dd>' + (proofStatus ? badge(proofStatus) : '<span class="compact-note">尚未提供验证状态</span>') + '</dd></dl>' + (path ? '<div style="margin-top:12px">' + sourcePath(path) + '</div>' : '<p class="compact-note" style="margin-top:12px">尚未提供原始基线文件路径。</p>') + (baseline.camera_path ? '<div style="margin-top:8px">' + sourcePath(baseline.camera_path) + '</div>' : '') + (proof ? '<div class="drawer-section">' + rawDetails(proof, '检查逐帧投影对应证明') + '</div>' : '<p class="compact-note" style="margin-top:12px">未提供逐帧投影对应证明；此处不宣称成功参数已经继承。</p>') + '<div style="margin-top:14px">' + rawDetails(baseline, '查看完整基线记录') + '</div></article>';
  }).join('');
  return '<div class="panel baseline-panel"><div class="panel-header"><div><h2>成功基线 · 原始参数与新模型对照</h2><p>独立导入成功 demo 数据，逐镜头核对源时钟与逐帧投影对应</p></div><span class="badge outline">' + baselines.length + ' 份已登记记录</span></div>' + (baselines.length ? '<div class="panel-body"><div class="baseline-grid">' + cards + '</div></div>' : emptyState('成功 demo 基线尚未登记', '后端尚未提供旧 9 镜头 demo 与 S01 成功基线的实际记录。找到原始参数后，分别保留导入状态和投影对应证明。', '↗')) + '<div class="panel-footer">初始化的数据路径、导入结果与投影验证各自记录，缺少证明的部分明确保留。</div></div>';
}

function renderReview() {
  const host = $('#view-review');
  const unavailable = readyMarkup('review');
  if (unavailable) { host.innerHTML = unavailable; return; }
  const review = resources.review.data;
  const evidence = asArray(review.evidence);
  const timeline = asArray(review.timeline);
  const conclusions = asArray(review.conclusions);
  host.innerHTML = resourceNotice('review') + '<div class="review-status"><div><h3>完整镜头与独立证据</h3><p>' + escapeHtml(review.message || review.description || '标线、广告边界与编号全部开启，连续审阅完整镜头。数值指标不替代真实证据。') + '</p></div>' + badge(review.status || '未声明状态') + '</div>' + renderBaselinePanel(review) + '<div class="review-layout"><div class="monitor-stack"><div class="panel"><div class="panel-header"><div><h2>可审阅证据</h2><p>原片来源、源时间与本地文件位置</p></div><span class="badge outline">' + evidence.length + ' 项</span></div>' + (evidence.length ? '<div class="panel-body"><div class="evidence-list">' + evidence.map(reviewEvidenceCard).join('') + '</div></div>' : emptyState('尚未导入影像证据', '当前接口没有可审阅的影像或独立验证记录。不能由代价、平滑或单元测试推定模型准确。', '□')) + '<div class="panel-footer">视频仅提供本地路径与源时间，在 PotPlayer 中播放。</div></div><div class="panel"><div class="panel-header"><h2>验证结论与限制</h2></div><div class="panel-body">' + (conclusions.length ? '<div class="evidence-list">' + conclusions.map(conclusion => typeof conclusion === 'object' && conclusion !== null ? '<div class="evidence-card"><div class="evidence-title"><span>' + escapeHtml(conclusion.title || conclusion.label || conclusion.shot || '结论') + '</span>' + (conclusion.status ? badge(conclusion.status) : '') + '</div><p class="evidence-description">' + escapeHtml(noteText(conclusion.message || conclusion.description || conclusion.conclusion || conclusion.notes || conclusion)) + '</p></div>' : '<p class="compact-note">' + escapeHtml(display(conclusion)) + '</p>').join('') + '</div>' : '<p class="compact-note">后端尚未登记验证结论。</p>') + '</div></div></div><div class="monitor-stack"><div class="panel"><div class="panel-header"><h2>证据与验证时间线</h2></div>' + (timeline.length ? '<div class="timeline">' + timeline.map(item => {
    if (typeof item !== 'object' || item === null) return '<div class="timeline-entry"><div class="timeline-title">' + escapeHtml(display(item)) + '</div></div>';
    return '<div class="timeline-entry"><div class="timeline-title">' + escapeHtml(item.title || item.label || item.event || '记录') + '</div><p class="timeline-detail">' + escapeHtml(noteText(item.description || item.message || item.notes)) + '</p><p class="timeline-time">' + escapeHtml(dateText(item.timestamp || item.time || item.at)) + '</p>' + (item.status ? badge(item.status) : '') + '</div>';
  }).join('') + '</div>' : '<div class="panel-body"><p class="compact-note">暂无已登记的时间线。</p></div>') + '</div><div class="note-card"><h3>审阅责任边界</h3><p>当前镜头的完整原片与未参与调参的真实证据分别检查。场地拟合仅接收已核实属于冰面或围挡本体的静止材料点；缺失与腾空保留原状。</p><button class="subtle-link" data-action="navigate" data-view="constraints">核对适用约束 →</button></div><div class="panel"><div class="panel-header"><h2>后端审阅记录</h2></div><div class="panel-body">' + rawDetails(review) + '</div></div></div></div>';
}


let modelDrawRevision = null;
let modelSelectedKey = null;
let modelResizePending = false;
function modelEntries(model) {
  return [...asArray(model.landmarks).map(item => ({type:'landmark',item})),...asArray(model.paths).map(item => ({type:'path',item})),...asArray(model.ads).map(item => ({type:'ad',item})),...asArray(model.rigs).map(item => ({type:'rig',item})),...asArray(model.entities).map(item => ({type:'entity',item}))];
}
function modelItemId(item) { return String(item.id ?? item.label ?? '未提供标识'); }
function modelItemLabel(item) { return String(item.short_label || (/^Ad\d+$/.test(String(item.id || '')) ? item.id : null) || item.label || item.name || item.id || '未提供标识'); }
function modelAxes(model) {
  const spec = model.top_view_axes || {};
  const indices = Array.isArray(spec.xy_indices) && spec.xy_indices.length === 2 ? spec.xy_indices : [0,1];
  return {indices,horizontal:spec.horizontal || 'x',vertical:spec.vertical || 'y',up:spec.up || 'z',yPositive:spec.y_positive || 'up'};
}
function modelXY(point, model) {
  const axes = modelAxes(model);
  if (!Array.isArray(point)) return null;
  const x = numeric(point[axes.indices[0]]), y = numeric(point[axes.indices[1]]);
  return x === null || y === null ? null : [x,y];
}
function modelGeometryPoints(model) {
  return [...asArray(model.paths).flatMap(item => asArray(item.points)),...asArray(model.ads).flatMap(item => asArray(item.footprint))].map(point => modelXY(point,model)).filter(Boolean);
}
function modelBounds(model, includeRigs) {
  const geometry = modelGeometryPoints(model);
  let points = [...geometry];
  const declared = !includeRigs && Array.isArray(model.rink_bounds) ? model.rink_bounds : model.bounds;
  if (Array.isArray(declared) && declared.length >= 4 && declared.slice(0,4).every(value => numeric(value) !== null)) {
    if (includeRigs || !geometry.length || Array.isArray(model.rink_bounds)) points.push([declared[0],declared[1]],[declared[2],declared[3]]);
  }
  if (includeRigs) asArray(model.rigs).forEach(rig => {
    [rig.position_xyz,rig.pivot_xyz,rig.current_optical_center_xyz].forEach(point => { const xy=modelXY(point,model); if(xy) points.push(xy); });
  });
  if (!points.length) return null;
  const result=[Infinity,Infinity,-Infinity,-Infinity];
  points.forEach(point=>{result[0]=Math.min(result[0],point[0]);result[1]=Math.min(result[1],point[1]);result[2]=Math.max(result[2],point[0]);result[3]=Math.max(result[3],point[1]);});
  if (result[0] === result[2] || result[1] === result[3]) return null;
  return result;
}
function modelUnitLabel(model) {return model.unit==='conditional_scene_units'?'场景单位（米制未验证）':display(model.unit);}
function modelCoordinateText(point,model) {
  if (!Array.isArray(point)) return '位置未知';
  return '(' + point.map(value => numeric(value) === null ? '?' : numberText(value,2)).join(', ') + ')';
}
function modelEmptyMarkup(title,message) {
  return '<div class="map-empty"><strong>' + escapeHtml(title) + '</strong><span>' + escapeHtml(message) + '</span></div>';
}
function renderModelDock(force = false) {
  const resource=resources.model;
  if (!resource.data) {
    $('#model-status').className='badge '+(resource.error?'bad':'');
    $('#model-status').textContent=resource.error?'模型接口未连接':'正在读取';
    $('#model-notice').className='model-notice'+(resource.error?' error':'');
    $('#model-notice').textContent=resource.error || '正在读取真实模型数据，不生成示意几何。';
    const empty=modelEmptyMarkup('尚未取得模型',resource.error?'请连接 /api/model；未知几何和相机位置保持为空。':'等待后端登记实体与相机位置。');
    $('#model-overview-map').innerHTML=empty; $('#model-rink-map').innerHTML=empty;
    $('#model-rig-legend').innerHTML='<p class="compact-note">尚未取得三机位记录。</p>';
    return;
  }
  const model=resource.data;
  const clock=model.shot_clock, exposure=model.selected_exposure;
  if(clock&&exposure){
    modelSelectedPts=exposure.native_pts;
    $('#model-frame-controls').innerHTML='<label>源时间 <span id="model-frame-time">'+Number(exposure.source_seconds).toFixed(3)+' 秒 · PTS '+exposure.native_pts+'</span><input id="model-native-pts" type="range" min="'+clock.first_pts+'" max="'+clock.last_pts+'" step="'+clock.step_pts+'" value="'+exposure.native_pts+'" aria-label="选择S01原生曝光查看相机光心与方向"></label><button class="link-button" data-action="projection-browser">查看此帧原片与当前网格</button> · <button class="link-button" data-action="model-parameters">编辑当前参数</button>';
  }else $('#model-frame-controls').innerHTML='';
  const revision=display(model.revision);
  const drawVersion=display(model.version || model.parameter_sha256 || revision);
  const paths=asArray(model.paths),ads=asArray(model.ads),rigs=asArray(model.rigs);
  $('#model-status').className='badge '+(resource.error?'warn':tone(model.status));
  $('#model-status').textContent=resource.error?'保留上次模型':model.parameter_sha256?'当前模型候选；整镜头待验收':label(model.status || '未声明状态');
  $('#model-unit').textContent=modelUnitLabel(model)+' · '+modelAxes(model).horizontal.toUpperCase()+'/'+modelAxes(model).vertical.toUpperCase()+(asArray(model.landmarks).length?' · ＋'+asArray(model.landmarks).length+' 固定材料点':'');
  $('#model-element-count').textContent=paths.length+' 几何 / '+ads.filter(ad=>asArray(ad.footprint).some(point=>modelXY(point,model))).length+' 有位置广告';
  $('#model-revision').textContent='模型 '+display(model.model_id)+' · r'+revision+' / 参数 r'+display(model.parameter_revision);
  $('#model-revision').title='模型 '+display(model.model_id)+' · 修订 '+revision;
  $('#model-updated').textContent=(resource.error?'上次成功 ':'同步 ')+shortTime(resource.at);
  const source=model.current_model_source || model.source;
  const sourceName=typeof source==='object' && source!==null ? source.label || source.name || source.id || source.path || noteText(source) : noteText(source);
  const warnings=asArray(model.warnings).map(noteText).filter(Boolean);
  const missingNotice=asArray(model.missing_ad_ids).length?'有编号但当前包缺失几何：'+asArray(model.missing_ad_ids).join('、')+'。 ':'';
  const initial=paths.some(item=>item.initial_geometry_only) || model.initial_geometry_only || String(model.status).includes('initial');
  $('#model-notice').className='model-notice'+(resource.error?' error':'');
  $('#model-notice').innerHTML=(resource.error?'<strong>连接异常，显示上次成功模型。</strong> ':'')+escapeHtml(missingNotice)+(model.parameter_sha256?'参数 r'+display(model.parameter_revision)+'。 ':'')+(initial?'当前候选使用实际参数状态，尚未通过整镜头真实验收。 ':'')+(warnings.length?escapeHtml(warnings.join('；')):'实体定义、来源与未确认语义可逐项检查。')+' <button class="link-button" data-action="model-source">检查来源</button> <button class="link-button" data-action="model-parameters">编辑当前参数</button>';
  $('#model-rig-legend').innerHTML=rigs.length?rigs.map(rig=>'<button class="model-rig-row" data-action="model-entity" data-entity-type="rig" data-id="'+escapeHtml(modelItemId(rig))+'"><span><span class="rig-legend-name">'+escapeHtml(modelItemLabel(rig))+'</span><span class="rig-legend-status">'+escapeHtml(rig.position_xyz?label(rig.status || '未声明状态'):'位置未知')+'</span></span><span class="rig-legend-coordinate">'+escapeHtml(modelCoordinateText(rig.position_xyz,model))+'</span></button>').join(''):'<p class="compact-note">API尚未返回机位记录；不摆放示意相机。</p>';
  const missing=asArray(model.missing_ad_ids).map(String);
  const adIndex=new Map(ads.map(ad=>[modelItemId(ad),{type:'ad',id:modelItemId(ad),label:modelItemLabel(ad),missing:missing.includes(modelItemId(ad))||!asArray(ad.footprint).some(point=>modelXY(point,model))}]));
  missing.forEach(id=>{if(!adIndex.has(id))adIndex.set(id,{type:'missing-ad',id,label:id,missing:true});});
  $('#model-ad-buttons').innerHTML=[...adIndex.values()].sort((a,b)=>a.id.localeCompare(b.id,'en',{numeric:true})).map(ad=>'<button class="model-ad-button'+(ad.missing?' missing':'')+'" data-action="model-entity" data-entity-type="'+ad.type+'" data-id="'+escapeHtml(ad.id)+'" aria-label="'+escapeHtml(ad.label+(ad.missing?'：有编号但当前包缺失几何':''))+'">'+escapeHtml(ad.label)+'</button>').join('');
  $('#model-missing-ads').textContent=missing.length?'有编号但当前包缺失几何：'+missing.join('、'):'当前API没有登记缺失广告编号。';
  const fullBounds=modelBounds(model,true),rinkOnlyBounds=modelBounds(model,false);
  const mapWidth=$('#model-overview-map').getBoundingClientRect().width;
  const mapsHeight=$('.model-maps').getBoundingClientRect().height;
  const fullScale=fullBounds?Math.min(Math.max(1,mapWidth-62)/(fullBounds[2]-fullBounds[0]),Math.max(1,mapsHeight-82)/(fullBounds[3]-fullBounds[1])):0;
  const needsInset=!!(fullBounds&&rinkOnlyBounds&&((rinkOnlyBounds[2]-rinkOnlyBounds[0])*fullScale<200||(rinkOnlyBounds[3]-rinkOnlyBounds[1])*fullScale<100));
  $('.model-maps').classList.toggle('single-map',!needsInset);
  if (force || modelDrawRevision!==drawVersion || !$('#model-overview-map svg') || needsInset&&!$('#model-rink-map svg')) {
    modelDrawRevision=drawVersion;
    drawModelMap($('#model-overview-map'),model,true);
    if(needsInset)drawModelMap($('#model-rink-map'),model,false);
  }
}
function modelSvgPath(points,model,project,closed) {
  let path='',hasPoint=false,validCount=0,invalidCount=0;
  asArray(points).forEach(point=>{
    const xy=modelXY(point,model);
    if (!xy) {hasPoint=false;invalidCount++;return;}
    const screen=project(xy); path+=(hasPoint?' L ':' M ')+screen[0].toFixed(2)+' '+screen[1].toFixed(2); hasPoint=true;validCount++;
  });
  if (closed && validCount>2 && invalidCount===0) path+=' Z';
  return path.trim();
}
function modelAnchor(points,model,project) {
  const valid=asArray(points).map(point=>modelXY(point,model)).filter(Boolean);
  if(!valid.length)return null;
  const point=valid[Math.floor(valid.length/2)];
  return project(point);
}
function drawModelMap(host,model,includeRigs) {
  const width=Math.floor(host.getBoundingClientRect().width),height=Math.floor(host.getBoundingClientRect().height);
  const bounds=modelBounds(model,includeRigs);
  if(width<20 || height<20)return;
  if(!bounds){host.innerHTML=modelEmptyMarkup('缺少可绘制的真实坐标','未提供二维范围或有效实体几何，不生成默认冰场。');return;}
  const padding=includeRigs?31:30;
  const spanX=bounds[2]-bounds[0],spanY=bounds[3]-bounds[1];
  const scale=Math.min(Math.max(1,width-padding*2)/spanX,Math.max(1,height-padding*2)/spanY);
  const contentWidth=spanX*scale,contentHeight=spanY*scale;
  const offsetX=(width-contentWidth)/2,offsetY=(height-contentHeight)/2;
  const axes=modelAxes(model);
  const project=point=>[offsetX+(point[0]-bounds[0])*scale,axes.yPositive==='down'?offsetY+(point[1]-bounds[1])*scale:height-offsetY-(point[1]-bounds[1])*scale];
  const occupied=[],pathLabels=[],adLabels=[];
  let geometry='',rigMarks='';
  const attr=(type,item)=>' data-action="model-entity" data-entity-type="'+type+'" data-id="'+escapeHtml(modelItemId(item))+'"';
  [...asArray(model.paths)].sort((a,b)=>(a.kind==='grid'?-1:0)-(b.kind==='grid'?-1:0)).forEach(item=>{
    const path=modelSvgPath(item.points,model,project,item.closed);
    if(!path)return;
    const kind=String(item.kind||'').toLowerCase();
    const board=kind.includes('board')||kind.includes('boundary')||kind.includes('rink_perimeter');
    const circle=kind.includes('circle')||kind.includes('target')||kind.includes('ring');
    const grid=kind==='grid';
    const selected=modelSelectedKey==='path:'+modelItemId(item);
    geometry+='<g class="entity-mark"'+attr('path',item)+' role="button" tabindex="0" aria-label="'+escapeHtml(modelItemLabel(item))+'"><path class="'+(grid?'grid-line':board?'rink-board':circle?'circle-line':'ice-line')+(item.initial_geometry_only||item.constraint_semantics_unassigned?' unconfirmed-geometry':'')+(selected?' entity-selected':'')+'" d="'+path+'"/><path class="entity-hit" d="'+path+'"/></g>';
    const anchor=modelAnchor(item.points,model,project);
    if(anchor && !grid && !kind.includes('advertisement') && modelItemLabel(item).length<=5 && !pathLabels.some(entry=>modelItemLabel(entry.item)===modelItemLabel(item)))pathLabels.push({type:'path',item,anchor});
  });
  asArray(model.landmarks).forEach(item=>{
    const xy=modelXY(item.position_world,model);if(!xy)return;
    const p=project(xy),selected=modelSelectedKey==='landmark:'+modelItemId(item);
    geometry+='<g class="entity-mark"'+attr('landmark',item)+' role="button" tabindex="0" aria-label="'+escapeHtml(modelItemLabel(item)+' 固定材料点')+'"><circle class="landmark-hit" cx="'+p[0]+'" cy="'+p[1]+'" r="8"/><path class="landmark-position'+(selected?' entity-selected':'')+'" d="M '+(p[0]-3)+' '+p[1]+' h 6 M '+p[0]+' '+(p[1]-3)+' v 6"/><title>'+escapeHtml(item.id+'：'+item.position_semantics)+'</title></g>';
    if(selected)pathLabels.push({type:'landmark',item,anchor:p});
  });
  asArray(model.ads).forEach(item=>{
    const path=modelSvgPath(item.footprint,model,project,item.closed===true);
    if(!path)return;
    geometry+='<g class="entity-mark"'+attr('ad',item)+' role="button" tabindex="0" aria-label="'+escapeHtml(modelItemLabel(item))+'"><path class="ad-footprint'+(modelSelectedKey==='ad:'+modelItemId(item)?' entity-selected':'')+'" d="'+path+'"/><path class="entity-hit" d="'+path+'"/></g>';
    const valid=asArray(item.footprint).map(point=>modelXY(point,model)).filter(Boolean);
    if(valid.length){const center=[valid.reduce((sum,point)=>sum+point[0],0)/valid.length,valid.reduce((sum,point)=>sum+point[1],0)/valid.length];adLabels.push({type:'ad',item,anchor:project(center),world:center});}
  });
  const labelBox=(text,x,y)=>{const w=Math.min(width-12,Math.max(18,Array.from(text).reduce((sum,c)=>sum+(c.charCodeAt(0)>255?11:6),0)+7));return{x:Math.max(4,Math.min(width-w-4,x)),y:Math.max(4,Math.min(height-18,y-9)),w,h:15};};
  const overlaps=box=>occupied.some(other=>box.x<other.x+other.w+3&&box.x+box.w+3>other.x&&box.y<other.y+other.h+2&&box.y+box.h+2>other.y);
  const placeLabel=(entry,preferred,cssClass)=>{
    const text=modelItemLabel(entry.item);
    let box=labelBox(text,preferred[0],preferred[1]);
    if(overlaps(box)){
      let found=false;
      for(let radius=13;radius<=Math.max(width,height)&&!found;radius+=13){
        for(const delta of [[0,-radius],[0,radius],[radius,0],[-radius,0],[radius,-radius],[-radius,radius],[radius,radius],[-radius,-radius]]){
          const candidate=labelBox(text,preferred[0]+delta[0],preferred[1]+delta[1]);
          if(!overlaps(candidate)){box=candidate;found=true;break;}
        }
      }
    }
    occupied.push(box);
    const labelX=box.x+3,labelY=box.y+11;
    const leader=Math.hypot(labelX-entry.anchor[0],labelY-entry.anchor[1])>13?'<path class="label-leader" d="M '+entry.anchor[0].toFixed(2)+' '+entry.anchor[1].toFixed(2)+' L '+labelX.toFixed(2)+' '+(labelY-3).toFixed(2)+'"/>':'';
    return leader+'<g class="entity-label '+cssClass+'"'+attr(entry.type,entry.item)+' role="button" tabindex="0" aria-label="'+escapeHtml(text)+'"><rect x="'+box.x.toFixed(1)+'" y="'+box.y.toFixed(1)+'" width="'+box.w+'" height="'+box.h+'" fill="transparent"/><text x="'+labelX.toFixed(1)+'" y="'+labelY.toFixed(1)+'">'+escapeHtml(text)+'</text></g>';
  };
  if(includeRigs)asArray(model.rigs).forEach(rig=>{
    const xy=modelXY(rig.position_xyz,model),pivot=modelXY(rig.pivot_xyz,model),current=modelXY(rig.current_optical_center_xyz,model);
    if(xy){const point=project(xy);rigMarks+='<g class="entity-mark"'+attr('rig',rig)+' role="button" tabindex="0" aria-label="'+escapeHtml(modelItemLabel(rig)+(rig.position_kind==='current_selected_exposure_rotation_derived_optical_center'?' 当前姿态光心':' 历史初值光心'))+'"><circle class="rig-position" cx="'+point[0]+'" cy="'+point[1]+'" r="5"/></g>';rigMarks+=placeLabel({type:'rig',item:rig,anchor:point},[point[0]+8,point[1]-8],'rig-label');}
    if(pivot){const point=project(pivot);rigMarks+='<g'+attr('rig',rig)+' role="button" tabindex="0"><path class="rig-pivot" d="M '+point[0]+' '+(point[1]-5)+' l 5 5 l -5 5 l -5 -5 Z"/><title>'+escapeHtml(modelItemLabel(rig)+' 固定pivot')+'</title></g>';}
    if(current){const point=project(current);rigMarks+='<g'+attr('rig',rig)+' role="button" tabindex="0"><circle class="rig-position" cx="'+point[0]+'" cy="'+point[1]+'" r="3"/><title>'+escapeHtml(modelItemLabel(rig)+' 当前光心')+'</title></g>';}
  });
  const rinkBounds=modelBounds(model,false);
  const rinkCenter=rinkBounds?[(rinkBounds[0]+rinkBounds[2])/2,(rinkBounds[1]+rinkBounds[3])/2]:[(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2];
  const sides={top:[],bottom:[],left:[],right:[]};
  adLabels.forEach(entry=>{
    const dx=(entry.world[0]-rinkCenter[0])/(rinkBounds?(rinkBounds[2]-rinkBounds[0])/2:spanX/2);
    const dy=(entry.world[1]-rinkCenter[1])/(rinkBounds?(rinkBounds[3]-rinkBounds[1])/2:spanY/2);
    const side=Math.abs(dx)>Math.abs(dy)?(dx>0?'right':'left'):(dy>0?(axes.yPositive==='down'?'bottom':'top'):(axes.yPositive==='down'?'top':'bottom'));
    sides[side].push(entry);
  });
  let labels='';
  Object.entries(sides).forEach(([side,entries])=>{
    const vertical=side==='left'||side==='right';
    entries.sort((a,b)=>a.anchor[vertical?1:0]-b.anchor[vertical?1:0]);
    entries.forEach((entry,index)=>{
      const preferred=vertical?[side==='left'?4:width-Math.max(25,modelItemLabel(entry.item).length*7)-8,offsetY+(index+.5)*contentHeight/Math.max(1,entries.length)]:[offsetX+(index+.5)*contentWidth/Math.max(1,entries.length),side==='top'?Math.max(13,offsetY-16):Math.min(height-12,height-offsetY+24)];
      labels+=placeLabel(entry,preferred,'ad-label');
    });
  });
  pathLabels.forEach(entry=>{labels+=placeLabel(entry,[entry.anchor[0]+3,entry.anchor[1]-3],'path-label');});
  const unit=modelUnitLabel(model);
  const dimensions='Δ'+axes.horizontal+' '+numberText(spanX,2)+' · Δ'+axes.vertical+' '+numberText(spanY,2)+' '+unit;
  const axis='<path class="axis-line" d="M 10 '+(height-17)+' h 15 M 10 '+(height-17)+' v -15"/><text class="axis-label" x="28" y="'+(height-13)+'">'+escapeHtml(axes.horizontal)+'</text><text class="axis-label" x="6" y="'+(height-35)+'">'+escapeHtml(axes.vertical)+'</text><text class="axis-label" x="'+(width-7)+'" y="'+(height-5)+'" text-anchor="end">'+escapeHtml(dimensions)+'</text>';
  host.innerHTML='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+width+' '+height+'" role="img" aria-label="'+escapeHtml((includeRigs?'完整空间与全部已知相机位置':'冰场完整标线围挡广告')+'，单位 '+unit)+'"><title>'+escapeHtml(model.model_id||'当前模型')+'</title><desc>全部导入几何始终显示。未确认或仅初始几何用虚线表示。相机光心与固定pivot分别标记。</desc>'+geometry+labels+rigMarks+axis+'</svg>';
}
function modelMetadataMarkup(item) {
  const omitted=new Set(['points','footprint','evidence']);
  const pairs=Object.entries(item).filter(([key])=>!omitted.has(key));
  return '<dl class="model-parameter-list">'+pairs.map(([key,value])=>'<dt>'+escapeHtml(key)+'</dt><dd>'+escapeHtml(noteText(value)||'—')+'</dd>').join('')+'</dl>';
}
function openModelEntity(type,id) {
  const model=resources.model.data;
  if(!model)return;
  const entry=type==='missing-ad' && asArray(model.missing_ad_ids).map(String).includes(String(id)) ? {type,item:{id,label:id,status:'当前包缺失几何',geometry:null,source:'GET /api/model.missing_ad_ids',notes:'保留编号；没有几何，不在俯视图中摆放推测位置。'}} : modelEntries(model).find(entry=>entry.type===type&&modelItemId(entry.item)===String(id));
  if(!entry){toast('当前模型修订中没有这个实体。',true);return;}
  const metadata=asArray(model.entities).find(entity=>modelItemId(entity)===String(id));
  const item={...(metadata||{}),...entry.item};
  modelSelectedKey=type+':'+id;
  renderModelDock(true);
  $('#model-selection').textContent=modelItemLabel(item)+' · '+type+' · 模型修订 '+display(model.revision);
  const initial=item.initial_geometry_only||String(item.status).includes('initial')||type==='rig';
  const unassigned=item.constraint_semantics_unassigned;
  const constraints=asArray(item.constraint_ids || item.related_constraints).map(value=>typeof value==='object'&&value!==null?value.id:value).filter(Boolean);
  let content='<div class="badge-row"><span class="badge outline">'+escapeHtml(type==='ad'||type==='missing-ad'?'围挡广告':type==='rig'?'摄像机':type==='path'?'冰场标线 / 边界':type==='landmark'?'固定材料点':'模型实体')+'</span>'+badge(item.status||model.status||'未声明状态')+(initial?'<span class="badge warn">当前候选 · 整镜头待验收</span>':'')+(unassigned?'<span class="badge warn">约束语义尚未分配</span>':'')+'</div><p class="drawer-intro">模型 '+escapeHtml(display(model.model_id))+' · 修订 '+escapeHtml(display(model.revision))+' · 坐标单位 '+escapeHtml(display(model.unit))+'</p>';
  if(type==='rig')content+='<div class="model-source-warning">'+(item.position_kind==='current_selected_exposure_rotation_derived_optical_center'?'position_xyz 是当前参数在所选原生PTS推导的光心，随姿态和固定杆臂移动；pivot_xyz 是同一安装的固定转动枢轴。':'position_xyz 是明确参考PTS的历史光心初值，固定安装尚待估计，不能当作pivot。')+' 机位显示不表示后续镜头已进入求解。</div>';
  if(initial||unassigned)content+='<p class="model-selected-geometry">'+(initial?'当前实体的初值来源保留供追查，正在显示实际参数候选，尚未通过整镜头验收。 ':'')+(unassigned?'此实体的显示几何尚未分配为新的物理约束。':'')+'</p>';
  if(type==='ad'&&item.actual_template_binding)content+='<section class="drawer-section"><h3>当前实际模板绑定</h3>'+badge(item.template_status)+'<p class="compact-note">'+item.actual_template_binding.source_bound_curve_count+' / '+item.actual_template_binding.geometry_curve_count+' 条当前几何曲线声明了源模板。此关系不替代逐源像素实体核实，素材文件存在也不表示已经用于当前投影。</p>'+rawDetails(item.actual_template_binding,'查看源模板与对象契约关系')+'</section>';
  content+='<section class="drawer-section"><h3>定义、参数与关联字段</h3>'+modelMetadataMarkup(item)+'</section><section class="drawer-section"><h3>来源证据</h3>'+evidenceMarkup(item.evidence)+'</section>';
  if(type==='landmark')content+='<section class="drawer-section"><p>'+escapeHtml(item.position_semantics)+'</p><button class="button button-secondary" data-action="observation-browser" data-query="'+escapeHtml(item.source_query || item.id)+'">核对该材料的原片取点</button></section>';
  if(type==='path')content+='<section class="drawer-section"><button class="button button-secondary" data-action="observation-browser" data-curve-id="'+escapeHtml(id)+'">查看对应原片取点</button></section>';
  if(constraints.length)content+='<section class="drawer-section"><h3>关联约束台账</h3><div class="badge-row">'+constraints.map(constraintId=>'<button class="button button-secondary button-small" data-action="constraint-detail" data-id="'+escapeHtml(constraintId)+'">'+escapeHtml(constraintId)+'</button>').join('')+'</div></section>';
  content+='<section class="drawer-section">'+rawDetails(item,'查看完整实体与原始坐标')+'</section>';
  openDrawer('MODEL ENTITY',modelItemLabel(item),content,{type:'model',id,entityType:type,revision:model.revision});
}

let modelRegistryFilter='';
function modelRegistryEntries(model) {
  const entries=modelEntries(model);
  const geometryIds=new Set(entries.filter(entry=>entry.type!=='entity').map(entry=>modelItemId(entry.item)));
  return [...entries.filter(entry=>entry.type!=='entity'||!geometryIds.has(modelItemId(entry.item))),...asArray(model.missing_ad_ids).map(id=>({type:'missing-ad',item:{id,label:id,status:'当前包缺失几何'}}))];
}
function openModelRegistry() {
  const model=resources.model.data;if(!model)return;
  modelRegistryFilter='';
  openDrawer('MODEL ENTITY INDEX','全部几何与实体','<p class="drawer-intro">全部导入几何始终绘制。网格线保留点击和查询，不在图面堆放重复文字。缺失编号没有推测位置。</p><input id="model-registry-search" class="model-registry-search" type="search" placeholder="搜索标线、广告、机位、实体ID或类型" aria-label="搜索全部模型实体"><p id="model-registry-count" class="compact-note" style="margin-bottom:10px"></p><div id="model-registry-list" class="model-registry-list"></div>',{type:'model-registry',revision:model.revision});
  renderModelRegistryList();
}
function renderModelRegistryList() {
  const model=resources.model.data,host=$('#model-registry-list');if(!model||!host)return;
  const entries=modelRegistryEntries(model),query=modelRegistryFilter.toLocaleLowerCase().trim();
  const filtered=entries.filter(entry=>!query||[entry.item.id,entry.item.label,entry.item.kind,entry.type].join(' ').toLocaleLowerCase().includes(query));
  $('#model-registry-count').textContent='显示 '+filtered.length+' / '+entries.length+' 项 · 模型修订 '+display(model.revision);
  host.innerHTML=filtered.map(entry=>'<button class="model-registry-row" data-action="model-entity" data-entity-type="'+entry.type+'" data-id="'+escapeHtml(modelItemId(entry.item))+'"><span>'+escapeHtml(modelItemId(entry.item))+'<small style="display:block;margin-top:3px">'+escapeHtml(modelItemLabel(entry.item))+'</small></span><small>'+escapeHtml(entry.item.kind||entry.type)+'</small></button>').join('')||(query?'<p class="compact-note">没有匹配的实体。</p>':'<p class="compact-note">当前没有实体。</p>');
}

function openModelSource() {
  const model=resources.model.data;
  if(!model)return;
  const metadata=Object.fromEntries(Object.entries(model).filter(([key])=>!['paths','ads','entities'].includes(key)));
  openDrawer('MODEL PROVENANCE','当前模型来源与空间定义','<div class="model-source-warning">初始化参照、显示几何与新物理拟合验证状态分别登记。成功 demo 不等于新模型已经拟合通过；其他成功参照各自保留。</div>'+modelMetadataMarkup(metadata)+'<section class="drawer-section">'+rawDetails(metadata)+'</section>',{type:'model-source',revision:model.revision});
}
function scheduleModelResize(){if(modelResizePending)return;modelResizePending=true;requestAnimationFrame(()=>{modelResizePending=false;if(resources.model.data)renderModelDock(true);});}

function openDrawer(eyebrow, title, content, state) {
  if(state?.type!=='projection')disposeProjection();
  if ($('#detail-drawer').hidden) drawerOpener = document.activeElement;
  drawerState = state;
  $('#drawer-eyebrow').textContent = eyebrow;
  $('#drawer-title').textContent = title;
  $('#drawer-content').innerHTML = content;
  $('#detail-drawer').hidden = false;
  $('#drawer-backdrop').hidden = false;
  document.body.style.overflow = 'hidden';
  $('#drawer-content').scrollTop = 0;
  $('#close-drawer').focus();
}
function closeDrawer() {
  if ($('#confirm-dialog').open) return;
  disposeProjection();
  $('#detail-drawer').hidden = true;
  $('#drawer-backdrop').hidden = true;
  document.body.style.overflow = '';
  drawerState = null;
  if (drawerOpener?.isConnected) drawerOpener.focus();
}
function toast(message, error = false) {
  const item = document.createElement('div');
  item.className = 'toast' + (error ? ' error' : '');
  item.textContent = message;
  $('#toast-region').append(item);
  setTimeout(() => item.remove(), error ? 8000 : 4500);
}
async function copyText(text) {
  try {
    if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
    else {
      const field = document.createElement('textarea');
      field.value = text; field.style.position = 'fixed'; field.style.opacity = '0';
      document.body.append(field); field.select();
      const result = document.execCommand('copy'); field.remove();
      if (!result) throw new Error('复制未完成');
    }
    toast('已复制，可粘贴到本地工具。');
  } catch { toast('剪贴板不可用，请选中路径后手动复制。', true); }
}


function sourceExclusionText(record) {
  const reasons={current_not_admitted:'当前未获拟合资格',current_not_fit_role:'当前不作为拟合源',current_model_retired:'对应旧模型已退出',current_withheld:'保留作检查，未参与求解',no_exact_current_geometry_crosswalk:'缺少精确实体映射',corner_semantics_not_fixed_entity:'角点尚未确认为固定实体',selected_path_outside_current_allowed_paths:'不在当前允许的有限曲线上',query_descriptor_missing:'缺少当前Query定义',intersection_support_replacement:'被新固定交点替代，保留原支持',unverified_current_source_entity_association:'原片未核清实体对应，退出当前拟合',unresolved_current_finite_glyph_boundary:'字形有限轮廓未核清，退出当前拟合',Ad5_specific_source_association_requires_finite_branch_review:'Ad5具体源点对应错误或有限轮廓未核清',explicit_selection_or_complete_W_group_withdrawal:'主动撤出或整W组随成员退出'};
  return asArray(record.baseline_exclusion_reasons).map(item=>reasons[item.code] || item.code).join('；');
}

let projectionBrowser=null;
function disposeProjection(){
  if(projectionBrowser?.controller)projectionBrowser.controller.abort();
  if(projectionBrowser?.url)URL.revokeObjectURL(projectionBrowser.url);
  projectionBrowser=null;
}
async function openProjection(){
  disposeProjection();
  projectionBrowser={nativePts:modelSelectedPts ?? 256256,loading:false,url:null,identity:null,error:null,sourceOnly:false,zoom:1};
  openDrawer('SAME MODEL PROJECTION','原片与当前网格','',{type:'projection'});
  await loadProjection();
}
async function loadProjection(){
  const p=projectionBrowser;if(!p || p.loading)return;
  p.loading=true;p.error=null;p.controller=new AbortController();
  modelSelectedPts=p.nativePts;loadResource('model',true);renderProjection();
  const timer=setTimeout(()=>p.controller.abort(),45000);
  try{
    const response=await fetch('/api/model/projection?native_pts='+p.nativePts,{signal:p.controller.signal,cache:'no-store'});
    if(!response.ok){let body=await response.json();throw Error(body.message || '网格预览读取失败');}
    const identity=JSON.parse(response.headers.get('X-Projection-Identity') || 'null');
    if(!identity || identity.native_pts!==p.nativePts || !identity.all_model_paths_enabled)throw Error('预览状态身份或全路径声明不一致');
    const blob=await response.blob();if(projectionBrowser!==p)return;
    if(p.url)URL.revokeObjectURL(p.url);
    p.url=URL.createObjectURL(blob);p.identity=identity;p.sourceOnly=false;
  }catch(error){if(projectionBrowser===p)p.error=error.name==='AbortError'?'预览请求已取消或超时，已有叠图保留。':error.message;}
  finally{clearTimeout(timer);p.loading=false;if(projectionBrowser===p)renderProjection();}
}
function projectionNotice(){
  const d=projectionBrowser?.identity,m=resources.model.data;
  if(!d)return '';
  return d.model_id===m?.model_id && d.model_revision===m?.revision && d.parameter_sha256===m?.parameter_sha256?'叠图与右侧总览参数一致。':'叠图保留下面的冻结参数；右侧总览已有其他参数状态，点击刷新叠图核对。';
}
function updateProjectionNotice(){
  const host=$('#projection-live-state');if(host)host.textContent=projectionNotice();
}
function renderProjection(){
  const p=projectionBrowser;if(!p || drawerState?.type!=='projection')return;
  const d=p.identity,m=resources.model.data;
  const matches=d && m && d.model_id===m.model_id && d.model_revision===m.revision && d.parameter_sha256===m.parameter_sha256;
  const status=d?'<p id="projection-live-state" class="inline-note '+(matches?'':'warn')+'">'+projectionNotice()+'</p><dl class="key-values"><dt>此图源时间</dt><dd>'+d.source_seconds.toFixed(3)+' 秒 · PTS '+d.native_pts+'</dd><dt>此图模型 / 参数</dt><dd>模型 r'+d.model_revision+' · 参数 r'+d.parameter_revision+'</dd><dt>参数 SHA256</dt><dd class="observation-id mono">'+escapeHtml(d.parameter_sha256)+'</dd><dt>全开路径 / 可见</dt><dd>'+d.all_path_count+' / '+d.visible_path_count+'</dd><dt>完整读出投影证书</dt><dd>'+escapeHtml(d.certificate_policy_id || '旧版完整读出检查')+'<br>子区间补证 '+(d.partition_certificate_point_count ?? 0)+' 点；仍未证 '+(d.formal_projection_status_counts?.['6'] ?? 0)+' 点</dd></dl>':'';
  const controls='<form id="projection-form" class="observation-query"><label class="form-field">S01 原生 PTS<input name="native_pts" type="number" min="256256" max="575488" step="256" value="'+p.nativePts+'" required'+(p.loading?' disabled':'')+'></label><button class="button button-primary" type="submit"'+(p.loading?' disabled':'')+'>'+(p.loading?'正在投影当前完整模型…':'刷新此帧当前网格')+'</button></form>';
  const image=d&&p.url?'<div class="toolbar"><button class="button button-secondary button-small" data-action="projection-toggle">'+(p.sourceOnly?'查看全开网格':'只看同帧原片')+'</button><button class="button button-secondary button-small" data-action="projection-zoom">放大 '+p.zoom+'×</button><button class="button button-secondary button-small" data-action="observation-browser" data-native-pts="'+d.native_pts+'">核对并撤出具体源点</button></div><div class="observation-image"><div class="observation-source-inner" style="width:'+(p.zoom*100)+'%"><img src="'+(p.sourceOnly?'/api/observations/frame?native_pts='+d.native_pts:p.url)+'" alt="S01原生曝光 '+d.native_pts+'；冻结参数 '+d.parameter_revision+'；'+(p.sourceOnly?'原片':'全部标线、广告边界和编号')+'"></div></div>':'';
  const old=$('.observation-image'),scroll=old?[old.scrollLeft,old.scrollTop]:[0,0];
  $('#drawer-content').innerHTML='<p class="compact-note">按需读取同一完整物理模型；所有标线、广告边界、编号与网格全开。单帧用于找错，S01整段仍未验收。人物没有遮挡蒙版；无有效传感器行根的投影不补造。</p>'+controls+(p.error?'<p role="alert">'+escapeHtml(p.error)+'</p>':'')+status+image+(d?rawDetails(d,'此帧投影状态与完整身份'):'');
  const next=$('.observation-image');if(next){next.scrollLeft=scroll[0];next.scrollTop=scroll[1];}
}

let pointBrowser=null;
async function openObservations(options={}) {
  const pts=options.nativePts ?? modelSelectedPts ?? 256256;
  modelSelectedPts=pts;
  pointBrowser={nativePts:pts,query:options.query || '',curveId:options.curveId || null,offset:0,data:null,selectedId:null,zoom:1};
  openDrawer('SOURCE OBSERVATIONS','原片取点与优化选择','<p>正在读取当前稳定源ID…</p>',{type:'observations'});
  await Promise.all([loadResource('model',true),loadObservations()]);
}
async function loadObservations() {
  if(!pointBrowser)return;
  const p=pointBrowser,query=new URLSearchParams({native_pts:p.nativePts,offset:p.offset,limit:64});
  if(p.query)query.set('query',p.query);
  if(p.curveId)query.set('curve_id',p.curveId);
  try {
    const data=await requestJson('/api/observations?'+query.toString());
    if(pointBrowser!==p)return;
    p.data=data;
    if(!data.records.some(r=>r.observation_id===p.selectedId))p.selectedId=data.records[0]?.observation_id || null;
    renderObservations();
  } catch(error) {if(drawerState?.type==='observations')$('#drawer-content').innerHTML='<p role="alert">'+escapeHtml(error.message)+'</p>';}
}
function renderObservations() {
  const p=pointBrowser,d=p?.data;if(!d || drawerState?.type!=='observations')return;
  const oldImage=$('.observation-image');const oldScroll=oldImage?[oldImage.scrollLeft,oldImage.scrollTop]:[0,0];
  const r=d.records.find(v=>v.observation_id===p.selectedId);
  const points=d.records.map(v=>'<g data-action="observation-select" data-id="'+escapeHtml(v.observation_id)+'" class="observation-mark '+(v.observation_id===p.selectedId?'selected ':'')+(v.effective_enabled?'active':'withdrawn')+'"><circle cx="'+v.uv_native_pixels[0]+'" cy="'+v.uv_native_pixels[1]+'" r="'+(v.observation_id===p.selectedId?7:4)+'"/><title>'+escapeHtml(v.observation_id)+'</title></g>').join('');
  const selection=r?'<section class="drawer-section"><h3>所选原始观测</h3><p class="mono observation-id">'+escapeHtml(r.observation_id)+'</p><dl class="key-values"><dt>原生坐标 (u,v)</dt><dd>'+r.uv_native_pixels.map(v=>numberText(v,3)).join(', ')+'</dd><dt>当前实体</dt><dd>'+escapeHtml(r.curve?.label || r.curve?.id || '未获当前实体绑定')+'</dd><dt>优化状态</dt><dd>'+escapeHtml(r.effective_enabled?'当前准备内有效':r.explicit_disabled?'主动退出优化':r.withdrawn_group_id?'因整组相关W退出':'当前数据原本未获拟合资格')+'</dd><dt>基础退出原因</dt><dd>'+escapeHtml(sourceExclusionText(r) || '无基础退出')+'</dd><dt>原统计标准差</dt><dd>'+escapeHtml(r.statistical_noise_kind==='original_correlated_group_W'?'按整组相关W使用；逐点σ不适用':r.sigma_px===null?'未知（不补造）':display(r.sigma_px)+' px')+'</dd><dt>工程像素预算 E</dt><dd>'+escapeHtml(display(r.engineering_budget_px))+' px</dd><dt>相关源W组</dt><dd>'+escapeHtml(r.statistical_group_id || '无当前整组W')+(r.statistical_group_size?' · '+r.statistical_group_size+'个源点':'')+'</dd></dl>'+(r.statistical_group_id?'<p class="inline-note warn">撤此点会使整个原W组退出优化；原始像素与矩阵保留，不裁剪W或改成独立噪声。</p>':'')+(r.edit?'<p>编辑原因：'+escapeHtml(r.edit.reason)+'</p>':'')+'<form id="observation-selection-form"><label class="form-field">具体修改原因<textarea name="reason" rows="2" required placeholder="例如：该像素实际在K上，却被绑定为N；或人物遮挡无法确认材料边界"></textarea></label><p id="observation-error" role="alert"></p><button class="button button-primary" type="submit"'+(!r.baseline_active?' disabled':'')+'>'+escapeHtml(!r.baseline_active?'基础实体或用途未核实，当前不参与优化':r.explicit_disabled?'恢复此前主动退出':'将此源点退出优化')+'</button></form>'+rawDetails(r,'完整源坐标、用途和依赖字段')+'</section>':'<p>当前查询无原始取点。</p>';
  $('#drawer-content').innerHTML='<p class="compact-note">当前数据绑定模型 r'+d.model_revision+' · 取点选择 r'+d.revision+' · 原UV/PTS/ID保留。</p>'+((resources.model.data?.model_id!==d.model_id || resources.model.data?.revision!==d.model_revision)?'<p class="inline-note warn">此份当前数据与右侧可见模型结构尚未对齐，不能作为该模型实际拟合进度。</p>':'')+'<form id="observation-query-form" class="observation-query"><label class="form-field">原生 PTS<input name="native_pts" type="number" min="256256" max="575488" step="256" value="'+p.nativePts+'" required></label><label class="form-field">源ID / 实体检索<input name="query" value="'+escapeHtml(p.query)+'" placeholder="Ad7、C或原始源ID"></label><button class="button button-secondary" type="submit">读取原片与取点</button></form>'+(p.curveId?'<p>当前曲线筛选：<code>'+escapeHtml(p.curveId)+'</code> <button class="subtle-button" data-action="observation-unfilter">取消实体筛选</button></p>':'')+'<p class="compact-note">'+(p.nativePts/12800).toFixed(3)+' 秒 · 当前页 '+d.records.length+' / '+d.total+' 点。红圈为当前准备可用，灰圈为退出优化，橙圈为所选点；实际求解范围以当前运行阶段为准。</p><button class="button button-secondary button-small" data-action="observation-zoom">图像放大 '+p.zoom+'×</button><div class="observation-image"><div class="observation-source-inner" style="width:'+(100*p.zoom)+'%"><img src="/api/observations/frame?native_pts='+p.nativePts+'" alt="官方原片原生PTS '+p.nativePts+'，无网格修饰"><svg viewBox="0 0 1920 1080" aria-label="当前页原始源坐标">'+points+'</svg></div></div><div class="observation-pagination"><button class="button button-secondary button-small" data-action="observation-page" data-direction="-1"'+(p.offset===0?' disabled':'')+'>上一页</button><span>'+Math.min(p.offset+1,d.total)+'–'+Math.min(p.offset+d.records.length,d.total)+' / '+d.total+'</span><button class="button button-secondary button-small" data-action="observation-page" data-direction="1"'+(p.offset+d.records.length>=d.total?' disabled':'')+'>下一页</button></div>'+selection+'<details><summary>展开当前页全部源点记录</summary><div class="table-wrap"><table class="data-table"><thead><tr><th>原实体 / 像素</th><th>优化用途</th></tr></thead><tbody>'+d.records.map(v=>'<tr><td><button class="subtle-button" data-action="observation-select" data-id="'+escapeHtml(v.observation_id)+'">'+escapeHtml(v.curve?.label || v.curve?.id || v.observation_id)+'</button><small class="compact-note" style="display:block">'+v.uv_native_pixels.map(x=>numberText(x,2)).join(', ')+'</small></td><td>'+escapeHtml(v.effective_enabled?'有效':v.explicit_disabled?'主动退出':v.withdrawn_group_id?'整W组退出':'原本未纳入')+'</td></tr>').join('')+'</tbody></table></div></details><section class="drawer-section"><h3>本次可恢复选择</h3><p>主动退出 '+d.explicit_disabled_count+' 个稳定源ID；因整W组连带退出 '+d.collateral_raw_count+' 个原始点。恢复编辑不能越过既有实体/材料资格。</p><p class="compact-note">选择修订保存到当前数据；数值运行须按该修订重建实际残差图后才生效。</p>'+sourcePath(d.prepared)+'</section>';
  const image=$('.observation-image');if(image){image.scrollLeft=oldScroll[0];image.scrollTop=oldScroll[1];}
}
async function saveObservationSelection(form) {
  const p=pointBrowser,r=p?.data?.records.find(v=>v.observation_id===p.selectedId);
  if(!r || mutationPending)return;
  const reason=form.elements.reason.value.trim();if(!reason){$('#observation-error').textContent='请填写实际修改原因。';return;}
  try {
    mutationPending=true;form.querySelector('button').disabled=true;
    await requestJson('/api/observations/selection',{method:'PUT',body:JSON.stringify({expected_revision:p.data.revision,observation_ids:[r.observation_id],enabled:r.explicit_disabled,reason})});
    await loadObservations();toast('已保存可恢复的取点选择修订；原始像素未改。');
  } catch(error){$('#observation-error').textContent=error.message;}
  finally {mutationPending=false;const button=$('#observation-selection-form button');if(button)button.disabled=!r.baseline_active;}
}

let parameterEditor = null;
async function openParameters(preferredId) {
  try {
    const data=await requestJson('/api/model/parameters');
    parameterEditor={data,blockId:preferredId || data.blocks.find(b=>b.id==='scene/board')?.id || data.blocks[0]?.id};
    openDrawer('CURRENT PHYSICAL PARAMETERS','当前模型参数','', {type:'parameters'});
    renderParameterEditor();
  } catch(error) {toast(error.message,true);}
}
function renderParameterEditor() {
  if(!parameterEditor)return;
  const {data,blockId}=parameterEditor, block=data.blocks.find(b=>b.id===blockId);
  if(!block)return;
  const options=data.blocks.map(b=>'<option value="'+escapeHtml(b.id)+'"'+(b.id===blockId?' selected':'')+'>'+escapeHtml(b.id)+' · '+escapeHtml(b.owner.kind)+' / '+b.values.length+'值</option>').join('');
  const compact=block.values.length<=20;
  const fields=compact?block.components.map((c,i)=>'<label class="form-field">'+escapeHtml(c.name)+' <span class="compact-note">'+escapeHtml(c.unit)+'</span><input class="input" type="number" step="any" name="p'+i+'" value="'+escapeHtml(block.values[i])+'" required></label>').join(''):'<label class="form-field">完整参数数组（'+block.values.length+'个，顺序按当前结构）<textarea class="input" id="parameter-json" rows="12">'+escapeHtml(JSON.stringify(block.values))+'</textarea></label><details><summary>逐分量名称与单位</summary><pre class="code-block">'+escapeHtml(JSON.stringify(block.components,null,2))+'</pre></details>';
  $('#drawer-content').innerHTML='<p class="compact-note">同一模型 r'+data.model_revision+' · 参数 r'+data.revision+' · '+escapeHtml(data.state_id)+'</p><p>'+escapeHtml(data.notes)+'</p><label class="form-field">物理参数块<select class="input" id="parameter-block">'+options+'</select></label><p class="compact-note">所有者：'+escapeHtml(block.owner.kind+' / '+block.owner.id)+' · '+escapeHtml(block.mode)+'</p><form id="parameter-form">'+fields+'<p id="parameter-error" role="alert"></p><button class="button button-primary" type="submit"'+(block.mode!=='free'?' disabled':'')+'>保存参数修订</button></form>';
}
async function saveParameters(form) {
  if(!parameterEditor || mutationPending)return;
  const {data,blockId}=parameterEditor, block=data.blocks.find(b=>b.id===blockId);
  const errorBox=$('#parameter-error');
  try {
    const values=block.values.length<=20?block.values.map((v,i)=>{const text=form.elements['p'+i].value.trim();if(!text)throw new Error('参数不能为空');return Number(text);}):JSON.parse($('#parameter-json').value);
    if(!Array.isArray(values)||values.length!==block.values.length||values.some(v=>typeof v!=='number'||!Number.isFinite(v)))throw new Error('数组长度必须与当前参数块相同，所有值须为有限数字');
    mutationPending=true;form.querySelector('button[type="submit"]').disabled=true;
    const result=await requestJson('/api/model/parameters',{method:'PUT',body:JSON.stringify({block_id:blockId,values,expected_revision:data.revision})});
    await loadResource('model',true);await openParameters(blockId);toast('已保存参数 r'+result.revision+'；俯视图已使用同一新状态。');
  } catch(error){if(errorBox)errorBox.textContent=error.message;toast(error.message,true);}
  finally{mutationPending=false;const save=$('#parameter-form button[type="submit"]');if(save)save.disabled=block.mode!=='free';}
}

document.addEventListener('click', async event => {
  const navigation = event.target.closest('.nav-item[data-view]');
  if (navigation) { switchView(navigation.dataset.view); return; }
  const control = event.target.closest('[data-action]');
  if (!control) return;
  const action = control.dataset.action;
  const id = control.dataset.id;
  if (action === 'navigate') switchView(control.dataset.view);
  else if (action === 'retry-all') await refreshAll();
  else if (action === 'retry') await loadResource(control.dataset.resource);
  else if (action === 'copy') await copyText(control.dataset.value || '');
  else if (action === 'module-detail') openModule(id);
  else if (action === 'module-type') {closeDrawer();contractQuery='';dataMode='contracts';switchView('data');const index=contractTypes(resources.contracts.data).findIndex(type=>type.name===control.dataset.type);const card=document.getElementById('contract-type-'+index);if(card){card.open=true;card.scrollIntoView({behavior:'smooth',block:'start'});}}
  else if (action === 'constraint-detail') openConstraint(id);
  else if (action === 'constraint-edit') openConstraint(id, true);
  else if (action === 'constraint-confirm') openConfirm(id);
  else if (action === 'toggle-constraint') await toggleConstraint(id);
  else if (action === 'dataset-detail') openDataset(id);
  else if (action === 'model-entity') openModelEntity(control.dataset.entityType,id);
  else if (action === 'model-source') openModelSource();
  else if (action === 'projection-browser') await openProjection();
  else if (action === 'projection-toggle') {if(projectionBrowser){projectionBrowser.sourceOnly=!projectionBrowser.sourceOnly;renderProjection();}}
  else if (action === 'projection-zoom') {if(projectionBrowser){projectionBrowser.zoom=projectionBrowser.zoom===3?1:projectionBrowser.zoom+1;renderProjection();}}
  else if (action === 'observation-browser') await openObservations({curveId:control.dataset.curveId,nativePts:control.dataset.nativePts?Number(control.dataset.nativePts):undefined,query:control.dataset.query || ''});
  else if (action === 'observation-zoom') {pointBrowser.zoom=pointBrowser.zoom===3?1:pointBrowser.zoom+1;renderObservations();}
  else if (action === 'observation-select') {if(pointBrowser){pointBrowser.selectedId=id;renderObservations();}}
  else if (action === 'observation-unfilter') {pointBrowser.curveId=null;pointBrowser.offset=0;await loadObservations();}
  else if (action === 'observation-page') {pointBrowser.offset=Math.max(0,pointBrowser.offset+Number(control.dataset.direction)*64);await loadObservations();}
  else if (action === 'model-parameters') await openParameters();
  else if (action === 'model-registry') openModelRegistry();
  else if (action === 'data-mode') {dataMode=control.dataset.mode;history.replaceState(null,'',dataMode==='contracts'?'#data/contracts':'#data');renderHeader();renderData();}
  else if (action === 'contract-jump') {const card=document.getElementById('contract-type-'+control.dataset.index);if(card){card.open=true;card.scrollIntoView({behavior:'smooth',block:'start'});}}
});
document.addEventListener('input', event => {
  const control = event.target;
  if (control.id === 'constraint-search') { constraintFilters.query = control.value; renderConstraintTable(); }
  if (control.id === 'dataset-search') { dataQuery = control.value; renderDatasetTable(); }
  if (control.id === 'model-registry-search') { modelRegistryFilter=control.value;renderModelRegistryList(); }
  if (control.id === 'model-native-pts') {const c=resources.model.data.shot_clock;$('#model-frame-time').textContent=(Number(control.value)*c.time_base_num/c.time_base_den).toFixed(3)+' 秒 · PTS '+control.value;}
  if (control.id === 'contract-search') {contractQuery=control.value;renderContractTypes();}
  if (control.closest('#constraint-form')) { const dirty = $('#constraint-dirty'); if (dirty) dirty.textContent = '有尚未保存的修改'; }
});
document.addEventListener('change', event => {
  const control = event.target;
  if (control.id === 'model-native-pts') {modelSelectedPts=Number(control.value);const query=new URLSearchParams(location.search);query.set('native_pts',modelSelectedPts);history.replaceState(null,'',location.pathname+'?'+query.toString()+location.hash);loadResource('model',true);}
  if (control.id === 'parameter-block' && parameterEditor) {parameterEditor.blockId=control.value;renderParameterEditor();}
  if (control.id === 'filter-approval') constraintFilters.approval = control.value;
  if (control.id === 'filter-origin') constraintFilters.origin = control.value;
  if (control.id === 'filter-enabled') constraintFilters.enabled = control.value;
  if (['filter-approval','filter-origin','filter-enabled'].includes(control.id)) renderConstraintTable();
  if (control.id === 'confirm-explicit') $('#confirm-submit').disabled = !control.checked || mutationPending;
  if (control.id === 'pause-polling') {
    pollPaused = control.checked;
    $('#polling-footer').textContent = pollPaused ? '运行状态自动刷新已暂停' : '运行状态每 3 秒同步';
    if (!pollPaused) loadResource('state', true);
  }
});
document.addEventListener('submit', event => {
  if (event.target.id === 'projection-form') {event.preventDefault();projectionBrowser.nativePts=Number(event.target.elements.native_pts.value);loadProjection();}
  if (event.target.id === 'constraint-form') { event.preventDefault(); saveConstraint(event.target); }
  if (event.target.id === 'observation-query-form') {event.preventDefault();pointBrowser.nativePts=Number(event.target.elements.native_pts.value);pointBrowser.query=event.target.elements.query.value.trim();pointBrowser.offset=0;modelSelectedPts=pointBrowser.nativePts;loadResource('model',true);loadObservations();}
  if (event.target.id === 'observation-selection-form') {event.preventDefault();saveObservationSelection(event.target);}
  if (event.target.id === 'parameter-form') { event.preventDefault(); saveParameters(event.target); }
});
document.addEventListener('keydown', event => {
  if ((event.key === 'Enter' || event.key === ' ') && event.target.closest('svg [data-action="model-entity"]')) { event.preventDefault(); const entity=event.target.closest('[data-action]'); openModelEntity(entity.dataset.entityType,entity.dataset.id); return; }
  if ($('#confirm-dialog').open) return;
  if (!$('#detail-drawer').hidden) {
    if (event.key === 'Escape') { event.preventDefault(); closeDrawer(); }
    if (event.key === 'Tab') {
      const focusable = $$('button:not([disabled]),a[href],input:not([disabled]),textarea:not([disabled]),select:not([disabled]),summary,[tabindex="0"]', $('#detail-drawer')).filter(element => element.getClientRects().length);
      const first = focusable[0], last = focusable[focusable.length-1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
  }
});
$('#refresh-all').addEventListener('click', refreshAll);
$('#close-drawer').addEventListener('click', closeDrawer);
$('#drawer-backdrop').addEventListener('click', closeDrawer);
$('#confirm-submit').addEventListener('click', submitConfirmation);
$('#confirm-dialog').addEventListener('close', () => { confirmTarget = null; });
window.addEventListener('hashchange', () => switchView(location.hash.slice(1)));
window.addEventListener('online', () => { if (!refreshPending) refreshAll(); });
setInterval(() => { if (!pollPaused && !document.hidden && !refreshPending) Promise.all([loadResource('state', true),loadResource('model', true),loadResource('constraints',true)]); }, 3000);
document.addEventListener('visibilitychange', () => { if (!document.hidden && !pollPaused && !refreshPending) Promise.all([loadResource('state', true),loadResource('model', true),loadResource('constraints',true)]); });
const modelResizeObserver=new ResizeObserver(scheduleModelResize);
modelResizeObserver.observe($('#model-overview-map'));
modelResizeObserver.observe($('#model-rink-map'));
switchView(location.hash.slice(1) || 'architecture');
renderModelDock();
refreshAll().then(()=>{const query=new URLSearchParams(location.search);if(query.get('panel')==='projection')openProjection();if(query.get('panel')==='parameters')openParameters();if(query.get('panel')==='observations')openObservations({query:query.get('point_query')||''});});
