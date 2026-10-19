/* Read-only catalog UI. Source PTS is displayed independently from browser seeks. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const labels = {supported:'有证据支持', multiple_candidates:'多候选', unknown:'未知', not_applicable:'不适用', unprocessed:'未处理', candidate:'候选', unresolved:'未解决', uncertain:'未确定', arena:'本场馆', rink:'冰场画面', non_target:'非目标素材', interview:'采访', ceremony:'颁奖', black:'黑屏', replay:'回放', possible_replay:'可能回放', live:'直播', unassessed:'未检查', unresolved_content:'画面类型未决', non_target_material:'非目标素材', source_start:'来源起点', source_end:'来源终点', supported_within_session:'来源内有支持', human_confirmed:'人工已确认', user_confirmed:'用户已确认', supported_within_broadcast_scope:'播出来源内有支持', unresolved_transition:'转场范围未决', not_replay:'非回放', exact:'精确帧', hard_cut:'硬切', transition:'过渡', continuous:'连续', supported_candidate:'有支持的候选'};
  const colors = {supported:'#258277', multiple_candidates:'#b78226', unknown:'#8296a1', not_applicable:'#bcbdc7', unprocessed:'#d07b73'};
  Object.assign(labels, {supported_in_broadcast_scope:'播出来源内有支持', human_confirmed_position:'人工确认的位置', off_rink_arena_kiss_and_cry:'场内等分区', remote_interview:'远程访谈', off_rink_remote_interview:'远程访谈', graphic_transition:'图形转场', dissolve_transition:'叠化转场', unresolved_dissolve:'叠化范围未决', machine_supported_cut:'程序支持的切镜', unresolved_short_event:'短暂变化未决', sampled_same_image_replay_supported:'采样画面支持回放关联', probable_visual_transition_evidence:'图形转场支持的可能回放'});
  Object.assign(labels, {static_context_region_matches:'静态场景多区域匹配', end_wall_windows:'端墙窗格', grandstand_structure:'看台结构', rink_surface_context:'冰面／围挡', backdrop:'后幕', chair_contour:'椅子轮廓', side_flowers:'侧边花饰', foreground_flowers:'前景花篮'});
  Object.assign(labels, {shot:'镜头区间', boundary_unresolved:'边界未决区间', unresolved_range:'范围未决', interval_start:'区间起点', interval_end:'区间终点', explicit_interval_start:'显式区间起点', explicit_interval_end:'显式区间终点', after_unresolved_interval:'未决范围之后', dissolve:'叠化', graphic_wipe:'图形擦拭'});
  Object.assign(labels, {arena_skating:'场内滑行', arena_skating_replay_transition:'场内滑行／回放转场线索', arena_warmup:'场内热身', mixed_kiss_and_cry_then_high_overview:'等分区后切高位全景', non_target_interview_composite:'非目标访谈合成画面', target_venue:'目标场馆', target_venue_replay_candidate:'目标场馆回放候选', target_venue_warmup:'目标场馆热身', target_venue_with_transition_and_kiss_and_cry:'场馆／转场／等分区', target_venue_with_transition_graphics_and_replay:'场馆／图形转场／回放线索'});
  Object.assign(labels, {human_confirmed_position_group:'用户确认的同一机位'});
  labels.multiple_candidates='候选待确认';
  const keys = {source_id:'来源', shot_id:'镜头编号', status:'状态', reason:'原因', start_seconds:'起点（秒）', end_seconds:'终点（秒）', start_pts:'起点 PTS', end_pts:'终点 PTS', frame_count:'帧数', time_base:'time_base', physical_camera_id:'物理机位', installation_id:'场次安装版本', appearance_group_id:'外观候选组', candidate_installation_ids:'候选安装', source_path:'视频轨路径', boundary_count:'边界数量', shot_count:'镜头数量', duration_seconds:'时长（秒）', missing_media:'媒体缺失', coverage:'覆盖检查', summary:'概况', limitations:'限制', automated_checks:'程序检查', independent_visual_review:'独立画面检查', visual_review:'真实画面检查', all_eight_sources_processed:'八场全量处理', all_sources_processed:'全部来源已处理', gaps:'缺口', overlaps:'重叠', per_source:'各来源', sources:'各来源', passed:'通过', checks:'检查', findings:'发现', source_pts:'源 PTS', source_time_seconds:'源时间（秒）', image_path:'证据图', evidence:'证据', source_reference:'来源引用', method:'方法', interpretation:'含义', support:'支持', opposition:'反对', kind:'类型'};
  Object.assign(keys, {interval_type:'区间类型', boundary_status:'边界状态', candidate_points:'区间内候选变化点', transition_kinds:'转场线索', review_evidence_paths:'局部复核图', uncertainty_reason:'未决原因', capture_session_id:'实际拍摄场次', broadcast_session_id:'播出场次', broadcast_scope_source_id:'播出来源范围'});
  Object.assign(keys, {decision_id:'判断编号', user_statement:'用户原话', label:'名称', installation_ids:'关联安装版本', scope:'适用范围'});
  Object.assign(keys, {processing_scope:'本场冰场处理范围', target_seconds:'入选冰场时长（秒）', ignored_seconds:'已忽略无关时长（秒）', pending_seconds:'场景待判定时长（秒）', target_frame_count:'入选冰场帧数', ignored_frame_count:'已忽略帧数', pending_frame_count:'场景待判定帧数', scene_screening_complete:'场景筛选完成', migration:'历史记录迁移', target_intervals:'入选冰场区间', ignored_intervals:'已忽略区间', pending_intervals:'场景待判定区间', end_pts_exclusive:'结束 PTS（不含）'});
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const label = value => labels[value] || (value == null || value === '' ? '未记录' : value);
  const asArray = value => Array.isArray(value) ? value : value == null ? [] : [value];
  const fileURL = path => '/api/file?path=' + encodeURIComponent(path);
  const time = value => {
    if (value == null || !Number.isFinite(Number(value))) return '—';
    const seconds = Math.max(0, Number(value));
    return `${Math.floor(seconds/3600).toString().padStart(2,'0')}:${Math.floor(seconds%3600/60).toString().padStart(2,'0')}:${(seconds%60).toFixed(3).padStart(6,'0')}`;
  };
  const duration = value => `${(Number(value || 0)/3600).toFixed(2)} 小时`;
  const badge = status => `<span class="badge ${escape(status)}">${escape(label(status))}</span>`;
  const pretty = value => `<pre>${escape(JSON.stringify(value, null, 2))}</pre>`;
  const api = async (path, params={}) => {
    const response = await fetch(path + '?' + new URLSearchParams(params));
    if (!response.ok) throw new Error((await response.json()).error || response.statusText);
    return response.json();
  };
  const filters = {'source-filter':'source_id','status-filter':'status','camera-filter':'physical_camera_id','installation-filter':'installation_id','appearance-filter':'appearance_group_id','special-filter':'special','content-filter':'content_kind','query-filter':'q'};
  let meta, sourceMap = new Map(), referenceMap = new Map(), installationMap = new Map(), offset = 0, total = 0, pageSize = 48, timeline = [], timelineDuration = 0, activeShot, queryVersion = 0, timelineVersion = 0;

  const scopeColors={target:'#e5efeb',ignored:'#d3d9dd',pending:'#fff0c6'};
  const activeInstallations=()=> (meta.installations||[]).filter(item=>(item.shot_ids||[]).length);
  const activeCameras=()=> {
    const installations=activeInstallations();
    return (meta.physical_cameras||[]).filter(camera=>installations.some(item=>item.physical_camera_id===camera.physical_camera_id||(camera.installation_ids||[]).includes(item.installation_id)));
  };
  function sourceTimeBase(source) {
    const base=source?.track?.time_base||source?.time_base;
    if(typeof base==='number')return base>0?base:null;
    if(base&&typeof base==='object') {
      const numerator=Number(base.numerator??base.num),denominator=Number(base.denominator??base.den);
      return numerator>0&&denominator>0?numerator/denominator:null;
    }
    const parts=String(base||'').split('/').map(Number);
    return parts.length===2&&parts[0]>0&&parts[1]>0?parts[0]/parts[1]:null;
  }
  function scopeIntervals(source,kind) {
    const base=sourceTimeBase(source);
    if(!base)return [];
    return (source?.processing_scope?.[kind+'_intervals']||[]).map(interval=>({
      start_seconds:Number(Array.isArray(interval)?interval[0]:interval.start_pts)*base,
      end_seconds:Number(Array.isArray(interval)?interval[1]:interval.end_pts_exclusive??interval.end_pts)*base
    })).filter(interval=>Number.isFinite(interval.start_seconds)&&Number.isFinite(interval.end_seconds)&&interval.end_seconds>interval.start_seconds);
  }
  function sceneAt(sourceId,seconds) {
    const source=sourceMap.get(sourceId);
    if(!source)return 'pending';
    const end=Number(source.duration_seconds??source.track?.duration_seconds);
    if(seconds<0||(Number.isFinite(end)&&seconds>=end))return 'outside';
    for(const kind of ['ignored','pending','target'])if(scopeIntervals(source,kind).some(interval=>interval.start_seconds<=seconds&&seconds<interval.end_seconds))return kind;
    return 'pending';
  }
  function scopeMessage(kind,seconds) {
    const prefix=`${time(seconds)} · `;
    if(kind==='ignored')return prefix+'该时段已忽略：画面未纳入本场冰场范围，不显示镜头。';
    if(kind==='pending')return prefix+'该时段的场景仍待判定，尚未进入分镜，也未计作已忽略。';
    if(kind==='outside')return prefix+'已超出该来源的有效时间。';
    return prefix+'该时段已纳入本场冰场范围，当前尚未列出对应镜头。';
  }
  function showScopeMessage(kind,seconds) {
    if($('shot-dialog').open)$('shot-dialog').close();
    $('jump-message').textContent=scopeMessage(kind,seconds);
  }

  function error(message) { $('load-error').hidden = false; $('load-error').textContent = message; }
  function options(id, items, idKey, nameKey='label') {
    for (const item of items || []) {
      const value = typeof item === 'string' ? item : item[idKey];
      if (!value) continue;
      const option = document.createElement('option'); option.value = value;
      option.textContent = typeof item === 'string' ? label(item) : `${item[nameKey] || value}${item[nameKey] && item[nameKey] !== value ? ' · '+value : ''}`;
      $(id).append(option);
    }
  }

  function params() {
    const p = {offset, limit:pageSize};
    for (const [id,key] of Object.entries(filters)) if ($(id).value) p[key] = $(id).value;
    return p;
  }

  function setTab(name) {
    for (const button of document.querySelectorAll('[data-tab]')) button.classList.toggle('active',button.dataset.tab===name);
    for (const tab of ['shots','cameras','validation']) $(tab+'-tab').hidden = tab !== name;
    if (name === 'shots') drawTimeline();
  }

  function imagePath(observation) { return typeof observation === 'string' ? observation : observation?.image_path || observation?.path || observation?.frame_path; }
  function firstImage(shot) { return (shot.observations || []).find(item => imagePath(item)) || shot.representative_observation; }
  function thumbnail(observation) {
    const path = imagePath(observation);
    return path ? `<img class="thumb" src="${fileURL(path)}" loading="lazy" alt="原始来源画面">` : '<div class="thumb no-image">缺少代表画面 · 可打开原视频</div>';
  }
  const candidateNames = {main_long_side:'长边主机位', low_end:'低位端部', scoreboard_end_R:'记分牌端 R', long_side_or_end_unresolved:'长边或端部（尚未区分）', high_overview:'高位斜俯全景', human_A_B:'人工确认 A/B 的同一机位', kiss_and_cry:'等分区固定构图'};
  const candidateIDs = c => [...new Set(c.candidate_installation_ids||[])];
  const candidateName = id => candidateNames[id.split(':').at(-1)] || installationMap.get(id)?.label || '未命名候选';
  function classificationBadges(c) {
    const count=candidateIDs(c).length;
    const candidates=count?`<span class="badge multiple_candidates">${count} 个候选，未确认</span>`:'';
    return c.status==='multiple_candidates' ? candidates||'<span class="badge unknown">候选信息缺失</span>' : badge(c.status||'unprocessed')+(c.status==='unknown'?candidates:'');
  }
  function candidateList(c) {
    const ids=candidateIDs(c);
    return ids.length?`<div class="candidate-names"><span>候选：</span>${ids.map(id=>`<span>${escape(candidateName(id))}</span>`).join('、')}</div>`:'';
  }
  function candidateEvidence(shot,id) {
    const entries=[];
    const add=(observation,match)=>entries.push({observation,match,reference:referenceMap.get(match.reference_id),referenceId:match.reference_id});
    for(const evidence of shot.classification?.supporting_evidence||[]) {
      if(!evidence||typeof evidence!=='object')continue;
      for(const match of evidence.matches||[]) {
        const ids=match.kind==='high_overview'?[`install:${shot.source_id}:high_overview`]:(match.candidate_group_ids||[]).map(group=>`candidate:${shot.source_id}:${group}`);
        if(ids.includes(id))add(evidence,match);
      }
      if(evidence.kind==='human_reference_retrieval' && evidence.candidate_installation_id===id) {
        for(const observation of evidence.matches_to_confirmed_observations||[])for(const match of observation.reference_matches||[])add(observation,match);
      }
      if(evidence.kind==='static_context_region_matches') {
        for(const observation of evidence.observations||[])if(observation.candidate_installation_id===id)for(const match of observation.matches||[])add(observation,match);
      }
    }
    // Keep the strongest recorded example of each reference, not unrelated group thumbnails.
    const unique=new Map();
    for(const entry of entries.sort((a,b)=>(b.match.inliers||0)-(a.match.inliers||0)))if(!unique.has(entry.referenceId))unique.set(entry.referenceId,entry);
    return [...unique.values()];
  }
  function fallbackReference(shot,id) {
    const kind=id.split(':').at(-1);
    if(kind==='human_A_B') {
      const item=meta.human_evidence?.cases?.find(c=>c.installation_id===id);
      if(item?.observations?.length)return {observation:item.observations[0]};
    }
    return [...referenceMap.values()].find(ref=>{
      const sameSource=(ref.source_id||ref.observation?.source_id)===shot.source_id;
      return sameSource&&(kind==='high_overview'?ref.kind==='high_overview':kind==='kiss_and_cry'?ref.reference_id.startsWith('context:'):(ref.legacy_candidate_group_ids||[]).includes(kind));
    });
  }
  function observationSeconds(obs,shot) {
    const direct=obs?.source_time_seconds??obs?.source_seconds;
    if(direct!=null)return direct;
    const saved=shot.observations?.find(o=>o.source_pts===obs?.source_pts);
    if(saved?.source_time_seconds!=null)return saved.source_time_seconds;
    const base=String(obs?.time_base||shot.time_base||'').split('/').map(Number);
    return obs?.source_pts!=null&&base.length===2&&base[1]>0?obs.source_pts*base[0]/base[1]:null;
  }
  function comparisonImage(obs,title,shot,sourceId) {
    const path=imagePath(obs), seconds=observationSeconds(obs,shot);
    const source=sourceMap.get(obs?.source_id||sourceId)?.title||obs?.source_id||sourceId||'来源未记录';
    return `<figure class="candidate-frame"><figcaption><strong>${escape(title)}</strong><span>${time(seconds)}</span></figcaption>${path?`<a href="${fileURL(path)}" target="_blank" rel="noopener"><img src="${fileURL(path)}" alt="${escape(title)}，${escape(source)}，${time(seconds)}"></a>`:'<div class="thumb no-image">未记录对应图片</div>'}<div class="candidate-source">${escape(source)}</div></figure>`;
  }
  function candidatesHTML(shot) {
    const c=shot.classification||{}, ids=candidateIDs(c);
    if(!ids.length)return c.status==='multiple_candidates'?'<section class="evidence-block"><h3>候选信息缺失</h3><p>这条记录标为候选待确认，但没有列出候选机位，当前无法比较。</p></section>':'';
    if(c.status==='supported')return '';
    const rows=ids.map(id=>({id,entries:candidateEvidence(shot,id)}));
    const shared=new Map();
    for(const row of rows)for(const entry of row.entries)if(entry.referenceId){if(!shared.has(entry.referenceId))shared.set(entry.referenceId,new Set());shared.get(entry.referenceId).add(row.id);}
    const sharedGroups=[...new Set([...shared.values()].filter(group=>group.size>1).map(group=>[...group].map(candidateName).join('、')))];
    const cards=rows.map((row,index)=>{
      const best=row.entries.find(e=>imagePath(e.reference?.observation))||row.entries[0];
      const ref=best?.reference||(!best?fallbackReference(shot,row.id):null), query=best?.observation;
      const regions=Object.entries(best?.match.regions||best?.match.region_inliers||{}).filter(([,count])=>count>0).map(([name])=>label(name));
      const sharedWith=best?.referenceId?[...(shared.get(best.referenceId)||[])].filter(id=>id!==row.id):[];
      const oldLabel=['main_long_side','low_end','scoreboard_end_R','long_side_or_end_unresolved'].includes(row.id.split(':').at(-1));
      const extras=row.entries.filter(entry=>entry!==best);
      return `<article class="candidate-card"><div class="candidate-heading"><span class="candidate-number">${index+1}</span><h4>${escape(candidateName(row.id))}</h4></div><p class="candidate-kind">${oldLabel?'旧候选标签，位置未确认':'当前镜头尚未确认属于此机位'}</p><div class="candidate-pair">${comparisonImage(query||firstImage(shot),query?'本镜头匹配帧':'本镜头示例',shot,shot.source_id)}${comparisonImage(ref?.observation,best?'匹配参照':'候选参照',shot,ref?.source_id)}</div><p class="candidate-basis">${best?`程序将本镜头与这张参照图关联。${regions.length?`记录的对应区域：${escape(regions.join('、'))}。`:''}`:'没有记录本镜头的具体匹配帧；参照图仅用于说明候选。'}${best&&!ref?' 参照图记录缺失。':''}</p>${sharedWith.length?`<p class="candidate-shared">此参照也被用于：${escape(sharedWith.map(candidateName).join('、'))}。</p>`:''}${extras.length?`<details><summary>其他匹配参照（${extras.length}）</summary>${extras.map(entry=>`<div class="candidate-pair">${comparisonImage(entry.observation,'本镜头匹配帧',shot,shot.source_id)}${comparisonImage(entry.reference?.observation,'匹配参照',shot,entry.reference?.source_id)}</div>`).join('')}</details>`:''}<button data-installation="${escape(row.id)}">查看这个候选的相关镜头</button><details class="candidate-record"><summary>候选编号与原始依据</summary><div class="id">${escape(row.id)}</div>${pretty(row.entries.map(entry=>({reference_id:entry.referenceId,source_pts:entry.observation.source_pts,regions:entry.match.regions||entry.match.region_inliers})))}</details></article>`;
    }).join('');
    return `<section class="candidate-section" aria-label="候选机位对照"><h3>尚未确定机位 · 现有记录列出 ${ids.length} 个候选</h3><p>下面直接对照本镜头与各候选的参照画面。点击图片可放大。</p>${sharedGroups.length?`<div class="candidate-notice"><strong>部分候选共用同一张参照图</strong><p>${sharedGroups.map(escape).join('；')}。这些是尚未分清的标签，不能据此认定为不同机位。</p></div>`:''}${c.status==='unknown'?'<p class="candidate-notice">这段仍标为“未知”。以下候选线索被保留，尚未形成机位归属。</p>':''}<div class="candidate-grid">${cards}</div></section>`;
  }
  function shotCard(shot) {
    const c = shot.classification || {}, source = sourceMap.get(shot.source_id), content = shot.content || {};
    return `<article class="shot-card" tabindex="0" role="button" data-shot="${escape(shot.shot_id)}" aria-label="打开 ${escape(shot.shot_id)}">${thumbnail(firstImage(shot))}<div class="card-body"><div class="source-label">${escape(source?.title || shot.source_id)}</div><div class="card-time timing">${time(shot.start_seconds)} → ${time(shot.end_seconds)}</div><div class="badges">${classificationBadges(c)}${shot.interval_type && shot.interval_type!=='shot'?badge(shot.interval_type):''}${content.kind ? badge(content.kind) : ''}${shot.duplicate_links?.length ? '<span class="badge">重复关联</span>' : ''}</div>${c.status!=='supported'?candidateList(c):''}<div class="reason">${escape(c.installation_id?candidateName(c.installation_id):c.reason || '未记录物理安装判断')}</div><div class="shot-id">${escape(shot.shot_id)}</div></div></article>`;
  }

  async function loadShots() {
    const version = ++queryVersion;
    try {
      const result = await api('/api/shots', params()); if (version !== queryVersion) return;
      total = result.total;
      $('results-count').textContent = `${total.toLocaleString()} 条入选冰场分段／镜头 · ${duration(result.duration_seconds)}`;
      $('page-label').textContent = total ? `${Math.floor(offset/pageSize)+1} / ${Math.ceil(total/pageSize)}` : '0 / 0';
      $('prev-page').disabled = offset === 0; $('next-page').disabled = offset + pageSize >= total;
      $('shot-grid').innerHTML = result.shots.length ? result.shots.map(shotCard).join('') : '<div class="empty">此筛选下没有镜头。</div>';
    } catch (err) { error('目录读取失败：'+err.message); }
  }

  async function loadTimeline() {
    const version = ++timelineVersion, sourceId = $('source-filter').value;
    $('timeline-hover').textContent='';$('jump-message').textContent='';
    if (!sourceId) { timeline = []; timelineDuration=0; $('timeline-title').textContent='选择一场查看源时间轴'; drawTimeline(); return; }
    timeline=[];timelineDuration=0;drawTimeline();
    const result = await api('/api/timeline',{source_id:sourceId}); if (version !== timelineVersion) return;
    timeline=result; timelineDuration = sourceMap.get(sourceId)?.duration_seconds || Math.max(0,...timeline.map(s=>s.end_seconds));
    $('timeline-title').textContent = `${sourceMap.get(sourceId)?.title || sourceId} · ${timeline.length.toLocaleString()} 条入选冰场分段／镜头 · 源时长 ${time(timelineDuration)}`;
    drawTimeline();
  }

  function drawTimeline() {
    const canvas=$('timeline'), rect=canvas.getBoundingClientRect(), ratio=window.devicePixelRatio||1;
    if (!rect.width) return;
    canvas.width=rect.width*ratio; canvas.height=70*ratio;
    const ctx=canvas.getContext('2d'); ctx.scale(ratio,ratio); const width=rect.width;
    ctx.fillStyle='#eef2f4'; ctx.fillRect(0,5,width,36);
    if (!timelineDuration) { ctx.fillStyle='#788e98';ctx.font='12px system-ui';ctx.fillText('选择来源，查看入选冰场镜头与场景筛选范围',12,28);return; }
    ctx.fillStyle=scopeColors.pending;ctx.fillRect(0,7,width,32);
    const source=sourceMap.get($('source-filter').value);
    for(const kind of ['target','ignored','pending'])for(const interval of scopeIntervals(source,kind)) {
      const x=Math.max(0,interval.start_seconds/timelineDuration*width),end=Math.min(width,interval.end_seconds/timelineDuration*width);
      if(end>x){ctx.fillStyle=scopeColors[kind];ctx.fillRect(x,7,end-x,32);}
    }
    for (const shot of timeline) {
      const x=shot.start_seconds/timelineDuration*width, end=shot.end_seconds/timelineDuration*width;
      ctx.fillStyle=colors[shot.status] || colors.unprocessed;ctx.fillRect(x,7,Math.max(.7,end-x),32);
      if (shot.boundary_issue) {ctx.fillStyle='#d29131';ctx.fillRect(x,3,Math.max(1,end-x),3);}
    }
    ctx.font='11px system-ui';ctx.fillStyle='#6c808a';
    for(let i=0;i<=6;i++) {const x=width*i/6;ctx.textAlign=i===0?'left':i===6?'right':'center';ctx.fillText(time(timelineDuration*i/6).slice(0,8),x,59);}
  }
  function timelineAt(event) {
    if (!timelineDuration) return null;
    const rect=$('timeline').getBoundingClientRect(), seconds=Math.min(timelineDuration-.000001, Math.max(0,(event.clientX-rect.left)/rect.width*timelineDuration));
    const kind=sceneAt($('source-filter').value,seconds);
    return {seconds,kind,shot:kind==='target'?timeline.find(shot=>shot.start_seconds<=seconds && shot.end_seconds>seconds):null};
  }
  function parseTime(value) {
    if(!value.trim())return NaN;
    const parts=value.trim().split(':').map(Number);
    if (parts.some(n=>!Number.isFinite(n)||n<0)||parts.length>3) return NaN;
    return parts.reduce((total,n)=>total*60+n,0);
  }

  function renderOverview() {
    const counts=meta.viewer.status_counts,scope=meta.processing_scope||{},summary=scope.summary||{};
    const stat=(value,title,detail='',kind='')=>`<div class="stat ${kind?'stat-'+kind:''}"><div class="stat-value">${escape(value)}</div><div class="stat-label">${escape(title)}</div>${detail?`<small>${escape(detail)}</small>`:''}</div>`;
    const scopeDuration=value=>value==null?'未记录':duration(value);
    const frameCount=value=>value==null?'':`${Number(value).toLocaleString()} 帧`;
    $('overview').innerHTML=stat(scopeDuration(summary.target_seconds),'入选本场冰场',frameCount(summary.target_frame_count),'target')+stat(scopeDuration(summary.ignored_seconds),'已忽略无关画面',frameCount(summary.ignored_frame_count),'ignored')+stat(scopeDuration(summary.pending_seconds),'场景待判定','尚未进入分镜；不计作已忽略','pending')+stat(Number(meta.viewer.shot_count||0).toLocaleString(),'当前冰场分段／镜头',`${meta.sources?.length||0} 场来源`)+stat((counts.supported?.shot_count||0).toLocaleString(),'当前镜头归属有支持',duration(counts.supported?.duration_seconds))+stat(((counts.unknown?.shot_count||0)+(counts.multiple_candidates?.shot_count||0)).toLocaleString(),'当前镜头机位尚未确定',duration((counts.unknown?.duration_seconds||0)+(counts.multiple_candidates?.duration_seconds||0)))+`<p class="scope-status ${scope.scene_screening_complete===true?'':'scope-incomplete'}">${scope.scene_screening_complete===true?'场景筛选已完成；机位归属状态另行展示。':'场景筛选尚未完成。待判定画面尚未进入分镜，也没有被当作无关画面忽略；当前镜头数只覆盖已经入选的冰场范围。'}</p>`;
  }

  function evidenceHTML(value) {
    if (value == null || (Array.isArray(value) && !value.length)) return '<p class="muted">未记录</p>';
    if (typeof value === 'string') return `<p class="muted">${escape(value)}</p>`;
    const references = new Map();
    const collect = item => {
      if (!item || typeof item !== 'object') return;
      if (item.reference_id && referenceMap.has(item.reference_id)) references.set(item.reference_id,referenceMap.get(item.reference_id));
      for (const nested of Object.values(item)) if (nested && typeof nested === 'object') collect(nested);
    };
    collect(value);
    const comparison = references.size ? `<details><summary>对照参考帧（${references.size}）</summary><p class="muted">这些帧参与了本版本分类，不能作为本版本的独立检查。参考帧时间属于其自身来源。</p><div class="observations">${Array.from(references.values()).map(ref=>`<div><p class="muted">${escape(ref.reference_id)} · ${escape(ref.observation?.source_id||ref.source_id||'')}<br>${escape(ref.visual_reason||ref.visual_basis||ref.new_usage||ref.use||'')}</p>${observationCard(ref.observation||{},false)}</div>`).join('')}</div></details>` : '';
    const items=asArray(value), hasStructured=items.some(item=>item && typeof item==='object');
    const summary=items.map(item=> {
      if(typeof item==='string') return `<li>${escape(item)}</li>`;
      if(!item || typeof item!=='object') return `<li>${escape(item??'未记录')}</li>`;
      const regions=new Set((item.matches||[]).flatMap(match=>Object.entries(match.regions||match.region_inliers||{}).filter(([,count])=>count>0).map(([name])=>label(name))));
      const description=[item.kind?label(item.kind):'',item.source_pts!=null?`源 PTS ${item.source_pts}`:'',item.matches?.length?`${item.matches.length} 项参考对应`:'',Array.from(regions).join('、'),item.reference_id||'',item.reason||item.interpretation||'',item.observations?.length?`${item.observations.length} 条观察证据`:''].filter(Boolean).join(' · ');
      return `<li>${escape(description||'已记录的证据，完整字段可展开查看')}</li>`;
    }).join('');
    const records=hasStructured?`<details><summary>完整证据字段与数值</summary>${items.map(item=>renderObject(item,1)).join('')}</details>`:'';
    return `<ul class="evidence-list">${summary}</ul>${comparison}${records}`;
  }
  function renderObject(value,depth=0) {
    if (value == null) return '<span class="muted">未记录</span>';
    if (typeof value === 'boolean') return value?'是':'否';
    if (typeof value !== 'object') {
      if (typeof value==='string' && /\.(jpg|jpeg|png|webp|mp4|json|html)$/i.test(value) && (value.includes('/')||value.includes('\\'))) return `<a class="artifact-link" href="${fileURL(value)}" target="_blank" rel="noopener">${escape(value)}</a>`;
      return escape(label(value));
    }
    if (depth>2) return pretty(value);
    if(Array.isArray(value)) return value.length ? `<ul class="evidence-list">${value.map(item=>'<li>'+renderObject(item,depth+1)+'</li>').join('')}</ul>` : '<span class="muted">无记录</span>';
    return `<dl class="kv">${Object.entries(value).map(([key,val])=>`<dt>${escape(keys[key]||key)}</dt><dd>${renderObject(val,depth+1)}</dd>`).join('')}</dl>`;
  }

  function installationCard(item) {
    return `<article class="camera-card installation-card">${thumbnail(item.representative_observations?.[0])}<div class="card-body"><h3>${escape(item.label||item.installation_id)}</h3><div class="id">${escape(item.installation_id)}</div><div class="source-label">${escape(sourceMap.get(item.source_id)?.title||item.source_id||'拍摄场次未确定')}</div><div class="status-line">${badge(item.status||'candidate')}</div><p class="muted">${(item.shot_ids||[]).length} 条直接镜头关联</p><button data-installation="${escape(item.installation_id)}">查看此版本支持／候选镜头</button><details><summary>支持与反对证据</summary><h3>支持</h3>${evidenceHTML(item.supporting_evidence)}<h3>反对／限制</h3>${evidenceHTML(item.opposing_evidence)}${item.reason?`<p class="muted">${escape(item.reason)}</p>`:''}${pretty(item)}</details></div></article>`;
  }

  function renderCameras() {
    const installations=activeInstallations(), cameras=activeCameras();
    const blocks=[], groupedInstallations=new Set();
    for(const camera of cameras) {
      const reps=camera.representative_observations||[], related=installations.filter(item=>item.physical_camera_id===camera.physical_camera_id || (camera.installation_ids||[]).includes(item.installation_id));
      related.forEach(item=>groupedInstallations.add(item.installation_id));
      blocks.push(`<article class="camera-card physical-camera-card">${thumbnail(reps[0]||related[0]?.representative_observations?.[0])}<div class="card-body"><h3>${escape(camera.label||camera.physical_camera_id)}</h3><div class="id">物理机位 · ${escape(camera.physical_camera_id)}</div>${badge(camera.identity_status||camera.status||'candidate')}<p class="reason">${escape(camera.reason||camera.description||'详见关联安装版本的证据')}</p><button data-camera="${escape(camera.physical_camera_id)}">查看全部关联镜头</button><details class="installation-versions"><summary>展开 ${related.length} 个场次安装版本</summary><div class="installation-list">${related.map(installationCard).join('')}</div></details><details><summary>归属证据</summary>${evidenceHTML(camera.supporting_evidence)}${evidenceHTML(camera.opposing_evidence)}</details></div></article>`);
    }
    const ungrouped=installations.filter(item=>!groupedInstallations.has(item.installation_id));
    if(ungrouped.length) blocks.push(`<details class="ungrouped-installations"><summary>未关联已列物理机位的安装候选（${ungrouped.length}）</summary><div class="camera-grid">${ungrouped.map(installationCard).join('')}</div></details>`);
    $('camera-grid').innerHTML=blocks.join('')||'<div class="empty">当前入选冰场镜头尚无关联机位记录。可在冰场镜头中查看尚未确定项。</div>';

  }

  let appearanceOffset = 0, appearanceTotal = 0;
  async function loadAppearances() {
    const result = await api('/api/appearance-groups',{q:$('appearance-search').value,source_id:$('appearance-source').value,offset:appearanceOffset,limit:24});
    appearanceTotal=result.total;
    $('appearance-count').textContent=`${appearanceTotal.toLocaleString()} 组 · ${Math.floor(appearanceOffset/24)+1} / ${Math.max(1,Math.ceil(appearanceTotal/24))}`;
    $('appearance-prev').disabled=appearanceOffset===0;$('appearance-next').disabled=appearanceOffset+24>=appearanceTotal;
    $('appearance-list').innerHTML=result.groups.map(group=>`<article class="camera-card">${thumbnail(group.representative_observations?.[0])}<div class="card-body"><div class="id">${escape(group.appearance_group_id)}</div><p class="muted">${(group.shot_ids||[]).length} 条镜头关联 · 外观检索线索</p><button data-appearance="${escape(group.appearance_group_id)}">查看关联镜头</button><details><summary>原始记录</summary>${renderObject(group)}</details></div></article>`).join('')||'<p class="muted">没有匹配的外观组</p>';
  }

  function observationCard(obs, showSeek=true) {
    const path=imagePath(obs), seconds=obs.source_time_seconds??obs.source_seconds??obs.time_seconds??obs.time, pts=obs.source_pts??obs.pts;
    return `<div class="observation"><figure>${path?`<a href="${fileURL(path)}" target="_blank" rel="noopener"><img src="${fileURL(path)}" loading="lazy" alt="源时间 ${time(seconds)} 的证据帧"></a>`:'<div class="thumb no-image">未提供此帧图片</div>'}<figcaption>${escape(obs.role||obs.label||'源画面')}<br>源时间 ${time(seconds)}<br>PTS ${escape(pts??'未记录')}${obs.time_base?` · time_base ${escape(typeof obs.time_base==='object'?JSON.stringify(obs.time_base):obs.time_base)}`:''}${(obs.image_sha256||obs.sha256)?`<br>图像 SHA-256 ${escape(obs.image_sha256||obs.sha256)}`:''}</figcaption></figure>${seconds!=null&&showSeek?`<button data-seek="${Number(seconds)}">原片跳转到此处</button>`:''}</div>`;
  }
  function boundaryHTML(boundary,title) {
    if(!boundary) return `<div class="boundary-card"><h3>${title}</h3><p>未记录边界证据</p></div>`;
    const observations=[...(boundary.observations||boundary.frames||boundary.neighbour_frames||[]),...asArray(boundary.before),...asArray(boundary.after),...asArray(boundary.previous_frame),...asArray(boundary.first_frame)];
    const rest={...boundary}; for(const key of ['observations','frames','neighbour_frames','before','after','previous_frame','first_frame']) delete rest[key];
    return `<div class="boundary-card"><h3>${title} ${badge(boundary.status||'unknown')}</h3>${renderObject(rest)}<div class="observations">${observations.map(obs=>observationCard(obs)).join('')}</div></div>`;
  }

  function intervalHTML(shot) {
    if(shot.interval_type==='shot' || !shot.interval_type) return '';
    return `<div class="warning"><strong>${escape(label(shot.interval_type))} · ${escape(label(shot.boundary_status))}</strong><p>此区间完整保留在源时间轴中；区间端点用于逐帧分配，内部切镜或转场仍按下列范围记录。</p>${renderObject({interval_type:shot.interval_type,boundary_status:shot.boundary_status,start_seconds:shot.start_seconds,end_seconds:shot.end_seconds,start_pts:shot.start_pts,end_pts:shot.end_pts_exclusive??shot.end_pts,candidate_points:shot.candidate_points||[],transition_kinds:shot.transition_kinds||[],review_evidence_paths:shot.review_evidence_paths||[],uncertainty_reason:shot.uncertainty_reason})}</div>`;
  }

  async function openShot(id, seekSeconds) {
    try {
      const shot=await api('/api/shot',{shot_id:id});activeShot=shot; const initialTime=seekSeconds??shot.start_seconds;
      const source=sourceMap.get(shot.source_id)||{}, c=shot.classification||{}, track=source.track||{}, sourcePath=track.source_path||source.source_path;
      $('detail-source').textContent=`${source.title||shot.source_id} · ${shot.source_id}`;
      $('detail-title').textContent=`${time(shot.start_seconds)} → ${time(shot.end_seconds)}`;
      const video=sourcePath?`<video id="source-video" controls preload="metadata" src="${fileURL(sourcePath)}#t=${Number(initialTime)}"></video><div class="video-buttons"><button data-seek="${shot.start_seconds}">镜头开头</button><button data-seek="${Math.max(0,shot.start_seconds-2)}">边界前 2 秒</button><button data-seek="${Math.max(shot.start_seconds,shot.end_seconds-2)}">镜头结尾前 2 秒</button><a href="${fileURL(sourcePath)}#t=${shot.start_seconds},${shot.end_seconds}" target="_blank" rel="noopener">单独打开原片区间</a></div>`:'<div class="warning">未提供可播放的本地源轨路径。此镜头保留在时间轴中。</div>';
      $('detail-content').innerHTML=`${candidatesHTML(shot)}<div class="detail-layout"><div class="video-panel">${video}<div class="time-note">播放器按秒跳转可能落在邻近帧；下列 PTS、time_base 与边界邻帧用于精确核对。</div><details><summary>原始视频轨与每帧范围</summary>${renderObject({source_id:shot.source_id,shot_id:shot.shot_id,source_path:sourcePath,start_seconds:shot.start_seconds,end_seconds:shot.end_seconds,start_pts:shot.start_pts,end_pts:shot.end_pts,frame_count:shot.frame_count,time_base:track.time_base,track})}</details></div><div class="evidence-block"><div class="badges">${classificationBadges(c)}${shot.interval_type && shot.interval_type!=='shot'?badge(shot.interval_type):''}${badge(shot.content?.kind||'unknown')}</div><h3>物理机位归属</h3><p>${escape(c.installation_id?candidateName(c.installation_id):'尚未确定')}</p><p class="muted">${escape(c.reason||'未记录归属原因')}</p><details><summary>查看匹配依据与尚未确定的原因</summary><h3>支持证据</h3>${evidenceHTML(c.supporting_evidence)}<h3>反对证据／限制</h3>${evidenceHTML(c.opposing_evidence)}</details><details><summary>外观组与归属原始记录</summary>${pretty(c)}</details></div></div><section class="evidence-block"><h3>镜头内背景观察序列</h3><p class="muted">这些图片用于检查构图变化；代表帧数量不等于机位数量。点击图片查看原尺寸。</p><div class="observations">${(shot.observations||[]).map(obs=>observationCard(obs)).join('')||'<p class="muted">未提供观察图；需查看完整原镜头。</p>'}</div></section><section class="evidence-block"><h3>边界范围与邻帧证据</h3>${intervalHTML(shot)}<div class="boundary-grid">${boundaryHTML(shot.boundary_start,'起点')}${boundaryHTML(shot.boundary_end,'终点')}</div></section><section class="evidence-block"><h3>回放、重复画面与历史来源</h3><p class="muted">重复拍摄画面不作为独立验证；播出场次与拍摄场次分别以记录为准。</p>${renderObject(shot.content||{})}<h3>重复与回放关联</h3>${evidenceHTML(shot.duplicate_links)}<h3>历史标签／旧镜头引用</h3>${evidenceHTML(shot.historical_links)}</section><details><summary>完整镜头 JSON</summary>${pretty(shot)}</details><div class="detail-pager"><button id="previous-shot">同场上一个镜头</button><button id="next-shot">同场下一个镜头</button></div>`;
      if(!$('shot-dialog').open)$('shot-dialog').showModal();
      $('shot-dialog').scrollTop=0;
      history.replaceState(null,'','#shot='+encodeURIComponent(id));
      const player=$('source-video'); if(player) {
        player.addEventListener('loadedmetadata',()=>{player.currentTime=initialTime;},{once:true});
        player.addEventListener('error',()=>{const note=document.createElement('div');note.className='warning';note.textContent='该源轨未能在浏览器播放，请核对媒体缺失记录或源轨编码。';player.after(note);},{once:true});
      }
      $('previous-shot').onclick=()=>adjacentShot(-1);$('next-shot').onclick=()=>adjacentShot(1);
    }catch(err){error('镜头读取失败：'+err.message);}
  }

  async function adjacentShot(direction) {
    const rows=activeShot.source_id===$('source-filter').value?timeline:await api('/api/timeline',{source_id:activeShot.source_id});
    const index=rows.findIndex(row=>row.shot_id===activeShot.shot_id), next=rows[index+direction];
    if(next)await openShot(next.shot_id);
  }
  function clearFilters() { for(const id of Object.keys(filters))$(id).value=''; offset=0; }
  function selectGroup(id,value) { if($('shot-dialog').open){$('shot-dialog').close();history.replaceState(null,'',location.pathname+location.search);} clearFilters(); if($(id).tagName==='SELECT' && !Array.from($(id).options).some(o=>o.value===value)) {const option=document.createElement('option');option.value=value;option.textContent=value;$(id).append(option);} $(id).value=value;setTab('shots');loadShots();loadTimeline();window.scrollTo({top:200,behavior:'smooth'}); }

  async function start() {
    if(location.protocol==='file:') {
      const server=new URL(document.currentScript?.src || [...document.scripts].find(s=>s.src.includes('camera_classification.js')).src).origin;
      document.body.innerHTML=`<main class="standalone"><h1>RinkSpace · 本场冰场镜头</h1><p>源视频需要本地服务提供分段读取。<a href="${server}/">打开可播放的冰场镜头目录</a></p><p>如服务尚未启动，请运行：</p><pre>.venv\\Scripts\\python.exe scripts/serve_camera_classification.py --port ${new URL(server).port||8767}</pre></main>`;
      return;
    }
    try {
      meta=await api('/api/catalog');sourceMap=new Map((meta.sources||[]).map(s=>[s.source_id,s]));installationMap=new Map((meta.installations||[]).map(item=>[item.installation_id,item]));
      const referenceData=await api('/api/references');referenceMap=new Map((referenceData.references||[]).map(ref=>[ref.reference_id,ref]));
      options('source-filter',meta.sources,'source_id','title');options('camera-filter',activeCameras(),'physical_camera_id');options('installation-filter',activeInstallations().map(item=>({...item,label:candidateName(item.installation_id)+' · '+(sourceMap.get(item.source_id)?.title||item.source_id||'')})),'installation_id');options('content-filter',meta.viewer.content_kinds);
      options('appearance-source',meta.sources,'source_id','title');
      $('catalog-download').href=fileURL(meta.viewer.catalog_path);renderOverview();renderCameras();
      const human=await api('/api/human-evidence');
      const activeInstallationIDs=new Set(activeInstallations().map(item=>item.installation_id));
      human.cases=(human.cases||[]).filter(item=>activeInstallationIDs.has(item.installation_id));
      human.physical_identity_decisions=(human.physical_identity_decisions||[]).filter(item=>(item.installation_ids||[]).some(id=>activeInstallationIDs.has(id))).map(item=>({...item,installation_ids:(item.installation_ids||[]).filter(id=>activeInstallationIDs.has(id))}));
      if(human.cases?.length || human.physical_identity_decisions?.length) {
        const panel=document.createElement('section');panel.className='evidence-block';
        const decisions=(human.physical_identity_decisions||[]).map(d=>`<div class="evidence-block"><h3>${escape(d.label||d.decision_id)} ${badge('human_confirmed_position_group')}</h3><div class="id">${escape(d.decision_id)}</div><p><strong>用户原话：${escape(d.user_statement)}</strong></p>${renderObject({scope:d.scope})}${d.physical_camera_id?`<button data-camera="${escape(d.physical_camera_id)}">查看此机位全部关联镜头</button>`:''}<details><summary>关联安装版本（${(d.installation_ids||[]).length}）</summary><ul class="evidence-list">${(d.installation_ids||[]).map(id=>`<li><button data-installation="${escape(id)}">${escape(id)}</button></li>`).join('')}</ul></details><details><summary>判断原始记录</summary>${renderObject(d)}</details></div>`).join('');
        panel.innerHTML='<h2>人工确认的机位关联</h2><p class="muted">保留用户原话，并分别展示每项判断的适用范围。</p>'+decisions+(human.cases||[]).map(c=>`<div class="evidence-block"><h3>${escape(c.case_id)} ${badge(c.status)}</h3><p>${escape(c.question)}</p><p><strong>用户判断：${escape(c.response||c.user_statement)}</strong></p><p class="muted">${escape(c.propagation_scope||c.scope||'')}</p><div class="observations">${(c.observations||[]).map(obs=>observationCard(obs,false)).join('')}</div><p>${c.installation_id?`<button data-installation="${escape(c.installation_id)}">查看此安装的关联镜头</button>`:''}</p><details><summary>人工判断原始记录与来源</summary>${renderObject(c)}</details></div>`).join('');
        $('camera-grid').before(panel);
      }
      const validation=await api('/api/validation'); const checkIndex=await api('/api/checks');
      const historicalValidation=`<div class="evidence-block"><h3>历史完整镜头检查区间</h3><p class="muted">片段保留原始帧与源 PTS 映射。联系表是采样序列；检查结论及独立性以下方真实画面报告为准。</p><div class="camera-grid">${(checkIndex.cases||[]).map(c=>`<article class="camera-card"><div class="card-body"><h3>${escape(c.source_id)}</h3><div class="card-time">${time(c.start_seconds)} → ${time(c.end_seconds)}</div><p class="muted">${escape(c.case_id)}<br>${escape(c.visual_review_status||'请查看报告')}</p><a href="${fileURL(c.clip_path)}" target="_blank" rel="noopener">播放完整检查片段</a><details><summary>采样页与源时间映射</summary>${(c.strips||[]).map((p,i)=>`<a class="artifact-link" href="${fileURL(p)}" target="_blank" rel="noopener">第 ${i+1} 页</a>`).join('')}<p class="muted">源轨 ${escape(c.source_path)}<br>PTS ${escape(c.start_pts)} → ${escape(c.end_pts_exclusive)} · time_base ${escape(c.time_base)}</p>${c.source_pts_mapping?`<a href="${fileURL(c.source_pts_mapping)}">源 PTS 映射</a>`:''}</details></div></article>`).join('')||'<p class="muted">尚未生成可播放的检查片段</p>'}</div></div><div class="evidence-block"><h3>目录概况与限制</h3>${renderObject(meta.summary||{})}</div>${Object.entries(validation).map(([key,value])=>`<div class="evidence-block"><h3>${escape(keys[key]||key)}</h3>${renderObject(value)}</div>`).join('')}`;
      const scope=meta.processing_scope||{},scopeSummary=scope.summary||{};
      $('validation-content').innerHTML=`<div class="evidence-block"><h3>当前本场冰场范围</h3><p>${scope.scene_screening_complete===true?'场景筛选已完成；以下镜头归属仍按当前证据展示。':'场景筛选尚未完成。场景待判定的画面尚未进入分镜，也未计作已忽略。'}</p>${renderObject(scopeSummary)}<p class="muted">当前展示 ${Number(meta.viewer.shot_count||0).toLocaleString()} 条入选冰场分段／镜头；是否入选只看画面中是否有本场冰场，不按节目类型排除。</p><details><summary>范围记录与迁移依据</summary>${renderObject(scope)}</details></div><details class="historical-validation"><summary>历史检查与片段（不作为当前范围已完成的证明）</summary><p class="warning">以下报告和片段来自历史处理范围，可能包含本轮已忽略或仍待判定的画面。历史完整性数字不能用于说明当前场景筛选已完成。</p>${historicalValidation}</details>`;
      await loadShots();drawTimeline();
      const shotFromHash=new URLSearchParams(location.hash.slice(1)).get('shot');if(shotFromHash)await openShot(shotFromHash);
    }catch(err){error('无法读取分类目录：'+err.message+'。请确认 catalog.json 已生成，且页面通过本地服务打开。');}
  }

  document.querySelectorAll('[data-tab]').forEach(button=>button.onclick=()=>setTab(button.dataset.tab));
  for(const id of Object.keys(filters)) if(id!=='query-filter')$(id).addEventListener('change',()=>{offset=0;loadShots();if(id==='source-filter')loadTimeline().catch(err=>error(err.message));});
  let queryTimer;$('query-filter').addEventListener('input',()=>{clearTimeout(queryTimer);queryTimer=setTimeout(()=>{offset=0;loadShots();},250);});
  $('prev-page').onclick=()=>{offset=Math.max(0,offset-pageSize);loadShots();};$('next-page').onclick=()=>{offset+=pageSize;loadShots();};
  $('reset-filters').onclick=()=>{clearFilters();loadShots();loadTimeline();};
  $('timeline').addEventListener('mousemove',event=>{const hit=timelineAt(event);$('timeline-hover').textContent=hit?.shot?`${time(hit.shot.start_seconds)} → ${time(hit.shot.end_seconds)} · ${label(hit.shot.status)}`:hit?scopeMessage(hit.kind,hit.seconds):'';});
  $('timeline').addEventListener('mouseleave',()=>{$('timeline-hover').textContent='';});
  $('timeline').addEventListener('click',event=>{const hit=timelineAt(event);if(!hit)return;if(hit.shot){$('jump-message').textContent='';openShot(hit.shot.shot_id,hit.seconds);}else showScopeMessage(hit.kind,hit.seconds);});
  $('jump-go').onclick=async()=>{
    const seconds=parseTime($('jump-time').value),sourceId=$('source-filter').value;
    if(!sourceId||!Number.isFinite(seconds)){ $('jump-message').textContent='请先选择来源并输入有效时间';return;}
    const kind=sceneAt(sourceId,seconds);
    if(kind!=='target'){showScopeMessage(kind,seconds);return;}
    try {
      const result=await api('/api/shots',{source_id:sourceId,at_seconds:seconds,limit:1});
      if(sourceId!==$('source-filter').value)return;
      if(result.total){$('jump-message').textContent='';openShot(result.shots[0].shot_id,seconds);}
      else showScopeMessage(kind,seconds);
    }catch(err){$('jump-message').textContent='镜头查询失败：'+err.message;}
  };
  $('jump-time').addEventListener('keydown',event=>{if(event.key==='Enter')$('jump-go').click();});
  $('close-detail').onclick=()=>{$('shot-dialog').close();history.replaceState(null,'',location.pathname+location.search);};
  $('shot-dialog').addEventListener('close',()=>{const player=$('source-video');if(player)player.pause();});
  document.addEventListener('click',event=>{const target=event.target.closest('[data-shot],[data-seek],[data-camera],[data-installation],[data-appearance]');if(!target)return;if(target.dataset.shot)openShot(target.dataset.shot);else if(target.dataset.seek!=null){const player=$('source-video');if(player){player.currentTime=Number(target.dataset.seek);player.scrollIntoView({block:'center',behavior:'smooth'});}}else if(target.dataset.camera)selectGroup('camera-filter',target.dataset.camera);else if(target.dataset.installation)selectGroup('installation-filter',target.dataset.installation);else if(target.dataset.appearance)selectGroup('appearance-filter',target.dataset.appearance);});
  document.addEventListener('keydown',event=>{const card=event.target.closest('[data-shot]');if(card&&(event.key==='Enter'||event.key===' ')){event.preventDefault();openShot(card.dataset.shot);}});
  $('appearance-section').addEventListener('toggle',()=>{if($('appearance-section').open)loadAppearances();});
  $('appearance-prev').onclick=()=>{appearanceOffset=Math.max(0,appearanceOffset-24);loadAppearances();};$('appearance-next').onclick=()=>{appearanceOffset+=24;loadAppearances();};
  $('appearance-source').onchange=()=>{appearanceOffset=0;loadAppearances();};
  let appearanceTimer;$('appearance-search').oninput=()=>{clearTimeout(appearanceTimer);appearanceTimer=setTimeout(()=>{appearanceOffset=0;loadAppearances();},250);};
  window.addEventListener('resize',drawTimeline);
  start();
})();
