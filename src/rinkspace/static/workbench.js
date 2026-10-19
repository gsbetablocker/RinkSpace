"use strict";

/* All editing coordinates are image pixels. CSS scaling and devicePixelRatio do
   not enter stored data: canvases have source-pixel backing dimensions. */
const $ = (id) => document.getElementById(id);
const ui = {state:null, video:null, frame:null, playing:false, request:0, busy:false,
  mode:"inspect", draft:[], pair:{source:null,reference:null}, reference:false,
  selectedShot:null, activeRun:null, noticeTimer:null};
const labels = {programme:"节目",live:"节目",replay:"回放",non_programme:"非节目",unknown:"未定"};
const shotLabel = (shot) => `${labels[shot.label] || shot.label || "未定"}${shot.status==="corrected"?"":"候选"}`;
const roles = {development:"开发镜头",development_full_shot:"完整开发镜头",development_mask_correction:"JIN 逐帧去污染开发对照",held_out:"同赛事保留镜头",holdout_full_shot:"完整保留镜头",failure_control:"失败对照",static_wide_group_protected_experiment:"清冰后固定广角 · 60秒独立组对照"};
const runStatuses = {candidate:"候选",candidate_not_independently_validated:"候选 / 未独立验收",candidate_not_formal_acceptance:"静态对照 / 未正式验收"};
const backends = {photometric_affine:"Affine 光度候选",rotation_zoom_experimental:"旋转与变焦约束实验",baseline:"旧基线"};
const reasons = {independent_landmark_validation_pending:"局部映射已生成；尚待独立物理地标检查。",new_local_reference_requires_independent_validation:"建立新的局部参照；独立检查尚未完成。",camera_cut_or_implausible_interframe_motion:"检测到切镜或不可信的大幅运动，配准已断链。",insufficient_features:"可用冰面纹理不足，配准已断链。",insufficient_tracks:"有效跟踪点不足，配准已断链。",insufficient_inliers:"符合冰面模型的特征不足，配准已断链。",insufficient_spatial_support:"特征分布过于集中，缺少足够冰面支持。",large_time_gap:"采样时间间隔过大，局部映射不再沿用。",camera_cut:"相机切镜，已终止上一条映射。"};
const reasonText = (reason) => reasons[reason] || reason;
Object.assign(reasons,{photometric_candidate_requires_independent_geometry_check:"光度拟合候选；仍需独立物理组检查，光度采样不是几何误差证据。",first_observed_reference_only:"镜头参考帧自身；不计作配准或独立精度成功。",recovery_gap_exceeded_requires_explicit_new_component:"失配已超过恢复时间；保持缺失，须明确切镜后才可建立新参考。",insufficient_safe_photometric_support:"安全冰面支撑或光度匹配不足；本帧没有映射。",composed_reference_horizon_crosses_fitting_support:"累计变换的参考地平线穿过拟合区域，映射已拒绝。"});
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const finite = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
const num = (v, n=3) => finite(v) ? Number(v).toFixed(n) : "—";
const rate = (v) => {if(finite(v))return Number(v);const match=String(v ?? "").match(/^(\d+)\/(\d+)$/);return match && Number(match[2])?Number(match[1])/Number(match[2]):null;};
const clock = (v) => { if (!finite(v)) return "—"; const s=Number(v), h=Math.floor(s/3600); return `${h?String(h).padStart(2,"0")+":":""}${String(Math.floor(s/60)%60).padStart(2,"0")}:${(s%60).toFixed(3).padStart(6,"0")}`; };
const currentVideoId = () => ui.video?.id ?? ui.video?.video_id;
const allVideos = () => {const v=ui.state?.project?.videos ?? ui.state?.videos ?? [];return Array.isArray(v)?v:Object.entries(v).map(([id,x])=>({id,...x}));};
const allShots = () => (ui.state?.project?.shots ?? []).filter(s => !s.video_id || s.video_id===currentVideoId());
const annotations = () => {const a=ui.state?.project?.annotations ?? [];return (Array.isArray(a)?a:Object.values(a).flat()).filter(a=>!a.video_id || a.video_id===currentVideoId());};
const frameAnnotations = () => annotations().filter(a=>Number(a.source_frame_index)===Number(ui.frame?.source_frame_index));
const frameCount = () => Number(ui.video?.frame_count ?? ui.video?.decoded_frame_count ?? 0);

function notice(message, error=false, persistent=false) {
  clearTimeout(ui.noticeTimer); $("notice").textContent=message; $("notice").classList.toggle("error",error); $("notice").hidden=!message;
  if (message && !error && !persistent) ui.noticeTimer=setTimeout(()=>{$("notice").hidden=true;},6000);
}
async function api(path, body) {
  const response=await fetch(path, body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  let data; const type=response.headers.get("content-type") ?? "";
  if(type.includes("json")) data=await response.json(); else data={error:await response.text()};
  if(!response.ok || data.error) throw new Error(typeof data.error==="string"?data.error:data.message || `请求失败（HTTP ${response.status}）`);
  return data;
}
async function perform(fn) {try{return await fn();}catch(error){notice(error.message || String(error),true);return null;}}
function setBusy(value) {ui.busy=value;updateControls();}
function stopPlayback() {ui.playing=false;$("playButton").textContent="播放";}
function updateControls() {
  const selected=Boolean(ui.video), indexed=selected && frameCount()>0 && ui.video.indexed!==false;
  const ready=indexed && !ui.busy, exists=selected && ui.video.exists!==false;
  for(const id of ["playButton","frameInput","goFrame","frameRange"]) $(id).disabled=!ready;
  $("playButton").disabled=!ready || !ui.frame;
  const evidence=Boolean(ui.frame && (ui.frame.fixed_url || ui.frame.mask_url || finite(ui.frame.next_evidence_frame) || finite(ui.frame.previous_evidence_frame)));
  $("playMode").options[1].disabled=!evidence;$("playMode").disabled=!ready;
  $("playMode").title=evidence?"证据播放只走本次运行采样的真实源帧；前后箭头始终逐一源帧步进。":"当前源帧没有对应运行证据，请从运行记录定位证据帧。";
  if(!evidence && $("playMode").value==="evidence")$("playMode").value="source";
  $("prevFrame").disabled=!ready || Number(ui.frame?.source_frame_index ?? 0)<=0;
  $("nextFrame").disabled=!ready || Number(ui.frame?.source_frame_index ?? 0)>=frameCount()-1;
  for(const id of ["markKeyframe","setRunStart","setRunEnd"]) $(id).disabled=!ready || !ui.frame;
  for(const id of ["runButton","runStart","runEnd","runRole","runBackend","runEveryN","saveShot","compareBoundary"]) $(id).disabled=!ready || !exists;
  $("scanButton").disabled=!exists || ui.busy;
  $("scanButton").textContent=indexed?"重新扫描镜头":"建立索引与分镜";
  $("saveButton").disabled=!ui.state || ui.busy;
  $("undoButton").disabled=!ui.state || ui.busy || ui.state.project?.can_undo===false;
  $("referenceToggle").disabled=!ui.frame?.reference_url;
  $("saveMask").disabled=!ready || ui.draft.length<3 || !ui.frame || ($("maskReason").value==="heldout_protection" && !$("maskGroupId").value.trim());
  $("saveLandmark").disabled=!ready || !ui.pair.source || !ui.pair.reference || !$("landmarkId").value.trim() || !$("landmarkPointId").value.trim();
  document.querySelectorAll("[data-mode]").forEach(button=>{button.disabled=!ready || !ui.frame;});
  $("mediaSeekSeconds").disabled=!ui.video?.media_url;
  $("mediaSeekButton").disabled=!ui.video?.media_url || $("sourcePlayer").readyState<1;
  $("frameRange").max=String(Math.max(0,frameCount()-1));
  $("frameInput").max=String(Math.max(0,frameCount()-1));
  if(selected && !indexed) $("decodeNote").textContent=exists?"尚未完成源帧索引；请先建立索引与分镜，帧号不会按 FPS 猜测。":"原片路径不存在，请配置实际本地文件。";
  else $("decodeNote").textContent=$("playMode").value==="evidence"?"按本次运行的采样源帧同步播放；显示真实源 PTS，速度受解码限制。← → 始终单帧。":"逐一读取真实源帧；播放速度受本地解码限制。空格播放 / 暂停，← → 单帧。";
}
function videoSummary() {
  const v=ui.video;if(!v)return;
  const mediaUrl=v.media_url;
  $("mediaBrowser").hidden=!mediaUrl;
  if(mediaUrl && $("sourcePlayer").getAttribute("src")!==mediaUrl){
    $("sourcePlayer").pause();$("sourcePlayer").src=mediaUrl;
    $("mediaBrowser").open=true;
  }
  $("mediaInfo").textContent=v.indexed?"完整原片连续播放；精确源帧与PTS请在下方已索引视图中检查。":"完整原片可以连续播放和拖动取材；尚未建立全场完整解码帧索引，播放器时间不冒充源帧编号。";
  $("sourcePath").value=v.path || "";
  const player=$("sourcePlayer"), playerReady=Boolean(mediaUrl && player.getAttribute("src")===mediaUrl && player.readyState>=1);
  const duration=finite(v.duration_s)?`${clock(v.duration_s)}`:playerReady && finite(player.duration)?`${clock(player.duration)}（播放器读取）`:"时长待读取 / 待索引";
  const dimensions=finite(v.width)&&finite(v.height)?`${v.width} × ${v.height}`:playerReady && player.videoWidth?`${player.videoWidth} × ${player.videoHeight}（播放器读取）`:null;
  const bits=[v.codec_name || v.codec || v.metadata?.codec,dimensions,finite(rate(v.fps))?`${num(rate(v.fps),2)} FPS（流声明）`:null,duration,v.exists===false?"文件不存在":v.indexed===false?"待建立源帧索引":frameCount()?`${frameCount().toLocaleString()} 源帧`:null].filter(Boolean);
  $("sourceSummary").textContent=bits.join(" · ");
  $("sourceHash").textContent=v.sha256 || v.source_sha256 || v.metadata?.source_sha256 || v.metadata?.video_id || (v.id && v.id.length>40?v.id:"SHA-256 在导入 / 分析时核验");
  $("provenanceText").textContent=v.path ? `输入：${v.path}` : "输入位置未配置。";
}
async function refreshState({keepFrame=true}={}) {
  const state=await api("/api/state");ui.state=state;
  const oldId=currentVideoId();const videos=allVideos();
  const selected=state.project?.selected_video_id ?? oldId;
  ui.video=videos.find(v=>(v.id ?? v.video_id)===selected) || videos[0] || null;
  $("projectName").textContent=state.project?.name || state.project?.project_name || "本地项目";
  $("revision").textContent=`版本 ${state.project?.revision ?? 0}`;
  $("videoSelect").innerHTML=videos.length?videos.map(v=>`<option value="${esc(v.id ?? v.video_id)}">${esc(v.label || v.name || v.id)}${v.exists===false?" · 缺少文件":""}</option>`).join(""):"<option>尚未配置输入视频</option>";
  if(ui.video)$("videoSelect").value=currentVideoId();
  if(!keepFrame || oldId!==currentVideoId() || ui.video?.indexed===false){ui.frame=null;ui.request++;ui.draft=[];ui.pair={source:null,reference:null};ui.reference=false;clearViews();}
  videoSummary();renderShots();renderAnnotations();renderRuns(state.runs ?? []);updateControls();
  document.dispatchEvent(new CustomEvent("rinkspace:source",{detail:{video:ui.video}}));
  const pick=state.capabilities?.pick_file;
  if(pick===false || pick?.available===false){$("pickPath").disabled=true;$("pickPath").title=pick?.reason || "本机文件选择器不可用，请粘贴完整路径。";}
  if(ui.video && ui.video.indexed!==false && frameCount()>0 && (!ui.frame || !keepFrame)) await loadFrame(0);
  return state;
}
function clearViews() {
  for(const kind of ["source","fixed","mask","error"]){$(`${kind}Image`).removeAttribute("src");$(`${kind}Empty`).hidden=false;}
  $("frameIndex").textContent="未索引";$("sourceTime").textContent="—";$("sourcePTS").textContent="—";
  $("frameDimensions").textContent="源帧规格待索引";$("sourceCaption").textContent="尚未读取已索引源帧；上方播放器仅用于连续浏览。";
  $("fixedCaption").textContent="当前没有源帧映射；未沿用其它视频的 H。";$("maskCaption").textContent="当前没有源帧掩膜证据。";
  $("fixedHeading").textContent="固定冰面";$("referenceToggle").textContent="查看参考帧";$("frameInput").value="";$("frameRange").value="0";
  $("mediaTime").textContent="播放器时长待读取；不是源帧索引。";renderQuality();pairReadout();drawOverlays();
}
function preload(url) {return new Promise(resolve=>{if(!url){resolve(false);return;}const im=new Image();im.onload=()=>resolve(true);im.onerror=()=>resolve(false);im.src=url;});}
async function loadFrame(index,{quiet=false}={}) {
  if(!ui.video || ui.video.indexed===false || frameCount()===0)return;
  const target=Math.max(0,Math.min(frameCount()-1,Math.trunc(Number(index))));
  if(!Number.isFinite(target))throw new Error("源帧号必须为整数。");
  const request=++ui.request;const videoId=currentVideoId();
  const q=new URLSearchParams({video_id:videoId,index:String(target)});if(ui.activeRun)q.set("run_id",ui.activeRun);
  const meta=await api(`/api/frame-meta?${q}`);
  if(!finite(meta.source_frame_index) || !finite(meta.pts))throw new Error("后端未返回源帧身份或 PTS，不能将此图冒充源帧。");
  const urls={source:meta.image_url,fixed:ui.reference?meta.reference_url:meta.fixed_url,mask:meta.mask_url,error:meta.error_url};
  const loaded=await Promise.all(Object.entries(urls).map(async([kind,url])=>[kind,url,await preload(url)]));
  if(request!==ui.request || videoId!==currentVideoId())return;
  if(!loaded.find(x=>x[0]==="source")?.[2])throw new Error("源帧图像无法读取。请检查视频路径及解码日志。");
  ui.frame=meta;ui.draft=[];ui.pair={source:null,reference:null};
  for(const [kind,url,ok] of loaded){if(url && ok){$(`${kind}Image`).src=url;$(`${kind}Empty`).hidden=true;}else{$(`${kind}Image`).removeAttribute("src");$(`${kind}Empty`).hidden=false;}}
  $("frameIndex").textContent=String(meta.source_frame_index);$("sourceTime").textContent=`${clock(meta.source_time_s)} / ${num(meta.source_time_s)} s`;
  $("sourcePTS").textContent=`${meta.pts} × ${meta.time_base ?? "未知时基"}`;
  $("frameDimensions").textContent=`源像素 ${meta.width ?? ui.video.width ?? "—"} × ${meta.height ?? ui.video.height ?? "—"} · 流声明 ${num(rate(meta.fps ?? ui.video.fps),2)} FPS`;
  $("frameInput").value=String(meta.source_frame_index);$("frameRange").value=String(meta.source_frame_index);
  $("sourceCaption").textContent=`源帧 #${meta.source_frame_index} · ${num(meta.source_time_s)} s · image_px`;
  $("fixedHeading").textContent=ui.reference?"配对参考帧":"固定冰面";
  $("referenceToggle").textContent=ui.reference?"返回固定冰面":"查看参考帧";
  const reg=meta.registration ?? {};
  const rejectedPlane=(reg.review_disposition || meta.review_disposition)==="rejected_plane_contamination";
  if(rejectedPlane)$("fixedHeading").textContent=ui.reference?"失败基线的参考原帧":"固定冰面（失败诊断）";
  $("fixedCaption").textContent=rejectedPlane?`已否定的污染基线；${ui.reference?"参考原帧仅供审阅":"变换图仅保留作失败诊断，不是有效冰面补偿"}。`:ui.reference?`参考源帧 #${meta.reference_frame_index ?? "—"}；点击静止冰面地标。`:reasonText(reg.reason) || meta.fixed_reason || (meta.fixed_url?`参考源帧 #${meta.reference_frame_index ?? reg.reference_frame_index ?? "—"} · 仅已观测区域` : "无有效映射；不沿用其它帧的 H。");
  $("maskCaption").textContent=rejectedPlane?"失败基线的原掩膜证据；包含已确认的非冰面污染，不能视作有效冰面区域。":meta.mask_reason || (meta.mask_url?"可用冰面与排除区的当前帧证据；掩膜仍需目视检查。":"当前源帧尚未运行掩膜。相邻采样帧的掩膜不会沿用。");
  if(meta.photometric_audit_url){const link=document.createElement("a");link.href=meta.photometric_audit_url;link.textContent=" 下载本帧完整拟合足迹（gzip）";link.setAttribute("download","");$("maskCaption").append(link);}
  renderQuality(reg);renderAnnotations();renderShots();updateControls();pairReadout();requestAnimationFrame(drawOverlays);
  document.dispatchEvent(new CustomEvent("rinkspace:frame",{detail:meta}));
  if(!quiet && !meta.fixed_url && meta.nearest_evidence_frame!==undefined)$("fixedCaption").textContent=`当前帧未采样；最近证据源帧 #${meta.nearest_evidence_frame}。未沿用其映射。`;
  return meta;
}
async function play() {
  if(ui.playing){stopPlayback();return;}if(!ui.frame)return;ui.playing=true;$("playButton").textContent="暂停";
  while(ui.playing && ui.frame){
    const evidenceMode=$("playMode").value==="evidence";
    const next=evidenceMode?ui.frame.next_evidence_frame:Number(ui.frame.source_frame_index)+1;
    if(!finite(next) || next<=ui.frame.source_frame_index || next>=frameCount()){stopPlayback();break;}
    const before=performance.now();const lastPTS=ui.frame.source_time_s;
    try{await loadFrame(Number(next),{quiet:true});}catch(e){stopPlayback();notice(e.message,true);break;}
    const dt=finite(lastPTS)&&finite(ui.frame.source_time_s)?Math.max(0,(ui.frame.source_time_s-lastPTS)*1000):0;
    await new Promise(resolve=>setTimeout(resolve,Math.max(0,dt-(performance.now()-before))));
  }
}
function renderQuality(reg={}) {
  const status=reg.status || reg.quality_status || "unavailable";
  const audit=reg.independent ?? reg.independent_audit ?? reg.audit ?? reg;
  const rejectedPlane=(reg.review_disposition || ui.frame?.review_disposition)==="rejected_plane_contamination";
  const accepted=!rejectedPlane && (audit.accepted===true || ["accepted","validated","passed","independently_validated_local_region"].includes(status)), failure=rejectedPlane || ["failed","rejected","disconnected","broken"].includes(status);
  const label=rejectedPlane?"失败：非冰面污染 / 漂移":accepted?"局部独立验证通过":failure?"已断链 / 不可用":["candidate","available"].includes(status)?"候选":"不可用";
  const explanation=rejectedPlane?`${reg.review_note || ui.frame?.review_note || "此运行已因非冰面污染与漂移被否定。"} 当前图像和原始数值仅保留作失败诊断，不是有效冰面补偿。`:accepted?"此处被独立观测覆盖的局部区域满足声明的误差门槛；不代表全场或米制精度。":reasonText(reg.reason) || "当前帧暂无有效的局部配准与独立验证证据。";
  $("qualitySummary").innerHTML=`<div class="quality-row"><span class="quality-dot ${accepted?"":failure?"failed":status==="candidate"?"candidate":"unknown"}"></span><div><strong>局部配准 <em>${label}</em></strong><p>${esc(explanation)}</p>${audit.status==="independent_checks_incomplete_or_failed"?'<p>独立验收未通过或覆盖尚不完整，详见误差与运行报告。</p>':""}</div></div>`;
  const points=audit.independent_points ?? audit.check_count ?? audit.n_checks ?? audit.n_visible ?? audit.visible_count;
  const count=Array.isArray(points)?points.length:points;
  const photo=reg.fit_points_role==="photometric_sampling_not_independent_correspondences";
  const fitMetrics=photo?[["光度采样总数（非独立检查）",reg.fit_points_full_count ?? "—"],["显示采样子集",reg.fit_current_xy?.length ?? 0],["光度相关性（非几何误差）",num(reg.photometric_final_correlation,3)]]:[["拟合内点数量（非验收）",reg.train_inliers ?? reg.inlier_count ?? reg.inliers ?? "—"],["拟合 P95 / 参考px",num(reg.inlier_p95_reference_px,2)]];
  const metrics=[["源帧身份",ui.frame?`#${ui.frame.source_frame_index}`:"—"],["参考源帧",ui.frame?.reference_frame_index ?? reg.reference_frame_index ?? "—"],["独立检查点",count ?? "—"],["独立中位误差 / px",num(audit.median_error_px ?? audit.median_px,2)],["独立 P95 误差 / px",num(audit.p95_error_px ?? audit.p95_px,2)],...fitMetrics,["几何层级",reg.motion_model==="affine"?"局部 affine 近似 / 非米制":"local_projective"]];
  $("qualityMetrics").innerHTML=metrics.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  $("errorTag").textContent=rejectedPlane?"失败基线诊断":accepted?"局部独立检查通过":audit.status==="independent_checks_incomplete_or_failed"?"未通过 / 覆盖不足":ui.frame?.error_url?"查看独立误差":"等待独立检查";
  $("errorCaption").textContent=ui.frame?.error_reason || (finite(audit.median_error_px ?? audit.median_px)?`中位 ${num(audit.median_error_px ?? audit.median_px,2)} px · P95 ${num(audit.p95_error_px ?? audit.p95_px,2)} px · 以报告所声明的参考分辨率计量。`:"检查点独立于拟合；源分辨率、可见性与分布见运行报告。");
}
function renderShots() {
  const shots=allShots();const container=$("shotTimeline");
  if(!shots.length){container.innerHTML='<div class="timeline-empty">尚未生成镜头候选；建立索引与分镜后可检查切点。</div>';$("shotEditor").hidden=true;return;}
  const total=shots.reduce((sum,s)=>sum+Math.max(1,Number(s.end_frame ?? 0)-Number(s.start_frame ?? 0)+1),0);
  container.innerHTML=shots.map((s,i)=>{const length=Math.max(1,Number(s.end_frame ?? 0)-Number(s.start_frame ?? 0)+1);const active=ui.selectedShot===s.id || (ui.selectedShot==null && ui.frame && ui.frame.source_frame_index>=s.start_frame && ui.frame.source_frame_index<=s.end_frame);return `<button class="shot-segment ${esc(s.label ?? "unknown")} ${active?"active":""}" data-shot="${esc(s.id ?? i)}" style="flex:${length/total};" title="${esc(shotLabel(s))} · 源帧 ${s.start_frame}–${s.end_frame}">${esc(shotLabel(s))}<span>${num(s.start_s,1)}–${num(s.end_s,1)} s</span></button>`;}).join("");
  container.querySelectorAll("button").forEach(button=>button.addEventListener("click",()=>perform(async()=>{stopPlayback();const shot=shots.find((s,i)=>String(s.id ?? i)===button.dataset.shot);ui.selectedShot=shot.id ?? button.dataset.shot;populateShot(shot);await loadFrame(shot.start_frame);})));
  if(ui.selectedShot){const shot=shots.find(s=>String(s.id)===String(ui.selectedShot));if(shot)populateShot(shot);}
}
function populateShot(shot) {$("shotEditor").hidden=false;$("shotLabel").value=shot.label==="live"?"programme":shot.label ?? "unknown";$("shotStart").value=shot.start_frame;$("shotEnd").value=shot.end_frame;}
async function mutate(action, data={}) {
  stopPlayback();const result=await api("/api/project",{action,...data});
  await refreshState();if(ui.frame)await loadFrame(ui.frame.source_frame_index);return result;
}
async function waitJob(result,statusElement="runStatus") {
  let data=result;const jobId=result.job_id;
  if(!jobId)return data;
  for(;;){
    data=await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    const status=data.status ?? "running";$(statusElement).textContent=data.message || data.progress_message || `本地处理：${status}`;
    if(["complete","completed","done","success","succeeded"].includes(status))return data.result ?? data;
    if(["failed","error","cancelled"].includes(status))throw new Error(data.error || data.message || "本地任务失败，请检查日志。");
    await new Promise(resolve=>setTimeout(resolve,1200));
  }
}
async function prepare() {stopPlayback();setBusy(true);$("runProgress").hidden=false;try{notice("正在按真实解码顺序建立源帧索引与镜头候选；原片保持只读。",false,true);const result=await api("/api/prepare",{video_id:currentVideoId()});await waitJob(result);await refreshState({keepFrame:false});notice("源帧索引与镜头候选已更新。候选角色可在时间线上修正。");}finally{setBusy(false);$("runProgress").hidden=true;}}
async function runAnalysis() {
  if(!ui.video || ui.video.indexed===false || frameCount()===0)throw new Error("请先建立实际源帧索引，再运行局部配准；连续播放器时间不是帧索引。");
  stopPlayback();const start=Number($("runStart").value),end=Number($("runEnd").value);
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start)throw new Error("请填写有效源时间区间，结束时间应晚于起始时间。");
  setBusy(true);$("runProgress").hidden=false;$("runStatus").textContent="正在启动本地分析…";
  try{let result=await api("/api/run",{video_id:currentVideoId(),start_s:start,end_s:end,every_n:Number($("runEveryN").value),backend:$("runBackend").value,role:$("runRole").value});if(result.job_id)result=await waitJob(result);else if(result.run_id && !["complete","completed","done","success"].includes(result.status)){while(true){result=await api(`/api/run/${encodeURIComponent(result.run_id)}`);$("runStatus").textContent=result.message || result.status || "正在处理…";if(["complete","completed","done","success","succeeded"].includes(result.status))break;if(["error","failed"].includes(result.status))throw new Error(result.error || result.message || "分析失败");await new Promise(resolve=>setTimeout(resolve,1200));}}
    ui.activeRun=result.run_id ?? result.id ?? null;await refreshState();if(finite(result.first_frame_index))await loadFrame(result.first_frame_index);else if(ui.frame)await loadFrame(ui.frame.source_frame_index);if(!$("playMode").options[1].disabled)$("playMode").value="evidence";$("runStatus").textContent=`${backends[result.backend] || "本次运行"}已保存（每 ${result.every_n ?? 1} 个源帧）。请检查独立证据、掩膜与缺失位置。`;notice("局部分析结果已保存。候选是否通过以独立检查结果为准。");
  }finally{setBusy(false);$("runProgress").hidden=true;}
}
function renderRuns(runs) {
  const list=Array.isArray(runs)?runs:[];
  if(!list.length){$("runList").innerHTML='<p class="muted">尚无本次运行记录。选择源时间区间，生成第一个局部镜头结果。</p>';return;}
  $("runList").innerHTML=list.map(run=>{
    const rejectedPlane=run.review_disposition==="rejected_plane_contamination",status=run.review_disposition || run.quality_status || run.status || "candidate";
    const passed=!rejectedPlane && ["passed","accepted","validated"].includes(status),failed=rejectedPlane || ["failed","rejected","failure_control"].includes(status);
    const badge=rejectedPlane?"失败：非冰面污染 / 漂移":passed?"独立通过":failed?"失败 / 断链":runStatuses[status] || status;
    return `<article class="run-item"><div class="run-item-top"><h3>${esc(run.label || run.name || roles[run.role] || run.run_id || run.id)}</h3><span class="run-badge ${passed?"passed":failed?"failed":""}">${esc(badge)}</span></div>${run.thumbnail_url?`<img class="run-preview" loading="lazy" src="${esc(run.thumbnail_url)}" alt="${esc(run.label || "运行关键帧")}">`:""}<p>${esc(backends[run.backend] || "方法见报告")} · 每 ${esc(run.every_n ?? "—")} 个源帧 · ${esc(run.video_label || run.video_id || "源视频")} · ${num(run.start_s ?? run.source_start_s)}–${num(run.end_s ?? run.source_end_s)} s</p><p>${esc(run.review_note || run.reason || run.summary || "独立误差、源时间与必要配置见完整报告。")}</p><div class="run-links">${run.report_url?`<a href="${esc(run.report_url)}" target="_blank" rel="noopener">打开证据报告 ↗</a>`:""}${run.full_fitting_audit_url?`<a href="${esc(run.full_fitting_audit_url)}" target="_blank" rel="noopener">本机完整拟合足迹 ↗</a>`:""}${run.review_url?`<a href="${esc(run.review_url)}" download>下载 review.zip ↓</a>`:""}${finite(run.first_frame_index)?`<button class="text-button" data-open-run="${esc(run.run_id || run.id)}" data-video="${esc(run.video_id || currentVideoId())}" data-frame="${run.first_frame_index}">定位证据帧</button>`:""}</div></article>`;
  }).join("");
  $("runList").querySelectorAll("[data-open-run]").forEach(button=>button.addEventListener("click",()=>perform(async()=>{stopPlayback();ui.activeRun=button.dataset.openRun;if(button.dataset.video!==currentVideoId()){await api("/api/project",{action:"select_video",video_id:button.dataset.video});await refreshState({keepFrame:false});}await loadFrame(Number(button.dataset.frame));if(!$("playMode").options[1].disabled)$("playMode").value="evidence";updateControls();$("sourceStage").scrollIntoView({behavior:"smooth",block:"center"});})));
}
function setMode(mode) {
  stopPlayback();ui.mode=mode;ui.draft=[];ui.pair={source:null,reference:null};
  document.querySelectorAll("[data-mode]").forEach(b=>b.classList.toggle("selected",b.dataset.mode===mode));
  $("maskControls").hidden=!["rectangle","polygon"].includes(mode);$("landmarkControls").hidden=mode!=="landmark";
  const instructions={inspect:"在原片上查看已保存的修正。所有坐标存为原图像素。",rectangle:"在原片依次点击矩形的两个对角，再保存排除区。仅作用于当前源帧。",polygon:"在原片依次点击多边形顶点（至少 3 点），再保存排除区。仅作用于当前源帧。",landmark:"先点击原片上的静止冰面地标，再点击参考原图中的同一物理地标。检查点不参与拟合。"};
  $("editInstructions").textContent=instructions[mode];
  if(mode==="landmark" && ui.frame?.reference_url && !ui.reference){ui.reference=true;perform(()=>loadFrame(ui.frame.source_frame_index));}
  else if(mode==="landmark" && !ui.frame?.reference_url)$("editInstructions").textContent="当前帧尚无参考原图，暂不能配对。请先运行局部镜头分析并定位有映射的源帧。";
  pairReadout();updateControls();drawOverlays();
}
function canvasGeometry(which) {
  const image=$(`${which}Image`),stage=$(`${which}Stage`),canvas=$(`${which}Canvas`);
  if(!image.hasAttribute("src") || !image.naturalWidth || !ui.frame){canvas.style.display="none";return null;}
  const ir=image.getBoundingClientRect(),sr=stage.getBoundingClientRect();
  const width=which==="source"?Number(ui.frame.width ?? ui.video.width ?? image.naturalWidth):Number(ui.frame.reference_width ?? ui.frame.width ?? image.naturalWidth);
  const height=which==="source"?Number(ui.frame.height ?? ui.video.height ?? image.naturalHeight):Number(ui.frame.reference_height ?? ui.frame.height ?? image.naturalHeight);
  if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;
  Object.assign(canvas.style,{left:`${ir.left-sr.left}px`,top:`${ir.top-sr.top}px`,width:`${ir.width}px`,height:`${ir.height}px`,display:"block"});
  return {canvas,width,height,rect:ir};
}
function imagePoint(which,event) {
  const g=canvasGeometry(which);if(!g || !g.rect.width || !g.rect.height)return null;
  return [Math.min(g.width-1,Math.max(0,(event.clientX-g.rect.left)*g.width/g.rect.width)),Math.min(g.height-1,Math.max(0,(event.clientY-g.rect.top)*g.height/g.rect.height))].map(v=>Math.round(v*100)/100);
}
function drawPolygon(ctx,points,color,fill=false) {if(!points.length)return;ctx.strokeStyle=color;ctx.fillStyle=fill?color.replace(/1\)$/, ".18)"):color;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));if(points.length>=3)ctx.closePath();if(fill && points.length>=3)ctx.fill();ctx.stroke();for(const p of points){ctx.beginPath();ctx.arc(p[0],p[1],ctx.lineWidth*1.7,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();}}
function drawPoint(ctx,point,text,color) {if(!point)return;ctx.strokeStyle=color;ctx.fillStyle=color;const r=ctx.lineWidth*4;ctx.beginPath();ctx.arc(...point,r,0,Math.PI*2);ctx.moveTo(point[0]-r*1.5,point[1]);ctx.lineTo(point[0]+r*1.5,point[1]);ctx.moveTo(point[0],point[1]-r*1.5);ctx.lineTo(point[0],point[1]+r*1.5);ctx.stroke();ctx.fillText(text,point[0]+r*2,point[1]-r);}
function drawOverlays() {
  for(const which of ["source","fixed"]){const g=canvasGeometry(which);if(!g)continue;const ctx=g.canvas.getContext("2d");ctx.clearRect(0,0,g.width,g.height);ctx.lineWidth=Math.max(1.5,g.width/440);ctx.font=`${Math.max(13,g.width/75)}px sans-serif`;const editable=which==="source"?ui.mode!=="inspect":ui.mode==="landmark" && ui.reference;$(`${which}Stage`).classList.toggle("editing",editable);g.canvas.style.pointerEvents=editable?"auto":"none";
    for(const a of frameAnnotations()){if(which==="source" && (a.points || a.polygon))drawPolygon(ctx,a.points || a.polygon,"rgba(243,156,149,1)",true);if(a.source_xy && (which==="source" || ui.reference))drawPoint(ctx,which==="source"?a.source_xy:a.reference_xy,a.physical_landmark_id || a.id || "地标",a.role==="training"?"#efc781":"#8fdecf");}
    if(which==="source"){drawPolygon(ctx,ui.draft,"rgba(143,222,207,1)",true);drawPoint(ctx,ui.pair.source,"源图点","#8fdecf");}else if(ui.reference)drawPoint(ctx,ui.pair.reference,"参考图点","#8fdecf");
  }
}
function sourceClick(event) {
  if(!ui.frame || ui.mode==="inspect")return;const point=imagePoint("source",event);if(!point)return;
  if(ui.mode==="landmark")ui.pair.source=point;
  else if(ui.mode==="polygon")ui.draft.push(point);
  else if(ui.mode==="rectangle"){if(ui.draft.length!==1)ui.draft=[point];else{const p=ui.draft[0];ui.draft=[[Math.min(p[0],point[0]),Math.min(p[1],point[1])],[Math.max(p[0],point[0]),Math.min(p[1],point[1])],[Math.max(p[0],point[0]),Math.max(p[1],point[1])],[Math.min(p[0],point[0]),Math.max(p[1],point[1])]];}}
  pairReadout();updateControls();drawOverlays();
}
function pairReadout() {const format=p=>p?`(${p.map(n=>n.toFixed(1)).join(", ")}) px`:"—";$("pairReadout").textContent=`源图 ${format(ui.pair.source)} / 参考图 ${format(ui.pair.reference)}`;}
function renderAnnotations() {
  const rows=frameAnnotations();const masks=rows.filter(a=>a.points || a.polygon);$("maskCount").textContent=masks.length?`${masks.length} 个人工排除区`:"";
  $("annotationList").innerHTML=rows.map(a=>`<div class="annotation-row"><span>${esc(a.point_id || a.physical_landmark_id || a.reason || a.type || "关键帧")} · ${esc(a.role || a.provenance?.type || "人工修正")}</span>${a.id || a.annotation_id?`<button data-delete="${esc(a.id || a.annotation_id)}" title="删除这条标注（可撤销）">删除</button>`:""}</div>`).join("");
  $("annotationList").querySelectorAll("[data-delete]").forEach(button=>button.addEventListener("click",()=>perform(()=>mutate("delete_annotation",{annotation_id:button.dataset.delete,video_id:currentVideoId()}))));
}
async function compareBoundary() {
  const start=Number($("shotStart").value);if(!Number.isFinite(start))return;
  const indices=[Math.max(0,start-1),start];$("boundaryTitle").textContent=`镜头起始源帧 #${start}；检查边界前后，确认转场与断链。`;
  $("boundaryImages").innerHTML='<p class="muted">正在读取边界源帧…</p>';$("boundaryDialog").showModal();
  const metas=await Promise.all(indices.map(index=>api(`/api/frame-meta?${new URLSearchParams({video_id:currentVideoId(),index:String(index)})}`)));
  $("boundaryImages").innerHTML=metas.map((m,i)=>`<figure><img src="${esc(m.image_url)}" alt="${i===0?"切点前":"切点后"}源帧"><figcaption>${i===0?"切点前":"镜头开始"} · 源帧 #${m.source_frame_index} · PTS ${m.pts} × ${esc(m.time_base)} · ${num(m.source_time_s)} s</figcaption></figure>`).join("");
}
function bind() {
  const mediaReadout=()=>{const player=$("sourcePlayer");$("mediaTime").textContent=finite(player.duration)?`播放器 ${num(player.currentTime)} / ${num(player.duration)} s · 非源帧索引`:"播放器时长待读取；不是源帧索引。";};
  $("sourcePlayer").addEventListener("loadedmetadata",()=>{videoSummary();mediaReadout();updateControls();});
  $("sourcePlayer").addEventListener("durationchange",()=>{videoSummary();mediaReadout();updateControls();});
  $("sourcePlayer").addEventListener("timeupdate",mediaReadout);
  $("mediaSeekButton").onclick=()=>perform(()=>{const player=$("sourcePlayer"),seconds=Number($("mediaSeekSeconds").value);if(!ui.video?.media_url || player.readyState<1)throw new Error("请等待完整原片播放器读取时长。");if(!$("mediaSeekSeconds").value.trim() || !Number.isFinite(seconds) || seconds<0 || (finite(player.duration) && seconds>player.duration))throw new Error("请输入原片时长内的有效秒数。");player.currentTime=seconds;mediaReadout();});
  $("mediaSeekSeconds").onkeydown=event=>{if(event.key==="Enter"){event.preventDefault();$("mediaSeekButton").click();}};
  $("pathToggle").onclick=()=>{$("pathForm").hidden=!$("pathForm").hidden;};
  $("pathForm").onsubmit=event=>{event.preventDefault();perform(async()=>{stopPlayback();setBusy(true);try{const r=await api("/api/config",{video_id:currentVideoId(),path:$("sourcePath").value.trim()});await waitJob(r);await refreshState({keepFrame:false});notice("本地视频路径已保存。");}finally{setBusy(false);}});};
  $("pickPath").onclick=()=>perform(async()=>{const r=await api("/api/pick",{});if(r.path)$("sourcePath").value=r.path;else if(!r.cancelled)notice("未选择文件，可直接粘贴完整本地路径。");});
  $("videoSelect").onchange=()=>perform(async()=>{stopPlayback();ui.activeRun=null;ui.selectedShot=null;await api("/api/project",{action:"select_video",video_id:$("videoSelect").value});await refreshState({keepFrame:false});});
  $("prevFrame").onclick=()=>perform(()=>{stopPlayback();return loadFrame(Number(ui.frame?.source_frame_index ?? 0)-1);});
  $("nextFrame").onclick=()=>perform(()=>{stopPlayback();return loadFrame(Number(ui.frame?.source_frame_index ?? 0)+1);});
  $("playButton").onclick=()=>perform(play);
  $("playMode").onchange=()=>{stopPlayback();updateControls();};
  $("goFrame").onclick=()=>perform(()=>{stopPlayback();return loadFrame($("frameInput").value);});
  $("frameInput").onkeydown=event=>{if(event.key==="Enter")$("goFrame").click();};
  $("frameRange").onchange=()=>perform(()=>{stopPlayback();return loadFrame($("frameRange").value);});
  $("frameRange").oninput=()=>{$("frameInput").value=$("frameRange").value;};
  $("saveButton").onclick=()=>perform(async()=>{await mutate("save");notice("项目已保存，包含来源、镜头、关键帧和修正版本。");});
  $("undoButton").onclick=()=>perform(async()=>{await mutate("undo");notice("已撤销最近一次项目修改。");});
  $("scanButton").onclick=()=>perform(prepare);
  $("runButton").onclick=()=>perform(runAnalysis);
  $("runBackend").onchange=()=>{$("runBackendNote").textContent=$("runBackend").value==="rotation_zoom_experimental"?"假定光心固定、主点居中、初始焦距为图宽4倍；这是未标定的候选模型，仍须检查独立误差。":$("runBackend").value==="photometric_affine"?"Affine 是近似模型；完整拟合足迹保存在本机。运行完成不代表独立精度通过。":"旧基线用于复现既有实验，失配后可能产生多个参考分量；其候选状态不是验收通过。";};
  $("setRunStart").onclick=()=>{$("runStart").value=ui.frame.source_time_s;};
  $("setRunEnd").onclick=()=>{$("runEnd").value=ui.frame.source_time_s;};
  $("refreshRuns").onclick=()=>perform(async()=>{const result=await api("/api/runs");renderRuns(result.runs ?? result);});
  $("saveShot").onclick=()=>perform(async()=>{const start=Number($("shotStart").value),end=Number($("shotEnd").value);if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end<start||end>=frameCount())throw new Error("镜头边界需为有效源帧号，结束帧不得早于起始帧。");await mutate("edit_shot",{video_id:currentVideoId(),shot_id:ui.selectedShot,label:$("shotLabel").value,start_frame:start,end_frame:end});notice("镜头修正已保存，后续运行将使用新的边界与角色。");});
  $("compareBoundary").onclick=()=>perform(compareBoundary);$("closeBoundary").onclick=()=>$("boundaryDialog").close();
  $("referenceToggle").onclick=()=>perform(()=>{ui.reference=!ui.reference;return loadFrame(ui.frame.source_frame_index);});
  $("markKeyframe").onclick=()=>perform(async()=>{await mutate("add_keyframe",{video_id:currentVideoId(),source_frame_index:ui.frame.source_frame_index,pts:ui.frame.pts,time_base:ui.frame.time_base});notice("当前源帧已保存为关键帧。");});
  document.querySelectorAll("[data-mode]").forEach(button=>button.onclick=()=>setMode(button.dataset.mode));
  $("sourceCanvas").addEventListener("click",sourceClick);
  $("fixedCanvas").addEventListener("click",event=>{if(ui.mode!=="landmark" || !ui.reference)return;ui.pair.reference=imagePoint("fixed",event);pairReadout();updateControls();drawOverlays();});
  $("landmarkId").oninput=updateControls;
  $("landmarkPointId").oninput=updateControls;
  $("maskGroupId").oninput=updateControls;
  $("maskReason").onchange=()=>{$("maskGroupField").hidden=$("maskReason").value!=="heldout_protection";updateControls();};
  $("clearDrawing").onclick=()=>{ui.draft=[];updateControls();drawOverlays();};
  $("saveMask").onclick=()=>perform(async()=>{await mutate("add_mask",{video_id:currentVideoId(),source_frame_index:ui.frame.source_frame_index,pts:ui.frame.pts,time_base:ui.frame.time_base,coordinate_space:"image_px",shape:"polygon",points:ui.draft,reason:$("maskReason").value,operation:"exclude",...($("maskReason").value==="heldout_protection"?{physical_landmark_id:$("maskGroupId").value.trim()}:{})});ui.draft=[];notice("排除区已保存。重新运行该区间后生效；现有运行证据保留其原始版本。");});
  $("saveLandmark").onclick=()=>perform(async()=>{await mutate("add_landmark",{video_id:currentVideoId(),source_frame_index:ui.frame.source_frame_index,reference_frame_index:ui.frame.reference_frame_index,physical_landmark_id:$("landmarkId").value.trim(),point_id:$("landmarkPointId").value.trim(),role:$("landmarkRole").value,source_xy:ui.pair.source,reference_xy:ui.pair.reference,coordinate_space:"image_px",pts:ui.frame.pts,time_base:ui.frame.time_base});ui.pair={source:null,reference:null};notice("地标配对已保存。检查点与拟合地标分开记录，重新运行后进入对应审计。");});
  new ResizeObserver(()=>drawOverlays()).observe($("sourceStage"));new ResizeObserver(()=>drawOverlays()).observe($("fixedStage"));
  document.addEventListener("keydown",event=>{if(["INPUT","SELECT","TEXTAREA","BUTTON","VIDEO"].includes(document.activeElement.tagName)||$("boundaryDialog").open)return;if(event.code==="Space"){event.preventDefault();if(!$("playButton").disabled)$("playButton").click();}else if(event.key==="ArrowLeft"){event.preventDefault();if(!$("prevFrame").disabled)$("prevFrame").click();}else if(event.key==="ArrowRight"){event.preventDefault();if(!$("nextFrame").disabled)$("nextFrame").click();}});
}
bind();perform(()=>refreshState({keepFrame:false}));

// Exposed only to local QA for checking CSS/source-pixel round trips.
window.RinkSpaceUI={state:ui,imagePoint,canvasGeometry,loadFrame,refreshState};
