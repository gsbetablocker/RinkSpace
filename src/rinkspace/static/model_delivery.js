/* Offline review only: no fetch, no external dependency, no new localization. */
(function(){
  "use strict";
  const data=JSON.parse(document.getElementById("review-data").textContent),G=window.RinkSpaceGridGeometry;
  const $=id=>document.getElementById(id),video=$("video"),model=data.model;
  const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const fmt=value=>typeof value==="number"&&Number.isFinite(value)?value.toFixed(3):"—";
  const text=value=>typeof value==="string"?value:JSON.stringify(value);
  const colors={observed:"#82e4ca",predicted:"#9bbaff",candidate:"#ffcb72",ambiguous:"#ffcb72",unavailable:"#6e8491"};
  const labels={observed:"定位有证据",predicted:"连续推算",candidate:"待验证候选",ambiguous:"待验证候选",unavailable:"无法定位"};
  let run=null,frame=null,frameIndex=-1,time=0,mapTransform=null,presentedTime=null;
  const geometry=({projective:"位置坐标网（非等距）",euclidean_unscaled:"等距坐标网（任意单位）",metric_assumed:"米制网格（尺寸假设）",metric_verified:"米制网格（尺度有证据）"})[model.geometry_level]||"位置坐标网（非等距）";
  $("geometryLabel").textContent=geometry;
  $("modelIdentity").textContent=`${model.model_id} / ${model.revision} · ${(model.sources||[]).length} 个来源 · ${(model.reference_views||[]).length} 个参考画面 · ${model.status||"候选"}`;
  const regions=model.coverage_regions||[];
  $("coverage").textContent=Array.isArray(regions)?regions.map(r=>{const v=(model.reference_views||[]).find(v=>v.id===r.id);return r.label_zh||r.label||r.description||v?.label_zh||G.evidenceLabel(r.status)||r.region_id||r.id||text(r);}).join("；"):text(regions);
  const ambiguities=model.known_ambiguities||[];
  $("ambiguities").innerHTML=ambiguities.length?`<ul>${ambiguities.map(a=>`<li>${esc(typeof a==="string"?a:a.description||a.reason||text(a))}</li>`).join("")}</ul>`:"<p>完整边界、欧氏尺度及独立定位精度仍以模型记录为准；没有列出疑点不代表全场已验证。</p>";
  $("missingImages").textContent=data.missing_reference_images.length?`本包缺少 ${data.missing_reference_images.length} 张来源图片：${data.missing_reference_images.join("；")}`:"来源画面的原始时间与哈希保留在模型记录中。";
  function imageHTML(o){return `<figure>${o.image_url?`<a href="${esc(o.image_url)}" target="_blank"><img src="${esc(o.image_url)}" alt="${esc(o.label_zh||o.label||o.id||o.view_id||o.source_id||"参考画面")}" loading="lazy"></a>`:"<p>来源图片未装入本包</p>"}<figcaption>${esc(o.label_zh||o.label||o.view_id||o.id||"")} · ${esc(o.source_id||"")}<br>${fmt(o.source_time_s)} s · PTS ${esc(o.source_pts??o.pts??"—")} × ${esc(o.source_time_base??o.time_base??"—")}<br>${esc(G.evidenceLabel(o.model_contribution_zh||o.contribution||o.role||o.status||""))}</figcaption></figure>`;}
  $("referenceViews").innerHTML=(model.reference_views||[]).map(imageHTML).join("");
  if(model.overview_url){$("overviewLink").hidden=false;$("overviewLink").href=model.overview_url;}
  $("landmarks").innerHTML=(model.landmarks||[]).map((l,i)=>`<button data-landmark="${i}">${esc(G.landmarkLabel(l))}</button>`).join("");
  const landmarkPoint=G.landmarkPoint;
  function sourceDimensions(f){return [f?.image_width||f?.source_width||run?.image_width||run?.source_width,f?.image_height||f?.source_height||run?.image_height||run?.source_height];}
  function poly(ctx,points,transform=p=>p){ctx.beginPath();let n=0;for(const p of points||[]){const q=transform(p);if(q){n++?ctx.lineTo(...q):ctx.moveTo(...q);}}ctx.closePath();}
  function canvasContext(){
    const canvas=$("gridCanvas"),[width,height]=sourceDimensions(frame),w=video.videoWidth,h=video.videoHeight;
    if(!width||!height||!w||!h){canvas.width=1;canvas.height=1;return null;}
    if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;
    const vr=video.getBoundingClientRect(),sr=$("videoStage").getBoundingClientRect(),scale=Math.min(vr.width/w,vr.height/h),dw=w*scale,dh=h*scale;
    Object.assign(canvas.style,{width:`${dw}px`,height:`${dh}px`,left:`${vr.left-sr.left+(vr.width-dw)/2}px`,top:`${vr.top-sr.top+(vr.height-dh)/2}px`});
    return canvas.getContext("2d");
  }
  function drawMap(validFrame){
    const canvas=$("mapCanvas"),ctx=canvas.getContext("2d"),bounds=model.grid?.bounds;
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#0a1720";ctx.fillRect(0,0,canvas.width,canvas.height);if(!bounds||bounds.length!==4)return;
    const[x0,y0,x1,y1]=bounds,scale=Math.min((canvas.width-44)/(x1-x0),(canvas.height-50)/(y1-y0));if(!Number.isFinite(scale)||scale<=0)return;
    const ox=(canvas.width-(x1-x0)*scale)/2,oy=(canvas.height-(y1-y0)*scale)/2,transform=([x,y])=>[ox+(x-x0)*scale,oy+(y-y0)*scale];mapTransform={x0,y0,scale,ox,oy};
    for(const view of model.reference_views||[]){const points=(view.ice_polygon_image||[]).map(p=>G.project(view.H_model_from_image,p)).filter(Boolean);if(points.length>=3){poly(ctx,points,transform);ctx.fillStyle="#327a7240";ctx.strokeStyle="#468d8477";ctx.fill();ctx.stroke();}}
    ctx.lineWidth=1;ctx.strokeStyle="#37555d";for(const line of G.gridLines(model)){ctx.beginPath();line.forEach((p,i)=>i?ctx.lineTo(...transform(p)):ctx.moveTo(...transform(p)));ctx.stroke();}
    if(validFrame){poly(ctx,validFrame.support_polygon_image,p=>{const q=G.project(validFrame.H_model_from_image,p);return q?transform(q):null;});ctx.fillStyle="#c2f3de20";ctx.strokeStyle=colors[validFrame.status]||colors.candidate;ctx.lineWidth=3;ctx.fill();ctx.stroke();}
    ctx.font="12px sans-serif";for(const landmark of model.landmarks||[]){const point=landmarkPoint(landmark);if(!point)continue;const q=transform(point);ctx.fillStyle="#ffd88a";if(landmark.model_xy){ctx.beginPath();ctx.arc(...q,4,0,Math.PI*2);ctx.fill();}ctx.fillText(G.landmarkLabel(landmark),q[0]+6,q[1]-7);}
    ctx.fillStyle="#a8c3c9";ctx.fillText("共同坐标 · 曲线编号位置不是材料点/圆心",16,canvas.height-12);
  }
  function draw(seconds){
    time=seconds;frame=G.frameAtTime(run?.frames||[],seconds,run||{});frameIndex=frame?run.frames.indexOf(frame):-1;
    let reason=G.drawable(frame,model,run,run?.source_sha256),status=frame?.status||"unavailable";
    if(!reason&&sourceDimensions(frame).some(v=>!Number.isFinite(v)||v<=0))reason="原图尺寸未登记，无法把代理像素当原图坐标绘网";
    const ctx=canvasContext();if(ctx)ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
    $("status").textContent=labels[reason&&status!=="ambiguous"?"unavailable":status]||labels.candidate;$("status").style.color=colors[reason&&status!=="ambiguous"?"unavailable":status]||colors.candidate;
    $("frameIdentity").textContent=frame?`源 ${fmt(frame.source_time_s)} s · 源帧 ${frame.source_frame_index??"未完整索引"} · PTS ${frame.source_pts??frame.pts} × ${frame.source_time_base??frame.time_base} · 镜头 ${frame.shot_id??"—"}`:`代理 ${fmt(seconds)} s · 此时间无定位记录`;
    $("reason").textContent=reason?G.reasonText(reason):`${G.reasonText(frame.reason)||"存在模型图像对应，独立几何未验证。"} 畸变未校正；人物排除仍为候选。锚 ${frame.anchor_id||"—"}。`;
    if(frameIndex>=0)$("frameSlider").value=frameIndex;
    $("previous").disabled=frameIndex<=0;$("next").disabled=frameIndex<0||frameIndex>=run.frames.length-1;
    if(ctx&&!reason){
      ctx.save();poly(ctx,frame.support_polygon_image);ctx.clip();G.clipForeground(ctx,frame);
      ctx.strokeStyle=colors[status]||colors.candidate;ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=Math.max(1.5,ctx.canvas.width/700);ctx.setLineDash(status==="observed"?[]:[ctx.canvas.width/130,ctx.canvas.width/210]);
      const inverse=G.invert(frame.H_model_from_image);
      if($("showGrid").checked)for(const line of G.gridLines(model))for(const part of G.projectedLine(inverse,line)){ctx.beginPath();part.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();}
      if($("showLandmarks").checked){ctx.font=`${Math.max(15,ctx.canvas.width/90)}px sans-serif`;for(const landmark of model.landmarks||[]){const p=G.project(inverse,landmarkPoint(landmark));if(p){if(landmark.model_xy){ctx.beginPath();ctx.arc(...p,ctx.canvas.width/300,0,Math.PI*2);ctx.fill();}ctx.fillText(G.landmarkLabel(landmark),p[0]+ctx.canvas.width/200,p[1]-ctx.canvas.width/200);}}}ctx.restore();
    }
    drawMap(reason?null:frame);
  }
  function select(){
    video.pause();presentedTime=null;run=data.runs[Number($("programmeSelect").value)];frame=null;frameIndex=-1;video.src=run.proxy_file;$("frameSlider").max=run.frames.length-1;$("frameSlider").value=0;
    const window=run.selected_window,counts={};for(const f of run.frames)counts[f.status]=(counts[f.status]||0)+1;
    $("windowSummary").textContent=`完整选定片段 ${run.frames.length} 帧 · ${fmt(run.frames[0].source_time_s)}–${fmt(run.frames.at(-1).source_time_s)} s`;
    $("runSummary").textContent=`${window.selection_basis}；${Object.entries(counts).map(([status,count])=>`${labels[status]||status} ${count}/${run.frames.length}`).join("，")}。${G.evidenceLabel(run.review_note||run.role)||"所有选定帧均保留，包括无法定位。"}`;
    $("firstCandidate").disabled=!run.frames.some(f=>f.status==="candidate"&&f.H_model_from_image);
    const strip=$("statusStrip"),ctx=strip.getContext("2d");for(let x=0;x<strip.width;x++){const f=run.frames[Math.min(run.frames.length-1,Math.floor(x*run.frames.length/strip.width))];ctx.fillStyle=colors[f.status]||colors.candidate;ctx.fillRect(x,0,1,strip.height);}
    draw(run.frames[0].media_time_s);
  }
  function seek(index){index=Math.max(0,Math.min(run.frames.length-1,index));video.pause();presentedTime=null;video.currentTime=G.seekFrameTime(run.frames,index,run);draw(run.frames[index].media_time_s);}
  function show(title,body){$("provenanceTitle").textContent=title;$("provenanceBody").innerHTML=body;$("provenance").showModal();}
  function showLandmark(index){const landmark=model.landmarks[index];show(G.landmarkLabel(landmark),`<p>模型 ${esc(model.model_id)} / ${esc(model.revision)} · ${esc(landmark.status||"候选")}</p><div class="references">${(landmark.observations||[]).map(imageHTML).join("")}</div><details><summary>完整来源与观测</summary><pre>${esc(JSON.stringify(landmark,null,2))}</pre></details>`);}
  $("landmarks").querySelectorAll("[data-landmark]").forEach(button=>button.onclick=()=>showLandmark(Number(button.dataset.landmark)));
  $("sourceChain").onclick=()=>{const views=(model.reference_views||[]).filter(v=>v.id===frame?.anchor_id||v.id===frame?.reference_view_id);show(frame?`源 ${fmt(frame.source_time_s)} s · PTS ${frame.pts} × ${frame.time_base}`:"当前时间无定位",`<p>使用地标：${esc((frame?.physical_ids||[]).join(" / ")||"见完整记录")}。有模型对应不代表独立精度通过。</p><p>以下为当前帧摘要；完整优化诊断与全部掩膜请打开 <a href="${esc(run.frames_file)}" target="_blank">完整逐帧定位记录</a>，按本帧 PTS 查找。人物排除配置保留在节目记录。</p><div class="references">${views.map(imageHTML).join("")}</div><pre>${esc(JSON.stringify(frame||{status:"unavailable",proxy_time_s:time},null,2))}</pre>`);};
  $("mapCanvas").onclick=event=>{if(!mapTransform)return;const r=event.currentTarget.getBoundingClientRect(),x=(event.clientX-r.left)*event.currentTarget.width/r.width,y=(event.clientY-r.top)*event.currentTarget.height/r.height,t=mapTransform;let found=-1,distance=30;(model.landmarks||[]).forEach((l,i)=>{const p=landmarkPoint(l);if(!p)return;const d=Math.hypot(t.ox+(p[0]-t.x0)*t.scale-x,t.oy+(p[1]-t.y0)*t.scale-y);if(d<distance){found=i;distance=d;}});if(found>=0)showLandmark(found);};
  $("closeProvenance").onclick=()=>$("provenance").close();$("programmeSelect").onchange=select;$("showGrid").onchange=()=>draw(time);$("showLandmarks").onchange=()=>draw(time);
  $("previous").onclick=()=>seek(frameIndex-1);$("next").onclick=()=>seek(frameIndex+1);$("frameSlider").oninput=()=>seek(Number($("frameSlider").value));
  $("firstCandidate").onclick=()=>{const i=run.frames.findIndex(f=>f.status==="candidate"&&f.H_model_from_image);if(i>=0)seek(i);};
  $("statusStrip").onclick=event=>{const r=event.currentTarget.getBoundingClientRect();seek(Math.floor((event.clientX-r.left)/r.width*run.frames.length));};
  const drawPlayback=()=>draw(G.presentedClockTime(video.currentTime,presentedTime,video.seeking,video.ended,run?.frames.at(-1)?.media_time_s));
  for(const name of["loadedmetadata","seeking"])video.addEventListener(name,()=>{presentedTime=null;drawPlayback();});
  for(const name of["seeked","pause","timeupdate","ended"])video.addEventListener(name,drawPlayback);
  if(video.requestVideoFrameCallback){const callback=(_,metadata)=>{presentedTime=metadata.mediaTime;drawPlayback();video.requestVideoFrameCallback(callback);};video.requestVideoFrameCallback(callback);}
  new ResizeObserver(()=>draw(time)).observe($("videoStage"));select();
})();
