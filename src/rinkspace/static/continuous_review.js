"use strict";
(() => {
  const data = JSON.parse(document.getElementById("review-data").textContent);
  const $ = id => document.getElementById(id);
  const video = $("decoder"), rows = data.rows, w = data.width, h = data.height;
  const original = $("original"), fixed = $("fixed"), fit = $("fit"), timeline = $("timeline");
  [original,fixed,fit].forEach(c => { c.width=w; c.height=h; });
  const ctx = original.getContext("2d"), fx = fixed.getContext("2d"), ft = fit.getContext("2d"), tx = timeline.getContext("2d");
  let index=0, desiredIndex=null, zoom=1, panX=0, panY=0, dragging=null;
  const useProjective=data.renderer==="projective_webgl_inverse_h";
  let projectiveRenderer=null, projectiveProblem=null;
  const lo=data.bounds[0], hi=data.bounds[1];
  const baseScale=Math.min(w/(hi[0]-lo[0]),h/(hi[1]-lo[1]));
  const baseX=(w-(hi[0]-lo[0])*baseScale)/2-lo[0]*baseScale;
  const baseY=(h-(hi[1]-lo[1])*baseScale)/2-lo[1]*baseScale;
  const f = (value,digits=2) => Number.isFinite(value) ? value.toFixed(digits) : "无观测";
  $("seek").max=rows.length-1;
  $("sourceTime").min=rows[0].t; $("sourceTime").max=rows.at(-1).t;
  $("intro").textContent=`${data.source} · ${rows.length} 个实际源帧 · ${f(rows[0].t)}–${f(rows.at(-1).t)} 秒 · 全部样本保留，按原始 PTS 播放`;
  $("provenance").textContent=`源文件 SHA256：${data.sha256}。样图 ${w}×${h}，源帧号与 PTS 来自实际解码索引。固定显示参照：样本 ${data.reference_sample} / 源 ${f(data.reference_time)} 秒；坐标为局部投影参照像素，无米制尺度。显示方式：${useProjective?"透视逆投影（WebGL）":"仿射映射（Canvas）"}。此 MP4 是从源采样 JPEG 编码的显示代理，原始视频文件未改写。`;
  if(data.audit.model_selection_independence_claimed===false)$("provenance").textContent+=" 本轮用途为开发诊断（"+(data.audit.evaluation_role||"development_diagnostic")+"）；模型开发已查看过这组误差，不宣称模型选择独立性，也不称为新的盲测。";
  data.audit.limitations.forEach(text=>{const item=document.createElement("li");item.textContent=text;$("auditLimitations").appendChild(item);});
  const s=data.audit.summary, raw=data.audit.raw_summary;
  if(!data.audit.present)$("auditNotice").textContent="连续显示候选；本次未附独立观测，精度未验证。所有时间点均保留，拟合残差不能替代独立误差。";
  $("summaryStatus").textContent=s ? `整段所有独立观测：补偿后中位 ${f(s.median_1920_px)} / P95 ${f(s.p95_1920_px)} px；未补偿 ${f(raw?.median_1920_px)} / ${f(raw?.p95_1920_px)} px。物理组 ${data.audit.physical_groups}，正式精度验收未通过。` : "本次没有附带独立轮廓观测，不能以拟合结果代替验收。";

  function path(context,points,close=true) {
    if (!points?.length) return;
    context.moveTo(points[0][0],points[0][1]);
    for (let j=1;j<points.length;j++) context.lineTo(points[j][0],points[j][1]);
    if(close) context.closePath();
  }
  function checker(context) {
    context.setTransform(1,0,0,1,0,0);
    context.fillStyle="#1a2229";context.fillRect(0,0,w,h);
    context.fillStyle="#222e37";
    for(let y=0;y<h;y+=20)for(let x=0;x<w;x+=20)if(((x/20+y/20)&1)===0)context.fillRect(x,y,20,20);
  }
  function overlays(context,row) {
    if(!$("showAudit").checked)return;
    context.save();context.lineWidth=1.2;
    context.strokeStyle="#41f2a5";context.beginPath();
    row.contours.forEach(p=>path(context,p,false));context.stroke();
    context.strokeStyle="rgba(255,145,90,.65)";context.lineWidth=.7;context.beginPath();
    row.checks.forEach((p,k)=>{if(k%6===0){context.moveTo(p[0],p[1]);context.lineTo(p[2],p[3]);}});context.stroke();
    context.fillStyle="#ff565e";
    row.checks.forEach(p=>context.fillRect(p[2]-1,p[3]-1,2,2));
    context.restore();
  }
  function clipDisplay(context,row) {
    context.beginPath();row.regions.forEach(p=>path(context,p));context.clip();
    // Subtract each polygon in a separate clip; overlapping exclusions must
    // stay excluded, not cancel one another through an even-odd XOR.
    row.display_exclude.forEach(p=>{context.beginPath();path(context,[[0,0],[w,0],[w,h],[0,h]]);path(context,p);context.clip("evenodd");});
  }
  function inverse3(a) {
    const [b,c,d]=a[0],[e,f,g]=a[1],[h,i,j]=a[2];
    const cof=[[f*j-g*i,d*i-c*j,c*g-d*f],[g*h-e*j,b*j-d*h,d*e-b*g],[e*i-f*h,c*h-b*i,b*f-c*e]];
    const det=b*cof[0][0]+c*cof[1][0]+d*cof[2][0];
    if(!Number.isFinite(det)||Math.abs(det)<1e-14)throw new Error("当前投影不可逆");
    return cof.map(r=>r.map(v=>v/det));
  }
  function initProjective() {
    const source=document.createElement("canvas");source.width=w;source.height=h;
    const context=source.getContext("2d"), canvas=document.createElement("canvas");canvas.width=w;canvas.height=h;
    const gl=canvas.getContext("webgl",{alpha:true,premultipliedAlpha:false,antialias:false,preserveDrawingBuffer:true});
    if(!gl)throw new Error("浏览器未提供 WebGL，透视画布不可显示");
    const compile=(kind,code)=>{const shader=gl.createShader(kind);gl.shaderSource(shader,code);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error("透视着色器编译失败");return shader;};
    const vertex=compile(gl.VERTEX_SHADER,"attribute vec2 a_position;void main(){gl_Position=vec4(a_position,0.0,1.0);}");
    const fragment=compile(gl.FRAGMENT_SHADER,`precision highp float;
      uniform sampler2D u_source;uniform mat3 u_inverse;uniform vec2 u_size;
      void main(){
        vec2 destination=vec2(gl_FragCoord.x,u_size.y-gl_FragCoord.y)-vec2(0.5);
        vec3 q=u_inverse*vec3(destination,1.0);
        if(abs(q.z)<0.00000001)discard;
        vec2 p=q.xy/q.z;
        if(p.x<0.0||p.y<0.0||p.x>u_size.x-1.0||p.y>u_size.y-1.0)discard;
        gl_FragColor=texture2D(u_source,(p+vec2(0.5))/u_size);
      }`);
    const program=gl.createProgram();gl.attachShader(program,vertex);gl.attachShader(program,fragment);gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error("透视着色器连接失败");
    gl.useProgram(program);
    const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
    const location=gl.getAttribLocation(program,"a_position");gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,2,gl.FLOAT,false,0,0);
    const texture=gl.createTexture();gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texture);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,false);
    gl.uniform1i(gl.getUniformLocation(program,"u_source"),0);gl.uniform2f(gl.getUniformLocation(program,"u_size"),w,h);
    const inverseLocation=gl.getUniformLocation(program,"u_inverse");gl.viewport(0,0,w,h);
    canvas.addEventListener("webglcontextlost",e=>{e.preventDefault();projectiveProblem="WebGL 上下文丢失，透视画布不可显示";render();});
    return {draw(row){
      if(gl.isContextLost())throw new Error("WebGL 上下文丢失");
      context.setTransform(1,0,0,1,0,0);context.clearRect(0,0,w,h);context.save();clipDisplay(context,row);
      context.drawImage(video,0,0,w,h);overlays(context,row);
      if(row.person){const [x0,y0,x1,y1]=row.person;context.strokeStyle="#72d8ea";context.lineWidth=1.2;context.strokeRect(x0,y0,x1-x0,y1-y0);}
      context.restore();
      const a=row.h,s=zoom*baseScale,dx=zoom*baseX+panX,dy=zoom*baseY+panY;
      const matrix=[a[0].map((v,k)=>s*v+dx*a[2][k]),a[1].map((v,k)=>s*v+dy*a[2][k]),a[2]];
      const inv=inverse3(matrix);
      gl.uniformMatrix3fv(inverseLocation,false,new Float32Array([inv[0][0],inv[1][0],inv[2][0],inv[0][1],inv[1][1],inv[2][1],inv[0][2],inv[1][2],inv[2][2]]));
      gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,source);
      gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
      if(gl.getError()!==gl.NO_ERROR)throw new Error("透视纹理渲染失败");
      fx.drawImage(canvas,0,0,w,h);
    }};
  }
  if(useProjective){try{projectiveRenderer=initProjective();}catch(error){projectiveProblem=error.message;}}
  function renderFixed(row) {
    checker(fx);
    if(!row.h) {
      fx.fillStyle="#ffd38a";fx.font="24px sans-serif";fx.fillText("本帧配准不可用 · 不沿用旧画面",40,h/2);return;
    }
    if(useProjective){
      if(!projectiveProblem){try{projectiveRenderer.draw(row);return;}catch(error){projectiveProblem=error.message;}}
      fx.fillStyle="#ffd38a";fx.font="21px sans-serif";fx.fillText("透视画布不可用 · 未使用仿射替代",30,h/2-18);fx.fillText(projectiveProblem,30,h/2+18);return;
    }
    fx.save();fx.setTransform(zoom,0,0,zoom,panX,panY);
    fx.transform(baseScale,0,0,baseScale,baseX,baseY);
    const a=row.h;fx.transform(a[0][0],a[1][0],a[0][1],a[1][1],a[0][2],a[1][2]);
    clipDisplay(fx,row);
    fx.drawImage(video,0,0,w,h);overlays(fx,row);
    if(row.person){const [x0,y0,x1,y1]=row.person;fx.strokeStyle="#72d8ea";fx.lineWidth=1/(baseScale*zoom);fx.strokeRect(x0,y0,x1-x0,y1-y0);}
    fx.restore();
  }
  function renderFit(row) {
    ft.setTransform(1,0,0,1,0,0);ft.drawImage(video,0,0,w,h);
    ft.fillStyle="rgba(5,13,22,.8)";ft.fillRect(0,0,w,h);
    ft.save();ft.beginPath();row.mask_paths.forEach(p=>path(ft,p));ft.clip("evenodd");ft.drawImage(video,0,0,w,h);ft.restore();
    ft.fillStyle="#ffac38";row.fit.forEach(p=>ft.fillRect(p[0]-1,p[1]-1,2,2));
    ft.beginPath();path(ft,row.support);ft.strokeStyle="#70e98b";ft.lineWidth=1.2;ft.stroke();
  }
  const plot={x:65,y:19,width:1195,height:178};
  const maxValue=Math.max(10,...rows.flatMap(r=>[r.p95,r.raw_p95]).filter(Number.isFinite));
  const ceiling=Math.ceil(maxValue/50)*50;
  const plotX=i=>plot.x+(rows[i].t-rows[0].t)/(rows.at(-1).t-rows[0].t||1)*plot.width;
  const plotY=v=>plot.y+plot.height-v/ceiling*plot.height;
  function renderTimeline() {
    tx.clearRect(0,0,timeline.width,timeline.height);tx.fillStyle="#14202b";tx.fillRect(0,0,1280,260);
    for(let j=0;j<rows.length;j++) {
      const r=rows[j];if(r.p95===null||!r.h){tx.fillStyle=!r.h?"#5b3c26":"#283440";const x=plotX(j);const next=j+1<rows.length?plotX(j+1):x+2;tx.fillRect(x,plot.y,Math.max(1,next-x),plot.height);}
    }
    tx.font="24px sans-serif";tx.lineWidth=1;
    for(let j=0;j<=4;j++){const value=ceiling*j/4,y=plotY(value);tx.strokeStyle="#334554";tx.beginPath();tx.moveTo(plot.x,y);tx.lineTo(plot.x+plot.width,y);tx.stroke();tx.fillStyle="#aebdcc";tx.fillText(value.toFixed(0),8,y+5);}
    tx.fillStyle="#aebdcc";tx.fillText("px",8,12);
    for(let j=0;j<=5;j++){const t=rows[0].t+(rows.at(-1).t-rows[0].t)*j/5;const x=plot.x+plot.width*j/5;tx.fillText(t.toFixed(2)+"s",Math.min(1195,x-18),229);}
    tx.setLineDash([6,4]);tx.strokeStyle="#c2a361";tx.beginPath();tx.moveTo(plot.x,plotY(10));tx.lineTo(plot.x+plot.width,plotY(10));tx.stroke();tx.setLineDash([]);
    for(const [key,color] of [["raw_p95","#79b8ff"],["p95","#ff8585"],["median","#61e3ad"]]){
      tx.beginPath();tx.strokeStyle=color;tx.lineWidth=2;let active=false;
      rows.forEach((r,j)=>{if(!Number.isFinite(r[key])){active=false;return;}const x=plotX(j),y=plotY(r[key]);if(active)tx.lineTo(x,y);else tx.moveTo(x,y);active=true;});tx.stroke();
      // Sparse checked frames remain individual observations, never joined
      // across unobserved times; draw isolated points so none disappear.
      tx.fillStyle=color;
      rows.forEach((r,j)=>{if(Number.isFinite(r[key])&&!Number.isFinite(rows[j-1]?.[key])&&!Number.isFinite(rows[j+1]?.[key])){tx.beginPath();tx.arc(plotX(j),plotY(r[key]),4,0,2*Math.PI);tx.fill();}});
    }
    rows.forEach((r,j)=>{if(r.ref||r.audit_ref){tx.fillStyle=r.ref?"#f4ce80":"#c4a4fa";tx.fillRect(plotX(j)-1,plot.y,2,plot.height);}});
    tx.strokeStyle="#fff";tx.lineWidth=1.5;tx.beginPath();tx.moveTo(plotX(index),plot.y);tx.lineTo(plotX(index),plot.y+plot.height);tx.stroke();
    tx.fillStyle="#e1e9f1";tx.fillText(`当前 ${f(rows[index].t)} s`, Math.min(1100,Math.max(65,plotX(index)-60)),252);
  }
  function render() {
    const r=rows[index];if(video.readyState>=2){ctx.setTransform(1,0,0,1,0,0);ctx.drawImage(video,0,0,w,h);overlays(ctx,r);renderFixed(r);renderFit(r);}
    $("seek").value=index;if(document.activeElement!==$("sourceTime"))$("sourceTime").value=f(r.t);
    $("frameStatus").textContent=`${index+1}/${rows.length} · 源 ${f(r.t)} s · 帧 ${r.frame}`;
    $("sourceStatus").textContent=`PTS ${r.pts} × ${r.tb} · ${r.checks.length?"独立组形状检查":"独立轮廓未观测／未标注，不填补观测"}`;
    $("fixedStatus").textContent=`${projectiveProblem?"显示不可用 / "+projectiveProblem:!r.h?"不可用 / 明确断链":r.ref?"首个参考帧 / 恒等，不计成功":"候选变换 / 未通过独立验收"} · 缩放 ${Math.round(zoom*100)}% · 参照 ${f(data.reference_time)} s${useProjective?" · 透视逆投影":""}`;
    $("errorStatus").textContent=Number.isFinite(r.p95)?`${r.audit_ref?"形状参考帧自比 · ":""}当前中位 ${f(r.median)} / P95 ${f(r.p95)} px；未补偿 ${f(r.raw_median)} / ${f(r.raw_p95)} px` : "当前没有可用的独立轮廓观测，误差未知。";
    $("errorStatus").className="quality "+(Number.isFinite(r.p95)&&r.p95>10?"red":"warn");
    renderTimeline();
    // Read-only QA state exposes exactly the displayed source identity.
    document.body.dataset.sampleIndex=String(r.i);document.body.dataset.sourcePts=String(r.pts);document.body.dataset.zoom=String(zoom);document.body.dataset.renderer=data.renderer||"affine_canvas_2d";document.body.dataset.renderProblem=projectiveProblem||"";
  }
  function nearest(time) {let lo=0,hi=rows.length-1;while(lo<hi){const m=Math.floor((lo+hi)/2);if(rows[m].dt<time)lo=m+1;else hi=m;}if(lo>0&&Math.abs(rows[lo-1].dt-time)<Math.abs(rows[lo].dt-time))lo--;return lo;}
  function seekTo(i) {video.pause();i=Math.max(0,Math.min(rows.length-1,i));desiredIndex=i;index=i;video.currentTime=rows[i].dt;renderTimeline();if(video.readyState>=2&&Math.abs(video.currentTime-rows[i].dt)<1e-8&&!video.seeking){render();desiredIndex=null;}}
  function onFrame(_now,metadata) {
    const decodedIndex=nearest(metadata.mediaTime);
    // A seek may still deliver one callback for the previous decoded picture.
    // Never paint that picture with the newly requested frame's geometry.
    if(desiredIndex===null||decodedIndex===desiredIndex){index=decodedIndex;render();}
    video.requestVideoFrameCallback(onFrame);
  }
  if("requestVideoFrameCallback" in video)video.requestVideoFrameCallback(onFrame);
  else {const tick=()=>{if(!video.paused&&desiredIndex===null){index=nearest(video.currentTime);render();}requestAnimationFrame(tick);};requestAnimationFrame(tick);}
  video.addEventListener("loadeddata",()=>{index=0;render();});
  video.addEventListener("seeked",()=>{index=desiredIndex??nearest(video.currentTime);desiredIndex=null;render();});
  video.addEventListener("play",()=>{$("play").textContent="暂停";});
  video.addEventListener("pause",()=>{$("play").textContent="播放";});
  video.addEventListener("ended",()=>{index=rows.length-1;render();});
  video.addEventListener("error",()=>{$("sourceStatus").textContent="源片段加载失败；请把 source.mp4 与 HTML 保持在同一文件夹。";});
  $("play").onclick=()=>{if(video.paused){if(video.ended)seekTo(0);video.play().catch(e=>{$("sourceStatus").textContent="播放失败："+e.message;});}else video.pause();};
  $("previous").onclick=()=>seekTo(index-1);$("next").onclick=()=>seekTo(index+1);
  $("seek").oninput=()=>seekTo(Number($("seek").value));
  $("go").onclick=()=>seekTo(nearest(Number($("sourceTime").value)-rows[0].t));
  $("sourceTime").onkeydown=e=>{if(e.key==="Enter")$("go").click();};
  $("showAudit").onchange=render;
  function zoomAt(factor,x=w/2,y=h/2){const old=zoom;zoom=Math.max(.25,Math.min(12,zoom*factor));panX=x-(x-panX)*zoom/old;panY=y-(y-panY)*zoom/old;render();}
  $("zoomIn").onclick=()=>zoomAt(1.3);$("zoomOut").onclick=()=>zoomAt(1/1.3);
  $("reset").onclick=()=>{zoom=1;panX=0;panY=0;render();};
  fixed.addEventListener("wheel",e=>{e.preventDefault();const b=fixed.getBoundingClientRect();zoomAt(Math.exp(-e.deltaY*.0015),(e.clientX-b.left)*w/b.width,(e.clientY-b.top)*h/b.height);},{passive:false});
  fixed.addEventListener("pointerdown",e=>{dragging={x:e.clientX,y:e.clientY,panX,panY};fixed.setPointerCapture(e.pointerId);});
  fixed.addEventListener("pointermove",e=>{if(!dragging)return;const b=fixed.getBoundingClientRect();panX=dragging.panX+(e.clientX-dragging.x)*w/b.width;panY=dragging.panY+(e.clientY-dragging.y)*h/b.height;render();});
  fixed.addEventListener("pointerup",()=>{dragging=null;});fixed.addEventListener("pointercancel",()=>{dragging=null;});
  timeline.onclick=e=>{const b=timeline.getBoundingClientRect(),x=(e.clientX-b.left)*timeline.width/b.width;const t=rows[0].t+Math.max(0,Math.min(1,(x-plot.x)/plot.width))*(rows.at(-1).t-rows[0].t);seekTo(nearest(t-rows[0].t));};
  document.addEventListener("keydown",e=>{if(["INPUT","BUTTON"].includes(document.activeElement?.tagName))return;if(e.code==="Space"){e.preventDefault();$("play").click();}else if(e.code==="ArrowRight"){e.preventDefault();seekTo(index+1);}else if(e.code==="ArrowLeft"){e.preventDefault();seekTo(index-1);}});
  render();
})();
