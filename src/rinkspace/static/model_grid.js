/* Shared-model display. Matrices and support stay in original image pixels;
   CSS, letterboxing and device pixel ratio never alter the saved geometry. */
(function () {
  "use strict";
  const isNumber = value => typeof value === "number" && Number.isFinite(value);
  const statusText={observed:"定位有证据",anchored:"定位有证据",model_observed:"定位有证据",predicted:"连续推算",propagated:"连续推算",candidate:"待验证候选",model_reobserved:"认出场地",image_tracked:"持续跟踪",motion_predicted:"暂时推算",unavailable:"暂时无法定位",unlocalized:"暂时无法定位",missing:"暂时无法定位",ambiguous:"存在冲突"};
  const statusColor={observed:"#80edd4",anchored:"#80edd4",model_observed:"#80edd4",predicted:"#9dc7ff",propagated:"#9dc7ff",candidate:"#ffce7a",model_reobserved:"#80edd4",image_tracked:"#9dc7ff",motion_predicted:"#ffce7a",ambiguous:"#ff91b6"};
  const statusLabel=(status,run={})=>(run?.landmark_mode?{model_reobserved:"地标定位候选",image_tracked:"沿图像连续推算"}[status]:null)||statusText[status]||"待验证候选";
  const runModeLabel=run=>String(run?.landmark_mode||"").startsWith("manual_")?"人工干预对照":"";
  const runDisplayLabel=run=>[run?.label||run?.run_id||"",runModeLabel(run)].filter(Boolean).join(" · ");
  const displayStatusState=(status,run={})=>run?.landmark_mode&&status==="model_reobserved"?"candidate":status||"candidate";
  function continuitySummary(frame) {
    if(!frame)return "";
    const parts=[],uncertainty=frame.uncertainty,jointUncertainty=frame.camera_uncertainty;
    if(isNumber(frame.last_image_observation_age_s))parts.push(`距最近图像观测 ${frame.last_image_observation_age_s.toFixed(2)} 秒`);
    if(isNumber(jointUncertainty?.engineering_range_sample_px))parts.push(`工程误差估计 ${jointUncertainty.engineering_range_sample_px.toFixed(1)} 样图像素（非统计置信界）`);
    else if(isNumber(uncertainty?.estimated_error_px_at_960))parts.push(`工程误差估计 ${uncertainty.estimated_error_px_at_960.toFixed(1)} 像素（960宽，非统计置信界）`);
    if(frame.future_anchor_used)parts.push("使用了后续画面的场地观测");
    const validated=frame.independent_validation_passed===true||frame.independent_geometry_validated===true||frame.independent_validation?.passed===true;
    parts.push(validated?"独立几何检查通过（仅登记范围）":"尚未独立验准");
    return parts.join(" · ");
  }
  function invert(matrix) {
    if (!Array.isArray(matrix) || matrix.length !== 3 || matrix.some(row => !Array.isArray(row) || row.length !== 3 || row.some(v => !isNumber(v)))) return null;
    const [a,b,c,d,e,f,g,h,i] = matrix.flat();
    const det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
    if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;
    return [[e*i-f*h,c*h-b*i,b*f-c*e],[f*g-d*i,a*i-c*g,c*d-a*f],[d*h-e*g,b*g-a*h,a*e-b*d]].map(row => row.map(v => v/det));
  }
  function project(H, point) {
    if (!H || !Array.isArray(point) || point.length < 2 || !point.slice(0,2).every(isNumber)) return null;
    const [x,y] = point, z = H[2][0]*x+H[2][1]*y+H[2][2];
    if (!Number.isFinite(z) || Math.abs(z) < 1e-8) return null;
    const result = [(H[0][0]*x+H[0][1]*y+H[0][2])/z,(H[1][0]*x+H[1][1]*y+H[1][2])/z];
    return result.every(isNumber) ? result : null;
  }
  const frameTime = (frame, run) => isNumber(frame.media_time_s) ? frame.media_time_s : frame.source_time_s-(run.media_time_offset_s || 0);
  function frameDuration(frames,index,run={}) {
    const frame=frames[index],durations=[frame?.source_duration_s,run.frame_duration_s].filter(v=>isNumber(v)&&v>0);
    for(const neighbor of [frames[index-1],frames[index+1]])if(neighbor){const dt=Math.abs(frameTime(neighbor,run)-frameTime(frame,run));if(dt>0&&dt<=.1)durations.push(dt);}
    return Math.min(.1,durations.length?Math.min(...durations):.02);
  }
  function frameAtTime(frames, seconds, run = {}) {
    if (!frames.length || !isNumber(seconds)) return null;
    let low=0, high=frames.length;
    while (low<high) {const mid=(low+high)>>1; if (frameTime(frames[mid],run)<=seconds+1e-7) low=mid+1; else high=mid;}
    const index=low-1,best=frames[index];if(!best)return null;
    // A paused media clock can lie between PTS values. Keep the containing
    // displayed frame rather than selecting the not-yet-displayed next frame.
    // Explicitly bounded duration still leaves sparse/unrecorded gaps blank.
    return seconds-frameTime(best,run)<frameDuration(frames,index,run)+.0005 ? best : null;
  }
  function seekFrameTime(frames,index,run={}) {
    if(!frames[index])return null;
    // Seek just inside the selected frame, avoiding decoder boundary rounding.
    return frameTime(frames[index],run)+Math.min(.001,frameDuration(frames,index,run)/4);
  }
  function presentedClockTime(currentTime,presentedTime,seeking=false,ended=false,lastFrameTime=null) {
    if(ended&&isNumber(lastFrameTime))return lastFrameTime;
    return !seeking&&isNumber(presentedTime)?presentedTime:currentTime;
  }
  function reasonText(reason) {
    if(!reason)return "";
    const labels={
      no_compatible_camera_reference:"尚无经冰面检查支持的机位参照，当前帧不画全场网格。",
      reference_library_retrieval:"从赛事参照库匹配机位，并重新求本帧朝向和变焦；仍是待验准候选。",
      observed_LK_rotation_zoom:"共享安装参数，以当前静止背景更新朝向和变焦；来源和冰面检查见记录。",
      no_model_ice_anchor:"当前画面没有找到足够的冰面图案对应，暂时无法定位到共同模型。",
      typed_surfaces_shared_camera_candidate:"冰面、接地边候选与板面分别提供观测，共同估计这一镜头的相机转向和变焦；仍需检查冰面精度。",
      typed_ice_contact_camera_candidate:"使用实际冰面观测和接地边候选估计相机；板面没有参与这一对照。",
      reference_camera_refitted_fixed_geometry:"场地结构保持不变，已重新检查参考帧相机；当前是待验准候选。",
      no_fixed_model_ice_reobservation:"当前画面未能重新对应到整场草模的固定冰面参考，暂时无法定位。",
      fixed_model_reference_ice_reobserved:"已在固定模型参考图中重新观测到真实冰面对应，仍是条件草模候选。",
      model_anchor_conflict:"不同场地定位解释相互冲突，暂停叠线并切断传播；保留前后观测供核对。",
      observed_ice_chain_since_model_reobservation:"用当前画面的静止冰面继续跟踪，位置仍连接到已识别的场地。",
      accumulated_uncertainty_exceeded:"连续传播的工程误差估计已过大，暂停叠线，等待新的场地观测。",
      local_observation_lost:"新的静止冰面对应丢失，不能继续把相机移动当作已观测。",
      unanchored_local_component:"已跟踪这一段画面的相对移动，但尚未确定它在冰场的哪里。",
      cut_waiting_for_relocalization:"刚切换机位，正在重新确定相机与同一冰场的关系。",
      local_component_without_model_anchor:"已跟踪这一段画面的相对移动，但尚未确定它在冰场的哪里。",
      accumulated_image_path_uncertainty_exceeded:"图像观测链越传越远，工程误差估计已过大，暂停叠线。",
      joint_camera_uncertainty_exceeded:"前后观测联合约束后的工程误差估计仍过大，暂停叠线。",
      offline_joint_actual_image_edges_and_model_anchors:"使用前后场地观测和中间实际冰面跟踪，共同约束这一镜头的位置。",
      forward_new_model_anchor_and_actual_path_recovers_after_conflict:"在没有跨过冲突的这一段内，已重新连接真实场地观测并持续跟踪；仍是待验准候选。",
      reverse_actual_ice_path_to_model_anchor_with_local_consistent_overlap:"从后面的真实场地观测倒序跟踪到此帧，并检查了与正向结果的一致区间；仍需验准。",
      directional_camera_conflict_barrier:"前后图像观测给出的场地位置存在冲突，暂停叠线。",
      unresolved_directional_conflict_requires_new_model_anchor:"此前的定位冲突尚未解决；另一条跟踪暂时失踪不能作为恢复网格的依据。",
      unavailable_after_conflict_without_new_anchor_and_observed_path:"冲突后还没有新的场地观测和可接通的图像链，暂时无法定位。",
      known_anchor_conflict_no_real_anchor_in_surviving_observed_segment:"已知场地解释存在冲突，这一段缺少不跨冲突的新场地观测，继续暂停叠线。",
      unresolved_model_anchor_conflict_over_observed_component:"同一段镜头仍存在相互冲突的场地解释，保留多个解释并暂停叠线。",
      connected_actual_image_observations_from_real_model_anchor:"沿真实冰面观测连续跟踪，位置来源可以追溯到场地参考。",
      bounded_ice_motion_since_model_reobservation:"从最近固定模型参考的冰面重观测短时传播，仍需检查漂移。",
      bounded_reverse_ice_motion_from_future_model_reobservation:"离线使用后面出现的真实模型锚向前短时推算；使用了未来帧，尚未独立验证。",
      direct_fixed_model_print_reobservation_geometry_candidate:"已重新匹配共同模型的冰面图案，当前仍是待验证候选。",
      bounded_local_ice_propagation_from_model_anchor:"从最近的模型锚沿冰面跟踪连续推算，尚未独立验证精度。",
      bounded_masked_ice_photometric_propagation_from_model_anchor:"从最近的模型锚按冰面纹理连续推算，尚未独立验证精度。",
      insufficient_bidirectional_mask_overlap:"可用冰面重叠范围不足，暂时无法定位。",
      no_safe_camera_proposal:"尚未找到可靠的相同机位候选，暂时无法定位。",
      no_model_correspondence:"没有找到与共同模型对应的冰面图案。",
      insufficient_spatial_support:"冰面对应集中在过小区域，不能支撑当前网格。",
      no_global_model_anchor:"缺少共同模型锚，局部跟踪不能单独提供全场位置。",
      anchor_ablation:"模型锚已移除，局部跟踪不能单独提供全场位置。",
      camera_cut:"已切换机位，需要重新定位到共同模型。",
      unavailable:"当前帧无法定位。",ambiguous:"存在多个未能区分的定位候选。"
    };
    if(labels[reason])return labels[reason];
    return /[\u3400-\u9fff]/.test(String(reason))?String(reason):"当前帧未满足模型定位条件，具体原因见来源诊断。";
  }
  function landmarkLabel(landmark) {
    const labels={R1_gray_near:"R1 近侧灰靶",R2_red_middle:"R2 中部红靶",R3_gray_middle:"R3 中部灰靶",R4_red_far:"R4 远侧红靶"};
    const id=landmark?.physical_id||"";
    return labels[id]||(landmark?.label&&landmark.label!==id?landmark.label:`地标 ${id}`);
  }
  function evidenceLabel(value) {
    return ({direct_image_candidate_geometry:"实拍图像支持的候选几何，尚未独立验证",
      propagated_prior_unknown_cumulative_drift:"由相邻参考帧传播的候选范围，累计漂移尚未验证",
      new_source_inference_after_freeze:"模型冻结后的新来源推理"})[value]||value||"";
  }
  const naturalOrder=(a,b)=>String(a??"").localeCompare(String(b??""),"en",{numeric:true,sensitivity:"base"});
  function defaultModelIndex(models,explicit=null) {
    const retained=explicit?models.findIndex(m=>m.model_id===explicit.model_id&&String(m.revision)===String(explicit.revision)):-1;
    if(retained>=0)return retained;
    const preferred=models.findIndex(m=>m.preferred_delivery===true);if(preferred>=0)return preferred;
    return models.map((model,index)=>({model,index})).sort((a,b)=>naturalOrder(b.model.revision,a.model.revision)||naturalOrder(a.model.model_id,b.model.model_id))[0]?.index??-1;
  }
  function compareRuns(a,b) {
    const diagnostic=r=>Number(r.anchor_ablation===true||r.diagnostic_preview_only===true||r.experimental===true||["ablation","anchor_ablation","diagnostic","experimental","historical"].includes(r.role));
    const complete=r=>Number(r.all_selected_frames_in_denominator===true);
    const cadence=r=>Number(isNumber(r.reobserve_every_source_frames)&&r.reobserve_every_source_frames>0);
    const stamp=r=>{for(const value of[r.updated_at,r.created_at,r.modified_at,r.mtime]){const n=typeof value==="number"?value:Date.parse(value);if(Number.isFinite(n))return n;}return 0;};
    const start=r=>isNumber(r.start_s)?r.start_s:isNumber(r.selected_window?.start_s)?r.selected_window.start_s:-Infinity;
    // Use declared run metadata, never success counts or a programme filename.
    // Current catalog records do not contain creation times; their explicit
    // reobservation cadence distinguishes current runs from older experiments.
    return diagnostic(a)-diagnostic(b)||complete(b)-complete(a)||naturalOrder(b.algorithm_version,a.algorithm_version)
      ||cadence(b)-cadence(a)||stamp(b)-stamp(a)||start(b)-start(a)||naturalOrder(b.run_id,a.run_id);
  }
  function defaultRunId(runs,explicit=null) {
    if(explicit&&runs.some(r=>r.run_id===explicit))return explicit;
    return [...runs].sort(compareRuns)[0]?.run_id??"";
  }
  function drawable(frame, model, run, sourceHash) {
    if (!frame || !model || !run) return "当前时间没有模型定位";
    if (String(frame.model_id ?? run.model_id)!==String(model.model_id) || String(frame.model_revision ?? frame.revision ?? run.model_revision ?? run.revision)!==String(model.revision)) return "模型 ID / 版本不匹配";
    if (sourceHash && (frame.source_sha256 || run.source_sha256) && sourceHash!==(frame.source_sha256 || run.source_sha256)) return "原片 SHA-256 与定位记录不匹配";
    if (["unavailable","unlocalized","missing","failed","rejected","ambiguous"].includes(frame.status)) return frame.reason || "此帧无法定位";
    if(model.structure&&frame.projection_chain){
      const c=frame.projection_chain,d=c.reference_radial_distortion;
      if(![c.H_reference_undistorted_from_model,c.H_current_sample_from_reference_raw,c.H_source_image_from_sample].every(invert))return "整场投影链缺少有效矩阵";
      if(![-1,1].includes(c.reference_visible_branch_sign??1))return "参考图可见投影分支无效";
      if(![-1,1].includes(c.raw_transfer_visible_branch_sign??1))return "原始像素传递的可见分支无效";
      if(d&&(!isNumber(d.k1)||!isNumber(d.scale_image)||d.scale_image<=0||!Array.isArray(d.centre_image)||d.centre_image.length!==2||!d.centre_image.every(isNumber)))return "候选畸变参数无效";
      const validPolygon=p=>Array.isArray(p)&&p.length>=3&&p.every(q=>Array.isArray(q)&&q.length===2&&q.every(isNumber));
      if(!Array.isArray(frame.ice_polygons_sample)||!frame.ice_polygons_sample.length||!frame.ice_polygons_sample.every(validPolygon))return "没有可用的整幅冰面遮罩";
      return null;
    }
    if (!invert(frame.H_model_from_image)) return "此帧无有效模型矩阵";
    if (!Array.isArray(frame.support_polygon_image) || frame.support_polygon_image.length<3 || frame.support_polygon_image.some(p=>!Array.isArray(p)||p.length!==2||!p.every(isNumber))) return "此帧没有有效冰面支持范围";
    if (frame.matrix_direction && frame.matrix_direction!=="H_model_from_image") return "矩阵方向不是原图到模型";
    if (frame.image_domain && !["original_image_px","image_px","raw_image_px","raw_approximate","original_distorted_approximation","raw_px_distortion_unchecked"].includes(frame.image_domain)) return "镜头坐标域尚无原片投影接线";
    return null;
  }
  function gridLines(model) {
    if (Array.isArray(model.grid_lines_model)) return model.grid_lines_model;
    const grid=model.grid || {}, bounds=grid.bounds;
    if (!Array.isArray(bounds) || bounds.length!==4 || !bounds.every(isNumber)) return [];
    const [x0,y0,x1,y1]=bounds;
    return [...(grid.x_values||[]).filter(isNumber).slice(0,300).map(x=>[[x,y0],[x,y1]]),...(grid.y_values||[]).filter(isNumber).slice(0,300).map(y=>[[x0,y],[x1,y]])];
  }
  function projectedLine(H,line) {
    const segments=[];let part=[],lastSign=null;
    for(let n=1;n<line.length;n++){
      const a=line[n-1],b=line[n];if(!Array.isArray(a)||!Array.isArray(b)||!a.every(isNumber)||!b.every(isNumber))continue;
      for(let k=0;k<=32;k++){
        const t=k/32,p=[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t],z=H[2][0]*p[0]+H[2][1]*p[1]+H[2][2],q=project(H,p),sign=Math.sign(z);
        if(!q || lastSign!==null&&sign!==lastSign){if(part.length>1)segments.push(part);part=[];}
        if(q)part.push(q);lastSign=q?sign:null;
      }
    }
    if(part.length>1)segments.push(part);return segments;
  }
  function structureObjects(model) {
    return model?.structure?[model.structure.boundary,...(model.structure.primitives||[])].filter(Boolean):[];
  }
  function modelLandmarks(model) {
    if(!model?.structure)return model?.landmarks||[];
    return structureObjects(model).map(p=>({...p,physical_id:p.id,label:p.display_label||p.label||p.id,
      curve_model:p.points_model,observations:(model.view_poses||[]).filter(v=>(p.source_view_ids||[]).includes(v.id))}));
  }
  function wholeProjectedLine(chain,line,closed=false) {
    if(!chain||!Array.isArray(line)||line.length<2)return [];
    const points=closed?[...line,line[0]]:line,segments=[];let part=[],lastSign=null;
    for(let n=1;n<points.length;n++)for(let k=0;k<=32;k++){
      const a=points[n-1],b=points[n],t=k/32,p=[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t];
      const h=chain.H_reference_undistorted_from_model,z=h[2][0]*p[0]+h[2][1]*p[1]+h[2][2];
      let q=(chain.reference_visible_branch_sign??1)*z>1e-8?project(h,p):null,sign=null;
      if(q&&chain.reference_radial_distortion){const d=chain.reference_radial_distortion,dx=q[0]-d.centre_image[0],dy=q[1]-d.centre_image[1],r2=(dx*dx+dy*dy)/(d.scale_image*d.scale_image),s=1+d.k1*r2;q=1+3*d.k1*r2>0?[d.centre_image[0]+dx*s,d.centre_image[1]+dy*s]:null;}
      if(q){const m=chain.H_current_sample_from_reference_raw,d=m[2][0]*q[0]+m[2][1]*q[1]+m[2][2];sign=Math.sign(d);q=(chain.raw_transfer_visible_branch_sign??1)*d>1e-8?project(m,q):null;if(q)q=project(chain.H_source_image_from_sample,q);}
      if(!q||lastSign!==null&&sign!==lastSign){if(part.length>1)segments.push(part);part=[];}
      if(q&&q.every(isNumber))part.push(q);lastSign=q?sign:null;
    }
    if(part.length>1)segments.push(part);return segments;
  }
  function drawWholeOverlay(ctx,model,frame,{geometry=true,structure=geometry,grid=geometry,labels=false}={}) {
    const chain=frame.projection_chain,S=chain.H_source_image_from_sample;
    ctx.save();ctx.beginPath();for(const polygon of frame.ice_polygons_sample||[]){polygon.forEach((p,i)=>{const q=project(S,p);if(q)i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.closePath();}ctx.clip();
    clipForeground(ctx,{foreground_polygons_image:(frame.foreground_polygons_sample||[]).map(p=>p.map(q=>project(S,q)).filter(Boolean))});
    const width=ctx.canvas.width;
    const all=[...(structure||labels?structureObjects(model):[]),...(grid?gridLines({...model,grid:{...(model.grid||{}),bounds:model.structure.bounds}}).map(points_model=>({points_model,status:"assumed",kind:"line",display_grid:true})):[])];
    for(const primitive of all){if(!Array.isArray(primitive.points_model))continue;ctx.strokeStyle=primitive.display_grid?"#a7c9ea":"#ff77d9";ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=Math.max(1.5,width/650);ctx.setLineDash(primitive.display_grid?[width/130,width/210]:[width/240,width/160]);
      const closed=primitive===model.structure.boundary||primitive.closed||["circle","ring","polygon","closed_curve","rounded_boundary"].includes(primitive.kind);
      const segments=wholeProjectedLine(chain,primitive.points_model,closed);
      if(primitive.display_grid?grid:structure)for(const segment of segments){ctx.beginPath();segment.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();}
      if(labels&&primitive.id&&segments.length){const p=segments[0][Math.floor(segments[0].length/2)];if(p&&p[0]>=0&&p[0]<width&&p[1]>=0&&p[1]<ctx.canvas.height){ctx.font=`${Math.max(14,width/95)}px sans-serif`;ctx.fillText(primitive.display_label||primitive.label||primitive.id,p[0]+6,p[1]-6);}}
    }ctx.restore();
  }
  function landmarkPoint(l) {
    if(Array.isArray(l.model_xy)&&l.model_xy.length===2&&l.model_xy.every(isNumber))return l.model_xy;
    const p=l.polygon_model||l.curve_model;
    if(Array.isArray(p)&&p.length&&p.every(q=>Array.isArray(q)&&q.length===2&&q.every(isNumber)))return p.reduce((a,q)=>[a[0]+q[0]/p.length,a[1]+q[1]/p.length],[0,0]);
    return null;
  }
  function clipForeground(ctx,frame) {
    // Subtract each region separately: one evenodd path containing overlapping
    // boxes would reopen their intersection and draw on a person.
    for(const b of frame.person_boxes_image||[]){if(Array.isArray(b)&&b.length===4&&b.every(isNumber)){ctx.beginPath();ctx.rect(0,0,ctx.canvas.width,ctx.canvas.height);ctx.rect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.clip("evenodd");}}
    for(const points of frame.foreground_polygons_image||[]){if(!Array.isArray(points)||points.length<3||points.some(p=>!Array.isArray(p)||p.length!==2||!p.every(isNumber)))continue;ctx.beginPath();ctx.rect(0,0,ctx.canvas.width,ctx.canvas.height);points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.clip("evenodd");}
  }
  function drawSceneObservations(ctx,frame,enabled={}) {
    const channels=frame?.scene_observations;if(!channels)return;
    const colors={ice_marking:"#ff50cb",ice_board_contact:"#ffec33",board_surface:"#7dff69"};
    const sx=ctx.canvas.width/(frame.width||1920),sy=ctx.canvas.height/(frame.height||1080);
    ctx.save();ctx.lineWidth=Math.max(1.3,ctx.canvas.width/1100);ctx.setLineDash([]);
    for(const [name,color] of Object.entries(colors)){
      if(!enabled[name])continue;const channel=channels[name];if(!channel)continue;
      ctx.strokeStyle=color;ctx.fillStyle=color;
      for(const p of channel.points_source_px||[]){if(!p.every(isNumber))continue;ctx.beginPath();ctx.arc(p[0]*sx,p[1]*sy,Math.max(1.4,ctx.canvas.width/800),0,Math.PI*2);ctx.fill();}
      for(const p of channel.direct_reference_points_source_px||[]){if(!p.every(isNumber))continue;ctx.strokeRect(p[0]*sx-3,p[1]*sy-3,6,6);}
      for(const p of channel.used_points_source_px||[]){if(!p.every(isNumber))continue;ctx.beginPath();ctx.arc(p[0]*sx,p[1]*sy,4,0,Math.PI*2);ctx.stroke();}
      for(const item of channel.identified_line_candidates||[]){ctx.lineWidth=Math.max(2,ctx.canvas.width/600);ctx.beginPath();item.points_source_px.forEach((p,i)=>i?ctx.lineTo(p[0]*sx,p[1]*sy):ctx.moveTo(p[0]*sx,p[1]*sy));ctx.stroke();ctx.lineWidth=Math.max(1.3,ctx.canvas.width/1100);}
      for(const line of channel.lines_source_px||[]){ctx.beginPath();line.forEach((p,i)=>i?ctx.lineTo(p[0]*sx,p[1]*sy):ctx.moveTo(p[0]*sx,p[1]*sy));ctx.stroke();}
    }
    ctx.restore();
  }
  function landmarkObservationState(observation) {
    const status=observation?.association_status;
    if(status==="conflict"||status==="geometry_conflict")return "conflict";
    if(status==="masked_out"||observation?.rejection_stage==="mask")return "masked_out";
    if(status==="not_detected")return "not_detected";
    if(observation?.used_by_solver===true)return "used";
    if(observation?.selected_model_id)return "associated";
    return "unknown";
  }
  function landmarkObservationStatus(observation) {
    return ({conflict:"已对应但几何冲突",masked_out:"被遮罩删掉",not_detected:"没提取出来",used:"已参与求解，尚未独立验准",associated:"已对应，未参与求解",unknown:"看见但不知是哪个"})[landmarkObservationState(observation)];
  }
  function landmarkConfirmationLabel(row) {
    if(row.user_confirmed===true)return "用户确认";
    if(row.manual_confirmed||row.not_automatic||row.confirmation_actor==="agent_source_reading_not_user_confirmation")return "人工读点候选";
    return "";
  }
  function landmarkRows(frame) {return Array.isArray(frame?.landmark_observations)?frame.landmark_observations:[];}
  function observationBindingReason(observation,frame,run={},sourceHash=null) {
    if(!frame)return "当前没有源帧记录";
    const hashes=[sourceHash,run.source_sha256,frame.source_sha256,observation?.source_sha256].filter(Boolean);
    if(new Set(hashes).size>1)return "地标观测与当前原片不匹配";
    const actualPts=observation?.source_pts??observation?.pts,currentPts=frame.source_pts??frame.pts;
    const actualBase=observation?.source_time_base??observation?.time_base,currentBase=frame.source_time_base??frame.time_base;
    // Integer PTS and its declared time base are identity, not a nearby-time hint.
    if(actualPts!=null&&actualPts!==currentPts)return "地标观测与当前源PTS不匹配";
    if(actualBase!=null&&actualBase!==currentBase)return "地标观测与当前源时基不匹配";
    return null;
  }
  function observedLandmarkSegments(observation,frame) {
    let segments=observation.visible_segments_source_px;
    if(!Array.isArray(segments)&&Array.isArray(observation.visible_segment_ranges)&&Array.isArray(observation.points_source_px))segments=observation.visible_segment_ranges.filter(r=>Array.isArray(r)&&r.length===2&&r.every(Number.isInteger)&&r[0]>=0&&r[1]>r[0]&&r[1]<=observation.points_source_px.length).map(([start,stop])=>observation.points_source_px.slice(start,stop));
    if(!Array.isArray(segments)&&Array.isArray(observation.points_source_px))segments=[observation.points_source_px];
    if(!Array.isArray(segments)){
      const scale=observation.image_scale_from_source??frame?.image_scale_from_source;
      const sample=observation.visible_segments_sample_px??(Array.isArray(observation.points_sample_px)?[observation.points_sample_px]:null);
      if(!Array.isArray(sample)||!isNumber(scale)||scale<=0)return [];
      segments=sample.map(segment=>Array.isArray(segment)?segment.map(p=>Array.isArray(p)?p.map(v=>v/scale):p):[]);
    }
    // Never bridge masked/invalid gaps or complete an ellipse from its fit.
    const visible=[];
    for(const segment of segments){let part=[];if(!Array.isArray(segment))continue;for(const p of segment){if(Array.isArray(p)&&p.length===2&&p.every(isNumber))part.push(p);else{if(part.length)visible.push(part);part=[];}}if(part.length)visible.push(part);}
    return visible;
  }
  function landmarkObservationSummary(frame) {
    if(!frame)return "当前没有源帧记录。";
    if(!Array.isArray(frame.landmark_observations))return "本次运行没有逐帧地标轮廓记录；无名跟踪点不计为地标识别。";
    const rows=landmarkRows(frame);
    if(!rows.length)return "此帧没有已保存的地标候选；这不等于原片没有可见地标。";
    const counts={};for(const row of rows){const label=landmarkObservationStatus(row);counts[label]=(counts[label]||0)+1;}
    const details=rows.filter(row=>row.selected_model_id||row.candidate_model_ids?.length||row.rejection_reason).slice(0,6).map(row=>{
      const id=row.selected_model_id||(row.candidate_model_ids||[]).join(" / ")||row.observation_id||row.id||"未编号";
      const provenance=landmarkConfirmationLabel(row)?`（${landmarkConfirmationLabel(row)}）`:"";
      return `${id}${provenance}：${landmarkObservationStatus(row)}`;
    });
    return `${Object.entries(counts).map(([label,count])=>`${label} ${count} 组`).join("；")}。${details.length?" "+details.join("；")+"。":""} 组数不是独立验证数；完整原因见当前帧来源链。`;
  }
  function drawLandmarkObservations(ctx,frame,{enabled=true,labels=true,run={},sourceHash=null}={}) {
    if(!enabled||!frame)return;
    const colors={used:"#63ff9b",associated:"#72d8ff",unknown:"#ffe36b",conflict:"#ff665e",masked_out:"#b8bbc7",not_detected:"#b8bbc7"};
    const sx=ctx.canvas.width/(frame.source_size?.[0]||frame.source_width||frame.width||ctx.canvas.width),sy=ctx.canvas.height/(frame.source_size?.[1]||frame.source_height||frame.height||ctx.canvas.height);
    ctx.save();ctx.lineWidth=Math.max(2,ctx.canvas.width/700);ctx.font=`${Math.max(13,ctx.canvas.width/115)}px sans-serif`;
    for(const row of landmarkRows(frame)){
      if(observationBindingReason(row,frame,run,sourceHash))continue;
      const segments=observedLandmarkSegments(row,frame),status=landmarkObservationState(row);
      ctx.strokeStyle=colors[status];ctx.fillStyle=colors[status];ctx.setLineDash(status==="masked_out"?[5,5]:[]);
      for(const segment of segments){ctx.beginPath();if(segment.length===1){ctx.arc(segment[0][0]*sx,segment[0][1]*sy,Math.max(2,ctx.canvas.width/700),0,Math.PI*2);ctx.fill();}else{segment.forEach((p,i)=>i?ctx.lineTo(p[0]*sx,p[1]*sy):ctx.moveTo(p[0]*sx,p[1]*sy));ctx.stroke();}}
      if(labels&&segments.length){const p=segments[0][0],id=row.selected_model_id||(row.candidate_model_ids||[]).join("/")||"身份未知";ctx.fillText(`${id} · ${landmarkObservationStatus(row)}`,p[0]*sx+5,p[1]*sy-6);}
    }
    ctx.restore();
  }
  const helpers={invert,project,frameAtTime,seekFrameTime,presentedClockTime,statusLabel,runModeLabel,runDisplayLabel,displayStatusState,continuitySummary,reasonText,landmarkLabel,evidenceLabel,defaultModelIndex,defaultRunId,drawable,gridLines,projectedLine,landmarkPoint,clipForeground,structureObjects,modelLandmarks,wholeProjectedLine,drawWholeOverlay,drawSceneObservations,landmarkObservationState,landmarkObservationStatus,landmarkConfirmationLabel,observationBindingReason,observedLandmarkSegments,landmarkObservationSummary,drawLandmarkObservations};
  if (typeof module!=="undefined" && module.exports) module.exports=helpers;
  if (typeof document==="undefined") return;
  if (document.documentElement.hasAttribute("data-static-model-delivery")) {window.RinkSpaceGridGeometry=helpers;return;}
  const el=id=>document.getElementById(id), fmt=(v,n=3)=>isNumber(v)?v.toFixed(n):"—";
  const escape=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const state={models:[],runs:[],model:null,run:null,frames:[],frame:null,video:null,request:0,mapTransform:null,mediaTime:0,explicitModel:null,explicitRuns:{}};
  const get=async url=>{const response=await fetch(url);const body=await response.json();if(!response.ok)throw new Error(body.error||"模型接口读取失败");return body;};
  function link(id,url){el(id).hidden=!url;if(url)el(id).href=url;else el(id).removeAttribute("href");}
  function sourceHash(){const v=state.video;return v?.source_sha256||v?.sha256||v?.metadata?.source_sha256;}
  function runSelectionKey(){return JSON.stringify([state.video?.id||state.video?.key||sourceHash()||"",state.model?.model_id,state.model?.revision]);}
  function modelStatusLabel(model){return model.review_label_zh||({conditional_candidate:"条件草模候选",candidate:"待验证候选",rejected:"已否定，保留历史"})[model.status]||model.status||"候选";}
  function geometryLabel(model){return ({projective:"位置坐标网（非等距）",local_projective:"位置坐标网（非等距）",euclidean_conditional:"整场条件草模（比例与角部含假设，非实测尺寸）",euclidean_unscaled:"等距坐标网（任意单位）",metric_assumed:"米制网格（尺寸假设）",metric_verified:"米制网格（有尺度证据）"})[model?.geometry_level]||"位置坐标网（非等距）";}
  async function refresh(video=window.RinkSpaceUI?.state.video) {
    state.video=video;const request=++state.request;
    state.run=null;state.frames=[];state.frame=null;draw(state.mediaTime);
    const [models,runs]=await Promise.all([get("/api/rink-models"),get(`/api/model-localizations?video_id=${encodeURIComponent(video?.id||video?.key||"")}`)]);
    if(request!==state.request)return;
    state.models=models;state.runs=runs;
    el("rinkModelSelect").innerHTML=models.length?models.map((m,i)=>`<option value="${i}">${escape(m.model_id)} / ${escape(m.revision)} · ${escape(modelStatusLabel(m))}${m.preferred_delivery?" · 当前交付":""}</option>`).join(""):"<option value=''>尚无共同模型</option>";
    const selected=defaultModelIndex(models,state.explicitModel);el("rinkModelSelect").value=selected<0?"":String(selected);
    await chooseModel();
  }
  async function chooseModel(){
    state.model=state.models[Number(el("rinkModelSelect").value)]||null;
    const model=state.model;
    const runs=state.runs.filter(r=>model&&r.model_id===model.model_id&&String(r.model_revision??r.revision)===String(model.revision)).sort(compareRuns);
    el("localizationSelect").innerHTML=runs.length?runs.map(r=>`<option value="${escape(r.run_id)}">${escape(runDisplayLabel(r))} · ${fmt(r.start_s)}–${fmt(r.end_s)} s</option>`).join(""):"<option value=''>此原片暂无对应模型定位</option>";
    const preferredRun=model?.preferred_run_ids?.[state.video?.id||state.video?.key||""];
    el("localizationSelect").value=defaultRunId(runs,state.explicitRuns[runSelectionKey()]||preferredRun);
    el("gridGeometryLabel").textContent=geometryLabel(model);
    link("modelDocumentLink",model?.document_url);link("modelEvidenceLink",model?.evidence_url);link("modelOverviewLink",model?.overview_url);
    const sources=model?.sources||[], views=model?.reference_views||model?.view_poses||[], landmarks=modelLandmarks(model);
    el("modelSummary").textContent=model?`${model.model_id} / ${model.revision} · ${model.reference_source_count??sources.length} 个来源 · ${views.length} 个参考画面 · ${landmarks.length} 个地标。${modelStatusLabel(model)}；图像对应不是独立几何验证。`:"模型尚未生成。";
    const regions=model?.coverage_regions||[], ambiguity=model?.known_ambiguities||[];
    el("modelCoverage").textContent=`${Array.isArray(regions)?regions.map(r=>{const v=views.find(v=>v.id===r.id);return [r.label_zh||r.label||v?.label_zh||r.region_id,evidenceLabel(r.status)].filter(Boolean).join("：");}).filter(Boolean).join("；"):"几何、实拍与验证覆盖分别见证据"}${ambiguity.length?" · 缺口 / 歧义："+ambiguity.map(a=>typeof a==="string"?a:a.description||a.reason||a.id).join("；"):""}`;
    el("modelLandmarks").innerHTML=landmarks.map((l,i)=>`<button class="text-button" data-landmark="${i}">${escape(landmarkLabel(l))}</button>`).join("");
    el("modelLandmarks").querySelectorAll("[data-landmark]").forEach(button=>button.onclick=()=>showLandmark(Number(button.dataset.landmark)));
    await chooseRun();
  }
  async function chooseRun(){
    const request=++state.request, id=el("localizationSelect").value;
    state.run=null;state.frames=[];state.frame=null;draw(state.mediaTime);
    const entry=state.runs.find(r=>r.run_id===id);link("localizationDocument",entry?.document_url);
    el("seekLocalization").disabled=!entry;
    if(!entry)return;
    const run=await get(`/api/model-localization?run_id=${encodeURIComponent(id)}`);
    if(request!==state.request)return;
    state.run=run;state.frames=(run.frames||[]).filter(f=>isNumber(frameTime(f,run))).sort((a,b)=>frameTime(a,run)-frameTime(b,run));
    draw(el("sourcePlayer").currentTime);
  }
  function canvasAtVideo(){
    const video=el("sourcePlayer"), canvas=el("videoGridCanvas"), stage=el("modelVideoStage"), vr=video.getBoundingClientRect(),sr=stage.getBoundingClientRect();
    const width=video.videoWidth, height=video.videoHeight;if(!width||!height){canvas.width=1;canvas.height=1;return null;}
    const scale=Math.min(vr.width/width,vr.height/height),dw=width*scale,dh=height*scale;
    if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;
    Object.assign(canvas.style,{width:`${dw}px`,height:`${dh}px`,left:`${vr.left-sr.left+(vr.width-dw)/2}px`,top:`${vr.top-sr.top+(vr.height-dh)/2}px`});
    return canvas.getContext("2d");
  }
  function polygon(ctx,points,transform=p=>p){ctx.beginPath();let count=0;for(const point of points){const p=transform(point);if(!p)continue;count++===0?ctx.moveTo(...p):ctx.lineTo(...p);}ctx.closePath();}
  function draw(seconds){
    state.mediaTime=seconds;const frame=frameAtTime(state.frames,seconds,state.run||{});state.frame=frame;
    const model=state.model,run=state.run,reason=drawable(frame,model,run,sourceHash()),status=reason&&frame?.status!=="ambiguous"?"unavailable":frame?.status;
    const ctx=canvasAtVideo();if(ctx)ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
    el("localizationStatus").textContent=[runModeLabel(run),statusLabel(status,run)].filter(Boolean).join(" · ");el("localizationStatus").dataset.state=displayStatusState(status,run);
    el("localizationFrame").textContent=frame?`源时间 ${fmt(frame.source_time_s)} s · 源帧 ${frame.source_frame_index??"未完整索引"} · PTS ${frame.source_pts??frame.pts??"—"} × ${frame.source_time_base??frame.time_base??"未知时基"} · 镜头 ${frame.shot_id??"—"}`:`播放器 ${fmt(seconds)} s · 本时间无定位记录`;
    el("localizationReason").textContent=`${reasonText(reason||frame?.reason)||"当前画面有模型对应。"} ${continuitySummary(frame)}`;
    if(el("landmarkObservationSummary"))el("landmarkObservationSummary").textContent=observationBindingReason(null,frame,run||{},sourceHash())||landmarkObservationSummary(frame);
    el("inspectLocalizationFrame").disabled=!frame||!Number.isInteger(frame.source_frame_index)||!state.video?.indexed;
    if(ctx&&!reason&&model){
      if(model.structure&&frame.projection_chain){drawWholeOverlay(ctx,model,frame,{grid:el("gridToggle").checked,structure:el("predictedContourToggle")?.checked??true,labels:el("landmarkToggle").checked});}
      else {
      const inverse=invert(frame.H_model_from_image),width=ctx.canvas.width;
      ctx.save();polygon(ctx,frame.support_polygon_image);ctx.clip();
      clipForeground(ctx,frame);
      ctx.strokeStyle=statusColor[displayStatusState(status,run)]||"#ffce7a";ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=Math.max(1.5,width/700);ctx.setLineDash(["predicted","propagated","candidate","image_tracked","motion_predicted","model_reobserved"].includes(status)?[width/130,width/210]:[]);
      if(el("gridToggle").checked)for(const line of gridLines(model))for(const segment of projectedLine(inverse,line)){
        ctx.beginPath();segment.forEach((point,i)=>i?ctx.lineTo(...point):ctx.moveTo(...point));ctx.stroke();
      }
      if(el("predictedContourToggle")?.checked){ctx.strokeStyle="#ff77d9";for(const landmark of model.landmarks||[]){const curve=landmark.curve_model||landmark.polygon_model;if(Array.isArray(curve))for(const segment of projectedLine(inverse,curve)){ctx.beginPath();segment.forEach((point,i)=>i?ctx.lineTo(...point):ctx.moveTo(...point));ctx.stroke();}}}
      if(el("landmarkToggle").checked){ctx.font=`${Math.max(15,width/90)}px sans-serif`;for(const landmark of model.landmarks||[]){const p=project(inverse,landmarkPoint(landmark));if(p){if(landmark.model_xy){ctx.beginPath();ctx.arc(...p,width/300,0,Math.PI*2);ctx.fill();}ctx.fillText(landmarkLabel(landmark),p[0]+width/200,p[1]-width/200);}}}
      ctx.restore();
      }
    }
    if(ctx&&!observationBindingReason(null,frame,run||{},sourceHash()))drawSceneObservations(ctx,frame,{ice_marking:el("iceObservationToggle")?.checked,ice_board_contact:el("contactObservationToggle")?.checked,board_surface:el("boardObservationToggle")?.checked});
    if(ctx)drawLandmarkObservations(ctx,frame,{enabled:el("detectedContourToggle")?.checked,labels:el("landmarkToggle").checked,run:run||{},sourceHash:sourceHash()});
    drawMap(reason?null:frame);
  }
  function drawMap(frame){
    const canvas=el("modelMapCanvas"),ctx=canvas.getContext("2d"),model=state.model;
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#07161d";ctx.fillRect(0,0,canvas.width,canvas.height);
    const bounds=model?.structure?.bounds||model?.grid?.bounds;if(!bounds||bounds.length!==4)return;
    const[x0,y0,x1,y1]=bounds,scale=Math.min((canvas.width-48)/(x1-x0),(canvas.height-48)/(y1-y0));
    if(!isNumber(scale)||scale<=0)return;
    const ox=(canvas.width-(x1-x0)*scale)/2,oy=(canvas.height-(y1-y0)*scale)/2,transform=([x,y])=>[ox+(x-x0)*scale,oy+(model.structure?y1-y:y-y0)*scale];
    state.mapTransform={scale,x0,y0,y1,ox,oy,whole:Boolean(model.structure)};
    // Reference support polygons show actual connected observation footprints.
    for(const region of model.coverage_regions||[]){const points=region.polygon_model;if(points?.length>=3){polygon(ctx,points,transform);ctx.fillStyle="#327a7240";ctx.strokeStyle="#468d8477";ctx.fill();ctx.stroke();}}
    ctx.strokeStyle="#3c555d";ctx.lineWidth=1;for(const line of gridLines(model)){ctx.beginPath();line.forEach((p,i)=>{const q=transform(p);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.stroke();}
    for(const p of structureObjects(model)){ctx.strokeStyle=({observed:"#43d7b4",inferred:"#86aeff",assumed:"#ffba53"})[p.status]||"#ffba53";ctx.setLineDash(p.status==="observed"?[]:[4,5]);ctx.beginPath();(p.points_model||[]).forEach((v,i)=>i?ctx.lineTo(...transform(v)):ctx.moveTo(...transform(v)));if(p===model.structure.boundary)ctx.closePath();ctx.stroke();}ctx.setLineDash([]);
    if(frame&&!model.structure){polygon(ctx,frame.support_polygon_image,p=>{const q=project(frame.H_model_from_image,p);return q?transform(q):null;});ctx.fillStyle="#a7eadd2b";ctx.strokeStyle=statusColor[frame.status]||"#ffce7a";ctx.lineWidth=3;ctx.fill();ctx.stroke();}
    ctx.font="12px sans-serif";for(const landmark of modelLandmarks(model)){const point=landmarkPoint(landmark);if(!point)continue;const p=transform(point);ctx.fillStyle="#ffda8d";if(landmark.model_xy){ctx.beginPath();ctx.arc(...p,4,0,Math.PI*2);ctx.fill();}ctx.fillText(landmarkLabel(landmark),p[0]+7,p[1]-6);}
    ctx.fillStyle="#aac1c5";ctx.fillText(model.structure?"整场条件草模 · 非实测尺寸 · 不是全场验证范围":"共同坐标 · 曲线编号位置不是材料点/圆心",18,canvas.height-12);
  }
  function observationHTML(o){return `<figure>${o.image_url?`<a href="${escape(o.image_url)}" target="_blank" rel="noopener"><img src="${escape(o.image_url)}" alt="${escape(o.label_zh||o.label||o.view_id||o.source_id||"模型来源画面")}"></a>`:""}<figcaption>${escape(o.label_zh||o.label||o.view_id||o.id||"")} · ${escape(o.source_id||"")} · ${fmt(o.source_time_s)} s<br>PTS ${escape(o.source_pts??o.pts??"—")} × ${escape(o.source_time_base??o.time_base??"—")} · 源帧 ${escape(o.source_frame_index??"未完整索引")}<br><code>${escape(o.source_sha256||"")}</code></figcaption></figure>`;}
  function showLandmark(index){const landmark=modelLandmarks(state.model)[index];if(!landmark)return;el("modelProvenanceTitle").textContent=landmarkLabel(landmark);el("modelProvenanceBody").innerHTML=`<p>${escape(landmark.status||"候选")} · 模型 ${escape(state.model.model_id)} / ${escape(state.model.revision)} · 坐标 ${escape(JSON.stringify(landmark.model_xy||landmark.polygon_model||null))}</p><div class="model-observations">${(landmark.observations||[]).map(observationHTML).join("")}</div><details><summary>完整来源记录</summary><pre>${escape(JSON.stringify(landmark,null,2))}</pre></details>`;el("modelProvenanceDialog").showModal();}
  function showFrame(){const frame=state.frame,ids=frame?.physical_ids||[];el("modelProvenanceTitle").textContent=frame?`源时间 ${fmt(frame.source_time_s)} s · 模型定位来源链`:"当前时间无模型定位";const referenceIds=new Set([frame?.anchor_id,frame?.reference_view_id,...(frame?.anchor_lineage||[]).map(a=>a.reference_id||a.anchor_id)]);const views=(state.model?.reference_views||state.model?.view_poses||[]).filter(v=>referenceIds.has(v.id));el("modelProvenanceBody").innerHTML=`<p>模型 ${escape(state.model?.model_id||"—")} / ${escape(state.model?.revision||"—")} · 使用地标 ${escape(ids.join(" / ")||"见定位记录")}。有对应不代表独立验证通过。</p><div class="model-observations">${views.map(observationHTML).join("")}</div><pre>${escape(JSON.stringify(frame||{status:"unavailable",media_time_s:state.mediaTime},null,2))}</pre>`;el("modelProvenanceDialog").showModal();}
  function fail(error){el("localizationReason").textContent=error.message||String(error);el("localizationStatus").textContent="无法读取定位";}
  document.addEventListener("rinkspace:source",event=>refresh(event.detail.video).catch(fail));
  el("rinkModelSelect").onchange=()=>{const m=state.models[Number(el("rinkModelSelect").value)];state.explicitModel=m?{model_id:m.model_id,revision:m.revision}:null;chooseModel().catch(fail);};
  el("localizationSelect").onchange=()=>{state.explicitRuns[runSelectionKey()]=el("localizationSelect").value;chooseRun().catch(fail);};el("refreshModels").onclick=()=>refresh().catch(fail);
  el("gridToggle").onchange=()=>draw(state.mediaTime);el("landmarkToggle").onchange=()=>draw(state.mediaTime);
  for(const id of ["iceObservationToggle","contactObservationToggle","boardObservationToggle","detectedContourToggle","predictedContourToggle"]){if(el(id))el(id).onchange=()=>draw(state.mediaTime);}
  el("seekLocalization").onclick=()=>{if(state.frames.length)el("sourcePlayer").currentTime=frameTime(state.frames[0],state.run);};
  el("inspectLocalizationFrame").onclick=()=>{if(state.frame&&Number.isInteger(state.frame.source_frame_index))window.RinkSpaceUI.loadFrame(state.frame.source_frame_index).then(()=>el("sourceStage").scrollIntoView({behavior:"smooth",block:"center"})).catch(fail);};
  el("showFrameProvenance").onclick=showFrame;el("closeModelProvenance").onclick=()=>el("modelProvenanceDialog").close();
  el("modelMapCanvas").onclick=event=>{const t=state.mapTransform;if(!t)return;const r=event.currentTarget.getBoundingClientRect(),x=(event.clientX-r.left)*event.currentTarget.width/r.width,y=(event.clientY-r.top)*event.currentTarget.height/r.height;let chosen=-1,distance=30;for(const[l,landmark]of modelLandmarks(state.model).entries()){const point=landmarkPoint(landmark);if(!point)continue;const[lx,ly]=point,d=Math.hypot(t.ox+(lx-t.x0)*t.scale-x,t.oy+(t.whole?t.y1-ly:ly-t.y0)*t.scale-y);if(d<distance){chosen=l;distance=d;}}if(chosen>=0)showLandmark(chosen);};
  const video=el("sourcePlayer");let presentedTime=null;
  const drawPlayback=()=>draw(presentedClockTime(video.currentTime,presentedTime,video.seeking));
  for(const name of["loadedmetadata","seeking"])video.addEventListener(name,()=>{presentedTime=null;drawPlayback();});
  for(const name of["seeked","pause","timeupdate","ended"])video.addEventListener(name,drawPlayback);
  if(video.requestVideoFrameCallback){const callback=(_,metadata)=>{presentedTime=metadata.mediaTime;drawPlayback();video.requestVideoFrameCallback(callback);};video.requestVideoFrameCallback(callback);}
  new ResizeObserver(()=>draw(state.mediaTime)).observe(el("modelVideoStage"));
  window.RinkSpaceGrid={state,...helpers,draw,refresh,showFrame};
  if(window.RinkSpaceUI?.state.video)refresh().catch(fail);
})();
