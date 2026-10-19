"""Player event order and visible provenance checks; no geometry acceptance."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_portable_player_pause_steps_and_ended_keep_source_pts_and_raw_diagnostics():
    node=shutil.which('node')
    if not node:
        pytest.skip('Node unavailable')
    static=Path(__file__).parents[1]/'src/rinkspace/static'
    check=r'''
const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const G=require(process.argv[1]);
const model={model_id:'test',revision:'v0',geometry_level:'projective',sources:[],reference_views:[],landmarks:[{physical_id:'R1_gray_near',label:'R1_gray_near'}]};
const frames=[0,.02,.04].map((t,i)=>({media_time_s:t,source_time_s:214.30+t,source_pts:2743040+i*256,pts:2743040+i*256,source_time_base:'1/12800',time_base:'1/12800',source_frame_index:10715+i,status:'unavailable',reason:'no_model_ice_anchor',H_model_from_image:null}));
const data={model,missing_reference_images:[],runs:[{frames,model_id:'test',model_revision:'v0',source_sha256:'hash',proxy_file:'test.mp4',role:'new_source_inference_after_freeze',selected_window:{selection_basis:'synthetic three-frame UI test'}}]};
const events={},elements=new Map();
const canvas={clearRect(){},fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fill(){},arc(){},closePath(){},fillText(){},clip(){},save(){},restore(){},setLineDash(){}};
function element(id){if(!elements.has(id))elements.set(id,{id,value:'0',textContent:'',innerHTML:'',style:{},dataset:{},disabled:false,width:640,height:360,hidden:false,checked:false,getContext(){return canvas;},querySelectorAll(){return [];},getBoundingClientRect(){return {left:0,top:0,width:640,height:360};},showModal(){this.open=true;},close(){this.open=false;}});return elements.get(id);}
element('review-data').textContent=JSON.stringify(data);
const video=element('video');Object.assign(video,{currentTime:0,videoWidth:640,videoHeight:360,paused:true,seeking:false,ended:false,pause(){this.paused=true;},addEventListener(name,fn){(events[name]??=[]).push(fn);},requestVideoFrameCallback(fn){this.callback=fn;}});
const fakeDocument={getElementById:element};
vm.runInNewContext(fs.readFileSync(process.argv[2],'utf8'),{window:{RinkSpaceGridGeometry:G},document:fakeDocument,ResizeObserver:class{observe(){}},console});
const emit=name=>(events[name]||[]).forEach(fn=>fn());
assert.match(element('reason').textContent,/没有找到足够的冰面图案对应/);
assert.equal(element('status').textContent,'无法定位');
assert.match(element('landmarks').innerHTML,/R1 近侧灰靶/);
assert.ok(!element('landmarks').innerHTML.includes('R1_gray_near'));
assert.match(element('runSummary').textContent,/模型冻结后的新来源推理/);
assert.ok(!element('runSummary').textContent.includes('new_source_inference_after_freeze'));
assert.match(G.evidenceLabel('direct_image_candidate_geometry'),/候选几何，尚未独立验证/);
assert.match(G.evidenceLabel('propagated_prior_unknown_cumulative_drift'),/累计漂移尚未验证/);
video.currentTime=.039;video.callback(null,{mediaTime:.02});emit('pause');
assert.match(element('frameIdentity').textContent,/2743296/);
element('next').onclick();assert.ok(video.currentTime>.04&&video.currentTime<.06);
video.callback(null,{mediaTime:.04});emit('seeked');
assert.match(element('frameIdentity').textContent,/2743552/);
video.currentTime=.06;video.ended=true;emit('ended');emit('timeupdate');
assert.match(element('frameIdentity').textContent,/214\.340/);
assert.match(element('frameIdentity').textContent,/2743552/);
assert.equal(element('previous').disabled,false);assert.equal(element('next').disabled,true);
element('sourceChain').onclick();assert.match(element('provenanceBody').innerHTML,/no_model_ice_anchor/);
video.ended=false;element('previous').onclick();assert.ok(video.currentTime>.02&&video.currentTime<.04);
video.callback(null,{mediaTime:.02});emit('seeked');assert.match(element('frameIdentity').textContent,/2743296/);
'''
    result=subprocess.run([node,'-e',check,str(static/'model_grid.js'),str(static/'model_delivery.js')],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr


def test_workbench_defaults_use_natural_revision_and_run_metadata_not_success_counts():
    node=shutil.which('node')
    if not node:
        pytest.skip('Node unavailable')
    script=Path(__file__).parents[1]/'src/rinkspace/static/model_grid.js'
    check=r'''
const assert=require('node:assert/strict'),G=require(process.argv[1]);
const models=[{model_id:'rink',revision:'v0.1'},{model_id:'rink',revision:'v0.3'},{model_id:'rink',revision:'v0.10'}];
assert.equal(G.defaultModelIndex(models),2);
assert.equal(G.defaultModelIndex([...models].reverse(),{model_id:'rink',revision:'v0.1'}),2);
assert.equal(G.defaultModelIndex(models,{model_id:'missing',revision:'v99'}),2);
const delivery=models.map((m,i)=>({...m,preferred_delivery:i===0}));
assert.equal(G.defaultModelIndex(delivery),0); // Explicit reviewed delivery beats later failed drafts.
assert.equal(G.defaultModelIndex(delivery,{model_id:'rink',revision:'v0.10'}),2); // User choice still wins.
const common={algorithm_version:'model-v2',all_selected_frames_in_denominator:true,anchor_ablation:false};
const current={...common,run_id:'ordinary-result',reobserve_every_source_frames:5,start_s:201.44,status_counts:{unavailable:556,candidate:13}};
const oldExperiment={...common,run_id:'zzzzz-old-experiment',start_s:201.44,status_counts:{candidate:646}};
const ablation={...current,run_id:'aaaa-ablation',anchor_ablation:true,algorithm_version:'model-v99',created_at:'2099-01-01'};
const oldWindow={...current,run_id:'zzzzz-previous-window',start_s:75.5,status_counts:{candidate:689}};
const runs=[ablation,oldExperiment,oldWindow,current];
assert.equal(G.defaultRunId(runs),'ordinary-result');
assert.equal(G.defaultRunId([...runs].reverse()),'ordinary-result');
assert.equal(G.defaultRunId(runs,'aaaa-ablation'),'aaaa-ablation'); // explicit historical choice persists
assert.equal(G.defaultRunId(runs,'missing'),'ordinary-result');
assert.equal(G.defaultRunId(runs,'zzzzz-previous-window'),'zzzzz-previous-window'); // Same selector accepts explicit manifest preference.
const created=[{...current,run_id:'old',created_at:'2026-01-01'},{...current,run_id:'new',created_at:'2026-01-02'}];
assert.equal(G.defaultRunId(created),'new');
assert.equal(runs.length,4);assert.equal(G.defaultRunId([]),'');assert.equal(G.defaultModelIndex([]),-1);
'''
    result=subprocess.run([node,'-e',check,str(script)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr


def test_whole_structure_nonlinear_canvas_toggle_keeps_raw_domain_and_front_branch():
    node=shutil.which('node')
    if not node: pytest.skip('Node unavailable')
    script=Path(__file__).parents[1]/'src/rinkspace/static/model_grid.js'
    check=r'''
const assert=require('node:assert/strict'),G=require(process.argv[1]),I=[[1,0,0],[0,1,0],[0,0,1]];
const chain={H_reference_undistorted_from_model:I,H_current_sample_from_reference_raw:I,H_source_image_from_sample:[[2,0,0],[0,2,0],[0,0,1]],reference_radial_distortion:{k1:-.2,centre_image:[250,200],scale_image:500}};
const model={model_id:'m',revision:'whole-v1',structure:{bounds:[0,0,500,300],boundary:{id:'B',kind:'boundary',status:'assumed',points_model:[[0,0],[500,0],[500,300],[0,300]]},primitives:[{id:'L',kind:'line',status:'observed',points_model:[[100,100],[400,100]]}]}};
const run={model_id:'m',model_revision:'whole-v1',source_sha256:'original'};
const frame={status:'candidate',projection_chain:chain,H_model_from_image:null,image_domain:'raw_reference_transfer_approximation',ice_polygons_sample:[[[0,0],[500,0],[500,300],[0,300]]],foreground_polygons_sample:[[[100,100],[200,100],[200,200],[100,200]]]};
assert.equal(G.drawable(frame,model,run,'original'),null);
assert.match(G.drawable({...frame,status:'unavailable'},model,run,'original'),/无法定位/);
assert.equal(G.drawable({...frame,status:'ambiguous',reason:'model_anchor_conflict'},model,run,'original'),'model_anchor_conflict');
assert.match(G.reasonText('model_anchor_conflict'),/暂停叠线并切断传播/);
assert.match(G.drawable(frame,model,run,'changed'),/SHA-256/);
assert.match(G.drawable({...frame,projection_chain:{...chain,H_source_image_from_sample:null}},model,run,'original'),/矩阵/);
const p=G.wholeProjectedLine(chain,[[100,100],[400,100]])[0][0];
assert.ok(Math.abs(p[0]-207.8)<1e-7&&Math.abs(p[1]-205.2)<1e-7);
const back={...chain,H_reference_undistorted_from_model:[[1,0,0],[0,1,100],[.01,0,1]]};
assert.equal(G.wholeProjectedLine(back,[[-200,-300],[-110,-300]]).length,0);
const folded={...chain,reference_radial_distortion:{k1:-1,centre_image:[0,0],scale_image:100}};
assert.equal(G.wholeProjectedLine(folded,[[80,40],[110,40]]).length,0);
const rescaled={...chain,H_reference_undistorted_from_model:I.map(r=>r.map(x=>-x)),reference_visible_branch_sign:-1};
assert.deepEqual(G.wholeProjectedLine(rescaled,[[100,100],[400,100]]),G.wholeProjectedLine(chain,[[100,100],[400,100]]));
const transferScaled={...chain,H_current_sample_from_reference_raw:I.map(r=>r.map(x=>-x)),raw_transfer_visible_branch_sign:-1};
assert.deepEqual(G.wholeProjectedLine(transferScaled,[[100,100],[400,100]]),G.wholeProjectedLine(chain,[[100,100],[400,100]]));
assert.equal(G.wholeProjectedLine({...transferScaled,raw_transfer_visible_branch_sign:1},[[100,100],[400,100]]).length,0);
let strokes=0,clips=0;const ctx={canvas:{width:1000,height:600},save(){},restore(){},beginPath(){},moveTo(){},lineTo(){},closePath(){},rect(){},clip(){clips++;},setLineDash(){},stroke(){strokes++;},fillText(){}};
G.drawWholeOverlay(ctx,model,frame,{geometry:false,labels:false});assert.equal(strokes,0);
G.drawWholeOverlay(ctx,model,frame,{geometry:true,labels:false});assert.ok(strokes>0&&clips>=4);
assert.equal(frame.H_model_from_image,null); // Never fabricate an inverse for radial raw pixels.
assert.match(G.reasonText('fixed_model_reference_ice_reobserved'),/条件草模候选/);
'''
    result=subprocess.run([node,'-e',check,str(script)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr


def test_continuity_states_do_not_promote_independent_accuracy_and_keep_old_labels():
    node=shutil.which('node')
    if not node: pytest.skip('Node unavailable')
    script=Path(__file__).parents[1]/'src/rinkspace/static/model_grid.js'
    check=r'''
const assert=require('node:assert/strict'),G=require(process.argv[1]),I=[[1,0,0],[0,1,0],[0,0,1]];
for(const [status,label] of Object.entries({model_reobserved:'认出场地',image_tracked:'持续跟踪',motion_predicted:'暂时推算',ambiguous:'存在冲突',unavailable:'暂时无法定位',candidate:'待验证候选',predicted:'连续推算'}))assert.equal(G.statusLabel(status),label);
const autoRun={landmark_mode:'automatic_named_landmarks'},manualRun={run_id:'comparison',landmark_mode:'manual_source_reading_development_diagnostic'};
assert.equal(G.statusLabel('model_reobserved',autoRun),'地标定位候选');
assert.equal(G.statusLabel('image_tracked',autoRun),'沿图像连续推算');
assert.equal(G.statusLabel('model_reobserved',manualRun),'地标定位候选');
assert.equal(G.displayStatusState('model_reobserved',manualRun),'candidate');
assert.equal(G.displayStatusState('model_reobserved'),'model_reobserved');
assert.match(G.runDisplayLabel(manualRun),/人工干预对照/);
assert.equal(G.runModeLabel(autoRun),'');
assert.equal(G.landmarkConfirmationLabel({manual_confirmed:true,user_confirmed:false,confirmation_actor:'agent_source_reading_not_user_confirmation'}),'人工读点候选');
assert.equal(G.landmarkConfirmationLabel({manual_confirmed:true}),'人工读点候选');
assert.equal(G.landmarkConfirmationLabel({user_confirmed:true}),'用户确认');
const model={model_id:'m',revision:'v07'},run={model_id:'m',model_revision:'v07'},frame={status:'image_tracked',H_model_from_image:I,support_polygon_image:[[0,0],[20,0],[20,20]],anchor_age_s:25,last_image_observation_age_s:0,uncertainty:{estimated_error_px_at_960:4.2,kind:'engineering_not_calibrated_confidence'}};
for(const status of ['model_reobserved','image_tracked','motion_predicted','candidate','predicted'])assert.equal(G.drawable({...frame,status},model,run),null);
for(const status of ['ambiguous','unavailable'])assert.ok(G.drawable({...frame,status},model,run));
assert.match(G.continuitySummary(frame),/距最近图像观测 0.00 秒/);
assert.match(G.continuitySummary(frame),/4.2 像素（960宽，非统计置信界）/);
assert.match(G.continuitySummary(frame),/尚未独立验准/);
assert.match(G.continuitySummary({...frame,status:'model_reobserved'}),/尚未独立验准/);
assert.match(G.continuitySummary({...frame,future_anchor_used:true}),/后续画面/);
assert.match(G.continuitySummary({...frame,camera_uncertainty:{engineering_range_sample_px:12.3}}),/12.3 样图像素/);
assert.ok(!G.continuitySummary({...frame,camera_uncertainty:{engineering_range_sample_px:12.3}}).includes('4.2'));
assert.match(G.reasonText('unanchored_local_component'),/尚未确定它在冰场的哪里/);
assert.match(G.reasonText('cut_waiting_for_relocalization'),/同一冰场/);
'''
    result=subprocess.run([node,'-e',check,str(script)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr


def test_landmark_overlay_keeps_observed_gaps_unknown_ids_and_missing_camera_separate():
    node=shutil.which('node')
    if not node: pytest.skip('Node unavailable')
    script=Path(__file__).parents[1]/'src/rinkspace/static/model_grid.js'
    check=r'''
const assert=require('node:assert/strict'),G=require(process.argv[1]);
const observation={source_pts:2560,source_time_base:'1/12800',source_sha256:'source',
  visible_segments_source_px:[[[20,30],[30,40]],[[70,70],[80,80]]],
  candidate_model_ids:['R1','R2'],selected_model_id:null,association_status:'detected_unidentified',used_by_solver:false};
const frame={status:'unavailable',source_sha256:'source',pts:2560,time_base:'1/12800',width:1920,height:1080,projection_chain:null,landmark_observations:[observation]};
let strokes=0,labels=[],lines=[],moves=[],closes=0;
const ctx={canvas:{width:1920,height:1080},save(){},restore(){},beginPath(){},setLineDash(){},stroke(){strokes++;},fill(){},arc(){},moveTo(x,y){moves.push([x,y]);},lineTo(x,y){lines.push([x,y]);},closePath(){closes++;},fillText(text){labels.push(text);}};
G.drawLandmarkObservations(ctx,frame,{enabled:false});assert.equal(strokes,0);
G.drawLandmarkObservations(ctx,frame,{sourceHash:'source'});
assert.equal(strokes,2);assert.equal(closes,0);assert.deepEqual(moves,[[20,30],[70,70]]);assert.deepEqual(lines,[[30,40],[80,80]]);
assert.match(labels[0],/R1\/R2.*看见但不知是哪个/);assert.equal(frame.projection_chain,null);
const previous=strokes;
G.drawLandmarkObservations(ctx,frame,{sourceHash:'different'});assert.equal(strokes,previous);
G.drawLandmarkObservations(ctx,{...frame,pts:2816},{sourceHash:'source'});assert.equal(strokes,previous);
assert.match(G.observationBindingReason(observation,{...frame,time_base:'1/50'}),/时基/);
assert.match(G.landmarkObservationSummary(frame),/看见但不知是哪个 1 组/);
assert.match(G.landmarkObservationSummary({landmark_observations:[]}),/不等于原片没有可见地标/);
assert.match(G.landmarkObservationSummary({}),/没有逐帧地标轮廓记录/);
assert.match(G.landmarkObservationStatus({used_by_solver:true,selected_model_id:'R1'}),/已参与求解，尚未独立验准/);
assert.equal(G.landmarkObservationStatus({association_status:'conflict',used_by_solver:true}),'已对应但几何冲突');
assert.equal(G.landmarkObservationStatus({association_status:'masked_out'}),'被遮罩删掉');
assert.deepEqual(G.observedLandmarkSegments({points_sample_px:[[10,15],[15,20]],image_scale_from_source:.5},frame),[[[20,30],[30,40]]]);
assert.deepEqual(G.observedLandmarkSegments({points_sample_px:[[10,15]]},frame),[]);
assert.deepEqual(G.observedLandmarkSegments({points_source_px:[[1,2],null,[4,5]],fitted_ellipse:[5,5,10,2]},frame),[[[1,2]],[[4,5]]]);
assert.deepEqual(G.observedLandmarkSegments({fitted_ellipse:[5,5,10,2]},frame),[]); // Fits are not observed contours.
assert.deepEqual(G.observedLandmarkSegments({points_source_px:[[1,2],[3,4],[10,11],[12,13]],visible_segment_ranges:[[0,2],[2,4]]},frame),[[[1,2],[3,4]],[[10,11],[12,13]]]);
assert.deepEqual(G.observedLandmarkSegments({points_source_px:[[1,2],[3,4]],visible_segment_ranges:[[0,7]]},frame),[]); // Invalid range cannot reconnect a missing arc.
'''
    result=subprocess.run([node,'-e',check,str(script)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr


def test_predicted_structure_grid_and_contour_controls_are_independent():
    node=shutil.which('node')
    if not node: pytest.skip('Node unavailable')
    static=Path(__file__).parents[1]/'src/rinkspace/static'
    markup=(static/'index.html').read_text(encoding='utf8')
    for control in ('detectedContourToggle','predictedContourToggle','gridToggle',
                    'iceObservationToggle','contactObservationToggle','boardObservationToggle'):
        assert markup.count('id="'+control+'"')==1
    check=r'''
const assert=require('node:assert/strict'),G=require(process.argv[1]),I=[[1,0,0],[0,1,0],[0,0,1]];
const model={structure:{bounds:[0,0,100,100],boundary:{id:'edge',points_model:[[0,0],[100,0]]},primitives:[{id:'midline',points_model:[[50,0],[50,100]]}]},grid:{x_values:[20,80],y_values:[20,80]}};
const frame={projection_chain:{H_reference_undistorted_from_model:I,H_current_sample_from_reference_raw:I,H_source_image_from_sample:I},ice_polygons_sample:[[[0,0],[100,0],[100,100],[0,100]]],foreground_polygons_sample:[]};
let strokes=0;const colors=[];
const ctx={canvas:{width:100,height:100},save(){},restore(){},beginPath(){},moveTo(){},lineTo(){},closePath(){},clip(){},setLineDash(){},stroke(){strokes++;colors.push(this.strokeStyle);},fillText(){}};
G.drawWholeOverlay(ctx,model,frame,{structure:false,grid:false});assert.equal(strokes,0);
G.drawWholeOverlay(ctx,model,frame,{structure:true,grid:false});assert.equal(strokes,2);assert.deepEqual(new Set(colors),new Set(['#ff77d9']));
strokes=0;colors.length=0;
G.drawWholeOverlay(ctx,model,frame,{structure:false,grid:true});assert.equal(strokes,4);assert.deepEqual(new Set(colors),new Set(['#a7c9ea']));
strokes=0;G.drawWholeOverlay(ctx,model,frame,{geometry:false});assert.equal(strokes,0); // Historical callers remain compatible.
'''
    result=subprocess.run([node,'-e',check,str(static/'model_grid.js')],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
