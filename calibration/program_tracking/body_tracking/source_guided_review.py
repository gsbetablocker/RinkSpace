"""Reproduce whole-S05 native source comparisons from saved results; no fit or GUI."""
from pathlib import Path
import json,hashlib,numpy as np
from PIL import Image,ImageDraw,ImageFont
from rinkspace.viewer3d.source import SourceVideo
from rinkspace.viewer3d.blade_review import overlay_primitives
from rinkspace.path_policy import guard_project_write
from calibration.program_tracking.body_tracking.source_guided_metrics import compare_metrics
from calibration.program_tracking.body_tracking.source_support_locations import _runner_witness
from calibration.program_tracking.body_tracking.blade_projection import SavedBladeProjection
from calibration.program_tracking.body_tracking.source_visible_model import camera_centres
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest
from calibration.program_tracking.body_tracking.fullbody_round import write_json
def build(project,root,selection_path,*,delivery_directory='delivery',output_directory='review'):
 from rinkspace.storage import AssetStore
 P=Path(project).resolve();R=Path(root).resolve();D=AssetStore(P).data;B=R.parent;N=B/'body_sequence_1105';E=P/'events/ankara_2026/programmes/okada_mei_sp';out=R/output_directory
 out.mkdir(exist_ok=False)
 delivery=json.loads((R/delivery_directory/'delivery.json').read_text(encoding='utf-8'));selection=json.loads(Path(selection_path).read_text(encoding='utf-8'));final=selection['final_variant']
 base_metrics=Path(selection.get('baseline_metrics',R/'baseline_direction/baseline_metrics.json'));contract=N/'source_contract_review_1106/known_foot_contracts.json';old22=N/'full_structure_22/annotations_combined.json'
 metrics={};summary={};curves={}
 definitions=dict(delivery['variants'])
 for vid,row in definitions.items():
  stage=Path(row['stage']);metric=base_metrics if vid=='sequence_source_structure'else stage/'final_metrics.json'
  record=compare_metrics(base_metrics,metric,old22,contract);write_json(out/(vid+'_vs_baseline.json'),record);metrics[vid]=record
  with np.load(row['display_result'],allow_pickle=False)as z:
   keys=['pts','time_base','P_world_mid','Q_world_mid','local_R_mid','native22_world_mid','COCO_world_mid','COCO_uv','com_world_mid','skate_R_world_mid','skate_origin_world_mid','skate_witness_world_mid','blade_material_world_mid','skate_edge_world_mid','skate_clearance_m_mid','foot_cavity_max_violation_m_mid','source_support_mode_mid','sequence_contact_modes','sequence_contact_status']
   a={k:z[k].copy()for k in keys if k in z.files}
  query=SavedBladeProjection(P,stage,Path(row['parameter']));Rshoe=a['skate_R_world_mid'];O=a['skate_origin_world_mid']/query.ratio
  exact=[_runner_witness(query.binding,Rshoe[:,f],O[:,f])for f in (0,1)]
  a['runner_exact_witness_world_mid']=np.stack([x['world']for x in exact],axis=1)*query.ratio
  a['runner_exact_feature_index']=np.stack([x['feature_index']for x in exact],axis=1)
  a['coordinate_semantics_json']=np.array(json.dumps(dict(world='same frozen camera/conditional metre frame; not independent 3D truth',material='fixed model rear/middle/front profile stations; not observed contact',witness='moving geometric minimum; not source contact material',source_time_base=[1,12800])))
  curves[vid]=a
  if vid!='sequence_source_structure':np.savez_compressed(out/(vid+'_compact_trajectory.npz'),**a)
  t=a['pts']/12800.;dt=np.diff(t)
  def jumps(x):
   step=np.linalg.norm(np.diff(x,axis=0),axis=-1);flat=np.argsort(step.ravel())[-8:][::-1];result=[]
   for k in flat:
    ind=np.unravel_index(k,step.shape);i=ind[0];result.append(dict(before_pts=int(a['pts'][i]),after_pts=int(a['pts'][i+1]),source_seconds=[float(t[i]),float(t[i+1])],channel=[int(v)for v in ind[1:]],step_conditional_m=float(step[ind]),speed_conditional_m_s=float(step[ind]/dt[i])))
   return result
  def span(x,lo,hi):return np.ptp(x[(t>=lo)&(t<=hi),:2],axis=0).tolist()
  data=json.loads(metric.read_text(encoding='utf-8'));xy=data['source_support_locations'];dist=np.array([np.linalg.norm(x['outside_source_range_xy_conditional_m'])for x in xy]);lower=[x for x in data['visible_runner_contours']if x['structure_feature']=='runner_lower_edge'and x.get('source_structure_feature','runner_lower_edge')=='runner_lower_edge'and '_component'not in x['id']]
  summary[vid]=dict(body_pixels=data['body_pixel_distance'],canonical_lower_foot_rows=len(lower),canonical_lower_pixel_samples_diagnostic=np.percentile(np.concatenate([x['distance_px']for x in lower]),[50,95,100]).tolist(),xy_rows=len(xy),xy_conditional_metres=np.percentile(dist,[50,95,100]).tolist(),physical=data['physical'],
   root_jumps=jumps(a['P_world_mid']),COM_proxy_jumps=jumps(a['com_world_mid']),shoe_origin_jumps=jumps(a['skate_origin_world_mid']),fixed_station_jumps=jumps(a['blade_material_world_mid']),runner_witness_jumps=jumps(a['runner_exact_witness_world_mid']),
   support_spin_88_92_xy_span=dict(origin=span(a['skate_origin_world_mid'][:,1],88,92),fixed_middle=span(a['blade_material_world_mid'][:,1,1],88,92),runner_minimum=span(a['runner_exact_witness_world_mid'][:,1],88,92)),
   whole_source_951=True,support_coordinate_span_is_not_accuracy=True)
  for key,fn in [('source_contour_contract',contract),('source_episode',N/'source_contract_review_1106/source_episode_review.json')]:
   target=guard_project_write(E/('spin_'+key+'_1106.json'),project_root=P)
   if target.exists():
    if digest(target)!=digest(fn):raise ValueError('Existing final evidence changed')
   else:target.write_bytes(fn.read_bytes())
 write_json(out/'trajectory_and_source_summary.json',summary)
 write_json(out/'source_guided_vs_final.json',compare_metrics(Path(delivery['variants'][selection['source_guided_variant']]['stage'])/'final_metrics.json',Path(delivery['variants'][final]['stage'])/'final_metrics.json',old22,contract))
 source=json.loads((N/'full_structure_1106_inputs/structures_fit_v1.json').read_text(encoding='utf-8'));known=source['frames'];unknown=source['unassigned_identity_observations'];write_json(out/'coverage.json',dict(known_foot_observations=len(known),known_native_exposures=len({r['pts']for r in known}),unassigned_foot_observations=len(unknown),unassigned_native_exposures=len({r['pts']for r in unknown}),all_native_exposures=len({r['pts']for r in known+unknown}),whole_estimated_exposures=951,independent_heldout=0,new_source_pixel_exposures_this_round=0))
 asset=json.loads((R/delivery_directory/'sequence_blade_review.json').read_text(encoding='utf-8'));arrays=np.load(R/delivery_directory/'sequence_blade_review.npz',allow_pickle=False);by={(int(r['pts']),r['foot']):r for r in asset['source_structures']if r.get('foot')in(0,1)}
 video_path=json.loads((N/'source_sequence.json').read_text(encoding='utf-8'))['source_video']['path']
 video=SourceVideo(video_path,(1,12800));font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);queries={v:SavedBladeProjection(P,Path(definitions[v]['stage']),Path(definitions[v]['parameter']))for v in ['sequence_source_structure','source_guided',final]}
 for second in (81.62,87.64,91.2,92.8):
  pt=round(second*12800);frame=by[pt,1];points=np.concatenate([np.asarray(f['points_uv'])for k,f in frame['features'].items()if k in ['runner_lower_edge','runner_upper_edge','blade_assembly_outline','toe_pick','rear_extension']and len(f.get('points_uv',[]))])
  lo=np.maximum(np.floor(points.min(0)-18),[0,0]).astype(int);hi=np.minimum(np.ceil(points.max(0)+18),[1920,1080]).astype(int);bounds=(*lo,*hi);im=Image.fromarray(video.read(pt));sheet=Image.new('RGB',(2500,650),'white');draw=ImageDraw.Draw(sheet)
  for col,(vid,title)in enumerate([(None,'干净原片'),('source','原片读出的结构'),('sequence_source_structure','旧基线：模型下刃'),('source_guided','源引导：模型下刃'),(final,'新候选：模型下刃')]):
   panel=im.copy();pd=ImageDraw.Draw(panel,'RGBA')
   if vid=='source':
    for primitive in overlay_primitives(dict(model=None,source_structures=[frame]),include_model_reference=False,include_model_contours=False,include_source_structures=True,include_source_legacy=False):pd.line([tuple(p)for p in primitive['points']],fill=tuple(round(c*255)for c in primitive['color']),width=1)
   elif vid:
    q=queries[vid];i=int(np.searchsorted(q.pts,pt));R0=curves[vid]['skate_R_world_mid'][i,1];O0=curves[vid]['skate_origin_world_mid'][i,1]/q.ratio;C=camera_centres(q.camera,np.array([i]),np.array([np.median(points[:,1])]))[0];side=int((R0.T@(C-O0))[1]>=0)
    uv=arrays[asset['variants'][vid]+'_runner_edges_uv'][i,1,side];pd.line([tuple(p)for p in uv],fill=(235,0,190,255),width=2)
   crop=panel.crop(bounds);scale=min(480/crop.width,470/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.NEAREST);sheet.paste(crop,(col*500+(500-crop.width)//2,80+(470-crop.height)//2));draw.text((col*500+12,22),title,font=font,fill='black')
  draw.text((12,570),f'{second:.2f}s / exact PTS {pt} / foot1。源：绿下刃、深绿上缘、紫刀架、红齿区、蓝后尾；虚线保留歧义。',font=font,fill='black')
  draw.text((12,605),'品红仅为相机近侧的实际模型钢刃；不是源观测，也不是已确认接冰线。前三维状态未做图像平移或缩放对齐。',font=font,fill='black');sheet.save(out/f'closeup_{pt}.png')
 video.close();write_json(out/'report_manifest.json',dict(summary=str(out/'trajectory_and_source_summary.json'),coverage=str(out/'coverage.json'),new_programme_evidence_files=[str(E/'spin_source_contour_contract_1106.json'),str(E/'spin_source_episode_1106.json')],no_fit=True,no_inference=True,no_gui=True));print(json.dumps(dict(output=str(out),variants=list(summary),final_variant=final),ensure_ascii=False),flush=True)


def main():
 import argparse
 ap=argparse.ArgumentParser()
 for name in ('project','root','selection'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--delivery-directory',default='delivery');ap.add_argument('--output-directory',default='review')
 a=ap.parse_args();build(a.project,a.root,a.selection,delivery_directory=a.delivery_directory,output_directory=a.output_directory)

if __name__=='__main__':main()
