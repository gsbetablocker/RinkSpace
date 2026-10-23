"""Real-source fixed2D anchor guard: full existing cohort, no RGB/curve evaluation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,argparse,time,hashlib,collections
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[4]
from rinkfit_joint.io_utils import read_json,write_json,file_sha256
from rinkfit_joint.joint_metrics import PacketArrays
from rinkfit_joint.joint_model import graph_from_joint,as_event_view
from rinkfit_joint.camera_graph import CameraGraph
from rinkfit_joint.joint_replay import render_visibility,material_visibility_records
from rinkfit_joint.recovery_metrics import _same_denominator_statistics
from rinkfit_start.native_scene import NativeScene

def ref(f):return dict(file=str(Path(f).resolve()),sha256=file_sha256(f))
def find_overlay(packet_file,p):
 visited=set();proofs=[]
 while True:
  key=str(Path(packet_file).resolve())
  if key in visited:raise ValueError('Cycle in actual numeric prefix lineage')
  visited.add(key);d=p.get('dense_numeric_auxiliary_overlay')
  if d:
   f=Path(d['file']);f=f if f.is_absolute()else Path(packet_file).parent/f
   if file_sha256(f)!=d['sha256']:raise ValueError('Changed numeric parent prefix overlay')
   x=read_json(f);proofs.append(ref(f))
   pieces=[v for v in x.get('pieces',[])if v['kind']=='complete_native_checkpoint_prefix'and v['start']==0]
   if pieces:
    if len(pieces)!=1:raise ValueError('Exact one complete parent prefix required')
    piece=pieces[0]
    for k in ['explicit_nuisance_manifest','complete_state_verification','state_binding','complete_checkpoint']:
     if file_sha256(piece[k]['file'])!=piece[k]['sha256']:raise ValueError('Changed restored parent proof '+k)
    return piece,dict(overlay=ref(f),traversed_numeric_references=proofs,parent_complete_checkpoint=piece['complete_checkpoint'],parent_numeric_manifest=piece['explicit_nuisance_manifest'])
  parent=p.get('dense_parent_packet')
  if not parent:raise ValueError('No actual restored complete parent prefix in packet lineage')
  q=Path(parent['file']);q=q if q.is_absolute()else Path(packet_file).parent/q
  if file_sha256(q)!=parent['sha256']:raise ValueError('Changed actual numeric packet parent')
  packet_file=q;p=read_json(q)
def source_col(d,root):
 f=Path(d['file']);f=f if f.is_absolute()else root/f
 if file_sha256(f)!=d['sha256']:raise ValueError('Changed source numeric column')
 return np.memmap(f,dtype=d['dtype'],mode='r',offset=int(d.get('offset_bytes',0)),shape=tuple(d['shape']))
def evaluate(a):
 t=time.monotonic();out=Path(a.out)
 if out.exists():raise FileExistsError('Use a new immutable live-guard output directory')
 ml=Path(a.model);m=read_json(ml);mref=ref(ml)
 loader=PacketArrays(a.packet);p=loader.packet;n=p['row_count']
 arrays={k:loader.load(k)for k in ['kind','material_index','admission','fit_role','uv','native_pts','shot_index','E']}
 labels=np.asarray([x['label']for x in m['materials']]);material=arrays['material_index'];isboard=np.asarray([not v.startswith('ice.')for v in labels])
 prefix,auxref=find_overlay(a.packet,p);end=int(prefix['count'])
 rows=np.flatnonzero((np.arange(n)<end)&(arrays['kind']==1)&(arrays['admission']==1)&(arrays['fit_role']==1)&isboard[material])
 if not len(rows):raise ValueError('The original complete fixed2D cohort is empty')
 if a.baseline and len(rows)!=int(read_json(a.baseline)['rows']):raise ValueError('Complete fixed2D cohort count changed; rebind instead of silently comparing a subset')
 branch=source_col(prefix['arrays']['path'],Path(auxref['overlay']['file']).parent)[rows+int(prefix['source_offset'])]
 frozenrow=source_col(prefix['arrays']['row'],Path(auxref['overlay']['file']).parent)[rows+int(prefix['source_offset'])]
 station=np.zeros(len(rows))
 scene=NativeScene(as_event_view(m,'s01'));world,finite,reason=scene.world_rows(material[rows],branch,station)
 for mi,pi in np.unique(np.c_[material[rows],branch],axis=0):
  path=scene.path(int(mi),int(pi))
  if path.get('physical_point')is not True:raise ValueError('Fixed2D cohort contains sliding material')
  if not any(lo<=0<=hi for lo,hi in path.get('allowed_station_intervals',[[0.,1.]])):raise ValueError('Fixed2D station0 outside actual material domain')
 cg=CameraGraph(graph_from_joint(m));uv=arrays['uv'][rows];pts=arrays['native_pts'][rows];shot=arrays['shot_index'][rows];E=arrays['E'][rows].reshape(-1);mi=material[rows]
 pred=np.full_like(uv,np.nan);observed=np.full_like(uv,np.nan);rs=np.zeros(len(rows),bool);vis=rs.copy();obsvalid=rs.copy();bracket=rs.copy();cert=rs.copy();eq=np.full(len(rows),np.nan);target=labels[mi]
 for sn,s in enumerate(p['shot_ids']):
  ix=np.flatnonzero(shot==sn)
  if not len(ix):continue
  cam=cg.native_camera(s);local=cg.local_seconds(s,pts[ix]);w=world[ix].copy();w[~finite[ix]]=0.
  rr=frozenrow[ix]if s=='s01'else uv[ix,1]
  q=cam.at(w,local+cam.tau*(rr/1079.-.5));observed[ix]=q['uv'];obsvalid[ix]=finite[ix]&q['valid']&np.isfinite(rr)&(rr>=0)&(rr<=1079)
  q=cam.project_rs(w,local);pred[ix]=q['uv'];rs[ix]=finite[ix]&q['valid'];bracket[ix]=q['bracketed'];cert[ix]=q['certificate']['certified'];eq[ix]=q['root_equality_px']
  records=material_visibility_records(scene,mi[ix],branch[ix]);vis[ix]=render_visibility(cam,scene,w,records,local,q['uv'])
 valid=finite&rs&vis;distance=np.linalg.norm(pred-uv,axis=1);od=np.linalg.norm(observed-uv,axis=1);order=np.arange(len(rows));status=finite.astype('u1')|rs.astype('u1')*2|vis.astype('u1')*4
 cohort_hash=hashlib.sha256()
 for i in rows:cohort_hash.update(str(int(i)).encode()+b'\t'+p['source_row_ids'][int(i)].encode('utf8')+b'\n')
 def summary(ix):
  ix=np.asarray(ix,int)
  return dict(true_RS_rawUV=_same_denominator_statistics(distance,ix,valid,E,'true_RS_fixed2D_full_source_cohort'),observed_row_rawUV=_same_denominator_statistics(od,ix,obsvalid,E,'source_observed_row_fixed2D_full_source_cohort'),finite_invalid=int((~finite[ix]).sum()),true_RS_invalid=int((~rs[ix]).sum()),visibility_invalid=int((rs[ix]&~vis[ix]).sum()),RS_endpoint_not_bracketed=int((~bracket[ix]).sum()),RS_certificate_unknown=int((~cert[ix]).sum()))
 report=dict(schema='rinkfit.full_existing_fixed2D_source_anchor_guard.v1',model=mref,implementation=ref(__file__),packet=ref(a.packet),source_visibility_operator=ref(P/'calibration/rinkfit/src/rinkfit_joint/joint_replay.py'),parent_numeric_reference=auxref,rows=len(rows),full_original_parent_fixed2D_cohort=True,cohort_row_ID_sha256=cohort_hash.hexdigest(),source_value_column_descriptors={k:p['arrays'][k]for k in ['uv','native_pts','shot_index','material_index','kind','E']},raw_source_rows=n,station_semantics='physical_point material evaluated at station0; no sliding station approximation',source_entity_fixed_does_not_mean_world_coordinates_calibrated=True,observed_row_semantics='S01 frozen true parent complete prefix row; S02-S06 original actual sourceUVy',candidate_complete_checkpoint_claimed=False,statistical_sigma_or_independence_claimed=False,original_E_is_engineering_review_scale=True,new_curve_rows_projected=0,RGB_decoded=0,sourceUV_modified=False,overall=summary(order),by_shot={s:summary(np.flatnonzero(shot==sn))for sn,s in enumerate(p['shot_ids'])if np.any(shot==sn)},by_entity={s:summary(np.flatnonzero(target==s))for s in sorted(set(target.tolist()))},maximum_true_RS_equality_px=float(np.max(abs(eq[valid]),initial=0.)),quality_accepted=False)
 if a.baseline:
  b=read_json(a.baseline)
  if b['cohort_row_ID_sha256']!=report['cohort_row_ID_sha256']or b['source_visibility_operator']['sha256']!=report['source_visibility_operator']['sha256']:raise ValueError('Guard baseline must bind exact same cohort/operator')
  if b['source_value_column_descriptors']!=report['source_value_column_descriptors']:
   # Appending independently measured sources changes full-column descriptors.
   # Compare every actual original cohort value instead of confusing that append
   # with a modification to the original UV/PTS/material/operator.
   bp=Path(b['packet']['file'])
   if file_sha256(bp)!=b['packet']['sha256']:raise ValueError('Changed baseline packet')
   bl=PacketArrays(bp)
   for key in ('uv','native_pts','shot_index','material_index','kind','E'):
    before=np.ascontiguousarray(bl.load(key)[rows]);after=np.ascontiguousarray(arrays[key][rows])
    if before.dtype!=after.dtype or before.shape!=after.shape or before.tobytes()!=after.tobytes():raise ValueError('Original fixed-source cohort changed '+key)
   report['source_append_original_cohort_columns_verified_bit_exact']=True
  f=Path(b['distances']['file'])
  if file_sha256(f)!=b['distances']['sha256']:raise ValueError('Changed immutable baseline anchor residual cache')
  with np.load(f)as z:
   if not np.array_equal(z['packet_rows'],rows):raise ValueError('Baseline cohort order differs')
   bd=z['true_RS_distance'].copy();bv=(z['status']&7)==7;bo=z['observed_row_distance'].copy();bov=z['observed_row_valid'].astype(bool)
  def changes(ix):
   ix=np.asarray(ix,int);currentabove=(~valid[ix])|(distance[ix]>E[ix]);oldabove=(~bv[ix])|(bd[ix]>E[ix]);both=valid[ix]&bv[ix]
   return dict(N=len(ix),baseline_true_RS=b['overall']['true_RS_rawUV']if len(ix)==len(rows)else _same_denominator_statistics(bd,ix,bv,E,'baseline_true_RS_same_fixed2D_cohort'),candidate_true_RS=summary(ix)['true_RS_rawUV'],newly_above_original_E=int(np.sum(currentabove&~oldabove)),returned_inside_original_E=int(np.sum(~currentabove&oldabove)),newly_invalid=int(np.sum(~valid[ix]&bv[ix])),newly_valid=int(np.sum(valid[ix]&~bv[ix])),sum_squared_error_delta_on_common_valid=float(np.sum(distance[ix][both]**2-bd[ix][both]**2)),sum_squared_E_normalized_delta_on_common_valid=float(np.sum((distance[ix][both]**2-bd[ix][both]**2)/E[ix][both]**2)))
  report['baseline']=ref(a.baseline);report['packet_wrapper_may_differ_but_entire_fixed2D_cohort_ID_UV_PTS_material_E_exact']=True;report['same_source_comparison']=dict(overall=changes(order),by_shot={s:changes(np.flatnonzero(shot==sn))for sn,s in enumerate(p['shot_ids'])if np.any(shot==sn)},by_entity={s:changes(np.flatnonzero(target==s))for s in sorted(set(target.tolist()))})
  report['same_source_comparison']['by_shot_entity']=[dict(shot_id=s,feature_id=str(label),comparison=changes(np.flatnonzero((shot==sn)&(target==label))))for sn,s in enumerate(p['shot_ids'])for label in sorted(set(target[shot==sn].tolist()))]
  findings=[];worst_by_entity={}
  for i in order:
   if (not valid[i]and bv[i])or(valid[i]and bv[i]and distance[i]>bd[i]):
    findings.append(dict(packet_row=int(rows[i]),source_row_id=p['source_row_ids'][int(rows[i])],shot_id=p['shot_ids'][int(shot[i])],feature_id=str(target[i]),native_pts=int(pts[i]),baseline_error_px=float(bd[i])if bv[i]else None,candidate_error_px=float(distance[i])if valid[i]else None,original_E=float(E[i]),squared_error_increase=float(distance[i]**2-bd[i]**2)if valid[i]and bv[i]else None))
  for subject in findings:
   key=(subject['shot_id'],subject['feature_id']);old=worst_by_entity.get(key);rank=lambda x:(x['candidate_error_px']is None,x['squared_error_increase'] or 0)
   if old is None or rank(subject)>rank(old):worst_by_entity[key]=subject
  report['worsened_rows']=len(findings);report['worst_worsened_subject_by_entity']=list(worst_by_entity.values());report['all_worsened_rows_retained_in_compact_cache']=True
 out.mkdir(parents=True);f=out/'anchor_distances.npz';np.savez_compressed(f,packet_rows=rows.astype('<u4'),true_RS_distance=np.where(valid,distance,np.inf),observed_row_distance=np.where(obsvalid,od,np.inf),observed_row_valid=obsvalid,status=status,worsened_packet_rows=np.asarray([x['packet_row']for x in findings],dtype='<u4')if a.baseline else np.empty(0,'<u4'));report['distances']=ref(f);report['elapsed_seconds']=time.monotonic()-t
 if file_sha256(ml)!=mref['sha256']:raise ValueError('Candidate model changed during evaluation; use atomic immutable snapshot')
 write_json(out/'report.json',report);print(json.dumps(dict(report=ref(out/'report.json'),overall=report['overall'],comparison=report.get('same_source_comparison',{}).get('overall'),seconds=report['elapsed_seconds'])))
 return report

def evaluate_fixed_anchors(model,packet,output,baseline=None):
 """Evaluate every original fixed board point, retaining all invalid rows."""
 return evaluate(argparse.Namespace(model=model,packet=packet,out=output,baseline=baseline))

if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--model',required=True);q.add_argument('--packet',required=True);q.add_argument('--out',required=True);q.add_argument('--baseline');a=q.parse_args();evaluate(a)
