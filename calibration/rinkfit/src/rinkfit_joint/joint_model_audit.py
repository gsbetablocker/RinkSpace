"""Exhaustive solver-side physical binding audit; pixel semantics remain source-audited."""
from pathlib import Path
import argparse,gzip,json,re
from collections import Counter
import numpy as np
from .joint_model import as_event_view,validate_physical_domain,graph_from_joint
from .camera_graph import CameraGraph
from .io_utils import file_sha256,write_json
from .joint_metrics import genuine_query_operators

def audit(model_file,packet_file,out):
 model_file=Path(model_file);packet_file=Path(packet_file);out=Path(out)
 if out.exists():raise ValueError('immutable model audit exists')
 out.mkdir(parents=True);m=json.loads(model_file.read_text(encoding='utf8'));p=json.loads(packet_file.read_text(encoding='utf8'));cache={}
 def array(name):
  d=p['arrays'][name];q=Path(d['file']);q=q if q.is_absolute() else packet_file.parent/q;key=(str(q),d['sha256'])
  if key not in cache:
   if file_sha256(q)!=d['sha256']:raise ValueError('changed source array '+name)
   cache[key]=True
  return np.fromfile(q,d['dtype'],count=int(np.prod(d['shape'])),offset=d.get('offset_bytes',0)).reshape(d['shape'])
 a={k:array(k) for k in ('admission','fit_role','kind','material_index','path','native_pts','shot_index','uv')};active=(a['admission']==1)&(a['fit_role']==1);rr=np.flatnonzero(active);query_mask,_=genuine_query_operators(p,int(p['row_count']));queries=set(np.flatnonzero(query_mask).tolist());violations=[];surface=Counter();bykind=Counter();byshot=Counter();bylabel=Counter();width=Counter();target=Counter();printonly=Counter();board=Counter()
 def hit(i,why):violations.append(dict(joint_row=int(i),source_dataset_id=p['source_dataset_ids'][i],source_row_id=p['source_row_ids'][i],shot_id=p['shot_ids'][int(a['shot_index'][i])],native_pts=int(a['native_pts'][i]),source_uv=a['uv'][i].tolist(),reason=why))
 for i in rr:
  mi=int(a['material_index'][i]);pi=int(a['path'][i]);kind=int(a['kind'][i]);bykind[kind]+=1;byshot[p['shot_ids'][int(a['shot_index'][i])]]+=1
  if mi>=len(m['materials']):hit(i,'active_material_index_unbound');continue
  material=m['materials'][mi];label=material['label'];bylabel[label]+=1
  if pi>=len(material['paths']):hit(i,'active_finite_branch_unbound');continue
  q=material['paths'][pi];pk=int(q['kind']);plane='ice_z0' if pk in (0,1,4,5) else 'continuous_board_course' if pk in (2,3) else 'unknown';surface[plane]+=1
  if plane=='unknown':hit(i,'unsupported_or_unknown_depth_primitive')
  if re.search(r'(reflection|spectator|person|subtitle|stand_point|unknown_depth)',label,re.I):hit(i,'forbidden_physical_identity_label')
  if kind==1 and not q.get('physical_point',False):hit(i,'fixed_entity_bound_to_sliding_curve')
  if kind==1 and pk==2 and len(q.get('points',[]))!=1:hit(i,'entity_has_more_than_one_unfolded_material_coordinate')
  if kind==1 and q.get('geometry_block',-1)>=0:hit(i,'outer_entity_illegally_depends_on_print_shape')
  if kind==2 and i not in queries:hit(i,'genuine_profile_missing_frozen_query')
  if i in queries and kind not in (0,2):hit(i,'fixed_entity_illegally_uses_sliding_profile_query')
  if pk==3:
   expected=1. if label.startswith('board.top') else 0. if label.startswith('board.contact') else None
   if expected is None:hit(i,'board_role_not_designated_blue_upper_or_yellow_lower')
   elif q.get('geometry_block',-1)>=0 or q.get('fixed_height_fraction')!=expected:hit(i,'board_outer_boundary_not_at_shared_H_or_0')
   board[label]+=1
  if pk==2:
   ab=q.get('advertisement_block',-1)
   if ab<0 or len(m['blocks'][ab]['values'])!=4 or not m['blocks'][ab]['name'].startswith('ad/'):hit(i,'advertisement_not_on_continuous_unfolded_board_chart')
  stem=label.split(':')[0]
  if stem in {'ice.B1','ice.B2','ice.C','ice.C0','ice.A1','ice.A2','ice.A3','ice.A4'}:width[label]+=1
  if re.match(r'^ice.[FR][1-4]:',label):target[label]+=1
  ad=re.match(r'^Ad(8|9|10|11|12).',label)
  if ad and '.print.'not in label:printonly[label]+=1
 path_issues=[]
 for mi,material in enumerate(m['materials']):
  for pi,q in enumerate(material['paths']):
   if q['kind']==2 and not q.get('physical_point',False):
    xy=np.asarray(q.get('points',[]),float)
    if len(xy)<2 or np.max(np.linalg.norm(xy-xy[0],axis=1),initial=0)<1e-14:path_issues.append(dict(material=material['label'],path=pi,reason='non_entity_print_contour_degenerate',active_rows=int(bylabel[material['label']])))
   if q['kind']==3 and q.get('geometry_block',-1)>=0:path_issues.append(dict(material=material['label'],path=pi,reason='free_inner_board_height_in_physical_model',active_rows=int(bylabel[material['label']])))
 domain=validate_physical_domain(m);cg=CameraGraph(graph_from_joint(m));camera={s:cg.native_camera(s).optical_domain_certificate() for s in m['shots']};excluded=[]
 for i in (25537,34524):excluded.append(dict(original_index=i,source_row_id=p['source_row_ids'][i],admission=int(a['admission'][i]),fit_role=int(a['fit_role'][i]),active=bool(active[i])))
 with gzip.open(out/'illegal_bindings.jsonl.gz','wt',encoding='utf8') as f:
  for q in violations:f.write(json.dumps(q,ensure_ascii=False)+'\n')
 report=dict(schema='rinkfit.joint_solver_physical_binding_audit.v1',model=dict(path=str(model_file.resolve()),sha256=file_sha256(model_file)),packet=dict(path=str(packet_file.resolve()),sha256=file_sha256(packet_file)),raw_rows=int(p['row_count']),active_rows=len(rr),temporarily_held_admitted=int(np.sum((a['admission']==1)&~active)),active_by_shot=dict(byshot),active_by_kind=dict(bykind),physical_surface_counts=dict(surface),illegal_binding_count=len(violations),illegal_bindings='illegal_bindings.jsonl.gz',all_material_path_issues=path_issues,finite_geometry_domain=domain,optical_domain_by_shot=camera,s01_excluded_sources=excluded,genuine_profile_queries=len(queries),active_genuine_profiles=int(np.sum(active&query_mask)),active_kind0_genuine_profiles=int(np.sum(active&query_mask&(a['kind']==0))),query_presence_definition='Exact packet queries row identities; a sliding finite material kind0 may have a genuine Query',legacy_kind2_missing_query_is_illegal=True,width_midline_rows=dict(width),FR_material_boundary_rows=dict(target),board_designated_outer_rows=dict(board),Ad8_12_external_source_rows=dict(printonly),all_physical_model_materials=len(m['materials']),all_finite_paths=sum(len(x['paths']) for x in m['materials']),unknown_per_row_depth_parameters=False,measurement_semantics_are_source_evidence_required=True,pixel_identity_or_visibility_verified_by_this_solver_audit=False,solver_binding_passed=not violations and not any(q['active_rows'] for q in path_issues) and all(not q['active']for q in excluded),all_current_physical_domains_passed=domain['passed'] and all(c['passed'] for c in camera.values()),joint_fit_complete=False)
 write_json(out/'report.json',report);print(json.dumps(dict(active=report['active_rows'],illegal=len(violations),path_issues=len(path_issues),geometry_domain_passed=domain['passed'],geometry_violations=domain.get('violations',[]),report=str(out/'report.json')),ensure_ascii=False));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--packet',required=True);p.add_argument('--out',required=True);a=p.parse_args();audit(a.model,a.packet,a.out)
