"""Actual first4 engineering packet from the current source generation.
Original S01 numeric arrays and W are copied exactly. New source E is an
engineering pixel budget; unknown joint source W stays rank zero explicitly.
"""
from pathlib import Path
from collections import Counter,defaultdict
from copy import deepcopy
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .packet_storage import write_packet
from .source_generation_input_v2 import QualifiedSourceGenerationInput

def _ref(p):
 p=Path(p).resolve();return {'file':str(p),'sha256':file_sha256(p)}
def build_first4_packet(s01_packet,input_binding,query_manifest,statistics_manifest,model_path,material_map,out,*,dependency_closure,engineering_E=2.):
 src=Path(s01_packet).resolve();out=Path(out).resolve()
 if out.exists():raise ValueError('New immutable native packet output required')
 qin=QualifiedSourceGenerationInput(input_binding);rows=list(qin.rows());p=read_json(src);model=read_json(model_path);mapping=read_json(material_map)
 qm=read_json(query_manifest);sm=read_json(statistics_manifest);closure=read_json(dependency_closure)
 row_units=closure['source_row_to_dependency_component']
 if closure.get('input_binding',{}).get('sha256',qin.identity)!=qin.identity:raise ValueError('Dependency closure source binding differs')
 if qm['statistics_manifest']['sha256']!=file_sha256(statistics_manifest):raise ValueError('Query statistics binding differs')
 if qm['input_binding']['input_binding']['sha256']!=qin.identity:raise ValueError('Query source binding differs')
 table_ref=qm['queries'];qp=Path(table_ref.get('path',table_ref.get('file','')))
 if file_sha256(qp)!=table_ref['sha256']:raise ValueError('Query table changed')
 qt={r['source_row_id']:r for r in read_jsonl(qp)}
 if set(qt)!={r['source_row_id']for r in rows}:raise ValueError('Query full row denominator differs')
 maps=mapping['semantic_to_material'];rowmaps=mapping.get('row_to_material',{});N0=int(p['row_count']);N=len(rows);checked={}
 def locate(ref):
  f=Path(ref['file']);f=f if f.is_absolute()else src.parent/f
  if str(f)not in checked:
   if file_sha256(f)!=ref['sha256']:raise ValueError('S01 numeric dependency changed')
   checked[str(f)]=True
  return f.resolve()
 A={k:np.fromfile(locate(v),v['dtype']).reshape(v['shape'])for k,v in p['arrays'].items()}
 prefix={k:a.tobytes()for k,a in A.items()};allowed=deepcopy(p.get('allowed_paths',{}));queries=deepcopy(p.get('queries',[]));source_domains=deepcopy(p.get('source_station_domains',{}));domainrows=source_domains.get('rows',[])
 extra={k:[]for k in A};decisions=[];active_new=[];unknownunits=defaultdict(list);parent={}
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def union(a,b):
  a=find(a);b=find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 for r in rows:
  q=qt[r['source_row_id']];ids=[v['source_statistics_group_id']for v in q.get('engineering_family_dependency_ids',[])];ids=ids or ['sourcefamily:'+r['source_family_id']]
  for x in ids[1:]:union(ids[0],x)
 for j,r in enumerate(rows):
  i=N0+j;rid=r['source_row_id'];q=qt[rid];desc=rowmaps.get(rid,maps.get(r.get('joint_material_id'),{}))
  if rid in rowmaps:
   for key in ('source_uv','native_pts','shot_id','source_packet_sha256'):
    if key not in desc or desc[key]!=r[key]:raise ValueError('Partial physical alias redirects current source '+key)
  mi=int(desc.get('material_index')or 0);pi=int(desc.get('path_index')or 0);reasons=[]
  identity=r.get('physical_identity_status')=='typed_identity_current_pixel_supported' and r.get('source_admitted')is not False
  if not identity:reasons.append('current_source_identity_or_full_footprint_not_admitted')
  ready=bool(desc.get('geometry_ready',False))
  if not ready:reasons.append(desc.get('reason','no_source_confirmed_native_material_crosswalk'))
  ok=identity and ready
  if ok and (mi<0 or mi>=len(model['materials'])or pi<0 or pi>=len(model['materials'][mi]['paths'])):raise ValueError('Physical source map outside true model')
  corner=bool(q['kind']==1 or r.get('point_correspondence'))
  if ok and corner and not model['materials'][mi]['paths'][pi].get('physical_point',False):raise ValueError('Fixed entity corner became sliding')
  uv=np.asarray(q['engineering_pixel_uv'],float)
  if uv.shape!=(2,)or not np.isfinite(uv).all():raise ValueError('Nonfinite actual source UV')
  has_query=ok and not corner and bool(q['query_available'])
  quv=q['query_uv']if has_query else uv.tolist();normal=q['profile_normal']if has_query else [0.,0.];alpha=float(q['initial_alpha'])if has_query else 0.;bounds=q['alpha_bounds']if has_query else [0.,0.]
  vals=dict(uv=uv,native_pts=int(r['native_pts']),shot_index=int(r['shot_id'][1:])-1,kind=1 if corner else 0,material_index=mi,path=pi,admission=1 if ok else 0,fit_role=1,E=float(engineering_E),station=0. if corner else .5,row=float(np.clip(uv[1],0,1079)),alpha=alpha,source_sigma=1.,profile_normal=normal,query_uv=quv,alpha_bounds=bounds,stroke_pair_index=-1,stroke_source_side=-1)
  if set(A)!=set(vals):raise ValueError('Native source ABI array change requires explicit implementation: '+str(set(A)^set(vals)))
  for k in A:extra[k].append(vals[k])
  if ok:
   active_new.append(i);paths=desc.get('allowed_paths',[pi]);allowed[str(i)]=paths
   if has_query:queries.append({**deepcopy(q['query']),'row_index':i})
   intervals=desc.get('allowed_station_intervals')
   if intervals and not corner:domainrows.append({'row_index':i,'source_row_id':rid,'path_index':pi,'allowed_station_intervals':intervals})
   ids=[v['source_statistics_group_id']for v in q.get('engineering_family_dependency_ids',[])]or['sourcefamily:'+r['source_family_id']]
   unknownunits[row_units.get(rid,'fixed-entity:'+rid if corner else find(ids[0]))].append(i)
  decisions.append({'row_index':i,'source_row_id':rid,'shot_id':r['shot_id'],'native_pts':int(r['native_pts']),'target_id':r.get('target_id'),'material_index':mi,'path_index':pi,'active':ok,'reasons':reasons,'kind':1 if corner else 0,'original_source_uv':r['source_uv'],'audited_current_uv':r.get('current_uv'),'engineering_pixel_uv':uv.tolist(),'engineering_pixel_basis':q['engineering_pixel_basis'],'query_available':has_query,'query_source_available':bool(q['query_available']),'query_unavailable_reason':q.get('query_unavailable_reason'),'actual_source_components':q.get('source_component_ids',[]),'engineering_family_dependency_ids':q.get('engineering_family_dependency_ids',[]),'engineering_exposure_id':q['engineering_exposure_id'],'E_is_sigma':False,'source_noise_W_supplied':False})
 out.mkdir(parents=True);arrays={}
 for k,a in A.items():
  tail=np.asarray(extra[k],dtype=a.dtype);b=np.concatenate((a,tail))
  if b[:N0].tobytes()!=prefix[k]:raise ValueError('S01 prefix bytes changed')
  f=out/(k+'.bin');b.tofile(f);arrays[k]={'file':str(f),'sha256':file_sha256(f),'dtype':b.dtype.str,'shape':list(b.shape),'layout':'C','byte_order':'little'}
 groups=[]
 for g in p['groups']:
  W=g['W'];groups.append({**g,'W':{**W,'file':str(locate(W))}if isinstance(W,dict)else W})
 refs=[_ref(input_binding),_ref(query_manifest),_ref(statistics_manifest),_ref(material_map),_ref(model_path),_ref(qp),_ref(dependency_closure)]
 all_source_refs=refs+[{'file':str(path),'sha256':sha}for path,sha in sorted(qin._refs.items(),key=lambda x:str(x[0]))]
 all_source_refs=list({(v['file'],v['sha256']):v for v in all_source_refs}.values())
 for gid,ii in sorted(unknownunits.items()):groups.append({'id':'first4_unknown_joint_W:'+gid,'source_group_id':gid,'rows':ii,'rank':0,'W':[],'statistic_kind':'constraint_only_unknown_noise','unknown_noise_source_refs':refs[:3],'source_noise_covariance_claimed':False,'engineering_E_separate':True})
 oldactive=(A['admission']==1)&(A['fit_role']==1)
 pixel_units=defaultdict(list)
 for i in np.flatnonzero(oldactive):
  label=model['materials'][int(A['material_index'][i])]['label'];family='board'if label.startswith('board.')else'advertisement'if label.startswith(('Ad','advertisement'))else'ice'
  pixel_units['S01_original_exposure:'+str(int(A['native_pts'][i]))+':'+family].append(int(i))
 for unit,ii in unknownunits.items():pixel_units['new_source_dependency:'+unit].extend(ii)
 dependency_policy={'schema':'rinkfit.engineering_pixel_dependency_groups.v1','policy_id':'source_raw_dependency_components_v2','source_dependency_closure':_ref(dependency_closure),'groups':[{'id':unit,'rows':ii}for unit,ii in sorted(pixel_units.items())],'normalization_order':['shot','physical_surface_family','native_exposure','actual_source_dependency_component','rows_in_component'],'source_dependency_components_are_not_independent_noise_samples':True,'S01_original_exposure_uniform_rows_retained':True}

 table=[{'index':i,'label':m['label'],'paths':[{'index':j,'id':path['id'],'physical_point':bool(path.get('physical_point',False))}for j,path in enumerate(m['paths'])]}for i,m in enumerate(model['materials'])]
 binding={'schema':'rinkfit.joint_material_binding.v1','model_material_prefix_count':len(table),'materials':table,'source_material_map':_ref(material_map),'model_topology_ref':_ref(model_path),'topological_identity_table_sha256':object_sha256(table),'numeric_model_values_are_part_of_identity_binding':False,'allow_append_only_model_material_suffix':True}
 result={**p,'shot_ids':['s01','s02','s03','s04'],'row_count':N0+N,'arrays':arrays,'source_dataset_ids':p['source_dataset_ids']+['first4_actual_source:'+r['shot_id']for r in rows],'source_row_ids':p['source_row_ids']+[r['source_row_id']for r in rows],'source_family_ids':p['source_family_ids']+[r['source_family_id']for r in rows],'groups':groups,'queries':queries,'allowed_paths':allowed,'active_rows':int(oldactive.sum()+len(active_new)),'new_source_start':N0,'new_source_statistics_rank':0,'joint_statistics_rank':int(p['current_original_W_rank']),'material_binding':binding,'recapture_rows':active_new,'source_manifests':p['source_manifests']+all_source_refs,'initialization_objective_only':True,'physical_source_audit_complete':False,'source_noise_statistics_supplied_for_appended_rows':False,'formal_new_source_statistical_optimum_claimed':False,'first4_source_generation':{'input_binding':_ref(input_binding),'statistics_manifest':_ref(statistics_manifest),'query_manifest':_ref(query_manifest),'source_profile_components_are_read_not_reconstructed':True,'new_source_E_is_engineering_not_sigma':True,'source_sigma_suffix_is_unused_ABI_storage_not_statistic':True,'full_W_ready':False,'all_unavailable_source_rows_retained':True},'engineering_pixel_dependency_groups':dependency_policy,'engineering_scope':{'schema':'rinkfit.engineering_active_shots.v1','shot_ids':['s01','s02','s03','s04'],'three_rigs_one_scene':True}}
 if domainrows:
  source_domains['rows']=domainrows;source_domains.setdefault('schema','rinkfit.source_row_station_domains.v1');source_domains.setdefault('source_refs',[]).extend(refs[:3]);result['source_station_domains']=source_domains
 partition=np.zeros(N0+N,dtype=np.int16);partition[np.asarray(result['independent_rows'],int)]+=1
 for g in groups:partition[g['rows']]+=1
 expected=np.r_[oldactive,np.asarray(extra['admission'],bool)]
 if not np.array_equal(partition,expected.astype(np.int16)):raise ValueError('Full active source W/unknown partition mismatch')
 write_packet(out/'packet.json',result);write_jsonl(out/'source_membership.jsonl.gz',decisions)
 report={'schema':'rinkfit.first4_actual_engineering_packet.v1','packet':_ref(out/'packet.json'),'input_binding':_ref(input_binding),'S01_prefix_rows':N0,'S01_active':int(oldactive.sum()),'S01_all_numeric_arrays_byteexact':True,'S01_original_W_unchanged':True,'full_new_rows':N,'new_active':len(active_new),'new_inactive':N-len(active_new),'new_kind_counts':dict(Counter(d['kind']for d in decisions if d['active'])),'new_active_Query':sum(d['query_available']for d in decisions),'new_unavailable_Query_retained_as_sourceUV_pixel':sum(d['active']and not d['query_available']for d in decisions),'unknown_joint_W_dependency_groups':len(unknownunits),'formal_rank_preserved':result['joint_statistics_rank'],'full_W_not_invented':True,'dependency_closure':_ref(dependency_closure),'engineering_pixel_dependency_components':len(pixel_units),'dependency_policy_id':dependency_policy['policy_id'],'E':float(engineering_E),'E_is_sigma':False,'new_active_by_shot':dict(Counter(d['shot_id']for d in decisions if d['active'])),'all_failed_source_rows_kept_in_full_denominator':True,'sourceUV_changed_by_fit_residual':False,'source_profile_support_components_consumed':True,'quality_accepted':False}
 write_json(out/'receipt.json',report);return report
