"""Append real signed ice interfaces and reviewed print boundaries together.

Source-only boundary. Frozen predecessor readers are replayed, all inherited
source/auxiliary scalars remain exact, and no W, world point or CP resume is issued.
"""
from pathlib import Path
from collections import Counter,defaultdict
import copy,gzip,hashlib,json,tempfile
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .packet_storage import write_packet
from .r_inner_prefix_v1 import array
from .geometry_source_append_v2 import append_geometry_sources
from . import ice_ring_source_v3 as ice
from . import source_shape_crosswalk_v1 as shape
from . import capture_primal_prefix_v4 as parent_guard
SCHEMA='rinkfit.actual_ice_and_reviewed_shape_source_append.v1'

def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def checked(r):
 p=Path(r.get('file',r.get('path'))).resolve();require(file_sha256(p)==r['sha256'],'Changed source append dependency '+str(p));return p

def compile_sources(ice_manifest,shape_report):
 im=read_json(ice_manifest);sr=read_json(shape_report)
 ice_receipt=ice.assert_current(ice_manifest)
 require(sr['producer']==ref(shape.__file__)and sr['actual_native_profiles_replayed']and not sr['quality_accepted'],'Shape consumer identity differs')
 with tempfile.TemporaryDirectory(prefix='rink_ice_shape_')as td:
  out=Path(td)/'shape';actual=shape.qualify(checked(sr['packet']),checked(sr['model']),checked(sr['witness']),out)
  require(actual==sr,'Shape actual RGB replay/semantic witness changed')
  expected=list(read_jsonl(out/'qualified_rows.jsonl'));stored=list(read_jsonl(Path(shape_report).parent/'qualified_rows.jsonl'));shape.validate_rows(stored,expected)
  require(stored==expected,'Shape full source membership changed')
 raw=list(read_jsonl(checked(im['files']['raw_ledger.jsonl.gz'])));geo=list(read_jsonl(checked(im['files']['geometry_rows.jsonl.gz'])));queries={r['source_row_id']:r for r in read_jsonl(checked(im['files']['finite_queries.jsonl.gz']))}
 w=read_json(checked(sr['witness']));pr=read_json(checked(w['source_position_review']));records=read_json(checked(pr['source_readings']));byid={r['source_row_id']:r for r in records}
 rgb={}
 for r in raw:
  key=(r['shot_id'],int(r['native_pts']));v=r['current_native_receipt']['decoded_native_RGB_sha256'];require(key not in rgb or rgb[key]==v,'Same native exposure pixels disagree');rgb[key]=v
 for r in stored:
  original=byid[r['reuses_original_source_row_id']];profile=original['current_source_profile'];mu=np.asarray(r['source_uv'],float);normal=np.asarray(r['normal'],float);alpha=np.asarray(profile['actual_source_normal_offsets_px'],float)
  require(original['original_UV']==r['source_uv']==r['current_uv']and np.array_equal(normal,profile['source_search_normal'])and profile['source_search_point_uv']==r['source_uv'],'Restored print original UV/normal differs')
  require(alpha.shape==(49,)and np.array_equal(alpha,np.arange(-24,25))and all(profile['actual_source_profile_valid']),'Restored print actual finite 49 profile differs')
  q=dict(alpha=alpha.tolist(),native_uv=(mu[None]+alpha[:,None]*normal[None]).tolist(),profile=profile['actual_source_luminance_profile'],profile_smoothed=profile['actual_source_smoothed_luminance_profile'],normal_uv=normal.tolist(),query_origin_uv=mu.tolist(),actual_selected_profile_offset=0.,source_mean_uv=mu.tolist(),material_transition=profile['material_transition'],sample_count=49,whole_actual_sampler_support_clear=True,formal_noise_statistics_ready=False,original_UV_retained_instead_of_candidate_q50=True)
  z={**r,'physical_surface':'board','finite_query':q,'source_noise_W_supplied':False,'source_admitted':True,'independence_claimed':False,'source_position_interval':dict(normal_interval_px=profile['observed_normal_interval_px'],definition='actual signed material width plus declared engineering margin; not sigma or confidence')}
  raw.append(z);geo.append(z);queries[z['source_row_id']]={'source_row_id':z['source_row_id'],**q};rgb[(r['shot_id'],int(r['native_pts']))]=pr['source_RGB']['decoded_native_RGB_sha256']
 ids=[r['source_row_id']for r in raw];require(len(set(ids))==len(ids),'Combined real source IDs duplicated')
 require(len(geo)==im['geometry_qualified_new_rows']+sr['qualified_rows'],'Complete source geometry denominator differs')
 return dict(raw=raw,geometry=geo,queries=queries,rgb=rgb,ice_receipt=ice_receipt,shape_receipt=actual,unknown_query_rows=im['unknown_query_rows'],all_ice_queries=im['all_source_query_rows'])

def source_continuation(ancestor_path,parent_path):
 a=read_json(ancestor_path);p=read_json(parent_path);require(a['row_count']==p['row_count'],'Source proof ancestor is not a numeric-only continuation')
 for k in ('source_row_ids','source_dataset_ids','source_family_ids','queries','groups','allowed_paths','material_binding','shot_ids','source_station_domains'):
  require(a.get(k)==p.get(k),'Source proof ancestor changed '+k)
 seen=set()
 for k in a['arrays']:
  if k not in ('station','row','alpha','path'):require(array(a,k,Path(ancestor_path).parent,seen).tobytes()==array(p,k,Path(parent_path).parent,seen).tobytes(),'Numeric continuation changed source '+k)

def finite_query(i,r,q,qualification):
 return dict(row_index=i,source_row_id=r['source_row_id'],alpha=q['alpha'],native_uv=q['native_uv'],normal=r['normal'],actual_acquisition_window_preserved=True,source_finite_profile_proof=qualification,actual_source_sample_count=len(q['alpha']),source_profile_material_transition=q['material_transition'],deterministic_query_is_not_tangential_noise_precision=True,source_mean_is_original_qualified_UV=True)

def closure(parent,p,A,raw,n):
 N=p['row_count'];d=np.arange(N,dtype=np.int64);active=(A['admission']==1)&(A['fit_role']==1)
 def root(i):
  while d[i]!=i:d[i]=d[d[i]];i=int(d[i])
  return i
 def union(a,b):
  a=root(a);b=root(b)
  if a!=b:d[max(a,b)]=min(a,b)
 oldgroups=parent['engineering_pixel_dependency_groups']['groups']
 for g in oldgroups:
  for i in g['rows'][1:]:union(g['rows'][0],i)
 newby=defaultdict(list)
 for j,r in enumerate(raw):
  if active[n+j]:newby[(r['shot_id'],int(r['native_pts']))].append(n+j)
 byframe=defaultdict(list)
 for i in np.flatnonzero(active):
  key=(p['shot_ids'][int(A['shot_index'][i])],int(A['native_pts'][i]))
  if key in newby:byframe[key].append(int(i))
 for rr in byframe.values():
  for i in rr[1:]:union(rr[0],i)
 affected={root(i)for rr in newby.values()for i in rr};components=defaultdict(list);oldids=defaultdict(list)
 for i in np.flatnonzero(active):
  if root(i)in affected:components[root(i)].append(int(i))
 groups=[]
 for g in oldgroups:
  if g['rows']and root(g['rows'][0])in affected:oldids[root(g['rows'][0])].append(g['id'])
  else:groups.append(copy.deepcopy(g))
 for k,rr in sorted(components.items()):groups.append(dict(id='ice_shape_same_RGB_closure:'+hashlib.sha256(np.asarray(rr,dtype='<u8').tobytes()).hexdigest()[:24],rows=rr,inherited_dependency_groups=oldids[k],source_exposure_pixel_dependence_preserved=True,source_independence_or_noise_precision_claimed=False))
 return groups,[dict(shot_id=k[0],native_pts=k[1],old_joint_rows=[i for i in byframe[k]if i<n],new_joint_rows=v,independence_claimed=False)for k,v in sorted(newby.items())]

def build(parent_packet,topology_model,ice_manifest,shape_report,output):
 out=Path(output).resolve();require(not out.exists(),'New immutable combined source boundary required')
 parent_guard.assert_capture_prefix_current(parent_packet);parent=read_json(parent_packet);x=compile_sources(ice_manifest,shape_report);source_continuation(read_json(ice_manifest)['parent_source_packet']['file'],parent_packet)
 model=read_json(topology_model);raw=x['raw'];geo=x['geometry'];n=parent['row_count'];oldids=set(parent['source_row_ids']);require(not oldids.intersection(r['source_row_id']for r in raw),'New source generation duplicates inherited source ID')
 out.mkdir(parents=True);files={}
 for key,rows in (('raw_ledger',raw),('geometry_rows',geo)):
  f=out/(key+'.jsonl.gz')
  with gzip.open(f,'wt',encoding='utf8')as stream:
   for r in rows:stream.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
  files[key]=ref(f)
 mapping={'semantic_to_material':{}}
 for r in raw:
  mi=int(r['joint_material_index']);require(model['materials'][mi]['label']==r['target_id'],'Actual current model physical material differs')
  mapping['semantic_to_material'][r['joint_material_id']]=dict(material_index=mi,geometry_ready=True)
 write_json(out/'material_map.json',mapping);files['material_map']=ref(out/'material_map.json')
 manifest=dict(schema=SCHEMA,consumer=ref(__file__),parent_packet=ref(parent_packet),topology_model=ref(topology_model),ice_qualification=ref(ice_manifest),shape_qualification=ref(shape_report),shape_qualified_rows=ref(Path(shape_report).parent/'qualified_rows.jsonl'),files=files,new_raw_rows=len(raw),new_active_rows=len(geo),ice_all_query_rows=x['all_ice_queries'],ice_queries_without_mean_retained_in_source=x['unknown_query_rows'],numeric_parent_bound=False,noise_W_issued=False)
 write_json(out/'qualification.json',manifest);qref=ref(out/'qualification.json')
 append_geometry_sources(parent_packet,files['geometry_rows']['file'],files['raw_ledger']['file'],files['material_map']['file'],topology_model,out/'append',out/'qualification.json',write_provisional_W=False)
 p=read_json(out/'append/packet.json');seen=set();A={k:np.asarray(array(p,k,out/'append',seen)).copy()for k in p['arrays']};newgroups=defaultdict(list);qs=list(parent['queries'])
 for j,r in enumerate(raw):
  i=n+j
  if not r['source_geometry_eligible']:continue
  q=x['queries'][r['source_row_id']];A['alpha_bounds'][i]=[q['alpha'][0],q['alpha'][-1]];qs.append(finite_query(i,r,q,qref));newgroups[r['source_family_id']].append(i)
 f=out/'alpha_bounds.bin';A['alpha_bounds'].tofile(f);p['arrays']['alpha_bounds']={**p['arrays']['alpha_bounds'],**ref(f)};p['queries']=qs
 for family,rr in sorted(newgroups.items()):p['groups'].append(dict(id='ice_shape_geometry:'+family,source_group_id=family,rows=rr,rank=0,W=[],statistic_kind='constraint_only_unknown_noise',unknown_noise_source_refs=[qref],unknown_noise_source_has_statistical_precision=False,E_RS_Query_and_finite_material_constraints_retained=True))
 groups,crows=closure(parent,p,A,raw,n);dep=copy.deepcopy(parent['engineering_pixel_dependency_groups']);dep['groups']=groups;write_json(out/'same_RGB_closure.json',dict(schema='rinkfit.actual_ice_shape_same_RGB_closure.v1',rows=[{**r,'actual_RGB_sha256':x['rgb'][(r['shot_id'],r['native_pts'])]}for r in crows],not_noise_whitening=True,old_groups_not_split=True));dep['ice_shape_same_RGB_closure']=ref(out/'same_RGB_closure.json');p['engineering_pixel_dependency_groups']=dep
 binding=dict(schema=SCHEMA,parent_packet=ref(parent_packet),qualification=qref,source_dependency_closure=ref(out/'same_RGB_closure.json'),source_consumer=ref(__file__),prototype_only=True,accepted_physical_parent_bound=False,new_raw_rows=len(raw),new_active_geometry_rows=len(geo),new_Query_count=len(geo),source_noise_unknown=True,placeholder_source_sigma_not_a_1px_noise_claim=True,formal_Query_W_capture_mode_consumption_complete=False)
 if 'source_capture_complete_state_binding'in p:binding['archived_parent_capture_binding']=p.pop('source_capture_complete_state_binding')
 p['source_only_ice_shape_successor']=binding;p['all_new_source_nuisance_requires_fresh_physical_capture']=True;p['constraint_only_unknown_noise_rows']=sum(len(g['rows'])for g in p['groups']if g.get('statistic_kind')=='constraint_only_unknown_noise');p['constraint_only_unknown_noise_groups']=sum(g.get('statistic_kind')=='constraint_only_unknown_noise'for g in p['groups']);p['source_manifests']+=[qref,ref(__file__),ref(out/'same_RGB_closure.json')]
 p['physical_noise_likelihood_claimed']=False;p['formal_new_source_statistical_optimum_claimed']=False;write_packet(out/'packet.json',p)
 report=validate_payload(p,Path(parent_packet),x,out/'packet.json');write_json(out/'actual_new_source_guard.json',report);return report

def validate_payload(p,parent_path,x,packet_path):
 parent=read_json(parent_path);b=p['source_only_ice_shape_successor'];manifest=read_json(checked(b['qualification']));require(manifest['consumer']==ref(__file__)and b['source_consumer']==ref(__file__),'Wrong actual combined source consumer');require('source_capture_complete_state_binding'not in p and b['prototype_only']and not b['accepted_physical_parent_bound'],'Source-only append inherited stale numeric capture identity')
 raw=x['raw'];geo=x['geometry'];n=parent['row_count'];N=n+len(raw);require(p['row_count']==N,'Combined complete raw denominator changed');seen=set();A={k:array(p,k,Path(packet_path).parent,seen)for k in p['arrays']}
 for k in parent['arrays']:require(A[k][:n].tobytes()==array(parent,k,parent_path.parent,seen).tobytes(),'Inherited actual source/aux prefix changed '+k)
 for k in ('source_row_ids','source_dataset_ids','source_family_ids'):require(p[k][:n]==parent[k]and len(p[k])==N,'Inherited source identity prefix changed '+k)
 for k in ('queries','groups'):require(p[k][:len(parent[k])]==parent[k],'Inherited full Query/W changed '+k)
 for k in ('stroke_pairs','independent_rows','independent_statistic_kind','joint_statistics_rank','source_station_domains','material_binding','shot_ids'):require(p.get(k)==parent.get(k),'Inherited source mathematics/topology changed '+k)
 require(p['source_row_ids'][n:]==[r['source_row_id']for r in raw]and len(set(p['source_row_ids'][n:]))==len(raw)and not set(p['source_row_ids'][n:]).intersection(parent['source_row_ids']),'New actual source IDs differ/duplicated')
 expected=[];admitted=[];allowed=copy.deepcopy(parent['allowed_paths']);model=read_json(checked(manifest['topology_model']))
 for j,r in enumerate(raw):
  i=n+j;ok=r['source_geometry_eligible'];require(not r['point_correspondence']and not r['entity_station_fixed'],'Source curve promoted to a fixed world point')
  require(int(A['admission'][i])==int(ok)and int(A['fit_role'][i])==1,'Unknown source or diagnostic row promoted into fit');require(int(A['native_pts'][i])==r['native_pts']and p['shot_ids'][int(A['shot_index'][i])]==r['shot_id'],'New native source clock/shot changed')
  require(np.array_equal(A['uv'][i],r['source_uv'])and np.array_equal(A['query_uv'][i],r['source_uv'])and np.array_equal(A['profile_normal'][i],r['normal']),'Actual original source UV/normal changed');require(int(A['kind'][i])==0 and int(A['material_index'][i])==r['joint_material_index'],'Actual source circle/layer/shape material changed')
  require(float(A['alpha'][i])==0. and float(A['station'][i])==.5 and int(A['path'][i])==0,'New source latent initialized by an unsupported correspondence')
  require(float(A['E'][i])==2. and float(A['source_sigma'][i])==1. and int(A['stroke_pair_index'][i])==-1 and int(A['stroke_source_side'][i])==-1,'New geometry placeholder or stroke semantics changed')
  if ok:
   q=x['queries'][r['source_row_id']];require(q['alpha'][0]<=0<=q['alpha'][-1],'Actual finite source Query excludes preserved original mean');require(np.array_equal(A['alpha_bounds'][i],[q['alpha'][0],q['alpha'][-1]]),'Original finite Query bounds changed');expected.append(finite_query(i,r,q,b['qualification']));admitted.append(i);allowed[str(i)]=list(range(len(model['materials'][r['joint_material_index']]['paths'])))
  else:require(np.array_equal(A['alpha_bounds'][i],[0,0]),'Diagnostic row gained a finite fitting window')
 require(p['queries'][len(parent['queries']):]==expected,'Actual original finite Query literal or semantic definition changed');require(p['allowed_paths']==allowed,'Actual finite material path branches changed')
 added=p['groups'][len(parent['groups']):];rr=[]
 for g in added:require(g['rank']==0 and g['W']==[]and g['statistic_kind']=='constraint_only_unknown_noise'and not g['unknown_noise_source_has_statistical_precision'],'New source noise W/rank fabricated');rr+=g['rows']
 require(sorted(rr)==admitted,'Complete new unknown-noise row partition changed')
 groups,crows=closure(parent,p,A,raw,n);require(p['engineering_pixel_dependency_groups']['groups']==groups,'Actual same-RGB dependence closure changed');c=read_json(checked(b['source_dependency_closure']));require(c['rows']==[{**r,'actual_RGB_sha256':x['rgb'][(r['shot_id'],r['native_pts'])]}for r in crows],'Actual source RGB/dependence proof changed')
 active=(A['admission']==1)&(A['fit_role']==1);counts=Counter((int(A['shot_index'][i]),int(A['native_pts'][i]))for i in np.flatnonzero(active));require(max(counts.values())<=500,'Actual fitted per-native-frame count exceeds 500');require(p['active_rows']==int(active.sum()),'Complete active denominator changed')
 return dict(schema='rinkfit.actual_ice_shape_source_consumer_receipt.v1',passed=True,packet=ref(packet_path),parent_packet=ref(parent_path),qualification=b['qualification'],new_actual_native_mean_rows=len(raw),new_active_geometry_rows=len(admitted),new_finite_Queries=len(expected),ice_unknown_queries_preserved_in_parent_generation=x['unknown_query_rows'],rows=N,active_rows=p['active_rows'],old_all18_source_aux_prefix_exact=True,old_W_Query_domains_exact=True,old_engineering_groups_not_split=True,max_active_rows_per_native_frame=max(counts.values()),new_W_rank=0,source_noise_sigma_issued=False,numeric_parent_bound=False,solver_started=False)

def assert_current(packet_path,*,return_context=False):
 path=Path(packet_path).resolve();p=read_json(path);b=p['source_only_ice_shape_successor'];parent_path=checked(b['parent_packet']);parent_guard.assert_capture_prefix_current(parent_path);m=read_json(checked(b['qualification']));x=compile_sources(checked(m['ice_qualification']),checked(m['shape_qualification']));source_continuation(read_json(checked(m['ice_qualification']))['parent_source_packet']['file'],parent_path)
 for key in ('raw_ledger','geometry_rows'):require(list(read_jsonl(checked(m['files'][key])))==x['raw'if key=='raw_ledger'else'geometry'],'New source qualification full row identity changed')
 receipt=validate_payload(p,parent_path,x,path)
 return (receipt,p,x,parent_path)if return_context else receipt
