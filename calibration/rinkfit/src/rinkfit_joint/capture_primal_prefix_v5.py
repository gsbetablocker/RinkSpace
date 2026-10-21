"""Exact selected complete capture parent plus real ice/Ad3 source suffix.
Physical objective remains v4. New analytic circles need fresh station/RS;
new polyline rows additionally need fresh finite cells. Never an AL resume.
"""
from pathlib import Path
from collections import Counter
import copy, gzip, hashlib, json
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .r_inner_prefix_v1 import array
from .accepted_aux_prefix import AUX,_prefix_hash
from .board_middle_append_v1 import checked,require,ref
from . import first4_true_rs_capture_audit_v5 as V4

SCHEMA='rinkfit.capture_v5_ice_shape_source_append_primal_prefix.v1'
SELECTION='rinkfit.root_selected_capture_v5_source_boundary.v1'
OBJECTIVE=V4.OBJECTIVE

def inspect_complete_capture_v4(source_packet,state_dir):
 pp=Path(source_packet).resolve();state=Path(state_dir).resolve();p=read_json(pp);seen=set();A={k:array(p,k,pp.parent,seen)for k in p['arrays']};active=(A['admission']==1)&(A['fit_role']==1);n=p['row_count'];mp=state/'model.json';m=read_json(mp);bp=state/'source_uv_capture_state_binding.json';b=read_json(bp);npth=state/'explicit_nuisance/manifest.json';nm=read_json(npth)
 require(b['schema']=='rinkfit.sourceUV_trueRS_capture_state_binding.v1'and b['objective_id']==OBJECTIVE and b['packet_sha256']==file_sha256(pp)and b['model_sha256']==file_sha256(mp),'Selected v4 complete source/model/objective identity')
 require(b['row_alpha_optimization_dof']==0 and b['stored_valid_rows_are_derived_true_RS_roots']and not b['formal_Query_closed'],'Capture-v4 cannot become formal explicit state')
 require(nm['raw_rows']==n and nm['packet_sha256']==file_sha256(pp),'Capture-v4 full raw auxiliary denominator')
 aux={};desc={}
 for k,(filename,dtype)in AUX.items():
  f=state/'explicit_nuisance'/filename;require(f.stat().st_size==n*np.dtype(dtype).itemsize and file_sha256(f)==nm['files'][filename]['sha256'],'Capture-v4 actual auxiliary file '+k);v=np.memmap(f,dtype=dtype,mode='r',shape=(n,));require(np.isfinite(v).all(),'Capture-v4 nonfinite auxiliary '+k);aux[k]=v;desc[k]=dict(**ref(f),dtype=dtype,shape=[n])
 base=V4._base()._base();verification=base.verify_checkpoint(state/'engineering_checkpoint.json',file_sha256(pp),m,aux,active,{q['row_index']for q in p['queries']},b);station_rows=verification.pop('station_rows');require(station_rows==set(map(int,np.flatnonzero(active&(A['kind']!=1)))),'Capture-v4 incomplete station parameter set')
 cells=V4.verify_cells(pp,state);require(cells['all_saved_stations_bitwise_verified']and cells['all_actual_Ceres_constant_and_bounds_records_checked'],'Capture-v4 cells are not actual graph state')
 maps=V4._maps(state/'engineering_checkpoint.json');discrete=maps[0];require(discrete['station_cell_state_sha256']==cells['complete_CP_cell_state_digest']and discrete['station_cell_policy_sha256']==cells['policy']['sha256'],'Capture-v4 discrete cell state differs')
 status=state/'derived_auxiliary_binding/RS_status.u8';ss=np.fromfile(status,'u1');require(ss.shape==(n,)and np.all(ss[active]==1),'Selected saved source RS status is not complete; independent root audit still required')
 return dict(schema='rinkfit.complete_capture_v4_primal_cell_inventory.v1',source_packet=ref(pp),model=ref(mp),checkpoint=ref(state/'engineering_checkpoint.json'),capture_state_binding=ref(bp),explicit_nuisance_manifest=ref(npth),actual_RS_status=ref(status),raw_rows=n,active_rows=int(active.sum()),physical_blocks=len(m['blocks']),physical_scalar_count=sum(len(x['values'])for x in m['blocks']),auxiliaries=desc,state_token=b['state_token'],objective_id=OBJECTIVE,complete_checkpoint_verification=verification,cell_state=cells,discrete_cell_state_sha256=discrete['station_cell_state_sha256'],discrete_cell_policy_sha256=discrete['station_cell_policy_sha256'],formal_Query_closed=False,original_W_likelihood_consumed=False,quality_accepted=False)


def validate_selection(selection_manifest,source_packet,state_dir):
 seen=set();sp=Path(selection_manifest).resolve();s=read_json(sp);actual=inspect_complete_capture_v4(source_packet,state_dir)
 require(s['schema']==SELECTION and s['selected_for_next_source_batch']is True and Path(s['state_directory']).resolve()==Path(state_dir).resolve(),'Root v5 source-boundary selection missing/different')
 for k in ('source_packet','model','checkpoint','capture_state_binding'):
  require(s[k]==actual[k],'Root selected complete capture differs '+k)
 require(s['state_token']==actual['state_token']and s['objective_id']==OBJECTIVE and s['cell_state_binding']==actual['cell_state']['binding'],'Root selected physical/cell token differs')
 audit=read_json(checked(s['independent_audit'],seen));native=read_json(checked(s['native_domain_receipt'],seen));checked(s['configuration'],seen)
 require(audit['schema']=='rinkfit.first4_true_RS_capture_independent_audit.v5'and audit['packet']==actual['source_packet']and audit['model']==actual['model'],'Selected parent independent full-source consumer differs')
 require(audit['complete_CP']['FNV_token_independently_recomputed']and audit['complete_CP']['state_token']==actual['state_token'],'Selected parent requires a same-token full independent audit')
 require(audit['all']['raw_source_2D_UV']['failed']==0 and audit['all']['finite_station_or_material_failed']==0 and audit['all']['material_parameter_closed_bound_failed']==0 and audit['all']['no_legal_RS_bracket']==0 and audit['all']['RS_whole_interval_uncertified']==0,'Selected complete parent full-source domain/RS audit fails')
 require(native['state_token']==actual['state_token']and native['g_max_positive']==0 and native['g_over_1e_minus8']==0 and native['h_max_abs']==0,'Selected exact native parent domain receipt differs/fails')
 require(s['quality_accepted']is False and s['formal_Query_closed']is False,'Numerical parent is not source quality/formal acceptance')
 return actual

def _same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def inspect_source_successor(canonical_source_packet,old_source_packet):
 src=Path(canonical_source_packet).resolve();oldpath=Path(old_source_packet).resolve();p=read_json(src);old=read_json(oldpath);n=old['row_count'];N=p['row_count'];seen=set();b=p['source_only_ice_shape_successor']
 require(N>n and checked(b['parent_packet'],seen)==oldpath,'Actual ice/shape source ancestry differs')
 require(b['schema']=='rinkfit.actual_ice_and_reviewed_shape_source_append.v1'and b['prototype_only']and not b['accepted_physical_parent_bound']and 'source_capture_complete_state_binding'not in p,'Source-only suffix falsely claims complete physical parent')
 q=read_json(checked(b['qualification'],seen));checked(b['source_consumer'],seen);require(q['consumer']==b['source_consumer']and q['parent_packet']==b['parent_packet']and not q['numeric_parent_bound']and not q['noise_W_issued'],'Actual source qualification/consumer differs')
 A={k:array(old,k,oldpath.parent,seen)for k in old['arrays']};B={k:array(p,k,src.parent,seen)for k in p['arrays']};require(set(A)==set(B),'Source array interface changed')
 proofs={}
 for k,a in A.items():
  require(a.shape[0]==n and B[k].shape[0]==N and _same(a,B[k][:n]),'Actual all18 old source/aux prefix changed '+k)
  proofs[k]=dict(parent=old['arrays'][k],canonical=p['arrays'][k],exact_old_prefix_sha256=_prefix_hash(a))
 for k in ('source_row_ids','source_family_ids','source_dataset_ids'):
  require(p[k][:n]==old[k]and len(p[k])==N,'Old source ID prefix changed '+k)
 for k in ('queries','groups'):require(p[k][:len(old[k])]==old[k],'Old literal Query/full-W prefix changed '+k)
 for g in old['groups']:
  if isinstance(g.get('W'),dict):checked(g['W'],seen)
 for k in ('stroke_pairs','independent_rows','independent_statistic_kind','joint_statistics_rank','source_station_domains','material_binding','shot_ids'):
  require(p.get(k)==old.get(k),'Old source mathematics/topology changed '+k)
 require(len(p['source_row_ids'][n:])==len(set(p['source_row_ids'][n:]))and not set(p['source_row_ids'][n:]).intersection(old['source_row_ids']),'New source ID collides with parent')
 for key,v in old.get('allowed_paths',{}).items():require(p.get('allowed_paths',{}).get(key)==v,'Old finite path authority changed')
 rawpath=checked(q['files']['raw_ledger'],seen);geopath=checked(q['files']['geometry_rows'],seen)
 with gzip.open(rawpath,'rt',encoding='utf8')as f:raw=[json.loads(s)for s in f if s.strip()]
 with gzip.open(geopath,'rt',encoding='utf8')as f:geo=[json.loads(s)for s in f if s.strip()]
 require(len(raw)==N-n==q['new_raw_rows']and len(geo)==q['new_active_rows']and p['source_row_ids'][n:]==[r['source_row_id']for r in raw],'Actual raw source denominator/order changed')
 model=read_json(checked(q['topology_model'],seen));active=(B['admission']==1)&(B['fit_role']==1);new=np.flatnonzero(active&(np.arange(N)>=n));queries={int(r['row_index']):r for r in p['queries']};kinds=Counter();rows=[];counts=Counter();newgroups=p['groups'][len(old['groups']):]
 require(set(r['source_row_id']for r in geo)==set(p['source_row_ids'][i]for i in new),'Qualified geometry complete denominator differs')
 require(sorted(i for g in newgroups for i in g['rows'])==new.tolist(),'New unknown-noise full group membership differs')
 for g in newgroups:require(g['rank']==0 and g['W']==[]and g['statistic_kind']=='constraint_only_unknown_noise'and not g['unknown_noise_source_has_statistical_precision'],'New source W/sigma invented')
 for j,r in enumerate(raw):
  i=n+j;ok=r['source_geometry_eligible'];require(bool(active[i])==ok and not r['point_correspondence']and not r['entity_station_fixed'],'Diagnostic/source curve promoted into fixed world point')
  require(int(B['native_pts'][i])==r['native_pts']and p['shot_ids'][int(B['shot_index'][i])]==r['shot_id'],'Actual source clock changed')
  require(np.array_equal(B['uv'][i],r['source_uv'])and np.array_equal(B['query_uv'][i],r['source_uv'])and np.array_equal(B['profile_normal'][i],r['normal']),'Actual original source UV/normal changed')
  require(int(B['kind'][i])==0 and int(B['material_index'][i])==r['joint_material_index']and int(B['path'][i])==0 and float(B['station'][i])==.5 and float(B['alpha'][i])==0.,'Unsupported new curve/initial correspondence')
  if not ok:continue
  path=model['materials'][int(B['material_index'][i])]['paths'][int(B['path'][i])];k=int(path['kind']);kinds[k]+=1;require(k in (1,2),'Unexpected new source path kind')
  query=queries[i];require(query['source_row_id']==r['source_row_id']and query['source_mean_is_original_qualified_UV']and not query.get('world_station_fixed',False),'New finite Query identity changed')
  require(query['alpha']==r['finite_query']['alpha']and query['native_uv']==r['finite_query']['native_uv']and query['normal']==r['normal']and len(query['alpha'])==query['actual_source_sample_count'],'Actual finite material Query changed')
  require(np.array_equal(B['alpha_bounds'][i],[query['alpha'][0],query['alpha'][-1]])and query['alpha'][0]<=0<=query['alpha'][-1],'New finite Query support changed')
  if k==2:require(query['actual_source_sample_count']==49 and path.get('closed',False),'New Ad3 row is not existing closed-polyline finite-profile source')
  rows.append(dict(row_index=int(i),source_row_id=r['source_row_id'],shot_id=r['shot_id'],native_pts=r['native_pts'],target_id=r['target_id'],material_index=int(B['material_index'][i]),path_kind=k,source_family_id=p['source_family_ids'][i]));counts[(r['shot_id'],r['native_pts'])]+=1
 require(kinds==Counter({1:962,2:8})and len(new)==970 and int(active.sum())==p['active_rows'],'Actual 962 analytic + 8 closed-polyline suffix differs')
 from .ice_shape_source_append_v1 import closure as actual_closure
 closure=read_json(checked(b['source_dependency_closure'],seen));groups,crows=actual_closure(old,p,B,raw,n);require(p['engineering_pixel_dependency_groups']['groups']==groups,'Actual old/new same-RGB dependence closure changed');require(closure['old_groups_not_split']and closure['not_noise_whitening'],'Actual RGB closure semantics changed')
 shapeproof=read_json(checked(q['shape_qualification'],seen));witness=read_json(checked(shapeproof['witness'],seen));position=read_json(checked(witness['source_position_review'],seen));rgb={}
 for r in raw:
  key=(r['shot_id'],r['native_pts']);value=r.get('current_native_receipt',{}).get('decoded_native_RGB_sha256',position['source_RGB']['decoded_native_RGB_sha256']);require(key not in rgb or rgb[key]==value,'Same actual exposure has inconsistent RGB identity');rgb[key]=value
 require(closure['rows']==[{**r,'actual_RGB_sha256':rgb[(r['shot_id'],r['native_pts'])]}for r in crows],'Actual per-frame same-RGB source dependence proof changed')
 return dict(schema='rinkfit.ice_shape_actual_prefix_numeric_inspection.v1',passed=True,canonical_source_packet=ref(src),old_source_packet=ref(oldpath),preserved_raw_rows=n,current_raw_rows=N,source_prefix_array_proofs=proofs,new_active_rows=len(new),new_active_path_kind_counts={str(k):v for k,v in sorted(kinds.items())},new_active_analytic_circle_rows=int(kinds[1]),new_active_polyline_rows=int(kinds[2]),new_polyline_row_indices=[r['row_index']for r in rows if r['path_kind']==2],qualified_rows=rows,source_qualification=ref(checked(b['qualification'],seen)),source_dependency_closure=ref(checked(b['source_dependency_closure'],seen)),all_old_source_ID_UV_Query_W_and_finite_paths_exact=True,all18_canonical_old_source_aux_prefix_bitexact=True,full_RGB_reader_replayed_in_this_numeric_inspection=False,selected_numeric_parent_bound=False,quality_accepted=False,solver_started=False)

def bind_selected_primal(canonical_source_packet,state_dir,old_source_packet,output,*,selection_manifest):
 from . import ice_shape_source_append_v1 as source
 src=Path(canonical_source_packet).resolve();oldpath=Path(old_source_packet).resolve();out=Path(output).resolve();require(not out.exists(),'Immutable v5 prefix output exists')
 canonical_guard=source.assert_current(src);inspection=inspect_source_successor(src,oldpath);canonical=read_json(src);old=read_json(oldpath);actual=validate_selection(selection_manifest,oldpath,state_dir);n=old['row_count'];N=canonical['row_count'];seen=set();oldmodel=read_json(checked(old['material_binding']['model_topology_ref'],seen));parentmodel=read_json(actual['model']['file'])
 require(len(parentmodel['blocks'])==len(oldmodel['blocks'])and [x['name']for x in parentmodel['blocks']]==[x['name']for x in oldmodel['blocks']],'Selected physical named topology changed')
 for k in ('materials','shots','rigs'):require(parentmodel[k]==oldmodel[k],'Selected physical topology changed '+k)
 require(actual['cell_state']['rows']==96077,'Selected parent cell denominator changed; this source transition requires explicit review')
 out.mkdir(parents=True);doc=copy.deepcopy(canonical);auxprefix={}
 for k,(filename,dtype)in AUX.items():
  previous=np.memmap(actual['auxiliaries'][k]['file'],dtype=dtype,mode='r',shape=(n,));suffix=array(canonical,k,src.parent,seen)[n:];f=out/(k+('.u32'if k=='path'else'.f64'));a=np.memmap(f,dtype=dtype,mode='w+',shape=(N,));a[:n]=previous;a[n:]=suffix;a.flush();del a;doc['arrays'][k]=dict(**ref(f),dtype=dtype,shape=[N]);auxprefix[k]=dict(selected_descriptor=actual['auxiliaries'][k],current_descriptor=doc['arrays'][k],selected_prefix_sha256=_prefix_hash(previous))
 write_json(out/'canonical_source_guard.json',canonical_guard);write_json(out/'actual_source_prefix_inspection.json',inspection)
 contract=dict(schema=SCHEMA,consumer=ref(__file__),canonical_source_packet=ref(src),canonical_source_guard=ref(out/'canonical_source_guard.json'),actual_source_prefix_inspection=ref(out/'actual_source_prefix_inspection.json'),old_source_packet=ref(oldpath),selected_complete_capture_v4=actual,root_selection=ref(selection_manifest),source_prefix_array_proofs=inspection['source_prefix_array_proofs'],preserved_raw_rows=n,current_raw_rows=N,first_new_raw_row=n,new_active_rows=inspection['new_active_rows'],new_active_path_kinds=[1,2],new_active_path_kind_counts=inspection['new_active_path_kind_counts'],new_active_analytic_circle_rows=inspection['new_active_analytic_circle_rows'],new_active_polyline_rows=inspection['new_active_polyline_rows'],new_polyline_row_indices=inspection['new_polyline_row_indices'],new_cell_rows_required=inspection['new_active_polyline_rows'],old_cell_rows=actual['cell_state']['rows'],expected_total_cell_rows=actual['cell_state']['rows']+inspection['new_active_polyline_rows'],parent_cell_state_binding=actual['cell_state']['binding'],parent_actual_graph_receipt=actual['cell_state']['actual_graph_receipt'],parent_cell_policy=actual['cell_state']['policy'],parent_cell_discrete_state_sha256=actual['discrete_cell_state_sha256'],parent_cell_discrete_policy_sha256=actual['discrete_cell_policy_sha256'],primal_auxiliary_prefixes=auxprefix,all_physical_scalar_values_exact=True,all_old_four_auxiliary_values_exact=True,all_old_source_ID_UV_Query_W_and_finite_paths_exact=True,old_cell_anchor_KKT_and_graph_bounds_preserved=True,new_analytic_station_and_RS_require_fresh_initialization=True,new_polyline_cells_require_fresh_initialization=True,source_append_requires_new_graph_and_policy_identity=True,old_graph_or_AL_checkpoint_resumed=False,only_new_rows_may_be_profiled=True,only_new_active_rows_fresh_profiled=True,formal_Query_closed=False,original_W_likelihood_consumed=False,quality_accepted=False)
 write_json(out/'capture_prefix_contract.json',contract);doc['source_capture_complete_state_binding']=dict(schema=SCHEMA,contract=ref(out/'capture_prefix_contract.json'),canonical_source_packet=ref(src),model=actual['model'],state_token=actual['state_token'],objective_id=OBJECTIVE,old_raw_rows=n,new_raw_rows=N-n,parent_cell_state_binding=actual['cell_state']['binding'],formal_Query_closed=False,quality_accepted=False);doc['source_manifests'] +=[ref(out/'capture_prefix_contract.json'),ref(__file__),ref(selection_manifest)]
 from .packet_storage import write_packet
 write_packet(out/'packet.json',doc);return dict(schema=SCHEMA,packet=ref(out/'packet.json'),model=actual['model'],contract=ref(out/'capture_prefix_contract.json'),selected_token=actual['state_token'],old_raw_rows=n,new_raw_rows=N-n,new_active_rows=inspection['new_active_rows'],new_active_analytic_circle_rows=962,new_active_polyline_rows=8,physical_scalar_values_preserved=True,old_cell_state_preserved=True,native_prefix_and_new_policy_qualification_required=True,solver_started=False)

def assert_capture_prefix_current(packet_path,*,return_verified_packet=False):
 from . import ice_shape_source_append_v1 as source
 pp=Path(packet_path).resolve();p=read_json(pp);b=p['source_capture_complete_state_binding'];seen=set();require(b['schema']==SCHEMA and b['objective_id']==OBJECTIVE and not b['formal_Query_closed']and not b['quality_accepted'],'V5 source-append prefix objective/semantics')
 c=read_json(checked(b['contract'],seen));require(c['schema']==SCHEMA and c['consumer']==ref(__file__)and c['only_new_rows_may_be_profiled']and c['only_new_active_rows_fresh_profiled']and not c['old_graph_or_AL_checkpoint_resumed'],'Wrong versioned v5 source-append consumer')
 canonicalpath=checked(c['canonical_source_packet'],seen);canonical=read_json(canonicalpath);source.assert_current(canonicalpath);oldpath=checked(c['old_source_packet'],seen);inspection=inspect_source_successor(canonicalpath,oldpath);require(inspection==read_json(checked(c['actual_source_prefix_inspection'],seen)),'Actual source-prefix inspection changed')
 selected=c['selected_complete_capture_v4'];state=Path(checked(selected['model'],seen)).parent;actual=validate_selection(checked(c['root_selection'],seen),oldpath,state);require(actual==selected,'Selected complete physical/cell state changed');n=c['first_new_raw_row'];N=p['row_count'];require(n==actual['raw_rows']==b['old_raw_rows']==c['preserved_raw_rows']and N==c['current_raw_rows']and N-n==b['new_raw_rows']and b['model']==actual['model']and b['state_token']==actual['state_token'],'Full v5 prefix/raw/model/token mismatch')
 require(inspection['source_prefix_array_proofs']==c['source_prefix_array_proofs']and c['new_active_path_kind_counts']=={'1':962,'2':8}and c['new_cell_rows_required']==8 and c['expected_total_cell_rows']==96085 and c['new_polyline_row_indices']==inspection['new_polyline_row_indices'],'Actual mixed-path initialization/cell policy differs')
 for k in canonical['arrays']:
  a=array(p,k,pp.parent,seen);v=array(canonical,k,canonicalpath.parent,seen)
  if k in AUX:
   previous=np.memmap(actual['auxiliaries'][k]['file'],dtype=actual['auxiliaries'][k]['dtype'],mode='r',shape=(n,));require(_same(a[:n],previous)and _same(a[n:],v[n:]),'V5 selected old four-aux/new init changed '+k)
  else:require(_same(a,v),'V5 wrapper changed canonical source array '+k)
 expected=copy.deepcopy(canonical);expected['arrays']=copy.deepcopy(p['arrays']);expected['source_capture_complete_state_binding']=copy.deepcopy(b);expected['source_manifests'] +=[ref(checked(b['contract'],seen)),ref(__file__),c['root_selection']];require(p==expected,'V5 prefix wrapper changed source/Query/W/dependencies mathematics')
 report=dict(schema=SCHEMA,passed=True,packet=ref(pp),selected_model=actual['model'],selected_complete_checkpoint=actual['checkpoint'],selected_token=actual['state_token'],objective_id=OBJECTIVE,preserved_raw_rows=n,new_raw_rows=N-n,new_active_rows=970,new_active_path_kind_counts={'1':962,'2':8},old_cell_rows=96077,new_cell_rows_required=8,parent_cell_state_binding=actual['cell_state']['binding'],all_physical_and_old_four_auxiliary_bytes_exact=True,all_old_source_and_current_Query_W_exact=True,old_cells_and_actual_graph_bounds_identity_verified=True,only_new_rows_may_be_profiled=True,old_graph_or_AL_resumed=False,formal_Query_closed=False,original_W_likelihood_consumed=False,quality_accepted=False)
 return(report,p)if return_verified_packet else report
