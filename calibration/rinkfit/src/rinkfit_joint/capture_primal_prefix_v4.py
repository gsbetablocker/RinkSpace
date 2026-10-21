"""Exact capture-v4 source-append primal prefix; never a graph/AL resume.
Preserves the selected full physical/scalar and cell optimizer state identity.
"""
from pathlib import Path
import copy
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .r_inner_prefix_v1 import array
from .accepted_aux_prefix import AUX,_prefix_hash
from .board_middle_append_v1 import checked,require,ref
from . import first4_true_rs_capture_audit_v4 as V4

SCHEMA='rinkfit.capture_v4_source_append_primal_prefix.v1'
SELECTION='rinkfit.root_selected_capture_v4_source_boundary.v1'
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
 require(s['schema']==SELECTION and s['selected_for_next_source_batch']is True and Path(s['state_directory']).resolve()==Path(state_dir).resolve(),'Root v4 source-boundary selection missing/different')
 for k in('source_packet','model','checkpoint','capture_state_binding'):
  require(s[k]==actual[k],'Root selected capture-v4 differs at '+k)
 require(s['state_token']==actual['state_token']and s['objective_id']==OBJECTIVE and s['cell_state_binding']==actual['cell_state']['binding'],'Root selected physical/cell token differs')
 audit=read_json(checked(s['independent_audit'],seen));native=read_json(checked(s['native_domain_receipt'],seen));checked(s['configuration'],seen)
 require(audit['packet']==actual['source_packet']and audit['complete_CP']['FNV_token_independently_recomputed'],'Selected v4 independent full source audit differs')
 if audit['complete_CP']['state_token']==actual['state_token']:
  require(audit['model']==actual['model'],'Selected v4 independent model differs')
 else:
  # Reuse a complete independent forward only where the actual model/source
  # and every unchanged auxiliary byte are identical; changed rows require
  # the root's actual same-token delta replay, not a name/cost comparison.
  delta=read_json(checked(s['independent_delta_audit'],seen))
  for dr in delta['dependencies']:checked(dr,seen)
  require(delta['schema']=='rinkfit.actual_same_physics_complete_CP_auxiliary_delta_audit.v1'and delta['before_state_token']==audit['complete_CP']['state_token']and delta['after_state_token']==actual['state_token']and delta['complete_CP']==actual['complete_checkpoint_verification'],'Actual final delta/full CP identity differs')
  require(delta['physical_model_and_projection_metadata_exact']and delta['all_source_arrays_unchanged_by_SHA']and delta['all_actual_saved_auxiliaries_and_FNV_reverified']and audit['model']['sha256']==actual['model']['sha256']and read_json(checked(audit['model'],seen))==read_json(actual['model']['file']),'Full source forward cannot be reused across changed physics/source')
  priorstate=Path(audit['model']['file']).parent;allchanges=np.zeros(actual['raw_rows'],bool)
  for k,(filename,dtype)in AUX.items():
   before=np.memmap(priorstate/'explicit_nuisance'/filename,dtype=dtype,mode='r',shape=(actual['raw_rows'],));after=np.memmap(actual['auxiliaries'][k]['file'],dtype=dtype,mode='r',shape=(actual['raw_rows'],));changed=np.any(before.view('u1').reshape(len(before),np.dtype(dtype).itemsize)!=after.view('u1').reshape(len(after),np.dtype(dtype).itemsize),axis=1);allchanges|=changed;d=delta['by_auxiliary'][k];require(int(changed.sum())==d['changed_rows']and float(np.max(np.abs(np.asarray(before,dtype=float)-np.asarray(after,dtype=float))))==d['maximum_absolute_delta'],'Actual final four-aux delta differs '+k)
  require(int(allchanges.sum())==delta['changed_auxiliary_rows']and delta['changed_active_rows']==delta['paired_context_replayed_rows']==delta['replayed_RS_and_finite_valid']and delta['all_active_rows_valid']==actual['active_rows'],'Changed final source contexts lack complete independent replay')
  cells=delta['all_cells_independently_rechecked'];require(cells['binding']==actual['cell_state']['binding']and cells['complete_CP_cell_state_digest']==actual['cell_state']['complete_CP_cell_state_digest']and cells['actual_graph_receipt']==actual['cell_state']['actual_graph_receipt'],'Final cell state independent proof differs')
  require(delta['native_same_token_debt']==native and delta['quality_accepted']is False and delta['formal_Query_or_W_qualification_claimed']is False,'Final delta domain/quality semantics differs')
 require(audit['all']['raw_source_2D_UV']['failed']==0 and audit['all']['finite_station_or_material_failed']==0 and audit['all']['material_parameter_closed_bound_failed']==0 and audit['all']['no_legal_RS_bracket']==0 and audit['all']['RS_whole_interval_uncertified']==0,'Selected v4 full-source independent domain/RS audit fails')
 require(native['state_token']==actual['state_token']and native['g_max_positive']==0 and native['g_over_1e_minus8']==0 and native['h_max_abs']==0,'Selected exact native v4 domain receipt differs/fails')
 require(s['quality_accepted']is False and s['formal_Query_closed']is False,'Root numerical parent is not source quality/formal acceptance')
 return actual

def _source_prefix(canonical,path,old,oldpath,n):
 seen=set();N=canonical['row_count'];require(N>n and canonical['source_only_printed_board_successor']['parent_packet']==ref(oldpath),'Actual printed source ancestry differs')
 arrays={}
 for k in old['arrays']:
  a=array(old,k,oldpath.parent,seen);b=array(canonical,k,path.parent,seen);require(a.shape[0]==n and b.shape[0]==N and b[:n].tobytes()==a.tobytes(),'Canonical actual source prefix differs '+k);arrays[k]=dict(parent=old['arrays'][k],canonical=canonical['arrays'][k],exact_old_prefix_sha256=_prefix_hash(a))
 for k in('source_row_ids','source_family_ids','source_dataset_ids'):
  require(canonical[k][:n]==old[k],'Old per-ID source prefix changed '+k)
 require(canonical['queries'][:len(old['queries'])]==old['queries'],'Old original Query prefix changed')
 require(canonical['groups'][:len(old['groups'])]==old['groups'],'Old original W/group prefix changed')
 for key,allowed in old.get('allowed_paths',{}).items():require(canonical.get('allowed_paths',{}).get(key)==allowed,'Old finite path authority changed')
 require(canonical.get('source_station_domains')==old.get('source_station_domains'),'Original finite source domains changed')
 return arrays

def bind_selected_primal(canonical_source_packet,state_dir,old_source_packet,output,*,selection_manifest):
 from .board_print_chain_v1 import assert_source_identity_current
 src=Path(canonical_source_packet).resolve();oldpath=Path(old_source_packet).resolve();out=Path(output).resolve();require(not out.exists(),'Immutable v4 prefix output exists')
 canonical_guard=assert_source_identity_current(src);canonical=read_json(src);old=read_json(oldpath);actual=validate_selection(selection_manifest,oldpath,state_dir);n=old['row_count'];N=canonical['row_count'];prefix_sources=_source_prefix(canonical,src,old,oldpath,n);seen=set();oldmodel=read_json(checked(old['material_binding']['model_topology_ref'],seen));parentmodel=read_json(actual['model']['file']);require(len(parentmodel['blocks'])==len(oldmodel['blocks']),'Selected v4 physical topology size changed')
 for k in('materials','shots','rigs'):require(parentmodel[k]==oldmodel[k],'Selected v4 physical topology changed '+k)
 require([x['name']for x in parentmodel['blocks']]==[x['name']for x in oldmodel['blocks']],'Selected v4 named physical topology changed')
 active=np.asarray(array(canonical,'admission',src.parent,seen))==1;role=np.asarray(array(canonical,'fit_role',src.parent,seen));mat=np.asarray(array(canonical,'material_index',src.parent,seen));path=np.asarray(array(canonical,'path',src.parent,seen));newids=np.flatnonzero(active&(role==1)&(np.arange(N)>=n));kinds=[parentmodel['materials'][int(mat[i])]['paths'][int(path[i])]['kind']for i in newids];require(all(k==3 for k in kinds),'This v4 prefix contract only introduces smooth same-board carrier rows, not new cell paths')
 out.mkdir(parents=True);doc=copy.deepcopy(canonical);auxprefix={}
 for k,(filename,dtype)in AUX.items():
  previous=np.memmap(actual['auxiliaries'][k]['file'],dtype=dtype,mode='r',shape=(n,));suffix=array(canonical,k,src.parent,seen)[n:];f=out/(k+('.u32'if k=='path'else'.f64'));a=np.memmap(f,dtype=dtype,mode='w+',shape=(N,));a[:n]=previous;a[n:]=suffix;a.flush();del a;doc['arrays'][k]=dict(**ref(f),dtype=dtype,shape=[N]);auxprefix[k]=dict(selected_descriptor=actual['auxiliaries'][k],current_descriptor=doc['arrays'][k],selected_prefix_sha256=_prefix_hash(previous))
 write_json(out/'canonical_source_guard.json',canonical_guard);contract=dict(schema=SCHEMA,consumer=ref(__file__),canonical_source_packet=ref(src),canonical_source_guard=ref(out/'canonical_source_guard.json'),old_source_packet=ref(oldpath),selected_complete_capture_v4=actual,root_selection=ref(selection_manifest),source_prefix_array_proofs=prefix_sources,preserved_raw_rows=n,current_raw_rows=N,first_new_raw_row=n,new_active_smooth_board_rows=len(newids),new_active_path_kinds=sorted(set(kinds)),old_cell_rows=actual['cell_state']['rows'],parent_cell_state_binding=actual['cell_state']['binding'],parent_actual_graph_receipt=actual['cell_state']['actual_graph_receipt'],parent_cell_policy=actual['cell_state']['policy'],parent_cell_discrete_state_sha256=actual['discrete_cell_state_sha256'],parent_cell_discrete_policy_sha256=actual['discrete_cell_policy_sha256'],primal_auxiliary_prefixes=auxprefix,all_physical_scalar_values_exact=True,all_old_four_auxiliary_values_exact=True,all_old_source_ID_UV_Query_W_and_finite_paths_exact=True,old_cell_anchor_KKT_and_graph_bounds_preserved=True,source_append_requires_new_graph_and_policy_identity=True,old_graph_or_AL_checkpoint_resumed=False,only_new_rows_may_be_profiled=True,formal_Query_closed=False,original_W_likelihood_consumed=False,quality_accepted=False)
 write_json(out/'capture_prefix_contract.json',contract);doc['source_capture_complete_state_binding']=dict(schema=SCHEMA,contract=ref(out/'capture_prefix_contract.json'),canonical_source_packet=ref(src),model=actual['model'],state_token=actual['state_token'],objective_id=OBJECTIVE,old_raw_rows=n,new_raw_rows=N-n,parent_cell_state_binding=actual['cell_state']['binding'],formal_Query_closed=False,quality_accepted=False);doc['source_manifests'] +=[ref(out/'capture_prefix_contract.json'),ref(__file__),ref(selection_manifest)]
 from .packet_storage import write_packet
 write_packet(out/'packet.json',doc);return dict(schema=SCHEMA,packet=ref(out/'packet.json'),model=actual['model'],contract=ref(out/'capture_prefix_contract.json'),selected_token=actual['state_token'],old_raw_rows=n,new_raw_rows=N-n,new_active_rows=len(newids),physical_scalar_values_preserved=True,old_cell_state_preserved=True,native_prefix_and_new_policy_qualification_required=True,solver_started=False)

def assert_capture_prefix_current(packet_path,*,return_verified_packet=False):
 from .board_print_chain_v1 import assert_source_identity_current
 pp=Path(packet_path).resolve();p=read_json(pp);b=p['source_capture_complete_state_binding'];seen=set();require(b['schema']==SCHEMA and b['objective_id']==OBJECTIVE and not b['formal_Query_closed']and not b['quality_accepted'],'V4 source-append prefix objective/semantics')
 c=read_json(checked(b['contract'],seen));require(c['schema']==SCHEMA and c['consumer']==ref(__file__)and c['only_new_rows_may_be_profiled']and not c['old_graph_or_AL_checkpoint_resumed'],'Wrong versioned v4 source-append consumer')
 canonicalpath=checked(c['canonical_source_packet'],seen);canonical=read_json(canonicalpath);assert_source_identity_current(canonicalpath);oldpath=checked(c['old_source_packet'],seen);old=read_json(oldpath);selected=c['selected_complete_capture_v4'];state=Path(checked(selected['model'],seen)).parent;actual=validate_selection(checked(c['root_selection'],seen),oldpath,state);require(actual==selected,'Selected complete v4 physical/cell state changed');n=c['first_new_raw_row'];N=p['row_count'];require(n==actual['raw_rows']==b['old_raw_rows']==c['preserved_raw_rows']and N==c['current_raw_rows']and N-n==b['new_raw_rows']and b['model']==actual['model']and b['state_token']==actual['state_token'],'Full v4 prefix/raw/model/token mismatch')
 require(_source_prefix(canonical,canonicalpath,old,oldpath,n)==c['source_prefix_array_proofs'],'Actual original source prefix proof changed')
 for k in canonical['arrays']:
  a=array(p,k,pp.parent,seen);v=array(canonical,k,canonicalpath.parent,seen)
  if k in AUX:
   previous=np.memmap(actual['auxiliaries'][k]['file'],dtype=actual['auxiliaries'][k]['dtype'],mode='r',shape=(n,));require(a[:n].tobytes()==previous.tobytes()and a[n:].tobytes()==v[n:].tobytes(),'V4 selected old four-aux/new init changed '+k)
  else:require(a.tobytes()==v.tobytes(),'V4 wrapper changed canonical source array '+k)
 expected=copy.deepcopy(canonical);expected['arrays']=copy.deepcopy(p['arrays']);expected['source_capture_complete_state_binding']=copy.deepcopy(b);expected['source_manifests'] +=[ref(checked(b['contract'],seen)),ref(__file__),c['root_selection']];require(p==expected,'V4 prefix wrapper changed source/Query/W/dependencies mathematics')
 report=dict(schema=SCHEMA,passed=True,packet=ref(pp),selected_model=actual['model'],selected_complete_checkpoint=actual['checkpoint'],selected_token=actual['state_token'],objective_id=OBJECTIVE,preserved_raw_rows=n,new_raw_rows=N-n,new_active_smooth_board_rows=c['new_active_smooth_board_rows'],parent_cell_state_binding=actual['cell_state']['binding'],all_physical_and_old_four_auxiliary_bytes_exact=True,all_old_source_and_current_Query_W_exact=True,old_cells_and_actual_graph_bounds_identity_verified=True,only_new_rows_may_be_profiled=True,old_graph_or_AL_resumed=False,formal_Query_closed=False,original_W_likelihood_consumed=False,quality_accepted=False)
 return(report,p)if return_verified_packet else report
