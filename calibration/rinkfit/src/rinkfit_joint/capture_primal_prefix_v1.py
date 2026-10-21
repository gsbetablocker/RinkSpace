"""Versioned complete trueRS-capture primal prefix import.

Capture derived rows/stations and unchanged alpha are preserved as capture
state. Neither formal Query closure nor original W likelihood is asserted.
The canonical source authority remains independent of this numeric wrapper.
"""
from pathlib import Path
import copy,json
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .r_inner_prefix_v1 import array
from .accepted_aux_prefix import AUX,_prefix_hash
from .first4_true_rs_capture_audit_v1 import verify_checkpoint,OBJECTIVE_POLICIES
from .board_middle_append_v1 import checked,require,ref
SCHEMA='rinkfit.true_RS_capture_complete_primal_prefix_binding.v1'

def inspect_complete_capture(source_packet,state_dir,*,capture_receipt=None):
 pp=Path(source_packet).resolve();state=Path(state_dir).resolve();old=read_json(pp);seen=set();arrays={k:array(old,k,pp.parent,seen)for k in old['arrays']};active=(arrays['admission']==1)&(arrays['fit_role']==1);modelpath=state/'model.json';model=read_json(modelpath);binding=read_json(state/'source_uv_capture_state_binding.json');manifest=read_json(state/'explicit_nuisance/manifest.json');n=old['row_count'];digest=file_sha256(pp)
 require(binding['schema']=='rinkfit.sourceUV_trueRS_capture_state_binding.v1'and binding['objective_id']in OBJECTIVE_POLICIES and binding['packet_sha256']==digest and binding['model_sha256']==file_sha256(modelpath),'Complete capture source/model identity differs')
 require(binding['row_alpha_optimization_dof']==0 and binding['stored_valid_rows_are_derived_true_RS_roots']and not binding['formal_Query_closed'],'Capture semantics silently changed to formal explicit')
 require(manifest['raw_rows']==n and manifest['packet_sha256']==digest,'Complete capture raw auxiliary denominator differs');aux={};descriptors={}
 for k,(name,dtype)in AUX.items():
  f=state/'explicit_nuisance'/name;require(f.stat().st_size==n*np.dtype(dtype).itemsize and file_sha256(f)==manifest['files'][name]['sha256'],'Full capture array missing/changed '+k);aux[k]=np.memmap(f,dtype=dtype,mode='r',shape=(n,));require(np.isfinite(aux[k]).all(),'Nonfinite full capture auxiliary '+k);descriptors[k]=dict(**ref(f),dtype=dtype,shape=[n])
 qrows={q['row_index']for q in old['queries']};proof=verify_checkpoint(state/'engineering_checkpoint.json',digest,model,aux,active,qrows,binding);require(proof.pop('station_rows')==set(map(int,np.flatnonzero(active&(arrays['kind']!=1)))),'Complete capture finite-station set differs')
 receiptref=None
 if capture_receipt is not None:
  receipt=read_json(capture_receipt);require(receipt['source_packet_sha256']==digest and receipt['model_sha256']==binding['model_sha256']and receipt['checkpoint_sha256']==file_sha256(state/'engineering_checkpoint.json')and receipt['complete_state_token']==binding['state_token']and receipt['objective_id']==binding['objective_id']and receipt['configuration_sha256']==proof['runtime_fingerprint'],'Actual capture receipt/complete CP mismatch');receiptref=ref(capture_receipt)
 status=state/'derived_auxiliary_binding/RS_status.u8';v=np.fromfile(status,dtype='u1');require(v.shape==(n,)and np.all(v[active]==1),'Actual accepted capture has invalid old RS source rows')
 return dict(schema=SCHEMA,source_packet=ref(pp),model=ref(modelpath),checkpoint=ref(state/'engineering_checkpoint.json'),capture_state_binding=ref(state/'source_uv_capture_state_binding.json'),explicit_nuisance_manifest=ref(state/'explicit_nuisance/manifest.json'),actual_RS_status=ref(status),capture_receipt=receiptref,auxiliaries=descriptors,raw_rows=n,active_rows=int(active.sum()),complete_checkpoint_verification=proof,objective_id=binding['objective_id'],state_token=binding['state_token'],formal_Query_closed=False,original_W_likelihood_consumed=False,accepted_physical_step_claimed=False,quality_accepted=False)

def _at(doc,path):
 for k in path:doc=doc[k]
 return doc

def bind_capture_complete_primal(canonical_source_packet,state_dir,accepted_source_packet,output,*,selection_manifest,capture_configuration):
 """Bind only the root-selected immutable accepted physical snapshot.

 Selection supplies actual request/iteration/relinearization refs and the exact
 JSON path of the materialized token; it never substitutes a numeric status bit
 for the complete CP or for source/pixel quality approval.
 """
 from .board_middle_current_v2 import assert_current
 src=Path(canonical_source_packet).resolve();state=Path(state_dir).resolve();out=Path(output).resolve();require(not out.exists(),'Immutable capture prefix output exists');canonical=read_json(src);source_report=assert_current(src);sel=read_json(selection_manifest);seen=set();require(sel['selected_for_next_source_batch']and sel['state_directory']==str(state),'Physical parent was not selected by the root review boundary')
 evidence=sel['accepted_step_evidence'];request=read_json(checked(evidence['request'],seen));iterationfile=checked(evidence['iterations'],seen);iteration=[v for v in read_jsonl(iterationfile)if int(v['global_iteration'])==int(request['global_iteration'])];require(request['accepted_iteration']>0 and request['global_iteration']>0 and request['physical_model_changed']and not request['parameters_mutated_in_callback']and len(iteration)==1 and iteration[0]['accepted']and int(iteration[0].get('iteration',request['global_iteration']))>0,'Missing genuine accepted physical-step evidence')
 inspected=inspect_complete_capture(accepted_source_packet,state);relin=read_json(checked(evidence['relinearization'],seen));require(_at(relin,evidence['materialized_token_json_path'])==inspected['state_token'],'Accepted step did not materialize this complete capture state token')
 require(sel['selected_model']==inspected['model']and sel['selected_checkpoint']==inspected['checkpoint']and sel['selected_state_binding']==inspected['capture_state_binding'],'Root selected parent snapshot differs from accepted CP')
 old=read_json(accepted_source_packet);n=old['row_count'];N=canonical['row_count'];require(n==source_report['old_raw_rows']and canonical['source_only_board_successor']['parent_packet']==ref(accepted_source_packet),'Accepted capture is not the actual full old source parent')
 model=read_json(inspected['model']['file']);oldmodelpath=checked(old['material_binding']['model_topology_ref'],seen);canonical_model=read_json(oldmodelpath)
 for a,b in zip(canonical_model['blocks'],model['blocks']):
  aa=copy.deepcopy(a);bb=copy.deepcopy(b);aa.pop('values');bb.pop('values');require(aa==bb,'Accepted capture changed physical block topology/domain '+a['name'])
 require(len(model['blocks'])==len(canonical_model['blocks'])and model['materials']==canonical_model['materials']and model['shots']==canonical_model['shots']and model['rigs']==canonical_model['rigs'],'Accepted capture changed shared physical projection topology')
 out.mkdir(parents=True);doc=copy.deepcopy(canonical);prefix={}
 for k,(name,dtype)in AUX.items():
  accepted=np.memmap(inspected['auxiliaries'][k]['file'],dtype=dtype,mode='r',shape=(n,));suffix=array(canonical,k,src.parent,seen)[n:];f=out/(k+('.u32'if k=='path'else'.f64'));v=np.memmap(f,dtype=dtype,mode='w+',shape=(N,));v[:n]=accepted;v[n:]=suffix;v.flush();del v;doc['arrays'][k]=dict(**ref(f),dtype=dtype,shape=[N]);prefix[k]=dict(accepted_descriptor=inspected['auxiliaries'][k],current_descriptor=doc['arrays'][k],accepted_prefix_sha256=_prefix_hash(accepted))
 for k in old['arrays']:
  if k not in AUX:require(array(doc,k,out,seen)[:n].tobytes()==array(old,k,Path(accepted_source_packet).parent,seen).tobytes(),'Old source mathematics changed while rebinding capture '+k)
 contract=dict(schema=SCHEMA,consumer=ref(__file__),canonical_source_packet=ref(src),accepted_parent_source_packet=ref(accepted_source_packet),actual_accepted_capture=inspected,root_selection=ref(selection_manifest),configuration=ref(capture_configuration),preserved_raw_rows=n,current_raw_rows=N,old_complete_derived_capture_auxiliaries=prefix,fresh_source_start=n,new_source_station_and_RS_initialization_required=True,old_station_row_alpha_path_prefix_exact=True,source_UV_W_E_admission_roles_unchanged=True,old_AL_duals_resumed=False,formal_Query_closed=False,formal_Query_W_mode_transition_performed=False,original_W_likelihood_consumed=False,quality_acceptance_claimed=False,next_native_consumer_must_only_profile_new_rows=True)
 write_json(out/'capture_prefix_contract.json',contract);doc['source_capture_complete_state_binding']=dict(schema=SCHEMA,contract=ref(out/'capture_prefix_contract.json'),canonical_source_packet=ref(src),model=inspected['model'],state_token=inspected['state_token'],objective_id=inspected['objective_id'],old_raw_rows=n,new_raw_rows=N-n,formal_Query_closed=False,quality_accepted=False);doc['source_only_board_successor']['accepted_physical_parent_bound']=True;doc['source_only_board_successor']['prototype_only']=False;doc['source_only_board_successor']['capture_complete_prefix_contract']=ref(out/'capture_prefix_contract.json');doc['source_manifests']+=[ref(out/'capture_prefix_contract.json'),ref(__file__),ref(selection_manifest),ref(capture_configuration)]
 from .packet_storage import write_packet
 write_packet(out/'packet.json',doc);return dict(packet=ref(out/'packet.json'),model=inspected['model'],contract=ref(out/'capture_prefix_contract.json'),old_CP_modified=False,new_native_consumer_required=True,solver_started=False)


def assert_capture_prefix_current(packet_path):
 """Actual Python consumer for a capture-only primal successor wrapper."""
 from .board_middle_current_v2 import assert_current
 path=Path(packet_path).resolve();doc=read_json(path);binding=doc['source_capture_complete_state_binding'];seen=set();require(binding['schema']==SCHEMA and binding['objective_id']in OBJECTIVE_POLICIES and not binding['formal_Query_closed']and not binding['quality_accepted'],'Unknown capture-primal state semantics');contract=read_json(checked(binding['contract'],seen));require(contract['schema']==SCHEMA and contract['consumer']==ref(__file__)and not contract['old_AL_duals_resumed']and not contract['formal_Query_W_mode_transition_performed']and not contract['original_W_likelihood_consumed'],'Capture prefix consumer/claims differ')
 canonical_path=checked(contract['canonical_source_packet'],seen);sourceproof=assert_current(canonical_path);canonical=read_json(canonical_path);accepted=contract['actual_accepted_capture'];checked(contract['root_selection'],seen);checked(contract['configuration'],seen);state=Path(checked(accepted['model'],seen)).parent;oldpath=checked(contract['accepted_parent_source_packet'],seen);current=inspect_complete_capture(oldpath,state);require(current==accepted,'Complete capture prefix parent changed');n=contract['preserved_raw_rows'];N=contract['current_raw_rows'];require(n==current['raw_rows']==binding['old_raw_rows']and N==doc['row_count']and N-n==binding['new_raw_rows']and binding['model']==current['model']and binding['state_token']==current['state_token'],'Capture prefix state/denominator mismatch')
 for k in canonical['arrays']:
  a=array(doc,k,path.parent,seen);b=array(canonical,k,canonical_path.parent,seen)
  if k in AUX:
   old=np.memmap(current['auxiliaries'][k]['file'],dtype=current['auxiliaries'][k]['dtype'],mode='r',shape=(n,));require(a[:n].tobytes()==old.tobytes()and a[n:].tobytes()==b[n:].tobytes(),'Complete accepted capture auxiliary prefix or new seed changed '+k)
  else:require(a.tobytes()==b.tobytes(),'Numeric wrapper changed actual canonical source array '+k)
 for k in ('source_row_ids','source_dataset_ids','source_family_ids','queries','groups','stroke_pairs','source_station_domains','engineering_pixel_dependency_groups','material_binding'):require(doc[k]==canonical[k],'Capture wrapper changed canonical source mathematics '+k)
 return dict(schema=SCHEMA,passed=True,packet=ref(path),accepted_model=current['model'],accepted_complete_checkpoint=current['checkpoint'],accepted_token=current['state_token'],objective_id=current['objective_id'],preserved_raw_rows=n,new_source_rows=N-n,all_four_actual_accepted_auxiliary_prefixes_exact=True,all_other_canonical_source_arrays_and_W_Query_exact=True,only_new_rows_must_be_initialized=True,formal_Query_closed=False,original_W_likelihood_consumed=False,quality_acceptance_claimed=False,new_native_consumer_still_required=True)
