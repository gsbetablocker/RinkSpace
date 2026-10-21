"""Root-selected domain-qualified trueRS capture primal prefix, source immutable.
This is a new graph primal import, never formal Query/W or AL checkpoint resume.
"""
from pathlib import Path
import copy
import numpy as np
from .io_utils import read_json,write_json,file_sha256
from .r_inner_prefix_v1 import array
from .accepted_aux_prefix import AUX,_prefix_hash
from .board_middle_append_v1 import checked,require,ref
from .capture_primal_prefix_v1 import inspect_complete_capture
SCHEMA='rinkfit.selected_true_RS_capture_primal_prefix_binding.v2'

def validate_selection(selection_manifest,source_packet,state_dir):
 sel=read_json(selection_manifest);seen=set();state=Path(state_dir).resolve()
 require(sel['schema']=='rinkfit.root_selected_complete_capture_initial_primal.v1' and sel['selected_for_next_source_batch'] and sel['selection_kind']=='independently_domain_qualified_initial_capture' and sel['state_directory']==str(state),'Root selected initial capture identity differs')
 actual=inspect_complete_capture(source_packet,state,capture_receipt=checked(sel['native_receipt'],seen));ind=read_json(checked(sel['independent_audit'],seen));debt=read_json(checked(sel['native_constraint_debt'],seen));checked(sel['configuration'],seen)
 require(actual['state_token']==sel['selected_state_token']==debt['state_token']==ind['complete_CP']['state_token'] and sel['selected_model']==actual['model'] and sel['selected_checkpoint']==actual['checkpoint'] and sel['selected_state_binding']==actual['capture_state_binding'],'Selected complete capture token/model/CP mismatch')
 require(ind['packet']==actual['source_packet'] and ind['model']==actual['model'] and ind['complete_CP']['all_parameter_values_verified'] and ind['complete_CP']['FNV_token_independently_recomputed'],'Independent initial audit is not this exact complete state')
 require(debt['g_max_positive']==0 and debt['g_over_1e_minus8']==0 and debt['h_max_abs']==0 and ind['all']['raw_source_2D_UV']['failed']==0 and ind['all']['finite_station_or_material_failed']==0 and ind['all']['material_parameter_closed_bound_failed']==0 and ind['all']['no_legal_RS_bracket']==0 and ind['all']['RS_whole_interval_uncertified']==0,'Selected initial state lacks complete domain/RS qualification')
 require(not sel['quality_accepted'] and not sel['accepted_physical_step_claimed'] and not sel['formal_Query_closed'],'Root initial seed relabelled quality/accepted/formal')
 return actual

def bind_selected_initial_primal(canonical_source_packet,state_dir,old_source_packet,output,*,selection_manifest):
 from .joint_source_current_v3 import assert_source_identity_current
 src=Path(canonical_source_packet).resolve();state=Path(state_dir).resolve();out=Path(output).resolve();require(not out.exists(),'Immutable primal output already exists')
 assert_source_identity_current(src,allow_unbound_source_prototype=True);canonical=read_json(src);oldpath=Path(old_source_packet).resolve();old=read_json(oldpath);current=validate_selection(selection_manifest,oldpath,state);seen=set();n=old['row_count'];N=canonical['row_count']
 require(canonical['source_only_board_successor']['parent_packet']==ref(oldpath) and n==current['raw_rows'] and N>n,'Selected old capture is not this canonical source ancestry')
 # Source means/admission have their own immutable generations; they do not
 # numerically redefine an old trueRS row, material q, station, alpha or path.
 base=read_json(checked(old['material_binding']['model_topology_ref'],seen));model=read_json(current['model']['file'])
 require(len(base['blocks'])==len(model['blocks']),'Physical topology size changed')
 for a,b in zip(base['blocks'],model['blocks']):
  aa=copy.deepcopy(a);bb=copy.deepcopy(b);aa.pop('values');bb.pop('values');require(aa==bb,'Selected initial physical domain/topology changed '+a['name'])
 for key in ('materials','shots','rigs'):require(model[key]==base[key],'Selected initial physical topology changed '+key)
 out.mkdir(parents=True);doc=copy.deepcopy(canonical);prefix={}
 for k,(name,dtype)in AUX.items():
  saved=np.memmap(current['auxiliaries'][k]['file'],dtype=dtype,mode='r',shape=(n,));suffix=array(canonical,k,src.parent,seen)[n:];f=out/(k+('.u32'if k=='path'else'.f64'));values=np.memmap(f,dtype=dtype,mode='w+',shape=(N,));values[:n]=saved;values[n:]=suffix;values.flush();del values;doc['arrays'][k]=dict(**ref(f),dtype=dtype,shape=[N]);prefix[k]=dict(selected_descriptor=current['auxiliaries'][k],current_descriptor=doc['arrays'][k],selected_prefix_sha256=_prefix_hash(saved))
 contract=dict(schema=SCHEMA,consumer=ref(__file__),canonical_source_packet=ref(src),old_source_packet=ref(oldpath),selected_complete_capture=current,root_selection=ref(selection_manifest),preserved_raw_rows=n,current_raw_rows=N,primal_auxiliary_prefixes=prefix,first_new_raw_row=n,all_physical_values_exact=True,all_old_four_auxiliary_values_exact=True,canonical_source_UV_and_admission_changes_preserved=True,new_source_station_and_RS_initialization_required=True,only_new_rows_may_be_profiled=True,old_checkpoint_graph_resumed=False,old_AL_duals_resumed=False,formal_Query_closed=False,original_W_likelihood_consumed=False,cross_frame_E_numerical_discount_consumed=False,quality_accepted=False)
 write_json(out/'capture_prefix_contract.json',contract);doc['source_capture_complete_state_binding']=dict(schema=SCHEMA,contract=ref(out/'capture_prefix_contract.json'),canonical_source_packet=ref(src),model=current['model'],state_token=current['state_token'],objective_id=current['objective_id'],old_raw_rows=n,new_raw_rows=N-n,formal_Query_closed=False,quality_accepted=False);doc['source_manifests']+=[ref(out/'capture_prefix_contract.json'),ref(__file__),ref(selection_manifest)]
 from .packet_storage import write_packet
 write_packet(out/'packet.json',doc);return dict(packet=ref(out/'packet.json'),model=current['model'],contract=ref(out/'capture_prefix_contract.json'),old_CP_modified=False,physical_values_preserved=True,preserved_four_auxiliary_rows=n,new_rows=N-n,native_import_qualification_required=True,solver_started=False)

def assert_capture_prefix_current(packet_path):
 from .joint_source_current_v3 import assert_source_identity_current
 path=Path(packet_path).resolve();doc=read_json(path);binding=doc['source_capture_complete_state_binding'];seen=set();require(binding['schema']==SCHEMA and not binding['formal_Query_closed'] and not binding['quality_accepted'],'Invalid selected capture primal semantics')
 contract=read_json(checked(binding['contract'],seen));require(contract['schema']==SCHEMA and contract['consumer']==ref(__file__) and not contract['old_checkpoint_graph_resumed'] and not contract['old_AL_duals_resumed'] and not contract['original_W_likelihood_consumed'] and contract['only_new_rows_may_be_profiled'],'Capture primal contract/consumer changed')
 canonicalpath=checked(contract['canonical_source_packet'],seen);assert_source_identity_current(canonicalpath,allow_unbound_source_prototype=True);canonical=read_json(canonicalpath);parent=contract['selected_complete_capture'];state=Path(checked(parent['model'],seen)).parent;oldpath=checked(contract['old_source_packet'],seen);actual=validate_selection(checked(contract['root_selection'],seen),oldpath,state);require(actual==parent,'Selected complete initial state changed');n=contract['preserved_raw_rows'];N=contract['current_raw_rows'];require(n==actual['raw_rows']==binding['old_raw_rows']==contract['first_new_raw_row'] and N==doc['row_count'] and N-n==binding['new_raw_rows'] and binding['model']==actual['model'] and binding['state_token']==actual['state_token'] and binding['objective_id']==actual['objective_id'],'Capture prefix state/denominator differs')
 for k in canonical['arrays']:
  a=array(doc,k,path.parent,seen);b=array(canonical,k,canonicalpath.parent,seen)
  if k in AUX:
   old=np.memmap(actual['auxiliaries'][k]['file'],dtype=actual['auxiliaries'][k]['dtype'],mode='r',shape=(n,));require(a[:n].tobytes()==old.tobytes() and a[n:].tobytes()==b[n:].tobytes(),'Selected initial four-aux prefix/new seed changed '+k)
  else:require(a.tobytes()==b.tobytes(),'Capture wrapper changed canonical source array '+k)
 expected=copy.deepcopy(canonical);expected['arrays']=copy.deepcopy(doc['arrays']);expected['source_capture_complete_state_binding']=copy.deepcopy(binding);expected['source_manifests'] += [ref(checked(binding['contract'],seen)),ref(__file__),contract['root_selection']];require(doc==expected,'Capture wrapper altered canonical source metadata/Query/W/E mathematics')
 return dict(schema=SCHEMA,passed=True,packet=ref(path),selected_model=actual['model'],selected_complete_checkpoint=actual['checkpoint'],selected_token=actual['state_token'],objective_id=actual['objective_id'],preserved_raw_rows=n,new_raw_rows=N-n,all_four_actual_auxiliary_prefixes_exact=True,all_other_canonical_source_arrays_and_W_Query_exact=True,physical_values_exact=True,only_new_rows_may_be_profiled=True,formal_Query_closed=False,original_W_likelihood_consumed=False,cross_frame_E_numerical_discount_consumed=False,quality_accepted=False,native_import_qualification_required=True)
