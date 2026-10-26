"""Genuine complete-CP continuation through an explicit measured-operator change.

A correct old image midpoint is retained as a source fact. Its factor can be
superseded only by the two exact original interfaces, never by residual tests.
"""
from pathlib import Path
import copy,json,numpy as np
from .io_utils import file_sha256
from .accepted_aux_prefix import (AUX,OBJECTIVE,_ref,_verified,_array,_prefix_hash,_digest,_runtime_group_digest,_runtime_query_digest,verify_engineering_checkpoint,_verify_exact_repack)
from .paint_width_aux_prefix import _roles,_prefix_model
from .successor_aux_prefix import _definition
from .paint_operator_transition import SCHEMA,assert_operator_transition

SUFFIX_SCHEMA='rinkfit.qualified_physical_suffix_for_accepted_aux.v4_operator_transition_paint_width'

def operator_authority(packet,path):
    path=Path(path).resolve();cp=_verified(packet['board_material_source_qualification'],path.parent);cert=json.loads(cp.read_text('utf8'))
    if cert.get('schema')!=SCHEMA:raise ValueError('CP operator transition lacks its explicit original-source authority')
    assert_operator_transition(packet,path)
    bp=_verified(cert['operator_transition_bundle'],cp.parent);b=json.loads(bp.read_text('utf8'))
    return cp,cert,b

def _source_prefix_after_transition(name,old,indices):
    if name!='admission':return old
    expected=old.copy()
    if np.any(expected[indices]!=1):raise ValueError('Superseded midpoint was not an active original factor')
    expected[indices]=0;return expected

def _groups_after_transition(groups,indices):
    out=[]
    for g in groups:
        hits=indices.intersection(g['rows'])
        if hits:
            if g['statistic_kind']!='constraint_only_unknown_noise'or g['rank']!=0 or g['W']!=[]:raise ValueError('Superseded midpoint was not a geometry-only derived factor')
            rows=[i for i in g['rows']if i not in indices]
            if rows:out.append(dict(g,rows=rows))
        else:out.append(g)
    return out

def _issued_prefix(issued,accepted):
    value=copy.deepcopy(issued);value['blocks']=value['blocks'][:len(accepted['blocks'])]
    for key in ('measurement_widths','stroke_pair_model_bindings'):
        count=len(accepted.get(key,[]));items=value.get(key,[])[:count]
        if key in accepted:value[key]=items
        else:value.pop(key,None)
    return value

def merge_operator_width_suffix(accepted_model_path,source_packet_path,out_model):
    pp=Path(source_packet_path).resolve();packet=json.loads(pp.read_text('utf8'));cp,cert,b=operator_authority(packet,pp);accepted=json.loads(Path(accepted_model_path).read_text('utf8'));issued=json.loads(_verified(b['model'],cp.parent).read_text('utf8'));out=Path(out_model).resolve()
    if out.exists():raise FileExistsError('Immutable operator transition CP model exists')
    k=len(accepted['blocks'])
    if k>=len(issued['blocks'])or _roles(accepted)!=_roles(_issued_prefix(issued,accepted)):raise ValueError('Accepted CP has incompatible old physical topology/projection roles')
    merged=copy.deepcopy(accepted);merged['blocks']+=copy.deepcopy(issued['blocks'][k:])
    for key in ('measurement_widths','stroke_pair_model_bindings'):
        old=accepted.get(key,[])
        if issued[key][:len(old)]!=old:raise ValueError('Operator transition changed existing paired identity')
        merged[key]=copy.deepcopy(issued[key])
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return dict(model=_ref(out),accepted_model=_ref(accepted_model_path),source_packet=_ref(pp),old_physical_blocks=k,new_local_scalar_widths=len(merged['blocks'])-k,old_complete_CP_values_exact=True,operator_transition_certificate=packet['board_material_source_qualification'],new_Solve_started=False)

def qualify_operator_width_suffix(packet,packet_path,accepted_model,current_model_path,old_n):
    path=Path(packet_path).resolve();cp,cert,b=operator_authority(packet,path);currentpath=Path(current_model_path).resolve();current=json.loads(currentpath.read_text('utf8'));issued=json.loads(_verified(b['model'],cp.parent).read_text('utf8'));old_b=len(accepted_model['blocks']);old_m=len(accepted_model['materials']);oldpairs=len(accepted_model.get('stroke_pair_model_bindings',[]));pairs=packet['stroke_pairs'][oldpairs:]
    if current['materials']!=accepted_model['materials']or len(current['blocks'])!=len(issued['blocks'])or _definition(_issued_prefix(current,accepted_model))!=_definition(accepted_model):raise ValueError('Operator transition old physical prefix differs from genuine complete CP')
    if len(pairs)!=len(current['blocks'])-old_b or int(packet['row_count'])-old_n!=2*len(pairs):raise ValueError('Operator width suffix must consume exactly two new native interfaces per new scalar')
    for key in ('measurement_widths','stroke_pair_model_bindings'):
        if current[key]!=issued[key]:raise ValueError('Operator width scope or finite side branch changed')
    bindings=current['stroke_pair_model_bindings'][oldpairs:]
    for j,(pair,binding)in enumerate(zip(pairs,bindings)):
        bi=binding['half_width_block'];a=current['blocks'][bi];e=issued['blocks'][bi]
        if (bi!=old_b+j or pair['id']!=binding['pair_id']or pair['width_scope_id']!=binding['width_scope_id']or pair['rows']!=[old_n+2*j,old_n+2*j+1]or pair['material_index']!=145 or {k:v for k,v in a.items()if k!='values'}!={k:v for k,v in e.items()if k!='values'}or len(a['values'])!=1 or not 0<a['values'][0]<=1 or a['constant']is not False or a['fixed_indices']!=[]or a['lower']!=[1e-12]or a['upper']!=[1.]):raise ValueError('Operator transition scalar width or native flank binding differs')
    return dict(schema=SUFFIX_SCHEMA,current_model=_ref(currentpath),source_qualification_certificate=packet['board_material_source_qualification'],source_issued_model=b['model'],operator_transition_bundle=cert['operator_transition_bundle'],old_physical_blocks=old_b,new_physical_blocks=len(current['blocks']),old_materials=old_m,new_materials=old_m,qualified_suffix_blocks=len(pairs),qualified_suffix_materials=0,qualified_new_source_rows=int(packet['row_count'])-old_n,old_measurement_width_count=oldpairs,old_CP_model_is_not_replaced=True,old_physical_prefix_values_exact=True,old_projection_roles_and_topology_exact=True,source_qualification_required_before_consume=True,suffix_is_not_part_of_old_checkpoint=True,local_scalar_width_is_not_metric_truth=True)

def rebind_operator_transition_nuisance(packet_path, candidate, output_path, *, current_model_path=None, physical_suffix_qualifier=qualify_operator_width_suffix):
    """Bind candidate/explicit_nuisance, never independent-RS reprofile output."""
    src = Path(packet_path).resolve(); state = Path(candidate).resolve(); out = Path(output_path).resolve()
    if out.exists(): raise FileExistsError('immutable engineering auxiliary packet exists')
    if state.name != 'candidate': raise ValueError('engineering binding requires the candidate explicit state, not independent reprofile')
    receipt_path = state.parent/'engineering_receipt.json'; receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if receipt.get('schema') != 'rinkfit.joint_engineering_recovery_run.v1' or receipt.get('objective_id') != OBJECTIVE:
        raise ValueError('not a native engineering pixel candidate')
    if receipt.get('source_recalled', True) or receipt.get('formal_Phase_II_KKT_passed', True) or receipt.get('common_worst_budget_consumed', True) or receipt.get('original_W_likelihood_consumed', True) or receipt.get('callback_error', ''):
        raise ValueError('recalled, failed, or incompatible engineering candidate')
    model_path = state/'model.json'
    if file_sha256(model_path) != receipt['model_sha256']: raise ValueError('engineering candidate physical model changed')
    model = json.loads(model_path.read_text(encoding='utf8'))
    _verify_exact_repack(state,receipt,model)
    manifest_path = state/'explicit_nuisance/manifest.json'; manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    parent = Path(receipt['source_packet']).resolve()
    if file_sha256(parent) != receipt['source_packet_sha256'] or manifest['packet_sha256'] != receipt['source_packet_sha256']:
        raise ValueError('engineering explicit state source mismatch')
    original = json.loads(parent.read_text(encoding='utf8')); packet = json.loads(src.read_text(encoding='utf8'))
    operator_certificate,operator_cert,operator_bundle=operator_authority(packet,src)
    superseded={int(i)for i in operator_cert['superseded_derived_midpoint_indices']}
    transitioned_indices=sorted(i for i in superseded if i<int(manifest['raw_rows']))
    from .joint_solver import _source_membership_fingerprint
    for source,document in ((parent,original),(src,packet)):
        directories={source.parent}
        certificate=document.get('board_material_source_qualification')
        if certificate: directories.add(_verified(certificate,source.parent).parent)
        fingerprint=_source_membership_fingerprint(document)
        for ref in document.get('source_identity_watchlist',[]):
            if ref.get('source_membership_fingerprint')==fingerprint: directories.add(_verified(ref,source.parent).parent)
        for directory in directories:
            for marker in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
                if (directory/marker).exists(): raise ValueError('current engineering source qualification recalled '+str(directory/marker))
    old_n = int(manifest['raw_rows']); n = int(packet['row_count'])
    if int(original['row_count']) != old_n or n < old_n or original['shot_ids'] != packet['shot_ids']:
        raise ValueError('engineering appended scope or prefix count mismatch')
    for key in ('source_row_ids','source_dataset_ids'):
        if original[key] != packet[key][:old_n]: raise ValueError('changed engineering source ID prefix '+key)
    physical_suffix=None
    if current_model_path is not None:
        physical_suffix=(physical_suffix_qualifier or qualify_operator_width_suffix)(packet,src,model,current_model_path,old_n)
    old_queries = [q for q in original.get('queries',[]) if int(q['row_index']) < old_n]
    new_queries = [q for q in packet.get('queries',[]) if int(q['row_index']) < old_n]
    if old_queries != new_queries: raise ValueError('changed engineering genuine Query prefix')
    for key in ('stroke_pairs','independent_rows'):
        previous = original.get(key,[])
        if packet.get(key,[])[:len(previous)] != previous: raise ValueError('changed engineering source topology prefix '+key)
    old_domains = original.get('source_station_domains',{}).get('rows',[])
    new_domains = [q for q in packet.get('source_station_domains',{}).get('rows',[]) if int(q['row_index']) < old_n]
    if old_domains != new_domains: raise ValueError('changed engineering finite source station domains')
    contract_arrays = {}; arrays = {}
    for name, old_d in original['arrays'].items():
        if name in AUX: continue
        old_a = _array(original, name, parent.parent); new_a = _array(packet, name, src.parent)
        if old_a.shape[0] != old_n or new_a.shape[0] != n or old_a.dtype != new_a.dtype or not np.array_equal(_source_prefix_after_transition(name,old_a,transitioned_indices),new_a[:old_n]):
            raise ValueError('changed engineering source array prefix '+name)
        contract_arrays[name] = dict(current_descriptor=packet['arrays'][name], prefix_sha256=_prefix_hash(new_a[:old_n]),original_prefix_sha256=_prefix_hash(old_a),original_descriptor=old_d)
        arrays[name] = new_a
    verified_group_dependencies=set()
    def normalized_group(group, base):
        g = dict(group)
        if isinstance(g['W'],dict):
            d = dict(g['W']); d['file'] = str(_verified(d,base,verified_group_dependencies)); g['W'] = d
        return g
    old_original_groups = [normalized_group(g,parent.parent) for g in original['groups']]
    old_groups=_groups_after_transition(old_original_groups,set(transitioned_indices))
    groups = [normalized_group(g,src.parent) for g in packet['groups']]
    if groups[:len(old_groups)] != old_groups or any(any(int(i)<old_n for i in g['rows']) for g in groups[len(old_groups):]):
        raise ValueError('changed engineering W group prefix or cross-prefix appended group')
    active = (_array(original,'admission',parent.parent)==1)&(_array(original,'fit_role',parent.parent)==1)
    retained_active=(arrays['admission'][:old_n]==1)&(arrays['fit_role'][:old_n]==1)
    aux = {}
    for name,(filename,dtype) in AUX.items():
        d = manifest['files'][filename]; p = _verified(dict(file=str(state/'explicit_nuisance'/filename),sha256=d['sha256']),state)
        a = np.memmap(p,mode='r',dtype=dtype)
        if a.shape != (old_n,) or not np.isfinite(a).all(): raise ValueError('incomplete engineering explicit '+name)
        aux[name] = a
    checkpoint = state/'engineering_checkpoint.json'
    required_station = set(np.flatnonzero(active & (arrays['kind'][:old_n] != 1)).tolist())
    checked = verify_engineering_checkpoint(checkpoint,receipt,model,aux,active,[q['row_index'] for q in old_queries],required_station)
    out.parent.mkdir(parents=True,exist_ok=True)
    checked_files = set()
    def absolute(d):
        p = Path(d['file']); p = p if p.is_absolute() else src.parent/p; p = p.resolve()
        key = (str(p),d['sha256'])
        if key not in checked_files: _verified(d,src.parent); checked_files.add(key)
        return {**d,'file':str(p)}
    packet['arrays'] = {k:absolute(d) for k,d in packet['arrays'].items()}
    for g in packet['groups']:
        if isinstance(g['W'],dict): g['W'] = absolute(g['W'])
    for d in packet.get('source_manifests',[]): d.update(absolute(d))
    for name in contract_arrays: contract_arrays[name]['current_descriptor'] = packet['arrays'][name]
    aux_contract = {}
    for name,(filename,dtype) in AUX.items():
        fresh = _array(packet,name,src.parent)
        target = out.parent/('accepted_current_'+filename)
        if target.exists(): raise FileExistsError('immutable engineering merged auxiliary exists')
        with target.open('wb') as stream:
            np.asarray(aux[name],dtype=dtype).tofile(stream)
            np.asarray(fresh[old_n:],dtype=dtype).tofile(stream)
        descriptor = dict(file=str(target),sha256=file_sha256(target),dtype=dtype,shape=[n])
        packet['arrays'][name] = descriptor
        aux_contract[name] = dict(candidate_file=_ref(state/'explicit_nuisance'/filename),current_descriptor=descriptor,prefix_sha256=_prefix_hash(aux[name]))
    transitioned_original=dict(original,groups=old_groups)
    runtime_groups_digest,runtime_group_count=_runtime_group_digest(transitioned_original,parent.parent,verified_group_dependencies)
    contract = dict(schema='rinkfit.engineering_operator_transition_aux_contract.v1',old_raw_rows=old_n,new_raw_rows=n,
        old_source_packet=_ref(parent),appended_source_packet=_ref(src),source_arrays=contract_arrays,auxiliaries=aux_contract,
        source_ID_prefix_sha256=_digest([original['source_dataset_ids'],original['source_row_ids']]),
        query_prefix_sha256=_digest(old_queries),group_prefix_sha256=_digest(old_groups),old_group_count=len(old_groups),
        runtime_query_prefix_sha256=_runtime_query_digest(old_queries),runtime_group_prefix_sha256=runtime_groups_digest,
        runtime_old_group_count=runtime_group_count,shot_ids=original['shot_ids'],
        complete_Query_material_kind_normal_E_ID_PTS_UV_and_W_prefix_verified=True,
        full_candidate_explicit_arrays_preserved=True,old_active_rows=int(active.sum()),retained_active_rows=int(retained_active.sum()),expected_station_rows=len(required_station),
        operator_transition_certificate=packet['board_material_source_qualification'],superseded_original_factor_indices=transitioned_indices,
        source_UV_E_W_ID_query_unchanged=True,only_derived_midpoint_factor_membership_explicitly_changed=True,original_group_prefix_sha256=_digest(old_original_groups))
    if physical_suffix is not None:contract['qualified_physical_suffix']=physical_suffix
    contract_path = out.parent/'accepted_aux_prefix_contract.json'
    if contract_path.exists(): raise FileExistsError('immutable engineering prefix contract exists')
    contract_path.write_text(json.dumps(contract,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    packet['accepted_explicit_state_binding'] = dict(schema='rinkfit.accepted_operator_transition_state_packet_binding.v1',
        state_kind='engineering_candidate',parent_packet=_ref(src),accepted_source_packet=_ref(parent),
        accepted_receipt=_ref(receipt_path),accepted_model=_ref(model_path),explicit_nuisance_manifest=_ref(manifest_path),
        complete_engineering_checkpoint=_ref(checkpoint),prefix_contract=_ref(contract_path),
        model_sha256=receipt['model_sha256'],state_token=checked['state_token'],objective_id=OBJECTIVE,
        configuration_sha256=receipt['configuration_sha256'],preserved_raw_rows=old_n,
        native_replay_pass_claimed=False,formal_acceptance_claimed=False,old_AL_duals_resumed=False,
        source_UV_W_E_unchanged=True,source_factor_admission_operator_transition_explicit=True,new_appended_rows_must_be_initialized=True,operator_transition_certificate=packet['board_material_source_qualification'])
    if physical_suffix is not None:packet['accepted_explicit_state_binding']['qualified_physical_suffix']=physical_suffix
    out.write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    from .joint_solver import _inherit_source_recall_marker
    _inherit_source_recall_marker(src,out)
    current_model=physical_suffix['current_model']if physical_suffix is not None else _ref(model_path)
    return dict(packet=str(out),sha256=file_sha256(out),model=current_model['file'],model_sha256=current_model['sha256'],
                accepted_CP_model=_ref(model_path),qualified_physical_suffix=physical_suffix is not None,
                preserved_raw_rows=old_n,fresh_appended_rows=n-old_n,state_token=checked['state_token'],objective_id=OBJECTIVE)
