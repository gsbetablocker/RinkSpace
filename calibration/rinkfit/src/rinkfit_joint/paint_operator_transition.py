"""Explicit midpoint-factor supersession by its original two RGB interfaces.

This is an operator transition, not an append-only source certificate. Old raw
UV, identities and journals survive; only the derived factor membership changes.
"""
from pathlib import Path
import copy,json,hashlib
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .board_source_qualification_guard import _load_ref
from .static_board_prefix_qualification import _array,_json,_source_refs,AUX_ARRAYS
from . import curve_source_prefix_qualification as parent_guard
from . import paint_midpoint_supersession as acquisition
from . import tail_curve_source_append as source

SCHEMA='rinkfit.actual_source_operator_qualification.v9_original_midpoint_to_pair'
CONTRACT='original_native_C_midpoint_factor_to_its_two_interfaces_v1'
BUNDLE_SCHEMA='rinkfit.original_native_C_operator_transition_bundle.v1'
SELF=source.ref(__file__)
PARENT=source.ref(parent_guard.__file__)
ACQUISITION=source.ref(acquisition.__file__)

def _checked_source(path,seen):
    b=read_json(path)
    if b.get('schema')!=acquisition.BUNDLE_SCHEMA or b.get('reader_module_refs')!=[ACQUISITION] or b.get('qualified_midpoint_supersessions')!=13 or b.get('new_actual_flank_rows')!=26 or b.get('pending_old_rows')!=[] or b.get('old_raw_UV_and_ID_preserved')is not True or b.get('reason_is_operator_definition_not_residual')is not True:raise ValueError('No exact original 13-source operator authority')
    _source_refs(b,path.parent,seen)
    rows=list(read_jsonl(_load_ref(b['source_rows_ref'],path.parent,seen)));proof=list(read_jsonl(_load_ref(b['actual_replay'],path.parent,seen)));journals={}
    if len(rows)!=26 or len(proof)!=13:raise ValueError('Original operator denominator changed')
    for j,x in enumerate(proof):
        d=x['existing_source_derivation'];old=d['original_source_row'];r=old['source_operator_result']['source_material_receipt'];fl=np.asarray(d['source_flanks_uv'],float)
        if x.get('qualified_for_operator_supersession')is not True or x.get('old_UV_correct_and_retained')is not True or x.get('actor_exclusion_clear')is not True or any(x[k]>1e-10 for k in ('actual_flank_replay_max_abs_error_px','actual_midpoint_replay_max_abs_error_px','actual_five_profile_replay_error')) or x['raw_index']!=d['raw_index'] or x['source_row_id']!=d['source_row_id'] or np.max(abs(np.mean(fl,axis=0)-d['correct_original_packet_UV']))>1e-10:raise ValueError('Original correct midpoint is not exactly derived from the actual two interfaces')
        _load_ref(x['frozen_replay_module'],path.parent,seen)
        jp=_load_ref(d['journal'],path.parent,seen)
        if jp not in journals:journals[jp]=list(read_jsonl(jp))
        actual=[v for v in journals[jp]if v.get('source_row_id')==d['source_row_id']and v.get('native_pts')==d['native_pts']and v.get('current_uv')==d['correct_original_packet_UV']]
        if len(actual)!=1 or actual[0]!=old:raise ValueError('Original saved material receipt does not exactly match its immutable actual journal')
        if old['native_source_binding']['source_video_sha256']!=source.SOURCE_SHA or old['native_source_binding']['decoded_native_RGB_sha256']!=x['current_RGB_sha256']:raise ValueError('Original source native RGB identity changed')
        for side in (0,1):
            a=rows[2*j+side];q=a['source_acquisition_query'];seed=np.asarray(d['original_acquisition_seed_uv']);normal=np.asarray(d['source_normal']);bounds=d['original_normal_offsets_bounds'];uv=fl[side].tolist()
            if (a['source_row_id']!='native_C_operator_supersession_v1/'+d['source_row_id']+'/flank'+str(side) or a['current_uv']!=uv or a['source_uv']!=uv or a['source_flanks_uv']!=d['source_flanks_uv'] or a['old_correct_midpoint_raw_index']!=x['raw_index'] or a['old_correct_midpoint_UV']!=d['correct_original_packet_UV'] or a['derived_midpoint_source_row_id']!=d['source_row_id'] or a['source_side_index']!=side or a['native_pts']!=d['native_pts'] or a['current_rgb_sha256']!=x['current_RGB_sha256'] or a['shot_id']!='s09' or a['joint_material_index']!=145 or a['joint_material_id']!='ice.C:segment0' or a['normal']!=d['source_normal'] or a['measurement_operator_resolved']!=acquisition.OPERATOR or a['frozen_original_material_receipt']!=r or a['source_geometry_eligible']is not True or any(a[k]is not False for k in ('solver_eligible','statistical_objective_eligible','numeric_statistics_ready','pixels_from_model','point_correspondence','source_position_correction'))):raise ValueError('Operator replacement redirects a real original interface')
            if q['alpha']!=bounds or q['native_uv']!=[(seed+t*normal).tolist()for t in bounds] or q['normal']!=normal.tolist() or q['original_midpoint_source_row_id']!=d['source_row_id'] or a['source_navigation_seed_uv']!=seed.tolist() or a['source_measured_alpha']!=d['source_alpha_flanks'][side]:raise ValueError('Original finite source acquisition changed')
    return b,rows,proof

def build_operator_transition_bundle(source_manifest,parent_model_path,output):
    from .joint_model import as_event_view,graph_from_joint
    from .camera_graph import CameraGraph
    from rinkfit_start.native_scene import NativeScene
    from .tail_paint_pair_append import _inverse_source_plane
    sp=Path(source_manifest).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('Immutable operator transition generation exists')
    b,rows,proof=_checked_source(sp,set());parent=read_json(parent_model_path);model=copy.deepcopy(parent)
    camera=CameraGraph(graph_from_joint(parent)).native_camera('s09');scene=NativeScene(as_event_view(parent,'s09'))
    uv=np.asarray([x['current_uv']for x in rows]);pts=np.asarray([x['native_pts']for x in rows]);times=(pts-parent['shots']['s09']['time_origin_pts'])/12800+camera.tau*(uv[:,1]/1079-.5)
    stations=np.linspace(0,1,1001);curve=scene.world(145,0,stations);initial=np.empty((26,2))
    for pt in np.unique(pts):
        inds=np.flatnonzero(pts==pt);pred=camera.at(curve,(pt-parent['shots']['s09']['time_origin_pts'])/12800)['uv'];dist=((uv[inds,None,:]-pred[None,:,:])**2).sum(2);initial[inds]=curve[np.argmin(dist,axis=1),:2]
    inverse,error=_inverse_source_plane(camera,uv,times,initial);path=parent['materials'][145]['paths'][0];q=np.asarray(parent['blocks'][path['geometry_block']]['values']);q=q[path['geometry_indices']]if path.get('geometry_indices')else q[path.get('geometry_offset',0):];t=q[2:4]-q[:2];normal=np.array([-t[1],t[0]])/np.linalg.norm(t)
    widths=model.setdefault('measurement_widths',[]);bindings=model.setdefault('stroke_pair_model_bindings',[]);allocation=[]
    for j in range(13):
        r=rows[2*j];scope=r['half_width_scope_id'];delta=float(np.dot(inverse[2*j+1]-inverse[2*j],normal));width=abs(delta)/2
        if not 1e-12<width<1:raise ValueError('Original source local width inverse has no valid numerical seed')
        signs=[-1,1]if delta>0 else[1,-1];name='measurement/paint_half_width/'+hashlib.sha256(scope.encode()).hexdigest()[:24];bi=len(model['blocks'])
        model['blocks'].append(dict(name=name,values=[width],lower=[1e-12],upper=[1.],scales=[max(width,1e-5)],constant=False,fixed_indices=[]))
        widths.append(dict(width_scope_id=scope,half_width_block=bi,scope_is_source_confirmed_section_not_uniform_whole_stroke=True,positive_scalar_measurement_nuisance=True))
        bindings.append(dict(pair_id=scope+'/strict_pair',width_scope_id=scope,half_width_block=bi,material_index=145,initial_path_index=0,source_world_signs=signs,allowed_world_side_assignments=[[-1,1],[1,-1]],source_refs=[b['source_rows_ref']],finite_association_is_numerical_seed_not_material_identity=True))
        allocation.append(dict(original_midpoint_id=r['derived_midpoint_source_row_id'],original_raw_index=r['old_correct_midpoint_raw_index'],half_width_scope_id=scope,half_width_block=bi,block_name=name,actual_flank_UV=r['source_flanks_uv'],source_world_signs=signs,inverse_ice_XY=inverse[2*j:2*j+2].tolist(),inverse_forward_error_px=error[2*j:2*j+2].tolist(),world_width_is_measured_truth=False,station_shared_between_flanks=False,width_shared_across_exposures=False))
    out.mkdir(parents=True);write_json(out/'model.json',model)
    manifest=dict(schema=BUNDLE_SCHEMA,original_source_authority=source.ref(sp),parent_packet=b['parent_packet'],parent_model=source.ref(parent_model_path),model=source.ref(out/'model.json'),source_rows_ref=b['source_rows_ref'],actual_replay=b['actual_replay'],new_local_widths=allocation,reader_module_ref=SELF,inverse_initializer_module_ref=source.ref(__import__('rinkfit_joint.tail_paint_pair_append',fromlist=['x']).__file__),old_midpoints_remain_correct_source_facts=True,operator_replacement_not_residual_rejection=True,source_mean_or_sigma_modified=False)
    write_json(out/'manifest.json',manifest);write_json(out/'material_map.json',dict(semantic_to_material={'ice.C:segment0':dict(material_index=145,geometry_ready=True)}))
    write_json(out/'chunk.json',dict(schema='rinkfit.actual_source_gate_reference_adapter.v1',source_rows_ref=b['source_rows_ref'],geometry_rows_ref=b['source_rows_ref'],source_gate_manifest=source.ref(out/'manifest.json'),source_dataset_tag='native_original_C_interfaces_v1',actual_source_rows=26,all_requested_frames_accounted=True,retain_unmeasured_candidates_in_source_journal_only=True))
    roles=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_joint9_1018a/roles/frame_roles.json');return source.ref(out/'manifest.json'),source.ref(roles)

def _expected_pair(r,N,j,b):
    return dict(id=r['half_width_scope_id']+'/strict_pair',source_row_id=r['derived_midpoint_source_row_id'],rows=[N+2*j,N+2*j+1],material_index=145,target_id='ice.C:segment0',shot_id='s09',native_pts=r['native_pts'],width_scope_id=r['half_width_scope_id'],allowed_world_side_assignments=[[-1,1],[1,-1]],actual_flank_uv=r['source_flanks_uv'],source_normal=r['normal'],source_family_id=r['source_family_id'],source_refs=[b['source_rows_ref']])

def install_operator_transition(packet,parent,out,bundlepath):
    b=read_json(bundlepath);rows=list(read_jsonl(b['source_rows_ref']['file']));N=parent['row_count'];packet=source.install_actual_queries(packet,b['source_rows_ref']['file'],out,parent)
    for key in ('stroke_pair_index','stroke_source_side'):
        d=packet['arrays'][key];a=np.memmap(d['file'],dtype=d['dtype'],shape=tuple(d['shape']),mode='r+');a[N:]=np.arange(26)//2+len(parent['stroke_pairs'])if key=='stroke_pair_index'else np.arange(26)%2;a.flush();del a;d['sha256']=file_sha256(d['file'])
    packet['stroke_pairs']=parent['stroke_pairs']+[_expected_pair(rows[2*j],N,j,b)for j in range(13)]
    indices={int(r['old_correct_midpoint_raw_index'])for r in rows};d=packet['arrays']['admission'];a=np.memmap(d['file'],dtype=d['dtype'],shape=tuple(d['shape']),mode='r+');a[list(indices)]=0;a.flush();del a;d['sha256']=file_sha256(d['file'])
    groups=[];removed=[]
    for index,g in enumerate(packet['groups']):
        hits=indices.intersection(g['rows'])
        if hits:
            if g['statistic_kind']!='constraint_only_unknown_noise'or g['rank']!=0 or g['W']!=[]:raise ValueError('Derived midpoint cannot be removed from a statistical precision unit')
            remaining=[i for i in g['rows']if i not in indices];removed.append(dict(original_group_index=index,original_group=g,remaining_rows=remaining))
            if remaining:groups.append(dict(g,rows=remaining))
        else:groups.append(g)
    packet['groups']=groups;packet['active_rows']-=13;packet['constraint_only_unknown_noise_rows']-=13;packet['constraint_only_unknown_noise_groups']=sum(g['statistic_kind']=='constraint_only_unknown_noise'for g in groups);packet['recapture_rows']=[i for i in packet['recapture_rows']if i not in indices]
    packet['C_original_midpoint_operator_transition']=dict(schema=CONTRACT,original_midpoint_raw_indices=sorted(indices),original_midpoint_raw_source_UV_and_ID_unchanged=True,original_midpoint_source_facts_retained=True,factor_role='derived_midpoint_navigation_only_superseded_by_its_two_original_interfaces',removed_original_rank0_groups=removed,new_actual_flanks=26,new_independent_local_widths=13,source_residual_used_for_selection=False,statistical_rank_changed=False)
    return packet

def assert_operator_transition(packet,packet_path,*,certificate=None,parent_document=None,verified_dependency_cache=None):
    path=Path(packet_path).resolve();seen=set()if verified_dependency_cache is None else verified_dependency_cache;cp,cert=_json(packet['board_material_source_qualification'],path.parent,seen)if certificate is None else certificate
    if cert.get('schema')!=SCHEMA or cert.get('parent_recursion_contract')!=CONTRACT or cert.get('issuer_module_ref')!=SELF or cert.get('recursive_parent_guard_module_ref')!=PARENT or cert.get('original_source_acquisition_module_ref')!=ACQUISITION:raise ValueError('Explicit original-C operator transition code contract differs')
    for x in (SELF,PARENT,ACQUISITION):_load_ref(x,cp.parent,seen)
    pp=_load_ref(cert['parent_packet'],cp.parent,seen);parent=parent_document if parent_document is not None else read_json(pp);parent_guard.assert_board_source_qualification_current(parent,pp,verified_dependency_cache=seen)
    bp=_load_ref(cert['operator_transition_bundle'],cp.parent,seen);b=read_json(bp);_source_refs(b,bp.parent,seen);_,rows,proof=_checked_source(_load_ref(b['original_source_authority'],bp.parent,seen),seen)
    if b.get('schema')!=BUNDLE_SCHEMA or b['parent_packet']!=cert['parent_packet']or b['reader_module_ref']!=SELF:raise ValueError('Operator transition bundle differs')
    N=parent['row_count'];M=packet['row_count'];indices={r['old_correct_midpoint_raw_index']for r in rows}
    if M!=N+26 or cert['original_raw_rows']!=N or cert['raw_rows']!=M or cert['superseded_derived_midpoint_indices']!=sorted(indices)or packet['shot_ids']!=parent['shot_ids']or packet['material_binding']!=parent['material_binding']:raise ValueError('Original C transition physical/source scope differs')
    for key in ('source_row_ids','source_dataset_ids','source_family_ids'):
        if len(packet[key])!=M or packet[key][:N]!=parent[key]:raise ValueError('Original C raw source registry changed')
    for key in ('independent_rows','independent_statistic_kind','source_station_domains','joint_statistics_rank'):
        if packet.get(key)!=parent.get(key):raise ValueError('Original C changed independent statistics/domain')
    if packet['queries'][:len(parent['queries'])]!=parent['queries']or len(packet['queries'])!=len(parent['queries'])+26 or packet['stroke_pairs'][:len(parent['stroke_pairs'])]!=parent['stroke_pairs']or len(packet['stroke_pairs'])!=len(parent['stroke_pairs'])+13:raise ValueError('Original source Query/pair order changed')
    oldgroups=[];removed=[]
    for j,g in enumerate(parent['groups']):
        hits=indices.intersection(g['rows'])
        if hits:
            if g['statistic_kind']!='constraint_only_unknown_noise'or g['rank']!=0 or g['W']!=[]:raise ValueError('Operator transition changes existing statistical precision')
            remaining=[i for i in g['rows']if i not in indices];removed.append(dict(original_group_index=j,original_group=g,remaining_rows=remaining))
            if remaining:oldgroups.append(dict(g,rows=remaining))
        else:oldgroups.append(g)
    if packet['groups'][:len(oldgroups)]!=oldgroups:raise ValueError('Operator transition modifies more than exact derived-midpoint membership')
    suffix=[]
    for g in packet['groups'][len(oldgroups):]:
        if g['statistic_kind']!='constraint_only_unknown_noise'or g['rank']!=0 or g['W']!=[]or g.get('unknown_noise_source_has_statistical_precision')is not False:raise ValueError('Original flank factor invents precision')
        suffix+=g['rows']
    if sorted(suffix)!=list(range(N,M)):raise ValueError('Actual original interfaces incomplete source partition')
    transition=packet['C_original_midpoint_operator_transition']
    if transition!=dict(schema=CONTRACT,original_midpoint_raw_indices=sorted(indices),original_midpoint_raw_source_UV_and_ID_unchanged=True,original_midpoint_source_facts_retained=True,factor_role='derived_midpoint_navigation_only_superseded_by_its_two_original_interfaces',removed_original_rank0_groups=removed,new_actual_flanks=26,new_independent_local_widths=13,source_residual_used_for_selection=False,statistical_rank_changed=False):raise ValueError('Original midpoint source facts/role justification differs')
    issued=dict(packet,arrays=cert['issued_arrays']);arrays={}
    if packet['arrays'].keys()!=parent['arrays'].keys()or issued['arrays'].keys()!=parent['arrays'].keys():raise ValueError('Original operator transition array ABI differs')
    for key in parent['arrays']:
        old=_array(parent,key,pp.parent,seen);new=_array(issued,key,path.parent,seen);current=_array(packet,key,path.parent,seen)
        if new.shape!=(M,*old.shape[1:])or old.dtype!=new.dtype or current.shape!=new.shape or current.dtype!=new.dtype:raise ValueError('Original source ABI/dtype changed')
        expected=old.copy()if key=='admission'else old
        if key=='admission':expected[list(indices)]=0
        if not np.array_equal(new[:N],expected):raise ValueError('Changed an inherited original source value outside explicit operator admission')
        if key not in AUX_ARRAYS and not np.array_equal(new,current):raise ValueError('Runtime changed immutable original source arrays')
        arrays[key]=current
    changed=any(packet['arrays'][k]['sha256']!=cert['issued_arrays'][k]['sha256']for k in AUX_ARRAYS)
    if changed:
        from .joint_solver import _source_membership_fingerprint
        fp=_source_membership_fingerprint(packet);found=False
        for ref in packet.get('source_identity_watchlist',[]):
            if ref.get('source_membership_fingerprint')!=fp:continue
            _,wrapped=_json(ref,path.parent,seen)
            if wrapped.get('board_material_source_qualification')==packet.get('board_material_source_qualification')and _source_membership_fingerprint(wrapped)==fp:found=True;break
        if not found:raise ValueError('Operator transition runtime aux needs exact same-source wrapper')
        for key in AUX_ARRAYS:
            a=_array(issued,key,path.parent,seen)
            if not np.array_equal(a[:parent['new_source_start']],arrays[key][:parent['new_source_start']]):raise ValueError('Operator transition changed protected S01 aux')
    for j,r in enumerate(rows):
        i=N+j;oldi=r['old_correct_midpoint_raw_index']
        if (parent['source_row_ids'][oldi]!=r['derived_midpoint_source_row_id']or arrays['uv'][oldi].tolist()!=r['old_correct_midpoint_UV']or arrays['admission'][oldi]!=0 or arrays['fit_role'][oldi]!=1 or arrays['uv'][i].tolist()!=r['current_uv']or arrays['profile_normal'][i].tolist()!=r['normal']or arrays['native_pts'][i]!=r['native_pts']or arrays['kind'][i]!=0 or arrays['material_index'][i]!=145 or arrays['admission'][i]!=1 or arrays['fit_role'][i]!=1 or arrays['E'][i]!=2 or arrays['source_sigma'][i]!=1 or packet['source_row_ids'][i]!=r['source_row_id']or arrays['stroke_pair_index'][i]!=len(parent['stroke_pairs'])+j//2 or arrays['stroke_source_side'][i]!=j%2 or arrays['query_uv'][i].tolist()!=r['source_navigation_seed_uv']or arrays['alpha_bounds'][i].tolist()!=r['source_acquisition_query']['alpha']or packet['queries'][len(parent['queries'])+j]!=dict(row_index=i,source_factor_row_id=r['source_row_id'],**r['source_acquisition_query'])):raise ValueError('Original source interfaces are not actually consumed at native UV')
        if packet['stroke_pairs'][len(parent['stroke_pairs'])+j//2]!=_expected_pair(rows[2*(j//2)],N,j//2,b):raise ValueError('Original material interfaces lost their actual pairing')
    parentmodel=read_json(_load_ref(b['parent_model'],bp.parent,seen));model=read_json(_load_ref(b['model'],bp.parent,seen));n=len(parentmodel['blocks'])
    if model['blocks'][:n]!=parentmodel['blocks']or len(model['blocks'])!=n+13 or len(b['new_local_widths'])!=13:raise ValueError('Original operator transition changed inherited physical state')
    for k,v in parentmodel.items():
        if k not in ('blocks','measurement_widths','stroke_pair_model_bindings')and model.get(k)!=v:raise ValueError('Original operator transition changed scene/camera/material')
    for key in ('measurement_widths','stroke_pair_model_bindings'):
        if model[key][:len(parentmodel[key])]!=parentmodel[key]or len(model[key])!=len(parentmodel[key])+13:raise ValueError('Original operator transition changed inherited width scopes')
    for j,record in enumerate(b['new_local_widths']):
        r=rows[2*j];block=model['blocks'][n+j];mw=model['measurement_widths'][len(parentmodel['measurement_widths'])+j];binding=model['stroke_pair_model_bindings'][len(parentmodel['stroke_pair_model_bindings'])+j]
        if record['original_raw_index']!=r['old_correct_midpoint_raw_index']or record['original_midpoint_id']!=r['derived_midpoint_source_row_id']or record['half_width_block']!=n+j or block['name']!=record['block_name']or len(block['values'])!=1 or not 0<block['values'][0]<=1 or block['constant']is not False or block['fixed_indices']!=[]or block['lower']!=[1e-12]or block['upper']!=[1.]or mw['half_width_block']!=n+j or mw['width_scope_id']!=r['half_width_scope_id']or binding['half_width_block']!=n+j or binding['pair_id']!=r['half_width_scope_id']+'/strict_pair'or binding['width_scope_id']!=r['half_width_scope_id']or binding['material_index']!=145 or binding['source_world_signs']not in [[-1,1],[1,-1]]or binding['allowed_world_side_assignments']!=[[-1,1],[1,-1]]or record['world_width_is_measured_truth']is not False:raise ValueError('Original operator transition width is not one local nuisance per actual profile')
    partition=np.zeros(M,np.uint8);partition[packet['independent_rows']]+=1
    for g in packet['groups']:partition[g['rows']]+=1
    active=(arrays['admission']==1)&(arrays['fit_role']==1)
    if not np.array_equal(partition,active.astype(np.uint8))or packet['active_rows']!=int(active.sum()):raise ValueError('Operator transition dropped unknowns or lost full active-source partition')
    return dict(source_rows=rows,replay_proofs=proof,new_actual_flanks=26,operator_superseded_correct_midpoints=13,original_source_UV_ID_retained=True,statistical_precision_invented=False)

def finalize_operator_transition(parent_packet_path,packet,bundlepath,out_certificate,*,parent_document=None):
    pp=Path(parent_packet_path).resolve();out=Path(out_certificate).resolve();bp=Path(bundlepath).resolve();parent=parent_document if parent_document is not None else read_json(pp);b=read_json(bp)
    if out.exists():raise FileExistsError('Immutable explicit operator transition certificate exists')
    rows=list(read_jsonl(b['source_rows_ref']['file']));model=read_json(b['parent_model']['file']);cert=dict(schema=SCHEMA,parent_recursion_contract=CONTRACT,parent_packet=source.ref(pp),operator_transition_bundle=source.ref(bp),issuer_module_ref=SELF,recursive_parent_guard_module_ref=PARENT,original_source_acquisition_module_ref=ACQUISITION,original_raw_rows=parent['row_count'],raw_rows=packet['row_count'],superseded_derived_midpoint_indices=sorted({r['old_correct_midpoint_raw_index']for r in rows}),issued_arrays=copy.deepcopy(packet['arrays']),inherited_material_count=len(model['materials']),inherited_physical_block_count=len(model['blocks']),source_UV_and_ID_preserved=True,factor_admission_transition_explicit=True,reason_is_operator_definition_not_residual=True)
    result=assert_operator_transition(packet,out.parent/'packet.json',certificate=(out,cert),parent_document=parent)
    write_json(out.parent/'actual_operator_consumption.json',{k:v for k,v in result.items()if k not in ('source_rows','replay_proofs')});write_json(out,cert);packet['board_material_source_qualification']=source.ref(out);packet.pop('accepted_explicit_state_binding',None);packet['all_new_source_nuisance_requires_fresh_physical_capture']=True
    return packet
