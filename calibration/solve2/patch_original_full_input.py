"""Apply reviewed, specific source deltas to a current original-S01 input.

This is a current data/state adapter. It retains raw observations and original
factor matrices, and visits only changed finite feet. No optimizer, source
producer or historical qualification chain is run.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'calibration/rinkfit/src'))
from export_original_ad import checked,binary,save_json,sha
from original_full_model import register_full_model
from rinkfit_start.semantic_resolution import load_semantics


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def array(d):return np.fromfile(d.get('file',d.get('path')),d['dtype']).reshape(d['shape'])


def patch(input_dir,state_dir,output,delta_paths,compiled_dir,current_model=None):
    I=Path(input_dir).resolve();F=Path(state_dir).resolve();O=checked(output)
    O.relative_to(Path('D:/CodexData/RinkSpace_Data').resolve())
    if O.exists():raise FileExistsError('Refuse to overwrite a current input: '+str(O))
    pk=read(I/'packet/manifest.json');sem=read(I/'semantic/manifest.json');query=read(I/'query/manifest.json')
    parent=read(F/'report.json');model=read(F/'model.json');N=int(pk['observations'])
    source_model_change=None
    if current_model is not None:
        replacement=read(current_model)
        if replacement['blocks']!=model['blocks'] or replacement['camera']!=model['camera']:
            raise ValueError('Current static source update changed physical parameters or camera')
        previous_materials=model['materials'];new_materials=replacement['materials']
        if len(new_materials)<len(previous_materials):
            raise ValueError('Current static source update removed historical materials')
        metadata={'unused_legacy_witness','retired_alias','canonical_alias','canonical_curve_alias','alias_retirement','source_identity_status','provenance'}
        for old,new in zip(previous_materials,new_materials):
            if {k:v for k,v in old.items()if k not in metadata}!={k:v for k,v in new.items()if k not in metadata}:
                raise ValueError('Current static source update changed an existing mathematical path: '+old['label'])
        for material in new_materials[len(previous_materials):]:
            if not material.get('current_source_curve')or not material['label'].startswith('Ad'):
                raise ValueError('Only an explicit current advertisement source path may be appended')
            for path in material['paths']:
                if path['kind']not in (2,3)or not path.get('current_curve_id'):
                    raise ValueError('Appended source is not a declared finite print/board path')
                points=np.asarray(path.get('points',()),float)
                if points.ndim!=2 or not len(points)or not np.isfinite(points).all():
                    raise ValueError('Appended source curve has invalid finite vertices')
                for key in ('advertisement_block','geometry_block','whole_print_block'):
                    if key in path and not 0<=int(path[key])<len(model['blocks']):
                        raise ValueError('Appended source path uses an absent physical block')
        source_model_change=dict(current_model=ref(current_model),existing_mathematical_paths_and_physical_values_unchanged=True,
            appended_materials=len(new_materials)-len(previous_materials),existing_source_aliases_preserved=True,no_new_parameter_blocks=True)
        model=replacement
        for material in model['materials']:
            for path in material['paths']:
                variant_id=path['variant_id']
                if variant_id not in sem['resolved_variant_ids']:sem['resolved_variant_ids'].append(variant_id)
        sem['material_labels']=len(model['materials'])
    if parent['semantic']['sha256']!=sha(I/'semantic/manifest.json')or parent['packet']['sha256']!=sha(I/'packet/manifest.json'):
        raise ValueError('Delta state must belong to this current numerical input')
    raw={k:array(d)for k,d in pk['arrays'].items()};tables=read(pk['string_tables']['file']);ids=tables['stable_observation_ids']
    if len(ids)!=N or len(set(ids))!=N:raise ValueError('Current source IDs are not unique')
    id_to_row={sid:i for i,sid in enumerate(ids)}
    material=array(sem['resolved_material_index']).copy();variant=array(sem['resolved_variant_index']).copy()
    active=array(sem['active_source_mask']).astype(bool);previous_active=active.copy()
    oldstat=array(sem['statistical_active_mask']).astype(bool)
    state={k:np.fromfile(F/name,dt)for k,name,dt in [('station','station.f64','<f8'),('branch','path.u32','<u4'),('row','initial_row.f64','<f8'),('alpha','initial_alpha.f64','<f8')]}
    previous_state={k:v.copy()for k,v in state.items()}
    paths={}
    for mi,m in enumerate(model['materials']):
        for pi,p in enumerate(m['paths']):
            for cid in {p['id'],p.get('current_curve_id',p['id'])}:
                paths.setdefault(cid,[]).append((mi,pi,p))
    def source(record):
        sid=record.get('stable_id',record.get('source_id'));i=id_to_row.get(sid)
        if i is None:raise ValueError('Reviewed source is absent: '+str(sid))
        rr=record.get('original_row',record.get('prepared_row'));uv=record.get('immutable_uv',record.get('native_uv'))
        if rr!=int(raw['source_rows_full'][i])or record['native_pts']!=int(raw['native_pts'][i])or uv!=raw['native_uv'][i].tolist():raise ValueError('Reviewed raw source bytes differ: '+sid)
        p=model['materials'][int(material[i])]['paths'][int(state['branch'][i])]
        if record['old_curve_id'] not in (p['id'],p.get('current_curve_id')):raise ValueError('Reviewed old finite binding differs: '+sid)
        if record.get('engineering_budget_px')!=float(raw['engineering_budget_px'][i]):raise ValueError('Original engineering budget differs: '+sid)
        if not active[i]:raise ValueError('A specific delta must not silently revive a pending source: '+sid)
        return i
    changed=[];pending=[];withdraw_groups={};seen=set();receipts=[]
    for dp in delta_paths:
        d=read(dp);receipts.append(ref(dp))
        for record in d.get('records',d.get('rebind',[])):
            i=source(record)
            if i in seen:raise ValueError('A current source is edited by two deltas')
            seen.add(i);candidates=paths.get(record['new_curve_id'],[])
            if 'new_material_index'in record:candidates=[v for v in candidates if v[0]==record['new_material_index']and v[1]==record['new_path_index']]
            if len(candidates)!=1:raise ValueError('Reviewed new finite path is absent or ambiguous: '+record['new_curve_id'])
            mi,pi,p=candidates[0]
            if p['kind']not in (2,3)or p.get('physical_point',False):raise ValueError('Only the reviewed finite print/board-boundary interpretation may change here')
            seed=float(record.get('new_station_seed',record.get('station_seed')))
            if not np.isfinite(seed)or not 0<=seed<=1:raise ValueError('Invalid reviewed finite search seed')
            material[i]=mi;variant[i]=sem['resolved_variant_ids'].index(p['variant_id']);state['branch'][i]=pi;state['station'][i]=seed
            state['row'][i]=raw['native_uv'][i,1]
            changed.append(dict(row=i,source_id=ids[i],old_curve_id=record['old_curve_id'],new_curve_id=record['new_curve_id'],material=mi,branch=pi,statistical_before=bool(oldstat[i])))
        for record in d.get('pending_records',d.get('pending',[])):
            i=source(record)
            if i in seen:raise ValueError('A current source is edited by two deltas')
            seen.add(i);active[i]=False;pending.append(dict(row=i,source_id=ids[i],reason=record.get('reason',record.get('image_confirmed_boundary','specific native-image localization pending'))))
        for group in d.get('whole_W_groups',[]):
            if group['group_id']in withdraw_groups:raise ValueError('Duplicate whole-W withdrawal')
            withdraw_groups[group['group_id']]=group
    rows=np.asarray([v['row']for v in changed],int)
    # Recompute the true finite/RS feet only for explicitly changed bindings.
    # The same original fixed-model implementation is used; other accepted
    # stations, roots, projections and physical values stay untouched.
    from rinkfit_start.native_scene import NativeScene
    from rinkfit_start.material_foot import _certified_candidates
    from rinkfit_start.batched_grid import _true_rs_grid_recovery
    sys.path.insert(0,str(Path(compiled_dir).resolve()))
    from bound_camera_sensor import bind_camera_from_model
    camera=bind_camera_from_model(ref(F/'model.json'),model=model);scene=NativeScene(model)
    predictions=np.load(F/'candidate_predictions.npz');values={k:predictions[k].copy()for k in predictions.files}
    if len(rows):
        ss=state['station'][rows].copy();bb=state['branch'][rows].copy();times=(raw['native_pts'][rows]-pk['pts_origin'])/12800.
        capture=_certified_candidates(scene,camera,material[rows],bb,ss,times)
        report=_true_rs_grid_recovery(scene,camera,material[rows],raw['frame_index'][rows],times,raw['native_uv'][rows],np.ones(len(rows),bool),np.asarray(sem['resolved_variant_ids'])[variant[rows]],ss,bb,capture,same_branch_only=True,threshold_px=-1.,grid_count=1025,progress=None)
        if not capture['valid'].all():raise ValueError('Changed binding has no legal current true-RS foot')
        state['station'][rows]=ss;state['branch'][rows]=bb;state['row'][rows]=capture['row']
        values['uv'][rows]=capture['uv'];values['error_native_uv'][rows]=capture['uv']-raw['native_uv'][rows]
        values['coverage_error_px'][rows]=np.linalg.norm(values['error_native_uv'][rows],axis=1);values['RS_valid'][rows]=capture['valid']
    else:report={}
    # Keep complete positive factors. A withdrawn group is not split, and its
    # other genuine source points remain engineering observations.
    oldgroups=[json.loads(line)for line in Path(sem['assets']['groups.jsonl']['file']).read_text(encoding='utf8').splitlines()]
    members=array(sem['arrays']['group_rows']);weights=array(sem['arrays']['weights']);groups=[];newrows=[];newweights=[];partition=np.zeros(N,np.uint8);rank=0;removed_rank=0;found=set()
    for g in oldgroups:
        rr=members[g['member_start']:g['member_start']+g['member_count']].astype(int);W=weights[g['weight_start']:g['weight_start']+g['rank']*2*len(rr)]
        if g['group_id']in withdraw_groups:
            declaration=withdraw_groups[g['group_id']]
            if int(declaration['rank'])!=int(g['rank'])or declaration['full_rows']!=rr.tolist():raise ValueError('Reviewed whole-W support differs')
            removed_rank+=int(g['rank']);found.add(g['group_id']);continue
        if not active[rr].all():raise ValueError('A pending statistical source lacks an explicit whole-factor withdrawal')
        ng=deepcopy(g);ng.update(member_start=len(newrows),weight_start=len(newweights));groups.append(ng)
        newrows.extend(rr.tolist());newweights.extend(W.tolist());partition[rr]+=1;rank+=int(g['rank'])
    if found!=set(withdraw_groups):raise ValueError('A reviewed whole-W group is absent')
    ordinary=array(sem['arrays']['ordinary_rows']).astype(int);removed_independent=ordinary[~active[ordinary]];ordinary=ordinary[active[ordinary]]
    partition[ordinary]+=1;rank+=2*len(ordinary)
    if np.any(partition>1)or np.any((partition>0)&~active):raise ValueError('Current statistical partition is not exact')
    unsupported=[]
    for ad in range(8,13):
        ai=next(j for j,b in enumerate(model['blocks'])if b['name']==f'ad/{ad}')
        supported=any(active[i]and model['materials'][int(material[i])]['paths'][int(state['branch'][i])].get('advertisement_block')==ai for i in range(N))
        if not supported:unsupported.append(ad)
    unsupported_values=[]
    original=read(I/'model.json')
    for ad in unsupported:
        qi=next(j for j,b in enumerate(model['blocks'])if b['name']==f'print/Ad{ad}_current_whole')
        unsupported_values.append(dict(advertisement=ad,block=qi,discarded_unobserved_candidate_values=model['blocks'][qi]['values'],preserved_confirmed_values=original['blocks'][qi]['values']))
        model['blocks'][qi]['values']=deepcopy(original['blocks'][qi]['values'])
    supported_geometry={int(model['materials'][int(material[i])]['paths'][int(state['branch'][i])].get('geometry_block',-1))for i in np.flatnonzero(active)}
    unsupported_geometry=[b['name']for j,b in enumerate(model['blocks'])if b['name'].startswith('ice/')and j not in supported_geometry]
    model,registration=register_full_model(model,unobserved_artwork_ids=unsupported,unobserved_geometry_blocks=unsupported_geometry)
    O.mkdir(parents=True);sd=O/'semantic';pd=O/'packet';qd=O/'query'
    for path in (sd,pd,qd):path.mkdir()
    sem['resolved_material_index']=binary(sd,'resolved_material.u32',material,'<u4');sem['resolved_variant_index']=binary(sd,'resolved_variant.u32',variant,'<u4')
    am=binary(sd,'active.u8',active,'|u1');sm=binary(sd,'statistical_active.u8',partition,'|u1')
    sem['arrays']['ordinary_rows']=binary(sd,'ordinary_rows.u64',ordinary,'<u8');sem['arrays']['group_rows']=binary(sd,'group_rows.u64',newrows,'<u8');sem['arrays']['weights']=binary(sd,'weights.f64',newweights,'<f8')
    gp=sd/'groups.jsonl';gp.write_text(''.join(json.dumps(g,ensure_ascii=False)+'\n'for g in groups),encoding='utf8');sem['assets']['groups.jsonl']=dict(file=str(gp),sha256=sha(gp))
    stats=deepcopy(sem['statistics']);stats.update(active_observations=int(active.sum()),statistical_observations=int(partition.sum()),formal_residual_rows=rank,original_group_count=len(groups),independent_rows=len(ordinary),active_source_mask_sha256=am['sha256'],statistical_active_mask_sha256=sm['sha256'],budget_only_observations=int(active.sum()-partition.sum()))
    save_json(sd/'active_statistics.json',stats);scope=deepcopy(sem['current_input_scope']);scope.update(group_count=len(groups),independent_count=len(ordinary),formal_W_rank=rank,statistical_active_mask=sm,print_only_ad_ids=model['print_only_ad_ids'],unobserved_geometry_blocks=unsupported_geometry)
    model['current_input_scope']=scope;sem.update(active_source_mask=am,statistical_active_mask=sm,active_observations=int(active.sum()),active_statistics=dict(file=str(sd/'active_statistics.json'),sha256=sha(sd/'active_statistics.json')),statistics=stats,current_input_scope=scope)
    receipt=dict(schema='rinkfit.current_full_S01_specific_delta.v1',current_static_source_model_update=source_model_change,parent_packet=ref(I/'packet/manifest.json'),parent_complete_state=ref(F/'report.json'),reviewed_deltas=receipts,rebindings=changed,pending_sources=pending,withdrawn_complete_group_ids=sorted(found),withdrawn_group_rank=removed_rank,withdrawn_independent_rows=removed_independent.tolist(),withdrawn_independent_rank=2*len(removed_independent),raw_observations_and_E_sigma_unchanged=True,remaining_W_matrices_and_support_unchanged=True,unobserved_objects_fixed=unsupported_values,source_qualification_replayed=False,optimizer_started=False,finite_foot_recovery=report)
    sem['current_specific_delta']=receipt
    save_json(O/'model.json',model);save_json(sd/'manifest.json',sem)
    query['semantic_manifest']=ref(sd/'manifest.json');save_json(qd/'manifest.json',query)
    pk.update(current_input_scope=scope,current_prepared_export=sem['current_prepared_export'],semantic_packet=dict(manifest=str(sd/'manifest.json'),sha256=sha(sd/'manifest.json')),query_packet=dict(manifest=str(qd/'manifest.json'),sha256=sha(qd/'manifest.json')),event_model=dict(manifest=str(O/'model.json'),sha256=sha(O/'model.json')),initial_state_directory=str(O))
    save_json(pd/'manifest.json',pk)
    for key,name,dt in [('station','station.f64','<f8'),('branch','path.u32','<u4'),('row','initial_row.f64','<f8'),('alpha','initial_alpha.f64','<f8')]:binary(O,name,state[key],dt)
    qrows=array(query['arrays']['source_rows']).astype(int);alpha=binary(O,'query_initial_alpha.f64',state['alpha'][qrows],'<f8');save_json(O/'query_initialization.json',dict(query_packet_sha256=sha(qd/'manifest.json'),alpha=alpha))
    save_json(O/'initializer_manifest.json',dict(schema='rinkfit.current_source_numerical_seed.v1',semantic_manifest=ref(sd/'manifest.json'),files={n:ref(O/n)for n in ('station.f64','path.u32','initial_row.f64','initial_alpha.f64')},current_complete_delta_feet_recovered=True,unchanged_accepted_four_aux_preserved=True))
    unaffected=np.ones(N,bool);unaffected[rows]=False
    if any(not np.array_equal(state[k][unaffected],v[unaffected])for k,v in previous_state.items()):raise AssertionError('An unchanged accepted auxiliary changed')
    _,_,_,validated=load_semantics(sd/'manifest.json',N,expected_sha256=sha(sd/'manifest.json'))
    if not np.array_equal(validated['source_exclusion']['statistical_active_mask'],partition.astype(bool)):raise AssertionError('Current selected statistics mask differs')
    # Cached projections remain actual at the same physical state. Only the
    # reviewed finite bindings above needed reprojection, not a whole full2.
    values.update(active_source_mask=active,stations=state['station'],branches=state['branch'],root_rows=state['row'],profile_alpha=state['alpha'])
    np.savez_compressed(O/'current_predictions.npz',**values)
    mapping=read(I/'mapping.json');mapping.update(packet=str(pd/'manifest.json'),semantic=str(sd/'manifest.json'),query=str(qd/'manifest.json'),model=str(O/'model.json'),active_rows=int(active.sum()),statistical_sources=int(partition.sum()),W_rank=rank,positive_W_groups=len(groups),independent_rows=len(ordinary),specific_current_delta=receipt)
    save_json(O/'mapping.json',mapping);save_json(O/'registration.json',registration);save_json(O/'delta_receipt.json',receipt)
    return dict(input=str(O),raw_rows=N,active_rows=int(active.sum()),W_rank=rank,changed_feet=len(rows),pending_sources=len(pending),unobserved_fixed=unsupported)


def scope_baseline(input_dir,output):
    """Evaluate the new factor scope from retained exact current projections.

    The parent full2 evaluated the same physical state, and patch() reprojected
    every changed binding. All remaining source projections are unchanged.
    Scope withdrawal is explicitly separated from numerical optimization.
    """
    I=Path(input_dir).resolve();O=checked(output)
    if O.exists():raise FileExistsError('Refuse overwriting a current complete scope state')
    pk=read(I/'packet/manifest.json');sem=read(I/'semantic/manifest.json');delta=read(I/'delta_receipt.json')
    previous=read(delta['parent_complete_state']['path']);pred=np.load(I/'current_predictions.npz')
    error=pred['error_native_uv'];active=array(sem['active_source_mask']).astype(bool);coverage=pred['coverage_error_px'];budget=array(pk['arrays']['engineering_budget_px'])
    if not pred['RS_valid'][active].all()or not np.isfinite(error[active]).all():raise ValueError('New current scope has unresolved source projections')
    members=array(sem['arrays']['group_rows']);weights=array(sem['arrays']['weights']);ordinary=array(sem['arrays']['ordinary_rows']).astype(int);sigma=array(sem['arrays']['source_sigma']);groups=[json.loads(line)for line in Path(sem['assets']['groups.jsonl']['file']).read_text(encoding='utf8').splitlines()]
    total=0.
    for g in groups:
        rr=members[g['member_start']:g['member_start']+g['member_count']].astype(int);W=weights[g['weight_start']:g['weight_start']+g['rank']*2*len(rr)].reshape(g['rank'],2*len(rr));r=W@error[rr].reshape(-1);total+=.5*float(r@r)
    total+=.5*float(np.sum((error[ordinary]/sigma[ordinary,None])**2));q=(coverage/budget)**2
    best=deepcopy(previous['best']);best.update(label='current_scope_baseline',valid=True,active_RS_valid_rows=int(active.sum()),max_q=float(q[active].max()),maximum_error_px=float(coverage[active].max()),quantiles_px=np.quantile(coverage[active],[.5,.95,.99,1]).tolist(),over2=int(np.count_nonzero(active&(coverage>2))),whole_original_W_half_cost=total,maximum_source_row=int(np.flatnonzero(active)[np.argmax(q[active])]),elapsed_seconds=0.,locally_captured_ordinary_candidates=len(delta['rebindings']),inner_active_set_history=[])
    O.mkdir(parents=True)
    for name in ('model.json','station.f64','path.u32','initial_row.f64','initial_alpha.f64'):(O/name).write_bytes((I/name).read_bytes())
    (O/'candidate_predictions.npz').write_bytes((I/'current_predictions.npz').read_bytes())
    report=deepcopy(previous);report.update(run=str(O.parent),packet=ref(I/'packet/manifest.json'),semantic=ref(I/'semantic/manifest.json'),query_packet=ref(I/'query/manifest.json'),model_parent=ref(I/'model.json'),active_rows=int(active.sum()),statistics_rank=sem['statistics']['formal_residual_rows'],best=best,history=[best],initialization_only=True,final_physical_fit_accepted=False,source_W_E_unchanged=True,only_physical_columns_changed=[],reviewed_rebound_rows=[v['row']for v in delta['rebindings']],current_scope_transition=ref(I/'delta_receipt.json'),previous_scope_cost_not_an_optimization_step=previous['best']['whole_original_W_half_cost'],retained_current_projection_cache=ref(I/'current_predictions.npz'),unchanged_source_projections_not_recomputed=True,changed_source_feet_only_recomputed=True,elapsed_seconds=0.)
    for key in ('compiled_camera_snapshot','GPU_camera','CPU_grid_batch','CPU_compiled_camera','stage_seconds'):report.pop(key,None)
    report['implementation_source']=ref(__file__);report['files']={p.name:sha(p)for p in O.iterdir()};save_json(O/'report.json',report)
    return dict(complete_state=str(O/'report.json'),new_scope_baseline_W_half_cost=total,new_scope_engineering_b=best['max_q'],scope_change_not_optimization=True)


def rebase_current(input_dir,reference_input,reference_state,output,compiled_dir):
    """Keep the corrected current data and restore a visually accepted state.

    All unchanged material/variant/branch/station/row/alpha tuples and actual
    projections come together from that state. Only current changed bindings
    need new finite/RS feet. The rejected camera is never an initializer.
    """
    I=Path(input_dir).resolve();RI=Path(reference_input).resolve();F=Path(reference_state).resolve();O=checked(output)
    if O.exists():raise FileExistsError('Refuse overwriting a current rebased input')
    pk=read(I/'packet/manifest.json');rp=read(RI/'packet/manifest.json');sem=read(I/'semantic/manifest.json');rs=read(RI/'semantic/manifest.json');query=read(I/'query/manifest.json');model=read(I/'model.json');reference=read(F/'model.json')
    if pk['arrays']!=rp['arrays']or pk['immutable_source_hashes']!=rp['immutable_source_hashes']or model['camera']!=reference['camera']or model['materials']!=reference['materials']:raise ValueError('A physical rebase cannot change source identities or static scene definitions')
    if [b['name']for b in model['blocks']]!=[b['name']for b in reference['blocks']]:raise ValueError('Physical rebase parameter registry differs')
    for b,old in zip(model['blocks'],reference['blocks']):b['values']=deepcopy(old['values'])
    material=array(sem['resolved_material_index']);variant=array(sem['resolved_variant_index']);oldmaterial=array(rs['resolved_material_index']);oldvariant=array(rs['resolved_variant_index']);active=array(sem['active_source_mask']).astype(bool)
    changed=np.flatnonzero(active&((material!=oldmaterial)|(variant!=oldvariant)))
    state={k:np.fromfile(F/name,dt)for k,name,dt in [('station','station.f64','<f8'),('branch','path.u32','<u4'),('row','initial_row.f64','<f8'),('alpha','initial_alpha.f64','<f8')]}
    before={k:v.copy()for k,v in state.items()};uv=array(pk['arrays']['native_uv']);pts=array(pk['arrays']['native_pts']);frames=array(pk['arrays']['frame_index'])
    for key,name,dt in [('station','station.f64','<f8'),('branch','path.u32','<u4')]:state[key][changed]=np.fromfile(I/name,dt)[changed]
    state['row'][changed]=uv[changed,1]
    from rinkfit_start.native_scene import NativeScene
    from rinkfit_start.material_foot import _certified_candidates
    from rinkfit_start.batched_grid import _true_rs_grid_recovery
    sys.path.insert(0,str(Path(compiled_dir).resolve()));from bound_camera_sensor import bind_camera_from_model
    scene=NativeScene(model);camera=bind_camera_from_model(ref(F/'model.json'),model=reference);pred=np.load(F/'candidate_predictions.npz');values={k:pred[k].copy()for k in pred.files}
    if len(changed):
        ss=state['station'][changed].copy();bb=state['branch'][changed].copy();times=(pts[changed]-pk['pts_origin'])/12800.
        capture=_certified_candidates(scene,camera,material[changed],bb,ss,times)
        recovery=_true_rs_grid_recovery(scene,camera,material[changed],frames[changed],times,uv[changed],np.ones(len(changed),bool),np.asarray(sem['resolved_variant_ids'])[variant[changed]],ss,bb,capture,same_branch_only=True,threshold_px=-1.,grid_count=1025,progress=None)
        if not capture['valid'].all():raise ValueError('A changed binding has no legal foot at the accepted reference state')
        state['station'][changed]=ss;state['branch'][changed]=bb;state['row'][changed]=capture['row'];values['uv'][changed]=capture['uv'];values['RS_valid'][changed]=capture['valid'];values['error_native_uv'][changed]=capture['uv']-uv[changed];values['coverage_error_px'][changed]=np.linalg.norm(values['error_native_uv'][changed],axis=1)
    else:recovery={}
    unchanged=np.ones(len(active),bool);unchanged[changed]=False
    if any(not np.array_equal(v[unchanged],before[k][unchanged])for k,v in state.items()):raise AssertionError('Unchanged accepted complete tuple was not preserved')
    O.mkdir(parents=True)
    for name in ('packet','semantic','query'):(O/name).mkdir()
    save_json(O/'model.json',model);sem['current_visual_rebase']=dict(accepted_reference_state=ref(F/'report.json'),physical_values_restored=True,unchanged_complete_auxiliary_tuple_restored=True,changed_active_bindings_only=len(changed),source_data_and_W_selection_unchanged=True)
    save_json(O/'semantic/manifest.json',sem);query['semantic_manifest']=ref(O/'semantic/manifest.json');save_json(O/'query/manifest.json',query)
    pk.update(semantic_packet=dict(manifest=str(O/'semantic/manifest.json'),sha256=sha(O/'semantic/manifest.json')),query_packet=dict(manifest=str(O/'query/manifest.json'),sha256=sha(O/'query/manifest.json')),event_model=dict(manifest=str(O/'model.json'),sha256=sha(O/'model.json')),initial_state_directory=str(O));save_json(O/'packet/manifest.json',pk)
    for key,name,dt in [('station','station.f64','<f8'),('branch','path.u32','<u4'),('row','initial_row.f64','<f8'),('alpha','initial_alpha.f64','<f8')]:binary(O,name,state[key],dt)
    qrows=array(query['arrays']['source_rows']).astype(int);qa=binary(O,'query_initial_alpha.f64',state['alpha'][qrows],'<f8');save_json(O/'query_initialization.json',dict(query_packet_sha256=sha(O/'query/manifest.json'),alpha=qa))
    save_json(O/'initializer_manifest.json',dict(schema='rinkfit.current_source_numerical_seed.v1',semantic_manifest=ref(O/'semantic/manifest.json'),files={name:ref(O/name)for name in ('station.f64','path.u32','initial_row.f64','initial_alpha.f64')},accepted_reference_complete_tuple=True,changed_binding_feet_recomputed=True))
    values.update(active_source_mask=active,stations=state['station'],branches=state['branch'],root_rows=state['row'],profile_alpha=state['alpha']);np.savez_compressed(O/'current_predictions.npz',**values)
    receipt=dict(schema='rinkfit.current_full_S01_visual_rebase.v1',parent_packet=ref(I/'packet/manifest.json'),parent_complete_state=ref(F/'report.json'),current_corrected_data_input=str(I),rebindings=[dict(row=int(i))for i in changed],accepted_reference_physical_values_bitexact=True,complete_unchanged_auxiliary_tuple_bitexact=True,no_rejected_camera_or_geometry_inherited=True,no_W_E_sigma_or_source_pixels_changed=True,source_qualification_replayed=False,optimizer_started=False,finite_foot_recovery=recovery)
    save_json(O/'delta_receipt.json',receipt);mapping=read(I/'mapping.json');mapping.update(packet=str(O/'packet/manifest.json'),semantic=str(O/'semantic/manifest.json'),query=str(O/'query/manifest.json'),model=str(O/'model.json'),current_visual_rebase=receipt);save_json(O/'mapping.json',mapping);save_json(O/'registration.json',read(I/'registration.json'))
    load_semantics(O/'semantic/manifest.json',pk['observations'],expected_sha256=sha(O/'semantic/manifest.json'))
    return dict(input=str(O),restored_physical_blocks=len(model['blocks']),unchanged_complete_tuples=int(unchanged.sum()),changed_binding_feet=len(changed))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('input','state','output','compiled-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--delta',action='append',required=True);p.add_argument('--baseline-output');p.add_argument('--current-model',help='Current append-only static advertisement source model; no physical parameter update');a=p.parse_args()
    result=patch(a.input,a.state,a.output,a.delta,a.compiled_dir,a.current_model)
    if a.baseline_output:result['baseline']=scope_baseline(a.output,a.baseline_output)
    print(json.dumps(result))
