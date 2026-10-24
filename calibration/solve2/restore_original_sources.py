"""Restore explicitly confirmed current sources and their complete original W.

This current-data adapter does not run an optimizer or repeat source production.
Raw UV/PTS/IDs/E/sigma stay immutable. Only delivered restored feet are visited.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,importlib.util,json,sys
import numpy as np
from patch_original_full_input import read,ref,array,scope_baseline
from export_original_ad import checked,binary,save_json,sha
from original_full_model import register_full_model

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'calibration/rinkfit/src'))
from rinkfit_start.native_scene import NativeScene
from rinkfit_start.material_foot import _certified_candidates,_implicit_capture,select_certified_upper_bound
from rinkfit_start.batched_grid import _true_rs_grid_recovery
from rinkfit_start.semantic_resolution import load_semantics


def restore(input_dir,state_dir,delta_path,output,compiled_dir,native_factory=None):
    I=Path(input_dir).resolve();F=Path(state_dir).resolve();O=checked(output)
    if O.exists():raise FileExistsError('Refuse overwriting a current input')
    pk=read(I/'packet/manifest.json');sem=read(I/'semantic/manifest.json');query=read(I/'query/manifest.json');model=read(F/'model.json');parent=read(F/'report.json');d=read(delta_path)
    if parent['packet']!=ref(I/'packet/manifest.json')or parent['semantic']!=ref(I/'semantic/manifest.json'):raise ValueError('Restoration requires the exact current complete state')
    N=int(pk['observations']);raw={k:array(v)for k,v in pk['arrays'].items()};table=read(pk['string_tables']['file']);ids=table['stable_observation_ids'];lookup={sid:i for i,sid in enumerate(ids)}
    active=array(sem['active_source_mask']).astype(bool);stat=array(sem['statistical_active_mask']).astype(bool);material=array(sem['resolved_material_index']);variant=array(sem['resolved_variant_index']);sigma=array(sem['arrays']['source_sigma'])
    state={k:np.fromfile(F/name,dt)for k,name,dt in [('station','station.f64','<f8'),('branch','path.u32','<u4'),('row','initial_row.f64','<f8'),('alpha','initial_alpha.f64','<f8')]};before={k:v.copy()for k,v in state.items()};restored=[]
    for r in d['records']:
        i=lookup.get(r['stable_id'])
        if i is None or i!=r['full_row']or raw['native_pts'][i]!=r['native_pts']or raw['native_uv'][i].tolist()!=r['immutable_uv']or float(raw['engineering_budget_px'][i])!=r['engineering_budget_px']or float(sigma[i])!=r['original_sigma']:raise ValueError('Restored source differs from original immutable data')
        if active[i]or stat[i]:raise ValueError('A restored source is already active')
        path=model['materials'][int(material[i])]['paths'][int(state['branch'][i])]
        if 'previous_curve_id'in r:
            if r['previous_curve_id']not in (path['id'],path.get('current_curve_id')):raise ValueError('Restored source old finite binding differs')
            mi=int(r['current_material_index']);pi=int(r['current_path_index']);path=model['materials'][mi]['paths'][pi]
            if path['id']!=r['current_native_path_id']or path['kind']not in (2,3)or path.get('physical_point',False):raise ValueError('Restored source target is not the confirmed finite current entity')
            if path['variant_id']not in sem['resolved_variant_ids']:sem['resolved_variant_ids'].append(path['variant_id'])
            material[i]=mi;variant[i]=sem['resolved_variant_ids'].index(path['variant_id']);state['branch'][i]=pi
        if int(material[i])!=r['current_material_index']or int(state['branch'][i])!=r['current_path_index']or path['id']!=r['current_native_path_id']or sem['resolved_variant_ids'][int(variant[i])]!=path['variant_id']:raise ValueError('Restored source lacks the exact current physical crosswalk')
        active[i]=True;state['station'][i]=r['new_station_seed'];state['row'][i]=raw['native_uv'][i,1];restored.append(i)
    rows=np.asarray(restored,int)
    if len(np.unique(rows))!=len(rows):raise ValueError('Duplicate restored source')
    for r in d['remain_excluded']:
        i=int(r.get('full_row',r.get('original_row')))
        if active[i]:raise ValueError('An explicit retained exclusion was revived')
    # The existing original finite/RS machinery; no point identity is inferred
    # from the nearest curve, and the physical state is held exactly unchanged.
    sys.path.insert(0,str(Path(compiled_dir).resolve()));from bound_camera_sensor import bind_camera_from_model
    scene=NativeScene(model);camera=bind_camera_from_model(ref(F/'model.json'),model=model);times=(raw['native_pts'][rows]-pk['pts_origin'])/12800.;ss=state['station'][rows].copy();bb=state['branch'][rows].copy();capture=_certified_candidates(scene,camera,material[rows],bb,ss,times)
    proposal=_implicit_capture(scene,camera,material[rows],bb,ss,times,raw['native_uv'][rows],iterations=16,initial_row=capture['row']);trial=_certified_candidates(scene,camera,material[rows],bb,proposal,times)
    ss,bb,capture,_=select_certified_upper_bound(ss,bb,capture,proposal,bb,trial,raw['native_uv'][rows])
    recovery=_true_rs_grid_recovery(scene,camera,material[rows],raw['frame_index'][rows],times,raw['native_uv'][rows],np.ones(len(rows),bool),np.asarray(sem['resolved_variant_ids'])[variant[rows]],ss,bb,capture,same_branch_only=True,threshold_px=8.,grid_count=1025,progress=None)
    if not capture['valid'].all():raise ValueError('A restored source has no legal current true-RS foot')
    state['station'][rows]=ss;state['branch'][rows]=bb;state['row'][rows]=capture['row'];pred=np.load(F/'candidate_predictions.npz');values={k:pred[k].copy()for k in pred.files};values['uv'][rows]=capture['uv'];values['error_native_uv'][rows]=capture['uv']-raw['native_uv'][rows];values['coverage_error_px'][rows]=np.linalg.norm(values['error_native_uv'][rows],axis=1);values['RS_valid'][rows]=capture['valid']
    groups=[json.loads(line)for line in Path(sem['assets']['groups.jsonl']['file']).read_text(encoding='utf8').splitlines()];members=array(sem['arrays']['group_rows']).tolist();weights=array(sem['arrays']['weights']).tolist();group_ids={g['group_id']for g in groups};rank=int(sem['statistics']['formal_residual_rows']);new_group_rank=0
    if d['restore_whole_W_groups']:
        payload_path=Path(d['source_matrix_payload'].get('file',d['source_matrix_payload'].get('path')));payload=np.load(payload_path)
        if sha(payload_path)!=d['source_matrix_payload']['sha256']:raise ValueError('Original matrix payload differs')
    for g in d['restore_whole_W_groups']:
        if g['group_id']in group_ids:raise ValueError('An original W factor is already selected')
        rr=np.asarray(g['full_rows'],int);W=payload[g['matrix_npz_key']]
        if W.shape!=tuple(g['matrix_shape'])or W.dtype.str!=g['matrix_descriptor']['dtype']or hashlib.sha256(W.tobytes()).hexdigest()!=g['matrix_sha256']:raise ValueError('Original complete matrix bytes differ')
        if [ids[i]for i in rr]!=g['full_stable_ids']or not active[rr].all()or stat[rr].any():raise ValueError('A complete restored factor overlaps or lacks active support')
        ng=dict(group_id=g['group_id'],rank=int(g['rank']),member_start=len(members),member_count=len(rr),weight_start=len(weights),matrix_shape=list(W.shape),matrix_sha256=g['matrix_sha256'],native_pts=int(raw['native_pts'][rr[0]]),frame=int(raw['frame_index'][rr[0]]),covariance_origin=g['covariance_origin'],one_statistical_measurement_per_original_group=True,source_prepared_kind='original_restored_exact_entity')
        groups.append(ng);members.extend(rr.tolist());weights.extend(W.ravel().tolist());group_ids.add(g['group_id']);stat[rr]=True;rank+=len(W);new_group_rank+=len(W)
    ordinary=array(sem['arrays']['ordinary_rows']).astype(int).tolist();new_independent=[]
    for r in d['restore_independent_sources']:
        i=int(r['full_row'])
        if ids[i]!=r['stable_id']or not active[i]or stat[i]or float(sigma[i])!=r['original_sigma']or r['rank']!=2:raise ValueError('Original independent factor is duplicate or lacks its saved sigma')
        ordinary.append(i);stat[i]=True;rank+=2;new_independent.append(i)
    ordinary.sort();addition=None
    if native_factory:
        spec=importlib.util.spec_from_file_location('_current_native_source_factory',native_factory);factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory);updated,addition=factory.apply_native(model)
        if updated['blocks']!=model['blocks']or updated['camera']!=model['camera']or updated['materials'][:-1]!=model['materials']:raise ValueError('A detached static addition changed an existing physical definition')
        model=updated;sem['material_labels']=len(model['materials']);table['material_labels']=[v['label']for v in model['materials']]
    registration=read(I/'registration.json')
    if model.get('current_input_scope',{}).get('full_physical_solver',False):
        supported_ad_blocks={int(model['materials'][int(material[i])]['paths'][int(state['branch'][i])].get('advertisement_block',-1))for i in np.flatnonzero(active)}
        unsupported=[ad for ad in range(8,13)if next(j for j,b in enumerate(model['blocks'])if b['name']==f'ad/{ad}')not in supported_ad_blocks]
        supported_geometry={int(model['materials'][int(material[i])]['paths'][int(state['branch'][i])].get('geometry_block',-1))for i in np.flatnonzero(active)}
        unsupported_geometry=[b['name']for j,b in enumerate(model['blocks'])if b['name'].startswith('ice/')and j not in supported_geometry]
        model,registration=register_full_model(model,unobserved_artwork_ids=unsupported,unobserved_geometry_blocks=unsupported_geometry)
    O.mkdir(parents=True)
    for name in ('packet','semantic','query'):(O/name).mkdir()
    sd=O/'semantic';am=binary(sd,'active.u8',active,'|u1');sm=binary(sd,'statistical_active.u8',stat,'|u1');sem['resolved_material_index']=binary(sd,'resolved_material.u32',material,'<u4');sem['resolved_variant_index']=binary(sd,'resolved_variant.u32',variant,'<u4');sem['arrays']['ordinary_rows']=binary(sd,'ordinary_rows.u64',ordinary,'<u8');sem['arrays']['group_rows']=binary(sd,'group_rows.u64',members,'<u8');sem['arrays']['weights']=binary(sd,'weights.f64',weights,'<f8')
    gp=sd/'groups.jsonl';gp.write_text(''.join(json.dumps(g,ensure_ascii=False)+'\n'for g in groups),encoding='utf8');sem['assets']['groups.jsonl']=dict(file=str(gp),sha256=sha(gp));stats=deepcopy(sem['statistics']);stats.update(active_observations=int(active.sum()),statistical_observations=int(stat.sum()),formal_residual_rows=rank,original_group_count=len(groups),independent_rows=len(ordinary),active_source_mask_sha256=am['sha256'],statistical_active_mask_sha256=sm['sha256'],budget_only_observations=int(active.sum()-stat.sum()));save_json(sd/'active_statistics.json',stats)
    scope=deepcopy(sem['current_input_scope']);scope.update(group_count=len(groups),independent_count=len(ordinary),formal_W_rank=rank,statistical_active_mask=sm)
    if scope.get('full_physical_solver',False):scope.update(print_only_ad_ids=model['print_only_ad_ids'],unobserved_geometry_blocks=registration['unobserved_geometry_blocks_fixed'])
    model['current_input_scope']=scope;sem.update(active_source_mask=am,statistical_active_mask=sm,active_observations=int(active.sum()),statistics=stats,active_statistics=dict(file=str(sd/'active_statistics.json'),sha256=sha(sd/'active_statistics.json')),current_input_scope=scope)
    transition=dict(schema='rinkfit.current_exact_source_restoration.v1',delta=ref(delta_path),parent_packet=ref(I/'packet/manifest.json'),parent_complete_state=ref(F/'report.json'),restored_source_rows=restored,restored_complete_group_count=len(d['restore_whole_W_groups']),restored_group_rank=new_group_rank,restored_independent_rows=new_independent,restored_original_rank=new_group_rank+2*len(new_independent),retained_exclusions=d['remain_excluded'],raw_UV_PTS_ID_E_sigma_unchanged=True,all_existing_physical_values_unchanged=True,all_existing_W_matrices_unchanged=True,finite_foot_recovery=recovery,detached_static_addition=addition,source_qualification_replayed=False,optimizer_started=False)
    sem['current_exact_source_restoration']=transition;save_json(O/'model.json',model);save_json(sd/'manifest.json',sem);query['semantic_manifest']=ref(sd/'manifest.json');save_json(O/'query/manifest.json',query)
    if addition:save_json(O/'packet/string_tables.json',table);pk['string_tables']=dict(file=str(O/'packet/string_tables.json'),sha256=sha(O/'packet/string_tables.json'))
    pk.update(current_input_scope=scope,current_prepared_export=sem['current_prepared_export'],semantic_packet=dict(manifest=str(sd/'manifest.json'),sha256=sha(sd/'manifest.json')),query_packet=dict(manifest=str(O/'query/manifest.json'),sha256=sha(O/'query/manifest.json')),event_model=dict(manifest=str(O/'model.json'),sha256=sha(O/'model.json')),initial_state_directory=str(O));save_json(O/'packet/manifest.json',pk)
    for key,name,dt in [('station','station.f64','<f8'),('branch','path.u32','<u4'),('row','initial_row.f64','<f8'),('alpha','initial_alpha.f64','<f8')]:binary(O,name,state[key],dt)
    qrows=array(query['arrays']['source_rows']).astype(int);qa=binary(O,'query_initial_alpha.f64',state['alpha'][qrows],'<f8');save_json(O/'query_initialization.json',dict(query_packet_sha256=sha(O/'query/manifest.json'),alpha=qa));save_json(O/'initializer_manifest.json',dict(schema='rinkfit.current_source_numerical_seed.v1',semantic_manifest=ref(sd/'manifest.json'),files={name:ref(O/name)for name in ('station.f64','path.u32','initial_row.f64','initial_alpha.f64')},only_restored_source_feet_recomputed=True))
    unchanged=np.ones(N,bool);unchanged[rows]=False
    if any(not np.array_equal(v[unchanged],before[k][unchanged])for k,v in state.items()):raise AssertionError('An unchanged accepted auxiliary tuple changed')
    values.update(active_source_mask=active,stations=state['station'],branches=state['branch'],root_rows=state['row'],profile_alpha=state['alpha']);np.savez_compressed(O/'current_predictions.npz',**values)
    load_semantics(sd/'manifest.json',N,expected_sha256=sha(sd/'manifest.json'))
    mapping=read(I/'mapping.json');mapping.update(packet=str(O/'packet/manifest.json'),semantic=str(sd/'manifest.json'),query=str(O/'query/manifest.json'),model=str(O/'model.json'),active_rows=int(active.sum()),statistical_sources=int(stat.sum()),W_rank=rank,positive_W_groups=len(groups),independent_rows=len(ordinary),current_exact_source_restoration=transition);save_json(O/'mapping.json',mapping);save_json(O/'registration.json',registration);save_json(O/'delta_receipt.json',dict(parent_complete_state=ref(F/'report.json'),rebindings=[dict(row=i)for i in restored],restoration=transition))
    return dict(input=str(O),active_rows=int(active.sum()),W_rank=rank,restored_sources=len(rows),restored_original_rank=new_group_rank+2*len(new_independent),detached_static_addition=addition and addition['new_curve_id'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('input','state','delta','output','compiled-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--native-factory');p.add_argument('--baseline-output');a=p.parse_args();r=restore(a.input,a.state,a.delta,a.output,a.compiled_dir,a.native_factory)
    if a.baseline_output:r['baseline']=scope_baseline(a.output,a.baseline_output)
    print(json.dumps(r))
