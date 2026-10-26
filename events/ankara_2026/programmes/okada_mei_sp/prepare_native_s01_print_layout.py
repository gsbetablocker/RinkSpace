"""Stage an explicit native print-layout revision against the current point input.

Preserves camera, board, ice, size/installation values and raw observations.
The new component shapes start from the selected native reference artwork;
the retired independent glyph parameters are archived, not imported as facts.
"""
from pathlib import Path
import argparse,importlib.util,io,json,hashlib,os,shutil
import numpy as np
spec=importlib.util.spec_from_file_location('prep',Path(__file__).with_name('prepare_native_s01.py'))
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.print_parameters import parameterize_print_layout
from rinkspace.domain.scene_assets import read_scene_assets,scene_content_identity,snapshot_scene
from rinkspace.path_policy import guard_project_write


def stage(out=None,plan_path=None,*,preserve_unchanged_shapes=False):
    out=Path(out) if out is not None else prep.RUN/'print_component_layout'
    if out.exists():raise FileExistsError('Existing staged state must be consumed, not overwritten: '+str(out))
    plan_path=Path(plan_path) if plan_path is not None else prep.SCENE.parent.parent/'print_layout_components.json';plan=prep.read(plan_path)
    source_scene=prep.read(prep.SCENE)
    old_model=SceneParameterModel(source_scene,prep.SCENE,optimize_scene=True)
    blob=(prep.RUN/'accepted_pixels_latest.npz').read_bytes()
    with np.load(io.BytesIO(blob)) as f:saved={k:f[k] for k in f.files}
    old_state=ParameterState(old_model.initial_state.layout,saved['scene_values'])
    input_hash=prep.digest(prep.RUN/'point_observations.npz')
    if str(saved['point_input_sha256'])!=input_hash:raise ValueError('Current checkpoint/input mismatch')
    out.mkdir()
    (out/'before_state.npz').write_bytes(blob)
    before,_=snapshot_scene(prep.SCENE,out/'before_scene')
    candidate,_=snapshot_scene(prep.SCENE,out/'scene')
    candidate_path=out/'scene/scene.json'
    records={r['template_id']:r for r in plan['records']};changes=[]
    for ref,asset in read_scene_assets(candidate,candidate_path):
        record=records[ref['template_id']]
        if record['source_asset_sha256']!=ref['sha256']:raise ValueError('Component evidence belongs to another native template')
        new=parameterize_print_layout(asset,{ref['template_id']:record['groups']})
        for old_t,new_t in zip(asset['templates'],new['templates']):
            assert old_t['paths']==new_t['paths']
        for key in ('instances','sources','observations'):assert new[key]==asset[key]
        destination=candidate_path.parent/ref['path'];prep.write(destination,new)
        changes.append(dict(ad=ref['advertisement_number'],path=ref['path'],before_sha256=ref['sha256'],after_sha256=prep.digest(destination)))
        ref['sha256']=prep.digest(destination)
    candidate['print_layout_revision']=dict(schema=plan['schema'],definition_sha256=prep.digest(plan_path),
        policy=plan.get('numerical_policy','Shared affine artwork components; independent nonlinear glyph warps removed'),
        source_path_identities_and_vertices_unchanged=True,physical_ad_size_not_declared_measured=True)
    prep.write(candidate_path,candidate)
    new_model=SceneParameterModel(candidate,candidate_path,optimize_scene=True)
    values=new_model.initial_state.values.copy();copied=[];reset=[]
    for identity,block in new_model.initial_state.blocks.items():
        if not block['available']:continue
        if block['kind']=='print_shape' and not (preserve_unchanged_shapes and identity in old_state
                and old_state.blocks[identity]['independent_local_indices']==block['independent_local_indices']):
            reset.append(identity);continue
        if identity not in old_state:raise ValueError('Unexpected non-shape identity change: '+identity)
        prior=old_state[identity]
        if prior is None or len(prior)!=block['size']:raise ValueError('Cannot preserve current native block: '+identity)
        values[block['offset']:block['offset']+block['size']]=prior;copied.append(identity)
    new_state=ParameterState(new_model.initial_state.layout,values)
    snapshot={**saved,'scene_values':values}
    np.savez(out/'seed_state.npz',**snapshot)
    # Hardlink only the immutable seven-field observation file; initializers
    # and receipts are independent so diagnostics cannot modify the live run.
    os.link(prep.RUN/'point_observations.npz',out/'point_observations.npz')
    shutil.copy2(prep.RUN/'initialized_points.npz',out/'initialized_points.npz')
    receipt=prep.read(prep.RUN/'point_input.json');receipt['input']=str(out/'point_observations.npz')
    prep.write(out/'point_input.json',receipt)
    mapping=prep.read(prep.RUN/'mapping.json');mapping['scene_path']=str(candidate_path)
    mapping['scene_revision']=scene_content_identity(candidate,candidate_path)
    mapping['current_point_input']=str(out/'point_observations.npz')
    mapping['explicit_native_model_revision']=dict(definition=str(plan_path),definition_sha256=prep.digest(plan_path),
        dataset_overwrote_parameters=False,raw_observation_binding_unchanged=True)
    prep.write(out/'mapping.json',mapping)
    with np.load(out/'point_observations.npz') as points:
        observed={mapping['path_ids'][i] for i in points['path_indices']}
    dependencies={b['id'] for b in new_state.blocks.values() if b['kind'] in ('board_shape','shared_geometry')}
    for path in observed:
        if '::' in path:dependencies.update(new_model.dependencies(path))
    active=set(new_state.active_columns(dependencies).tolist());counts={}
    for b in new_state.blocks.values():
        if b['available']:counts[b['kind']]=counts.get(b['kind'],0)+sum(b['offset']+i in active for i in b['independent_local_indices'])
    result=dict(schema='rinkspace.s01_print_layout_migration.v1',before_scene_revision=old_state.layout['scene_content_revision'],
        after_scene_revision=new_state.layout['scene_content_revision'],before_snapshot_sha256=hashlib.sha256(blob).hexdigest(),
        seed_snapshot_sha256=prep.digest(out/'seed_state.npz'),point_input_sha256=input_hash,
        assets=changes,preserved_nonshape_blocks=copied,reset_shared_artwork_blocks=reset,active_coordinates=counts,
        source_paths_and_vertices_unchanged=True,raw_point_file_unchanged=prep.digest(out/'point_observations.npz')==input_hash,
        camera_controls_rig_board_ice_size_and_placement_preserved=True,
        nonlinear_shape_coordinates_active=False,production_installed=False,full_shot_fit_complete=False)
    prep.write(out/'migration.json',result)
    print(json.dumps(dict(directory=str(out),active_coordinates=counts,raw_point_file_unchanged=True),ensure_ascii=True))


def install():
    out=prep.RUN/'print_component_layout';migration=prep.read(out/'migration.json')
    if migration['production_installed']:raise ValueError('Layout revision already installed')
    old=prep.read(prep.SCENE);candidate_path=out/'scene/scene.json';candidate=prep.read(candidate_path)
    if scene_content_identity(old,prep.SCENE)!=migration['before_scene_revision']:
        raise ValueError('Current native model changed during layout review')
    if prep.digest(prep.RUN/'accepted_pixels_latest.npz')!=migration['before_snapshot_sha256']:
        raise ValueError('Current native accepted state changed during layout review')
    if prep.digest(prep.RUN/'point_observations.npz')!=migration['point_input_sha256']:
        raise ValueError('Current source input changed during layout review')
    if scene_content_identity(candidate,candidate_path)!=migration['after_scene_revision']:
        raise ValueError('Staged native model changed after validation')
    if not (prep.RUN/'stop_requested').exists() or not (out/'stop_requested').exists():
        raise ValueError('Both native runs must be stopped at the saved states')
    repair=prep.read(out/'ad6_binding_review/input/replacement_receipt.json')
    validation=repair['gpu_input_validation']
    if repair['stage']!='installed' or not validation['passed'] or validation['invalid_rows']:
        raise ValueError('Source binding correction has not passed full native validation')
    if prep.digest(out/'point_observations.npz')!=validation['input_sha256']:
        raise ValueError('Validated point input changed')
    with np.load(io.BytesIO((out/'accepted_pixels_latest.npz').read_bytes())) as f:
        saved={k:f[k] for k in f.files}
    if str(saved['point_input_sha256'])!=validation['input_sha256']:
        raise ValueError('Staged accepted state/input mismatch')
    model=SceneParameterModel(candidate,candidate_path,optimize_scene=True)
    state=ParameterState(model.initial_state.layout,saved['scene_values'])
    for block in state.blocks.values():
        if block['kind']=='print_shape' and (any(i>=4 for i in block['independent_local_indices'])
                or np.any(state[block['id']][4:])):
            raise ValueError('A retired glyph warp has returned')
    backup=out/'production_before_install';backup.mkdir(exist_ok=False)
    names=('point_observations.npz','initialized_points.npz','mapping.json','point_input.json','accepted_pixels_latest.npz')
    for name in names:shutil.copy2(prep.RUN/name,backup/name)
    staged=[]
    for before,after in zip(old['advertisement_assets'],candidate['advertisement_assets']):
        if before['template_id']!=after['template_id']:raise ValueError('Template order changed')
        destination=guard_project_write((prep.SCENE.parent/before['path']).resolve())
        pending=guard_project_write(destination.with_name(destination.name+'.layout_pending'))
        pending.write_bytes((candidate_path.parent/after['path']).read_bytes())
        staged.append((pending,destination));after['path']=before['path']
    pending_scene=guard_project_write(prep.SCENE.with_name(prep.SCENE.name+'.layout_pending'))
    prep.write(pending_scene,candidate)
    for pending,destination in staged:pending.replace(destination)
    pending_scene.replace(prep.SCENE)
    if scene_content_identity(prep.read(prep.SCENE),prep.SCENE)!=migration['after_scene_revision']:
        raise ValueError('Installed native revision differs from validated candidate')
    mapping=prep.read(out/'mapping.json');mapping['scene_path']=str(prep.SCENE)
    mapping['current_point_input']=str(prep.RUN/'point_observations.npz')
    receipt=prep.read(out/'point_input.json');receipt['input']=str(prep.RUN/'point_observations.npz')
    for name in names:
        pending=prep.RUN/(name+'.layout_pending')
        if name=='mapping.json':prep.write(pending,mapping)
        elif name=='point_input.json':prep.write(pending,receipt)
        else:shutil.copy2(out/name,pending)
    for name in names:(prep.RUN/(name+'.layout_pending')).replace(prep.RUN/name)
    prep.load_pixel_input(prep.RUN)
    migration.update(production_installed=True,current_point_input_sha256=validation['input_sha256'],
        explicit_source_binding_revision=str(Path(__file__).with_name('ad6_disconnected_contour_review_20261026.json')),
        production_numeric_state_sha256=prep.digest(prep.RUN/'accepted_pixels_latest.npz'),
        installed_scene_path=str(prep.SCENE),before_current_input_and_state=str(backup))
    prep.write(out/'migration.json',migration)
    print(json.dumps(dict(installed=True,scene_revision=migration['after_scene_revision'],
        input_sha256=validation['input_sha256'],retained_columns=validation['retained_columns'],source_rows=validation['rows'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',nargs='?',choices=['stage','install'],default='stage')
    args=parser.parse_args();stage() if args.action=='stage' else install()
