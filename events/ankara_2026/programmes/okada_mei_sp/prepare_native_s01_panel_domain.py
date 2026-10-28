"""Stage/install source-backed poster containment without changing observations."""
from pathlib import Path
import argparse, importlib.util, io, json, os, shutil
import numpy as np

spec=importlib.util.spec_from_file_location('prep',Path(__file__).with_name('prepare_native_s01.py'))
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.print_assets import sample_template_paths
from rinkspace.domain.print_parameters import validate_print_parameters
from rinkspace.domain.scene_assets import read_scene_assets, scene_content_identity, snapshot_scene
from rinkspace.path_policy import guard_project_write

OUT=prep.RUN/'print_component_layout/panel_domain'
LIBRARY=prep.DATA/'build/native_joint_panel_20261027/rinkspace_core.dll'
SOURCE={2:307456,3:326400,4:353024,6:423424,7:444672}


def snapshot(path):
    with np.load(io.BytesIO(path.read_bytes())) as f:return {k:f[k] for k in f.files}


def stage():
    if OUT.exists():raise FileExistsError('Do not overwrite an existing model revision')
    OUT.mkdir()
    current=snapshot(prep.RUN/'accepted_pixels_latest.npz')
    old=SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True).initial_state
    points,mapping,_=prep.load_pixel_input(prep.RUN);camera=prep.read(prep.CAMERA)
    prep.validate_snapshot_identity(current,old.layout,camera,mapping,points=points,
        point_input_sha256=prep.digest(prep.RUN/'point_observations.npz'))
    if str(current['point_input_sha256'])!=prep.digest(prep.RUN/'point_observations.npz'):
        raise ValueError('Current state/input mismatch')
    snapshot_scene(prep.SCENE,OUT/'before_scene')
    scene,_=snapshot_scene(prep.SCENE,OUT/'scene');scene_path=OUT/'scene/scene.json'
    shutil.copy2(prep.RUN/'accepted_pixels_latest.npz',OUT/'before_state.npz')
    changes=[]
    for ref,assets in read_scene_assets(scene,scene_path):
        number=ref['advertisement_number']
        if number not in SOURCE:continue
        pt=SOURCE[number]
        folder='print_component_layout/shape_review_solver_fixed' if number==6 else 'print_shape_audit'
        source=prep.RUN/folder/f'Ad{number:02d}_source_{pt}.png'
        if not source.is_file():raise FileNotFoundError(source)
        template=next(t for t in assets['templates'] if t['id']==ref['template_id'])
        template['material_domain']=dict(kind='physical_unit_panel',
            boundary_path_ids=[p['id'] for p in template['paths'] if p.get('role')=='panel_outer_boundary'],
            evidence=dict(kind='actual_source_visible_colored_poster_contains_printing',source_image=str(source),
                source_image_sha256=prep.digest(source),source_pts=pt,
                decision='The observed colored poster boundary contains the printed wordmarks and logos; no numeric margin or measured size is asserted'))
        validate_print_parameters(assets)
        path=scene_path.parent/ref['path'];before=ref['sha256'];prep.write(path,assets);ref['sha256']=prep.digest(path)
        changes.append(dict(ad=number,before_sha256=before,after_sha256=ref['sha256']))
    scene['print_panel_domain_revision']=dict(kind='physical_unit_panel',advertisements=sorted(SOURCE),
        source_points_unchanged=True,parameter_freedoms_unchanged=True)
    prep.write(scene_path,scene)
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    # This metadata revision must not reorder or reset any selected parameter.
    if model.initial_state.blocks!=old.blocks:raise ValueError('Unexpected parameter block change')
    values=current['scene_values'].copy();state=ParameterState(model.initial_state.layout,values)
    core=NativeCore(LIBRARY);repairs=[]
    for ref,assets in read_scene_assets(scene,scene_path):
        template=assets['templates'][0]
        if template.get('material_domain') is None:continue
        shapes=sample_template_paths(assets,template['id'],core,np.exp(state[template['size_class_id']]),parameter_state=state)
        sampled={p['id']:np.asarray(p['material_uv']) for p in shapes}
        for group in template['internal_geometry']['groups']:
            uv=np.vstack([sampled[p] for p in group['path_ids']]);lo=uv.min(0);hi=uv.max(0)
            delta=np.maximum(-lo,0.)+np.minimum(1.-hi,0.)
            if not np.any(delta):continue
            if np.any(hi-lo>1.):raise ValueError('Cannot restore containment by translation alone')
            block=state.blocks[group['parameter_block_id']]
            if group['parameterization']!='monotone_axis_warp.v1' or not {0,1}<=set(block['independent_local_indices']):
                raise ValueError('Unexpected non-affine containment repair')
            values[block['offset']:block['offset']+2]+=delta
            repairs.append(dict(ad=ref['advertisement_number'],block=block['id'],before_bounds=[lo.tolist(),hi.tolist()],
                translation_delta=delta.tolist(),reason='Minimal feasible initial translation under the explicit physical panel relation; remains jointly fitted'))
    current['scene_values']=values
    os.link(prep.RUN/'point_observations.npz',OUT/'point_observations.npz')
    shutil.copy2(prep.RUN/'initialized_points.npz',OUT/'initialized_points.npz')
    receipt=prep.read(prep.RUN/'point_input.json');receipt['input']=str(OUT/'point_observations.npz');prep.write(OUT/'point_input.json',receipt)
    mapping=prep.read(prep.RUN/'mapping.json');mapping.update(scene_path=str(scene_path),
        scene_revision=scene_content_identity(scene,scene_path),current_point_input=str(OUT/'point_observations.npz'))
    prep.write(OUT/'mapping.json',mapping)
    prep.bind_snapshot_identity(current,model.initial_state.layout,camera,mapping,native_library=LIBRARY)
    np.savez(OUT/'seed_state.npz',**current)
    result=dict(schema='rinkspace.s01_print_panel_domain_revision.v1',assets=changes,initial_feasibility_repairs=repairs,
        before_scene_revision=scene_content_identity(prep.read(prep.SCENE),prep.SCENE),
        after_scene_revision=mapping['scene_revision'],before_snapshot_sha256=prep.digest(OUT/'before_state.npz'),
        point_input_sha256=prep.digest(OUT/'point_observations.npz'),native_library=str(LIBRARY),
        native_library_sha256=prep.digest(LIBRARY),parameter_freedoms_unchanged=True,
        camera_motion_board_ice_size_placement_unchanged=True,source_points_unchanged=True,installed=False)
    prep.write(OUT/'migration.json',result);print(json.dumps(result,ensure_ascii=True))


def install():
    report=prep.read(OUT/'migration.json');candidate_path=OUT/'scene/scene.json';candidate=prep.read(candidate_path)
    if report['installed']:raise ValueError('Revision already installed')
    checks=[(scene_content_identity(prep.read(prep.SCENE),prep.SCENE),report['before_scene_revision']),
        (scene_content_identity(candidate,candidate_path),report['after_scene_revision']),
        (prep.digest(prep.RUN/'accepted_pixels_latest.npz'),report['before_snapshot_sha256']),
        (prep.digest(prep.RUN/'point_observations.npz'),report['point_input_sha256'])]
    if any(a!=b for a,b in checks):raise ValueError('Current model/state/input changed during review')
    valid=prep.read(OUT/'fit_input_validation.json')
    if not valid.get('passed') or valid['invalid_rows'] or valid['input_sha256']!=report['point_input_sha256']:
        raise ValueError('Candidate input has not passed native validation')
    if not (OUT/'stop_requested').exists() or not (prep.RUN/'stop_requested').exists():
        raise ValueError('Stop both numeric runs before installing')
    saved=snapshot(OUT/'accepted_pixels_latest.npz')
    model=SceneParameterModel(candidate,candidate_path,optimize_scene=True)
    points,mapping,_=prep.load_pixel_input(OUT)
    prep.validate_snapshot_identity(saved,model.initial_state.layout,prep.read(prep.CAMERA),mapping,points=points,
        point_input_sha256=report['point_input_sha256'])
    if str(saved['point_input_sha256'])!=report['point_input_sha256']:raise ValueError('Candidate state/input mismatch')
    audit=prep.read(OUT/'shape_review/report.json')
    if audit['snapshot_sha256']!=prep.digest(OUT/'accepted_pixels_latest.npz') or any(
            r['new_crossing_pairs'] or r['new_self_crossing_paths']
            or r.get('physical_panel_domain',{}).get('max_outside_material',0.)>1e-10 for r in audit['records']):
        raise ValueError('Candidate needs current all-template topology and containment audit')
    backup=OUT/'production_before_install';backup.mkdir()
    for name in ('mapping.json','accepted_pixels_latest.npz'):shutil.copy2(prep.RUN/name,backup/name)
    original=prep.read(prep.SCENE);pending=[]
    for old,new in zip(original['advertisement_assets'],candidate['advertisement_assets']):
        if old['template_id']!=new['template_id']:raise ValueError('Template order changed')
        if old['sha256']!=new['sha256']:
            target=guard_project_write((prep.SCENE.parent/old['path']).resolve())
            temporary=guard_project_write(target.with_name(target.name+'.panel_pending'))
            temporary.write_bytes((candidate_path.parent/new['path']).read_bytes());pending.append((temporary,target))
        new['path']=old['path']
    temporary=guard_project_write(prep.SCENE.with_name(prep.SCENE.name+'.panel_pending'));prep.write(temporary,candidate)
    for a,b in pending:a.replace(b)
    temporary.replace(prep.SCENE)
    if scene_content_identity(prep.read(prep.SCENE),prep.SCENE)!=report['after_scene_revision']:raise ValueError('Installed revision mismatch')
    mapping=prep.read(OUT/'mapping.json');mapping.update(scene_path=str(prep.SCENE),current_point_input=str(prep.RUN/'point_observations.npz'))
    prep.write(prep.RUN/'mapping.panel_pending.json',mapping);(prep.RUN/'mapping.panel_pending.json').replace(prep.RUN/'mapping.json')
    shutil.copy2(OUT/'accepted_pixels_latest.npz',prep.RUN/'accepted_pixels_latest.panel_pending.npz')
    (prep.RUN/'accepted_pixels_latest.panel_pending.npz').replace(prep.RUN/'accepted_pixels_latest.npz')
    for name in ('evaluation_pixels.npz','current_error_hotspots.json'):
        if (OUT/name).is_file():shutil.copy2(OUT/name,prep.RUN/name)
    report.update(installed=True,production_snapshot_sha256=prep.digest(prep.RUN/'accepted_pixels_latest.npz'))
    prep.write(OUT/'migration.json',report);print('Installed',report['after_scene_revision'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','install']);args=parser.parse_args()
    stage() if args.action=='stage' else install()
