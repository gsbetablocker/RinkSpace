"""Build a detached Ad3 official-shape completion; no observations or fitting.

CPU native checks use one immutable numerical snapshot. Existing material paths,
parameter blocks and values stay unchanged; content identity hashes must change.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import io
import json

import numpy as np

from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.print_parameters import validate_print_parameters
from rinkspace.domain.scene_assets import read_scene_assets
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.session.gpu_geometry import compile_geometry
from rinkspace.path_policy import guard_project_write


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def digest(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ref(path):return dict(path=str(Path(path).resolve()),sha256=digest(path))
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def equal_bytes(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def composition_affine(template,composition):
    contours={c['id']:c for c in composition['contours']}
    common=[p for p in template['paths']if p.get('original_id')in contours]
    x=[];y=[]
    for path in common:
        source=np.asarray(contours[path['original_id']]['xy_composition'],float)
        target=np.asarray(path['material_uv'],float)
        if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=2:
            raise ValueError('Official/common contour vertex correspondence differs')
        x.extend(source);y.extend(target)
    x=np.c_[np.asarray(x),np.ones(len(x))];y=np.asarray(y)
    matrix,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
    if rank!=3 or not np.isfinite(matrix).all():raise ValueError('Common official affine is not identifiable')
    error=float(np.max(np.abs(x@matrix-y)))
    if error>1e-12:raise ValueError('Official composition is not one existing material affine')
    return matrix,common,error


def expanded_bounds_same_center(bounds,vertices):
    """Enlarge only necessary axes, retaining the native floating-point centre."""
    result=np.asarray(bounds,float).copy();vertices=np.asarray(vertices,float)
    for axis in (0,1):
        lo,hi=result[axis],result[axis+2]
        minimum,maximum=vertices[:,axis].min(),vertices[:,axis].max()
        if minimum>=lo and maximum<=hi:continue
        center=lo+(hi-lo)/2
        half=max(center-lo,hi-center,center-minimum,maximum-center)
        a,b=center-half,center+half
        # Search adjacent representable bounds, rather than accepting a shifted
        # centre or moving a single protected material vertex at an endpoint.
        candidates_a=[a];candidates_b=[b];a_down=a_up=a;b_down=b_up=b
        for _ in range(8):
            a_down=np.nextafter(a_down,-np.inf);a_up=np.nextafter(a_up,np.inf)
            b_down=np.nextafter(b_down,-np.inf);b_up=np.nextafter(b_up,np.inf)
            candidates_a.extend([a_down,a_up]);candidates_b.extend([b_down,b_up])
        chosen=next(((u,v)for u in candidates_a for v in candidates_b
            if u<=minimum and v>=maximum and u+(v-u)/2==center),None)
        if chosen is None:raise ValueError('Cannot extend material bounds with the exact original centre')
        result[axis],result[axis+2]=chosen
    return result.tolist()


def complete_asset(assets,composition,composition_reference,svg_reference):
    result=deepcopy(assets)
    template=next(t for t in result['templates']if t['id']=='print_template/Ad3')
    matrix,common,error=composition_affine(template,composition)
    if len(common)!=15 or len(template['paths'])!=19:raise ValueError('Expected the current 15 official contours plus four panel edges')
    new_contours=[c for c in composition['contours']if c['id'].startswith('official_svg.path9.')]
    if len(composition['contours'])!=36 or len(new_contours)!=21:raise ValueError('Expected 21 missing official Daikichi contours')
    internal=template['internal_geometry']
    if len(internal['groups'])!=1:raise ValueError('Ad3 must retain one complete-artwork shape group')
    group=internal['groups'][0];shape=next(b for b in result['parameter_blocks']if b['id']==group['parameter_block_id'])
    if shape['independent_local_indices']!=[0,1,2,3]or any(shape['values'][4:]):
        raise ValueError('Ad3 completion requires the existing affine-only activity contract')
    added=[]
    for contour in new_contours:
        vertices=np.asarray(contour['xy_composition'],float)
        material=np.c_[vertices,np.ones(len(vertices))]@matrix
        identity='Ad3.print.'+contour['id']
        parent=contour['parent_id']
        path=dict(id=identity,original_id=contour['id'],template_id=template['id'],material_uv=material.tolist(),
            closed=contour['closed'],role='letter_hole_boundary'if parent is not None else'letter_material_boundary',
            parent_contour_key=None if parent is None else'Ad3.print.'+parent,component_id=contour['component_id'],
            official_svg_path_ordinal=contour['original_svg_path_ordinal'],
            official_svg_subpath_ordinal=contour['original_svg_subpath_ordinal'],
            original_path_d_sha256=contour['original_path_d_sha256'],
            official_shape_provenance=dict(composition=composition_reference,contour_id=contour['id'],svg=svg_reference,
                composition_to_material_uv=matrix.tolist(),current_source_observations_created=0,
                retired_source_uv_bound=False,pixel_accuracy_accepted=False,user_visual_acceptance=False))
        template['paths'].append(path);added.append(identity)
        internal['path_bindings'][identity]=dict(kind='shape_group',group_id=group['id'])
    old_bounds=list(group['reference_bounds'])
    all_vertices=np.vstack([p['material_uv']for p in template['paths']if p['id']in group['path_ids']or p['id']in added])
    group['reference_bounds']=expanded_bounds_same_center(old_bounds,all_vertices)
    group['path_ids'].extend(added)
    policy=next(g for g in internal['layout_policy']['groups']if g['id']==group['component_identity'])
    policy['path_ids'].extend(added)
    template['text']='買取大吉'
    result['counts'].update(unique_template_paths=40,current_glyph_and_logo_boundaries=36)
    result['official_shape_completion_candidate']=dict(added_path_ids=added,source_observations_created=0,
        retired_source_uv_bound=False,old_material_uv_unchanged=True,parameter_blocks_unchanged=True,
        old_reference_bounds=old_bounds,new_reference_bounds=group['reference_bounds'],
        existing_affine_center_preserved=True,current_source_photo_review_required=True,
        user_visual_acceptance=False,pixel_accuracy_accepted=False)
    validate_print_parameters(result)
    assert result['parameter_blocks']==assets['parameter_blocks']
    assert result['instances']==assets['instances']
    assert template['paths'][:19]==assets['templates'][0]['paths']
    for name in ('observations','aliases','observation_store'):
        assert result.get(name)==assets.get(name)
    return result,dict(matrix=matrix.tolist(),common_paths=len(common),common_vertex_max_abs_error=error,
        added_path_ids=added,old_reference_bounds=old_bounds,new_reference_bounds=group['reference_bounds'])


def layout_without_content_identity(layout):
    return {k:v for k,v in layout.items()if k not in ('scene_content_revision','layout_revision')}


def verify_native(old_scene,old_path,new_scene,new_path,state,manifest,core):
    old_model=SceneParameterModel(old_scene,old_path,optimize_scene=True)
    new_model=SceneParameterModel(new_scene,new_path,optimize_scene=True)
    if layout_without_content_identity(old_model.initial_state.layout)!=layout_without_content_identity(new_model.initial_state.layout):
        raise ValueError('Completion changed the actual shared parameter layout')
    assert equal_bytes(old_model.initial_state.values,new_model.initial_state.values)
    old_state=ParameterState(old_model.initial_state.layout,state['scene_values'])
    new_state=ParameterState(new_model.initial_state.layout,state['scene_values'])
    shape=next(b for b in old_state.layout['blocks']if b['id'].startswith('print_shape/print_template/Ad3/'))
    values=state['scene_values'][shape['offset']:shape['offset']+shape['size']]
    if np.any(values[4:]!=0):raise ValueError('Current numerical Ad3 state has nonzero nonlinear warp coordinates')
    old_current,old_native=compile_geometry(old_scene,old_path,core,parameter_state=old_state)
    _,new_native=compile_geometry(new_scene,new_path,core,parameter_state=new_state)
    assert set(old_native)<=set(new_native)
    xyz_old=[];xyz_new=[];path_results=[];max_active=0.
    for identity,path in old_native.items():
        if path['layer']=='advertisement':
            stations=np.linspace(0,1,9)
            a=old_model.evaluate_print(identity,stations,core,state=old_state)
            b=new_model.evaluate_print(identity,stations,core,state=new_state)
            left,right=a['xyz'],b['xyz']
            assert np.array_equal(a['status'],b['status'])and not np.any(a['status'])
            for key,columns in a['active_jacobians_by_block'].items():
                max_active=max(max_active,float(np.max(np.abs(columns-b['active_jacobians_by_block'][key])))if columns.size else 0.)
        else:
            other=new_native[identity]
            assert path==other
            support=[0.,core.board_perimeter(old_current['board'])]if path['kind']==7 else(path.get('support')or[0.,1.])
            stations=np.linspace(*support,9)
            left=np.asarray(core.sample_path(old_current['board'],path['kind'],path['parameters'],path.get('vertices',[]),stations))
            right=left.copy()
        error=float(np.max(np.abs(left-right)))
        if error>1e-13:raise ValueError('Old native world geometry changed: '+identity)
        xyz_old.append(left);xyz_new.append(right);path_results.append(dict(path_id=identity,samples=9,max_abs_xyz_difference=error))
    left,right=np.vstack(xyz_old),np.vstack(xyz_new)
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    projections=[]
    for shot in manifest['shots']:
        step=int(manifest.get('source_clock',{}).get('native_pts_step',256))
        pt=shot['first_pts']+((shot['last_pts']-shot['first_pts'])//(2*step))*step
        time=(pt-shot['reference_pts'])*shot['time_base_num']/shot['time_base_den']+timeline.offset(shot)
        args=(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,time)
        a=core.project_motion(*args,left);b=core.project_motion(*args,right)
        assert np.array_equal(a['status'],b['status'])
        valid=a['status']==0
        error=float(np.max(np.abs(a['uv'][valid]-b['uv'][valid])))if valid.any()else 0.
        if error>1e-9:raise ValueError('Old native source projection changed')
        projections.append(dict(shot=shot['id'],source_pts=pt,valid_samples=int(valid.sum()),
            failed_samples=int((~valid).sum()),max_abs_uv_difference=error))
    for identity in sorted(set(new_native)-set(old_native)):
        data=new_model.evaluate_print(identity,np.linspace(0,1,9),core,state=new_state)
        if np.any(data['status']):raise ValueError('New official curve has invalid native geometry')
    return dict(old_layout=old_model.initial_state.layout,new_layout=new_model.initial_state.layout,
        parameter_layout_except_content_identity_exact=True,scene_values_byte_exact=equal_bytes(old_state.values,new_state.values),
        initial_values_byte_exact=True,current_Ad3_shape_values=values.tolist(),
        max_active_print_jacobian_difference=max_active,old_native_paths=len(old_native),
        new_native_paths=len(new_native),added_localized_native_path_ids=sorted(set(new_native)-set(old_native)),
        old_path_checks=path_results,projection_checks=projections,
        inactive_nonlinear_jacobian_not_claimed_unchanged=True,GPU_resident_or_fit_created=False)


def build(args):
    output=guard_project_write(args.output.resolve())
    if output.exists():raise FileExistsError(output)
    new_asset=guard_project_write(args.asset_output.resolve())
    new_scene=guard_project_write(args.scene_output.resolve())
    if new_asset.exists()or new_scene.exists():raise FileExistsError('Preserve existing candidate asset/scene')
    scene_path=args.scene.resolve();scene=read(scene_path)
    protected={str(p.resolve()):digest(p)for p in (scene_path,args.composition,args.prepared/'joint_manifest.json')}
    references=read_scene_assets(scene,scene_path)
    matches=[(reference,asset)for reference,asset in references if any(t['id']=='print_template/Ad3'for t in asset['templates'])]
    if len(matches)!=1:raise ValueError('Scene must have one explicit Ad3/Ad16 asset')
    reference,assets=matches[0];asset_path=Path(reference['path'])
    asset_path=(asset_path if asset_path.is_absolute()else scene_path.parent/asset_path).resolve()
    protected[str(asset_path)]=digest(asset_path)
    composition=read(args.composition)
    svg=Path(composition['source_svg']).resolve()
    if digest(svg)!=composition['source_sha256']:raise ValueError('Official SVG source hash differs')
    completed,report=complete_asset(assets,composition,ref(args.composition),ref(svg))
    manifest=read(args.prepared/'joint_manifest.json')
    if Path(manifest['scene_path']).resolve()!=scene_path:raise ValueError('Frozen prepared and candidate parent scene differ')
    blob=args.state.read_bytes()
    with np.load(io.BytesIO(blob),allow_pickle=False)as z:state=dict(z)
    if str(state['point_input_sha256'])!=manifest['point_input_sha256']:raise ValueError('Frozen numerical state and prepared observations differ')
    if str(state['native_library_provenance_sha256'])!=digest(args.library):raise ValueError('Frozen state and explicit native library differ')
    output.mkdir(parents=True)
    (output/'frozen_state.npz').write_bytes(blob)
    new_asset.parent.mkdir(parents=True,exist_ok=True);write(new_asset,completed)
    candidate=deepcopy(scene)
    for original,new_reference in zip(scene['advertisement_assets'],candidate['advertisement_assets']):
        path=Path(original['path']);path=(path if path.is_absolute()else scene_path.parent/path).resolve()
        new_reference['path']=str(path)
        if original is reference or original==reference:
            new_reference.update(path=str(new_asset),sha256=digest(new_asset),template_paths=40,
                binding_status='official_shape_completion_candidate_not_source_pixel_acceptance',
                previous_template_acceptance=new_reference.pop('template_acceptance',None),
                template_acceptance=dict(user_visual_acceptance=False,pixel_accuracy_accepted=False,current_source_photo_review_required=True))
            new_reference['model_content']=[*original.get('model_content',[]),'大吉_official_shape_candidate']
            new_reference['excluded_from_model']=[v for v in original.get('excluded_from_model',[])if v not in ('大','吉')]
    candidate['revision']=scene['revision']+1
    candidate['ad3_completion_candidate']=dict(original_scene=ref(scene_path),current_pointer_updated=False,
        source_observations_created=0,retired_source_uv_bound=False,pixel_accuracy_accepted=False,user_visual_acceptance=False)
    new_scene.parent.mkdir(parents=True,exist_ok=True);write(new_scene,candidate)
    verification=verify_native(scene,scene_path,candidate,new_scene,state,manifest,NativeCore(args.library))
    for path,value in protected.items():
        if digest(path)!=value:raise ValueError('Static candidate input changed: '+path)
    result=dict(schema='rinkspace.ad3_official_completion_candidate.v1',**report,**verification,
        asset=ref(new_asset),scene=ref(new_scene),frozen_state=ref(output/'frozen_state.npz'),
        numerical_state_source=str(args.state.resolve()),static_inputs=protected,
        Ad16_instances_unchanged=True,Ad16_remains_unlocalized=True,new_independent_parameters=0,
        source_observations_created=0,retired_source_uv_bound=False,pixel_accuracy_accepted=False,
        current_pointer_updated=False,prepared_rebased=False,fit_executed=False)
    write(output/'report.json',result)
    write(output/'rebind_plan.json',dict(schema='rinkspace.scene_only_prepared_rebind_plan.v1',executed=False,
        candidate_scene=ref(new_scene),state_selection='At explicit rebind time use the latest complete accepted state matching current parent observations, not this build-time snapshot.',
        existing_entry_support='prepare_native_source_update has no scene override; reuse prepare_spatial_refinement._write_prepared with a new SceneParameterModel and copied manifest/mapping.',
        changes=['Verify exact blocks/offsets/activity and all original geometry against the selected latest state.',
            'Update manifest scene_path, scene_file_sha256 and scene_revision; mapping scene_revision gets new content identity.',
            'Keep mapping path order and all existing seven-field arrays byte-identical; append new localized paths only, if needed.',
            'Call _write_prepared with identical numeric stations and link_input=parent point_observations.npz; bind_snapshot updates identity metadata only.',
            'Recheck every numeric block and station/source row bytes; old LM internal checkpoints are not reused across a changed scene.'],
        source_row_or_parameter_changes_required=False,new_scene_or_layout_revision_must_not_be_forged=True))
    print(json.dumps(dict(output=str(output),added_paths=len(report['added_path_ids']),
        common_vertex_max_abs_error=report['common_vertex_max_abs_error'],
        parameter_layout_except_content_identity_exact=True,scene_values_byte_exact=result['scene_values_byte_exact'],
        old_native_paths=result['old_native_paths'],new_native_paths=result['new_native_paths'],fit_executed=False),ensure_ascii=False),flush=True)


def rebind_prepared(prepared_root,state_path,candidate_scene,output,library):
    """Rebind one explicit completed numerical state; no station initialization."""
    import prepare_native_multicamera as prepared
    from prepare_spatial_refinement import _write_prepared
    root,state_path,candidate_scene,output,library=map(lambda p:Path(p).resolve(),
        (prepared_root,state_path,candidate_scene,output,library))
    output=guard_project_write(output)
    if output.exists()or output.is_relative_to(root):raise FileExistsError('Rebind output must be new and separate')
    manifest=read(root/'joint_manifest.json');mapping=read(root/'mapping.json')
    with np.load(state_path,allow_pickle=False)as z:state=dict(z)
    with np.load(root/'point_observations.npz',allow_pickle=False)as z:points=dict(z)
    old_scene_path=Path(manifest['scene_path']).resolve();old_scene=read(old_scene_path)
    scene=read(candidate_scene)
    old_model=SceneParameterModel(old_scene,old_scene_path,optimize_scene=True)
    model=SceneParameterModel(scene,candidate_scene,optimize_scene=True)
    if layout_without_content_identity(old_model.initial_state.layout)!=layout_without_content_identity(model.initial_state.layout):
        raise ValueError('Scene rebind changed numeric parameter layout')
    prepared.validate_snapshot(state,old_model.initial_state.layout,mapping,manifest,points)
    if digest(root/'point_observations.npz')!=manifest['point_input_sha256']:raise ValueError('Parent source input changed')
    if str(state['native_library_provenance_sha256'])!=digest(library):raise ValueError('Rebind numerical native provenance differs')
    current=ParameterState(model.initial_state.layout,state['scene_values'])
    _,paths=compile_geometry(scene,candidate_scene,NativeCore(library),parameter_state=current)
    if not set(mapping['path_ids'])<=set(paths):raise ValueError('New scene omits existing observation paths')
    new_mapping=deepcopy(mapping)
    additions=[path for path in paths if path not in set(mapping['path_ids'])]
    # Complete the observable inventory explicitly while preserving every old
    # index. Other previously unobserved model paths are not silently appended.
    additions=[path for path in additions if path.startswith('print_instance/Ad3::Ad3.print.official_svg.path9.')]
    if len(additions)!=21:raise ValueError('Expected exactly 21 new localized Ad3 mapping paths')
    new_mapping['path_ids'].extend(additions)
    new_mapping['scene_revision']=model.initial_state.layout['scene_content_revision']
    target_manifest=deepcopy(manifest)
    target_manifest.update(scene_path=str(candidate_scene),scene_file_sha256=digest(candidate_scene),
        scene_revision=model.initial_state.layout['scene_content_revision'],
        native_library=str(library),native_library_sha256=digest(library))
    provenance=dict(schema='rinkspace.scene_only_prepared_rebind.v1',parent_root=str(root),parent_state=ref(state_path),
        previous_scene=ref(old_scene_path),candidate_scene=ref(candidate_scene),appended_path_ids=additions,
        existing_path_indices_unchanged=True,parameter_layout_except_content_identity_exact=True,
        numeric_globals_byte_exact=True,source_rows_and_UV_unchanged=True,station_initialization_performed=False,fit_executed=False)
    previous_spatial=deepcopy(manifest.get('spatial_refinement_preparation'))
    updated,new_mapping=_write_prepared(output,points,state['stations'],manifest['shots'],state,
        target_manifest,new_mapping,model,provenance,link_input=root/'point_observations.npz')
    if previous_spatial is not None:updated['spatial_refinement_preparation']=previous_spatial
    else:updated.pop('spatial_refinement_preparation',None)
    updated['scene_only_rebind']=provenance;prepared.write(output/'joint_manifest.json',updated)
    with np.load(output/'initial_pixels.npz',allow_pickle=False)as z:saved=dict(z)
    prepared.validate_snapshot(saved,model.initial_state.layout,new_mapping,updated,points)
    for key in state.keys()-{'joint_numeric_dependencies_json'}:
        if not equal_bytes(state[key],saved[key]):raise ValueError('Scene rebind changed numeric state: '+key)
    if digest(output/'point_observations.npz')!=manifest['point_input_sha256']:raise ValueError('Scene rebind changed source array bytes')
    write(output/'scene_rebind_receipt.json',dict(provenance,output=str(output),rows=len(points['pts']),
        old_mapping_paths=len(mapping['path_ids']),new_mapping_paths=len(new_mapping['path_ids']),
        scene_values_byte_exact=True,all_numeric_stations_byte_exact=True,point_input_sha256=manifest['point_input_sha256']))
    print(json.dumps(dict(output=str(output),rows=len(points['pts']),appended_mapping_paths=21,
        numeric_state_byte_exact=True,source_input_byte_exact=True,fit_executed=False)),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','composition','prepared','state','library','output','asset-output','scene-output'):
        parser.add_argument('--'+name,type=Path,required=True)
    build(parser.parse_args())
